// Heptapod: the forward pass and both samplers, in plain JavaScript.
// The same file runs in the page's worker and in Node (tests/), so the numbers the tests check
// come from exactly the code the page runs.
//
// Weights: one Int8Array of row-quantised matrices and one Float32Array of per-row scales and
// LayerNorm parameters, in the order export.py writes them.
(function (root) {
  "use strict";
  const V = 66, MASK = 65, NCH = 65, CTX = 256, D = 128, H = 4, HD = 32, L = 4, F4 = 512;

  // ---- weights ----------------------------------------------------------------------------
  function load(q8, f32) {
    let qi = 0, fi = 0;
    const mat = (rows, cols) => {           // int8 rows * per-row scale -> Float32Array
      const m = new Float32Array(rows * cols);
      for (let r = 0; r < rows; r++) {
        const s = f32[fi++];
        for (let c = 0; c < cols; c++) m[r * cols + c] = q8[qi++] * s;
      }
      return m;
    };
    const vec = (n) => { const v = f32.slice(fi, fi + n); fi += n; return v; };
    const w = { tok: mat(V, D), pos: mat(CTX, D), blocks: [] };
    for (let l = 0; l < L; l++) {
      w.blocks.push({
        ln1w: vec(D), ln1b: vec(D), qkv: mat(3 * D, D), proj: mat(D, D),
        ln2w: vec(D), ln2b: vec(D), fc: mat(F4, D), fc2: mat(D, F4),
      });
    }
    w.lnfw = vec(D); w.lnfb = vec(D);
    if (qi !== q8.length || fi !== f32.length) throw new Error("weight size mismatch");
    return w;
  }

  // ---- kernels ------------------------------------------------------------------------------
  // y[T][O] = x[T][I] . W[O][I]^T  (four rows of x share each pass over a row of W)
  function matmul(y, x, W, T, I, O) {
    let t = 0;
    for (; t + 3 < T; t += 4) {
      const x0 = t * I, x1 = x0 + I, x2 = x1 + I, x3 = x2 + I;
      for (let o = 0; o < O; o++) {
        const w0 = o * I;
        let a = 0, b = 0, c = 0, d = 0;
        for (let i = 0; i < I; i++) {
          const wv = W[w0 + i];
          a += x[x0 + i] * wv; b += x[x1 + i] * wv; c += x[x2 + i] * wv; d += x[x3 + i] * wv;
        }
        y[t * O + o] = a; y[(t + 1) * O + o] = b; y[(t + 2) * O + o] = c; y[(t + 3) * O + o] = d;
      }
    }
    for (; t < T; t++) {
      const x0 = t * I;
      for (let o = 0; o < O; o++) {
        const w0 = o * I;
        let a = 0;
        for (let i = 0; i < I; i++) a += x[x0 + i] * W[w0 + i];
        y[t * O + o] = a;
      }
    }
  }

  function layernorm(y, x, g, b, T) {
    for (let t = 0; t < T; t++) {
      const o = t * D;
      let m = 0;
      for (let i = 0; i < D; i++) m += x[o + i];
      m /= D;
      let v = 0;
      for (let i = 0; i < D; i++) { const z = x[o + i] - m; v += z * z; }
      const r = 1 / Math.sqrt(v / D + 1e-5);
      for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - m) * r * g[i] + b[i];
    }
  }

  const GC = Math.sqrt(2 / Math.PI);
  function gelu(x, n) {
    for (let i = 0; i < n; i++) {
      const v = x[i];
      x[i] = 0.5 * v * (1 + Math.tanh(GC * (v + 0.044715 * v * v * v)));
    }
  }

  // ---- full-sequence forward (the diffusion model; also used for GPT probes) ---------------
  function makeForward(w, causal) {
    const T = CTX;
    const x = new Float32Array(T * D), h = new Float32Array(T * D), qkv = new Float32Array(T * 3 * D);
    const att = new Float32Array(T * D), tmp = new Float32Array(T * D), ff = new Float32Array(T * F4);
    const row = new Float32Array(T), logits = new Float32Array(T * V);
    const sc = 1 / Math.sqrt(HD);
    return function forward(idx) {          // idx: length-256 token ids -> logits [256][66]
      for (let t = 0; t < T; t++) {
        const e = idx[t] * D, p = t * D;
        for (let i = 0; i < D; i++) x[p + i] = w.tok[e + i] + w.pos[p + i];
      }
      for (const B of w.blocks) {
        layernorm(h, x, B.ln1w, B.ln1b, T);
        matmul(qkv, h, B.qkv, T, D, 3 * D);
        for (let hh = 0; hh < H; hh++) {
          const qo = hh * HD, ko = D + hh * HD, vo = 2 * D + hh * HD;
          for (let i = 0; i < T; i++) {
            const qi = i * 3 * D + qo, n = causal ? i + 1 : T;
            let mx = -Infinity;
            for (let j = 0; j < n; j++) {
              const kj = j * 3 * D + ko;
              let s = 0;
              for (let c = 0; c < HD; c++) s += qkv[qi + c] * qkv[kj + c];
              s *= sc; row[j] = s; if (s > mx) mx = s;
            }
            let sum = 0;
            for (let j = 0; j < n; j++) { const e = Math.exp(row[j] - mx); row[j] = e; sum += e; }
            const ao = i * D + hh * HD;
            for (let c = 0; c < HD; c++) att[ao + c] = 0;
            for (let j = 0; j < n; j++) {
              const p = row[j] / sum, vj = j * 3 * D + vo;
              for (let c = 0; c < HD; c++) att[ao + c] += p * qkv[vj + c];
            }
          }
        }
        matmul(tmp, att, B.proj, T, D, D);
        for (let i = 0; i < T * D; i++) x[i] += tmp[i];
        layernorm(h, x, B.ln2w, B.ln2b, T);
        matmul(ff, h, B.fc, T, D, F4);
        gelu(ff, T * F4);
        matmul(tmp, ff, B.fc2, T, F4, D);
        for (let i = 0; i < T * D; i++) x[i] += tmp[i];
      }
      layernorm(h, x, w.lnfw, w.lnfb, T);
      matmul(logits, h, w.tok, T, D, V);
      return logits;
    };
  }

  // ---- one-position-at-a-time forward with a key/value cache (the GPT) --------------------
  function makeStepper(w) {
    const K = w.blocks.map(() => new Float32Array(CTX * D));
    const Vc = w.blocks.map(() => new Float32Array(CTX * D));
    const x = new Float32Array(D), h = new Float32Array(D), qkv = new Float32Array(3 * D);
    const att = new Float32Array(D), tmp = new Float32Array(D), ff = new Float32Array(F4);
    const row = new Float32Array(CTX), logits = new Float32Array(V);
    const sc = 1 / Math.sqrt(HD);
    return function step(tok, t) {          // feed token at position t -> logits for t+1
      for (let i = 0; i < D; i++) x[i] = w.tok[tok * D + i] + w.pos[t * D + i];
      w.blocks.forEach((B, l) => {
        layernorm(h, x, B.ln1w, B.ln1b, 1);
        matmul(qkv, h, B.qkv, 1, D, 3 * D);
        K[l].set(qkv.subarray(D, 2 * D), t * D);
        Vc[l].set(qkv.subarray(2 * D, 3 * D), t * D);
        for (let hh = 0; hh < H; hh++) {
          const o = hh * HD;
          let mx = -Infinity;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            for (let c = 0; c < HD; c++) s += qkv[o + c] * K[l][j * D + o + c];
            s *= sc; row[j] = s; if (s > mx) mx = s;
          }
          let sum = 0;
          for (let j = 0; j <= t; j++) { const e = Math.exp(row[j] - mx); row[j] = e; sum += e; }
          for (let c = 0; c < HD; c++) att[o + c] = 0;
          for (let j = 0; j <= t; j++) {
            const p = row[j] / sum;
            for (let c = 0; c < HD; c++) att[o + c] += p * Vc[l][j * D + o + c];
          }
        }
        matmul(tmp, att, B.proj, 1, D, D);
        for (let i = 0; i < D; i++) x[i] += tmp[i];
        layernorm(h, x, B.ln2w, B.ln2b, 1);
        matmul(ff, h, B.fc, 1, D, F4);
        gelu(ff, F4);
        matmul(tmp, ff, B.fc2, 1, F4, D);
        for (let i = 0; i < D; i++) x[i] += tmp[i];
      });
      layernorm(h, x, w.lnfw, w.lnfb, 1);
      matmul(logits, h, w.tok, 1, D, V);
      return logits;
    };
  }

  // ---- sampling helpers ---------------------------------------------------------------------
  function rng(seed) {                       // mulberry32
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // softmax over the 65 characters (never [MASK]) at temperature temp; returns [char, prob]
  const pr = new Float64Array(NCH);
  function sampleChar(logits, off, temp, rand) {
    let mx = -Infinity;
    for (let c = 0; c < NCH; c++) { const v = logits[off + c] / temp; pr[c] = v; if (v > mx) mx = v; }
    let s = 0;
    for (let c = 0; c < NCH; c++) { pr[c] = Math.exp(pr[c] - mx); s += pr[c]; }
    let u = rand() * s, c = 0;
    for (; c < NCH - 1; c++) { u -= pr[c]; if (u < 0) break; }
    return [c, pr[c] / s];
  }

  const NL = 0;                              // '\n' is character 0; both models start after one

  // The diffusion sampler. Position 0 holds the given newline; the other 255 start as [MASK].
  // Each of K steps runs the network once over all 256 positions, samples a letter for every
  // masked position, and commits a random subset of them (MDLM's ancestral order) so that
  // round(255*s/K) are committed after step s. order:"confidence" commits the most confident
  // instead (LLaDA-style); with this small model that repeats its safest word, so the page
  // uses random order. With remasking on, in the middle half of the run a quarter as many of the
  // least confident earlier commitments (confidence as it was when committed) go back to [MASK]
  // to be predicted again with more context (ReMDM-style). onStep gets every step's record.
  function diffuse(forward, opt, onStep) {
    const { K, temp = 0.8, seed = 1, remask = true, order = "random" } = opt;
    const rand = rng(seed), N = CTX - 1;
    const x = new Uint8Array(CTX).fill(MASK); x[0] = NL;
    const conf = new Float32Array(CTX);       // confidence at commit time
    const cand = new Uint8Array(CTX), cc = new Float32Array(CTX), key = new Float64Array(CTX);
    let done = 0;
    for (let s = 1; s <= K; s++) {
      const logits = forward(x);
      const masked = [];
      for (let i = 1; i < CTX; i++) if (x[i] === MASK) {
        const [c, p] = sampleChar(logits, i * V, temp, rand);
        cand[i] = c; cc[i] = p; masked.push(i); key[i] = order === "random" ? rand() : p;
      }
      const target = Math.round(N * s / K);
      let m = 0;
      if (remask && s >= Math.ceil(K / 4) && s < Math.floor(3 * K / 4)) {
        m = Math.round(0.25 * (target - Math.round(N * (s - 1) / K)));
      }
      const committedBefore = [];
      for (let i = 1; i < CTX; i++) if (x[i] !== MASK) committedBefore.push(i);
      m = Math.min(m, committedBefore.length);
      const reveal = Math.min(masked.length, target - (done - m));
      masked.sort((a, b) => key[b] - key[a] || a - b);
      const committed = masked.slice(0, reveal);
      committedBefore.sort((a, b) => conf[a] - conf[b] || a - b);
      const remasked = committedBefore.slice(0, m);
      for (const i of remasked) { x[i] = MASK; conf[i] = 0; }
      for (const i of committed) { x[i] = cand[i]; conf[i] = cc[i]; }
      done += committed.length - remasked.length;
      if (onStep) onStep({ step: s, x: x.slice(), committed, remasked, conf: conf.slice() });
    }
    return x;
  }

  // The GPT twin: the same newline at position 0, then 255 letters, one network pass each.
  function write(stepper, opt, onStep) {
    const { temp = 0.8, seed = 1 } = opt;
    const rand = rng(seed);
    const x = new Uint8Array(CTX).fill(MASK); x[0] = NL;
    for (let t = 0; t < CTX - 1; t++) {
      const logits = stepper(x[t], t);
      const [c, p] = sampleChar(logits, 0, temp, rand);
      x[t + 1] = c;
      if (onStep) onStep({ step: t + 1, pos: t + 1, char: c, p });
    }
    return x;
  }

  root.Heptapod = { V, MASK, NCH, CTX, D, L, load, makeForward, makeStepper, diffuse, write, rng };
})(typeof self !== "undefined" ? self : globalThis);
