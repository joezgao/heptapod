// JS forward pass vs PyTorch, on the 20 fixed probes per model that train/export.py wrote.
// PyTorch ran the same dequantised int8 weights. Pass: max |logit difference| < 1e-3 and the
// same top character at every position. For the GPT, the cached one-step-at-a-time forward the
// page uses to write is also checked against the full forward.
import fs from "node:fs";
import { H, weights, root } from "./load.mjs";
const N = 20, T = 256, V = 66, TOL = 1e-3;
let ok = true;
for (const kind of ["diffusion", "gpt"]) {
  const w = weights(kind);
  const buf = fs.readFileSync(root + `tests/probes-${kind}.bin`);
  const xs = new Uint8Array(buf.buffer, buf.byteOffset, N * T);
  const ref = new Float32Array(buf.buffer.slice(buf.byteOffset + N * T, buf.byteOffset + buf.length));
  const fwd = H.makeForward(w, kind === "gpt");
  let worst = 0, argmaxMiss = 0, stepWorst = 0;
  for (let p = 0; p < N; p++) {
    const x = xs.subarray(p * T, (p + 1) * T);
    const lg = fwd(x);
    for (let t = 0; t < T; t++) {
      let a = 0, b = 0;
      for (let c = 0; c < V; c++) {
        const r = ref[(p * T + t) * V + c];
        worst = Math.max(worst, Math.abs(lg[t * V + c] - r));
        if (lg[t * V + c] > lg[t * V + a]) a = c;
        if (r > ref[(p * T + t) * V + b]) b = c;
      }
      if (a !== b) argmaxMiss++;
    }
    if (kind === "gpt") {
      const step = H.makeStepper(w);
      for (let t = 0; t < T; t++) {
        const s = step(x[t], t);
        for (let c = 0; c < V; c++) stepWorst = Math.max(stepWorst, Math.abs(s[c] - lg[t * V + c]));
      }
    }
  }
  const pass = worst < TOL && argmaxMiss === 0 && stepWorst < TOL;
  ok = ok && pass;
  console.log(`${kind}: ${N} probes, max |JS - PyTorch| = ${worst.toExponential(2)}, top-1 mismatches ${argmaxMiss}/${N * T}` +
    (kind === "gpt" ? `, cached-step vs full max diff ${stepWorst.toExponential(2)}` : "") + (pass ? "  PASS" : "  FAIL"));
}
console.log(ok ? "VERDICT: PASS" : "VERDICT: FAIL");
process.exit(ok ? 0 : 1);
