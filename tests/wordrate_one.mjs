// One setting of the real-word measure (see wordrate_all.mjs).
//   node tests/wordrate_one.mjs gpt|diffusion K remask firstSeed lastSeed
import fs from "node:fs";
import { H, weights, root } from "./load.mjs";
const text = fs.readFileSync(root + "data/input.txt", "utf8");
const chars = [...new Set(text)].sort();
export const words = (s) => (s.match(/[A-Za-z]+(?:'[A-Za-z]+)*/g) || []).map((w) => w.toLowerCase());
const vocab = new Set(words(text));
const [kind, K, remask, s0, s1] = process.argv.slice(2).map((v, i) => (i ? Number(v) : v));
const decode = (x) => Array.from(x.subarray(1), (c) => chars[c]).join("");
const samples = [];
let remasks = 0;
if (kind === "gpt") {
  const st = H.makeStepper(weights("gpt"));
  for (let s = s0; s <= s1; s++) samples.push(decode(H.write(st, { seed: s, temp: 0.8 })));
} else {
  const fwd = H.makeForward(weights("diffusion"), false);
  for (let s = s0; s <= s1; s++)
    samples.push(decode(H.diffuse(fwd, { K, remask: remask === 1, seed: s, temp: 0.8 }, (r) => { remasks += r.remasked.length; })));
}
let real = 0, all = 0, distinct = 0;
for (const smp of samples) {
  const ws = words(smp);
  for (const w of ws) { all++; if (vocab.has(w)) real++; }
  distinct += ws.length ? new Set(ws).size / ws.length : 0;  // summed per sample; averaged in wordrate_all
}
console.log(JSON.stringify({ kind, K, remask, s0, s1, real, all, distinct, remasks, samples }));
