// Compute time of one run of each model in Node, with the page's code and weights (seed 1, K 32).
//   node tests/timing.mjs > docs/timing-<date>.json
import os from "node:os";
import { H, weights } from "./load.mjs";
const fwd = H.makeForward(weights("diffusion"), false);
const st = H.makeStepper(weights("gpt"));
fwd(new Uint8Array(256).fill(65));                    // warm up
let t = performance.now();
H.diffuse(fwd, { K: 32, seed: 1, remask: true });
const dms = performance.now() - t;
t = performance.now();
H.write(st, { seed: 1 });
const gms = performance.now() - t;
console.log(JSON.stringify({ date: new Date().toISOString(), node: process.version, cpu: os.cpus()[0].model,
  diffusion_K32_ms: Math.round(dms), diffusion_ms_per_pass: Math.round(dms / 32), gpt_255_letters_ms: Math.round(gms) }, null, 1));
