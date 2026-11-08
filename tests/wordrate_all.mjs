// Real-word rates (and the mean share of distinct words per sample, a check against repetition)
// for every setting the page offers, over seeds 1..50, with the page's own code
// and weights. Runs the settings in parallel processes and writes weights/wordrate.json.
//   node tests/wordrate_all.mjs [seeds=50]
import { execFile } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import { root } from "./load.mjs";
const seeds = Number(process.argv[2] || 50);
const KS = [8, 16, 32, 64, 128];
// split each setting's seeds into chunks of 5 so every core stays busy; largest jobs first
const jobs = [["gpt", 0, 0, 1, seeds]];
for (const K of [...KS].reverse()) for (const r of [1, 0])
  for (let a = 1; a <= seeds; a += 5) jobs.push(["diffusion", K, r, a, Math.min(seeds, a + 4)]);
const run = (j) => new Promise((ok, bad) =>
  execFile("node", [root + "tests/wordrate_one.mjs", ...j.map(String)], { maxBuffer: 1 << 26 },
    (e, out) => e ? bad(e) : ok(JSON.parse(out))));
const t0 = Date.now(), res = [];
let next = 0;
async function lane() { while (next < jobs.length) { const j = jobs[next++]; res.push([j, await run(j)]); } }
await Promise.all(Array.from({ length: Math.min(jobs.length, os.availableParallelism() - 1) }, lane));
const out = { seeds }, acc = {}, samples = {};
for (const [[kind, K, r], v] of res) {
  const key = kind === "gpt" ? "gpt" : `${K}${r ? "r" : ""}`;
  const a = (acc[key] ||= { real: 0, all: 0, remasks: 0, distinct: 0 });
  a.real += v.real; a.all += v.all; a.remasks += v.remasks; a.distinct += v.distinct;
  for (let i = 0; i < v.samples.length; i++) (samples[key] ||= {})[v.s0 + i] = v.samples[i];
}
for (const [key, a] of Object.entries(acc)) {
  if (key === "gpt") { out.gpt = a.real / a.all; out.gptWords = [a.real, a.all]; out.gptDistinct = a.distinct / seeds; }
  else out[key] = { diffusion: a.real / a.all, words: [a.real, a.all], distinct: a.distinct / seeds, remasksPerRun: a.remasks / seeds };
}
// every sample, so the number can be checked by hand
fs.writeFileSync(root + "tests/wordrate-samples.json", JSON.stringify(samples, null, 1));
fs.writeFileSync(root + "weights/wordrate.json", JSON.stringify(out, null, 1));
console.log(JSON.stringify(out), `${Math.round((Date.now() - t0) / 1000)} s`);
