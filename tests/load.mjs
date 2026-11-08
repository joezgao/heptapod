// Load src/model.js (a plain script) and the exported weights into Node.
import fs from "node:fs";
import vm from "node:vm";
const root = new URL("..", import.meta.url).pathname;
vm.runInThisContext(fs.readFileSync(root + "src/model.js", "utf8"));
export const H = globalThis.Heptapod;
export function weights(kind) {
  const q = fs.readFileSync(root + `weights/${kind}.q8`);
  const f = fs.readFileSync(root + `weights/${kind}.f32`);
  return H.load(new Int8Array(q.buffer, q.byteOffset, q.length),
                new Float32Array(f.buffer.slice(f.byteOffset, f.byteOffset + f.length)));
}
export { root };
