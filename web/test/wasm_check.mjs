// Decode .volc chunk files with the viewer's wasm build and print sha256 prefixes
// of plain / CT-smooth (2.0, gated|zero_guard) / prediction-smooth (0.6, zero_guard)
// decodes, in the same format as web/test/ref_checksums.py.
// usage: node web/test/wasm_check.mjs <volcomp.wasm> <chunk.volc>...
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { basename } from "node:path";

const [wasmPath, ...files] = process.argv.slice(2);
let mem;
const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {
  env: { emscripten_notify_memory_growth: () => {} },
});
const X = instance.exports;
X._initialize();
const sha = (u8) => createHash("sha256").update(u8).digest("hex").slice(0, 16);
function run(enc, fn) {
  const p = X.vc_in(enc.length);
  new Uint8Array(X.memory.buffer, p, enc.length).set(enc);
  const t = performance.now();
  const st = fn(enc.length);
  const ms = performance.now() - t;
  if (st !== 0) return ["err" + st, ms];
  return [sha(new Uint8Array(X.memory.buffer, X.vc_out(), 128 ** 3)), ms];
}
for (const f of files) {
  const enc = readFileSync(f);
  const a = run(enc, (n) => X.vc_decode(n));
  const b = run(enc, (n) => X.vc_decode_smooth(n, 2.0, 3));
  const c = run(enc, (n) => X.vc_decode_smooth(n, 0.6, 1));
  console.log(basename(f), a[0], b[0], c[0], `# ms ${a[1].toFixed(1)} ${b[1].toFixed(1)} ${c[1].toFixed(1)}`);
}
