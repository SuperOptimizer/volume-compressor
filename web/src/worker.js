// volcomp decode worker. One WebAssembly instance per worker (web/src/shim.c).
// The main thread posts {type:"init", bytes} (the .wasm) once, then {type:"decode", ...}
// jobs; every job answers with the decoded 128^3 chunk (transferred) or an error.
"use strict";

let X = null; // wasm exports

function init(bytes) {
  // synchronous compile is allowed in a worker at any size
  const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
    env: { emscripten_notify_memory_growth: () => {} },
  });
  X = inst.exports;
  X._initialize();
  const cstr = (p) => {
    const m = new Uint8Array(X.memory.buffer);
    let e = p;
    while (m[e]) e++;
    return new TextDecoder().decode(m.subarray(p, e));
  };
  return { version: cstr(X.vc_version()), kernels: cstr(X.vc_kernels()) };
}

// smooth: 0 = plain volcomp_decode, else volcomp_decode_smooth(strength = smooth, flags)
function decode(enc, smooth, flags) {
  const n = enc.byteLength;
  const p = X.vc_in(n);
  if (!p) throw new Error("wasm: out of memory for a " + n + " byte chunk");
  new Uint8Array(X.memory.buffer, p, n).set(new Uint8Array(enc));
  const mode = X.vc_chunk_mode(n);
  if (mode < 0) throw new Error("not a volcomp chunk (bad magic)");
  const q = X.vc_stream_q(n);
  const t0 = performance.now();
  const st = smooth > 0 ? X.vc_decode_smooth(n, smooth, flags) : X.vc_decode(n);
  const ms = performance.now() - t0;
  if (st !== 0) {
    const names = ["OK", "ERR_ARG", "ERR_CORRUPT", "ERR_VERSION", "ERR_NOMEM", "ERR_SHORT_BUF"];
    throw new Error("volcomp decode failed: " + (names[st] || st));
  }
  // copy out of wasm memory (it can move on growth) into a transferable buffer
  const out = new Uint8Array(128 * 128 * 128);
  out.set(new Uint8Array(X.memory.buffer, X.vc_out(), out.length));
  return { out, ms, mode, q };
}

self.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === "init") {
    try {
      self.postMessage({ type: "ready", info: init(m.bytes) });
    } catch (e) {
      self.postMessage({ type: "ready", error: String(e && e.message || e) });
    }
    return;
  }
  if (m.type === "decode") {
    try {
      const r = decode(m.enc, m.smooth || 0, m.flags || 0);
      self.postMessage({ type: "done", id: m.id, out: r.out.buffer, ms: r.ms, mode: r.mode, q: r.q },
        [r.out.buffer]);
    } catch (e) {
      self.postMessage({ type: "done", id: m.id, error: String(e && e.message || e) });
    }
  }
};
