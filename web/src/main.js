// volcomp viewer: streams volcomp-coded zarr v3 volumes and prediction stores
// straight from an HTTP mirror (Range requests into sharded stores) and shows
// three orthogonal slices with a prediction overlay. See web/README.md.
//
// Layers, top to bottom:
//   http      fetchRange / listDir, byte and request counters
//   Store     zarr v3 group/array metadata, OME multiscales, level geometry
//   Loader    wanted chunks -> shard index reads -> coalesced Range reads -> decode pool
//   Pool      Web Workers, one volcomp wasm instance each (worker.js)
//   LRU       decoded chunk cache (and a smaller compressed-bytes cache)
//   render    per-view slice images, overlay blend, crosshair, minimap
//   ui        pickers, mouse, keyboard, status bar, URL hash state
"use strict";

const DEFAULT_BASE = "https://dl.ash2txt.org/community-uploads/forrest/volcomp/";
// used when the mirror's directory index cannot be read
const FALLBACK_SAMPLES = [
  "PHerc0009B", "PHerc0125", "PHerc0139", "PHerc0172", "PHerc0175A", "PHerc0175B", "PHerc0191",
  "PHerc0211", "PHerc0257", "PHerc0268", "PHerc0306B", "PHerc0332", "PHerc0343", "PHerc0343P",
  "PHerc0358", "PHerc0483A", "PHerc0483B", "PHerc0490A", "PHerc0490B", "PHerc0500P2", "PHerc0800",
  "PHerc0813", "PHerc0814", "PHerc0826", "PHerc0841", "PHerc0846A", "PHerc0846B", "PHerc1203",
  "PHerc1218", "PHerc1299", "PHerc1447", "PHerc1451", "PHerc1545", "PHerc1667", "PHercMAN5",
  "PHercMANB", "PHercMANBp", "PHercParis3", "PHercParis4",
];
const C = 128;                 // inner chunk edge
const CV = C * C * C;          // voxels per chunk
const MUL = [C * C, C, 1];     // z-major strides inside a chunk
const AXES = ["z", "y", "x"];
const UV = [[2, 1], [2, 0], [1, 0]]; // screen (u, v) world axes of the z, y and x views
const AXIS_COLORS = ["#4da3ff", "#5bd16b", "#ffb347"];
const ZERO = new Uint8Array(0); // cache sentinel: chunk absent from the store = all zero (air)
const ERR = new Uint8Array(0);  // cache sentinel: fetch or decode failed
const SMOOTH_CT_FLAGS = 3;      // VOLCOMP_SMOOTH_GATED | VOLCOMP_DEBLOCK_ZERO_GUARD (docs/deblocking.md)
const SMOOTH_PRED_FLAGS = 1;    // VOLCOMP_DEBLOCK_ZERO_GUARD (Gaussian seam blur)
const MAX_FETCH = 10;           // concurrent HTTP requests
const COALESCE_GAP = 64 << 10;  // join two chunk ranges when the hole between them is at most this
const COALESCE_MAX = 8 << 20;   // never read more than this in one Range request
const PREFETCH_MAX_VISIBLE = 96; // prefetch the neighbouring chunk layers only below this many visible chunks

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const joinUrl = (a, b) => a.replace(/\/+$/, "") + "/" + b.replace(/^\/+/, "");
function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1 << 20) return (n / 1024).toFixed(1) + " KB";
  if (n < 1 << 30) return (n / (1 << 20)).toFixed(1) + " MB";
  return (n / (1 << 30)).toFixed(2) + " GB";
}

// ---------------------------------------------------------------- crc32c
const CRC32C = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0x82f63b78 : c >>> 1;
    t[i] = c >>> 0;
  }
  return t;
})();
function crc32c(u8) {
  let c = 0xffffffff;
  for (let i = 0; i < u8.length; i++) c = CRC32C[(c ^ u8[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------- http
const stats = {
  bytes: 0, requests: 0, indexReads: 0, rangeReads: 0, notFound: 0, errors: 0,
  decoded: 0, decodeMs: 0, lastDecodeMs: 0, zeroChunks: 0, lastError: "",
};

// GET url (optionally one Range). Returns an ArrayBuffer holding exactly the
// requested bytes, or null on 404. A server that ignores Range (200) is handled
// by slicing the full body.
async function fetchRange(url, range) {
  stats.requests++;
  let r;
  try {
    r = await fetch(url, range ? { headers: { Range: range } } : {});
  } catch (e) {
    stats.errors++;
    throw new Error("network/CORS error for " + url);
  }
  if (r.status === 404 || r.status === 403 && /\/c\//.test(url)) { stats.notFound++; return null; }
  if (!r.ok) { stats.errors++; throw new Error("HTTP " + r.status + " for " + url); }
  let buf = await r.arrayBuffer();
  stats.bytes += buf.byteLength;
  if (range && r.status === 200) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (m && m[1] === "") buf = buf.slice(Math.max(0, buf.byteLength - +m[2]));
    else if (m) buf = buf.slice(+m[1], m[2] === "" ? undefined : +m[2] + 1);
  }
  return buf;
}
async function fetchJSON(url) {
  const b = await fetchRange(url);
  if (!b) throw new Error("not found: " + url);
  return JSON.parse(new TextDecoder().decode(b));
}
// Subdirectory names from an HTML autoindex page (nginx/apache style).
async function listDir(url) {
  const b = await fetchRange(url.replace(/\/?$/, "/"));
  if (!b) throw new Error("not found: " + url);
  const html = new TextDecoder().decode(b);
  const out = [];
  for (const m of html.matchAll(/href="([^"?#]+)"/g)) {
    let h = decodeURIComponent(m[1]);
    if (h.startsWith("/") || h.startsWith("..") || /^[a-z]+:/i.test(h)) continue;
    if (!h.endsWith("/")) continue;
    h = h.slice(0, -1);
    if (h && !out.includes(h)) out.push(h);
  }
  return out;
}

// ---------------------------------------------------------------- LRU
class LRU {
  constructor(cap) { this.cap = cap; this.map = new Map(); this.bytes = 0; }
  get(k) {
    const e = this.map.get(k);
    if (e === undefined) return undefined;
    this.map.delete(k);
    this.map.set(k, e);
    return e.v;
  }
  peek(k) { const e = this.map.get(k); return e && e.v; }
  has(k) { return this.map.has(k); }
  set(k, v, size) {
    const old = this.map.get(k);
    if (old) { this.bytes -= old.n; this.map.delete(k); }
    this.map.set(k, { v, n: size });
    this.bytes += size;
    this.trim();
  }
  trim() {
    for (const [key, e] of this.map) {
      if (this.bytes <= this.cap) break;
      this.map.delete(key);
      this.bytes -= e.n;
    }
  }
  clear() { this.map.clear(); this.bytes = 0; }
  get size() { return this.map.size; }
}
const cache = new LRU(1536 << 20);  // decoded 128^3 chunks
const ccache = new LRU(256 << 20);  // compressed chunk bytes (toggling deblock re-decodes without re-downloading)

// ---------------------------------------------------------------- Store
let storeIds = new Map(); // url -> small int (cache keys)

class Store {
  constructor(url) {
    this.url = url.replace(/\/+$/, "");
    this.name = decodeURIComponent(this.url.split("/").pop());
    if (!storeIds.has(this.url)) storeIds.set(this.url, storeIds.size);
    this.id = storeIds.get(this.url);
    this.levels = [];
    this.unitUm = false;  // OME scales are in micrometres (prediction stores)
    this.pitch0 = null;   // um per level-0 voxel, when known
  }
  async open() {
    const root = await fetchJSON(joinUrl(this.url, "zarr.json"));
    this.attrs = root.attributes || {};
    let datasets;
    if (root.node_type === "array") {
      datasets = [{ path: "", scale: [1, 1, 1] }];
      this.levels = [];
    } else {
      const ms = (this.attrs.ome && this.attrs.ome.multiscales) || this.attrs.multiscales;
      if (!ms || !ms.length) throw new Error("no OME multiscales in " + this.url + "/zarr.json");
      const m = ms[0];
      this.unitUm = !!(m.axes && m.axes.some((a) => /micro|um|µm/i.test(a.unit || "")));
      datasets = m.datasets.map((d) => {
        const t = (d.coordinateTransformations || []).find((t) => t.type === "scale");
        return { path: d.path, scale: t ? t.scale.slice(-3) : [1, 1, 1] };
      });
    }
    const metas = await Promise.all(datasets.map((d) =>
      d.path === "" ? Promise.resolve(root) : fetchJSON(joinUrl(joinUrl(this.url, d.path), "zarr.json"))));
    this.levels = datasets.map((d, i) => parseLevel(joinUrl(this.url, d.path), d.path, d.scale, metas[i]));
    // level-0 pitch in micrometres: from the scales if they carry the unit, else
    // from the name ("...-8.640um-..."), else from the store's own attributes
    const vc = this.attrs.volcomp || {};
    if (this.unitUm) this.pitch0 = this.levels[0].scale.slice();
    else {
      const m = /(\d+(?:\.\d+)?)\s*um/i.exec(this.name);
      const p = m ? +m[1] : vc.native_voxel_size_um || null;
      if (p) this.pitch0 = [p, p, p];
    }
    return this;
  }
  // micrometres per voxel of level li, per axis (null if unknown)
  pitchUm(li) {
    const L = this.levels[li], L0 = this.levels[0];
    if (this.unitUm) return L.scale.slice();
    if (!this.pitch0) return null;
    return L.scale.map((s, a) => (s / L0.scale[a]) * this.pitch0[a]);
  }
}

function parseLevel(url, path, scale, meta) {
  if (meta.node_type !== "array") throw new Error(url + " is not an array");
  if (meta.data_type !== "uint8") throw new Error(url + ": dtype " + meta.data_type + " (only uint8)");
  if (meta.shape.length !== 3) throw new Error(url + ": not 3-D");
  const grid = meta.chunk_grid.configuration.chunk_shape;
  const kenc = meta.chunk_key_encoding || { name: "default" };
  const sep = (kenc.configuration && kenc.configuration.separator) || (kenc.name === "v2" ? "." : "/");
  const L = {
    url, path, scale, shape: meta.shape, fill: meta.fill_value || 0,
    keyPrefix: kenc.name === "v2" ? "" : "c" + sep, sep,
    sharded: false, shard: grid, inner: grid, cps: [1, 1, 1], nInner: 1, indexAtEnd: true, q: null,
  };
  let codecs = meta.codecs || [];
  const sh = codecs.find((c) => c.name === "sharding_indexed");
  if (sh) {
    const cf = sh.configuration;
    L.sharded = true;
    L.inner = cf.chunk_shape;
    L.cps = grid.map((g, i) => g / cf.chunk_shape[i]);
    L.nInner = L.cps[0] * L.cps[1] * L.cps[2];
    L.indexAtEnd = (cf.index_location || "end") === "end";
    L.indexCrc = (cf.index_codecs || []).some((c) => c.name === "crc32c");
    codecs = cf.codecs || [];
  }
  const vc = codecs.filter((c) => c.name !== "bytes");
  if (vc.length !== 1 || vc[0].name !== "volcomp")
    throw new Error(url + ": inner codecs " + JSON.stringify(codecs.map((c) => c.name)) + " (need exactly volcomp)");
  if (L.inner.some((n) => n !== C)) throw new Error(url + ": inner chunk " + L.inner + " (need 128^3)");
  L.q = vc[0].configuration && vc[0].configuration.q;
  L.mode = vc[0].configuration && vc[0].configuration.mode;
  L.grid = L.shape.map((n) => Math.ceil(n / C)); // inner chunks per axis
  return L;
}

// ---------------------------------------------------------------- decode pool
class Pool {
  constructor(wasmBytes, workerSrc, n) {
    this.n = n;
    this.free = [];
    this.jobs = [];
    this.pending = new Map();
    this.nextId = 1;
    this.info = null;
    this.ready = new Promise((resolve, reject) => {
      const url = URL.createObjectURL(new Blob([workerSrc], { type: "text/javascript" }));
      let up = 0;
      for (let i = 0; i < n; i++) {
        const w = new Worker(url);
        w.onmessage = (ev) => {
          const m = ev.data;
          if (m.type === "ready") {
            if (m.error) { reject(new Error(m.error)); return; }
            this.info = m.info;
            this.free.push(w);
            if (++up === n) resolve(this.info);
            this.pump();
          } else if (m.type === "done") {
            const p = this.pending.get(m.id);
            this.pending.delete(m.id);
            this.free.push(w);
            if (m.error) p.reject(new Error(m.error));
            else p.resolve({ out: new Uint8Array(m.out), ms: m.ms, mode: m.mode, q: m.q });
            this.pump();
          }
        };
        w.onerror = (e) => reject(new Error("worker: " + e.message));
        w.postMessage({ type: "init", bytes: wasmBytes });
      }
    });
  }
  decode(enc, smooth, flags) {
    return new Promise((resolve, reject) => {
      this.jobs.push({ enc, smooth, flags, resolve, reject });
      this.pump();
    });
  }
  pump() {
    while (this.free.length && this.jobs.length) {
      const w = this.free.pop(), j = this.jobs.shift(), id = this.nextId++;
      this.pending.set(id, j);
      w.postMessage({ type: "decode", id, enc: j.enc, smooth: j.smooth, flags: j.flags }, [j.enc]);
    }
  }
  get busy() { return this.pending.size + this.jobs.length; }
}

// ---------------------------------------------------------------- loader
// A wanted item: {key, ckey, st, li, c:[cz,cy,cx], smooth, flags, prio}
class Loader {
  constructor(pool) {
    this.pool = pool;
    this.queue = [];
    this.inflight = new Set();
    this.shards = new Map(); // shard url -> {state, dv (index DataView)}
    this.nFetch = 0;
  }
  want(items) {
    items.sort((a, b) => a.prio - b.prio);
    const seen = new Set();
    this.queue = items.filter((it) => {
      if (seen.has(it.key) || this.inflight.has(it.key) || cache.has(it.key)) return false;
      seen.add(it.key);
      return true;
    });
    this.pump();
  }
  shardOf(it) {
    const L = it.st.levels[it.li];
    const s = [0, 1, 2].map((a) => Math.floor(it.c[a] / L.cps[a]));
    const url = joinUrl(L.url, L.keyPrefix + s.join(L.sep));
    const l = [0, 1, 2].map((a) => it.c[a] - s[a] * L.cps[a]);
    return { url, L, inner: (l[0] * L.cps[1] + l[1]) * L.cps[2] + l[2] };
  }
  finish(it, arr, size) {
    this.inflight.delete(it.key);
    cache.set(it.key, arr, size);
    requestRender();
  }
  fail(it, err) {
    stats.lastError = String(err && err.message || err);
    this.finish(it, ERR, 64);
  }
  decode(it, enc) {
    this.pool.decode(enc.slice(0), it.smooth, it.flags).then((r) => {
      stats.decoded++;
      stats.decodeMs += r.ms;
      stats.lastDecodeMs = r.ms;
      this.finish(it, r.out, CV);
    }, (e) => this.fail(it, e));
  }
  pump() {
    const q = this.queue;
    let i = 0;
    while (i < q.length && this.nFetch < MAX_FETCH) {
      const it = q[i];
      if (cache.has(it.key) || this.inflight.has(it.key)) { q.splice(i, 1); continue; }
      const enc = ccache.get(it.ckey);
      if (enc) { q.splice(i, 1); this.inflight.add(it.key); this.decode(it, enc); continue; }
      const L = it.st.levels[it.li];
      const sh = this.shardOf(it);
      if (!L.sharded || L.nInner === 1) {
        // one file per chunk (unsharded, or a shard holding a single inner chunk,
        // as in the coarse levels of the prediction stores): one GET instead of
        // index + range; the file is that chunk plus a 20-byte index
        q.splice(i, 1);
        this.inflight.add(it.key);
        this.nFetch++;
        fetchRange(sh.url).then((b) => {
          if (!b) { stats.zeroChunks++; this.finish(it, ZERO, 64); return; }
          if (L.sharded) {
            const n = 16 + (L.indexCrc ? 4 : 0);
            if (b.byteLength < n) throw new Error("short shard " + sh.url);
            const dv = new DataView(b, L.indexAtEnd ? b.byteLength - n : 0, 16);
            if (dv.getUint32(0, true) === 0xffffffff && dv.getUint32(4, true) === 0xffffffff) {
              stats.zeroChunks++; this.finish(it, ZERO, 64); return;
            }
            const off = dv.getUint32(4, true) * 4294967296 + dv.getUint32(0, true);
            const len = dv.getUint32(12, true) * 4294967296 + dv.getUint32(8, true);
            b = b.slice(off, off + len);
          }
          ccache.set(it.ckey, b, b.byteLength);
          this.decode(it, b);
        }).catch((e) => this.fail(it, e)).finally(() => { this.nFetch--; this.pump(); });
        continue;
      }
      const s = this.shards.get(sh.url);
      if (!s) { this.readIndex(sh.url, L); i++; continue; }
      if (s.state === "pending") { i++; continue; }
      if (s.state !== "ok") { // missing shard (all zero) or unreadable index
        q.splice(i, 1);
        if (s.state === "missing") { stats.zeroChunks++; cache.set(it.key, ZERO, 64); requestRender(); }
        else this.fail(it, s.error);
        continue;
      }
      // every queued chunk of this shard (same smoothing) in one pass, coalesced
      const group = [];
      for (let j = i; j < q.length; j++) {
        const o = q[j];
        if (o.st !== it.st || o.li !== it.li || o.smooth !== it.smooth) continue;
        const so = this.shardOf(o);
        if (so.url !== sh.url) continue;
        if (cache.has(o.key) || this.inflight.has(o.key)) continue;
        group.push([o, so.inner]);
      }
      const gset = new Set(group.map((g) => g[0]));
      this.queue = q.filter((o) => !gset.has(o));
      this.readChunks(sh.url, s.dv, group);
      return this.pump();
    }
  }
  readIndex(url, L) {
    const s = { state: "pending" };
    this.shards.set(url, s);
    const n = L.nInner * 16 + (L.indexCrc ? 4 : 0);
    this.nFetch++;
    stats.indexReads++;
    fetchRange(url, L.indexAtEnd ? "bytes=-" + n : "bytes=0-" + (n - 1)).then((b) => {
      if (!b) { s.state = "missing"; return; }
      if (b.byteLength !== n) throw new Error("short shard index (" + b.byteLength + " of " + n + ") " + url);
      const u8 = new Uint8Array(b);
      if (L.indexCrc) {
        const want = new DataView(b).getUint32(n - 4, true);
        if (crc32c(u8.subarray(0, n - 4)) !== want) throw new Error("shard index crc32c mismatch " + url);
      }
      s.dv = new DataView(b);
      s.state = "ok";
    }).catch((e) => { s.state = "error"; s.error = e; stats.lastError = String(e.message || e); })
      .finally(() => { this.nFetch--; this.pump(); requestRender(); });
  }
  readChunks(url, dv, group) {
    const parts = [];
    for (const [it, k] of group) {
      const lo = dv.getUint32(k * 16, true), hi = dv.getUint32(k * 16 + 4, true);
      const nlo = dv.getUint32(k * 16 + 8, true), nhi = dv.getUint32(k * 16 + 12, true);
      if (lo === 0xffffffff && hi === 0xffffffff && nlo === 0xffffffff && nhi === 0xffffffff) {
        stats.zeroChunks++;
        cache.set(it.key, ZERO, 64);
        continue;
      }
      const off = hi * 4294967296 + lo, n = nhi * 4294967296 + nlo;
      if (n === 0) { cache.set(it.key, ZERO, 64); continue; }
      parts.push({ it, off, n });
      this.inflight.add(it.key);
    }
    requestRender();
    parts.sort((a, b) => a.off - b.off);
    const runs = [];
    for (const p of parts) {
      const r = runs[runs.length - 1];
      if (r && p.off - r.end <= COALESCE_GAP && p.off + p.n - r.start <= COALESCE_MAX) {
        r.parts.push(p);
        r.end = Math.max(r.end, p.off + p.n);
      } else runs.push({ start: p.off, end: p.off + p.n, parts: [p] });
    }
    for (const r of runs) {
      this.nFetch++;
      stats.rangeReads++;
      fetchRange(url, "bytes=" + r.start + "-" + (r.end - 1)).then((b) => {
        if (!b || b.byteLength < r.end - r.start) throw new Error("short range read " + url);
        for (const p of r.parts) {
          const enc = b.slice(p.off - r.start, p.off - r.start + p.n);
          ccache.set(p.it.ckey, enc, enc.byteLength);
          this.decode(p.it, enc);
        }
      }).catch((e) => { for (const p of r.parts) this.fail(p.it, e); })
        .finally(() => { this.nFetch--; this.pump(); });
    }
  }
  reset() { this.queue = []; this.shards.clear(); }
}

// ---------------------------------------------------------------- state
const S = {
  base: DEFAULT_BASE,
  sample: "",
  ct: null,        // base Store
  pred: null,      // overlay Store
  f: null,         // per store id: per level: world voxels per level voxel [z,y,x]
  cursor: [0, 0, 0],
  level: "auto",   // CT level: "auto" or an index
  predLevel: "auto",
  views: [],
  active: 0,
  maximized: -1,
  win: 255, lev: 128,
  ov: { on: true, opacity: 0.7, thresh: false },
  smooth: { on: false, pred: true, ct: 2.0, pr: 0.6 },
  crosshair: true,
  hover: null,
  warn: "",
};
let pool = null, loader = null;

// world = the base store's level-0 voxel grid. f(st, li)[a] = world voxels per voxel of that level.
function levelFactor(st, li) {
  const L = st.levels[li];
  if (st === S.ct || !S.ct) return L.scale.map((s, a) => s / st.levels[0].scale[a]);
  const wp = worldPitch();
  const p = st.pitchUm(li);
  if (p && wp) return p.map((x, a) => x / wp[a]);
  // no physical units anywhere: assume the overlay's level 0 is the base's level 0 grid
  return L.scale.map((s, a) => s / st.levels[0].scale[a]);
}
function worldPitch() {
  if (!S.ct) return null;
  if (S.ct.pitch0) return S.ct.pitch0;
  if (S.pred && S.pred.unitUm) return S.pred.levels[0].scale; // same grid assumed
  return null;
}
function worldShape() { return S.ct ? S.ct.levels[0].shape : [1, 1, 1]; }

// ---------------------------------------------------------------- LUTs
let grayLut = new Uint32Array(256);
let ovLut = new Uint32Array(65536); // (p << 8 | ct) -> RGBA
const MISSING_RGBA = pack(26, 30, 40);
const ERR_RGBA = pack(70, 18, 18);
function pack(r, g, b) { return (255 << 24 | b << 16 | g << 8 | r) >>> 0; }
function rebuildLuts() {
  const lo = S.lev - S.win / 2, w = Math.max(1, S.win);
  const g = new Uint8Array(256);
  for (let v = 0; v < 256; v++) g[v] = clamp(Math.round(((v - lo) / w) * 255), 0, 255);
  for (let v = 0; v < 256; v++) grayLut[v] = pack(g[v], g[v], g[v]);
  const op = S.ov.opacity;
  for (let p = 0; p < 256; p++) {
    const a = S.ov.thresh ? (p >= 128 ? op : 0) : (p / 255) * op;
    for (let v = 0; v < 256; v++) {
      const k = g[v] * (1 - a);
      ovLut[(p << 8) | v] = pack(Math.round(k + 255 * a), Math.round(k), Math.round(k));
    }
  }
  requestRender();
}

// ---------------------------------------------------------------- render
let renderQueued = false;
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; renderAll(); });
}

function smoothFor(st) {
  if (!S.smooth.on) return [0, 0];
  if (st === S.ct) return [S.smooth.ct, SMOOTH_CT_FLAGS];
  return S.smooth.pred ? [S.smooth.pr, SMOOTH_PRED_FLAGS] : [0, 0];
}
function chunkKeys(st, li, c, smooth) {
  const L = st.levels[li];
  const s = [0, 1, 2].map((a) => Math.floor(c[a] / L.cps[a]));
  const l = [0, 1, 2].map((a) => c[a] - s[a] * L.cps[a]);
  const inner = (l[0] * L.cps[1] + l[1]) * L.cps[2] + l[2];
  const ckey = st.id + "|" + li + "|" + s.join(".") + "|" + inner; // (store, level, shard, chunk)
  return [ckey + "|" + smooth, ckey];
}

// Pick the CT level of a view: explicit, or the coarsest whose voxel is at most one screen pixel.
function viewLevel(V) {
  const n = S.ct.levels.length;
  if (S.level !== "auto") return clamp(+S.level, 0, n - 1);
  const [ua] = UV[V.axis];
  let best = 0;
  for (let li = 0; li < n; li++) if (levelFactor(S.ct, li)[ua] <= 1 / V.zoom + 1e-6) best = li;
  return best;
}
// The overlay level closest in pitch to CT level li (ties: the finer one).
function predLevelFor(li) {
  if (!S.pred) return -1;
  if (S.predLevel !== "auto") return clamp(+S.predLevel, 0, S.pred.levels.length - 1);
  const want = Math.log(levelFactor(S.ct, li)[0]);
  let best = 0, bd = Infinity;
  S.pred.levels.forEach((_, i) => {
    const d = Math.abs(Math.log(levelFactor(S.pred, i)[0]) - want);
    if (d < bd - 1e-9) { bd = d; best = i; }
  });
  return best;
}

// Build a table of the chunk arrays covering a slice region; missing ones go to `wanted`.
// idxU / idxV: level voxel index per buffer column / row (-1 = outside).
function chunkTable(st, li, axis, s, idxU, idxV, wanted, prio0, center) {
  const [ua, va] = UV[axis];
  const L = st.levels[li];
  let u0 = Infinity, u1 = -1, v0 = Infinity, v1 = -1;
  for (const x of idxU) if (x >= 0) { u0 = Math.min(u0, x >> 7); u1 = Math.max(u1, x >> 7); }
  for (const x of idxV) if (x >= 0) { v0 = Math.min(v0, x >> 7); v1 = Math.max(v1, x >> 7); }
  if (u1 < 0 || v1 < 0 || s < 0 || s >= L.shape[axis]) return null;
  const nu = u1 - u0 + 1, nv = v1 - v0 + 1, cs = s >> 7;
  const [smooth, flags] = smoothFor(st);
  const tab = new Array(nu * nv);
  const miss = [];
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const c = [0, 0, 0];
    c[axis] = cs; c[ua] = u0 + i; c[va] = v0 + j;
    const [key, ckey] = chunkKeys(st, li, c, smooth);
    const a = cache.get(key);
    tab[j * nu + i] = a;
    if (a === undefined) {
      const du = u0 + i + 0.5 - center[0], dv = v0 + j + 0.5 - center[1];
      miss.push({ key, ckey, st, li, c, smooth, flags, prio: prio0 + Math.hypot(du, dv) });
    }
  }
  wanted.visible = (wanted.visible || 0) + nu * nv;
  const cap = Math.max(16, Math.floor(cache.cap / CV / 6));
  if (miss.length > cap) {
    miss.sort((a, b) => a.prio - b.prio);
    S.warn = `view needs ${nu * nv} chunks; loading the ${cap} nearest the centre (zoom in or pick a coarser level)`;
    miss.length = cap;
  }
  wanted.push(...miss);
  // prefetch the neighbouring chunk layers along the slice axis (renderAll trims these to the cache budget)
  if (wanted.pre && nu * nv <= PREFETCH_MAX_VISIBLE) {
    for (const d of [1, -1]) {
      const cn = cs + d;
      if (cn < 0 || cn >= L.grid[axis]) continue;
      for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
        const c = [0, 0, 0];
        c[axis] = cn; c[ua] = u0 + i; c[va] = v0 + j;
        const [key, ckey] = chunkKeys(st, li, c, smooth);
        if (!cache.has(key)) wanted.pre.push({ key, ckey, st, li, c, smooth, flags, prio: 1e6 + prio0 + Math.abs((s & 127) - (d > 0 ? 128 : -1)) });
      }
    }
  }
  return { tab, u0, v0, nu, cs };
}

// Render one slice of the base (and overlay) into an RGBA buffer of bw x bh
// pixels covering world rectangle [wu0, wu0+wspanU) x [wv0, wv0+wspanV).
function sliceImage(axis, sWorld, li, wu0, wv0, wspanU, wspanV, bwMax, bhMax, wanted, prio0, hist) {
  const [ua, va] = UV[axis];
  const L = S.ct.levels[li], f = levelFactor(S.ct, li);
  // visible level-voxel range
  const i0 = Math.max(0, Math.floor(wu0 / f[ua])), i1 = Math.min(L.shape[ua], Math.ceil((wu0 + wspanU) / f[ua]));
  const j0 = Math.max(0, Math.floor(wv0 / f[va])), j1 = Math.min(L.shape[va], Math.ceil((wv0 + wspanV) / f[va]));
  if (i1 <= i0 || j1 <= j0) return null;
  const nw = i1 - i0, nh = j1 - j0;
  const bw = Math.max(1, Math.min(nw, Math.ceil(bwMax))), bh = Math.max(1, Math.min(nh, Math.ceil(bhMax)));
  const s = Math.floor(sWorld / f[axis]);
  const colIdx = new Int32Array(bw), rowIdx = new Int32Array(bh);
  for (let i = 0; i < bw; i++) colIdx[i] = i0 + Math.floor(((i + 0.5) * nw) / bw);
  for (let j = 0; j < bh; j++) rowIdx[j] = j0 + Math.floor(((j + 0.5) * nh) / bh);
  const center = [(wu0 + wspanU / 2) / f[ua] / C, (wv0 + wspanV / 2) / f[va] / C];
  const T = chunkTable(S.ct, li, axis, s, colIdx, rowIdx, wanted, prio0, center);
  if (!T) return null;
  const out = new ImageData(bw, bh);
  const px = new Uint32Array(out.data.buffer);
  const colC = new Int32Array(bw), colO = new Int32Array(bw), rowC = new Int32Array(bh), rowO = new Int32Array(bh);
  for (let i = 0; i < bw; i++) { colC[i] = (colIdx[i] >> 7) - T.u0; colO[i] = (colIdx[i] & 127) * MUL[ua]; }
  for (let j = 0; j < bh; j++) { rowC[j] = ((rowIdx[j] >> 7) - T.v0) * T.nu; rowO[j] = (rowIdx[j] & 127) * MUL[va]; }
  const sO = (s & 127) * MUL[axis];

  // overlay geometry: world centre of each column/row -> overlay voxel
  let P = null;
  const pli = S.ov.on ? predLevelFor(li) : -1;
  if (pli >= 0) {
    const PL = S.pred.levels[pli], pf = levelFactor(S.pred, pli);
    const pcol = new Int32Array(bw), prow = new Int32Array(bh);
    for (let i = 0; i < bw; i++) {
      const x = Math.floor(((colIdx[i] + 0.5) * f[ua]) / pf[ua]);
      pcol[i] = x < PL.shape[ua] ? x : -1;
    }
    for (let j = 0; j < bh; j++) {
      const y = Math.floor(((rowIdx[j] + 0.5) * f[va]) / pf[va]);
      prow[j] = y < PL.shape[va] ? y : -1;
    }
    const ps = Math.floor(((s + 0.5) * f[axis]) / pf[axis]);
    const pcenter = [center[0] * f[ua] / pf[ua], center[1] * f[va] / pf[va]];
    const PT = ps < PL.shape[axis] ? chunkTable(S.pred, pli, axis, ps, pcol, prow, wanted, prio0 + 0.5, pcenter) : null;
    if (PT) {
      const pc = new Int32Array(bw), po = new Int32Array(bw), rc = new Int32Array(bh), ro = new Int32Array(bh);
      for (let i = 0; i < bw; i++) {
        pc[i] = pcol[i] < 0 ? -1 : (pcol[i] >> 7) - PT.u0;
        po[i] = (pcol[i] & 127) * MUL[ua];
      }
      for (let j = 0; j < bh; j++) {
        rc[j] = prow[j] < 0 ? -1 : ((prow[j] >> 7) - PT.v0) * PT.nu;
        ro[j] = (prow[j] & 127) * MUL[va];
      }
      P = { T: PT, pc, po, rc, ro, sO: (ps & 127) * MUL[axis] };
    }
  }

  const tab = T.tab;
  for (let j = 0; j < bh; j++) {
    const rbase = rowC[j], roff = sO + rowO[j], obase = j * bw;
    let prow = null, proff = 0, prc = -1;
    if (P && P.rc[j] >= 0) { prc = P.rc[j]; proff = P.sO + P.ro[j]; }
    let lastC = -1, arr = undefined, plastC = -1, parr = undefined;
    for (let i = 0; i < bw; i++) {
      const cc = colC[i];
      if (cc !== lastC) { lastC = cc; arr = tab[rbase + cc]; }
      let rgba;
      if (arr === undefined) rgba = MISSING_RGBA;
      else if (arr === ERR) rgba = ERR_RGBA;
      else {
        const v = arr === ZERO ? 0 : arr[roff + colO[i]];
        if (hist) hist[v]++;
        rgba = grayLut[v];
        if (prc >= 0) {
          const pci = P.pc[i];
          if (pci >= 0) {
            if (pci !== plastC) { plastC = pci; parr = P.T.tab[prc + pci]; }
            if (parr !== undefined && parr.length) {
              const p = parr[proff + P.po[i]];
              if (p) rgba = ovLut[(p << 8) | v];
            }
          }
        }
      }
      px[obase + i] = rgba;
    }
  }
  return { img: out, i0, j0, nw, nh, f, s, L };
}

function renderView(V, wanted) {
  const cv = V.canvas, ctx = V.ctx;
  const W = cv.width, H = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, W, H);
  if (!S.ct || W < 2 || H < 2) return;
  const [ua, va] = UV[V.axis];
  const z = V.zoom;
  const wu0 = V.center[0] - W / 2 / z, wv0 = V.center[1] - H / 2 / z;
  const li = viewLevel(V);
  V.li = li;
  const prio0 = V.axis === S.active ? 0 : 10;
  const hist = V.axis === S.active ? (S.hist = new Uint32Array(256)) : null;
  const r = sliceImage(V.axis, S.cursor[V.axis], li, wu0, wv0, W / z, H / z, W, H, wanted, prio0, hist);
  const ws = worldShape();
  if (r) {
    if (!V.buf || V.buf.width !== r.img.width || V.buf.height !== r.img.height) {
      V.buf = OffscreenCanvasOr(r.img.width, r.img.height);
    }
    V.buf.getContext("2d").putImageData(r.img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    const dx = (r.i0 * r.f[ua] - wu0) * z, dy = (r.j0 * r.f[va] - wv0) * z;
    ctx.drawImage(V.buf, dx, dy, r.nw * r.f[ua] * z, r.nh * r.f[va] * z);
  }
  // volume outline
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = 1;
  ctx.strokeRect((0 - wu0) * z + 0.5, (0 - wv0) * z + 0.5, ws[ua] * z, ws[va] * z);
  // crosshair: the other two views' slice planes
  if (S.crosshair) {
    const x = (S.cursor[ua] - wu0) * z, y = (S.cursor[va] - wv0) * z;
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = AXIS_COLORS[ua];
    ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, H); ctx.stroke();
    ctx.strokeStyle = AXIS_COLORS[va];
    ctx.beginPath(); ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(W, Math.round(y) + 0.5); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  const L = S.ct.levels[li];
  const f = levelFactor(S.ct, li);
  const sl = Math.floor(S.cursor[V.axis] / f[V.axis]);
  V.label.textContent = `${AXES[V.axis]} ${sl} / ${L.shape[V.axis] - 1}  ·  level ${L.path}${S.level === "auto" ? " (auto)" : ""}` +
    `  ·  ${(z * f[ua]).toFixed(z * f[ua] < 1 ? 2 : 1)} px/voxel` +
    (S.pred && S.ov.on ? `  ·  overlay ${S.pred.levels[predLevelFor(li)].path}` : "");
}

// OffscreenCanvas where available (it is in every current browser), else a detached <canvas>
function OffscreenCanvasOr(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

// minimap: the active view's slice at the coarsest CT level, with the viewport rectangle
function renderMinimap(wanted) {
  const cv = $("minimap"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, W, H);
  if (!S.ct) return;
  const V = S.views[S.active];
  const [ua, va] = UV[V.axis];
  const ws = worldShape();
  const z = Math.min(W / ws[ua], H / ws[va]);
  const ox = (W - ws[ua] * z) / 2, oy = (H - ws[va] * z) / 2;
  S.mini = { z, ox, oy, axis: V.axis };
  // coarsest level whose voxel is still at most ~one minimap pixel, but at least the view's level
  let li = S.ct.levels.length - 1;
  while (li > 0 && levelFactor(S.ct, li)[ua] * z > 1.5) li--;
  const r = sliceImage(V.axis, S.cursor[V.axis], li, 0, 0, ws[ua], ws[va], W, H, wanted, 5, null);
  if (r) {
    const b = OffscreenCanvasOr(r.img.width, r.img.height);
    b.getContext("2d").putImageData(r.img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(b, ox + r.i0 * r.f[ua] * z, oy + r.j0 * r.f[va] * z, r.nw * r.f[ua] * z, r.nh * r.f[va] * z);
  }
  const vw = V.canvas.width / V.zoom, vh = V.canvas.height / V.zoom;
  ctx.strokeStyle = AXIS_COLORS[V.axis];
  ctx.lineWidth = 1.5;
  ctx.strokeRect(ox + (V.center[0] - vw / 2) * z, oy + (V.center[1] - vh / 2) * z, vw * z, vh * z);
  $("minimap-label").textContent = `${AXES[V.axis]} view · level ${S.ct.levels[li].path}`;
}

let lastStatus = 0;
function renderAll() {
  const t0 = performance.now();
  const wanted = [];
  wanted.pre = [];
  S.warn = "";
  for (const V of S.views) {
    if (S.maximized >= 0 && V.axis !== S.maximized) continue;
    renderView(V, wanted);
  }
  renderMinimap(wanted);
  // prefetch only what fits next to the visible chunks in ~60% of the cache,
  // so prefetched chunks never evict the ones on screen
  const budget = Math.floor((cache.cap / CV) * 0.6) - (wanted.visible || 0);
  if (budget > 0) {
    wanted.pre.sort((a, b) => a.prio - b.prio);
    wanted.push(...wanted.pre.slice(0, budget));
  }
  if (loader) loader.want(wanted);
  stats.renderMs = performance.now() - t0;
  updateStatus();
}

// ---------------------------------------------------------------- status
function updateStatus() {
  const avg = stats.decoded ? stats.decodeMs / stats.decoded : 0;
  const parts = [
    `fetched ${fmtBytes(stats.bytes)} in ${stats.requests} req (${stats.indexReads} index, ${stats.rangeReads} range, ${stats.notFound} 404)`,
    `decoded ${stats.decoded} chunks · ${avg.toFixed(1)} ms avg · ${stats.lastDecodeMs.toFixed(1)} ms last · ${stats.zeroChunks} air`,
    `cache ${fmtBytes(cache.bytes)} / ${fmtBytes(cache.cap)} (${cache.size})`,
    `render ${(stats.renderMs || 0).toFixed(0)} ms`,
    loader ? `queue ${loader.queue.length} · fetching ${loader.nFetch} · decoding ${pool.busy}` : "",
  ];
  $("status-main").textContent = parts.filter(Boolean).join("   |   ");
  const w = S.warn || stats.lastError;
  $("status-warn").textContent = w;
  $("status-warn").title = w;
  if (S.hover) {
    const h = S.hover;
    const c = h.w.map((x) => Math.floor(x));
    let txt = `cursor z ${Math.floor(S.cursor[0])} y ${Math.floor(S.cursor[1])} x ${Math.floor(S.cursor[2])}` +
      `   ·   mouse z ${c[0]} y ${c[1]} x ${c[2]}`;
    const v = sampleAt(S.ct, h.li, h.w);
    if (v !== undefined) txt += `  CT ${v}`;
    if (S.pred) {
      const pli = predLevelFor(h.li);
      const p = sampleAt(S.pred, pli, h.w);
      if (p !== undefined) txt += `  p ${(p / 255).toFixed(3)}`;
    }
    $("info-hover").textContent = txt;
  }
}
// decoded value at a world point from the cache (undefined if not loaded)
function sampleAt(st, li, w) {
  if (!st || li < 0) return undefined;
  const L = st.levels[li], f = levelFactor(st, li);
  const v = w.map((x, a) => Math.floor(x / f[a]));
  if (v.some((x, a) => x < 0 || x >= L.shape[a])) return undefined;
  const [key] = chunkKeys(st, li, v.map((x) => x >> 7), smoothFor(st)[0]);
  const arr = cache.peek(key);
  if (arr === undefined || arr === ERR) return undefined;
  if (arr === ZERO) return 0;
  return arr[(v[0] & 127) * MUL[0] + (v[1] & 127) * MUL[1] + (v[2] & 127)];
}

// ---------------------------------------------------------------- views + input
function setupViews() {
  S.views = [0, 1, 2].map((axis) => {
    const el = $("view" + axis);
    const canvas = el.querySelector("canvas");
    return { axis, el, canvas, ctx: canvas.getContext("2d", { alpha: false }), label: el.querySelector(".vlabel"), zoom: 1, center: [0, 0], buf: null, li: 0 };
  });
  const ro = new ResizeObserver(() => { resizeViews(); requestRender(); });
  for (const V of S.views) {
    ro.observe(V.el);
    attachViewInput(V);
  }
}
function resizeViews() {
  const dpr = window.devicePixelRatio || 1;
  for (const V of S.views) {
    const r = V.el.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (V.canvas.width !== w || V.canvas.height !== h) {
      V.canvas.width = w; V.canvas.height = h; // zoom is per device pixel; the centre stays put
    }
  }
}
function fitView(V) {
  const [ua, va] = UV[V.axis];
  const ws = worldShape();
  const W = V.canvas.width, H = V.canvas.height;
  V.zoom = Math.min(W / ws[ua], H / ws[va]) * 0.95;
  V.center = [ws[ua] / 2, ws[va] / 2];
}
function fitAll() { for (const V of S.views) fitView(V); requestRender(); }

function eventWorld(V, ev) {
  const r = V.canvas.getBoundingClientRect();
  const dpr = V.canvas.width / r.width;
  const px = (ev.clientX - r.left) * dpr, py = (ev.clientY - r.top) * dpr;
  const [ua, va] = UV[V.axis];
  const w = S.cursor.slice();
  w[ua] = V.center[0] + (px - V.canvas.width / 2) / V.zoom;
  w[va] = V.center[1] + (py - V.canvas.height / 2) / V.zoom;
  return { w, px, py, dpr };
}
function setCursorWorld(w) {
  const ws = worldShape();
  S.cursor = w.map((x, a) => clamp(x, 0, ws[a] - 1e-3));
  saveHash();
  requestRender();
}
function stepSlice(V, n) {
  const li = V.li || 0;
  const f = levelFactor(S.ct, li)[V.axis];
  const w = S.cursor.slice();
  // move to the centre of the next voxel of the view's level
  w[V.axis] = (Math.floor(w[V.axis] / f) + n + 0.5) * f;
  setCursorWorld(w);
}
function zoomView(V, factor, px, py) {
  const [ua, va] = UV[V.axis];
  if (px === undefined) { px = V.canvas.width / 2; py = V.canvas.height / 2; }
  const wx = V.center[0] + (px - V.canvas.width / 2) / V.zoom;
  const wy = V.center[1] + (py - V.canvas.height / 2) / V.zoom;
  V.zoom = clamp(V.zoom * factor, 1e-3, 64);
  V.center[0] = wx - (px - V.canvas.width / 2) / V.zoom;
  V.center[1] = wy - (py - V.canvas.height / 2) / V.zoom;
  requestRender();
}
function attachViewInput(V) {
  const cv = V.canvas;
  let drag = null;
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("pointerdown", (ev) => {
    setActive(V.axis);
    cv.setPointerCapture(ev.pointerId);
    drag = { x: ev.clientX, y: ev.clientY, b: ev.button, moved: false, c: V.center.slice(), win: S.win, lev: S.lev, shift: ev.shiftKey };
  });
  cv.addEventListener("pointermove", (ev) => {
    if (!S.ct) return;
    const e = eventWorld(V, ev);
    S.hover = { w: e.w, li: V.li || 0 };
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.b === 2 || (drag.b === 0 && drag.shift)) { // window / level
        S.win = clamp(Math.round(drag.win + dx), 1, 255);
        S.lev = clamp(Math.round(drag.lev - dy), 0, 255);
        syncWinUi();
        rebuildLuts();
      } else if (drag.moved) { // pan
        V.center[0] = drag.c[0] - (dx * e.dpr) / V.zoom;
        V.center[1] = drag.c[1] - (dy * e.dpr) / V.zoom;
        requestRender();
      }
    }
    updateStatus();
  });
  cv.addEventListener("pointerup", (ev) => {
    if (drag && !drag.moved && drag.b === 0 && !drag.shift && S.ct) setCursorWorld(eventWorld(V, ev).w);
    drag = null;
  });
  cv.addEventListener("pointerleave", () => { S.hover = null; });
  cv.addEventListener("dblclick", () => toggleMax(V.axis));
  cv.addEventListener("wheel", (ev) => {
    if (!S.ct) return;
    ev.preventDefault();
    setActive(V.axis);
    if (ev.ctrlKey || ev.metaKey || ev.altKey) {
      const e = eventWorld(V, ev);
      zoomView(V, Math.exp(-ev.deltaY * (ev.deltaMode ? 0.05 : 0.002)), e.px, e.py);
    } else {
      const n = Math.sign(ev.deltaY) * (ev.shiftKey ? 10 : 1);
      if (n) stepSlice(V, n);
    }
  }, { passive: false });
}
function setActive(axis) {
  if (S.active === axis) return;
  S.active = axis;
  for (const V of S.views) V.el.classList.toggle("active", V.axis === axis);
  requestRender();
}
function toggleMax(axis) {
  S.maximized = S.maximized === axis ? -1 : axis;
  $("views").classList.toggle("max", S.maximized >= 0);
  for (const V of S.views) V.el.classList.toggle("maxed", V.axis === S.maximized);
  setActive(axis);
  requestAnimationFrame(() => { resizeViews(); requestRender(); });
}

function setupMinimap() {
  const cv = $("minimap");
  let down = false;
  const go = (ev) => {
    if (!S.mini) return;
    const r = cv.getBoundingClientRect();
    const dpr = cv.width / r.width;
    const V = S.views[S.active];
    V.center = [((ev.clientX - r.left) * dpr - S.mini.ox) / S.mini.z, ((ev.clientY - r.top) * dpr - S.mini.oy) / S.mini.z];
    requestRender();
  };
  cv.addEventListener("pointerdown", (ev) => { down = true; cv.setPointerCapture(ev.pointerId); go(ev); });
  cv.addEventListener("pointermove", (ev) => { if (down) go(ev); });
  cv.addEventListener("pointerup", () => { down = false; });
}

function onKey(ev) {
  const t = ev.target;
  if (t && (t.tagName === "INPUT" && t.type !== "range" && t.type !== "checkbox" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
  if (ev.ctrlKey || ev.metaKey) return;
  const V = S.views[S.active];
  const k = ev.key;
  let used = true;
  if (!S.ct && !"?h".includes(k)) return;
  switch (k) {
    case "ArrowUp": case "w": case "W": case "PageUp": stepSlice(V, ev.shiftKey || k === "PageUp" ? 10 : 1); break;
    case "ArrowDown": case "s": case "S": case "PageDown": stepSlice(V, ev.shiftKey || k === "PageDown" ? -10 : -1); break;
    case "ArrowLeft": V.center[0] -= 64 / V.zoom; requestRender(); break;
    case "ArrowRight": V.center[0] += 64 / V.zoom; requestRender(); break;
    case "+": case "=": zoomView(V, 1.25); break;
    case "-": case "_": zoomView(V, 0.8); break;
    case "0": case "r": case "R": fitAll(); break;
    case "1": case "2": case "3": setActive(+k - 1); break;
    case "f": case "F": toggleMax(S.active); break;
    case "Escape": if (S.maximized >= 0) toggleMax(S.maximized); else $("help").classList.remove("open"); break;
    case "d": case "D": $("smooth").click(); break;
    case "o": case "O": $("ov-on").click(); break;
    case "t": case "T": $("ov-thresh").click(); break;
    case "[": setOpacity(S.ov.opacity - 0.1); break;
    case "]": setOpacity(S.ov.opacity + 0.1); break;
    case ",": case "<": levelStep(1); break;
    case ".": case ">": levelStep(-1); break;
    case "a": case "A": levelSet("auto"); break;
    case "c": case "C": S.crosshair = !S.crosshair; requestRender(); break;
    case "l": case "L": autoWindow(); break;
    case "?": case "h": case "H": $("help").classList.toggle("open"); break;
    default: used = false;
  }
  if (used) ev.preventDefault();
}
function setOpacity(v) {
  S.ov.opacity = clamp(v, 0, 1);
  $("ov-opacity").value = Math.round(S.ov.opacity * 100);
  rebuildLuts();
}
function levelStep(d) { // + coarser
  const cur = S.level === "auto" ? viewLevel(S.views[S.active]) : +S.level;
  levelSet(String(clamp(cur + d, 0, S.ct.levels.length - 1)));
}
function levelSet(v) { S.level = v; $("level").value = v; saveHash(); requestRender(); }
function autoWindow() {
  const h = S.hist;
  if (!h) return;
  let n = 0;
  for (let v = 1; v < 256; v++) n += h[v]; // ignore masked air (0)
  if (!n) return;
  let acc = 0, lo = 1, hi = 255;
  for (let v = 1; v < 256; v++) { acc += h[v]; if (acc >= n * 0.005) { lo = v; break; } }
  acc = 0;
  for (let v = 255; v >= 1; v--) { acc += h[v]; if (acc >= n * 0.005) { hi = v; break; } }
  S.win = Math.max(1, hi - lo);
  S.lev = Math.round((hi + lo) / 2);
  syncWinUi();
  rebuildLuts();
}
function syncWinUi() {
  $("win").value = S.win; $("lev").value = S.lev;
  $("win-val").textContent = S.win; $("lev-val").textContent = S.lev;
}

// ---------------------------------------------------------------- pickers
function setOptions(sel, items, value) {
  sel.innerHTML = "";
  for (const it of items) {
    const o = document.createElement("option");
    if (typeof it === "string") { o.value = it; o.textContent = it; } else { o.value = it.value; o.textContent = it.label; }
    sel.appendChild(o);
  }
  if (value !== undefined) sel.value = value;
}
function msg(t, isErr) {
  const e = $("status-warn");
  e.textContent = t;
  e.classList.toggle("err", !!isErr);
  if (isErr) stats.lastError = t;
}

async function loadSamples(wantSample) {
  let names;
  try {
    names = (await listDir(S.base)).filter((n) => !/\.zarr$/.test(n));
    if (!names.length) throw new Error("empty index");
  } catch (e) {
    names = FALLBACK_SAMPLES.slice();
    msg("mirror index not readable (" + e.message + "); using the built-in sample list — paste a URL if a sample is missing", true);
  }
  setOptions($("sample"), [{ value: "", label: "— sample —" }, ...names]);
  const pick = wantSample && names.includes(wantSample) ? wantSample : "";
  $("sample").value = pick;
  if (pick) await selectSample(pick, true);
}
async function selectSample(name, keepSelection) {
  S.sample = name;
  setOptions($("volume"), [{ value: "", label: "loading…" }]);
  setOptions($("pred"), [{ value: "", label: "loading…" }]);
  if (!name) return;
  const sroot = joinUrl(S.base, name);
  const [vols, preds] = await Promise.all([
    listDir(joinUrl(sroot, "volumes")).catch(() => []),
    listDir(joinUrl(sroot, "representations/predictions/surfaces")).catch(() => []),
  ]);
  const vz = vols.filter((n) => /\.zarr$/.test(n)), pz = preds.filter((n) => /\.zarr$/.test(n));
  setOptions($("volume"), [{ value: "", label: vz.length ? "— volume —" : "(none found; paste a URL)" },
    ...vz.map((n) => ({ value: joinUrl(joinUrl(sroot, "volumes"), n), label: n }))]);
  setOptions($("pred"), [{ value: "", label: pz.length ? "none" : "(no predictions)" },
    ...pz.map((n) => ({ value: joinUrl(joinUrl(sroot, "representations/predictions/surfaces"), n), label: n }))]);
  if (keepSelection) return;
  if (vz.length) {
    $("volume").value = joinUrl(joinUrl(sroot, "volumes"), vz[0]);
    await openVolume($("volume").value);
  }
}

async function openVolume(url, keepView) {
  if (!url) return;
  msg("opening " + url + " …");
  try {
    const st = await new Store(url).open();
    S.ct = st;
    if (S.pred && !samePrefix(S.pred.url, st.url)) { S.pred = null; $("pred").value = ""; }
    setOptions($("level"), [{ value: "auto", label: "auto" },
      ...st.levels.map((L, i) => ({ value: String(i), label: `${L.path}  ${L.shape.join("×")}${L.q != null ? "  q" + L.q : ""}` }))]);
    if (S.level !== "auto" && +S.level >= st.levels.length) S.level = "auto";
    $("level").value = S.level;
    if (!keepView) {
      const ws = worldShape();
      S.cursor = ws.map((n) => n / 2);
      fitAll();
    }
    rebuildPredLevels();
    const pitch = st.pitch0 ? st.pitch0[0] + " µm" : "unknown pitch";
    $("info-vol").textContent = `${st.name}\n${st.levels[0].shape.join(" × ")} · ${pitch} · ${st.levels.length} levels`;
    msg("");
    saveHash();
    requestRender();
  } catch (e) {
    msg("could not open volume: " + e.message, true);
  }
}
function samePrefix(a, b) { // same sample root
  const s = (u) => u.split(/\/(volumes|representations)\//)[0];
  return s(a) === s(b);
}
async function openPred(url) {
  if (!url) { S.pred = null; rebuildPredLevels(); saveHash(); requestRender(); return; }
  msg("opening overlay " + url + " …");
  try {
    S.pred = await new Store(url).open();
    if (!S.ct) { await openVolume(url); S.pred = null; return; }
    rebuildPredLevels();
    const lv = S.pred.levels[0];
    const vc = S.pred.attrs.volcomp || {};
    $("info-pred").textContent = `${S.pred.name}\n${lv.shape.join(" × ")} · level 0 ${lv.path}` +
      (vc.content ? "\n" + vc.content : "");
    const wp = worldPitch();
    if (!S.ct.pitch0 && !S.pred.unitUm) msg("no voxel pitch on either store: overlay level 0 assumed on the CT level-0 grid", true);
    else if (!S.ct.pitch0 && wp) msg("CT pitch unknown: overlay level 0 assumed on the CT level-0 grid");
    else msg("");
    saveHash();
    requestRender();
  } catch (e) {
    S.pred = null;
    msg("could not open overlay: " + e.message, true);
  }
}
function rebuildPredLevels() {
  const items = [{ value: "auto", label: "auto" }];
  if (S.pred) S.pred.levels.forEach((L, i) => items.push({ value: String(i), label: `${L.path}  ${L.shape.join("×")}` }));
  setOptions($("pred-level"), items, S.predLevel);
  if ($("pred-level").value !== S.predLevel) { S.predLevel = "auto"; $("pred-level").value = "auto"; }
}

// Paste box: a mirror root, a sample directory, a volume (.zarr or one of its
// levels) or a prediction store.
async function openPasted(raw) {
  let url = raw.trim();
  if (!url) return;
  if (!/^https?:\/\//i.test(url)) { msg("need an http(s) URL", true); return; }
  const m = /^(.*?\.zarr)(\/.*)?$/i.exec(url);
  if (m) {
    const z = m[1];
    if (/\/representations\/|-prob\.zarr$|-th[\d.]+\.zarr$/i.test(z) && S.ct) {
      if (![...$("pred").options].some((o) => o.value === z)) $("pred").add(new Option(z.split("/").pop(), z));
      $("pred").value = z;
      await openPred(z);
    } else {
      if (![...$("volume").options].some((o) => o.value === z)) $("volume").add(new Option(z.split("/").pop(), z));
      $("volume").value = z;
      await openVolume(z);
    }
    return;
  }
  // a directory: sample (has volumes/) or mirror root
  url = url.replace(/\/+$/, "");
  const parts = url.split("/");
  const last = parts.pop();
  const subs = await listDir(url).catch(() => null);
  if (subs && subs.includes("volumes")) {
    S.base = parts.join("/") + "/";
    $("base-url").value = S.base;
    await loadSamples(last);
    if (!$("sample").value) { $("sample").add(new Option(last, last)); $("sample").value = last; }
    await selectSample(last);
  } else {
    S.base = url + "/";
    $("base-url").value = S.base;
    await loadSamples();
  }
}

// ---------------------------------------------------------------- URL hash state
let hashTimer = 0;
function saveHash() {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const p = new URLSearchParams();
    if (S.base !== DEFAULT_BASE) p.set("base", S.base);
    if (S.ct) p.set("vol", S.ct.url);
    if (S.pred) p.set("pred", S.pred.url);
    if (S.ct) p.set("c", S.cursor.map((x) => Math.floor(x)).join(","));
    if (S.level !== "auto") p.set("level", S.level);
    history.replaceState(null, "", "#" + p.toString());
  }, 300);
}
async function loadHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("base")) S.base = p.get("base");
  $("base-url").value = S.base;
  const vol = p.get("vol");
  let sample = "";
  if (vol) {
    const m = /^(.*)\/([^/]+)\/volumes\/[^/]+$/.exec(vol.replace(/\/+$/, ""));
    if (m) sample = m[2];
  }
  if (p.get("level")) S.level = p.get("level");
  await loadSamples(sample);
  if (vol) {
    if (![...$("volume").options].some((o) => o.value === vol)) $("volume").add(new Option(vol.split("/").pop(), vol));
    $("volume").value = vol;
    await openVolume(vol);
    if (p.get("c")) {
      const c = p.get("c").split(",").map(Number);
      if (c.length === 3 && c.every(Number.isFinite)) setCursorWorld(c.map((x) => x + 0.5));
    }
    const pred = p.get("pred");
    if (pred) {
      if (![...$("pred").options].some((o) => o.value === pred)) $("pred").add(new Option(pred.split("/").pop(), pred));
      $("pred").value = pred;
      await openPred(pred);
    }
  }
}

// ---------------------------------------------------------------- controls
function setupControls() {
  $("base-url").value = S.base;
  $("base-go").onclick = () => { S.base = $("base-url").value.trim().replace(/\/?$/, "/"); loadSamples(); };
  $("sample").onchange = (e) => selectSample(e.target.value);
  $("volume").onchange = (e) => openVolume(e.target.value);
  $("level").onchange = (e) => levelSet(e.target.value);
  $("pred").onchange = (e) => openPred(e.target.value);
  $("pred-level").onchange = (e) => { S.predLevel = e.target.value; requestRender(); };
  $("paste-go").onclick = () => openPasted($("paste-url").value);
  $("paste-url").addEventListener("keydown", (e) => { if (e.key === "Enter") openPasted(e.target.value); });
  $("win").oninput = (e) => { S.win = +e.target.value; syncWinUi(); rebuildLuts(); };
  $("lev").oninput = (e) => { S.lev = +e.target.value; syncWinUi(); rebuildLuts(); };
  $("wl-auto").onclick = autoWindow;
  $("wl-reset").onclick = () => { S.win = 255; S.lev = 128; syncWinUi(); rebuildLuts(); };
  $("ov-on").onchange = (e) => { S.ov.on = e.target.checked; requestRender(); };
  $("ov-thresh").onchange = (e) => { S.ov.thresh = e.target.checked; rebuildLuts(); };
  $("ov-opacity").oninput = (e) => { S.ov.opacity = +e.target.value / 100; rebuildLuts(); };
  $("smooth").onchange = (e) => { S.smooth.on = e.target.checked; requestRender(); };
  $("smooth-pred").onchange = (e) => { S.smooth.pred = e.target.checked; requestRender(); };
  $("smooth-ct-s").onchange = (e) => { S.smooth.ct = clamp(+e.target.value || 2, 0.1, 4); requestRender(); };
  $("smooth-pr-s").onchange = (e) => { S.smooth.pr = clamp(+e.target.value || 0.6, 0.1, 4); requestRender(); };
  $("cache-cap").onchange = (e) => { cache.cap = +e.target.value * 1048576; cache.trim(); requestRender(); };
  $("clear-cache").onclick = () => { cache.clear(); ccache.clear(); if (loader) loader.reset(); stats.lastError = ""; requestRender(); };
  $("fit").onclick = fitAll;
  $("help-btn").onclick = () => $("help").classList.toggle("open");
  $("help-close").onclick = () => $("help").classList.remove("open");
  window.addEventListener("keydown", onKey);
  syncWinUi();
}

// ---------------------------------------------------------------- boot
function b64bytes(s) {
  const bin = atob(s.replace(/\s+/g, ""));
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

async function boot() {
  setupControls();
  setupViews();
  setupMinimap();
  resizeViews();
  rebuildLuts();
  setInterval(updateStatus, 500);
  const wasm = b64bytes($("wasm-b64").textContent);
  const nWorkers = clamp(navigator.hardwareConcurrency || 4, 1, 16);
  pool = new Pool(wasm.buffer, $("worker-src").textContent, nWorkers);
  loader = new Loader(pool);
  try {
    const info = await pool.ready;
    $("info-wasm").textContent = `volcomp ${info.version} (wasm, ${info.kernels} kernels) · ${nWorkers} workers`;
  } catch (e) {
    msg("decoder failed to start: " + e.message, true);
    return;
  }
  await loadHash();
}

// Test/debug hook: decode one inner chunk (global chunk coords) of a level URL
// and return its checksums. Used by web/test/browser_check.mjs.
window.vcViewer = {
  S, stats, cache,
  async chunkChecksum(levelUrl, cz, cy, cx, smooth = 0, flags = 0) {
    const meta = await fetchJSON(joinUrl(levelUrl, "zarr.json"));
    const L = parseLevel(levelUrl, "", [1, 1, 1], meta);
    const s = [cz, cy, cx].map((c, a) => Math.floor(c / L.cps[a]));
    const l = [cz, cy, cx].map((c, a) => c - s[a] * L.cps[a]);
    const k = (l[0] * L.cps[1] + l[1]) * L.cps[2] + l[2];
    const url = joinUrl(levelUrl, L.keyPrefix + s.join(L.sep));
    const n = L.nInner * 16 + 4;
    const idx = await fetchRange(url, "bytes=-" + n);
    if (!idx) return { missing: "shard" };
    const dv = new DataView(idx);
    const off = dv.getUint32(k * 16 + 4, true) * 4294967296 + dv.getUint32(k * 16, true);
    const nb = dv.getUint32(k * 16 + 12, true) * 4294967296 + dv.getUint32(k * 16 + 8, true);
    if (dv.getUint32(k * 16, true) === 0xffffffff && dv.getUint32(k * 16 + 4, true) === 0xffffffff) return { missing: "chunk" };
    const enc = await fetchRange(url, `bytes=${off}-${off + nb - 1}`);
    const encCrc = crc32c(new Uint8Array(enc));
    const r = await pool.decode(enc, smooth, flags);
    const sha = crypto.subtle ? [...new Uint8Array(await crypto.subtle.digest("SHA-256", r.out))]
      .map((b) => b.toString(16).padStart(2, "0")).join("") : null;
    return { offset: off, nbytes: nb, encCrc32c: encCrc, crc32c: crc32c(r.out), sha256: sha, ms: r.ms, mode: r.mode, q: r.q };
  },
};

boot();
