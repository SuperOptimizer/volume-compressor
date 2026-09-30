// volcomp viewer: streams volcomp-coded zarr v3 stores straight from an HTTP
// mirror (Range requests into sharded stores) and shows three orthogonal slices
// of a stack of layers (CT, probability and mask stores, region sets), each
// placed in one world frame in micrometres and resampled into the view.
// See web/README.md.
//
// Parts, top to bottom:
//   http      fetchRange / listIndex, byte and request counters
//   LRU       decoded chunk cache with per-owner (per-store) caps under a total cap
//   Store     zarr v3 group/array metadata, OME multiscales, level geometry (pitch, origin)
//   RegionSet a directory of region_Z_Y_X.zarr stores seen as one sparse array
//   Loader    wanted chunks -> shard index reads -> coalesced Range reads -> decode pool
//   Pool      Web Workers, one volcomp wasm instance each (worker.js)
//   layers    layer stack, LUTs, blend modes
//   render    per-layer level choice, plane extraction, trilinear/nearest resampling, compositing
//   ui        pickers, add-layer panel, layer panel, mouse, keyboard, status bar, URL hash state
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
const VIEW_SHARE = 0.7;          // the views of one layer may use this much of its cache share (the rest: prefetch)
const TOO_FINE = 16;            // a layer whose coarsest level is > this many times finer than a screen pixel is not loaded
const PLANE_MAX = 16;           // ... nor one whose level voxels in view exceed this many per screen pixel (min 4M)
const REGION_SET_MIN = 20;      // a directory with more region_*.zarr stores than this is listed as one region set
const RENDER_MAX_PX = 1.5e6;     // a view larger than this (device pixels) is composited at reduced resolution
const W8 = 256, W24 = 16777216, HALF24 = 8388608; // fixed-point resampling weights (8 bits per axis)
const A_ONE = 65025;            // alpha x opacity scale (255 * 255)

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const joinUrl = (a, b) => a.replace(/\/+$/, "") + "/" + b.replace(/^\/+/, "");
const lastSeg = (u) => decodeURIComponent(u.replace(/\/+$/, "").split("/").pop());
function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1 << 20) return (n / 1024).toFixed(1) + " KB";
  if (n < 1 << 30) return (n / (1 << 20)).toFixed(1) + " MB";
  return (n / (1 << 30)).toFixed(2) + " GB";
}
const fmtUm = (x) => (Math.abs(x) >= 100 ? x.toFixed(1) : Math.abs(x) >= 10 ? x.toFixed(2) : x.toFixed(3)).replace(/\.?0+$/, "") + " µm";
// "...-2.215um-..." -> 2.215 (the CT naming convention)
function pitchFromName(name) { const m = /(\d+(?:\.\d+)?)\s*um(?![a-z])/i.exec(name || ""); return m ? +m[1] : null; }
function scanOf(name) { const m = /^(\d{14})/.exec(name || ""); return m ? m[1] : ""; }
function levelTagOf(name) { const m = /-L(\d+)-/.exec(name || ""); return m ? +m[1] : null; }
function sampleRootOf(url) { const m = /^(.*?)\/(volumes|representations)\//.exec(url + "/"); return m ? m[1] : ""; }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
// Entries of an HTML autoindex page (nginx/apache style): {dirs, files}, names without the slash.
const indexCache = new Map();
function listIndex(url) {
  const u = url.replace(/\/?$/, "/");
  if (!indexCache.has(u)) {
    const p = (async () => {
      const b = await fetchRange(u);
      if (!b) throw new Error("not found: " + u);
      const html = new TextDecoder().decode(b);
      const dirs = [], files = [], seen = new Set();
      for (const m of html.matchAll(/href="([^"?#]+)"/g)) {
        let h = decodeURIComponent(m[1]);
        if (h.startsWith("/") || h.startsWith("..") || /^[a-z]+:/i.test(h) || seen.has(h)) continue;
        seen.add(h);
        if (h.endsWith("/")) { h = h.slice(0, -1); if (h) dirs.push(h); } else files.push(h);
      }
      return { dirs, files };
    })();
    p.catch(() => indexCache.delete(u));
    indexCache.set(u, p);
  }
  return indexCache.get(u);
}
async function listDir(url) { return (await listIndex(url)).dirs; }

// ---------------------------------------------------------------- LRU
// One map for every layer; each entry has an owner (a store id). The total is
// capped, and so is each owner (its share of the total), so one layer can never
// evict the chunks another layer has on screen.
class LRU {
  constructor(cap) { this.cap = cap; this.map = new Map(); this.bytes = 0; this.own = new Map(); this.ownCap = new Map(); this.pinned = null; }
  get(k) {
    const e = this.map.get(k);
    if (e === undefined) return undefined;
    this.map.delete(k);
    this.map.set(k, e);
    return e.v;
  }
  peek(k) { const e = this.map.get(k); return e && e.v; }
  has(k) { return this.map.has(k); }
  set(k, v, size, owner = -1) {
    const old = this.map.get(k);
    if (old) this.del(k, old);
    this.map.set(k, { v, n: size, o: owner });
    this.bytes += size;
    this.own.set(owner, (this.own.get(owner) || 0) + size);
    this.trim();
  }
  del(k, e) {
    this.map.delete(k);
    this.bytes -= e.n;
    this.own.set(e.o, (this.own.get(e.o) || 0) - e.n);
  }
  // Evict least recently used entries until the total and every owner are
  // within their caps, never touching `pinned` (the chunks the current frame
  // draws): evicting those would make the next frame decode them again, and so on.
  trim() {
    const pin = this.pinned;
    if (this.bytes > this.cap) {
      for (const [k, e] of this.map) {
        if (this.bytes <= this.cap) break;
        if (pin && pin.has(k)) continue;
        this.del(k, e);
      }
    }
    for (const [o, cap] of this.ownCap) {
      if ((this.own.get(o) || 0) <= cap) continue;
      for (const [k, e] of this.map) {
        if (e.o !== o || pin && pin.has(k)) continue;
        if (this.own.get(o) <= cap) break;
        this.del(k, e);
      }
    }
  }
  clear() { this.map.clear(); this.bytes = 0; this.own.clear(); }
  get size() { return this.map.size; }
}
const cache = new LRU(3072 * 1048576); // decoded 128^3 chunks (the cache picker's default)
const ccache = new LRU(256 << 20);  // compressed chunk bytes (toggling deblock re-decodes without re-downloading)

// ---------------------------------------------------------------- Store
const storeIds = new Map(); // url -> small int (cache keys, cache owners)
function storeIdOf(url) { if (!storeIds.has(url)) storeIds.set(url, storeIds.size); return storeIds.get(url); }
const LEN_UM = { micrometer: 1, micron: 1, um: 1, "µm": 1, nanometer: 1e-3, nm: 1e-3, millimeter: 1e3, mm: 1e3,
  centimeter: 1e4, cm: 1e4, meter: 1e6, m: 1e6, angstrom: 1e-4 };

class Store {
  constructor(url) {
    this.url = url.replace(/\/+$/, "");
    this.name = lastSeg(this.url);
    this.id = storeIdOf(this.url);
    this.levels = [];
    this.attrs = {};
    this.vc = {};
    this.uf = null;       // micrometres per OME scale unit, when the axes carry a length unit
    this.hasTrans = false; // any OME translation
    this.notes = [];
  }
  async open() {
    const rb = await fetchRange(joinUrl(this.url, "zarr.json"));
    if (!rb) throw new Error(await notV3Reason(this.url));
    const root = JSON.parse(new TextDecoder().decode(rb));
    this.attrs = root.attributes || {};
    this.vc = this.attrs.volcomp || {};
    let datasets;
    if (root.node_type === "array") {
      datasets = [{ path: "", scale: [1, 1, 1], trans: [0, 0, 0] }];
    } else {
      const ms = (this.attrs.ome && this.attrs.ome.multiscales) || this.attrs.multiscales;
      if (!ms || !ms.length) throw new Error("no OME multiscales in " + this.url + "/zarr.json");
      const m = ms[0];
      this.omeType = m.type || "";
      const ax = (m.axes || []).slice(-3);
      const f = ax.map((a) => LEN_UM[String(a.unit || "").toLowerCase()]);
      if (ax.length === 3 && f.every((x) => x && x === f[0])) this.uf = f[0];
      const g = m.coordinateTransformations || [];
      const gs = ((g.find((t) => t.type === "scale") || {}).scale || [1, 1, 1]).slice(-3);
      const gt = ((g.find((t) => t.type === "translation") || {}).translation || null);
      if (gt) this.hasTrans = true;
      datasets = m.datasets.map((d) => {
        const ts = d.coordinateTransformations || [];
        const s = ((ts.find((t) => t.type === "scale") || {}).scale || [1, 1, 1]).slice(-3);
        const tr = (ts.find((t) => t.type === "translation") || {}).translation;
        if (tr) this.hasTrans = true;
        const t3 = tr ? tr.slice(-3) : [0, 0, 0], g3 = gt ? gt.slice(-3) : [0, 0, 0];
        return { path: d.path, scale: s.map((x, a) => x * gs[a]), trans: t3.map((x, a) => x * gs[a] + g3[a]) };
      });
    }
    const metas = await Promise.all(datasets.map((d) =>
      d.path === "" ? Promise.resolve(root) : fetchJSON(joinUrl(joinUrl(this.url, d.path), "zarr.json"))));
    this.levels = datasets.map((d, i) => Object.assign(parseLevel(joinUrl(this.url, d.path), d.path, d.scale, metas[i]), { trans: d.trans }));
    // an array store's own attributes (the region stores carry voxel_um / origin_zyx there)
    if (root.node_type === "array") this.arrayAttrs = this.attrs;
    return this;
  }
  // Level-0 pitch (µm, per axis) from this store's own metadata, and which rule gave it.
  // Contextual rules (the -L<k> name against an open CT layer, the fallback) live in relayout().
  intrinsicPitch() {
    const vc = this.vc, L0 = this.levels[0];
    // 1. prediction stores made from CT level k: pitch = CT pitch (from the source volume's name) x 2^k
    const sv = vc.source_volume, k = vc.source_level;
    if (sv && Number.isInteger(k)) {
      const p = pitchFromName(lastSeg(sv));
      if (p) return { p0: [p, p, p].map((x) => x * 2 ** k), rule: `source CT ${p} µm × 2^${k} (volcomp source_volume, source_level)` };
    }
    // 2. OME scales with a length unit
    if (this.uf) return { p0: L0.scale.map((s) => s * this.uf), rule: "OME scale (" + (this.uf === 1 ? "µm" : "× " + this.uf + " µm") + ")" };
    // 3. volcomp attributes
    const lv = vc.levels && vc.levels[0];
    if (lv && lv.voxel_size_um) return { p0: [1, 1, 1].map(() => +lv.voxel_size_um), rule: "volcomp levels[0].voxel_size_um" };
    const vu = this.attrs.voxel_um || vc.voxel_um;
    if (vu) return { p0: [1, 1, 1].map(() => +vu), rule: "attributes voxel_um" };
    // 4. the store name ("...-8.640um-...")
    const pn = pitchFromName(this.name);
    if (pn) return { p0: [pn, pn, pn], rule: "store name (" + pn + "um)" };
    return null;
  }
}

// Why a URL without zarr.json cannot be opened: a zarr v2 store (.zgroup/.zarray)
// is named as such, with the mirror's volcomp copy when the path suggests one.
async function notV3Reason(url) {
  for (const f of ["0/.zarray", ".zarray", ".zgroup"]) {
    const b = await fetchRange(joinUrl(url, f)).catch(() => null);
    if (!b) continue;
    let what = "zarr v2";
    try {
      const m = JSON.parse(new TextDecoder().decode(b));
      const za = f.endsWith(".zarray") ? m : null;
      if (za) what += ` (${za.dtype}, ${za.compressor ? za.compressor.id + (za.compressor.cname ? "/" + za.compressor.cname : "") : "raw"}, chunks ${za.chunks.join("×")})`;
    } catch { /* keep the generic name */ }
    const up = /amazonaws\.com\/(.+?)\/?$/.exec(url);
    const hint = up ? " — the volcomp copy is " + joinUrl(S.base, up[1]) : " — use its volcomp copy on the mirror";
    return `${lastSeg(url)} is a ${what} store; the viewer reads volcomp zarr v3 stores only${hint}`;
  }
  return "not a zarr v3 store (no zarr.json): " + url;
}

// A directory of region_Z_Y_X.zarr stores (one shard each, origins in level-0
// voxels on a grid of the region edge) seen as one sparse single-level array.
class RegionSet {
  constructor(url) {
    this.url = url.replace(/\/+$/, "");
    this.name = lastSeg(this.url) + " (regions)";
    this.id = storeIdOf(this.url);
    this.levels = [];
    this.attrs = {};
    this.vc = {};
    this.uf = null;
    this.hasTrans = false;
    this.notes = [];
    this.isRegionSet = true;
  }
  async open() {
    const { dirs } = await listIndex(this.url);
    const regs = [];
    for (const d of dirs) {
      const m = /^region_(\d+)_(\d+)_(\d+)\.zarr$/.exec(d);
      if (m) regs.push([+m[1], +m[2], +m[3]]);
    }
    if (!regs.length) throw new Error("no region_Z_Y_X.zarr stores in " + this.url);
    const first = regs[0].join("_");
    const meta = await fetchJSON(joinUrl(this.url, "region_" + first + ".zarr/zarr.json"));
    const R = parseLevel(joinUrl(this.url, "region_" + first + ".zarr"), "", [1, 1, 1], meta);
    const E = R.shape;
    if (E.some((e) => e % C)) throw new Error("region edge " + E + " is not a multiple of " + C);
    if (regs.some((r) => r.some((x, a) => x % E[a]))) throw new Error("region origins are not on a " + E.join("x") + " grid");
    this.attrs = meta.attributes || {};
    this.nRegions = regs.length;
    const present = new Set(regs.map((r) => r.join("_")));
    const shape = [0, 1, 2].map((a) => Math.max(...regs.map((r) => r[a])) + E[a]);
    const url = this.url, cpr = E.map((e) => e / C); // chunks per region
    // Each region store has its own layout (sharded or not): the loader reads a
    // region's zarr.json the first time it needs one of its chunks (region()).
    const regions = new Map();
    this.levels = [{
      url: this.url, path: "", scale: [1, 1, 1], trans: [0, 0, 0], shape, grid: shape.map((n) => Math.ceil(n / C)),
      cps: cpr, nInner: cpr[0] * cpr[1] * cpr[2], sharded: true, zstd: R.zstd, q: R.q, isRegions: true,
      // chunk c of the set -> {key, url} of its region (null: no such region, air) and the chunk inside it
      regionOf: (c) => {
        const r = c.map((x, a) => Math.floor(x / cpr[a]));
        const key = r.map((x, a) => x * E[a]).join("_");
        if (!present.has(key)) return null;
        return { key, url: url + "/region_" + key + ".zarr", c: c.map((x, a) => x - r[a] * cpr[a]) };
      },
      region: (ro) => {
        let e = regions.get(ro.key);
        if (!e) {
          e = { state: "pending" };
          regions.set(ro.key, e);
          e.p = fetchJSON(joinUrl(ro.url, "zarr.json")).then((m) => {
            const L = parseLevel(ro.url, "", [1, 1, 1], m);
            if (L.shape.some((n, a) => n !== E[a])) throw new Error("region " + ro.key + " has shape " + L.shape);
            e.L = L;
            e.state = "ok";
          }).catch((err) => { e.state = "error"; e.error = err; });
        }
        return e;
      },
    }];
    this.regionEdge = E;
    return this;
  }
  intrinsicPitch() {
    const vu = this.attrs.voxel_um;
    if (vu) return { p0: [vu, vu, vu], rule: "region attributes voxel_um" };
    const pn = pitchFromName(lastSeg(this.url));
    return pn ? { p0: [pn, pn, pn], rule: "directory name (" + pn + "um)" } : null;
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
    sharded: false, shard: grid, inner: grid, cps: [1, 1, 1], nInner: 1, indexAtEnd: true, q: null, zstd: false,
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
  // accepted inner codec chains: [volcomp] and [volcomp, zstd] ("bytes" entries are no-ops)
  const vc = codecs.filter((c) => c.name !== "bytes");
  const names = vc.map((c) => c.name);
  const ok = names.length >= 1 && names[0] === "volcomp" && (names.length === 1 || names.length === 2 && names[1] === "zstd");
  if (!ok) throw new Error(url + ": inner codecs " + JSON.stringify(codecs.map((c) => c.name)) + " (need volcomp, optionally then zstd)");
  if (L.inner.some((n) => n !== C)) throw new Error(url + ": inner chunk " + L.inner + " (need 128^3)");
  L.zstd = names.length === 2;
  L.q = vc[0].configuration && vc[0].configuration.q;
  L.mode = vc[0].configuration && vc[0].configuration.mode;
  L.grid = L.shape.map((n) => Math.ceil(n / C)); // inner chunks per axis
  return L;
}

// Open a store URL: a .zarr (group or array), or a directory of region stores.
const storePromises = new Map();
function openStore(url) {
  url = url.replace(/\/+$/, "");
  if (!storePromises.has(url)) {
    const p = (async () => {
      if (/\.zarr$/i.test(url)) return new Store(url).open();
      // not a .zarr: a region-set directory, or a store without the suffix
      const idx = await listIndex(url).catch(() => null);
      if (idx && idx.dirs.some((d) => /^region_\d+_\d+_\d+\.zarr$/.test(d))) return new RegionSet(url).open();
      return new Store(url).open();
    })();
    p.catch(() => storePromises.delete(url));
    storePromises.set(url, p);
  }
  return storePromises.get(url);
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
  decode(enc, smooth, flags, zstd) {
    return new Promise((resolve, reject) => {
      this.jobs.push({ enc, smooth, flags, zstd, resolve, reject });
      this.pump();
    });
  }
  pump() {
    while (this.free.length && this.jobs.length) {
      const w = this.free.pop(), j = this.jobs.shift(), id = this.nextId++;
      this.pending.set(id, j);
      w.postMessage({ type: "decode", id, enc: j.enc, smooth: j.smooth, flags: j.flags, zstd: j.zstd }, [j.enc]);
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
  // the array level and chunk an item reads: the store's level, or for a region
  // set the region store's own level (once its zarr.json is in, it.sub)
  levelOf(it) { return it.sub || { L: it.st.levels[it.li], c: it.c }; }
  shardOf(it) {
    const { L, c } = this.levelOf(it);
    const s = [0, 1, 2].map((a) => Math.floor(c[a] / L.cps[a]));
    const url = joinUrl(L.url, L.keyPrefix + s.join(L.sep));
    const l = [0, 1, 2].map((a) => c[a] - s[a] * L.cps[a]);
    return { url, L, inner: (l[0] * L.cps[1] + l[1]) * L.cps[2] + l[2] };
  }
  finish(it, arr, size) {
    this.inflight.delete(it.key);
    cache.set(it.key, arr, size, it.st.id);
    requestRender();
  }
  fail(it, err) {
    stats.lastError = String(err && err.message || err);
    this.finish(it, ERR, 64);
  }
  decode(it, enc) {
    this.pool.decode(enc.slice(0), it.smooth, it.flags, this.levelOf(it).L.zstd).then((r) => {
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
      const L0 = it.st.levels[it.li];
      if (L0.isRegions && !it.sub) {
        const ro = L0.regionOf(it.c);
        if (!ro) { // no such region: air, no request
          q.splice(i, 1);
          stats.zeroChunks++;
          cache.set(it.key, ZERO, 64, it.st.id);
          requestRender();
          continue;
        }
        const e = L0.region(ro);
        if (e.state === "pending") {
          if (!e.hooked) { e.hooked = true; this.nFetch++; e.p.finally(() => { this.nFetch--; this.pump(); }); }
          i++;
          continue;
        }
        if (e.state === "error") { q.splice(i, 1); this.fail(it, e.error); continue; }
        it.sub = { L: e.L, c: ro.c };
      }
      const sh = this.shardOf(it);
      const L = sh.L;
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
        if (s.state === "missing") { stats.zeroChunks++; cache.set(it.key, ZERO, 64, it.st.id); requestRender(); }
        else this.fail(it, s.error);
        continue;
      }
      // every queued chunk of this shard (same smoothing) in one pass, coalesced
      const group = [];
      for (let j = i; j < q.length; j++) {
        const o = q[j];
        if (o.st !== it.st || o.li !== it.li || o.smooth !== it.smooth) continue;
        if (o.st.levels[o.li].isRegions && !o.sub) continue;
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
        cache.set(it.key, ZERO, 64, it.st.id);
        continue;
      }
      const off = hi * 4294967296 + lo, n = nhi * 4294967296 + nlo;
      if (n === 0) { cache.set(it.key, ZERO, 64, it.st.id); continue; }
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
  layers: [],      // bottom -> top (drawn first -> last)
  sel: -1,         // selected layer (settings panel, keys, right-drag)
  cursor: [0, 0, 0], // world position in micrometres [z, y, x]
  views: [],
  active: 0,
  maximized: -1,
  smooth: { on: false, pred: true, ct: 2.0, pr: 0.6 },
  crosshair: true,
  outlines: true,
  hover: null,
  warn: "",
  hist: null,       // histogram of the selected layer in the active view (auto window/level)
  extraWant: null,  // chunks wanted by a test tile (renderTile)
};
let pool = null, loader = null;

// ---------------------------------------------------------------- geometry
// Every level of every layer gets a pitch (µm per voxel, per axis) and an origin
// (µm of the corner of voxel 0). Voxel i of a level covers [origin + i*pitch,
// origin + (i+1)*pitch) and its centre is origin + (i+0.5)*pitch. Pitch rules, in
// order (see README "Geometry"):
//   1. volcomp source_volume + source_level k (the m7 -L<k> stores): CT pitch (from the
//      source volume's name) x 2^k, i.e. level 0 of the prediction is CT level k;
//   2. OME scales with a length unit;  3. volcomp levels[0].voxel_size_um / voxel_um;
//   4. the store name ("...-2.215um-...");
//   5. a "-L<k>-" name next to a CT layer of the same scan: that CT's level-k pitch;
//   6. otherwise the level-0 grid of the lowest layer that has a pitch.
// Other levels: pitch0 x scale_i / scale_0. Origins: 0 (pooled pyramids share the
// corner of voxel 0); with OME translations, t is the centre of voxel 0 so the
// corner is t - pitch/2; array stores with origin_zyx (regions) start there.
function relayout() {
  for (const ly of S.layers) ly.geo = ly.st.intrinsicPitch();
  for (const ly of S.layers) {
    const st = ly.st, L0 = st.levels[0];
    let p0 = ly.geo && ly.geo.p0, rule = ly.geo && ly.geo.rule;
    ly.pitchNote = "";
    if (!p0) {
      const k = levelTagOf(st.name), scan = scanOf(st.name);
      const ct = k !== null && scan && S.layers.find((o) => o !== ly && o.geo && o.kind === "ct" && scanOf(o.st.name) === scan);
      if (ct) {
        const cl = ct.st.levels[k], c0 = ct.st.levels[0];
        p0 = cl ? ct.geo.p0.map((p, a) => p * cl.scale[a] / c0.scale[a]) : ct.geo.p0.map((p) => p * 2 ** k);
        rule = `level ${k} of the CT layer ${ct.st.name} (name -L${k}-)`;
      } else {
        const b = S.layers.find((o) => o !== ly && o.geo);
        if (b) { p0 = b.geo.p0.slice(); rule = "unknown: assumed the level-0 grid of " + b.st.name; }
        else { p0 = [1, 1, 1]; rule = "unknown: 1 unit per voxel"; }
        ly.pitchNote = "no voxel size in the metadata; " + rule;
      }
    }
    if (st.uf && !/^OME/.test(rule)) {
      const ome = L0.scale.map((s) => s * st.uf);
      if (ome.some((x, a) => Math.abs(x - p0[a]) > 1e-3 * p0[a]))
        ly.pitchNote = `the OME scale says ${fmtUm(ome[0])}; using ${fmtUm(p0[0])} (${rule})`;
    }
    ly.p0 = p0;
    ly.rule = rule;
    const oz = !st.isRegionSet && Array.isArray(st.attrs.origin_zyx) ? st.attrs.origin_zyx.slice(-3) : null;
    for (const L of st.levels) {
      L.pitch = p0.map((p, a) => p * L.scale[a] / L0.scale[a]);
      if (st.hasTrans) L.origin = L.trans.map((t, a) => t * (st.uf || L.pitch[a] / L.scale[a]) - L.pitch[a] / 2);
      else if (oz) L.origin = oz.map((o, a) => o * p0[a]);
      else L.origin = [0, 0, 0];
    }
  }
  applyCaps();
}
// every store gets an equal share of the decoded cache
function layerShare() { return cache.cap / Math.max(1, new Set(S.layers.map((l) => l.st.id)).size); }
function applyCaps() {
  cache.ownCap.clear();
  const share = Math.floor(layerShare());
  for (const ly of S.layers) cache.ownCap.set(ly.st.id, share);
  cache.trim();
}
// union of the visible layers' level-0 boxes (µm)
function worldBox() {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const ls = S.layers.filter((l) => l.visible && l.st.levels[0].pitch);
  for (const ly of ls.length ? ls : S.layers.filter((l) => l.st.levels[0].pitch)) {
    const L = ly.st.levels[0];
    for (let a = 0; a < 3; a++) {
      lo[a] = Math.min(lo[a], L.origin[a]);
      hi[a] = Math.max(hi[a], L.origin[a] + L.shape[a] * L.pitch[a]);
    }
  }
  if (!(hi[0] > lo[0])) return { lo: [0, 0, 0], hi: [1, 1, 1] };
  return { lo, hi };
}

// ---------------------------------------------------------------- layers
let nextLayerId = 1;
const HUES = {
  red: [255, 64, 64], green: [80, 230, 90], blue: [80, 140, 255], cyan: [40, 220, 230],
  magenta: [240, 80, 230], yellow: [255, 225, 60], orange: [255, 150, 40],
};
const HUE_ORDER = ["red", "green", "cyan", "magenta", "yellow", "blue", "orange"];
const LUT_NAMES = ["grey", ...Object.keys(HUES), "fire", "categorical"];
const BLENDS = ["normal", "add", "max"];
const KINDS = { ct: "CT", prob: "probability", mask: "mask", label: "labels" };

function classify(st) {
  if (st.isRegionSet) return "prob";
  const n = st.name, u = st.url + "/", vc = st.vc || {};
  if (/\/volumes\//.test(u)) return "ct";
  // structural signals only (free-text "content" mentions masked CT in probability stores)
  const enc = String((vc.encoding && vc.encoding.name) || "");
  if (/label|instance/i.test(n) || /label/i.test(enc) || Array.isArray(vc.labels) || vc.classes) return "label";
  if (/-th\d|[-_]mask(?!ed)/i.test(n) || /mask/i.test(enc) || st.omeType === "majority" || vc.threshold !== undefined) return "mask";
  if (/\/representations\//.test(u) || /prob|pred|ink|surface/i.test(n) || /probab/i.test(String(vc.content || ""))) return "prob";
  return "ct";
}
function nextHue() {
  const used = S.layers.filter((l) => l.kind !== "ct").map((l) => l.hue);
  return HUE_ORDER.find((h) => !used.includes(h)) || HUE_ORDER[S.layers.length % HUE_ORDER.length];
}
function newLayer(st, o = {}) {
  const kind = o.kind || classify(st);
  const hue = HUES[o.hue] ? o.hue : kind === "ct" ? "red" : nextHue();
  // the first CT volume becomes the "CT layer" that the volume picker replaces
  if (o.role === undefined && kind === "ct" && /\/volumes\//.test(st.url + "/") && !S.layers.some((l) => l.role === "ct")) o = Object.assign({}, o, { role: "ct" });
  const ly = {
    id: nextLayerId++, st, kind, hue, visible: true,
    lut: kind === "ct" ? "grey" : kind === "mask" || kind === "label" ? "categorical" : hue,
    opacity: kind === "ct" ? 1 : 0.7, win: 255, lev: 128, thrOn: false, thr: 128,
    blend: "normal", interp: "auto", level: "auto", role: o.role || "",
  };
  return Object.assign(ly, pickOpts(o));
}
function pickOpts(o) {
  const r = {};
  if (LUT_NAMES.includes(o.lut)) r.lut = o.lut;
  if (Number.isFinite(o.opacity)) r.opacity = clamp(o.opacity, 0, 1);
  if (Number.isFinite(o.win)) r.win = clamp(Math.round(o.win), 1, 255);
  if (Number.isFinite(o.lev)) r.lev = clamp(Math.round(o.lev), 0, 255);
  if (typeof o.thrOn === "boolean") r.thrOn = o.thrOn;
  if (Number.isFinite(o.thr)) r.thr = clamp(Math.round(o.thr), 0, 255);
  if (BLENDS.includes(o.blend)) r.blend = o.blend;
  if (["auto", "linear", "nearest"].includes(o.interp)) r.interp = o.interp;
  if (o.level === "auto" || Number.isInteger(o.level)) r.level = o.level === "auto" ? "auto" : String(o.level);
  if (typeof o.level === "string" && /^\d+$/.test(o.level)) r.level = o.level;
  if (typeof o.visible === "boolean") r.visible = o.visible;
  if (o.kind && KINDS[o.kind]) r.kind = o.kind;
  if (o.role) r.role = o.role;
  return r;
}
function interpOf(ly) {
  if (ly.interp !== "auto") return ly.interp;
  return ly.kind === "mask" || ly.kind === "label" || ly.lut === "categorical" ? "nearest" : "linear";
}
function smoothFor(ly) {
  if (!S.smooth.on) return [0, 0];
  if (ly.kind === "ct") return [S.smooth.ct, SMOOTH_CT_FLAGS];
  if (ly.kind === "prob") return S.smooth.pred ? [S.smooth.pr, SMOOTH_PRED_FLAGS] : [0, 0];
  return [0, 0]; // masks and labels are never smoothed
}
function layerColor(ly) {
  const c = ly.lut === "grey" ? [200, 200, 200] : ly.lut === "fire" ? [255, 140, 30] : ly.lut === "categorical" ? HUES[ly.hue] : HUES[ly.lut];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function shortName(ly) {
  const n = ly.st.name.replace(/\.zarr$/, "");
  const scan = scanOf(n);
  let s = n;
  if (scan) s = s.slice(scan.length + 1);
  s = s.replace(/^surface-\d{14}-/, "");
  return s || n;
}

// ---------------------------------------------------------------- LUTs
// Per layer and value v: colour rgb[v] and alpha a[v] in 0..255, times the opacity
// O = round(255 * opacity), give A[v] = a[v] * O in 0..65025. Blending into the
// 8-bit destination d (integers throughout, so a reference can reproduce it exactly):
//   normal   d = floor((d * (65025 - A) + c * A + 32512) / 65025)
//   add      d = min(255, d + floor((c * A + 32512) / 65025))
//   max      d = max(d, floor((c * A + 32512) / 65025))
// grey: c = w(v) (the window), a = 255 · hue ramps: c = hue, a = w(v) · fire: c = fire(w(v)), a = w(v)
// categorical: c = palette(v) (255 -> the layer hue), a = 255 for v > 0 · threshold: a = v >= thr ? a : 0
// (for ramps and fire the threshold makes a = 255 above it). w(v) = clamp(floor((v - lo) / win * 255 + 0.5)),
// lo = level - win / 2.
function hsl(h, s, l) {
  const k = (n) => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)].map((x) => Math.floor(x * 255 + 0.5));
}
const PALETTE = Array.from({ length: 256 }, (_, v) => hsl((v * 137.50776) % 360, 0.75, 0.58));
function lutRgb(ly, v, w) {
  switch (ly.lut) {
    case "grey": return [w, w, w];
    case "fire": return [Math.min(255, 3 * w), clamp(3 * w - 255, 0, 255), clamp(3 * w - 510, 0, 255)];
    case "categorical": return v === 255 ? HUES[ly.hue] : PALETTE[v];
    default: return HUES[ly.lut] || HUES.red;
  }
}
function layerLut(ly) {
  const key = [ly.lut, ly.hue, ly.opacity, ly.win, ly.lev, ly.thrOn, ly.thr, ly.blend].join(",");
  if (ly._lk === key) return ly._lut;
  const lo = ly.lev - ly.win / 2, win = Math.max(1, ly.win);
  const O = Math.floor(ly.opacity * 255 + 0.5);
  const A = new Int32Array(256), R = new Int32Array(256), G = new Int32Array(256), B = new Int32Array(256);
  const bl = BLENDS.indexOf(ly.blend);
  for (let v = 0; v < 256; v++) {
    const w = clamp(Math.floor(((v - lo) / win) * 255 + 0.5), 0, 255);
    const c = lutRgb(ly, v, w);
    let a = ly.lut === "grey" ? 255 : ly.lut === "categorical" ? (v ? 255 : 0) : w;
    if (ly.thrOn) a = v >= ly.thr ? (ly.lut === "categorical" ? (v ? 255 : 0) : 255) : 0;
    A[v] = a * O;
    const T = [R, G, B];
    for (let ch = 0; ch < 3; ch++) T[ch][v] = bl === 0 ? c[ch] * A[v] + 32512 : Math.floor((c[ch] * A[v] + 32512) / A_ONE);
  }
  ly._lk = key;
  ly._lut = { A, R, G, B, bl };
  return ly._lut;
}

// ---------------------------------------------------------------- resampling
let renderQueued = false;
function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => { renderQueued = false; renderAll(); });
}
function chunkKeys(st, li, c, smooth) {
  const L = st.levels[li];
  const s = [0, 1, 2].map((a) => Math.floor(c[a] / L.cps[a]));
  const l = [0, 1, 2].map((a) => c[a] - s[a] * L.cps[a]);
  const inner = (l[0] * L.cps[1] + l[1]) * L.cps[2] + l[2];
  const ckey = st.id + "|" + li + "|" + s.join(".") + "|" + inner; // (store, level, shard, chunk)
  return [ckey + "|" + smooth, ckey];
}
// Chunks one view may hold per layer: VIEW_SHARE of the layer's cache share,
// split between the views on screen (the minimap gets a small fixed part).
function viewBudget(T) {
  const sh = layerShare() / CV;
  if (T.minimap) return Math.max(8, Math.floor(sh * 0.05));
  return Math.max(16, Math.floor(sh * VIEW_SHARE / (S.maximized >= 0 ? 1 : 3)));
}
// Inner chunks a target needs from level li (the same ranges layerPlane reads).
function chunksNeeded(ly, li, T) {
  const L = ly.st.levels[li], axis = T.axis, [ua, va] = UV[axis];
  const lin = interpOf(ly) === "linear";
  const ss = sliceSamples((T.s - L.origin[axis]) / L.pitch[axis], L.shape[axis], lin);
  if (!ss) return 0;
  const span = (w0, N, a) => {
    const lo = Math.floor((w0 - L.origin[a]) / L.pitch[a] - 0.5), hi = Math.floor((w0 + N * T.du - L.origin[a]) / L.pitch[a] + 0.5);
    const a0 = clamp(lo, 0, L.shape[a] - 1), a1 = clamp(hi, 0, L.shape[a] - 1);
    return a1 < a0 ? 0 : (a1 >> 7) - (a0 >> 7) + 1;
  };
  const planes = ss[2] && ss[0] >> 7 !== ss[1] >> 7 ? 2 : 1;
  return span(T.u0, T.W, ua) * span(T.v0, T.H, va) * planes;
}
// Per view and layer: the coarsest level whose pitch (along the view's horizontal
// axis) is at most one screen pixel (du µm), or the layer's fixed level.
function pickLevel(ly, axis, du) {
  const Ls = ly.st.levels, ua = UV[axis][0];
  if (ly.level !== "auto") return clamp(+ly.level, 0, Ls.length - 1);
  let best = 0;
  for (let li = 0; li < Ls.length; li++) if (Ls[li].pitch[ua] <= du * (1 + 1e-6)) best = li;
  return best;
}
// Sample positions along one axis. World position of sample i: w = w0 + (i + 0.5) * d.
// Continuous level coordinate c = (w - origin) / pitch (voxel k spans [k, k+1)).
// nearest: a = floor(c). linear: t = c - 0.5, a = floor(t), weight f = floor((t - a) * 256 + 0.5)
// of the next voxel b = a + 1 (f = 256 -> a + 1 with f = 0), both clamped to the array (a == b -> f = 0).
// Samples with c outside [0, n) are outside the layer (a = -1).
function axisSamples(w0, d, N, org, pit, n, lin) {
  const A = new Int32Array(N), B = new Int32Array(N), F = new Int32Array(N);
  let mn = Infinity, mx = -1;
  for (let i = 0; i < N; i++) {
    const c = (w0 + (i + 0.5) * d - org) / pit;
    if (!(c >= 0 && c < n)) { A[i] = -1; continue; }
    let a, b, f;
    if (lin) {
      const t = c - 0.5;
      a = Math.floor(t);
      f = Math.floor((t - a) * W8 + 0.5);
      if (f === W8) { a++; f = 0; }
      b = a + 1;
      a = a < 0 ? 0 : a > n - 1 ? n - 1 : a;
      b = b < 0 ? 0 : b > n - 1 ? n - 1 : b;
      if (a === b) f = 0;
    } else { a = Math.floor(c); b = a; f = 0; }
    A[i] = a; B[i] = b; F[i] = f;
    if (a < mn) mn = a;
    const top = f ? b : a;
    if (top > mx) mx = top;
  }
  if (mx < 0) return null;
  return { A, B, F, mn, mx };
}
function sliceSamples(cs, n, lin) {
  if (!(cs >= 0 && cs < n)) return null;
  if (!lin) { const a = Math.floor(cs); return [a, a, 0]; }
  const t = cs - 0.5;
  let a = Math.floor(t), f = Math.floor((t - a) * W8 + 0.5);
  if (f === W8) { a++; f = 0; }
  let b = a + 1;
  a = clamp(a, 0, n - 1); b = clamp(b, 0, n - 1);
  if (a === b) f = 0;
  return [a, b, f];
}
function getBuf(owner, name, n, Type) {
  const k = "_b_" + name;
  if (!owner[k] || owner[k].length < n) owner[k] = new Type(Math.max(n, 1024));
  return owner[k];
}

// Stage 1: the layer's level voxels covering the target, interpolated along the
// slice axis, as an Int32 plane (-1 not loaded, -2 error). Linear planes hold
// v(s0) * (256 - ws) + v(s1) * ws, nearest planes hold v. Missing chunks go to `wanted`.
function layerPlane(ly, li, T, wanted, prio0, bufOwner) {
  const st = ly.st, L = st.levels[li], axis = T.axis, [ua, va] = UV[axis];
  const p = L.pitch, o = L.origin, n = L.shape;
  const lin = interpOf(ly) === "linear";
  const ss = sliceSamples((T.s - o[axis]) / p[axis], n[axis], lin);
  if (!ss) return null;
  const [s0, s1, ws] = ss;
  const cols = axisSamples(T.u0, T.du, T.W, o[ua], p[ua], n[ua], lin);
  const rows = axisSamples(T.v0, T.du, T.H, o[va], p[va], n[va], lin);
  if (!cols || !rows) return null;
  const nx = cols.mx - cols.mn + 1, ny = rows.mx - rows.mn + 1;
  if (!T.tile && nx * ny > Math.max(4 << 20, PLANE_MAX * T.W * T.H)) return { tooBig: true };
  const [smooth, flags] = smoothFor(ly);
  const cx0 = cols.mn >> 7, cx1 = cols.mx >> 7, cy0 = rows.mn >> 7, cy1 = rows.mx >> 7;
  const scs = ws ? [...new Set([s0 >> 7, s1 >> 7])] : [s0 >> 7];
  const ncx = cx1 - cx0 + 1, ncy = cy1 - cy0 + 1;
  const tab = new Array(scs.length * ncx * ncy);
  const miss = [];
  let have = 0; // decoded chunks of this view already in the cache
  // view centre in chunk units, for the loading order
  const mu = ((T.u0 + T.W * T.du / 2 - o[ua]) / p[ua]) / C, mv = ((T.v0 + T.H * T.du / 2 - o[va]) / p[va]) / C;
  for (let k = 0; k < scs.length; k++) for (let j = 0; j < ncy; j++) for (let i = 0; i < ncx; i++) {
    const c = [0, 0, 0];
    c[axis] = scs[k]; c[va] = cy0 + j; c[ua] = cx0 + i;
    const [key, ckey] = chunkKeys(st, li, c, smooth);
    const a = cache.get(key);
    tab[(k * ncy + j) * ncx + i] = a;
    if (T.pins) T.pins.add(key);
    if (a !== undefined && a.length) have++;
    if (a === undefined) {
      const d = Math.hypot(cx0 + i + 0.5 - mu, cy0 + j + 0.5 - mv);
      miss.push({ key, ckey, st, li, c, smooth, flags, prio: prio0 + d });
    }
  }
  wanted.visible = (wanted.visible || 0) + tab.length;
  if (!T.tile) {
    // never load more than the view's budget: past it, loading would evict what is on screen
    const cap = Math.max(0, viewBudget(T) - have);
    if (miss.length > cap) {
      miss.sort((a, b) => a.prio - b.prio);
      if (!T.minimap) S.warn = `${shortName(ly)}: the view needs ${tab.length} chunks at level ${li}; ${have + cap} fit its cache share (zoom in, pick a coarser level or a bigger cache)`;
      miss.length = cap;
    }
  }
  for (const m of miss) wanted.push(m);
  if (miss.length) T.pending = true;
  // prefetch the neighbouring chunk layers along the slice axis (renderAll trims these to the cache budget)
  if (wanted.pre && ncx * ncy <= PREFETCH_MAX_VISIBLE) {
    for (const dd of [1, -1]) {
      const cn = (dd > 0 ? Math.max(...scs) : Math.min(...scs)) + dd;
      if (cn < 0 || cn >= L.grid[axis]) continue;
      for (let j = 0; j < ncy; j++) for (let i = 0; i < ncx; i++) {
        const c = [0, 0, 0];
        c[axis] = cn; c[va] = cy0 + j; c[ua] = cx0 + i;
        const [key, ckey] = chunkKeys(st, li, c, smooth);
        if (!cache.has(key)) wanted.pre.push({ key, ckey, st, li, c, smooth, flags, prio: 1e6 + prio0 + Math.abs((s0 & 127) - (dd > 0 ? 128 : -1)) });
      }
    }
  }
  // fill the plane chunk by chunk
  const plane = getBuf(bufOwner, "plane", nx * ny, Int32Array);
  const k0 = scs.indexOf(s0 >> 7), k1 = ws ? scs.indexOf(s1 >> 7) : k0;
  const so0 = (s0 & 127) * MUL[axis], so1 = (s1 & 127) * MUL[axis];
  const mU = MUL[ua], mV = MUL[va], iw = W8 - ws;
  for (let j = 0; j < ncy; j++) for (let i = 0; i < ncx; i++) {
    const a0 = tab[(k0 * ncy + j) * ncx + i], a1 = tab[(k1 * ncy + j) * ncx + i];
    const y0 = Math.max(rows.mn, (cy0 + j) * C), y1 = Math.min(rows.mx, (cy0 + j) * C + C - 1);
    const x0 = Math.max(cols.mn, (cx0 + i) * C), x1 = Math.min(cols.mx, (cx0 + i) * C + C - 1);
    let bad = 0;
    if (a0 === undefined || a1 === undefined) bad = -1;
    if (a0 === ERR || a1 === ERR) bad = -2;
    const z0 = a0 === ZERO, z1 = a1 === ZERO;
    for (let y = y0; y <= y1; y++) {
      let pi = (y - rows.mn) * nx + (x0 - cols.mn);
      const rb = (y & 127) * mV;
      if (bad) { for (let x = x0; x <= x1; x++) plane[pi++] = bad; continue; }
      if (!lin) {
        if (z0) for (let x = x0; x <= x1; x++) plane[pi++] = 0;
        else for (let x = x0, q = so0 + rb; x <= x1; x++) plane[pi++] = a0[q + (x & 127) * mU];
      } else if (!ws) {
        if (z0) for (let x = x0; x <= x1; x++) plane[pi++] = 0;
        else for (let x = x0, q = so0 + rb; x <= x1; x++) plane[pi++] = a0[q + (x & 127) * mU] * W8;
      } else {
        for (let x = x0; x <= x1; x++) {
          const xo = (x & 127) * mU;
          const v0 = z0 ? 0 : a0[so0 + rb + xo], v1 = z1 ? 0 : a1[so1 + rb + xo];
          plane[pi++] = v0 * iw + v1 * ws;
        }
      }
    }
  }
  // sample indices relative to the plane
  for (let i = 0; i < T.W; i++) if (cols.A[i] >= 0) { cols.A[i] -= cols.mn; cols.B[i] -= cols.mn; }
  for (let j = 0; j < T.H; j++) if (rows.A[j] >= 0) { rows.A[j] = (rows.A[j] - rows.mn) * nx; rows.B[j] = (rows.B[j] - rows.mn) * nx; }
  return { plane, cols, rows, lin };
}

// Stage 2 + compositing: every visible layer, bottom to top, resampled at the
// target's pixel centres (bilinear on the plane = trilinear overall, or nearest)
// and blended. T = {axis, W, H, u0, v0, du, s} (µm); returns an ImageData.
const MISSING = [26, 30, 40], ERRC = [70, 18, 18];
function composite(T, wanted, prio0, hist, bufOwner) {
  const W = T.W, H = T.H, N = W * H;
  const dr = getBuf(bufOwner, "r", N, Uint8Array), dg = getBuf(bufOwner, "g", N, Uint8Array), db = getBuf(bufOwner, "b", N, Uint8Array);
  dr.fill(0, 0, N); dg.fill(0, 0, N); db.fill(0, 0, N);
  T.levels = new Map();
  let base = true;
  S.layers.forEach((ly, idx) => {
    if (!ly.visible) return;
    const isBase = base;
    base = false;
    const forced = T.force && T.force.get(ly.id);
    let li = forced !== undefined ? forced : pickLevel(ly, T.axis, T.du);
    // auto levels step coarser until the view's chunks fit the layer's budget
    if (forced === undefined && ly.level === "auto" && !T.tile) {
      const nL = ly.st.levels.length, bud = viewBudget(T), li0 = li, n0 = chunksNeeded(ly, li, T);
      while (li < nL - 1 && chunksNeeded(ly, li, T) > bud) li++;
      if (li !== li0 && !T.minimap)
        S.warn = S.warn || `${shortName(ly)}: drawn at level ${li} instead of ${li0} — its ${n0} chunks there exceed its ${fmtBytes(layerShare())} cache share (raise the cache)`;
    }
    const L = ly.st.levels[li];
    if (forced === undefined && L.pitch[UV[T.axis][0]] * TOO_FINE < T.du) {
      T.levels.set(ly.id, -2);
      if (!T.minimap) S.warn = S.warn || `${shortName(ly)}: zoom in to load it (its coarsest level is ${fmtUm(L.pitch[0])})`;
      return;
    }
    T.levels.set(ly.id, li);
    const P = layerPlane(ly, li, T, wanted, prio0 + idx * 0.25, bufOwner);
    if (!P) return;
    if (P.tooBig) {
      T.levels.set(ly.id, -2);
      if (!T.minimap) S.warn = S.warn || `${shortName(ly)}: zoom in to load it (level ${li} is too fine for this view)`;
      return;
    }
    const { plane, cols, rows, lin } = P;
    const cA = cols.A, cB = cols.B, cF = cols.F, rA = rows.A, rB = rows.B, rF = rows.F;
    const lut = layerLut(ly), LA = lut.A, LR = lut.R, LG = lut.G, LB = lut.B, bl = lut.bl;
    const h = hist && idx === S.sel ? hist : null;
    for (let j = 0; j < H; j++) {
      const ra = rA[j];
      if (ra < 0) continue;
      const rb = rB[j], wy = rF[j], iwy = W8 - wy, ob = j * W;
      for (let i = 0; i < W; i++) {
        const ca = cA[i];
        if (ca < 0) continue;
        let v;
        if (lin) {
          const wx = cF[i];
          const p00 = plane[ra + ca];
          if (wx === 0 && wy === 0) v = p00 < 0 ? p00 : Math.floor((p00 * 65536 + HALF24) / W24);
          else {
            const cb = cB[i];
            const p01 = wx ? plane[ra + cb] : 0, p10 = wy ? plane[rb + ca] : 0, p11 = wx && wy ? plane[rb + cb] : 0;
            if (p00 < 0 || p01 < 0 || p10 < 0 || p11 < 0) v = Math.min(p00, p01, p10, p11);
            else v = Math.floor(((p00 * (W8 - wx) + p01 * wx) * iwy + (p10 * (W8 - wx) + p11 * wx) * wy + HALF24) / W24);
          }
        } else v = plane[ra + ca];
        const k = ob + i;
        if (v < 0) {
          if (isBase) { const c = v === -2 ? ERRC : MISSING; dr[k] = c[0]; dg[k] = c[1]; db[k] = c[2]; }
          continue;
        }
        if (h) h[v]++;
        const a = LA[v];
        if (a === 0) continue;
        if (bl === 0) {
          const ia = A_ONE - a;
          dr[k] = Math.floor((dr[k] * ia + LR[v]) / A_ONE);
          dg[k] = Math.floor((dg[k] * ia + LG[v]) / A_ONE);
          db[k] = Math.floor((db[k] * ia + LB[v]) / A_ONE);
        } else if (bl === 1) {
          dr[k] = Math.min(255, dr[k] + LR[v]); dg[k] = Math.min(255, dg[k] + LG[v]); db[k] = Math.min(255, db[k] + LB[v]);
        } else {
          if (LR[v] > dr[k]) dr[k] = LR[v];
          if (LG[v] > dg[k]) dg[k] = LG[v];
          if (LB[v] > db[k]) db[k] = LB[v];
        }
      }
    }
  });
  const img = new ImageData(W, H);
  const px = new Uint32Array(img.data.buffer);
  for (let k = 0; k < N; k++) px[k] = (0xff000000 | db[k] << 16 | dg[k] << 8 | dr[k]) >>> 0;
  return img;
}

// ---------------------------------------------------------------- views
// the view's world rectangle; rs > 1 composites at 1/rs resolution (same top-left corner)
function viewTarget(V, rs = 1) {
  const W = V.canvas.width, H = V.canvas.height, du = 1 / V.zoom;
  return { axis: V.axis, W: Math.ceil(W / rs), H: Math.ceil(H / rs), du: du * rs, u0: V.center[0] - (W / 2) * du, v0: V.center[1] - (H / 2) * du, s: S.cursor[V.axis], pins: S.pins };
}
function renderView(V, wanted) {
  const cv = V.canvas, ctx = V.ctx;
  const W = cv.width, H = cv.height;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, W, H);
  if (!S.layers.length || W < 2 || H < 2) { V.label.textContent = ""; return; }
  const [ua, va] = UV[V.axis];
  const rs = W * H > RENDER_MAX_PX ? Math.sqrt(W * H / RENDER_MAX_PX) : 1;
  const T = viewTarget(V, rs);
  const prio0 = V.axis === S.active ? 0 : 10;
  const hist = V.axis === S.active ? (S.hist = new Uint32Array(256)) : null;
  const img = composite(T, wanted, prio0, hist, V);
  V.levels = T.levels;
  if (rs === 1) ctx.putImageData(img, 0, 0);
  else {
    if (!V.buf || V.buf.width !== img.width || V.buf.height !== img.height) V.buf = OffscreenCanvasOr(img.width, img.height);
    V.buf.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(V.buf, 0, 0, img.width * rs, img.height * rs);
  }
  const du = 1 / V.zoom;
  const X = (w) => (w - T.u0) / du, Y = (w) => (w - T.v0) / du;
  if (S.outlines) {
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.35;
    for (const ly of S.layers) {
      if (!ly.visible) continue;
      const L = ly.st.levels[0];
      ctx.strokeStyle = layerColor(ly);
      ctx.strokeRect(Math.round(X(L.origin[ua])) + 0.5, Math.round(Y(L.origin[va])) + 0.5,
        Math.round(L.shape[ua] * L.pitch[ua] / du), Math.round(L.shape[va] * L.pitch[va] / du));
    }
    ctx.globalAlpha = 1;
  }
  if (S.crosshair) {
    const x = X(S.cursor[ua]), y = Y(S.cursor[va]);
    ctx.globalAlpha = 0.55;
    ctx.strokeStyle = AXIS_COLORS[ua];
    ctx.beginPath(); ctx.moveTo(Math.floor(x) + 0.5, 0); ctx.lineTo(Math.floor(x) + 0.5, H); ctx.stroke();
    ctx.strokeStyle = AXIS_COLORS[va];
    ctx.beginPath(); ctx.moveTo(0, Math.floor(y) + 0.5); ctx.lineTo(W, Math.floor(y) + 0.5); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  const parts = [];
  for (const ly of S.layers) {
    if (!ly.visible) continue;
    const li = V.levels.get(ly.id);
    const L = li >= 0 ? ly.st.levels[li] : null;
    parts.push(`${shortName(ly).slice(0, 30)} ${L ? "L" + li + (ly.level === "auto" ? "" : "!") + " " + fmtUm(L.pitch[ua]) : "(zoom in)"}`);
  }
  V.label.textContent = `${AXES[V.axis]} ${fmtUm(S.cursor[V.axis])} · ${(V.zoom).toFixed(V.zoom < 1 ? 3 : 2)} px/µm\n` + parts.join("  ·  ");
}

// OffscreenCanvas where available (it is in every current browser), else a detached <canvas>
function OffscreenCanvasOr(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

// minimap: the active view's slice over the whole world box, with the viewport rectangle
function renderMinimap(wanted) {
  const cv = $("minimap"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#07090c";
  ctx.fillRect(0, 0, W, H);
  if (!S.layers.length) return;
  const V = S.views[S.active];
  const [ua, va] = UV[V.axis];
  const { lo, hi } = worldBox();
  const eu = hi[ua] - lo[ua], ev = hi[va] - lo[va];
  const du = Math.max(eu / W, ev / H);
  const T = { axis: V.axis, W, H, du, u0: lo[ua] + eu / 2 - (W / 2) * du, v0: lo[va] + ev / 2 - (H / 2) * du, s: S.cursor[V.axis], minimap: true, pins: S.pins };
  const img = composite(T, wanted, 5, null, cv);
  ctx.putImageData(img, 0, 0);
  S.mini = { u0: T.u0, v0: T.v0, du, axis: V.axis };
  const VT = viewTarget(V);
  ctx.strokeStyle = AXIS_COLORS[V.axis];
  ctx.lineWidth = 1.5;
  ctx.strokeRect((VT.u0 - T.u0) / du, (VT.v0 - T.v0) / du, VT.W * VT.du / du, VT.H * VT.du / du);
  const lv = [...T.levels.entries()].map(([id, li]) => li >= 0 ? "L" + li : "-").join(" ");
  $("minimap-label").textContent = `${AXES[V.axis]} view · levels ${lv}`;
}

let lastLayerUi = "";
function renderAll() {
  const t0 = performance.now();
  stats.renders = (stats.renders || 0) + 1;
  const wanted = [];
  wanted.pre = [];
  S.warn = "";
  const pins = new Set(); // every chunk this frame draws: the cache never evicts these
  S.pins = pins;
  for (const V of S.views) {
    if (S.maximized >= 0 && V.axis !== S.maximized) continue;
    renderView(V, wanted);
  }
  renderMinimap(wanted);
  cache.pinned = pins;
  if (S.extraWant) for (const it of S.extraWant) wanted.push(Object.assign({}, it, { prio: it.prio - 1e3 }));
  // prefetch per layer only into what is left of its share (90 %) after the chunks on screen
  const onScreen = new Map();
  for (const k of pins) { const o = +k.slice(0, k.indexOf("|")); onScreen.set(o, (onScreen.get(o) || 0) + 1); }
  const left = new Map();
  wanted.pre.sort((a, b) => a.prio - b.prio);
  for (const it of wanted.pre) {
    const o = it.st.id;
    if (!left.has(o)) left.set(o, Math.floor(layerShare() / CV * 0.9) - (onScreen.get(o) || 0));
    if (left.get(o) <= 0 || pins.has(it.key)) continue;
    left.set(o, left.get(o) - 1);
    wanted.push(it);
  }
  if (loader) loader.want(wanted);
  stats.renderMs = performance.now() - t0;
  // the layer list shows each layer's level in the active view
  const V = S.views[S.active];
  const sig = S.layers.map((l) => l.id + ":" + (V && V.levels ? V.levels.get(l.id) : "")).join(",");
  if (sig !== lastLayerUi) { lastLayerUi = sig; renderLayerList(); }
  updateStatus();
}

// ---------------------------------------------------------------- status
function updateStatus() {
  const avg = stats.decoded ? stats.decodeMs / stats.decoded : 0;
  const parts = [
    `fetched ${fmtBytes(stats.bytes)} in ${stats.requests} req (${stats.indexReads} index, ${stats.rangeReads} range, ${stats.notFound} 404)`,
    `decoded ${stats.decoded} chunks · ${avg.toFixed(1)} ms avg · ${stats.lastDecodeMs.toFixed(1)} ms last · ${stats.zeroChunks} air`,
    `cache ${fmtBytes(cache.bytes)} / ${fmtBytes(cache.cap)} (${cache.size}; ${fmtBytes(layerShare())} per layer)`,
    `render ${(stats.renderMs || 0).toFixed(0)} ms`,
    loader ? `queue ${loader.queue.length} · fetching ${loader.nFetch} · decoding ${pool.busy}` : "",
  ];
  $("status-main").textContent = parts.filter(Boolean).join("   |   ");
  const w = S.warn || stats.lastError;
  if (!$("status-warn").classList.contains("err") || !w) {
    $("status-warn").textContent = w;
    $("status-warn").title = w;
  }
  if (S.hover) {
    const h = S.hover;
    let txt = `cursor z ${fmtUm(S.cursor[0])} y ${fmtUm(S.cursor[1])} x ${fmtUm(S.cursor[2])}\n` +
      `mouse  z ${fmtUm(h.w[0])} y ${fmtUm(h.w[1])} x ${fmtUm(h.w[2])}`;
    for (const ly of S.layers) {
      if (!ly.visible) continue;
      const li = h.levels && h.levels.get(ly.id);
      if (li === undefined || li < 0) continue;
      const r = sampleAt(ly, li, h.w);
      if (!r) continue;
      const val = ly.kind === "prob" ? (r.v / 255).toFixed(3) : String(r.v);
      txt += `\n${shortName(ly).slice(0, 28)}: ${val}  [L${li} ${r.i.join(",")}]`;
    }
    $("info-hover").textContent = txt;
  }
}
// decoded value (nearest voxel) at a world point from the cache, or null
function sampleAt(ly, li, w) {
  const st = ly.st, L = st.levels[li];
  const i = w.map((x, a) => Math.floor((x - L.origin[a]) / L.pitch[a]));
  if (i.some((x, a) => x < 0 || x >= L.shape[a])) return null;
  const [key] = chunkKeys(st, li, i.map((x) => x >> 7), smoothFor(ly)[0]);
  const arr = cache.peek(key);
  if (arr === undefined || arr === ERR) return null;
  return { v: arr === ZERO ? 0 : arr[(i[0] & 127) * MUL[0] + (i[1] & 127) * MUL[1] + (i[2] & 127)], i };
}

// ---------------------------------------------------------------- views + input
function setupViews() {
  S.views = [0, 1, 2].map((axis) => {
    const el = $("view" + axis);
    const canvas = el.querySelector("canvas");
    return { axis, el, canvas, ctx: canvas.getContext("2d", { alpha: false }), label: el.querySelector(".vlabel"), zoom: 1, center: [0, 0], levels: new Map() };
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
  const { lo, hi } = worldBox();
  const W = V.canvas.width, H = V.canvas.height;
  V.zoom = Math.min(W / (hi[ua] - lo[ua]), H / (hi[va] - lo[va])) * 0.95;
  V.center = [(lo[ua] + hi[ua]) / 2, (lo[va] + hi[va]) / 2];
}
function fitAll() { for (const V of S.views) fitView(V); saveHash(); requestRender(); }
function centreCursor() { const { lo, hi } = worldBox(); S.cursor = lo.map((x, a) => (x + hi[a]) / 2); }

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
  const { lo, hi } = worldBox();
  S.cursor = w.map((x, a) => clamp(x, lo[a], hi[a] - 1e-6));
  saveHash();
  requestRender();
}
// step to the centre of the next voxel of the finest visible layer (at the view's levels)
function stepSlice(V, n) {
  const a = V.axis;
  let best = null;
  for (const ly of S.layers) {
    if (!ly.visible) continue;
    const li = V.levels && V.levels.get(ly.id);
    if (li === undefined || li < 0) continue;
    const L = ly.st.levels[li];
    if (!best || L.pitch[a] < best.pitch[a]) best = L;
  }
  if (!best) return;
  const p = best.pitch[a], o = best.origin[a];
  const w = S.cursor.slice();
  w[a] = o + (Math.floor((w[a] - o) / p) + n + 0.5) * p;
  setCursorWorld(w);
}
function zoomView(V, factor, px, py) {
  if (px === undefined) { px = V.canvas.width / 2; py = V.canvas.height / 2; }
  const wx = V.center[0] + (px - V.canvas.width / 2) / V.zoom;
  const wy = V.center[1] + (py - V.canvas.height / 2) / V.zoom;
  V.zoom = clamp(V.zoom * factor, 1e-5, 256);
  V.center[0] = wx - (px - V.canvas.width / 2) / V.zoom;
  V.center[1] = wy - (py - V.canvas.height / 2) / V.zoom;
  saveHash();
  requestRender();
}
function attachViewInput(V) {
  const cv = V.canvas;
  let drag = null;
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("pointerdown", (ev) => {
    setActive(V.axis);
    cv.setPointerCapture(ev.pointerId);
    const ly = S.layers[S.sel];
    drag = { x: ev.clientX, y: ev.clientY, b: ev.button, moved: false, c: V.center.slice(), win: ly ? ly.win : 255, lev: ly ? ly.lev : 128, shift: ev.shiftKey };
  });
  cv.addEventListener("pointermove", (ev) => {
    if (!S.layers.length) return;
    const e = eventWorld(V, ev);
    S.hover = { w: e.w, levels: V.levels };
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.b === 2 || (drag.b === 0 && drag.shift)) { // window / level of the selected layer
        const ly = S.layers[S.sel];
        if (ly) {
          ly.win = clamp(Math.round(drag.win + dx), 1, 255);
          ly.lev = clamp(Math.round(drag.lev - dy), 0, 255);
          layerChanged(ly, true);
        }
      } else if (drag.moved) { // pan
        V.center[0] = drag.c[0] - (dx * e.dpr) / V.zoom;
        V.center[1] = drag.c[1] - (dy * e.dpr) / V.zoom;
        requestRender();
      }
    }
    updateStatus();
  });
  cv.addEventListener("pointerup", (ev) => {
    if (drag && !drag.moved && drag.b === 0 && !drag.shift && S.layers.length) setCursorWorld(eventWorld(V, ev).w);
    else if (drag && drag.moved) saveHash();
    drag = null;
  });
  cv.addEventListener("pointerleave", () => { S.hover = null; });
  cv.addEventListener("dblclick", () => toggleMax(V.axis));
  cv.addEventListener("wheel", (ev) => {
    if (!S.layers.length) return;
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
    V.center = [S.mini.u0 + (ev.clientX - r.left) * dpr * S.mini.du, S.mini.v0 + (ev.clientY - r.top) * dpr * S.mini.du];
    requestRender();
  };
  cv.addEventListener("pointerdown", (ev) => { down = true; cv.setPointerCapture(ev.pointerId); go(ev); });
  cv.addEventListener("pointermove", (ev) => { if (down) go(ev); });
  cv.addEventListener("pointerup", () => { down = false; saveHash(); });
}

function onKey(ev) {
  const t = ev.target;
  if (t && (t.tagName === "INPUT" && t.type !== "range" && t.type !== "checkbox" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
  if (ev.ctrlKey || ev.metaKey) return;
  if ($("addpanel").classList.contains("open")) { if (ev.key === "Escape") closeAddPanel(); return; }
  const V = S.views[S.active];
  const k = ev.key;
  const ly = S.layers[S.sel];
  let used = true;
  if (!S.layers.length && !"?hHiI".includes(k) && k !== "Insert") return;
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
    case "n": selectLayer((S.sel + 1) % S.layers.length); break;
    case "N": selectLayer((S.sel - 1 + S.layers.length) % S.layers.length); break;
    case "v": case "V": if (ly) { ly.visible = !ly.visible; layerChanged(ly, true); } break;
    case "o": case "O": { // all layers above the lowest visible one on/off
      const over = S.layers.slice(1);
      const on = !over.some((l) => l.visible);
      for (const l of over) l.visible = on;
      layerChanged(null, true);
      break;
    }
    case "t": case "T": if (ly) { ly.thrOn = !ly.thrOn; layerChanged(ly, true); } break;
    case "[": if (ly) { ly.opacity = clamp(Math.round(ly.opacity * 10 - 1) / 10, 0, 1); layerChanged(ly, true); } break;
    case "]": if (ly) { ly.opacity = clamp(Math.round(ly.opacity * 10 + 1) / 10, 0, 1); layerChanged(ly, true); } break;
    case ",": case "<": levelStep(1); break;
    case ".": case ">": levelStep(-1); break;
    case "a": case "A": if (ly) { ly.level = "auto"; layerChanged(ly, true); } break;
    case "c": case "C": S.crosshair = !S.crosshair; $("xhair").checked = S.crosshair; requestRender(); break;
    case "l": case "L": autoWindow(); break;
    case "i": case "I": case "Insert": openAddPanel(); break;
    case "?": case "h": case "H": $("help").classList.toggle("open"); break;
    default: used = false;
  }
  if (used) ev.preventDefault();
}
function levelStep(d) { // + coarser, on the selected layer
  const ly = S.layers[S.sel];
  if (!ly) return;
  const V = S.views[S.active];
  const cur = ly.level === "auto" ? Math.max(0, V.levels.get(ly.id) || 0) : +ly.level;
  ly.level = String(clamp(cur + d, 0, ly.st.levels.length - 1));
  layerChanged(ly, true);
}
function autoWindow() {
  const h = S.hist, ly = S.layers[S.sel];
  if (!h || !ly) return;
  let n = 0;
  for (let v = 1; v < 256; v++) n += h[v]; // ignore 0 (masked air, no prediction)
  if (!n) return;
  let acc = 0, lo = 1, hi = 255;
  for (let v = 1; v < 256; v++) { acc += h[v]; if (acc >= n * 0.005) { lo = v; break; } }
  acc = 0;
  for (let v = 255; v >= 1; v--) { acc += h[v]; if (acc >= n * 0.005) { hi = v; break; } }
  ly.win = Math.max(1, hi - lo);
  ly.lev = Math.round((hi + lo) / 2);
  layerChanged(ly, true);
}

// ---------------------------------------------------------------- layer panel
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// something about a layer changed: redraw, update the panel, remember it in the hash
function layerChanged(ly, settingsToo) {
  renderLayerList();
  if (settingsToo) renderLayerSettings();
  saveHash();
  requestRender();
}
function selectLayer(i) {
  if (!S.layers.length) { S.sel = -1; return; }
  S.sel = clamp(i, 0, S.layers.length - 1);
  renderLayerList();
  renderLayerSettings();
  saveHash();
  requestRender(); // the histogram follows the selected layer
}
function moveLayer(i, d) {
  const j = i + d;
  if (j < 0 || j >= S.layers.length) return;
  const L = S.layers;
  [L[i], L[j]] = [L[j], L[i]];
  if (S.sel === i) S.sel = j; else if (S.sel === j) S.sel = i;
  layerChanged(null, true);
}
function removeLayer(i) {
  S.layers.splice(i, 1);
  if (S.sel >= S.layers.length) S.sel = S.layers.length - 1;
  relayout();
  syncVolumeSelect();
  layerChanged(null, true);
}
function renderLayerList() {
  const box = $("layer-list");
  box.innerHTML = "";
  if (!S.layers.length) { box.innerHTML = '<div class="k">no layers: pick a volume above, or add one with <b>+ layer</b></div>'; return; }
  const V = S.views[S.active];
  for (let i = S.layers.length - 1; i >= 0; i--) { // top of the list = drawn last (on top)
    const ly = S.layers[i];
    const li = V && V.levels ? V.levels.get(ly.id) : undefined;
    const row = document.createElement("div");
    row.className = "lrow" + (i === S.sel ? " sel" : "");
    row.dataset.idx = i;
    row.innerHTML = `<input type="checkbox" ${ly.visible ? "checked" : ""} title="visible (V)">` +
      `<span class="sw" style="background:${layerColor(ly)}"></span>` +
      `<span class="nm" title="${esc(ly.st.url)}">${esc(shortName(ly))}</span>` +
      `<span class="meta">${KINDS[ly.kind]} · ${li === undefined ? "" : li >= 0 ? "L" + li : "zoom in"}${ly.level === "auto" ? "" : " (fixed)"} · ${ly.blend}</span>` +
      `<button data-a="up" title="move up (drawn later)">↑</button><button data-a="down" title="move down">↓</button><button data-a="rm" title="remove">×</button>`;
    row.addEventListener("click", (e) => {
      const a = e.target.dataset && e.target.dataset.a;
      if (e.target.type === "checkbox") { ly.visible = e.target.checked; layerChanged(ly, false); return; }
      if (a === "up") moveLayer(i, 1);
      else if (a === "down") moveLayer(i, -1);
      else if (a === "rm") removeLayer(i);
      else selectLayer(i);
    });
    box.appendChild(row);
  }
}
function renderLayerSettings() {
  const box = $("layer-settings");
  const ly = S.layers[S.sel];
  if (!ly) { box.innerHTML = ""; return; }
  const st = ly.st, L0 = st.levels[0];
  const opt = (vals, cur, lab) => vals.map((v, i) => `<option value="${v}" ${String(v) === String(cur) ? "selected" : ""}>${esc(lab ? lab[i] : v)}</option>`).join("");
  const lvVals = ["auto", ...st.levels.map((_, i) => i)];
  const lvLab = ["auto", ...st.levels.map((L, i) => `L${i} ${L.path ? "(" + L.path + ") " : ""}${fmtUm(L.pitch[0])} ${L.shape.join("×")}`)];
  const org = L0.origin.some((x) => x) ? " · origin " + L0.origin.map((x) => x.toFixed(1)).join(", ") + " µm" : "";
  const extra = st.isRegionSet ? ` · ${st.nRegions} regions of ${st.regionEdge.join("×")}` : "";
  box.innerHTML = `
    <span class="k">store</span><span class="nmfull" style="word-break:break-all">${esc(st.name)}${L0.zstd ? " · volcomp+zstd" : ""}</span>
    <span class="k">grid</span><span>${L0.shape.join(" × ")} at ${fmtUm(ly.p0[0])}${org} · ${st.levels.length} level${st.levels.length > 1 ? "s" : ""}${extra}</span>
    <span class="k">pitch</span><span>${esc(ly.rule || "")}</span>
    ${ly.pitchNote ? `<span class="wide" style="color:var(--warn)">${esc(ly.pitchNote)}</span>` : ""}
    <span class="k">kind</span><span><select id="ls-kind" title="changes the default sampling and smoothing">${opt(Object.keys(KINDS), ly.kind, Object.values(KINDS))}</select>
      <select id="ls-lut" title="colour map">${opt(LUT_NAMES, ly.lut)}</select>
      <select id="ls-blend" title="blend mode">${opt(BLENDS, ly.blend, ["normal", "additive", "max"])}</select></span>
    <span class="k">opacity</span><span class="grp"><input id="ls-op" type="range" min="0" max="100" value="${Math.round(ly.opacity * 100)}"><span class="val">${Math.round(ly.opacity * 100)}</span></span>
    <span class="k">window</span><span class="grp"><input id="ls-win" type="range" min="1" max="255" value="${ly.win}"><span class="val">${ly.win}</span></span>
    <span class="k">level</span><span class="grp"><input id="ls-lev" type="range" min="0" max="255" value="${ly.lev}"><span class="val">${ly.lev}</span></span>
    <span class="k"></span><span class="grp"><button id="ls-wl-auto" title="from the active view's histogram of this layer (L)">auto W/L</button><button id="ls-wl-reset">reset</button>
      <label class="grp" title="draw only values >= the threshold, opaque (T)"><input id="ls-thr-on" type="checkbox" ${ly.thrOn ? "checked" : ""}>threshold ≥</label><input id="ls-thr" type="number" min="0" max="255" value="${ly.thr}"></span>
    <span class="k">sampling</span><span><select id="ls-interp">${opt(["auto", "linear", "nearest"], ly.interp, [`auto (${interpOf(Object.assign({}, ly, { interp: "auto" }))})`, "trilinear", "nearest"])}</select>
      <select id="ls-level" title="pyramid level (auto: per view, the coarsest level at most one screen pixel)">${opt(lvVals, ly.level, lvLab)}</select></span>`;
  const on = (id, ev, fn) => { const e = $(id); if (e) e.addEventListener(ev, fn); };
  on("ls-kind", "change", (e) => { ly.kind = e.target.value; layerChanged(ly, true); });
  on("ls-lut", "change", (e) => { ly.lut = e.target.value; layerChanged(ly, true); });
  on("ls-blend", "change", (e) => { ly.blend = e.target.value; layerChanged(ly, true); });
  on("ls-op", "input", (e) => { ly.opacity = +e.target.value / 100; e.target.nextSibling.textContent = e.target.value; layerChanged(ly, false); });
  on("ls-win", "input", (e) => { ly.win = +e.target.value; e.target.nextSibling.textContent = e.target.value; layerChanged(ly, false); });
  on("ls-lev", "input", (e) => { ly.lev = +e.target.value; e.target.nextSibling.textContent = e.target.value; layerChanged(ly, false); });
  on("ls-wl-auto", "click", autoWindow);
  on("ls-wl-reset", "click", () => { ly.win = 255; ly.lev = 128; layerChanged(ly, true); });
  on("ls-thr-on", "change", (e) => { ly.thrOn = e.target.checked; layerChanged(ly, false); });
  on("ls-thr", "change", (e) => { ly.thr = clamp(Math.round(+e.target.value || 0), 0, 255); layerChanged(ly, false); });
  on("ls-interp", "change", (e) => { ly.interp = e.target.value; layerChanged(ly, false); });
  on("ls-level", "change", (e) => { ly.level = e.target.value; layerChanged(ly, false); });
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
  e.title = t;
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
  if (wantSample && !pick) { $("sample").add(new Option(wantSample, wantSample)); }
  $("sample").value = wantSample || "";
  if (wantSample) await selectSample(wantSample, true);
}
async function selectSample(name, keepSelection) {
  S.sample = name;
  setOptions($("volume"), [{ value: "", label: "loading…" }]);
  if (!name) return;
  const sroot = joinUrl(S.base, name);
  const vols = await listDir(joinUrl(sroot, "volumes")).catch(() => []);
  const vz = vols.filter((n) => /\.zarr$/.test(n));
  setOptions($("volume"), [{ value: "", label: vz.length ? "— volume —" : "(none found; paste a URL)" },
    ...vz.map((n) => ({ value: joinUrl(joinUrl(sroot, "volumes"), n), label: n }))]);
  syncVolumeSelect();
  if (keepSelection) return;
  if (vz.length) {
    $("volume").value = joinUrl(joinUrl(sroot, "volumes"), vz[0]);
    await setCtLayer($("volume").value);
  }
}
function syncVolumeSelect() {
  const ct = S.layers.find((l) => l.role === "ct");
  const sel = $("volume");
  if (ct && ![...sel.options].some((o) => o.value === ct.st.url)) sel.add(new Option(ct.st.name, ct.st.url));
  sel.value = ct ? ct.st.url : "";
}

// The volume picker: replace the CT layer (or put one at the bottom). Layers of
// other scans are dropped: different scans are not co-registered.
async function setCtLayer(url, opts = {}) {
  if (!url) return null;
  msg("opening " + url + " …");
  try {
    const st = await openStore(url);
    const scan = scanOf(st.name);
    const i = S.layers.findIndex((l) => l.role === "ct");
    let ly;
    if (i >= 0) {
      ly = S.layers[i];
      ly.st = st;
      ly.kind = classify(st);
      ly.level = "auto";
    } else {
      ly = newLayer(st, Object.assign({ role: "ct" }, opts));
      S.layers.unshift(ly);
      if (S.sel >= 0) S.sel++;
    }
    const n0 = S.layers.length;
    if (!opts.keepOthers) S.layers = S.layers.filter((l) => l === ly || !scan || !scanOf(l.st.name) || scanOf(l.st.name) === scan);
    const dropped = n0 - S.layers.length;
    if (S.sel < 0 || S.sel >= S.layers.length) S.sel = S.layers.indexOf(ly);
    relayout();
    if (!opts.keepView) { centreCursor(); fitAll(); }
    syncVolumeSelect();
    layerChanged(ly, true);
    msg(dropped ? `removed ${dropped} layer(s) of other scans (not co-registered with ${st.name})` : "");
    return ly;
  } catch (e) {
    msg("could not open volume: " + e.message, true);
    return null;
  }
}
// Add any store (or region-set directory) as the top layer.
async function addLayer(url, opts = {}) {
  msg("opening " + url + " …");
  try {
    const st = await openStore(url);
    const ly = newLayer(st, opts);
    S.layers.push(ly);
    S.sel = S.layers.length - 1;
    relayout();
    if (S.layers.length === 1 && !opts.keepView) { centreCursor(); fitAll(); }
    syncVolumeSelect();
    layerChanged(ly, true);
    msg(ly.pitchNote ? ly.pitchNote : "");
    return ly;
  } catch (e) {
    msg("could not open " + url + ": " + e.message, true);
    return null;
  }
}

// ---------------------------------------------------------------- add-layer panel
// Everything under <sample>/representations/ (recursively, from the autoindex)
// plus the sample's volumes. A directory with many region_*.zarr stores is one
// region-set entry. Each store's metadata is read to show its kind, level, size
// and pitch; stores the viewer cannot read are listed greyed with the reason.
const repCache = new Map();
function crawlReps(root) {
  if (repCache.has(root)) return repCache.get(root);
  const p = (async () => {
    const out = [];
    let budget = 120;
    async function walk(url, rel, depth) {
      if (budget-- <= 0) return;
      let idx;
      try { idx = await listIndex(url); } catch { return; }
      if (idx.files.includes("zarr.json")) { out.push({ url, rel }); return; }
      const regs = idx.dirs.filter((d) => /^region_\d+_\d+_\d+\.zarr$/.test(d));
      if (regs.length > REGION_SET_MIN) out.push({ url, rel, regions: regs.length });
      else for (const d of idx.dirs.filter((d) => /\.zarr$/i.test(d))) out.push({ url: joinUrl(url, d), rel });
      if (depth < 5) await Promise.all(idx.dirs.filter((d) => !/\.zarr$/i.test(d)).map((d) => walk(joinUrl(url, d), rel ? rel + "/" + d : d, depth + 1)));
    }
    await walk(joinUrl(root, "representations"), "", 0);
    out.sort((a, b) => (a.rel + lastSeg(a.url)).localeCompare(b.rel + lastSeg(b.url)));
    return out;
  })();
  p.catch(() => repCache.delete(root));
  repCache.set(root, p);
  return p;
}
async function limitAll(items, n, fn) {
  let i = 0;
  await Promise.all(Array.from({ length: n }, async () => { while (i < items.length) { const k = i++; await fn(items[k], k); } }));
}
function sampleRootForAdd() {
  const ly = S.layers.find((l) => sampleRootOf(l.st.url));
  if (S.sample) return joinUrl(S.base, S.sample);
  return ly ? sampleRootOf(ly.st.url) : "";
}
async function openAddPanel() {
  $("addpanel").classList.add("open");
  const root = sampleRootForAdd();
  $("add-scope").textContent = root ? lastSeg(root) + "  (" + root + ")" : "(pick a sample first, or paste a URL)";
  if (!root) { $("add-list").innerHTML = ""; return; }
  await fillAddList(root);
}
function closeAddPanel() { $("addpanel").classList.remove("open"); }
async function fillAddList(root) {
  const box = $("add-list"), status = $("add-status");
  status.textContent = "listing " + root + "/representations/ …";
  box.innerHTML = "";
  const vols = (await listDir(joinUrl(root, "volumes")).catch(() => [])).filter((n) => /\.zarr$/.test(n))
    .map((n) => ({ url: joinUrl(joinUrl(root, "volumes"), n), rel: "volumes" }));
  const reps = await crawlReps(root).catch(() => []);
  const all = [...vols, ...reps];
  status.textContent = `${vols.length} volume(s), ${reps.length} representation store(s) under ${lastSeg(root)}/representations/ — click a row to add it on top`;
  const tb = document.createElement("table");
  tb.innerHTML = "<thead><tr><th>kind</th><th>type</th><th>scan</th><th>level</th><th>size (level 0)</th><th>pitch</th><th>levels</th><th>name</th></tr></thead>";
  const body = document.createElement("tbody");
  tb.appendChild(body);
  box.appendChild(tb);
  const rows = all.map((e) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(e.rel || "")}</td><td class="ty">…</td><td>${esc(scanOf(lastSeg(e.url)))}</td><td class="lv"></td><td class="sz"></td><td class="pi"></td><td class="nl"></td>` +
      `<td class="nm">${esc(lastSeg(e.url))}${e.regions ? ` (${e.regions} region stores)` : ""}</td>`;
    tr.dataset.url = e.url;
    body.appendChild(tr);
    return tr;
  });
  await limitAll(all, 6, async (e, k) => {
    const tr = rows[k];
    try {
      const st = await openStore(e.url);
      const g = st.intrinsicPitch();
      const lk = Number.isInteger(st.vc.source_level) ? st.vc.source_level : levelTagOf(st.name);
      tr.querySelector(".ty").textContent = KINDS[classify(st)] + (st.isRegionSet ? " (region set)" : "") + (st.levels[0].zstd ? " +zstd" : "");
      tr.querySelector(".lv").textContent = lk !== null && lk !== undefined ? "CT L" + lk : "";
      tr.querySelector(".sz").textContent = st.levels[0].shape.join("×");
      tr.querySelector(".pi").textContent = g ? fmtUm(g.p0[0]) : "?";
      tr.querySelector(".nl").textContent = st.levels.length;
      tr.className = "ok";
      tr.title = "add " + e.url;
      tr.addEventListener("click", async () => {
        tr.querySelector(".ty").textContent = "adding…";
        const ly = e.rel === "volumes" && !S.layers.some((l) => l.role === "ct") ? await setCtLayer(e.url) : await addLayer(e.url);
        tr.querySelector(".ty").textContent = ly ? "added ✓" : "failed";
      });
    } catch (err) {
      tr.className = "bad";
      tr.querySelector(".ty").textContent = "not viewable";
      tr.title = String(err.message || err);
      tr.querySelector(".nm").textContent += " — " + String(err.message || err);
    }
  });
}

// Paste box: a mirror root, a sample directory, a volume (.zarr or one of its
// levels), any other store (added as a layer) or a region-set directory.
async function openPasted(raw) {
  let url = raw.trim();
  if (!url) return;
  if (!/^https?:\/\//i.test(url)) { msg("need an http(s) URL", true); return; }
  const m = /^(.*?\.zarr)(\/.*)?$/i.exec(url);
  if (m) {
    const z = m[1];
    if (/\/volumes\//.test(z) && !/\/representations\//.test(z)) await setCtLayer(z);
    else await addLayer(z);
    return;
  }
  url = url.replace(/\/+$/, "");
  const idx = await listIndex(url).catch(() => null);
  if (idx && idx.dirs.some((d) => /^region_\d+_\d+_\d+\.zarr$/.test(d))) { await addLayer(url); return; }
  const parts = url.split("/");
  const last = parts.pop();
  if (idx && idx.dirs.includes("volumes")) {
    S.base = parts.join("/") + "/";
    $("base-url").value = S.base;
    await loadSamples(last);
    await selectSample(last);
  } else {
    S.base = url + "/";
    $("base-url").value = S.base;
    await loadSamples();
  }
}

// ---------------------------------------------------------------- URL hash state
// #l=<store url>*<opts>&l=...&cu=z,y,x (µm)&z=<zoom per view>&vc=<centre per view>&sel=i
// opts: lut:<name>,op:<0..1>,w:<win>,lv:<level>,t:<thr|->,bl:<blend>,ip:<interp>,lvl:<auto|i>,h:<1 hidden>,k:<kind>,hue:<hue>,r:ct
// The old #vol=&pred=&c=<CT voxel>&level= form still opens.
function layerOpts(ly) {
  const o = [`lut:${ly.lut}`, `op:${+ly.opacity.toFixed(3)}`, `w:${ly.win}`, `lv:${ly.lev}`, `t:${ly.thrOn ? ly.thr : "-"}`,
    `bl:${ly.blend}`, `ip:${ly.interp}`, `lvl:${ly.level}`, `k:${ly.kind}`, `hue:${ly.hue}`];
  if (!ly.visible) o.push("h:1");
  if (ly.role) o.push("r:" + ly.role);
  return o.join(",");
}
function parseOpts(s) {
  const o = {};
  for (const kv of (s || "").split(",")) {
    const i = kv.indexOf(":");
    if (i < 0) continue;
    const k = kv.slice(0, i), v = kv.slice(i + 1);
    if (k === "lut") o.lut = v;
    else if (k === "op") o.opacity = +v;
    else if (k === "w") o.win = +v;
    else if (k === "lv") o.lev = +v;
    else if (k === "t") { o.thrOn = v !== "-"; if (v !== "-") o.thr = +v; }
    else if (k === "bl") o.blend = v;
    else if (k === "ip") o.interp = v;
    else if (k === "lvl") o.level = v;
    else if (k === "h") o.visible = v !== "1";
    else if (k === "k") o.kind = v;
    else if (k === "hue") o.hue = v;
    else if (k === "r") o.role = v;
  }
  return o;
}
function buildHash() {
  const p = new URLSearchParams();
  if (S.base !== DEFAULT_BASE) p.set("base", S.base);
  for (const ly of S.layers) p.append("l", ly.st.url + "*" + layerOpts(ly));
  if (S.layers.length) {
    p.set("cu", S.cursor.map((x) => +x.toFixed(3)).join(","));
    p.set("z", S.views.map((V) => +V.zoom.toPrecision(6)).join(","));
    p.set("vc", S.views.map((V) => V.center.map((x) => +x.toFixed(2)).join(":")).join(","));
    p.set("sel", S.sel);
  }
  if (S.smooth.on) p.set("smooth", "1");
  return "#" + p.toString();
}
let hashTimer = 0, hashLoading = false;
function saveHash() {
  if (hashLoading) return;
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => history.replaceState(null, "", buildHash()), 300);
}
async function loadHash() {
  hashLoading = true;
  try {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("base")) S.base = p.get("base");
    $("base-url").value = S.base;
    if (p.get("smooth") === "1") { S.smooth.on = true; $("smooth").checked = true; }
    const ls = p.getAll("l").map((s) => { const i = s.lastIndexOf("*"); return i < 0 ? { url: s, o: {} } : { url: s.slice(0, i), o: parseOpts(s.slice(i + 1)) }; });
    const vol = p.get("vol"), pred = p.get("pred");
    const first = ls.length ? ls[0].url : vol || pred || "";
    const m = /\/([^/]+)\/(volumes|representations)\//.exec(first);
    await loadSamples(m ? m[1] : "");
    if (ls.length) {
      // open every store first (in parallel), then build the stack in order
      const sts = await Promise.all(ls.map((e) => openStore(e.url).catch((err) => { msg("could not open " + e.url + ": " + err.message, true); return null; })));
      S.layers = [];
      ls.forEach((e, i) => { if (sts[i]) S.layers.push(newLayer(sts[i], e.o)); });
      relayout();
      S.sel = clamp(+(p.get("sel") || 0), 0, S.layers.length - 1);
      centreCursor();
      fitAll();
      const cu = (p.get("cu") || "").split(",").map(Number);
      if (cu.length === 3 && cu.every(Number.isFinite)) S.cursor = cu;
      const z = (p.get("z") || "").split(",").map(Number), vc = (p.get("vc") || "").split(",").map((s) => s.split(":").map(Number));
      S.views.forEach((V, i) => {
        if (Number.isFinite(z[i]) && z[i] > 0) V.zoom = z[i];
        if (vc[i] && vc[i].length === 2 && vc[i].every(Number.isFinite)) V.center = vc[i];
      });
    } else if (vol || pred) {
      // the old form: #vol=<CT>&pred=<overlay>&c=<CT level-0 voxel>&level=<i>
      const ct = vol ? await setCtLayer(vol, { keepView: false }) : null;
      if (ct && p.get("level")) { ct.level = p.get("level"); }
      if (ct && p.get("c")) {
        const c = p.get("c").split(",").map(Number);
        const L = ct.st.levels[0];
        if (c.length === 3 && c.every(Number.isFinite)) S.cursor = c.map((x, a) => L.origin[a] + (x + 0.5) * L.pitch[a]);
      }
      if (pred) await addLayer(pred, { keepView: !!ct });
      if (!ct) { centreCursor(); fitAll(); }
    }
    syncVolumeSelect();
    renderLayerList();
    renderLayerSettings();
    requestRender();
  } finally {
    hashLoading = false;
  }
  if (S.layers.length) saveHash(); // rewrite an old-form hash in the current form
}

// ---------------------------------------------------------------- controls
function setupControls() {
  $("base-url").value = S.base;
  $("base-go").onclick = () => { S.base = $("base-url").value.trim().replace(/\/?$/, "/"); loadSamples(); };
  $("sample").onchange = (e) => selectSample(e.target.value);
  $("volume").onchange = (e) => setCtLayer(e.target.value);
  $("add-layer").onclick = openAddPanel;
  $("add-close").onclick = closeAddPanel;
  $("add-refresh").onclick = () => { const r = sampleRootForAdd(); if (r) { repCache.delete(r); indexCache.clear(); fillAddList(r); } };
  $("add-url-go").onclick = async () => { const u = $("add-url").value.trim(); if (u) { const ly = await addLayer(u.replace(/^(.*?\.zarr)\/.*$/i, "$1")); if (ly) closeAddPanel(); } };
  $("add-url").addEventListener("keydown", (e) => { if (e.key === "Enter") $("add-url-go").click(); });
  $("addpanel").addEventListener("click", (e) => { if (e.target === $("addpanel")) closeAddPanel(); });
  $("paste-go").onclick = () => openPasted($("paste-url").value);
  $("paste-url").addEventListener("keydown", (e) => { if (e.key === "Enter") openPasted(e.target.value); });
  $("smooth").onchange = (e) => { S.smooth.on = e.target.checked; saveHash(); requestRender(); };
  $("smooth-pred").onchange = (e) => { S.smooth.pred = e.target.checked; requestRender(); };
  $("smooth-ct-s").onchange = (e) => { S.smooth.ct = clamp(+e.target.value || 2, 0.1, 4); requestRender(); };
  $("smooth-pr-s").onchange = (e) => { S.smooth.pr = clamp(+e.target.value || 0.6, 0.1, 4); requestRender(); };
  $("cache-cap").onchange = (e) => { cache.cap = +e.target.value * 1048576; applyCaps(); requestRender(); };
  $("clear-cache").onclick = () => { cache.clear(); ccache.clear(); if (loader) loader.reset(); stats.lastError = ""; msg(""); requestRender(); };
  $("xhair").onchange = (e) => { S.crosshair = e.target.checked; requestRender(); };
  $("outlines").onchange = (e) => { S.outlines = e.target.checked; requestRender(); };
  $("fit").onclick = fitAll;
  $("copy-link").onclick = async () => {
    history.replaceState(null, "", buildHash());
    try { await navigator.clipboard.writeText(location.href); msg("link copied"); } catch { msg("copy the address bar: the link is in the URL hash"); }
  };
  $("help-btn").onclick = () => $("help").classList.toggle("open");
  $("help-close").onclick = () => $("help").classList.remove("open");
  window.addEventListener("keydown", onKey);
  window.addEventListener("hashchange", () => { if (location.hash !== buildHash()) loadHash(); });
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
  renderLayerList();
  setInterval(updateStatus, 500);
  const wasm = b64bytes($("wasm-b64").textContent);
  const nWorkers = clamp(navigator.hardwareConcurrency || 4, 1, 16);
  pool = new Pool(wasm.buffer, $("worker-src").textContent, nWorkers);
  loader = new Loader(pool);
  try {
    const info = await pool.ready;
    $("info-wasm").textContent = `volcomp ${info.version} (wasm, ${info.kernels} kernels${info.zstd ? ", zstd" : ""}) · ${nWorkers} workers`;
  } catch (e) {
    msg("decoder failed to start: " + e.message, true);
    return;
  }
  await loadHash();
  window.vcViewer.ready = true;
}

// Test/debug hooks (web/test/browser_check.cjs).
async function sha256hex(u8) {
  if (!crypto.subtle) return null;
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", u8))].map((b) => b.toString(16).padStart(2, "0")).join("");
}
window.vcViewer = {
  S, stats, cache, addLayer, setCtLayer, relayout, requestRender, buildHash, loadHash,
  layerState: () => S.layers.map((l) => ({ url: l.st.url, opts: layerOpts(l), p0: l.p0, rule: l.rule, levels: l.st.levels.map((L) => ({ pitch: L.pitch, origin: L.origin, shape: L.shape })) })),
  // decode one inner chunk (global chunk coords) of a level URL and return its checksums
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
    const r = await pool.decode(enc, smooth, flags, L.zstd);
    return { offset: off, nbytes: nb, zstd: L.zstd, encCrc32c: encCrc, crc32c: crc32c(r.out), sha256: await sha256hex(r.out), ms: r.ms, mode: r.mode, q: r.q };
  },
  // Composite one tile exactly as a view does (every visible layer, current settings),
  // waiting until all its chunks are loaded. spec: {axis, s, u0, v0, du, w, h, levels?: [li per layer]}
  // (µm). Returns the sha256 of the RGBA bytes and the levels used.
  async renderTile(spec) {
    const force = new Map();
    if (spec.levels) S.layers.forEach((l, i) => { if (spec.levels[i] !== undefined && spec.levels[i] !== null) force.set(l.id, spec.levels[i]); });
    const owner = {};
    for (let tries = 0; tries < 900; tries++) {
      const wanted = [];
      const T = { axis: spec.axis, W: spec.w, H: spec.h, u0: spec.u0, v0: spec.v0, du: spec.du, s: spec.s, force, tile: true };
      const img = composite(T, wanted, -1, null, owner);
      if (!wanted.length) {
        S.extraWant = null;
        const rgba = new Uint8Array(img.data.buffer.slice(0));
        let lit = 0;
        for (let k = 0; k < rgba.length; k += 4) if (rgba[k] !== rgba[k + 1] || rgba[k] !== rgba[k + 2]) lit++;
        return { sha256: await sha256hex(rgba), levels: [...T.levels.values()], coloured: lit, tries,
          rgbaHead: Array.from(rgba.slice(0, 16)) };
      }
      S.extraWant = wanted;
      requestRender();
      await sleep(200);
    }
    S.extraWant = null;
    throw new Error("renderTile: chunks did not load");
  },
};

boot();
