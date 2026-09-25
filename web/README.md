# volcomp viewer (browser)

`dist/volcomp_viewer.html` is a single self-contained page (about 170 KB: HTML,
JS, the decode worker and the volcomp decoder compiled to WebAssembly, inlined
as base64). Double-click it: it works from `file://` and from any plain HTTP
server, and needs nothing installed. It streams volcomp-coded zarr v3 stores
straight from the mirror at
<https://dl.ash2txt.org/community-uploads/forrest/volcomp/> (or any server that
sends `Access-Control-Allow-Origin: *`, allows the `Range` header and answers
byte ranges with 206). It shows three orthogonal slices of a CT volume with a
prediction overlay.

## Use

1. Open `dist/volcomp_viewer.html` in a current Chrome/Edge/Firefox.
2. Pick a **sample**. Its volumes (`<sample>/volumes/*.zarr`) and predictions
   (`<sample>/representations/predictions/surfaces/*.zarr`) are read from
   the mirror's directory index. The first volume opens straight away.
3. Choose an **overlay** (any prediction store of the sample). It is drawn in
   red with alpha = p × opacity, or as a solid mask of p ≥ 0.5 when
   **mask p≥0.5** is ticked. `-prob` stores hold round(255 p). `-th0.2` stores
   are 0/255 masks.
4. **level** `auto` picks, per view, the coarsest CT level whose voxel is at
   most one screen pixel. Picking a level forces it in every view. The overlay
   level follows the CT level by physical pitch: the CT pitch comes from the
   volume name (`…-8.640um-…`), the prediction pitch from its µm OME scales.
   So the 8.86 µm upstream masks line up with level 2 of a 2.215 µm scan.
5. **deblock (smooth)** decodes with `volcomp_decode_smooth`, using the
   settings from `docs/deblocking.md`: CT gets the gated face filter at
   strength 2 × q (`SMOOTH_GATED | ZERO_GUARD`), and predictions get the
   Gaussian seam blur σ 0.6 (`ZERO_GUARD`), which keeps a surface chunk's
   threshold exact. Both strengths can be edited. Compressed chunks are cached,
   so toggling re-decodes without downloading again.
6. The paste box takes any of:
   - a mirror root
   - a sample directory
   - a volume `.zarr` (or one of its levels)
   - a prediction `.zarr` (it becomes the overlay when a volume is open)

   The URL hash records the volume, the overlay, the level and the cursor, so a
   copied link reopens the same place.

Mouse and keys (press `?` in the page for the list):

| input | action |
|---|---|
| wheel | step the slice (shift ×10) |
| ctrl/alt + wheel, pinch | zoom about the mouse |
| drag | pan |
| click | set the cursor, which moves the other two views' slices |
| right-drag / shift-drag | window / level |
| double-click, `F` | maximise one view |
| `W`/`S`, arrows, PgUp/PgDn | step the slice in the active view |
| `+`/`-` | zoom |
| `R` | fit |
| `1` `2` `3` | make the z / y / x view active |
| `,` `.` | coarser / finer CT level |
| `A` | auto level |
| `L` | auto window/level (histogram of the active view, air excluded) |
| `D` | deblock |
| `O` | overlay on/off |
| `T` | threshold mask |
| `[` `]` | overlay opacity |
| `C` | crosshair |

The minimap (bottom right) shows the active view's slice at a coarse level with
that view's viewport. Click or drag it to move the view. The status bar shows:

- bytes fetched and requests (index / range / 404)
- chunks decoded, with average and last decode time per chunk (in the worker)
- chunks that were air
- cache use
- the last render time
- the loader's queue

## How it reads a store

- The viewer reads the group `zarr.json` (OME multiscales: level paths and
  scales), then every level's `zarr.json` (shape, shard shape, inner chunk,
  codec). The inner codec must be exactly `volcomp`, the dtype uint8 and the
  inner chunk 128³. Shard shapes may differ per level: the prediction stores
  use 1024, 512, 256 and 128.
- For each shard it needs, the viewer makes one Range request for the trailing
  index (`bytes=-(N·16+4)`: N (offset, nbytes) u64 LE pairs in C order, then
  the crc32c, which is checked). Indices are cached per shard. A missing shard
  (404) and an index entry of all ones both mean all zero (air).
- The chunks a view needs from one shard are sorted by offset. Ranges whose gap
  is at most 64 KB are read together, in reads of at most 8 MB. Whole shards are
  never downloaded. A shard that holds a single inner chunk (the coarse
  prediction levels) is fetched with one plain GET, because that file is the
  chunk plus a 20-byte index.
- Decoding runs in a pool of `navigator.hardwareConcurrency` Web Workers, each
  with its own wasm instance. Decoded 128³ chunks go into an LRU keyed by
  (store, level, shard, inner chunk, smoothing), capped at 1.5 GB by default
  (512 MB to 2 GB can be chosen). Compressed bytes go into a second 256 MB LRU.
- Loading order: visible chunks first, nearest the view centre first, the
  active view first. Then the neighbouring chunk layers along each view's
  slice axis are prefetched, but only while visible plus prefetched chunks fit
  in 60 % of the cache.

## Build

```sh
web/build.sh     # emcc (from ~/emsdk if not on PATH) + python3
```

`build.sh` compiles `src/shim.c` (a thin export layer over `../volcomp.h`)
with `emcc -O3 -msimd128 -sSTANDALONE_WASM` into `build/volcomp.wasm`. The
module needs no JS glue: its only import is `emscripten_notify_memory_growth`.
`build.py` then inlines `src/worker.js`, the base64 wasm and `src/main.js` into
`src/index.html` and writes `dist/volcomp_viewer.html`. Install Emscripten
with:

```sh
git clone https://github.com/emscripten-core/emsdk ~/emsdk
~/emsdk/emsdk install latest && ~/emsdk/emsdk activate latest
```

Sources:

| file | contents |
|---|---|
| `src/main.js` | HTTP, stores, loader, cache, rendering, UI |
| `src/worker.js` | decode worker |
| `src/index.html` | layout and CSS |
| `src/shim.c` | wasm exports |

The wasm exports are:

- `vc_decode`
- `vc_decode_smooth(n, strength, flags)`
- `vc_decode_block`
- chunk info: `vc_chunk_mode`, `vc_stream_q`, `vc_is_lossless`,
  `vc_surface_thr`, `vc_mask_dim`
- `vc_in(n)` (input buffer) and `vc_out()` (the 128³ output)

## Tests / verification

- `node web/test/wasm_check.mjs web/build/volcomp.wasm chunk.volc…` decodes
  `.volc` files with the wasm build and prints sha256 prefixes of three
  decodes: plain, CT smooth, prediction smooth.
  `python3 web/test/ref_checksums.py files chunk.volc…` prints the same line
  from `python/volcomp_zarr` (libvolcomp).
- `python3 web/test/ref_checksums.py url LEVEL_URL CZ CY CX` fetches one chunk
  from the mirror (index + Range, like the viewer) and prints the reference
  sha256 values.
- `NODE_PATH=…/node_modules node web/test/browser_check.cjs [page-url] [out-dir]`
  needs `npm i playwright && npx playwright install chromium`. It drives the
  built page in headless Chromium. By default it opens the page from `file://`;
  pass `http://…/volcomp_viewer.html` to test it served. It checks:
  - streaming of the example volume and overlay
  - the sample, volume and overlay pickers
  - the 2.215 µm volume with the 8.86 µm mask overlay
  - wheel, zoom, pan, click, window/level and keys

  It also prints in-browser sha256 values of chunks, computed with
  `window.vcViewer.chunkChecksum(levelUrl, cz, cy, cx, smooth, flags)`, and
  writes screenshots.

Results (2026-09-25):

- The wasm decoder matches the native C kernels (`-DVOLCOMP_NO_AVX2`) bit for
  bit, in both the SIMD and the scalar build. Seven chunks were tested: CT q8
  and q1 from the mirror, the m7 prob store (q8), a mode-5 mask, a mode-6
  surface chunk, and the q32 and surface goldens. Each was checked for the
  plain, gated-smooth and Gaussian-smooth decodes.
- Compared with the AVX2 python build, the plain and gated decodes are
  identical. The prediction Gaussian smooth differs within the documented
  ±1 LSB kernel tolerance, as the native C build does too.
- In the browser, from `file://` and from `http://`, CT level-0 chunk
  (22,16,16) gave sha256 `9a060fa8…` plain and `480eb840…` smoothed, and prob
  chunk (20,16,20) gave `b4daeae7…` plain. These equal the python decoder's
  values for the same bytes fetched over HTTP.

## Limits

- Slices are nearest-neighbour: no interpolation and no oblique planes.
- Deblocking is per chunk: `volcomp_decode_smooth` leaves chunk faces as
  decoded. The region-level `volcomp_deblock` over assembled chunks is not
  applied.
- The prediction Gaussian smooth can differ by ±1 from an AVX2 native build
  (the C kernels agree exactly). Plain decodes and gated smoothing are
  identical across builds.
- The overlay is aligned by pitch only: both grids are assumed to start at the
  same origin. OME translations are ignored. When no pitch is known, the
  overlay's level 0 is assumed to be the CT's level-0 grid.
- Supported stores: uint8, 3-D, inner chunk 128³, inner codec `volcomp` only.
  The stores can be sharded (index at the end or the start) or unsharded.
- Listing samples, volumes and predictions needs the server's HTML
  autoindex. Without it the viewer falls back to a built-in list of the 39
  samples, and volumes have to be pasted as URLs.
- Memory: the decoded cache holds 2 MB per chunk, up to the chosen cap, plus
  browser overhead. A view that needs more than cache/6 chunks (a fine level
  zoomed far out) loads only the chunks nearest its centre and says so in the
  status bar. Use `auto` or a coarser level.
- Missing shards are real 404s, so the browser console logs them. They are
  expected and mean air.
- Tested in headless Chromium (Playwright) only. Firefox and Safari are
  untested.
