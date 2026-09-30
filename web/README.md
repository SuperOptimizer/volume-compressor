# volcomp viewer (browser)

`dist/volcomp_viewer.html` is a single self-contained page (about 300 KB: HTML,
JS, the decode worker and the volcomp decoder plus a zstd decoder compiled to
WebAssembly, inlined as base64). Double-click it: it works from `file://` and
from any plain HTTP server, and needs nothing installed. It streams volcomp-coded
zarr v3 stores straight from the mirror at
<https://dl.ash2txt.org/community-uploads/forrest/volcomp/> (or any server that
sends `Access-Control-Allow-Origin: *`, allows the `Range` header and answers
byte ranges with 206). It shows three orthogonal slices of a stack of layers:
the CT, and any number of volumes from `representations/` (m7 surface
probabilities and masks, teacher-region sets, ink or label volumes when they
appear), each alone or overlaid.

## Use

1. Open `dist/volcomp_viewer.html` in a current Chrome/Edge/Firefox.
2. Pick a **sample**. Its volumes (`<sample>/volumes/*.zarr`) are read from
   the mirror's directory index and the first one opens as the **CT layer**.
   The **volume** picker replaces the CT layer; layers of other scans are then
   removed, because different scans are not co-registered.
3. **+ layer** (or `I`) opens the add-layer panel. It lists the sample's
   volumes and everything under `<sample>/representations/`, found by walking
   the autoindex recursively: every `*.zarr`, and every directory that holds
   more than 20 `region_Z_Y_X.zarr` stores as one **region set**. For each
   store it reads the metadata and shows the kind (the path under
   `representations/`), the type (CT / probability / mask / labels, region
   set, `+zstd`), the scan, the CT level it was made from (`CT L2` for an
   `-L2` store), the level-0 size, the pitch and the number of levels. Stores
   the viewer cannot read (not uint8, not 3-D, not volcomp) are greyed with
   the reason; files that are not zarr (checkpoints, JSON) are skipped. Click
   a row to add that store on top. The URL box adds any store URL (a `.zarr`,
   a level of one, or a region-set directory).
4. The **Layers** panel (bottom right) lists the stack, top row drawn on top.
   Each row has a visibility box, the colour, the kind, the level used in the
   active view and the blend mode, and buttons to move the layer up or down or
   remove it. Any layer can be the bottom one, and a prediction can be viewed
   alone with no CT. Click a row to select it: its settings appear below
   (see "Layer settings"), and the keys and right-drag act on it.
5. **deblock (smooth)** decodes with `volcomp_decode_smooth`, with the
   settings from `docs/deblocking.md`. Grey (CT) layers get the gated face
   filter at strength 2 × q (`SMOOTH_GATED | ZERO_GUARD`). Probability layers
   get the Gaussian seam blur σ 0.6 (`ZERO_GUARD`), which keeps a surface
   chunk's threshold exact. Masks and labels are never smoothed. Both
   strengths can be edited. Compressed chunks are cached, so toggling
   re-decodes without downloading again.
6. The paste box in the header takes a mirror root, a sample directory, a
   volume `.zarr` (it becomes the CT layer), or any other store or region-set
   directory (it is added as a layer).
7. The URL hash records the layers in order with all their settings, the
   cursor, each view's zoom and centre, the selected layer and the deblock
   switch. **link** copies it. A copied link reopens the same view. The old
   `#vol=…&pred=…&c=…&level=…` links still open (as a CT layer plus an
   overlay, the cursor converted from CT voxels).

### Layer settings

| setting | values | default |
|---|---|---|
| kind | CT, probability, mask, labels | from the store (below) |
| colour | `grey`; single-hue ramps `red` `green` `blue` `cyan` `magenta` `yellow` `orange`; `fire`; `categorical` | CT: grey; probability: the next free hue; mask/labels: categorical |
| blend | normal (alpha over), additive, max | normal |
| opacity | 0–100 % | CT 100, others 70 |
| window / level | 1–255 / 0–255 (right-drag, `L` for auto from the view's histogram) | 255 / 128 |
| threshold | off, or draw only values ≥ t (opaque) | off, 128 |
| sampling | auto, trilinear, nearest | auto: nearest for masks, labels and `categorical`, trilinear otherwise |
| level | auto, or a fixed pyramid level | auto |

The kind comes from the store. Stores under `volumes/` are CT. A `-th<x>`
name, a mask encoding, a `majority` pyramid or a `threshold` attribute make a
mask. A label/instance name or encoding makes labels. Anything else under
`representations/` is a probability.

The colour maps turn a value v into a colour c and an alpha a. The window gives
w(v) = clamp(floor((v − lo) / win · 255 + 0.5)), with lo = level − win/2.

- grey: c = (w, w, w), a = 255
- hue ramps: c = the hue, a = w(v), so with the default window a = v and a
  probability p is drawn with alpha p
- fire: c = black → red → yellow → white over w(v), a = w(v)
- categorical: a fixed 255-colour palette (golden-angle hues), a = 255 for
  v > 0. v = 255 takes the layer's hue, so a 0/255 mask is one solid colour.
- threshold: a = 255 for v ≥ t, 0 below

Blending is done in integers, bottom to top, onto black. With A = a · round(255
· opacity) in 0…65025:

- normal: d = floor((d · (65025 − A) + c · A + 32512) / 65025)
- additive: d = min(255, d + floor((c · A + 32512) / 65025))
- max: d = max(d, floor((c · A + 32512) / 65025))

Pixels whose chunks are still loading show the "not loaded" tint where the
bottom visible layer is missing. Upper layers are skipped there until they
arrive.

## Geometry and resampling

The views share one world frame in micrometres (z, y, x). Every level of every
layer has its own pitch (µm per voxel, per axis) and origin (the µm position of
the corner of voxel 0). Voxel i covers [origin + i·pitch, origin + (i+1)·pitch)
and its centre is origin + (i + ½)·pitch. The views, the cursor, the slice
steps and the URL hash all work in µm.

**Level-0 pitch**, the first rule that applies:

1. The store's `volcomp` attributes have `source_volume` and `source_level` k
   (every m7 `-L<k>` store on the mirror). Then pitch = the CT pitch from the
   source volume's name × 2^k. **Level 0 of an m7 `-L<k>` prediction is CT
   level k**: for the 2.215 µm PHerc0343P scan, `-L2` gives 8.86 µm, the same
   grid as CT level 2 (5179 × 3289 × 3289). Because the factor is a power of
   two and the origins are 0, CT level k and the prediction's level 0 have
   bit-identical pitches.
2. OME `multiscales` scales with a length unit (micrometer; nanometer and
   millimeter are converted).
3. `volcomp.levels[0].voxel_size_um`, or a `voxel_um` attribute (the
   teacher-region stores).
4. The store name (`…-2.215um-…`, the CT naming).
5. A `-L<k>-` name, with a CT layer of the same scan in the stack: that CT
   layer's level-k pitch.
6. Otherwise the level-0 grid of the lowest layer that has a pitch (1 unit
   per voxel if none does). The layer's settings say so in orange.

When rule 1, 3 or 4 is used and the store also has OME µm scales that differ
by more than 0.1 %, the settings show both. All 91 stores on the mirror agree
(checked 2026-09-30).

**Other levels**: pitch_i = pitch_0 × scale_i / scale_0 (the OME scales).

**Origins**: 0 for stores without OME translations. Pooled pyramids share the
corner of voxel 0, so level k voxel i covers CT level-0 voxels [i·2^k,
(i+1)·2^k). When OME translations are present, t is the centre of voxel 0, so
origin = t − pitch/2. An array store with `origin_zyx` (a single region) starts
at origin_zyx × pitch. A region set places each `region_Z_Y_X.zarr` at voxel
(Z, Y, X).

**Level choice**: this is done per view and per layer, independently. A layer
on `auto` uses the coarsest level whose pitch along the view's horizontal axis
is at most one screen pixel (µm per device pixel). When the view is zoomed in
past level 0, it uses level 0. So at 1 screen pixel per 2.215 µm CT voxel, the
CT uses level 0 and the m7 `-L2` prediction uses its level 0 (8.86 µm). A layer
is not loaded in a view (the label says "zoom in") in two cases: when even its
coarsest level is more than 16× finer than a pixel, or when the voxels in view
would exceed 16 per screen pixel. This happens with single-level stores such as
region sets.

**Resampling**: each screen pixel centre is a world point w. For each layer it
becomes a continuous level coordinate c = (w − origin) / pitch on each axis.

- nearest (masks, labels): voxel floor(c).
- trilinear (CT, probabilities): t = c − ½, a = floor(t), b = a + 1, and an
  8-bit weight f = floor((t − a) · 256 + ½) for b. f = 256 rounds to a + 1,
  f = 0. a and b are clamped to the array, and a == b gives f = 0. The value
  is the integer trilinear sum of the 8 voxels with weights (256 − f, f) per
  axis: floor((Σ + 2^23) / 2^24).
- A point outside [0, n) on any axis is outside the layer, which is
  transparent there.

The slice axis is interpolated the same way, so a prediction at 8.86 µm
between CT slices blends its two nearest slices. All of this is integer or
exact float arithmetic, and `test/ref_composite.py` reproduces it bit for bit.

Rendering has two stages per layer and view. First, the level voxels in view
are gathered into a plane, chunk by chunk, already interpolated along the slice
axis. Then each pixel is a bilinear (or nearest) lookup in that plane. A view
larger than 1.5 Mpx (hi-dpi, maximised) is composited at reduced resolution and
scaled up. At 1600 × 1000 CSS px, three views, the minimap and three layers
take about 60 ms per frame.

## Mouse and keys

Press `?` in the page for the list.

| input | action |
|---|---|
| wheel | step the slice by one voxel of the finest visible layer (shift ×10) |
| ctrl/alt + wheel, pinch | zoom about the mouse |
| drag | pan |
| click | set the cursor, which moves the other two views' slices |
| right-drag / shift-drag | window / level of the selected layer |
| double-click, `F` | maximise one view |
| `W`/`S`, arrows, PgUp/PgDn | step the slice in the active view |
| `+`/`-` | zoom |
| `R` | fit |
| `1` `2` `3` | make the z / y / x view active |
| `N` / shift+`N` | select the next / previous layer |
| `,` `.` | coarser / finer level of the selected layer |
| `A` | the selected layer back to auto level |
| `L` | auto window/level of the selected layer (ignores 0) |
| `V` | show / hide the selected layer |
| `O` | all layers above the bottom one on/off |
| `T` | threshold of the selected layer on/off |
| `[` `]` | opacity of the selected layer −/+ |
| `I`, Insert | add a layer |
| `D` | deblock |
| `C` | crosshair |

The minimap (bottom right) shows the active view's slice through the whole
stack at coarse levels, with that view's viewport. Click or drag it to move
the view. Each layer's extent is outlined in its colour (**outlines** turns
this off). The side panel shows the value of every layer under the mouse, with
the level and voxel index it came from. The status bar shows:

- bytes fetched and requests (index / range / 404)
- chunks decoded, with average and last decode time per chunk (in the worker)
- chunks that were air
- cache use and the per-layer share
- the last render time
- the loader's queue

### URL hash

`#l=<store>*<opts>&l=…&cu=z,y,x&z=<zoom per view>&vc=<u:v per view>&sel=<i>&smooth=1`

- `l`: one per layer, bottom first (URL-encoded)
- `cu`: the cursor in µm
- `z`: each view's zoom in px/µm
- `opts` is a comma list:
  - `lut:<map>`, `op:<0..1>`, `w:<window>`, `lv:<level>`, `t:<threshold|->`
  - `bl:<normal|add|max>`, `ip:<auto|linear|nearest>`, `lvl:<auto|i>`
  - `k:<ct|prob|mask|label>`, `hue:<name>`, `h:1` (hidden), `r:ct` (the
    CT layer)

A bare `l=<store>` takes the defaults.

## How it reads a store

- The viewer reads the group `zarr.json` (OME multiscales: level paths and
  scales), then every level's `zarr.json` (shape, shard shape, inner chunk,
  codec). The inner codecs must be `volcomp`, optionally followed by `zstd`
  (the teacher-region stores), the dtype uint8 and the inner chunk 128³.
  Shard shapes may differ per level: the prediction stores use 1024, 512, 256
  and 128.
- A region set is a directory of `region_Z_Y_X.zarr` array stores whose
  origins lie on a grid of the region edge (1024). It is read as one sparse
  single-level array. A region that is not in the directory listing is air and
  is never requested. Regions may differ in layout (on the mirror some hold one
  1024³ shard, others are unsharded 128³ chunks), so each region's `zarr.json`
  is read the first time one of its chunks is needed.
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
  with its own wasm instance. A `zstd` frame is undone in the same worker
  (zstd 1.5.7's decoder, compiled into the module) before the volcomp decode.
  Decoded 128³ chunks go into one LRU keyed by (store, level, shard, inner
  chunk, smoothing), capped at 3 GB by default (1 GB to 4 GB can be chosen).
  Each store in the stack gets an equal share of that total as its own cap, so
  one layer never evicts another layer's chunks. Compressed bytes go into a
  second 256 MB LRU.
- **Chunks on screen are never evicted.** Every frame pins the chunks it
  draws, and eviction skips them. A chunk evicted while visible
  would be decoded again the next frame, and so on for ever: that loop is
  what 1.5 GB shared by three layers did before this change. Instead, each
  layer has a budget per view: 70 % of its share, split between the views on
  screen. An `auto` layer whose level would need more chunks than that is
  drawn one or more levels coarser. The choice depends only on the view and
  the cache size, not on what has loaded, so it happens once per view change.
  The status bar then names the layer, the level used, the level it would
  have used, and its share. A layer that is fixed to a level, or that is
  already at its coarsest level, loads only the budget's worth of chunks
  nearest the centre.
- Loading order: visible chunks first, nearest the view centre first, the
  active view first, lower layers slightly before upper ones. Then the
  neighbouring chunk layers along each view's slice axis are prefetched, but
  only into what is left of each layer's share (90 %) after its chunks on
  screen.

## Build

```sh
web/build.sh     # emcc (from ~/emsdk if not on PATH) + python3
```

`build.sh` compiles `src/shim.c` (a thin export layer over `../volcomp.h`)
and zstd's decoder with `emcc -O3 -msimd128 -sSTANDALONE_WASM` into
`build/volcomp.wasm`. The zstd release tarball (1.5.7) is downloaded into
`build/` once and checked against its sha256. The module needs no JS glue: its
only import is `emscripten_notify_memory_growth`. The zstd decoder adds about
64 KB of wasm. Decodes are identical to the previous (volcomp-only) build on
the golden chunks.
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
| `src/main.js` | HTTP, stores, geometry, loader, cache, resampling, compositing, UI |
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
- `vc_unzstd(n)`: decompress a zstd frame in the input buffer in place
  (returns the new size); `vc_has_zstd()`

## Tests / verification

- `node web/test/wasm_check.mjs web/build/volcomp.wasm chunk.volc…` decodes
  `.volc` files with the wasm build and prints sha256 prefixes of three
  decodes: plain, CT smooth, prediction smooth.
  `python3 web/test/ref_checksums.py files chunk.volc…` prints the same line
  from `python/volcomp_zarr` (libvolcomp).
- `python3 web/test/ref_checksums.py url LEVEL_URL CZ CY CX` fetches one chunk
  from the mirror (index + Range, like the viewer) and prints the reference
  sha256 values.
- `python3 web/test/ref_composite.py spec.json` is the reference compositor.
  It fetches the chunks of a tile, decodes them with libvolcomp, derives each
  layer's pitch itself from the store metadata (rules 1, 2 and 4 above),
  resamples and blends in the same integer arithmetic, and prints the sha256
  of the RGBA bytes.
- `web/test/browser_check.cjs [page-url] [out-dir]` drives the built page in
  headless Chromium. By default it opens the page from `file://`; pass
  `http://…/volcomp_viewer.html` to test it served. It prints PASS/FAIL lines,
  writes screenshots to out-dir, and exits non-zero on any FAIL. It checks:
  1. The old `#vol=&pred=&c=` hash (8.64 µm CT + m7 L0). It also checks chunk
     checksums against `ref_checksums.py`.
  2. CT 2.215 µm + m7 `-L2` probabilities. Geometry: the prediction's level 0
     equals CT level 2 in pitch, shape and origin. At 1 px per CT voxel the z
     view uses CT L0 with prediction L0. Three composited tiles are compared
     with `ref_composite.py`: a z view at 1:1, an x view at 3.1 µm/px, and a y
     view with CT level 1.
  3. The prediction alone, with no CT, and its tile. Then a dpr-2 page with a
     maximised view, which is composited at reduced resolution.
  4. Two overlays over the CT: probabilities (additive) and the th0.2 mask
     (categorical, nearest). Two tiles are compared with `ref_composite.py`.
  5. A URL-hash round trip. Settings are changed through the panel (colour
     map, blend, threshold, fixed level, reorder, hide). The page is reopened
     from its hash, and the layers, order, settings, cursor, zooms and
     selection are compared.
  6. The pickers: the sample list, then the volume list, then the add-layer
     panel. The panel listing must show both volumes and the three prediction
     stores with kind, CT level, size and pitch. Clicking a row adds the store.
  7. Mouse and keys: wheel, ctrl+wheel, drag, click, right-drag window,
     `,` `F` `D` `N` `V`.
  8. A teacher region set (volcomp + zstd). A region chunk's checksum must
     equal python (`zstd -dc` + libvolcomp), and the set must render over the
     2.4 µm PHercParis4 CT.

  To set it up without a system node (nothing is committed; `web/.tools/` is
  ignored):

  ```sh
  curl -so- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | PROFILE=/dev/null bash
  . ~/.nvm/nvm.sh && nvm install --lts
  mkdir -p web/.tools && cd web/.tools && echo '{"name":"vc-tools","private":true}' > package.json
  npm i playwright && npx playwright install chromium && cd ../..
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
    env NODE_PATH=$PWD/web/.tools/node_modules node web/test/browser_check.cjs "" /tmp/vc-check
  ```

  It needs python3 with `python/volcomp_zarr` built, and the `zstd` CLI.
  `window.vcViewer` exposes the hooks it uses:
  - `chunkChecksum(levelUrl, cz, cy, cx, smooth, flags)`
  - `renderTile({axis, s, u0, v0, du, w, h, levels})`: composite a tile the
    way a view does, and wait for its chunks
  - `layerState()`
  - `buildHash()`

  9. Stability, with three layers whose visible sets together exceed the
     cache:
     - PHercParis4 CT + m7 L2 probabilities + th0.2 mask at the default view
       (1920 × 1080 at DPR 1.25, default cache), then after zooming to 1:1 and
       stepping 20 slices
     - PHerc0343P at 0.3 px/µm with a 1 GB cache
     Each must see 0 requests, 0 decodes and 0 redraws in 10 s of idle after
     settling. The cache-share warning must show, and a zarr v2 store must be
     rejected with a message. The phase also reports the renderer's memory.

Results (2026-09-30): 42 PASS, 0 FAIL, in 176 s from `file://`, under a 6 GB
memory cap.

- Browser tiles equal `ref_composite.py` bit for bit (sha256 prefixes):

  | tile | sha256 |
  |---|---|
  | CT + m7 L2, z 1:1 | `9bef57dd…` |
  | CT + m7 L2, x 3.1 µm/px | `9dcae66f…` |
  | CT + m7 L2, y with CT L1 | `042a1a4f…` |
  | m7 L2 alone | `c8d7f454…` |
  | two overlays, z | `c5df511c…` |
  | two overlays, x | `2deb75a3…` |

- The chunk checksums are unchanged from 2026-09-25: `9a060fa8…`,
  `480eb840…` and `b4daeae7…`.
- A zstd-wrapped teacher-region chunk decodes to `439f417c…`, as in python.
- Memory at three layers (Paris4 CT + m7 L2 prob + th0.2, 3 GB cap,
  1920 × 1080 at DPR 1.25):
  - after settling: decoded cache 1.15 GB, renderer RSS 1.9 GB
  - after zooming to 1:1 and 20 slices: 1.6 GB of cache and 2.3 GB of RSS
  - with the cache full, expect about 3 GB of cache plus about 0.7 GB of
    overhead (16 workers, the page)
- Render time at that size is 110–180 ms per frame: three views of about
  1200 × 640 device px, three layers, plus the minimap.

Earlier results (2026-09-25):

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

- Axis-aligned slices only: no oblique planes and no volume rendering.
- Trilinear weights are quantised to 1/256 of a voxel per axis, which is well
  below what the screen shows. They are quantised so that a reference can
  reproduce the image bit for bit.
- Deblocking is per chunk: `volcomp_decode_smooth` leaves chunk faces as
  decoded. The region-level `volcomp_deblock` over assembled chunks is not
  applied.
- The prediction Gaussian smooth can differ by ±1 from an AVX2 native build
  (the C kernels agree exactly). Plain decodes and gated smoothing are
  identical across builds.
- Geometry is pitch plus origin per axis. Rotations, affine transforms and
  non-z/y/x axis orders are not read. OME translations are honoured but no
  store on the mirror has any.
- Different scans of one sample are not co-registered. The volume picker drops
  layers of other scans, but layers added by hand are placed in the same µm
  frame with no registration.
- Supported stores: uint8, 3-D, inner chunk 128³, inner codecs `volcomp` or
  `volcomp` then `zstd`. The stores can be sharded (index at the end or the
  start) or unsharded. Multi-channel stores, other dtypes and `volcomp_label`
  multi-class chunks are listed as not viewable.
- zarr v2 stores (such as the published blosc/zstd 192³ OME-zarr masks on
  S3) are rejected when added. The message names the store's format and
  points to its volcomp copy on the mirror.
- A region set is single-level. It is drawn only once a view is zoomed in to
  at most 16 voxels per screen pixel along the view's horizontal axis (38 µm
  per pixel for the 2.4 µm teacher regions).
- Listing samples, volumes and representations needs the server's HTML
  autoindex. Without it the viewer falls back to a built-in list of the 39
  samples, and stores have to be pasted as URLs. The representations walk
  stops after 120 directories and 5 levels.
- Memory: the decoded cache holds 2 MB per chunk, up to the chosen total
  (3 GB by default), split equally between the layers' stores. Browser
  overhead comes on top, about 0.7 GB. On machines with little RAM, pick a
  smaller cache. The views then use coarser levels (the status bar says which
  layer), and they never thrash.
- Rendering is on the main thread: 110–180 ms per frame for three layers at
  1920 × 1080 (DPR 1.25). Views over 1.5 Mpx are composited at reduced
  resolution.
- Missing shards are real 404s, so the browser console logs them. They are
  expected and mean air.
- Tested in headless Chromium (Playwright) only. Firefox and Safari are
  untested.
