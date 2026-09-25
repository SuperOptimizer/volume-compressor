# volcomp

Single-header compressor for `uint8` volumetric data (micro-CT), built for the
Vesuvius Challenge scrolls. One knob (`q`), one format, no state, no threads,
no dependencies beyond libc/libm.

`q` picks the mode:

| `q` | mode | error | use it for |
|---|---|---|---|
| **0** (`VOLCOMP_Q_LOSSLESS`) | predictive, exact | none — decoded bytes are the source bytes | class maps, masks, valid/occupancy flags, anything whose values *are* the data (a wrong label is not a "small" error), and archival copies |
| **1..255** | 3-D DCT + dead-zone quantiser | P99 ≈ 2.5 q on scroll data | greyscale CT and model probability maps, where 20–200× at a controlled error is the point |

and there are two further modes for **binary masks** (below):
`volcomp_mask_encode`, where half the spatial resolution costs 30× less than
storing the mask exactly, and `volcomp_mask_encode_lossless`, which stores the
mask itself, voxel for voxel, for 4× that. A **surface** mode,
`volcomp_surface_encode`, is for probability maps that consumers threshold: the
DCT codec plus a refinement that keeps the 0.5 threshold exact for every voxel
(below and [`docs/surface_mode.md`](docs/surface_mode.md)).

Both modes share the geometry, the entropy coder, the API and the stream
header, so a reader does not need to know which one produced a chunk;
`volcomp_stream_q()` tells it, and `volcomp_decode_block()` works either way.
Rule of thumb: lossless gets 60–250× on label and mask volumes and 3–4× on a
smooth u8 field; lossy gets 20–200× on greyscale CT. Lossless never exceeds
the raw chunk plus 8 bytes, so it is always safe to reach for.

- **block** 16³ (transform + random access), **chunk** 128³ (independently
  decodable, = one zarr chunk), **shard** 1024³ (zarr v3 `sharding_indexed`).
- Lossy (`q >= 1`): 3-D DCT-16, dead-zone quantiser, 2-lane tANS with per-chunk tables; no
  in-format post-filter (blocks and chunks reconstruct independently, no
  seams). Optional `volcomp_deblock()` for callers who assemble a region and
  want block edges smoothed.
- Lossless (`q = 0`): per-block prediction one step back along z (y, then x, at
  the block edges), zigzag residuals, and per block a choice of CONST (a flat
  block costs a token and a byte), SPARSE (run/level), DENSE (a token per
  voxel) or RAW; same 2-lane tANS. An all-flat chunk is 9 bytes.
- Mask (`volcomp_mask_encode`): a 128³ binary mask, 2×2×2 majority-pooled to a
  64³ grid, stored exactly with an adaptive binary range coder over the 12
  causal neighbours (4096 contexts), and decoded **trilinearly interpolated**
  back to 128³. **0.0057 bits per voxel** on the PHercParis4 recto surface
  prediction — 11× smaller than packbits + zstd-19 of the same mask, 350 000× the
  raw chunk.
- Mask, spatially lossless (`volcomp_mask_encode_lossless`): the same coder and
  the same context model over the **full 128³** mask, no pool and no
  interpolation, decoding to 0/255 voxel for voxel. **0.0255 bits per voxel** on
  the same data — 4.5× mode 4, still 2.4× smaller than packbits + zstd-19.
- Format: [`spec/format.md`](spec/format.md). Header byte 5 records the mode,
  and volcomp 1.0 rejects a nonzero value there, so old readers cannot
  misread a lossless chunk; `q >= 1` streams are byte-identical to 1.0's.
- Decoding a 16³ block touches one substream (≤ 16 blocks) and is
  bit-identical to the same region of the full decode.

## Use

```c
#include "volcomp.h"

uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
size_t n;
volcomp_status st = volcomp_encode(src /* 128^3 z-major */, 8.0f, enc, VOLCOMP_ENCODE_BOUND, &n);
st = volcomp_encode(src, VOLCOMP_Q_LOSSLESS, enc, VOLCOMP_ENCODE_BOUND, &n);  /* exact */
st = volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);   /* same call for either mode */
float q; st = volcomp_stream_q(enc, n, &q);               /* 0 => the stream is exact */
st = volcomp_decode_block(enc, n, bz, by, bx, block, VOLCOMP_BLOCK_VOXELS);
volcomp_deblock(volume, nz, ny, nx, 8.0f);   /* optional, after assembling decoded chunks */
```

## Binary masks

```c
st = volcomp_mask_encode(mask /* 128^3, nonzero = set */, enc, VOLCOMP_ENCODE_BOUND, &n);
st = volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);   /* the same call as any other chunk */
st = volcomp_mask_info(enc, n, &dim);                     /* OK iff a mask chunk; dim = 64 or 128 */
st = volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_MASK_VOXELS);  /* the 64^3 grid itself */

st = volcomp_mask_encode_lossless(mask, enc, VOLCOMP_ENCODE_BOUND, &n);  /* exact: dim = 128 */
st = volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);   /* 0/255, voxel for voxel */
```

A mask chunk stores the 2×2×2 **majority** pool of the mask — a 64³ grid — and
nothing else; there is no knob. `volcomp_decode` returns that grid trilinearly
interpolated back to 128³ as `u8` 0..255, so a stored block centre comes back 0
or 255 exactly and the boundary carries a two-voxel ramp: a continuous training
target, not a staircase, from 1/8 of the bits a full-resolution mask would cost.
Every existing reader — `volcomp_decode_block`, `python/volcomp_zarr`, the shard
readers — works unchanged. `volcomp_is_lossless` is false for a mask chunk, and
a mask chunk is never larger than 32 777 bytes. The stored grid of one rung is
exactly the mask of the next coarser rung, so a mask pyramid is built by reading
grids, not by pooling pixels.

`volcomp_mask_encode_lossless` is the same thing without the pool: the whole
128³ mask coded with the same 12-neighbour model, so `volcomp_decode` gives back
0 and 255 exactly where the source was zero and nonzero. It costs 4.5× mode 4
(0.0255 bits per voxel on the recto) and is never larger than
`VOLCOMP_MASK_LL_BOUND` = 262 153 bytes. `volcomp_mask_info` reports a stored
edge of 128 for it and 64 for mode 4, which is how a reader tells them apart;
`volcomp_is_lossless` is false for both (a nonzero source voxel comes back 255,
not its own value).

Everything is `static`; include the header in the translation unit that uses
it. No arch flags are needed: on x86-64 the AVX2+FMA kernels are compiled with
a target attribute and chosen at runtime when the CPU has them, and every
target has plain C kernels (arm64, x86-64 without AVX2, or `-DVOLCOMP_NO_AVX2`
to force them). `volcomp_kernels()` reports `"avx2"` or `"c"`. The two agree to
±1 LSB on decode; with clang and GCC on x86-64 they produce identical bytes.
The C kernels run at 40–60 % of the AVX2 speed (see BENCHMARKS.md).
`q` is 0 (lossless) or the quantiser step in voxel units, 1..255 (typical
2..32); error percentiles scale with q (P99 ≈ 2.5q on scroll data). The
lossless path is plain C on every target — it has no float math, so its
streams are identical on every build. Define `VOLCOMP_MALLOC` / `VOLCOMP_FREE`
to override the one scratch allocation in `volcomp_encode`.

## Surface predictions (mode 6)

```c
st = volcomp_surface_encode(prob /* 128^3, u8 = 255 p */, 64.0f /* q */, 128 /* thr */,
                            enc, VOLCOMP_SURFACE_ENCODE_BOUND, &n);
st = volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);   /* the same call as any other chunk */
st = volcomp_surface_info(enc, n, &thr, &margin);         /* OK iff a surface chunk */
```

For smooth probability volumes: the recto and m7 surface-teacher predictions.
The chunk has two parts:

- A DCT base with a **flat** step law, where every AC step is q. It is
  reconstructed bit-exactly on every build by strict IEEE kernels, with no FMA
  and the same result from AVX2 and C.
- One context-coded bit for each voxel near the threshold, saying whether the
  base put it on the wrong side.

`(source >= thr) == (decoded >= thr)` holds for every voxel. The thresholded
mask, its skeleton and its distance field are therefore exactly the source's;
only the soft values carry the DCT error.

Measured on 201 M voxels of fresh float predictions per teacher from PHercParis4:

- **Surface q64** is 0.78× (recto) and 0.71× (m7) the size of plain q8. Its mask
  is exact, where q8 flips 0.66 % / 0.18 % of voxels and has a skeleton F1 of
  0.52 / 0.64. Its band MAE is 3.5/255, against 2.2-2.6 at q8.
- **Surface q32** matches q8's soft error at 1.0-1.17× q8's size.
- **Decode** runs at 0.96-1.5 GB/s single-thread, against 3.5-4.2 GB/s for q8.

q is the flat-law step, from 2 to 255; surface q48 is about the size of mode-0
q16. Details and the full candidate study: [`docs/surface_mode.md`](docs/surface_mode.md).
Format: [`spec/format.md`](spec/format.md) §13. A revision-3 reader refuses these
chunks cleanly.

## Decode-side deblocking (optional, any existing stream)

```c
volcomp_decode_smooth(enc, n, dst, VOLCOMP_CHUNK_VOXELS, 2.0f,
                      VOLCOMP_SMOOTH_GATED | VOLCOMP_DEBLOCK_ZERO_GUARD);   /* CT, q >= 4 */
volcomp_decode_smooth(enc, n, dst, VOLCOMP_CHUNK_VOXELS, 0.6f, VOLCOMP_DEBLOCK_ZERO_GUARD);  /* probability maps */
```

How it works:

1. Smooth the chunk's interior block faces.
2. Project every block back onto the quantisation cells its coefficients were
   decoded from.

The result is still a reconstruction the stream allows. Exact zeros (masked air)
stay 0 with the guard, and a surface chunk keeps its exact threshold.

- **Masked CT, q ≥ 4:** +0.1 to +0.6 dB PSNR, with no chunk worse than no filter.
  Cost: about 5 ms per chunk.
- **m7 q16:** the iso-surface shift drops from 0.095 to 0.075 voxel.

`volcomp_deblock` (region-level) now skips any face position that touches an
exact zero. The old filter blurred masked CT's chunk-aligned air edges at q ≥ 16.

The stream and `volcomp_decode` are unchanged. Readers opt in:
`volcomp_zarr.set_read_smoothing(2)`, or `volcomp decode --smooth`. Numbers and
the encoder-side alternatives that were rejected:
[`docs/deblocking.md`](docs/deblocking.md).

## Label volumes (`volcomp_label.h`)

Companion header for multi-class probability arrays: a 128³ label chunk holds
up to 255 class planes (u8 values 0..255 — model probabilities, or a class map
where the values are exact labels), stored as one zarr array of shape
`(255, Z, Y, X)` with only the present classes taking space. Class identity is an explicit tag per plane and is never lossy:
a decoded chunk cannot carry a class absent from the source, and two classes
cannot be confused. Each stored plane is a `volcomp.h` stream at **its own q**, so the
error statistics per plane are exactly volcomp's at that q — one chunk can
hold a lossless class map (`q = 0`) beside lossy probability planes
(`q = 8`). Thresholded or downscaled label maps are derived by the consumer
from the decoded probabilities, not stored.

```c
#include "volcomp_label.h"
volcomp_label_plane pl[] = {{1, ink}, {2, surface}, {7, fibers}};   /* class id, 128^3 u8 plane */
volcomp_label_encode(pl, 3, /*q*/ 8.0f, enc, VOLCOMP_LABEL_ENCODE_BOUND(3), &n);
const float qp[3] = {8.0f, 8.0f, VOLCOMP_Q_LOSSLESS};           /* class 7 stored exactly */
volcomp_label_encode_q(pl, 3, 8.0f, qp, enc, VOLCOMP_LABEL_ENCODE_BOUND(3), &n);
volcomp_label_classes(enc, n, cls, &count);                     /* which classes are stored */
volcomp_label_plane_q(enc, n, 7, &q);                           /* the q one plane used */
volcomp_label_decode(enc, n, 2, dst, VOLCOMP_CHUNK_VOXELS);     /* absent class -> zeros */
volcomp_label_decode_block(enc, n, 2, bz, by, bx, blk, VOLCOMP_BLOCK_VOXELS);
```

CLI: `label-encode DIR out.voll --q=Q [--q-plane=CLS=Q ...]` (DIR holds
`<cls>.u8` planes), `label-decode`, `label-verify` (per-class PSNR / MAE /
P90 / P95 / P99 / max).
Measured on the PHercParis4 ink-3d prediction (24 planes of 128³) at q = 8:
9.5 KB per plane (220×, 35× smaller than the blosc-zstd chunks it ships in),
MAE 0.5, P99 6, PSNR 43.7 dB.

## Browser viewer

[`web/dist/volcomp_viewer.html`](web/dist/volcomp_viewer.html) is one self-contained
page (the decoder compiled to WebAssembly and inlined). Double-click it to browse
the volcomp mirror at <https://dl.ash2txt.org/community-uploads/forrest/volcomp/>:
sharded Range streaming, three orthogonal slices, prediction overlays and
`decode_smooth` deblocking. It needs no server. Build and details are in
[`web/README.md`](web/README.md).

## Build the tools and tests

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
./build/release/volcomp encode chunk.u8 chunk.volc --q=8      # --q=0 for lossless
./build/release/volcomp encode prob.u8 prob.volc --q=64 --surface   # surface chunk, threshold 128
./build/release/volcomp verify chunk.volc chunk.u8
./build/release/volcomp shard-pack chunks/ shard.bin --q=8     # chunks/z_y_x.u8
./build/release/volcomp label-encode labels/ chunk.voll --q=8 --q-plane=12=0
tools/fetch_corpus.sh fetch tune && ./build/release/volcomp-bench --corpus=corpus/tune
```

Presets: `dev` (asan+ubsan), `release`, `bench` (`-march=native`), `fuzz`
(libFuzzer targets `fuzz_d_chunk`, `fuzz_rt_chunk`, `fuzz_rt_mask`, `fuzz_rt_mask_ll`). Requires clang or GCC and
C23; Linux x86-64 is where it is measured, arm64 compiles and uses the C
kernels. `-DVOLCOMP_STATIC_AVX2=ON` adds `-mavx2 -mfma` (no runtime dispatch),
`-DVOLCOMP_NO_AVX2=ON` builds the C kernels only; `ctest` always runs the
golden and codec tests on both kernel sets (`test_golden_c`, `test_codec_c`).

## Numbers

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Side-by-side images of raw CT
against q = 1..32 on 512³ cubes from four ESRF scans (0.55 / 1.13 / 2.40 / 9.36 µm):
[`docs/comparison/`](docs/comparison/README.md). Design decisions and the
measured graveyard of rejected ideas: [`docs/measured.md`](docs/measured.md).
