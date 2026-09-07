# volcomp

Single-header compressor for `uint8` volumetric data (micro-CT), built for the
Vesuvius Challenge scrolls. One knob (`q`), one format, no state, no threads,
no dependencies beyond libc/libm.

`q` picks the mode:

| `q` | mode | error | use it for |
|---|---|---|---|
| **0** (`VOLCOMP_Q_LOSSLESS`) | predictive, exact | none — decoded bytes are the source bytes | class maps, masks, valid/occupancy flags, anything whose values *are* the data (a wrong label is not a "small" error), and archival copies |
| **1..255** | 3-D DCT + dead-zone quantiser | P99 ≈ 2.5 q on scroll data | greyscale CT and model probability maps, where 20–200× at a controlled error is the point |

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

## Build the tools and tests

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
./build/release/volcomp encode chunk.u8 chunk.volc --q=8      # --q=0 for lossless
./build/release/volcomp verify chunk.volc chunk.u8
./build/release/volcomp shard-pack chunks/ shard.bin --q=8     # chunks/z_y_x.u8
./build/release/volcomp label-encode labels/ chunk.voll --q=8 --q-plane=12=0
tools/fetch_corpus.sh fetch tune && ./build/release/volcomp-bench --corpus=corpus/tune
```

Presets: `dev` (asan+ubsan), `release`, `bench` (`-march=native`), `fuzz`
(libFuzzer targets `fuzz_d_chunk`, `fuzz_rt_chunk`). Requires clang or GCC and
C23; Linux x86-64 is where it is measured, arm64 compiles and uses the C
kernels. `-DVOLCOMP_STATIC_AVX2=ON` adds `-mavx2 -mfma` (no runtime dispatch),
`-DVOLCOMP_NO_AVX2=ON` builds the C kernels only; `ctest` always runs the
golden and codec tests on both kernel sets (`test_golden_c`, `test_codec_c`).

## Numbers

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Design decisions and the
measured graveyard of rejected ideas: [`docs/measured.md`](docs/measured.md).
