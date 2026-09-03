# volcomp

Single-header lossy compressor for `uint8` volumetric data (micro-CT), built
for the Vesuvius Challenge scrolls. One knob (`q`), one format, no state, no
threads, no dependencies beyond libc/libm.

- **block** 16³ (transform + random access), **chunk** 128³ (independently
  decodable, = one zarr chunk), **shard** 1024³ (zarr v3 `sharding_indexed`).
- 3-D DCT-16, dead-zone quantiser, 2-lane tANS with per-chunk tables; no
  in-format post-filter (blocks and chunks reconstruct independently, no
  seams). Optional `volcomp_deblock()` for callers who assemble a region and
  want block edges smoothed. Format: [`spec/format.md`](spec/format.md).
- Decoding a 16³ block touches one substream (≤ 16 blocks) and is
  bit-identical to the same region of the full decode.

## Use

```c
#include "volcomp.h"

uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
size_t n;
volcomp_status st = volcomp_encode(src /* 128^3 z-major */, 8.0f, enc, VOLCOMP_ENCODE_BOUND, &n);
st = volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);
st = volcomp_decode_block(enc, n, bz, by, bx, block, VOLCOMP_BLOCK_VOXELS);
volcomp_deblock(volume, nz, ny, nx, 8.0f);   /* optional, after assembling decoded chunks */
```

Everything is `static`; include the header in the translation unit that uses
it. It requires AVX2 and FMA (`-mavx2 -mfma`, or `-march=native` on a capable
host; the CMake target adds them) and refuses to compile without them. `q` is the quantiser step in voxel units, 1..255 (typical 2..32); error
percentiles scale with q (P99 ≈ 2.5q on scroll data). Define `VOLCOMP_MALLOC`
/ `VOLCOMP_FREE` to override the one scratch allocation in `volcomp_encode`.

## Label volumes (`volcomp_label.h`)

Companion header for multi-class probability arrays: a 128³ label chunk holds
up to 255 class planes (u8 probabilities 0..255 straight from a model), stored
as one zarr array of shape `(255, Z, Y, X)` with only the present classes
taking space. Class identity is an explicit tag per plane and is never lossy:
a decoded chunk cannot carry a class absent from the source, and two classes
cannot be confused. Each stored plane is a `volcomp.h` stream at `q`, so the
error statistics per plane are exactly volcomp's at that q. Thresholded or
downscaled label maps are derived by the consumer from the decoded
probabilities, not stored.

```c
#include "volcomp_label.h"
volcomp_label_plane pl[] = {{1, ink}, {2, surface}, {7, fibers}};   /* class id, 128^3 u8 plane */
volcomp_label_encode(pl, 3, /*q*/ 8.0f, enc, VOLCOMP_LABEL_ENCODE_BOUND(3), &n);
volcomp_label_classes(enc, n, cls, &count);                     /* which classes are stored */
volcomp_label_decode(enc, n, 2, dst, VOLCOMP_CHUNK_VOXELS);     /* absent class -> zeros */
volcomp_label_decode_block(enc, n, 2, bz, by, bx, blk, VOLCOMP_BLOCK_VOXELS);
```

CLI: `label-encode DIR out.voll --q=Q` (DIR holds `<cls>.u8` planes),
`label-decode`, `label-verify` (per-class PSNR / MAE / P90 / P95 / P99 / max).
Measured on the PHercParis4 ink-3d prediction (24 planes of 128³) at q = 8:
9.5 KB per plane (220×, 35× smaller than the blosc-zstd chunks it ships in),
MAE 0.5, P99 6, PSNR 43.7 dB.

## Build the tools and tests

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
./build/release/volcomp encode chunk.u8 chunk.volc --q=8
./build/release/volcomp verify chunk.volc chunk.u8
./build/release/volcomp shard-pack chunks/ shard.bin --q=8     # chunks/z_y_x.u8
./build/release/volcomp label-encode labels/ chunk.voll --q=8   # labels/<cls>.u8
tools/fetch_corpus.sh fetch tune && ./build/release/volcomp-bench --corpus=corpus/tune
```

Presets: `dev` (asan+ubsan), `release`, `bench` (`-march=native`), `fuzz`
(libFuzzer targets `fuzz_d_chunk`, `fuzz_rt_chunk`). Requires clang, C23 and
an AVX2+FMA x86-64 CPU; Linux is the supported target for v1.

## Numbers

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Design decisions and the
measured graveyard of rejected ideas: [`docs/measured.md`](docs/measured.md).
