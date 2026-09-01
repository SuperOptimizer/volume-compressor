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

## Build the tools and tests

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
./build/release/volcomp encode chunk.u8 chunk.volc --q=8
./build/release/volcomp verify chunk.volc chunk.u8
./build/release/volcomp shard-pack chunks/ shard.bin --q=8     # chunks/z_y_x.u8
tools/fetch_corpus.sh fetch tune && ./build/release/volcomp-bench --corpus=corpus/tune
```

Presets: `dev` (asan+ubsan), `release`, `bench` (`-march=native`), `fuzz`
(libFuzzer targets `fuzz_d_chunk`, `fuzz_rt_chunk`). Requires clang, C23 and
an AVX2+FMA x86-64 CPU; Linux is the supported target for v1.

## Numbers

See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Design decisions and the
measured graveyard of rejected ideas: [`docs/measured.md`](docs/measured.md).
