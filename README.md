# volume-compressor

A fast, CPU-only **lossy compression codec for dense `u8` 3D volumetric data** —
built for Herculaneum / Vesuvius scroll micro-CT, where the workload is "decode
this specific small region, now" far more often than "stream the whole volume."

Single drop-in unit: **`src/vc/vc.c` + `src/vc/vc.h`** (pure C23, libc + libm only,
no GPU, no hand-written intrinsics — everything is compiler-autovectorizable).

## What it is

A frozen single pipeline, no modes or config knobs:

```
32³ ATOM  →  integer DCT-32 (partial-butterfly)
          →  dead-zone quant + 1-param HF-boost curve
          →  frequency scan + EOB  →  adaptive binary range coder (context coder)
```

- **32³ atom** is the unit of transform, quantization, entropy coding, and **random
  access** — any 32³ region decodes standalone, no neighbors needed.
- **8 independent LODs** (`0/`…`7/`, each 2× downsampled), each a standalone encode.
- **Self-describing archive** (magic + version + atom/chunk size, validated on open).
- **Coarse multithreading** (OpenMP, optional): independent atoms/chunks → near-linear.

## Why a 32³ atom

The whole design centers on **fast fine-grained random access**. Decoding one 32³
region costs ~70–230 µs versus tens of milliseconds for codecs that must inflate a
large (256³/128³) chunk to serve the same request — roughly **100–300× faster random
access**, while matching or beating their quality at the ratios scrolls are stored at
(10–20×) and trailing only modestly at extreme ratios.

## Performance (1024³ PHerc Paris 4 region, AVX-512, single socket)

| target | ratio | PSNR  | decode (1 thr) | decode (16 thr) | first-32³-block |
|-------:|------:|------:|---------------:|----------------:|----------------:|
|    10× |  ~9×  | 46 dB |    ~1.2 GB/s   |     ~1.2 GB/s   |      ~0.2 ms    |
|    50× | ~40×  | 39 dB |    ~0.5 GB/s   |     ~2.1 GB/s   |      ~0.1 ms    |
|   100× | ~80×  | 35 dB |    ~0.6 GB/s   |     ~2.5 GB/s   |      ~0.1 ms    |

Encode ~0.65–0.9 GB/s (16 threads). Lossy by design; holds c3d/c4d-class
PSNR-at-ratio with far faster random access.

## API

```c
#include "vc/vc.h"

// Encode a whole u8 volume (all 8 LODs) to one in-memory archive.
vc_status vc_encode(const uint8_t *vol, vc_dims dims, float target_ratio,
                    uint8_t **out_archive, size_t *out_len);   // free(*out_archive)

vc_archive *vc_open(const uint8_t *archive, size_t len);       // borrows the buffer
void        vc_close(vc_archive *a);

vc_status vc_lod_dims   (const vc_archive *a, int lod, vc_dims *out);
vc_status vc_decode_lod (vc_archive *a, int lod, uint8_t *out_vol, vc_dims *out_dims);
vc_status vc_decode_atom(vc_archive *a, int lod, int ax,int ay,int az,
                         uint8_t out[VC_ATOM3]);                // one 32³ block
vc_status vc_decode_region(vc_archive *a, int lod, vc_box box, uint8_t *out);
```

`vc_open` validates the archive (magic, version, geometry) and the decoder is
bounds-checked against malformed/untrusted input — a bad archive returns an error,
never a crash or out-of-bounds read.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build          # round-trip + random-access + format tests
```

CMake options:

| option | default | meaning |
|---|---|---|
| `VC_MARCH` | `native` | target ISA (`native` / `x86-64-v3` / `x86-64-v4` / `armv8-a+simd` / empty) |
| `VC_PORTABLE` | `OFF` | one portable binary; runtime AVX-512/AVX2/generic dispatch (no `-march`) |
| `VC_OPENMP` | `ON` | coarse multithreading (no-op if OpenMP absent) |
| `VC_LTO` | `ON` | link-time optimization |

The source is arch-generic: AVX2, AVX-512, and ARM NEON are all reached from the
same code via the compiler flag — no per-ISA code paths.

## Constraints (by design)

`u8` only · 3D only · lossy only · 10–100× operating range · CPU only · single
frozen pipeline (no selectable modes). Quality is judged on proxy metrics
(PSNR / MS-SSIM / GMSD).

## Testing

Round-trip + exhaustive per-atom random-access + region + format-rejection tests
(`tests/test_vc.c`); fuzzed three ways — libFuzzer (ASan+UBSan), MSan, and AFL++
(LAF-intel + CmpLog/Redqueen) — all clean; TSan-clean on the multithreaded path.

## License

See `LICENSE`.
