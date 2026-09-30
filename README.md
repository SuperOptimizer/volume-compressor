# volcomp

volcomp is a single-header C compressor (`volcomp.h`) for `uint8` volumes, built for the Vesuvius Challenge
scroll micro-CT and for the model predictions made from it. Each 16³ block is coded with a 3-D DCT, a
dead-zone quantiser and a tANS entropy coder. Blocks are grouped into independently decodable 128³ chunks,
and chunks sit in ordinary zarr v3 `sharding_indexed` shards of 1024³. There is one quality knob, `q`, plus
exact modes: lossless (`q = 0`), two binary-mask modes, and a surface mode that keeps a probability
threshold exact for every voxel. The decoder also compiles to WebAssembly. No threads, no dependencies
beyond libc/libm.

Released with it (September 2026 contest submission, full writeup in
[docs/SUBMISSION_2026-09.md](docs/SUBMISSION_2026-09.md)):

- the C library (version 1.3.0, format revision 4), a CLI, tests, fuzz targets and the
  [format specification](spec/format.md);
- a Python zarr v3 codec, `volcomp-zarr` ([python/](python/README.md)), registered as `"volcomp"`;
- a single-file browser viewer, [web/dist/volcomp_viewer.html](web/dist/volcomp_viewer.html);
- the mirror at <https://dl.ash2txt.org/community-uploads/forrest/volcomp/>: the open-data CT volumes of
  all 39 samples as volcomp zarr v3 stores, the published surface masks as volcomp mask stores, and
  61 new whole-volume m7 surface-probability stores;
- the VC3D integration, ScrollPrize/villa PR [#1704](https://github.com/ScrollPrize/villa/pull/1704)
  (open, not merged).

![PHercParis4 2.4 um centre z plane: raw, then q 1, 2, 4, 8, 16, 32](docs/comparison/PHercParis4_2.40um/zmid_all.png)

PHercParis4 2.4 µm, one 512² plane of a 512³ cube: the original CT, then q 1, 2, 4, 8, 16 and 32 (plain
decode). Up to q 8 the fibre structure is intact at 1:1; from q 16 block structure appears in smooth regions.

## Headline numbers

Single thread, Core Ultra 9 275HX. Copied from the submission's summary; each row names its source.

| what | number | source |
|---|---|---|
| PHercParis4 2.4 µm CT, 54 held-out 128³ chunks, q 8 | **52.9x**, PSNR 35.0 dB, MAE 3.2 | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| same, decode / encode throughput at q 8 | **1,375 / 784 MB/s** per core | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| 512³ cubes at q 8, four scans (0.55 / 1.13 / 2.40 / 9.36 µm) | 145.7x / 131.7x / 52.3x / 31.0x at 41.6 / 42.7 / 36.6 / 33.4 dB | [docs/comparison/README.md](docs/comparison/README.md) |
| PHerc0343P 8.64 µm CT, whole level 0 (138.04 Gvox, masked) | 239.3 MiB on the mirror | [fact sheet](docs/_factsheet_2026-09-30.md) 3g |
| our m7 surface stores (8-bit probability, q 8) vs the published binary m7 masks (blosc) | **1.63x to 2.34x smaller** at level 0 | [docs/comparison/surfaces/README.md](docs/comparison/surfaces/README.md) |
| compression-only cost on an m7 prediction (q 8 vs the same run's float output) | dice 0.978, MAE 4.10 / 255 | [docs/comparison/surfaces/PHerc0343P/README.md](docs/comparison/surfaces/PHerc0343P/README.md) |
| surface mode (exact threshold), q 64 | 0.78x (recto) / 0.71x (m7) the size of q 8, threshold exact for every voxel | [docs/surface_mode.md](docs/surface_mode.md) |

The PHerc0343P whole-volume figure is 550x, but most of that volume is masked air, which costs almost
nothing; the held-out chunk and cube rows are the numbers for dense papyrus.

## Quick start

### C

```c
#include "volcomp.h"

uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
size_t n;
volcomp_status st = volcomp_encode(src /* 128^3 z-major */, 8.0f, enc, VOLCOMP_ENCODE_BOUND, &n);
st = volcomp_encode(src, VOLCOMP_Q_LOSSLESS, enc, VOLCOMP_ENCODE_BOUND, &n);  /* exact */
st = volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);   /* same call for every mode */
float q; st = volcomp_stream_q(enc, n, &q);               /* 0 => the stream is exact */
st = volcomp_decode_block(enc, n, bz, by, bx, block, VOLCOMP_BLOCK_VOXELS);
```

Everything is `static`; include the header in the translation unit that uses it. Masks, surface chunks,
decode-side smoothing and the kernel details are in [docs/api.md](docs/api.md).

### Python

The package is `volcomp-zarr` (module `volcomp_zarr`, Python >= 3.10). It is not on PyPI; build the shared
library and install from the repository:

```sh
cmake --preset release && cmake --build --preset release --target volcomp_shim
cp build/release/libvolcomp.so python/volcomp_zarr/ && pip install ./python[zarr]
```

The bytes-level API works on one 128³ `uint8` chunk in z-major order. This snippet was run against the
repository's `libvolcomp.so` (submission section 3.6):

```python
import numpy as np
import volcomp_zarr as vc

z, y, x = np.mgrid[0:128, 0:128, 0:128]
chunk = 128 + 60 * np.sin(x / 9.0 + z / 13.0) + np.random.default_rng(0).normal(0, 6, x.shape)
chunk = np.clip(chunk, 0, 255).astype(np.uint8)

enc = vc.encode(chunk.tobytes(), 8)                  # lossy, q = 8
dec = np.frombuffer(vc.decode(enc), np.uint8).reshape(chunk.shape)
exact = vc.encode(chunk.tobytes(), vc.Q_LOSSLESS)    # q = 0, byte-exact
assert bytes(vc.decode(exact)) == chunk.tobytes()
blk = np.frombuffer(vc.decode_block(enc, 2, 3, 4), np.uint8).reshape(16, 16, 16)
assert np.array_equal(blk, dec[32:48, 48:64, 64:80])  # one block == that region of the full decode
sm = vc.decode_smooth(enc, 2.0, zero_guard=True, gated=True)   # opt-in deblocking for CT at q >= 4
```

As a zarr v3 codec it is a single serializer; pass `compressors=None`, because the chain must be volcomp only:

```python
import zarr
from volcomp_zarr import VolcompCodec
z = zarr.create_array(store, shape=shape, chunks=(128,) * 3, shards=(1024,) * 3, dtype="uint8",
                      serializer=VolcompCodec(q=8), compressors=None, fill_value=0)
# VolcompCodec(q=0) lossless; mode="mask" | "mask-lossless"; mode="surface", q=64, threshold=128
```

Importing `volcomp_zarr` registers the codec, so `zarr.open(...)` reads a local copy of any mirror store;
`volcomp_zarr.set_read_smoothing(2)` (or `VOLCOMP_SMOOTH=2`) turns on deblocked reads. More in
[python/README.md](python/README.md).

### The viewer

Download [web/dist/volcomp_viewer.html](web/dist/volcomp_viewer.html) (one self-contained page, about 300 KB,
decoder in WebAssembly) and double-click it. It works from `file://`, needs nothing installed, and streams
any store on the mirror by HTTP Range. Pick a sample; `I` adds a prediction layer, `D` toggles deblocking,
`?` lists the keys. The URL hash records the whole view, so a copied link reopens it. Tested in headless
Chromium only. Details: [web/README.md](web/README.md).

![viewer: PHerc0343P 2.215 um CT with m7 L2 probabilities](docs/img/viewer_ct_m7L2.png)

PHerc0343P 2.215 µm CT at level 0 with our m7 `-L2` probabilities (red), drawn from their own 8.86 µm level
0 over the finer CT.

## The mirror

Root: <https://dl.ash2txt.org/community-uploads/forrest/volcomp/>. It mirrors the open-data bucket layout:

| path | contents |
|---|---|
| `<Sample>/volumes/<scan>.zarr/<level>/` | CT, levels `0` to `5`; q 8 at level 0, then q 4 / 2 / 1 at levels 1 / 2 / 3 and above |
| `<Sample>/representations/predictions/surfaces/<name>.zarr/<pitch>/` | predictions, levels named by pitch in µm |
| `<Sample>/representations/predictions/surfaces/<scan>-surface-20260925170000-surface-m7-L<k>-prob.zarr` | our m7 stores: `uint8` = round(255·p), q 8, eight levels, shards 1024³ at level 0 shrinking to 128³ |

Inside a level, shards are at `c/<sz>/<sy>/<sx>` (as in [tools/export/README.md](tools/export/README.md)).

- **CT**: the open-data volumes of all 39 samples. Example: `PHerc0343P/volumes/20250521134555-8.640um-1.2m-116keV-masked.zarr`
  (level 0 239.3 MiB, the six-level pyramid 372.6 MiB).
- **Published surface masks**, re-exported as exact mask stores (mode 5; the PHercParis4 recto as mode 4).
- **61 m7 stores**: the m7 surface model (published `surface_m7_nnunet` checkpoint) run over all 31 volumes
  at 8-9 µm at level 0, plus 30 fine-pitch scans at the CT level nearest 9 µm (level 2 for 2.4 µm, 3 for
  1.1 µm, 4 for 0.5 µm, 1 for 4.3 µm). PHerc0343P 8.64 µm is 252.3 MiB, PHerc0009B 1.33 GiB. Example:
  `PHerc0343P/representations/predictions/surfaces/20260304131111-surface-20260925170000-surface-m7-L2-prob.zarr`.

**Reading one chunk over HTTP.** A shard is a plain file: chunk streams back to back, then 512
(u64 offset, u64 nbytes) pairs and a CRC-32C of those 8192 bytes ([spec §7](spec/format.md)). So one chunk
is two Range requests and one `decode`. The viewer does exactly this, and web/README.md records that the
Python decoder gives the same checksums as the browser on bytes fetched over HTTP. The function below
follows the spec; it was tested against a shard written by `volcomp shard-pack` with the Range requests
served from a local file, not against the mirror. It does not check the index CRC.

```python
import struct, urllib.request
import numpy as np
import volcomp_zarr as vc

def fetch(url, rng):                       # one HTTP Range request
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"})
    with urllib.request.urlopen(req) as r:
        return r.read()

def read_chunk(level_url, cz, cy, cx):
    """One 128^3 chunk of a level as (128, 128, 128) uint8; chunk coords in units of 128 voxels."""
    shard = f"{level_url}/c/{cz // 8}/{cy // 8}/{cx // 8}"
    index = fetch(shard, "-8196")[:8192]   # 512 x (u64 offset, u64 nbytes), then crc32c
    ci = (cz % 8) * 64 + (cy % 8) * 8 + (cx % 8)
    off, n = struct.unpack_from("<QQ", index, 16 * ci)
    if off == 0xFFFFFFFFFFFFFFFF:          # missing chunk = fill value 0 (air)
        return np.zeros((128, 128, 128), np.uint8)
    return np.frombuffer(vc.decode(fetch(shard, f"{off}-{off + n - 1}")), np.uint8).reshape(128, 128, 128)
```

A missing shard is an HTTP 404 and also means air. Reading the mirror through zarr-python with an HTTP
store has not been tested; the documented path is a local copy.

## The format in one screen

| unit | size | role |
|---|---|---|
| block | 16³ | transform unit; random-access unit inside a chunk |
| substream | 16 blocks | 32 per chunk; decoding one block touches one substream |
| chunk | 128³ = 8x8x8 blocks | independently decodable; one zarr chunk |
| shard | 1024³ = 8x8x8 chunks | zarr v3 `sharding_indexed`, index at the end, crc32c-checked |

**q** is the quantiser step in voxel units, 1 to 255 (typical 2 to 32); `q = 0` is lossless. The DC step
is q/8 and the AC step is q·(1+r)^0.65 with r = x+y+z. Error grows with q; there is no error bound.

| mode (header byte 5) | name | what it stores | measured size |
|---|---|---|---|
| 0 | lossy DCT | q = 1..255 as above | see Results |
| 1-3 | lossless (q = 0) | per-block prediction along z, zigzag residuals, CONST / SPARSE / DENSE / RAW per block; never more than raw + 8 bytes | 1.8x on synthetic CT; class maps and masks 60-250x |
| 4 | mask | 2x2x2 majority pool of a binary mask (64³), context-coded, decoded trilinearly to a 0..255 ramp | 0.0057 bits/voxel on the PHercParis4 recto, 11x smaller than packbits + zstd-19 |
| 5 | mask, spatially lossless | the full 128³ binary mask, same coder, decodes to exact 0/255 | 0.0255 bits/voxel, 2.4x smaller than packbits + zstd-19 |
| 6 | surface | a DCT base with a flat step law plus a refinement that keeps a threshold exact | 0.71-0.78x the size of q 8 at q 64 |

An older reader rejects a mode it does not know instead of misreading it. Every mode decodes through the
same `volcomp_decode` call.

**zarr codec chain.** The inner codec is `{"name": "volcomp", "configuration": {"q": 8}}` inside
`sharding_indexed` with `index_location: "end"` and `index_codecs = [bytes(little), crc32c]`, and nothing
else: no second compressor. A missing or all-zero chunk reads as the fill value 0.

**Deblocking is a reader option**, not part of the format. `volcomp_decode_smooth` smooths a chunk's
interior block faces and then projects each block back into the quantisation cells it was decoded from, so
the result is still a reconstruction the stream allows. Exact zeros stay zero with the zero guard, and a
surface chunk keeps its exact threshold.

| data | setting | effect | cost |
|---|---|---|---|
| masked CT, q >= 4 | gated face filter, strength 2, zero guard | +0.11 to +0.61 dB PSNR, no chunk worse than no filter on 4 volumes at q 4, 8, 16 | 5-6 ms per chunk (380-420 MB/s) |
| masked CT, q < 4 | none (decode_smooth turns itself off) | gated filters are within ±0.03 dB of no filter at q 2 | none |
| probability maps | Gaussian seam blur σ 0.6, zero guard | m7 q 16: dice 0.9913 to 0.9931, iso-surface shift 0.095 to 0.075 voxel | 13-17 ms per chunk |

Normative format: [spec/format.md](spec/format.md). Surface mode: [docs/surface_mode.md](docs/surface_mode.md).
Deblocking: [docs/deblocking.md](docs/deblocking.md). API: [docs/api.md](docs/api.md).

## Results

54 held-out 128³ chunks of PHercParis4 2.4 µm (masked), never used for design; single thread, AVX2 build,
medians of 3, about ±10 % timing noise under WSL2 ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| q | ratio | PSNR dB | SSIM | MAE | P99 | max | enc MB/s | dec MB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 17.56 | 41.33 | .9840 | 1.573 | 7 | 49 | 471 | 702 |
| 4 | 30.05 | 38.04 | .9700 | 2.282 | 11 | 77 | 574 | 906 |
| 8 | 52.92 | 35.00 | .9463 | 3.217 | 15 | 121 | 784 | 1375 |
| 16 | 95.58 | 32.22 | .9088 | 4.416 | 21 | 143 | 1033 | 2135 |
| 32 | 176.5 | 29.67 | .8532 | 5.911 | 28 | 155 | 1113 | 2484 |

512³ cubes (128 MiB each) of the original CT from four ESRF scans, plain decode
([docs/comparison/README.md](docs/comparison/README.md)):

| q | 0.55 µm PHerc0500P2 | 1.13 µm PHerc1667 | 2.40 µm PHercParis4 | 9.36 µm PHerc0191 |
|--:|--:|--:|--:|--:|
| 1 | 28.4x, 48.3 dB | 33.1x, 50.7 dB | 12.5x, 46.5 dB | 7.9x, 45.1 dB |
| 2 | 49.6x, 46.2 dB | 52.0x, 48.3 dB | 19.7x, 43.3 dB | 11.8x, 41.1 dB |
| 4 | 85.2x, 44.0 dB | 81.7x, 45.6 dB | 31.7x, 39.9 dB | 18.6x, 37.2 dB |
| 8 | 145.7x, 41.6 dB | 131.7x, 42.7 dB | 52.3x, 36.6 dB | 31.0x, 33.4 dB |
| 16 | 250.3x, 39.2 dB | 217.6x, 39.8 dB | 90.1x, 33.3 dB | 55.7x, 29.8 dB |
| 32 | 428.0x, 36.7 dB | 361.5x, 36.9 dB | 163.4x, 30.3 dB | 109.3x, 26.7 dB |

The same q behaves very differently across pitches: "about 50x at 35 dB" is a statement about 2.4 µm, not
about every scan.

- Images of every cube, every q, plain and deblocked, and error overlays: [docs/comparison/](docs/comparison/README.md).
- Our m7 stores against the published m7 masks, with images: [docs/comparison/surfaces/](docs/comparison/surfaces/README.md).
- Head-to-head against zstd, blosc, per-slice image codecs, video codecs, ZFP and JP3D on the same cubes:
  docs/bench/README.md, **in progress** (`tools/bench_codecs.py`). Until it lands, no claim is made that
  volcomp beats any of them on scroll CT.

## Downstream

**A teacher on decoded CT.** The recto surface teacher run on an interior 512³ of PHercParis4 decoded from
volcomp, against the same teacher on the uncompressed CT, scored on the central 256³: dice at 0.5 is 0.9777
at q 4, 0.9511 at q 8 and 0.9273 at q 16 ([docs/deblocking.md](docs/deblocking.md)). This teacher was not
trained on compressed data; fine-tuning on it has not been done.

**Our m7 stores against the published m7 masks** ([docs/comparison/surfaces/](docs/comparison/surfaces/README.md)):

| | PHerc0343P (2.215 µm scan, level 2) | PHerc0009B (2.401 µm scan, level 2) |
|---|---|---|
| published, level 0 (binary, threshold 0.2, blosc) | 370 MiB | 999 MiB |
| ours, level 0 (8-bit probability, volcomp q 8) | 158 MiB | 613 MiB |
| published / ours, level 0 | **2.34x** | **1.63x** |
| dice, ours p >= 0.5 vs published, dense 1024³ box | 0.637 | 0.655 |
| dice, ours p >= 0.2 (the published threshold), box | 0.736 | 0.744 |

The agreement is low mainly because these are separate inference runs of the same checkpoint (different
stride, blending, TensorRT fp16 and q 8 CT input; m7's mask moves with window placement, and stride 128
against stride 96 alone gives dice 0.946) and because the published masks are cut at 0.2 rather than 0.5.
Compression itself, measured against the same run's float output on a box of the PHerc0343P 8.64 µm scan,
costs dice 0.978 and MAE 4.10/255.

## Integration status

- **villa / VC3D**: PR [#1704](https://github.com/ScrollPrize/villa/pull/1704) (open, not merged) vendors
  volcomp 1.3.0 into `volume-cartographer/libs/volcomp`, adds a C++ codec shim, teaches `VcDataset`,
  `Volume` and the chunk fetcher to read volcomp stores, and fixes the C++ zarr reader and writer for
  shards with the index at the end and a crc32c index codec, with tests. Decode smoothing is exposed there
  but not yet called by any VC3D reader.
- **Export pipeline**: [tools/export/](tools/export/README.md) re-encodes the bucket with a coordinator
  (sqlite queue of (volume, level, shard) units, leased so workers can die and the run can resume) and
  workers that run `volcomp shard-pack` and `shard-verify` and upload by SFTP with an atomic rename.
- **Python**: the `volcomp-zarr` package registers the codec name `"volcomp"` through the `zarr.codecs`
  entry point (`volcomp = volcomp_zarr:VolcompCodec`). It is not in the zarr codecs registry.

## Limitations

- **No error bound.** Quality is empirical: measured P99 is 15 at q 8 and 28 at q 32 on the held-out set,
  and hostile content can ring arbitrarily.
- **Blocking.** 16³ blocks show at high zoom; the villa PR notes visible artefacts at q 8 above 2x zoom at
  native resolution. Deblocking is opt-in and per chunk in the viewer, so chunk faces are left as decoded.
- **Models not trained on compressed data are sensitive to it** (teacher dice 0.951 at q 8). No effect of
  compression on tracing or segmentation quality has been measured.
- **q 8 probability stores lose the threshold** on 0.18-0.66 % of voxels. The released m7 stores are q 8;
  surface mode fixes this but is not what they use, and a reader older than format revision 4 rejects it.
- **Comparison with other codecs is pending**; video codecs may win on ratio at equal PSNR.
- **Corpus.** The design and main tables use PHercParis4 2.4 µm; four cubes show the spread across pitch.
- **Platforms.** Measured on Linux x86-64 under WSL2 only; arm64 compiles and uses the C kernels,
  unmeasured. The viewer is tested in headless Chromium only and shows axis-aligned slices only.
- **Mask mode 4 is not exact** (1.7 % of thresholded voxels differ, up to 2 voxels); mode 5 is.
- **Lossless mode loses to zstd-19** by 23-34 % on busy, non-sparse channels.
- **The m7 release** was not made with identical settings throughout (stride 96 for two volumes, 128 for 59).

## Build and test

```sh
cmake --preset release && cmake --build --preset release && ctest --preset release
./build/release/volcomp encode chunk.u8 chunk.volc --q=8                # --q=0 for lossless
./build/release/volcomp encode prob.u8 prob.volc --q=64 --surface       # surface chunk, threshold 128
./build/release/volcomp decode chunk.volc out.u8 --smooth               # opt-in deblocking
./build/release/volcomp verify chunk.volc chunk.u8
./build/release/volcomp shard-pack chunks/ shard.bin --q=8              # chunks/z_y_x.u8
tools/fetch_corpus.sh fetch tune && ./build/release/volcomp-bench --corpus=corpus/tune
```

- Presets: `dev` (asan+ubsan), `release`, `bench` (`-march=native`), `fuzz` (libFuzzer targets
  `fuzz_d_chunk`, `fuzz_rt_chunk`, `fuzz_rt_mask`, `fuzz_rt_mask_ll`; sources in [fuzz/](fuzz/)).
- Requires clang or GCC with C23. On x86-64 the AVX2+FMA kernels are chosen at runtime; every target has
  plain C kernels. `-DVOLCOMP_STATIC_AVX2=ON` compiles AVX2 in without dispatch, `-DVOLCOMP_NO_AVX2=ON`
  builds the C kernels only, and `ctest` runs the golden and codec tests on both kernel sets.
- Python tests: `python/test_ctypes.py`, `python/test_zarr_roundtrip.py`. Viewer tests: [web/test/](web/test/).
  Export: `tools/export/test_export.py` (offline, synthetic bucket).

## Documentation

| file | what it is |
|---|---|
| [docs/SUBMISSION_2026-09.md](docs/SUBMISSION_2026-09.md) | the September 2026 contest writeup, with sources for every number |
| [docs/SUBMISSION_2026-09_short.md](docs/SUBMISSION_2026-09_short.md) | short version of the writeup |
| [docs/api.md](docs/api.md) | the C API in detail: modes, masks, surface chunks, decode smoothing, kernels, CLI |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | held-out and tune tables, throughput, kernel sets, lossless results |
| [docs/comparison/README.md](docs/comparison/README.md) | raw CT against q 1-32 on four 512³ cubes, images and tables |
| [docs/comparison/surfaces/README.md](docs/comparison/surfaces/README.md) | our m7 stores against the published m7 masks |
| [docs/surface_mode.md](docs/surface_mode.md) | surface mode (mode 6): measurements, candidates, design, results |
| [docs/deblocking.md](docs/deblocking.md) | deblocking survey, measurements, downstream teacher check, defaults |
| [docs/measured.md](docs/measured.md) | design decisions and the measured graveyard of rejected ideas |
| [docs/_factsheet_2026-09-30.md](docs/_factsheet_2026-09-30.md) | sourced fact sheet behind the writeup, including known disagreements |
| [docs/_landscape_2026-09-30.md](docs/_landscape_2026-09-30.md) | survey of other volumetric codecs (other data, not comparable) |
| [docs/VOLCOMP_V1_BRIEF.md](docs/VOLCOMP_V1_BRIEF.md) | the v1.0 implementation brief |
| [docs/REVIEW_NOTES_A.md](docs/REVIEW_NOTES_A.md) | review of the predecessor codec c5d that v1.0 was built from |
| [docs/img/](docs/img/) | viewer screenshots |
| docs/bench/README.md | codec head-to-head benchmark, in progress |
| [spec/format.md](spec/format.md) | the normative stream, shard and mode format (revision 4) |
| [python/README.md](python/README.md) | Python binding and zarr v3 codec |
| [web/README.md](web/README.md) | the viewer: use, layer settings, streaming, verification, limits |
| [tools/export/README.md](tools/export/README.md) | the bulk export pipeline |

## License and contact

MIT, see [LICENSE](LICENSE). If you use volcomp or the mirror's stores, please cite the repository,
<https://github.com/SuperOptimizer/volume-compressor>. The CT is the Vesuvius Challenge open data; the m7
stores come from the published `surface_m7_nnunet` checkpoint. Questions and issues: the GitHub repository.
