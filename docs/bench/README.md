# volcomp against other compressors on raw scroll CT

> **Status: interim.** Timings come from the contended 12-worker quality pass, and the quiet single-thread
> timing pass is still running. A few extended low-rate settings (x264/x265 crf 51, aom crf 63, zfp-acc 512/1024,
> volcomp q255) are also still pending. Sizes and quality numbers are final for every row present, because they
> do not depend on timing. This note goes away when the final update lands.

`tools/bench_codecs.py` encodes, decodes and scores every codec x setting x cube on the same uint8 CT.

## Contents

| file | what |
|---|---|
| [operating_points.md](operating_points.md) | **bits per voxel at 30 / 35 / 40 dB PSNR**, per cube, every codec |
| [speed_random_access.md](speed_random_access.md) | encode / decode MB/s and random-access bytes + ms, at each codec's cheapest setting reaching 35 dB |
| [rd_psnr.png](rd_psnr.png), [rd_ssim.png](rd_ssim.png) | rate-distortion, one panel per cube, log-x, colour = family, volcomp thick |
| [random_access_PHercParis4_2.40um.png](random_access_PHercParis4_2.40um.png) | random-access bytes read and decode time, one 128³ chunk |
| [results.csv](results.csv) / [results.json](results.json) / [results.jsonl](results.jsonl) | every row (the jsonl is the raw run log the report is built from) |
| [random_access.csv](random_access.csv) | the random-access rows behind the chart, all cubes |

## Data

These are the four 512³ cubes of **original uint8 CT** from [docs/comparison](../comparison/README.md): the same
origins, fetched from the open-data bucket's level-0 chunks, 128 MiB each. volcomp q8 reproduces
`comparison/report.json` byte for byte (PHerc0191 q8: 4 325 507 B in both).

| cube | scan | origin zyx |
|---|---|---|
| PHerc0500P2_0.55um | 20250821110041-0.550um-0.1m-65keV-masked | 5888, 7936, 1536 |
| PHerc1667_1.13um | 20260323082859-1.129um-0.2m-59keV-masked | 20864, 7168, 2944 |
| PHercParis4_2.40um | 20260411134726-2.400um-0.2m-78keV-masked | 37632, 22272, 22016 |
| PHerc0191_9.36um | 20250821151635-9.362um-1.2m-113keV-masked | 9216, 3584, 6016 |

**Only 512³ cubes were benchmarked.** A 1024³ PHercParis4 region was planned and dropped by the user's decision:
libaom and x265 at one thread took tens of minutes per setting on it. The script still knows the cube
(`PHercParis4_2.40um_1024`).

## Machine and versions

- **Host:** forlindesk2, AMD Ryzen 9 9950X (16 cores / 32 threads, AVX-512), 182 GB RAM, Ubuntu 26.04.1,
  kernel 7.0.0-31, gcc 15.2. The desk was shared the whole time. Two m7 inference workers and a
  surface_compare job were running, using about 5 cores and 60-90 GB of RAM.
- **volcomp:** this repo at the benchmark commit, `release` preset (`-O3`, runtime AVX2 dispatch), called through
  `python/volcomp_zarr` (ctypes) one 128³ chunk at a time.
- **ffmpeg 8.0.1** (Ubuntu) with libx264 0.165.3222, libx265 4.1, libsvtav1enc 2.3.0 and libaom 3.13.1 does the
  encoding. **PyAV 19.0.0** (bundled libavcodec 63.1) does the decoding: the native h264 and hevc decoders, and
  libdav1d for both AV1 encoders.
- **imagecodecs 2026.8.16:** libjpeg-turbo 3.2.0, OpenJPEG 2.5.4, libjxl 0.12.0, libwebp 1.6.0, SZ3 3.3.2,
  zstd 1.5.7, lz4 1.10.0.
- **Other libraries:** Pillow 12.3.0 (libavif 1.4.2 with aom), zfpy 1.0.1, numcodecs 0.17.0 (c-blosc 1.21.7),
  Python 3.14.4, numpy 2.5.3, OpenCV 5.0 (for the SSIM filter only).

## Threads and timing

- **One thread per codec, wherever the codec takes a setting:**
  - x264: `-threads 1`, `lookahead-threads=1`
  - x265: `pools=none:frame-threads=1`
  - SVT-AV1: `lp=1`
  - libaom: `-threads 1 -row-mt 0`
  - decoders: PyAV `thread_count = 1`
  - JPEG 2000 and JPEG XL: `numthreads=1`
  - AVIF: `max_threads=1`
  - blosc: `set_nthreads(1)`
  - `OMP_NUM_THREADS=1` for SZ3
  - volcomp, zfpy, zstd, lz4, JPEG and WebP are single-threaded by construction.
- **Quality pass** (every row): 12 worker processes, each pinned to its own core (4-15), with other jobs on the
  host. Those speeds are **contended**. Memory-bound codecs suffer most: volcomp measured 110-130 MB/s encode here,
  against 570 MB/s alone on the same host.
- **Timing pass** (`run --timing`): 4 workers pinned to cores 4, 6, 8 and 10, which are separate physical cores.
  Each encode and decode is the best of 3 when a repetition takes under 10 s. Quality metrics are skipped in this
  pass, and the bytes are asserted equal to the quality pass. `speed_random_access.md` marks every number that
  still comes from the contended pass with `(par)`.
- Encode and decode MB/s = 128 MiB / wall time for the whole cube, through the Python binding of each codec.
- **Video encode time runs through the ffmpeg CLI**, one process per 128-frame GOP. It includes process start
  (a few ms) and piping the raw frames.

## Storage model: what "bytes" and "random access" mean

| family | codecs | unit stored | random access to the centre 128³ chunk (z,y,x = 256:384) |
|---|---|---|---|
| chunked 3-D | volcomp, zfp, sz3, zstd-19, blosc-zstd5, lz4 | one blob per 128³ chunk (as a zarr store keeps them) | read and decode that one blob |
| per-slice 2-D | jpeg, jpeg2000, jpegxl, webp, avif | one image per full 512² z-slice | read and decode 128 full slices, then crop |
| video, GOP 128 | x264, x265, svtav1, aom | one closed 128-frame GOP (keyint 128, no scenecut) per bitstream, z = time | read and decode the whole GOP (128 full 512² frames), then crop |
| video, all-intra | x264/x265/svtav1 `-intra` (keyint 1), aom-intra (`-usage allintra`) | the same 128-frame files | the same GOP file (a codec could seek frame by frame, but still decodes full 512² frames) |

- **Bytes** are the sum of the codec payloads: raw Annex-B / IVF bitstreams for video, raw J2K codestreams, and
  the files as the codec writes them. No zarr shard index or container overhead is counted.
- **Random access** is the bytes of the units read, plus the wall time (best of 3) to reconstruct the chunk. The
  decoded chunk is checked equal to the same region of the full decode.
- The chart uses each codec's **cheapest setting reaching 35 dB** on that cube.

## Metrics

- **PSNR, MAE and max** are computed over all 134 M voxels (peak 255). **bpv** = 8 x bytes / voxels.
- **SSIM is 3-D SSIM**: a Gaussian window with sigma 1.5 applied separably along z, y and x (truncate 3.5,
  11 taps), K1 = 0.01, K2 = 0.03, L = 255. It is the mean of the SSIM map over the cube minus a 5-voxel border,
  computed in 64-slice z-slabs with a halo. It matches `scipy.ndimage.gaussian_filter`-based 3-D SSIM to 3e-7.
- **Operating points** at 30 / 35 / 40 dB are log-linear interpolations of bpv along each codec's sweep. In the
  table, `<=x` means even the lowest swept rate already exceeds the target. `-` means the codec never reached it.
- **ZFP and SZ3** work on float32: the cube is cast up, and the output is rounded and clipped to uint8.

## Settings swept

| codec | knob | values |
|---|---|---|
| volcomp | q | 1 2 4 8 16 32 64 128 (255 pending) |
| volcomp-smooth | q (same streams) | decoded with `volcomp_decode_smooth(sigma 2, GATED, ZERO_GUARD)`, the README's CT recipe |
| x264 | crf, `-preset medium -tune psnr` | 8 14 20 26 32 38 44 (51 pending) |
| x265 | crf, `-preset medium -tune psnr` | 8 14 20 26 32 38 44 (51 pending) |
| svtav1 | crf, `-preset 8`, `tune=1` (PSNR) | 8 18 28 38 48 58 63 |
| aom | crf, `-cpu-used 6 -b:v 0 -tune psnr` | 8 18 28 38 48 58 (63 pending) |
| x264/x265-intra | crf, keyint 1 | 8 16 24 32 40 |
| svtav1/aom-intra | crf | 8 23 38 53 63 |
| jpeg | quality, `optimize=True` | 2 5 10 25 50 70 85 95 |
| jpeg2000 | target PSNR (OpenJPEG `level`), 9/7, J2K codestream | 28 31 34 37 40 43 46 |
| jpegxl | distance, effort 7 | 0.5 1 2 3 5 8 12 18 25 |
| webp | quality, lossy, method 4 | 0 5 20 40 60 80 95 |
| avif | quality, speed 6, 4:0:0 | 0 5 10 25 40 55 70 85 95 |
| zfp-acc | fixed-accuracy tolerance | 0.25 ... 256 (512 1024 pending) |
| zfp-rate | fixed-rate bits/value | 0.125 0.25 0.5 1 1.5 2 3 4 |
| sz3 | absolute error bound | 0.5 1 2 4 8 16 32 |
| zstd-19 / blosc-zstd5 / lz4 | lossless | zstd level 19; Blosc(zstd, clevel 5, shuffle); lz4 block default |

## Caveats

- **Video codecs are 8-bit and see a grey image in two different ways:**
  - x264, x265 and libaom take `gray` directly (4:0:0).
  - libsvtav1 only takes yuv420p. We feed it constant-128 chroma planes, which cost a few hundred bytes per GOP.
  - No range conversion happens anywhere: the input and output pixel formats are the same, and the Y plane is
    read straight from the decoder.
  - All four encoders run PSNR-tuned: `-tune psnr` for x264, x265 and libaom (psy-rd and AQ off), `tune=1` for
    SVT-AV1. These are PSNR settings, not visual ones.
- **Video wins on rate by exploiting z-correlation across 128 frames with motion search.** It pays for that at
  read time: one chunk costs a full GOP of full-size frames, and the GOP length trades rate against access cost.
  - With keyint 128 the video GOP covers the same z-range as one chunk row but the whole xy plane: 16 chunks'
    worth of data at 512².
  - On a full volume (thousands of pixels per side) the xy cost grows with the frame size, while a chunk stays
    128³. A 512² frame is the most favourable case for video here.
- **Per-slice codecs ignore z entirely.** Their random access decodes 128 full slices; none of these libraries
  was asked for a region decode. JPEG 2000 and JPEG XL can do tiled or region decode, and libjpeg-turbo can crop.
  With tiles the bytes would fall toward (128/512)² of the numbers here, at some rate cost.
- **WebP has no grey mode.** The slice is replicated to RGB and coded as YUV 4:2:0 with neutral chroma. The
  green channel of the decoded RGB is compared.
- **AVIF** is written through Pillow as monochrome 4:0:0 (aom, speed 6). imagecodecs' AVIF encoder ignored
  `level` here (every level came out lossless), so it was not used.
- **JP3D (JPEG 2000 Part 10) is not in the results.** OpenJPEG 2.1.2's `opj_jp3d_compress` / `_decompress` build
  with gcc 15 (`-DBUILD_JP3D=ON`, `-std=gnu89`, at `-O2` and at `-O0`), but they do not round-trip. Even
  lossless (`-r 1`, and `-n 1,1,1`) decodes a 128³ chunk to noise (9 dB PSNR). OpenJPEG 2.2+ removed the JP3D code.
  The `jp3d` codec stays in the script (skipped by default) for anyone with a working build.
- **ZFP operates on float32**, so its accuracy mode spends bits on precision that uint8 data does not have. It is
  a scientific-float codec on integer data, measured as-is.
- **volcomp-smooth** is a decode-time option on the same streams, so it has the same bytes as volcomp.
  - It trades decode speed for quality: about 20 ms per chunk against about 3 ms (contended).
  - Its gain is +0.1 to +0.7 dB at q >= 8, and nothing at q <= 2, where the filter is off by design.
  - Downstream note (from [deblocking.md](../deblocking.md), not re-run here): for recto-teacher inference on
    q8 CT, plain decode was marginally better than smoothed (dice 0.951 vs 0.948-0.949), and smoothing helps
    at q16.
  - The model-inference check was skipped as instructed.
- **Masked volumes:** these cubes are densest-papyrus windows, so there is almost no masked air. volcomp's
  all-zero-chunk elision plays no part.

## Reproduce

```sh
# on the benchmark host
cmake --preset release && cmake --build --preset release --target volcomp_shim
python3 -m venv venv && venv/bin/pip install numpy scipy "zarr>=3" s3fs imagecodecs zfpy pillow \
    opencv-python-headless av matplotlib numcodecs
export VOLCOMP_LIB=$PWD/build/release/libvolcomp.so PYTHONPATH=$PWD/python
C=PHerc0500P2_0.55um,PHerc1667_1.13um,PHercParis4_2.40um,PHerc0191_9.36um
venv/bin/python tools/bench_codecs.py run --cubes $C --data cubes --out docs/bench --workers 12 --cpus 4,5,6,7,8,9,10,11,12,13,14,15
venv/bin/python tools/bench_codecs.py run --timing --cubes $C --data cubes --out docs/bench --workers 4 --cpus 4,6,8,10
venv/bin/python tools/bench_codecs.py report --out docs/bench
```

`run` fetches missing cubes (anonymous S3) and is resumable. `--codecs a,b` and `--mid-only` (one setting per
codec) are for smoke tests.
