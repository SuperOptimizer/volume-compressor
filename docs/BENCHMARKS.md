# volcomp — benchmarks (lossy 2026-09-01, lossless 2026-09-07)

Data: real PHercParis4 micro-CT (masked volume, 2.4 µm), 128³ u8 chunks
fetched with `tools/fetch_corpus.sh`: **tune** (54 chunks, levels 0–2,
z ∈ [150, 251]) used for every design choice, **held-out** (54 chunks,
z ∈ [350, 451]) touched only for the numbers below. Each set contains one
contiguous 2×2×2 octet for the chunk-seam metric. Class mix (tune):
29 dense interior, 21 smooth papyrus, 3 surface boundary, 1 air.

Host: Intel Core Ultra 9 275HX under WSL2, clang 23, `-O3 -march=native`
(`bench` preset), one thread pinned to one core, medians of 3 reps. WSL2
timing noise is about ±10%; the volcomp rows below were taken with an
unrelated 2-core job running on the host. c5d oracle: commit `c649fcf`, same
host, its own `bench` preset, 1 thread (measured earlier on an idle host).

## Held-out set — volcomp vs c5d

Bytes are total self-contained stream bytes. c5d rows use its normative
face deblock filter (its filter-off numbers equal volcomp's quality to 4
digits). volcomp rows are the AVX2 build (2026-09-01, second pass).

| q | codec | ratio | PSNR dB | SSIM | MAE | P90 | P95 | P99 | max | enc MB/s | dec MB/s |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | volcomp | **17.56** | 41.33 | .9840 | 1.573 | 3 | 4 | 7 | 49 | 471 | 702 |
| 2 | c5d | 17.45 | 41.31 | .9838 | 1.577 | 3 | 4 | 7 | 48 | 329 | 348 |
| 4 | volcomp | **30.05** | 38.04 | .9700 | 2.282 | 5 | 6 | 11 | 77 | 574 | 906 |
| 4 | c5d | 29.72 | 38.08 | .9706 | 2.267 | 5 | 6 | 11 | 72 | 382 | 426 |
| 8 | volcomp | **52.92** | 35.00 | .9463 | 3.217 | 7 | 9 | 15 | 121 | 784 | 1375 |
| 8 | c5d | 51.88 | 35.15 | .9489 | 3.151 | 7 | 9 | 15 | 117 | 452 | 523 |
| 16 | volcomp | **95.58** | 32.22 | .9088 | 4.416 | 10 | 13 | 21 | 143 | 1033 | 2135 |
| 16 | c5d | 92.18 | 32.53 | .9154 | 4.255 | 9 | 12 | 20 | 154 | 479 | 625 |
| 32 | volcomp | **176.5** | 29.67 | .8532 | 5.911 | 13 | 17 | 28 | 155 | 1113 | 2484 |
| 32 | c5d | 165.1 | 30.09 | .8652 | 5.642 | 13 | 16 | 26 | 162 | 522 | 618 |

- Bytes: volcomp is 0.6% (q2) to 6.9% (q32) smaller at the same q (8-byte
  header, compact 10-bit tANS tables, 8-byte directory entries).
- Quality at the same q: identical up to the effect of c5d's in-format face
  filter (0.02 dB at q2 rising to 0.42 dB / 0.012 SSIM at q32). volcomp has no
  in-format filter; the optional `volcomp_deblock()` post-process recovers it
  (see below), or choose a ~5–10% smaller q.
- Chunk faces: volcomp 128-plane / 16-plane gradient-error ratio 1.00–1.04
  at all q (no seams by construction). c5d's filter skips chunk faces:
  1.04× (q2), 1.72× (q4), 3.08× (q8) rougher than its interior planes.
- Speed, 1 thread: decode 2.0–4.0× c5d's, encode 1.4–2.2× (both use explicit
  AVX2 kernels; volcomp's scalar entropy loops are the remaining floor —
  see `docs/measured.md`, "AVX2 pass").

## Held-out set with the optional `volcomp_deblock()` post-process

Same streams; each decoded chunk post-filtered in place (per chunk, so chunk
faces are left as-is here — filtering an assembled region also treats them).
Decode cost of the AVX2 filter ≈ 2–8% (decode+filter 646 / 957 / 1323 /
1804 / 2425 MB/s at q 2 / 4 / 8 / 16 / 32).

| q | ratio | PSNR dB | SSIM | MAE | P99 | max | blocking amp | vs c5d in-format filter |
|--:|--:|--:|--:|--:|--:|--:|--:|---|
| 2 | 17.56 | 41.34 | 0.9840 | 1.571 | 7 | 49 | 1.193 | +0.03 dB |
| 4 | 30.05 | 38.10 | 0.9709 | 2.259 | 11 | 77 | 1.282 | +0.02 dB |
| 8 | 52.92 | 35.17 | 0.9494 | 3.139 | 15 | 121 | 1.310 | +0.02 dB |
| 16 | 95.58 | 32.54 | 0.9159 | 4.243 | 20 | 155 | 1.241 | +0.01 dB |
| 32 | 176.51 | 30.09 | 0.8656 | 5.629 | 27 | 161 | 1.160 | +0.00 dB |

## Tune set (used for design; volcomp only)

| q | ratio | PSNR | SSIM | MAE | P99 | max | enc MB/s | dec MB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 18.58 | 41.48 | .9831 | 1.526 | 7 | 53 | 538 | 788 |
| 4 | 31.96 | 38.25 | .9684 | 2.198 | 10 | 78 | 661 | 1026 |
| 8 | 56.47 | 35.27 | .9439 | 3.085 | 15 | 103 | 860 | 1430 |
| 16 | 102.4 | 32.52 | .9048 | 4.225 | 20 | 126 | 1040 | 1976 |
| 32 | 190.1 | 29.99 | .8466 | 5.649 | 27 | 161 | 1196 | 2611 |

Paired in-process A/B (both headers compiled into one binary, alternated per
chunk, best of 7 reps) of the AVX2 header against the previous plain-C one:
encode +22 / +36 / +45 / +60 / +57 %, decode +19 / +18 / +31 / +39 / +50 % at
q 2 / 4 / 8 / 16 / 32, bit-identical streams and voxels.

Blocking amplification (boundary/interior gradient-error RMSE on 16-planes):
1.19 / 1.36 / 1.53 / 1.76 / 1.93 at q 2 / 4 / 8 / 16 / 32.

## Error tails

Voxels with |err| > 3q: 1.24% (q2), 0.10% (q8), 0.0004% (q32). They sit in
high-contrast interior blocks (source range > 128: 86–99.6% of them), not at
the air mask (≤ 1.2%). See `docs/measured.md`.

## Kernel sets: AVX2 vs C (held-out, 1 thread, clang 23 `-O3`, 2026-09-04)

`volcomp.h` carries both kernel sets and picks AVX2+FMA at runtime on x86-64;
the C kernels serve arm64 and CPUs without AVX2. Bytes and decoded voxels
were identical between the two on every held-out chunk (bitstreams `q2`/`q32`
goldens included), so the numbers differ only in speed.

| q | kernels | enc MB/s | dec MB/s |
|---|---|---|---|
| 2 | avx2 (runtime dispatch, no `-mavx2`) | 480 | 665 |
| 2 | c | 295 | 368 |
| 8 | avx2 | 868 | 1406 |
| 8 | c | 514 | 610 |
| 32 | avx2 | 1222 | 2424 |
| 32 | c | 642 | 761 |

The static `-mavx2 -mfma` build measured within noise of the dispatch build
(383/724/1110 enc, 588/1262/2393 dec on the same run).

## Where the time goes (q8, 1 thread, AVX2 build)

Decode ≈ 1.4 ms per 2 MiB chunk: tANS decode + dequant ~60% (a scalar
dependency chain: state → table → bits → state, coupled across the two lanes
by the shared bit stream and the contexts), inverse DCT ~35% (x/y passes skip
empty planes and empty upper half-planes), scatter ~5%.
Encode ≈ 2.4 ms: gather + forward DCT ~35%, quantise + nonzero extraction +
sort + tokenise ~40%, tANS + bit writer ~25%. `volcomp_decode_block` re-parses the header and
tables (~20 µs) then decodes one 16-block substream.

## Lossless mode (`q = 0`, 2026-09-07)

Same host, `release` preset (`-O3`, no `-march=native`), one thread pinned to
one core, medians of 5 reps of 5 encodes / 5 decodes. The lossless path has no
SIMD kernels and no float math: these are the plain-C numbers on every target,
and the bytes are identical on every build.

Synthetic classes (`tests/test_lossless.c`, one 128³ chunk each; all decode
byte-identical to the source):

| class | bytes | ratio | enc MB/s | dec MB/s |
|---|--:|--:|--:|--:|
| constant (whole chunk one value) | 9 | 233017× | 50168 | 72328 |
| 2-class mask, 5 % foreground (blobs) | 12 395 | 169× | 1276 | 4356 |
| 8-class label map (Voronoi regions) | 39 244 | 53× | 546 | 2721 |
| smooth SDF-like u8 field | 550 152 | 3.8× | 122 | 290 |
| random noise (worst case) | 2 097 160 | 1.00× | 452 | 20550 |
| synthetic CT (`vt_synth_chunk`) | 1 139 759 | 1.8× | 131 | 224 |

Noise lands on `raw + 8` exactly: the encoder falls back to the raw chunk mode
whenever the coded form would reach 2 097 160 bytes, so a lossless stream is
never larger than the source plus the header. A chunk of 511 flat blocks and
one noise block is 4 826 bytes — the flat blocks cost a token and a byte each.

Real 128³ chunks from a TSM label store (`slab_faces_rv/labels/fine.zarr`,
z 0:128, y 2560:2688, x 4096:4224), for scale against `zstd`
(whole chunk, no dictionary):

| channel | distinct values | volcomp q=0 | ratio | dec MB/s | zstd -3 | zstd -19 |
|---|--:|--:|--:|--:|--:|--:|
| `faces_valid` (3-valued mask) | 3 | **8 346** | 251× | 8700 | 20 774 | 10 692 |
| `rv_class` (4-class map) | 4 | 31 069 | 67× | 4691 | 55 877 | **20 482** |
| `sdf_in` (smooth u8 field) | 234 | 663 690 | 3.2× | 336 | 699 969 | **510 196** |

volcomp is 22 % smaller than `zstd -19` on the mask, 44 % smaller than
`zstd -3` on the class map, and loses to `zstd -19` on the two busier channels
(by 34 % and 23 %) — zstd matches across the whole 2 MiB chunk, while volcomp
codes 16³ blocks that stay independently decodable and reuses one set of
per-chunk tables. What volcomp buys for that is `volcomp_decode_block()` (one
16³ block from one substream, no full-chunk decode), 4–9 GB/s decode on the
label channels, and one format and one API for both modes. Per-block choice of
the prediction axis is the obvious next thing to try on `rv_class`/`sdf_in`.

For reference, the same `rv_class` chunk through the **lossy** codec at q = 8
is 1 696 bytes (1237×) at PSNR 53.9 / max error 3 — which is worthless for a
class map, where an error of 1 is a different class. That is the whole reason
`q = 0` exists.

Fuzzing: 1000 chunks over constant, 5 %-mask, 2..40-class label, smooth-SDF,
noise, synthetic-CT, sparse-label and half-smooth/half-noise generators all
round-trip byte-identically (3.1× overall), and every block of a mixed chunk
decodes through `volcomp_decode_block()` to exactly the bytes of the full
decode. Truncations and single-bit flips are checked not to crash (the
`fuzz` preset covers this exhaustively) and the whole suite passes under
asan+ubsan (`dev` preset).

## Reproduce

```sh
tools/fetch_corpus.sh fetch tune && tools/fetch_corpus.sh fetch heldout
cmake --preset bench && cmake --build --preset bench
taskset -c 5 ./build/bench/volcomp-bench --corpus=corpus/heldout --q=2,4,8,16,32 --reps=3

cmake --preset release && cmake --build --preset release
taskset -c 5 ./build/release/test_lossless --bench       # lossless table above
# the "real" rows need tests/data/{faces_valid,rv_class,sdf_in}.u8 (2 MiB each,
# not in the repo); test_lossless skips them when they are absent
```
