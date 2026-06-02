# volume-compressor — BUILD_RESULTS

Frozen single-pipeline lossy codec built from scratch per `plan.txt` (clean unit
`src/vc/vc.{h,c}`). Pure C23, CPU-only, compiler-autovectorized, no hand
intrinsics, single code path. Measured on real PHerc Paris 4 256³ u8 volumes.

## What builds + is committed
- `src/vc/vc.h` / `src/vc/vc.c` — the entire codec (one drop-in unit, libc+libm only).
- Pipeline (all of plan §2/§3 present, single path, no knobs):
  integer DCT-16³ (Q14 orthonormal, separable, autovectorized) →
  dead-zone scalar quant (1-parameter HF-boost curve, dz=0.80, dequant offset 0.40) →
  3D low→high frequency scan + EOB (last-significant-position) →
  adaptive table-free binary range coder (CABAC-style, **reset per atom**, sig/mag
  contexts + sign/EG bypass) → per-member q via trivial encode-measure-adjust
  bisection to hit the target ratio (NO λ, NO per-block q) → uniform-atom
  fast-path → ABSENT-chunk handling → 8 fully-independent LODs (box-downsample +
  re-encode, no inter-LOD coding) → single-file archive (member dir + per-chunk
  per-atom directory, implicit offsets, footer w/ magic at both ends).
- Decode is the exact inverse; **deblock** (adaptive, atom-grid faces, strength
  0.4, ≤2 samples/side, edge-preserving clip) applied in `decode_lod`/
  `decode_region` only — single-atom decode stays `touched==1` (no neighbors).

## Test status — PASS
- `tests/test_vc.c` (synthetic): round-trip, single-atom random access (interior
  exact vs full decode), uniform atom exact, ABSENT-region path, LOD pyramid.
- `tests/test_realdata.c` (PHerc hires, 50×): **exhaustive** single-atom decode
  over all 4096 LOD0 atoms vs `decode_lod` interior → **0 / 7,077,888 mismatch**;
  all 8 LODs decode; region decode OK. Random access (touched==1) verified.
- Autovectorization: `gcc -O3 -ffast-math -fopt-info-vec` reports **20 loops
  vectorized** including the DCT matrix-vector kernels, the quant/dequant loops,
  and the deblock gradient loops. No hand intrinsics.

## Measured results (target ratios 10/20/50/100×)
`ratio` = whole 8-LOD archive vs raw LOD0 (so it sits ~12% under target because
the pyramid adds ~14% raw voxels per plan §5; LOD0 alone hits the target). PSNR
over the full volume; SSIM/GMSD on the central Z slice. enc/dec in MB/s of LOD0;
`atom_us` = mean single-16³-atom decode latency; `pyramid_MB` = total archive.

### hires_256.u8 (256³ = 16.78 MB)
| target | achieved | PSNR dB | SSIM | GMSD | enc MB/s | dec MB/s | atom µs | archive MB |
|-------:|---------:|--------:|-----:|-----:|---------:|---------:|--------:|-----------:|
| 10×  | 8.76×  | 46.97 | 0.9938 | 0.0130 | 5.6 | 90 | 37.6 | 1.92 |
| 20×  | 17.52× | 43.26 | 0.9853 | 0.0266 | 6.3 | 114 | 28.0 | 0.96 |
| 50×  | 43.70× | 37.71 | 0.9500 | 0.0619 | 6.7 | 139 | 22.3 | 0.38 |
| 100× | 87.45× | 33.49 | 0.8933 | 0.0951 | 6.9 | 152 | 20.3 | 0.19 |

### coarse_256.u8 (256³ = 16.78 MB)
| target | achieved | PSNR dB | SSIM | GMSD | enc MB/s | dec MB/s | atom µs | archive MB |
|-------:|---------:|--------:|-----:|-----:|---------:|---------:|--------:|-----------:|
| 10×  | 8.75×  | 37.83 | 0.9833 | 0.0448 | 5.7 | 88 | 37.8 | 1.92 |
| 20×  | 17.46× | 33.42 | 0.9517 | 0.0865 | 6.1 | 113 | 28.9 | 0.96 |
| 50×  | 43.75× | 28.34 | 0.8505 | 0.1390 | 6.5 | 133 | 22.8 | 0.38 |
| 100× | 87.12× | 25.04 | 0.7180 | 0.1808 | 6.9 | 150 | 20.9 | 0.19 |

Pyramid (all 8 LODs, hires): 256³…2³ = 8 members; total archive sizes above
(1.92 → 0.19 MB across 10×→100×). Coarse-LOD overhead over LOD0 is small and
shrinks at higher q, matching plan §5's ~+14%-raw geometric bound.

## Notes / deviations (minimal-first, documented)
- **Decode MB/s (85–152) ≫ encode MB/s (~6)**: encode is dominated by the
  encode-measure-adjust rate control (~10-iteration q bisection × 8 LODs ≈ 88
  full encodes). A single encode pass is ~10–11× faster (~60–70 MB/s); the search
  is the cost. This is the trivial q→ratio mapping sanctioned by plan §2.7 (no λ,
  no allocation). Could be sped with a calibrated curve later; not required.
- **Atom-payload 64B alignment relaxed.** plan §4 suggested 64B-aligning atom
  payloads, but at 100× atoms are ~tens of bytes, so 64B padding capped the ratio
  near 64× and bought nothing (entropy decode is byte-serial; the DCT runs on heap
  scratch, not the payload). Payloads are packed tightly; chunk directories and the
  payload region are 64B-aligned for mmap. Atom byte offsets are **implicit**
  (cumulative payload length), shrinking the per-atom directory to 8 B/entry — this
  recovered +3 to +11 dB PSNR at 50–100× vs the padded/u64-offset layout.
- **Footer is a fixed (versioned) struct**, not yet a TLV/skippable-tag block as
  plan §4 ideally wants. It has magic at both ends, version, and dir offset/len for
  a one-tail-read open. Adequate for the freeze; TLV evolution can be added later.

## Build commands
```
# library + tests
gcc -std=c23 -O3 -ffp-contract=fast -ffast-math src/vc/vc.c tests/test_vc.c -lm -o test_vc
gcc -std=c23 -O3 -ffast-math -D_POSIX_C_SOURCE=200809L -I. tests/test_realdata.c src/vc/vc.c -lm -o test_realdata
# benchmark
gcc -std=c23 -O3 -ffp-contract=fast -ffast-math -D_POSIX_C_SOURCE=200809L src/vc/vc.c harness/bench_vc.c -lm -o bench_vc
# autovectorization check
gcc -std=c23 -O3 -ffast-math -c src/vc/vc.c -fopt-info-vec
```
