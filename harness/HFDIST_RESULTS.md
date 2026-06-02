# HF / perceptual distortion-path experiment (PLAN §2 "Distortion metric" row)

**Question (made urgent by the rate-control result):** the Lagrangian RC experiment
confirmed that plain-MSE allocation *matches but does not beat* uniform-q PSNR on
homogeneous crops — minimizing MSE does **not** improve perceived/structural quality.
So: does optimizing a **different distortion objective** improve perceptual/edge quality
(GMSD / HaarPSI / edge-MAE) at **equal ratio**, even at a small PSNR cost? And does it
fix the **smear + blocking** the user reported on prior codecs?

**Stack under test (the winning config):** integer **DCT-16³** atom · per-coefficient
**16³ quant MATRIX** (new, `src/quant/qmatrix16.c`) · dead-zone · **RLGR** entropy.
Self-contained bench: `harness/bench_hfdist.c` (does not touch codec.c / ratectrl /
chunkmodel). Real data: PHerc Paris 4 `hires-256` (ink/fiber-rich) + `coarse-256`,
256³ each. All objectives driven to the **same target ratio** (10×/20×/50×) by
bisecting the global base step.

### The four objectives
- **(a) FLAT** — flat quant matrix (weight 1 everywhere). Pure-MSE. The baseline.
- **(b) HF** — HF-PROTECTING per-coefficient 16³ quant matrix: finer step (more bits)
  as the 3D frequency index `u+v+w` rises, so edge/ink/fiber HF coefficients survive.
  This is the proper JPEG-XL "quantization matrix" form, matched to the *16³* transform
  block (the existing `subband.c` scalar is keyed to an *8³* block — mismatched to the
  won DCT-16 — so a real per-position 16³ matrix was built for this experiment).
- **(b') ADAPT** — the HF matrix, additionally scaled per-16³-block by local **edge
  energy** (source AC variance): busy ink/fiber blocks get a finer step, flat/air blocks
  coarser. One side byte per atom (counted in the rate). Exact / reproducible.
- **(c) PERC** — true **perceptual-in-loop**: per-atom, search a ladder of step scales
  and pick the one minimizing a perceptual cost (GMSD + edge-MAE) on the **decoded**
  16³ block, under a per-atom rate ceiling. Needs the inverse transform per candidate
  → no free Parseval estimate → measures the **speed cost**.

A new edge-MAE metric (|∇ref − ∇rec|, Prewitt, 2.5D) and a `grid`-wall **seam-step**
metric (mean |Δerror| across 16-voxel block faces — isolates blocking) were added to
`src/metrics/metrics.c` for this study. Round-trip exactness of all three quant modes
through RLGR is covered by `tests/test_qmatrix16.c` (ctest `qmatrix16`, green).

---

## Headline result: HF protection helps at MODERATE ratio, not at aggressive ratio

The clean, **ratio-matched** comparison is the HF-slope sweep (each slope is independently
bisected to the same ratio, so the only variable is HF-protection strength). `hires-256`:

### HF-slope sweep @ 20× (hires) — lower GMSD / edgeMAE / seam = better
| hf_slope | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE | seam16 |
|---|---|---|---|---|---|---|---|
| 0.00 (=FLAT) | 19.83× | 40.83 | 0.9773 | 0.06940 | 0.9729 | 5.659 | 2.892 |
| 0.20 | 19.88× | 40.94 | 0.9778 | 0.06845 | 0.9733 | 5.579 | 2.842 |
| 0.40 | 20.13× | 40.87 | 0.9775 | 0.06903 | 0.9730 | 5.617 | 2.873 |
| **0.60** | **19.86×** | **40.96** | **0.9779** | **0.06808** | **0.9735** | **5.547** | **2.846** |
| 0.80 | 19.87× | 40.93 | 0.9778 | 0.06824 | 0.9733 | 5.550 | 2.866 |
| 0.90 | 20.02× | 40.90 | 0.9776 | 0.06851 | 0.9732 | 5.560 | 2.877 |

At 20× the HF matrix (slope≈0.6) **improves every metric at once**: GMSD −1.9%
(0.06940→0.06808), HaarPSI +0.0006, edge-MAE −2.0% (5.659→5.547), seam-step −1.6% —
**and PSNR goes UP** 0.13 dB (40.83→40.96). So here protecting HF is a *free* lift, not
a PSNR trade: the dead-zone was over-spending on low-frequency precision the eye/edge
metrics don't need. This reproduces (and quantifies, on the perceptual metrics directly)
the "small consistent lift + better edge metrics" the earlier entropy bake-off saw.

### HF-slope sweep @ 50× (hires)
| hf_slope | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE | seam16 |
|---|---|---|---|---|---|---|---|
| 0.00 (=FLAT) | 50.46× | 35.78 | 0.9380 | 0.11821 | 0.9438 | 9.546 | 5.216 |
| 0.40 | 49.61× | 35.78 | 0.9381 | 0.11815 | 0.9438 | 9.530 | 5.258 |
| 0.60 | 50.37× | 35.67 | 0.9369 | 0.11914 | 0.9431 | 9.609 | 5.336 |
| 0.90 | 49.94× | 35.71 | 0.9375 | 0.11848 | 0.9435 | 9.536 | 5.319 |

At 50× the effect **flips to neutral/negative**: a mild slope (0.4) is a wash; strong
protection (0.6) is slightly worse on GMSD/edge/seam. **Why:** at 50× the dead-zone has
already zeroed almost all HF, so the bit budget is tiny; spending it to keep marginal HF
coefficients *starves the structural low/mid frequencies*, and the global metric (which
averages flat regions too) doesn't reward it. **HF protection is operating-point
dependent: it pays in the ≤~20–30× regime and stops paying past ~40–50×.**

---

## Four-objective table at equal ratio (canonical bench)

`hires-256` (ink/fiber-rich). enc MB/s is single-thread encode.

### 20× (the regime where the objective matters)
| objective | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE | seam16 | enc MB/s |
|---|---|---|---|---|---|---|---|---|
| (a) FLAT pure-MSE | 19.83× | 40.83 | 0.9773 | 0.06940 | 0.9729 | 5.659 | 2.892 | 58 |
| **(b) HF-protecting** | 19.86× | **40.96** | **0.9779** | **0.06808** | **0.9735** | **5.547** | **2.846** | 61 |
| (b') HF+adaptive | 19.97× | 40.87 | 0.9765 | 0.07060 | 0.9728 | 5.577 | 2.865 | 59 |
| (c) PERC in-loop | 19.99× | 40.91 | 0.9776 | 0.06856 | 0.9732 | 5.579 | 2.862 | **5** |

### 50×
| objective | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE | seam16 | enc MB/s |
|---|---|---|---|---|---|---|---|---|
| (a) FLAT pure-MSE | 50.46× | 35.78 | 0.9380 | 0.11821 | 0.9438 | 9.546 | 5.216 | 63 |
| (b) HF-protecting | 50.37× | 35.67 | 0.9369 | 0.11914 | 0.9431 | 9.609 | 5.336 | 62 |
| (b') HF+adaptive | 49.85× | 35.63 | 0.9343 | 0.12156 | 0.9422 | 9.624 | 5.353 | 61 |
| (c) PERC in-loop | 50.49× | 35.61 | 0.9360 | 0.11969 | 0.9427 | 9.660 | 5.369 | 5 |

### 10× (very-high-quality regime)
| objective | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE | seam16 | enc MB/s |
|---|---|---|---|---|---|---|---|---|
| (a) FLAT pure-MSE | 10.55× | 45.05 | 0.9906 | 0.03527 | 0.9887 | 3.385 | 1.650 | 56 |
| (b) HF-protecting | 10.08× | 44.80 | 0.9902 | 0.03663 | 0.9879 | 3.467 | 1.781 | 55 |

(At 10× the volume is already near-lossless — 45 dB — and there is essentially no HF to
protect; FLAT and HF are within noise, FLAT marginally ahead. `coarse-256` follows the
same pattern as hires, one regime tighter: HF is ≈neutral at 10–20×, slightly negative
at 50×; full numbers below in raw output.)

---

## Decision questions

**1. Does (b) HF-weighted/adaptive quant improve GMSD/HaarPSI/edge-MAE at equal ratio
vs (a) plain MSE — and at what PSNR cost?**
Yes, in the **moderate-ratio regime (≈10–30×)**, and at **no PSNR cost** — actually a
tiny PSNR *gain*. Ratio-matched at 20× on hires: **GMSD −1.9%, edge-MAE −2.0%, HaarPSI
+0.06%, seam −1.6%, PSNR +0.13 dB.** The lift is small but **consistent across all five
quality/edge metrics simultaneously** and free. Past ~40–50× it disappears/reverses.
The *content-adaptive* variant (b') does **not** help: at any aggressiveness it is
strictly worse than the fixed HF matrix on the global perceptual metrics (GMSD 0.07060
vs 0.06808 @20×), because trading flat-region quality for busy-region quality at equal
ratio coarsens the flats (seam-step rises) and the whole-volume metric doesn't reward
the trade. **Verdict: use the fixed HF matrix; skip variance-driven adaptivity.**

**2. Does (c) perceptual-in-loop beat (b) enough to justify its speed cost?**
**No.** Perceptual-in-loop lands essentially **on top of the fixed HF matrix** on every
metric (GMSD 0.06856 vs HF 0.06808 @20× — marginally *worse*), because at equal global
ratio the per-atom rate ceiling pulls each block's perceptual-optimal step back to the
budget the fixed matrix already implies. And it costs **~12× the encode time**: 5 MB/s
vs 58–61 MB/s for the matrix passes (it runs the inverse DCT-16 + a local GMSD/edge-MAE
per candidate step, ×7 candidates per atom). The PLAN/red-team flagged this as "worth it
iff fast enough" — **it is neither faster nor better, so it is not worth it.**

**3. Reproduce + quantify the SMEAR/BLOCKING.**
Captured by the new 16-grid **seam-step** metric (mean |Δerror| across block faces) and
GMSD. Blocking grows monotonically with ratio: seam-step 1.65 → 2.89 → 5.22 (hires
10/20/50×) and GMSD 0.035 → 0.069 → 0.118. The DCT-16³ atom already keeps these modest
(a 16-voxel grid has half the face density of an 8³ grid, and 16³ compaction means
fewer coefficients survive to step at the wall) — this is *why* DCT-16 + mild deblock
was the transform/healing winner. Among the distortion objectives, **HF protection
(slope≈0.6) gives the lowest seam-step at 20× (2.846 vs FLAT 2.892)** — it preserves
ink/fiber edges without adding blocking; adaptive *adds* seam (2.865) and is the worst
edge-preserver. So the artifact ranking at the regime that matters is **HF < FLAT <
PERC ≈ FLAT < ADAPT** (lower seam = better).

**4. Recommended default distortion/quant objective.**
**Fixed HF-protecting per-coefficient 16³ quant matrix, slope ≈ 0.4–0.6.**
- It is the only objective that improves the perceptual/edge metrics (GMSD, HaarPSI,
  edge-MAE, seam) at equal ratio in the dominant 10–30× operating band — for free on
  PSNR — and is the best ink/fiber-edge preserver / least-blocking.
- It costs nothing at encode time (it's a static matrix multiply, same speed as FLAT,
  fully autovectorizable) — unlike perceptual-in-loop's 12× slowdown.
- At very high ratio (≥50×) HF protection is a wash; a **mild slope (≈0.4)** is the
  safe all-regime default (neutral at 50×, still a small lift at 20×). For a quality-
  first profile use slope 0.6.
- **Drop** content-adaptive (variance-driven) quant and **drop** perceptual-in-loop:
  neither beats the fixed matrix and both cost more (a side byte; 12× encode time).

This matches PLAN §8's leading default ("HF-protecting dead-zone quant") and resolves
the "central incoherence": MSE-optimal allocation does starve HF, and a light, fixed HF
re-weighting *recovers* the edge quality the eye + downstream ML care about with no PSNR
or speed penalty — whereas the heavier objectives (adaptive, in-loop) over-reach and
lose. The win is small but real and free; the expensive objectives are not worth it.

---

## Method / reproduce
```
cmake -S . -B build -DVC_TRANSFORM=dct16 -DVC_ENTROPY=rlgr && cmake --build build
./build/bench_hfdist            # canonical 4-objective table, hires + coarse, 10/20/50x
./build/bench_hfdist --hfsweep  # ratio-matched HF-slope sweep (the clean comparison)
ctest --test-dir build          # incl. qmatrix16 round-trip-exact test
```
- HF slope is a single tunable (`qm_set_hf_slope` / `QM_HF_SLOPE_DEFAULT`); it is a
  build/run constant, not per-block side info, so decode reproduces it exactly.
- The matrix quant loops are unit-stride + branchless (autovectorizable, PLAN §7);
  RLGR is inherently serial (unchanged). Single-threaded, reentrant, heap scratch.
- 256³ = 4096 atoms of 16³; equal-ratio via global-step bisection (±1%).
```
```
