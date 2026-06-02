# Learned (scroll-trained) static quant table vs generic hand-tuned matrix

Branch: `r2-learned-tables` (worktree, explore-only, NOT merged).
Date: 2026-06-02. Data: real PHerc Paris 4 `hires_256.u8` + `coarse_256.u8` (256³ u8).
Bench: `harness/bench_learned_tables.c` (throwaway, self-contained: won DCT-16³ +
won RLGR + metric bundle; does NOT touch codec.c / ratectrl / chunkmodel).

## Question (PLAN §6 parked-but-look)
Constraint: learned components AS COMPILE-TIME STATIC TABLES ONLY — no per-voxel
neural inference. Does a 16³ quant matrix TRAINED offline on PHerc Paris 4
coefficient statistics, baked as a static `const float[4096]` table (identical
hot-path cost to the generic matrix), beat the GENERIC hand-tuned HF-protecting
matrix (`src/quant/qmatrix16.c`, slope 0.60) at equal ratio? Watch for overfit.

## Method
- Pipeline atom = 16³ (won transform + random-access unit). Per-atom DC removed,
  integer DCT-16³, per-coefficient dead-zone quant matrix, RLGR. Exactly the
  `bench_hfdist.c` recipe.
- LEARNED table: collect mean-|coef| per coefficient position over ALL atoms of a
  training volume → per-position weight `w_i = clamp((E_i/E_p90)^(-p), wmin, 1)`
  (energy-aware static allocation; DC forced to weight 1). p, wmin baked once.
  This is the proper variance-aware static allocation learned from THIS data
  instead of the hand-drawn `s=u+v+w` line.
- Baselines: FLAT (pure-MSE flat matrix) and HF generic (slope 0.60, the stack
  default). All three driven to the SAME target ratio by bisecting the base step.
- Cross-validation for overfit: train hires→test coarse, and train coarse→test
  hires.
- Metrics at matched ratio: PSNR, MS-SSIM, GMSD, HaarPSI, edge-MAE, seam16.

## Headline result (default learned table p=0.50, wmin=0.30)

In-domain, train hires / test HIRES:

| target | matrix   | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE |
|--------|----------|-------|------|---------|------|---------|---------|
| 20x | FLAT       | 19.91 | 40.83 | 0.9772 | 0.06954 | 0.9728 | 5.655 |
| 20x | HF generic | 19.86 | **40.96** | **0.9779** | **0.06808** | **0.9735** | **5.547** |
| 20x | LEARNED    | 19.93 | 40.65 | 0.9769 | 0.07157 | 0.9722 | 5.905 |
| 50x | FLAT       | 49.07 | 35.79 | 0.9382 | 0.11807 | 0.9439 | 9.539 |
| 50x | HF generic | 50.37 | 35.67 | 0.9369 | 0.11914 | 0.9431 | 9.609 |
| 50x | LEARNED    | 49.99 | 35.74 | 0.9380 | 0.11901 | 0.9432 | 9.794 |

**The learned table LOSES to the generic HF matrix (and often to FLAT) on every
metric at every ratio, in-domain.** It is never better than ~noise; usually worse
on edge metrics (GMSD/edgeMAE), which are the ones that matter for ink/fiber.

## Overfitting: severe + breaks rate control cross-domain

Train hires / test COARSE @ target 100x: the learned table could not even reach
the target — bisection stuck at **60.71x** (vs FLAT/HF hitting 99–100x), PSNR
25.65. Train coarse / test COARSE @ 100x stuck at **67.16x**. The aggressive
energy weighting makes payload rate nearly insensitive to the base step in the
high-q regime → rate control cannot hit aggressive ratios. This is an overfit
pathology: a table tuned to one volume's energy profile misallocates on another
(and even on the same volume at a ratio the training didn't emphasize), and it
poisons the Lagrangian step search. It does NOT break per-atom random access
(every 16³ still decodes standalone, touched=1 holds — the table is global
static), but it breaks the ratio knob.

## Robustness: p-sweep (train hires / test hires @ 50x) — can ANY setting win?

| p | wmin | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE |
|------|------|-------|------|---------|------|---------|---------|
| HF generic | – | 50.37 | 35.67 | 0.9369 | 0.11914 | 0.9431 | 9.609 |
| -0.50 | 0.20 | 50.40 | 35.79 | 0.9381 | 0.11811 | 0.9438 | 9.536 |
| -0.25 | 0.20 | 49.07 | 35.79 | 0.9382 | 0.11807 | 0.9439 | 9.538 |
| +0.25 | 0.40 | 49.93 | 35.76 | 0.9377 | 0.11896 | 0.9431 | 9.730 |
| +0.50 | 0.40 | 50.17 | 35.81 | 0.9388 | 0.11806 | 0.9439 | 9.650 |
| +1.00 | 0.20 | 49.79 | 35.78 | 0.9388 | 0.11794 | 0.9439 | 9.701 |

The only learned settings that don't lose are the ones whose weights collapse
toward 1.0 — i.e. they rediscover FLAT (PSNR ~35.79, GMSD ~0.118 = the FLAT row).
The best "learned" outcome is within ~0.1 dB / 0.001 GMSD of FLAT/HF, inside
bisection noise. There is NO learned setting that delivers a real, consistent
ratio-at-quality gain over the generic matrix.

## Why it doesn't help (interpretation)
- The scroll's 16³ DCT energy spectrum is a smooth, near-monotonic falloff with
  3D frequency — almost exactly the shape the generic `1 - slope·(u+v+w)/45`
  curve already encodes. There is no scroll-specific anisotropy or band structure
  for a learned table to exploit beyond the hand-tuned line.
- Allocating FINER steps to high-energy (low-freq) coefficients (p>0) spends bits
  where the signal is already cheap and well-preserved, starving the HF/edge
  bands → worse GMSD/edgeMAE. The opposite sign just converges to FLAT.
- The HF-protecting generic matrix was already tuned (slope sweep in
  HFDIST_RESULTS.md) on this same scroll, so "generic" here is in practice
  already scroll-aware in the way that matters; an automatic table cannot beat the
  hand-tuned operating point and risks overfitting one volume's exact spectrum.

## Decision numbers
- Ratio-at-quality gain from scroll-tuned static table: **none** (≤0.1 dB,
  ≤0.001 GMSD, both inside bisection noise; edge metrics typically WORSE).
- Overfit cost: cross-domain it cannot reach high target ratios (100x → 60–67x),
  i.e. it actively breaks Lagrangian rate control.
- Hot-path cost if shipped: zero extra (static table multiply, same as generic).
- Random access: unaffected — touched=1 preserved (global static table).
- Encode/decode speed: identical to the generic-matrix path (same kernel; the
  training pass is offline/one-time, not in the codec).

## Verdict: NIX (PARK the idea, do not ship)
A scroll-tuned STATIC quant table is NOT worth it vs the generic hand-tuned
HF matrix. It never wins at quality, it overfits one scroll, and it breaks the
ratio knob in the high-compression regime. The generic HF matrix (slope 0.60),
itself already tuned on this scroll, is the right default. Keep the door open
only if a future, larger, multi-scroll corpus reveals band structure the
monotonic curve misses — but on PHerc Paris 4 there is nothing to learn.
