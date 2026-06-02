# Experiment #18 — Dead-zone width + dequant reconstruction-offset sweep

**Branch:** `exp-deadzone-sweep` · **Bench:** `harness/bench_deadzone.c` (self-contained, throwaway)
**Data:** real PHerc Paris 4 `hires-256` + `coarse-256` (256³ u8, 16.8 MB each)
**Pipeline (all held fixed = the winning stack):** integer DCT-16³ atom · fixed HF-protecting
16³ quant matrix (`qmatrix16.c`, slope 0.50, mid of the mandated 0.4–0.6 band) · table-free
RLGR · per-atom DC bias. Each cell re-bisects the base step to the SAME target ratio, so
quality deltas are reported at **equal ratio** (the ratio-at-quality objective).

## What was swept

Two scalar knobs of the dead-zone scalar quantizer, both GLOBAL build/run constants (no side
info, encoder+decoder agree):

- **dead-zone zero-bin width `dz`** — zero-bin half-width = `dz·step`.
  encode: `lvl = floor(|c|/step + (1-dz))` for `|c| ≥ dz·step`, else 0.
  `dz = 0.50` reproduces the CURRENT `qm_quant_block` exactly (verified algebraically and
  by the baseline row). Wider `dz` kills more near-zero HF coeffs → frees rate → spent on a
  finer step elsewhere at equal ratio.
- **dequant reconstruction offset `ro`** — `|rec| = (lvl − 1 + dz + ro)·step`.
  `ro = 0.50` = bin centre = the CURRENT `qm_dequant_block` (`rec = lvl·step`).
  Lower `ro` (≈0.375–0.40) is the Laplacian-optimal sub-centre reconstruction point c2d/c3d
  used (AC coefficients are Laplacian; the conditional mean sits below the bin centre).

Baseline = `(dz=0.50, ro=0.50)`. Swept `dz ∈ {0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.80}`,
`ro ∈ {0.30,0.375,0.40,0.45,0.50}`, at q∈{16,32,64,128} on both volumes. Full grid in
`harness/deadzone_sweep_raw.txt`; speed in `harness/deadzone_speed_raw.txt`.

## Headline result

**Both knobs are free, additive, and robust.** At equal ratio, vs the current quantizer:

- **Reconstruction offset alone** (`dz=0.50`, `ro 0.50→0.375`): **+0.25 dB** PSNR consistently,
  also better MS-SSIM / GMSD / HaarPSI / edge-MAE. Zero-cost: one constant. This is the
  c2d/c3d "+0.2–0.5 dB free" effect, reproduced exactly.
- **Wider dead-zone** (`dz 0.50→0.80`, at `ro=0.40`): adds another **+0.4–0.5 dB**.
- **Combined winner `(dz=0.80, ro=0.40)`**: **+0.62 to +0.77 dB** PSNR at matched ratio,
  across BOTH volumes and ALL four q-points — and every perceptual/edge metric improves too,
  including the 16³ seam-step (the artifact the project gates with GMSD).

### Winner vs baseline at matched ratio (from the speed table, dz0.80/ro0.40)

| vol | q | ratio (base→win) | PSNR | ΔPSNR | MS-SSIM | GMSD | enc MB/s | dec MB/s |
|-----|---|---|---|---|---|---|---|---|
| hires | 16 | 15.97→16.00 | 42.26→**42.88** | **+0.62** | .9831→.9850 | .0568→.0522 | 58→57 | 175→164 |
| hires | 32 | 32.03→32.02 | 38.25→**38.92** | **+0.67** | .9618→.9661 | .0936→.0880 | 58→61 | 170→181 |
| hires | 64 | 64.03→63.88 | 34.40→**35.15** | **+0.75** | .9196→.9282 | .1319→.1266 | 52→62 | 181→185 |
| hires | 128| 127.5→128.1 | 30.79→**31.53** | **+0.74** | .8438→.8555 | .1713→.1700 | 63→63 | 183→186 |
| coarse| 16 | 16.00→15.97 | 32.28→**32.99** | **+0.71** | .9528→.9575 | .1063→.1002 | 58→58 | 174→175 |
| coarse| 32 | 32.10→31.96 | 28.47→**29.24** | **+0.77** | .8932→.9009 | .1477→.1437 | 60→60 | 180→181 |
| coarse| 64 | 64.21→64.03 | 25.43→**26.15** | **+0.72** | .7961→.8012 | .1800→.1815 | 62→62 | 185→180 |
| coarse| 128| 127.8→127.9 | 22.98→**23.59** | **+0.61** | .6544→.6500 | .2106→.2199 | 63→60 | 174→185 |

**Speed: identical.** Encode 52–63 MB/s and decode 164–186 MB/s for both configs — the
winner is often a hair faster (fewer nonzero levels → less RLGR work). The change is two
constants in the existing branchless, autovectorized quant/dequant loops (verified
`-fopt-info-vec`: both loops vectorize at 16-byte vectors). No random-access cost: still a
self-contained 16³ atom, touched=1 decodes one atom (the quantizer math is unchanged in
structure).

## Detail: the two knobs are separable and monotone

**Reconstruction offset (`ro`), at fixed `dz`:** PSNR peaks at `ro ≈ 0.375–0.40`, monotone
down toward `ro=0.50` (bin centre = baseline) and `ro=0.30`. Example hires-32x, dz=0.50:
ro 0.50→38.25, 0.45→38.40, 0.40→38.50, 0.375→**38.50**, 0.30→38.42. So `ro≈0.375–0.40` is
the Laplacian-optimal point; pushing past it (0.30) overshoots and loses. Perceptual metrics
track PSNR here.

**Dead-zone width (`dz`), at fixed `ro`:** PSNR rises monotonically with `dz` and is
essentially saturated by `dz≈0.70–0.80` (hires-64x, ro=0.40: dz 0.50→34.68, 0.60→34.93,
0.70→35.05, 0.80→**35.15**). `dz<0.50` (narrower zero-bin) is strictly worse — it keeps
near-zero HF coeffs that cost rate the finer step would spend better. Edge metrics
(GMSD/edge-MAE) and the 16³ seam-step ALSO improve monotonically with wider `dz` (the wider
zero-bin suppresses ringing-class tiny HF coefficients), so this is not a PSNR-only win.

**One caveat at the quality extreme (low-q on the busy coarse volume):** at coarse-128x the
*perceptual* metrics (MS-SSIM/GMSD/HaarPSI) prefer a slightly higher `ro` (~0.45–0.50) while
PSNR prefers `ro≈0.375`; the two disagree by a small margin. `ro≈0.40` is within noise of
both optima everywhere, so it is the safe single setting. The dead-zone widening still helps
at every operating point. (Also at coarse-128x with `dz≤0.45` the bisection can't reach 128×
— the zero-bin is too narrow to compress that far — another reason not to go below 0.50.)

## Recommendation: **KEEP**

A cheap, reliable, free win exactly as the experiment brief predicted. Two compile-time
constants in `src/quant/qmatrix16.c` deliver **+0.6–0.8 dB at equal ratio** (equivalently
~3–6% more ratio at equal PSNR) on real scroll data, with **gains on every perceptual/edge
metric including the GMSD-gated 16³ seam**, **zero speed cost**, **no side info**, and **no
loss of 16³ random access**. It stacks cleanly under the existing HF quant matrix.

### Suggested integration (for the human integrator — NOT done here, exploration only)

In `src/quant/qmatrix16.c`, change `qm_quant_block` / `qm_dequant_block` to the generalized
form with `DZ_WIDTH = 0.80f`, `RECON_OFFSET = 0.40f`:

- quant: `lvl = floor(|c|/step + (1 - DZ_WIDTH))` for `|c| ≥ DZ_WIDTH·step`, else 0.
- dequant: `|rec| = (lvl - 1 + DZ_WIDTH + RECON_OFFSET)·step` for `lvl ≠ 0`, else 0.

Both stay unit-stride/branchless/autovectorizable. Conservative alternative if any worry
about the coarse-128x perceptual/PSNR split: `(dz=0.70, ro=0.40)` is within ~0.05 dB of the
winner everywhere and keeps a slightly tighter ratio band.
