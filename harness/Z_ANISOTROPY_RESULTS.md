# Z-directional anisotropy experiment (branch r2-z-anisotropy)

Self-contained throwaway bench `harness/bench_zaniso.c` (skeleton copied from
`bench_hfdist.c`): integer DCT-16^3 atom + per-coefficient quant matrix + RLGR
(measurable baseline, per task) run end-to-end at EQUAL RATIO on real PHerc
Paris 4 hires-256 + coarse-256. Baseline = the won isotropic HF-protecting 16^3
quant matrix (slope 0.5). No codec.c / ratectrl coupling.

## 0. Axis verification (harness/axis_corr.c) — REQUIRED FIRST STEP

Volume is ZYX, **Z = axis0 = outermost index = scroll page top->bottom** (per
fetch_corpus.py: coarse zarr shape (4071,2264,2264) Z,Y,X; hires 128^3 chunks).
Lag-1 Pearson correlation of adjacent voxels per axis:

| volume     | Z (axis0) rho | Y (axis1) rho | X (axis2) rho | mean\|d\| Z / Y / X |
|------------|---------------|---------------|---------------|---------------------|
| hires-256  | **0.99089**   | 0.97279       | 0.98883       | 3.34 / 6.07 / 3.84  |
| coarse-256 | 0.90650       | 0.92146       | 0.89553       | 10.3 / 9.8 / 10.8   |

**Finding: the user's domain insight is CONFIRMED on hires** — Z (page
top->bottom) is the single most-correlated axis (rho 0.991, smallest mean
|delta|). X is a close second; **Y is the weakest** axis on hires. On coarse the
anisotropy is mild and the ordering flips (Y slightly strongest) — different
voxel size/orientation, so Z-anisotropy is a hires-only signal.

## 1. (a) Anisotropic quant matrix

Per-axis weight `w = 1 - slope*(az*fz+ay*fy+ax*fx)/((az+ay+ax)*15)`; isotropic
baseline = az=ay=ax=1. Small az = protect Z freqs (finer step), large ay/ax =
coarsen Y/X. Zero side info — just a different fixed matrix. slope=0.5.

Equal-ratio deltas vs isotropic HF baseline (best aniso = Zprot .3/1.5/1.5):

| target | vol    | PSNR base→aniso | ΔPSNR | edgeMAE base→aniso | GMSD base→aniso  |
|--------|--------|-----------------|-------|--------------------|------------------|
| 16x    | hires  | 42.26 → 42.31   | +0.05 | 4.773 → 4.736 (-0.8%) | 0.05676→0.05620 (-1.0%) |
| 32x    | hires  | 38.25 → 38.27   | +0.02 | 7.431 → 7.402 (-0.4%) | 0.09363→0.09328 (-0.4%) |
| 64x    | hires  | 34.44 → 34.46   | +0.02 | 10.81 → 10.78 (-0.3%) | 0.13149→0.13126 (-0.2%) |
| 16x    | coarse | 32.28 → 32.35   | +0.07 | 15.12 → 14.98 (-0.9%) | 0.10634→0.10546 (-0.8%) |
| 64x    | coarse | 25.43 → 25.43   | 0.00  | 30.66 → 30.66 (0%)    | ~0%              |

- **Direction is correct**: protecting Z + coarsening Y/X helps; the *opposite*
  config "Zcoarse 2/1/1" is consistently the WORST row at every point (e.g.
  coarse-16x PSNR 32.20 vs baseline 32.28). So the physics is real.
- **Magnitude is negligible**: best case +0.07 dB / -1% edgeMAE, mostly +0.02 dB.
  All within run-to-run / metric noise. The DCT-16 already decorrelates each axis
  independently, so a per-axis frequency reweighting buys almost nothing on top.
- Cost: free (no side info, no speed cost: ~50 MB/s same as baseline). Does NOT
  break random access (still a fixed per-coef matrix, touched=1 holds).

## 2. (b) Z-only inter-atom DC prediction

Predict an atom's DC (per-atom mean) from its Z-neighbor atom (az-1) DC; code the
residual (1 byte if |resid|<64 else 2) instead of raw i16. Z-only, NOT 6/18/26-conn.

- DC stream halves: 8 KB → 4 KB per 256^3 volume.
- But DC is **~0.025% of the total stream** (4 KB saved / 16 MB raw, ratios 8–64x).
  Equal-ratio PSNR effect is +0.00–0.06 dB (biggest at coarse-64x: 25.43→25.49),
  i.e. freeing 4 KB barely relaxes the step.
- **Cost: breaks per-atom random access.** Decoding any atom now requires its
  Z-neighbor's DC first (serial az dependency); `touched=1` no longer holds for
  the DC term. Encode also ~10–40 MB/s slower (Z-major serialization).
- The bigger low-freq Z-column prediction (fz,0,0 band) was not pursued: the DC
  result already shows the addressable budget (the LF Z band) is tiny and the
  random-access cost is the same.

## 3. Decision

**PARK.** The user's domain insight is empirically TRUE (Z is the most-correlated
axis on hires; protecting Z / coarsening Y is directionally the right move — the
opposite config always loses). But after the won DCT-16 + HF matrix, the
*exploitable* anisotropy is already captured by the transform: the best
zero-cost anisotropic matrix yields only **+0.02 to +0.07 dB and <=1% edgeMAE**
at equal ratio, and the Z-only DC predictor saves 0.025% of the stream while
breaking random access. Neither clears the bar for integration.

PARK (not NIX) because: the anisotropic matrix is FREE and directionally correct,
so it's a harmless default knob to keep on the shelf if a future transform change
re-exposes axis structure; and the axis-correlation diagnostic itself is reusable.

### Decision numbers
- Ratio at quality: anisotropic matrix gives **≈0% ratio change** at matched
  quality (it IS matched-ratio; the gain shows as +0.02–0.07 dB / ≤1% edgeMAE).
- dB: best **+0.07 dB** (coarse-16x), typical **+0.02 dB** (hires).
- Speed: anisotropic matrix ~50 MB/s encode = baseline (no cost). DC predictor
  ~10–40 MB/s slower.
- Random access: anisotropic matrix — **preserved** (touched=1 holds). Z-only DC
  predictor — **BROKEN** (serial az-1 dependency).

## Repro
```
cmake --build build --target bench_zaniso
gcc -O2 -std=c23 harness/axis_corr.c -o build/axis_corr -lm && ./build/axis_corr
./build/bench_zaniso        # hires-256 then coarse-256, targets 8/16/32/64x
```
Pre-existing `ratectrl` ctest failure is unrelated to this branch (identical on
base commit 70d0871 / sibling worktrees); roundtrip/transforms/chunkmodel/
qmatrix16 all green.
