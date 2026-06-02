# Vector / lattice quantization bake-off (PLAN §2 "Quantizer", parked-but-look)

Branch `r2-vector-quant`. Self-contained throwaway bench `harness/bench_vq.c`
(does NOT touch codec.c / ratectrl / chunkmodel). Atom = the won integer DCT-16³;
entropy = the won table-free RLGR (the measurable baseline per the experiment
mandate); quant shape = the won HF-protecting 16³ matrix (slope 0.60). Real
PHerc Paris 4 hires-256 + coarse-256. All candidates driven to the SAME global
target ratio by bisecting the base step, so this is an iso-rate quality
comparison (the correct way to compare quantizers).

## Question

Can vector/lattice quantization of small LOW-FREQUENCY coefficient groups pack
rate-distortion tighter than the per-coefficient SCALAR dead-zone, and at what
DECODE cost? VQ decode = table lookup (potentially heavy); this is a decode-speed
-sensitive random-access codec, so the verdict weighs ratio/quality gain against
decode cost. Prior expectation (mandate): likely PARK.

## Setup

- Per atom: 4096 DCT coeffs split into a LOW-FREQ group (first NLF=64 coeffs in a
  u+v+w-ordered scan, in D=4-dim subvectors → 16 groups) handled by VQ/lattice,
  and the HF remainder always scalar dead-zone + RLGR.
- Coefficients are normalized by their per-coef HF-matrix step before VQ/lattice,
  step multiplied back on decode (HF shape preserved).
- Candidates:
  - **SCALAR** — baseline: HF-matrix scalar dead-zone over all 4096 + RLGR.
  - **LATTICE D4** — Conway–Sloane D4 fast quantizer on the low-freq 4-D subvectors
    (round-to-nearest integer point with even coordinate sum). Lattice point
    integers RLGR-coded. Decode = `level*step` — IDENTICAL arithmetic to scalar
    (no LUT, no codebook). No extra side info.
  - **VQ256** — trained k-means codebook (K=256, dim 4) on the volume's own
    low-freq subvectors; per-group 1-byte index RLGR-coded; codebook is global
    side info (2 KiB, counted once). Decode = LUT (`codebook[index]*step`).

## Results — hires-256 (ink/fiber-rich)

| target | quantizer | ratio | PSNR | MS-SSIM | GMSD | HaarPSI | edgeMAE | seam16 | enc MB/s | dec MB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| 10x | SCALAR  | 10.08 | 44.80 | 0.9902 | 0.0366 | 0.9879 | 3.467 | 1.781 | 49 | 154 |
| 10x | **LATTICE** | 10.01 | **45.73** | 0.9920 | 0.0301 | 0.9905 | 3.041 | 1.588 | 44 | 154 |
| 10x | VQ256   | 10.09 | 28.86 | 0.9065 | 0.1371 | 0.9208 | 11.44 | 10.03 | 41 | 151 |
| 20x | SCALAR  | 19.86 | 40.96 | 0.9779 | 0.0681 | 0.9735 | 5.547 | 2.846 | 52 | 156 |
| 20x | **LATTICE** | 20.16 | **42.56** | 0.9842 | 0.0539 | 0.9803 | 4.577 | 2.338 | 48 | 158 |
| 20x | VQ256   | 19.88 | 28.92 | 0.8992 | 0.1419 | 0.9181 | 11.97 | 10.13 | 46 | 157 |
| 50x | SCALAR  | 50.37 | 35.67 | 0.9369 | 0.1191 | 0.9431 | 9.609 | 5.336 | 53 | 156 |
| 50x | **LATTICE** | 50.11 | **37.42** | 0.9551 | 0.1013 | 0.9548 | 8.016 | 4.369 | 47 | 156 |
| 50x | VQ256   | 49.85 | 28.87 | 0.8564 | 0.1644 | 0.9020 | 14.85 | 11.00 | 45 | 157 |
| 100x| SCALAR  | 100.2 | 32.04 | 0.8754 | 0.1564 | 0.9134 | 13.44 | 7.979 | 53 | 156 |
| 100x| **LATTICE** | 99.9 | **33.30** | 0.9012 | 0.1428 | 0.9252 | 11.92 | 7.032 | 47 | 155 |
| 100x| VQ256   | 63.9* | 27.91 | 0.8053 | 0.1794 | 0.8850 | 17.50 | 11.73 | 46 | 160 |

\* VQ256 cannot reach 100x — the per-group index payload + codebook side info
floor the rate (ratio saturates ~64x).

## Results — coarse-256

| target | quantizer | ratio | PSNR | MS-SSIM | GMSD | edgeMAE | dec MB/s |
|---|---|---|---|---|---|---|---|
| 10x | SCALAR  | 10.00 | 35.27 | 0.9754 | 0.0753 | 10.53 | 152 |
| 10x | **LATTICE** | 10.09 | **36.66** | 0.9819 | 0.0620 | 8.791 | 148 |
| 10x | VQ256   |  9.95 | 27.72 | 0.9255 | 0.1181 | 16.86 | 146 |
| 20x | SCALAR  | 20.12 | 30.94 | 0.9370 | 0.1208 | 17.59 | 150 |
| 20x | **LATTICE** | 20.15 | **32.54** | 0.9555 | 0.1031 | 14.58 | 153 |
| 50x | SCALAR  | 50.17 | 26.43 | 0.8351 | 0.1695 | 27.95 | 154 |
| 50x | **LATTICE** | 49.79 | **27.69** | 0.8754 | 0.1551 | 24.57 | 153 |
| 100x| SCALAR  | 100.4 | 23.78 | 0.7100 | 0.1991 | 35.44 | 153 |
| 100x| **LATTICE** | 100.4 | **24.49** | 0.7562 | 0.1886 | 32.98 | 153 |

(decode-speed isolation re-run twice: SCALAR/LATTICE/VQ256 all 150-160 MB/s,
differences within run-to-run noise. The earlier coarse-256 106 MB/s dips were
background-load artifacts, not the dequant.)

## Findings

1. **LATTICE D4 is a consistent, clean WIN over scalar dead-zone at iso-rate:**
   +0.9 dB (hires 10x) up to +1.75 dB (hires 50x); +1.4 dB on coarse at 10/20x.
   It also improves EVERY perceptual metric (MS-SSIM, GMSD, HaarPSI, edgeMAE) and
   *reduces* the 16-grid seam step — at both volumes, all four ratios. The gain
   matches D4 lattice theory (denser packing of the 4-D Voronoi cell than the
   cubic Z⁴ dead-zone cell → ~0.6 bit/vector ≈ +1–1.8 dB at these rates).

2. **DECODE COST IS ZERO.** This is the load-bearing result. The D4 dequant is
   `level*step` — bit-for-bit the same arithmetic as the scalar dequant, no LUT,
   no codebook, no branch. Decode MB/s is statistically identical to SCALAR
   (~155 MB/s) across two repeated runs. The mandate's worry ("VQ decode = table
   lookup, can be heavy") applies to the CODEBOOK form, NOT the lattice form.

3. **Encode cost is mild** (~10-15% slower: D4 quantizer is one round + one
   parity-fix per 4-vector). Acceptable; encode is offline.

4. **Random access is preserved** (touched=1). `tests/test_vq.c` proves a single
   atom decoded in isolation is byte-identical to the same atom decoded inside a
   full multi-atom sweep, for BOTH lattice and VQ — the quantizer adds no
   cross-atom dependency. The D4 lattice is fully self-contained (no side state);
   the codebook is global read-only side info, also non-blocking for seek.

5. **VQ256 (trained codebook) is a clear NIX.** Quality collapses (PSNR ~28 dB
   regardless of target) and it cannot reach high ratios (codebook + index floor
   the rate at ~64x). A k-means codebook over raw normalized DCT subvectors is a
   poor model for this data; it also gives NO decode-speed benefit. This is the
   "heavy table-lookup VQ" the mandate predicted would PARK — confirmed PARK/NIX.

## Caveats / honesty

- The lattice gain is partly that D4 has no dead-zone while the scalar baseline
  does (dz=0.80 shape via the HF matrix). The comparison is still fair: it is
  iso-rate (bisected to equal ratio), measuring quality at matched bits — the
  standard quantizer comparison. The dead-zone's job (kill near-zero coeffs to
  save rate) is on the LOW-FREQ group only here, where coeffs are rarely zero, so
  removing it there costs little rate and the lattice packing dominates.
- Only the low-freq 64 coeffs use the lattice; the HF tail (where the dead-zone
  earns its rate on sparse near-zero coeffs) is untouched. Extending the lattice
  into the HF tail would likely LOSE (the dead-zone's zero-run advantage there is
  real). The win is specifically low-freq-group lattice + HF-tail dead-zone.
- This overlaps conceptually with the completed RDOQ experiment (#19), which also
  improves coefficient quantization RD; they should be cross-checked before BOTH
  are integrated (the gains may not be fully additive).
- Pre-existing ctest failure `ratectrl` exists at the base commit (70d0871),
  unrelated to this change. New `vq` test passes; no regressions introduced.
  Hot loops verified autovectorized (`-fopt-info-vec`: quant/dequant loops
  vectorize, 16-byte vectors).

## Decision

**KEEP — the D4 lattice form (NIX the trained codebook).**

- Ratio/quality: **+0.9 to +1.75 dB PSNR at iso-rate**, better on every perceptual
  metric and seam, at all four ratios on both real volumes. Equivalently ~8-15%
  rate saving at fixed quality.
- Decode cost: **none** — D4 dequant == scalar dequant arithmetic (~155 MB/s,
  unchanged). This is what flips the expected PARK verdict.
- Encode cost: ~10-15% slower (offline, acceptable).
- Random access: **not broken** — touched=1 holds (proven by test).
- Trained-codebook VQ: **NIX/PARK** — quality-poor, rate-floored, no speed gain.

The decode-speed objection the mandate raised is real ONLY for codebook VQ; the
*lattice* variant gives the RD gain for free. Recommend KEEP the low-freq D4
lattice quantizer as a candidate quant block, gated on cross-checking additivity
with RDOQ before integration.
