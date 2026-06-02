# Directional / Oriented-Basis Transform — bake-off results

PLAN "parked-but-look" axis: *directional/oriented transform basis*. Scroll
fiber/sheet structure is directional, so an oriented basis (AV1-directional-intra
spirit) might compact directional energy onto fewer separable basis functions
than the always-separable DCT-16³. **HIGH-RISK** (may fight the fast fixed-basis
design). This is the bounded LIGHT test the PLAN asked for.

## What was built

`harness/bench_directional.c` (throwaway, self-contained; does NOT touch
`codec.c`/entropy/quant/ratectrl). It #includes the **won** integer DCT-16³
(`dct_int16.c`) + the **won** HF-protecting 16³ quant matrix (`qmatrix16.c`,
slope 0.60 default), links the **won** RLGR coder + the metric bundle, and runs a
matched-ratio comparison of three configs at q∈{16,32,64,128} (mapped to ratio
targets ~10/20/45/90× on this data) on real PHerc Paris 4 **hires-256** +
**coarse-256**:

- **PLAIN** — DCT-16³ + HF dead-zone + RLGR. The mandated RLGR baseline.
- **DIR-PERM (6)** — per-atom best of 6 reversible **axis permutations** (orient
  the dominant structure onto a consistent transform axis), RD-selected by actual
  coded RLGR bytes. Side cost: 1 byte/atom.
- **DIR-FULL (12)** — permutations **+ 3 diagonal in-plane folds** (face-diagonal
  transposes) = the fuller oriented set, the lightest stand-in for a 45° oriented
  basis. Side cost: 1 byte/atom.

The "oriented basis" is realised as an exactly-reversible voxel **pre-fold**
(precomputed bijective index map) applied before the separable DCT and inverted
after the IDCT — integer-exact, autovectorizable (`-fopt-info-vec` confirmed
clean), and **keeps full 16³ random access**: the chosen mode is one side byte
the decoder reads, so touched=1 still decodes one atom standalone. Selection is
RD (min coded rate); since the basis is orthonormal the distortion is ~constant
across orientations, so min-rate is the correct criterion.

## Results (real data, matched ratio)

### hires-256 (ink/fiber-rich) — the case directional energy *should* help most
| q (ratio)  | config       | ratio  | PSNR  | MS-SSIM | GMSD    | enc MB/s | dec MB/s | non-id% |
|------------|--------------|--------|-------|---------|---------|----------|----------|---------|
| q16 (~10×) | PLAIN        | 10.08× | 44.80 | 0.9902  | 0.03663 | 47       | 54       | 0.0     |
|            | DIR-PERM (6) |  9.95× | 44.84 | 0.9902  | 0.03639 | 9        | 53       | 56.3    |
|            | DIR-FULL(12) |  9.95× | 44.84 | 0.9902  | 0.03639 | 4        | 44       | 56.3    |
| q32 (~20×) | PLAIN        | 19.86× | 40.96 | 0.9779  | 0.06808 | 48       | 53       | 0.0     |
|            | DIR-PERM (6) | 20.08× | 40.96 | 0.9779  | 0.06811 | 10       | 51       | 63.9    |
| q64 (~45×) | PLAIN        | 44.74× | 36.33 | 0.9444  | 0.11258 | 51       | 54       | 0.0     |
|            | DIR-PERM (6) | 45.29× | 36.33 | 0.9444  | 0.11259 | 11       | 54       | 67.6    |
| q128(~90×) | PLAIN        | 90.08× | 32.61 | 0.8875  | 0.15048 | 53       | 55       | 0.0     |
|            | DIR-PERM (6) | 89.22× | 32.72 | 0.8898  | 0.14934 | 11       | 56       | 66.5    |

vs PLAIN deltas (DIR-PERM): ratio **−1.3% / +1.1% / +1.2% / −0.9%**, PSNR
**+0.04 / −0.00 / −0.00 / +0.12 dB**.

### coarse-256
| q (ratio)  | config       | ratio  | PSNR  | MS-SSIM | enc MB/s | dec MB/s | non-id% |
|------------|--------------|--------|-------|---------|----------|----------|---------|
| q16 (~10×) | PLAIN        | 10.00× | 35.27 | 0.9754  | 49       | 57       | 0.0     |
|            | DIR-PERM (6) |  9.95× | 35.55 | 0.9768  | 10       | 55       | 76.0    |
| q32 (~20×) | PLAIN        | 20.12× | 30.94 | 0.9370  | 41       | 45       | 0.0     |
|            | DIR-PERM (6) | 19.89× | 31.21 | 0.9405  | 10       | 56       | 76.0    |
| q64 (~45×) | PLAIN        | 45.19× | 26.87 | 0.8498  | 53       | 57       | 0.0     |
|            | DIR-PERM (6) | 44.60× | 27.12 | 0.8578  | 11       | 54       | 74.8    |
| q128(~90×) | PLAIN        | 89.73× | 24.17 | 0.7336  | 55       | 57       | 0.0     |
|            | DIR-PERM (6) | 89.90× | 24.34 | 0.7429  | 10       | 51       | 74.4    |

vs PLAIN deltas (DIR-PERM): ratio **−0.5% / −1.2% / −1.3% / +0.2%**, PSNR
**+0.29 / +0.27 / +0.26 / +0.17 dB**.

**DIR-FULL (12) == DIR-PERM (6) to 5 digits in every cell.** The diagonal/face-fold
orientations (the actual AV1-directional-intra novelty) are *never preferred* over
plain axis permutations. The directional novelty adds literally nothing.

## Reading

- **The gain is noise.** Best case is coarse-256 at +0.26–0.29 dB PSNR — but it is
  bundled with a −0.5 to −1.3% ratio loss (the 1 side byte/atom + occasionally
  picking a slightly worse rate). Net R-D movement is inside the matched-ratio
  bisection's ±1% band. On hires-256 (the ink/fiber-rich case directional energy
  was *supposed* to help most) it is flat: ±0.12 dB.
- **56–76% of atoms pick a non-identity orientation, yet ratio barely moves.** That
  is the verdict: separable DCT-16³ already captures axis-aligned directional energy
  near-optimally. Re-ordering which axis is transformed first changes the coded rate
  by < 1.5% because the 3D HF quant matrix is near-isotropic and RLGR scans the whole
  4096-coef block — there is no scan/quant asymmetry for an orientation to exploit.
- **The diagonal folds are dead weight.** True off-axis (45°) energy is not what the
  scroll atoms carry at 16³, or the fold is too coarse to capture it; either way the
  oriented modes lose to permutations every time.
- **Encode cost is brutal: 4.5–10× slower** (47→4–11 MB/s) from the per-atom RD
  search over 6–12 orientations (each = full DCT+quant+RLGR). Decode is unchanged
  (~50 MB/s; one extra index-map pass is free and autovectorized).
- **Random access intact:** touched=1 holds; one mode side byte per atom; fold is an
  exact bijection (round-trip verified — identity at mode 0, reversible at all 12).

## Verdict: **PARK** (confirmed)

The bounded test does what the PLAN asked: it confirms the PARK. A LIGHT oriented
basis (axis permutations + face-diagonal folds, RD-selected) **does not beat plain
DCT-16³ on ratio-at-quality** — the movement is < 1.5% ratio and < 0.3 dB, inside
matched-ratio noise, even though most atoms opt into a non-identity orientation.
The genuinely directional (diagonal) modes are never chosen. Meanwhile it costs
**~5–10× encode throughput** and adds per-atom side info + mode-dispatch complexity
that fights the fast fixed-basis design.

A *heavier* directional transform (true arbitrary-angle steerable/rotated basis,
gradient-estimated per atom) is the only thing left that could move the needle, but
it (a) breaks the integer-exact reversibility, (b) needs a real angle search or
gradient estimator + interpolation (non-autovectorizable, non-trivial decode), and
(c) the light test already shows the *axis-aligned* directional headroom is ~0 —
the separable DCT eats it. The expected payoff does not justify that risk/complexity.

**The gain is not worth the complexity. PARK stays PARK.** Do not integrate.

### Decision numbers
- Ratio vs RLGR/DCT-16³ baseline: **−1.3% … +1.2%** (noise; both volumes, all q)
- Quality: **+0.0 … +0.29 dB PSNR**, MS-SSIM +0.000…+0.009 (coarse only; hires flat)
- Encode: **4–11 MB/s vs 47–55 MB/s baseline → ~5–10× slower**
- Decode: **44–56 MB/s vs 53–57 MB/s → unchanged** (within noise)
- Random access: **NOT broken** — touched=1 preserved, +1 mode byte/atom, exact fold
- DIR-FULL (the diagonal/oriented novelty) ≡ DIR-PERM: the directional part is inert

## Repro
```
cmake -S . -B build && cmake --build build --target bench_directional
./build/bench_directional               # hires-256 + coarse-256
./build/bench_directional <raw.u8> D H W
```

## Note on test suite
`ctest` shows `ratectrl` failing — this is **pre-existing at base commit 70d0871**
(a rate-control allocator tolerance test: "per-chunk hits ~10x" / "per-block hits
20x"), independent of this experiment. This branch's change is purely additive (one
new throwaway bench + an additive CMake block); it touches no shipped source. The
other 6 tests (roundtrip, entropy, transforms, healing, chunkmodel, qmatrix16) pass.
