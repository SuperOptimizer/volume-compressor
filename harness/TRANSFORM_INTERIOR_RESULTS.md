# EXP #13 — Transform-interior improvements (PLAN §2 transform / quant / entropy)

**Question.** Inside the *won* 16³ DCT atom, which of three interior improvements buys
ratio/quality at acceptable cost, each measured as a swappable variant vs the winning
baseline at **matched ratio** on real PHerc Paris 4 data?

- **(a) tuned per-coefficient QUANT-MATRIX shape** — beyond the existing linear HF slope:
  L2-radial, separable per-axis, and a data-derived energy-tuned per-position step.
- **(b) intra-atom LAPPED / POT transform** — overlap *within* the 16³ atom only (8³ DCT
  tiling + a POT/LBT pre/post filter across the single internal seam; never crosses the
  atom face, so 16³ random access is preserved).
- **(c) coefficient CONTEXT modeling** — JPEG-XL/HEVC-style binary range coder that codes
  each coefficient's significance + magnitude conditioned on already-decoded causal
  neighbor coefficients *within the atom*, replacing RLGR on the per-atom stream.

**Baseline (the winning stack):** integer DCT-16³ · fixed HF-protecting per-coefficient
16³ quant matrix (slope 0.6) · RLGR. Self-contained bench
`harness/bench_transform_interior.c` (does NOT touch codec.c). Round-trip + structural
tests in `tests/test_transform_interior.c` (ctest `transform_interior`, green). All
objectives driven to the same target ratio (10/20/50×) by bisecting the base step.
Atom = 16³ (contexts reset per atom → random access intact, touched=1 preserved).

---

## Headline: (c) coefficient context modeling is a large, free ratio/quality win;
## (a) and (b) do not pay.

### Apples-to-apples entropy comparison — CTX vs RLGR at IDENTICAL quantization
(same coefficients, same distortion → the pure ratio gain context modeling buys at
**matched quality**):

| volume | 10× | 20× | 50× |
|---|---|---|---|
| hires-256 | **−26.7%** (→13.8×) | **−29.1%** (→28.3×) | **−17.2%** (→62.4×) |
| coarse-256 | **−31.3%** (→14.6×) | **−31.6%** (→29.7×) | **−18.8%** (→63.3×) |

Context modeling shrinks the RLGR payload by **~27–32% in the dominant 10–20× band** and
~17–19% at 50×, at *identical* quantization. That is on the same order as (or larger
than) the already-won EG2024 3-band split, and it stacks with it (it is a strictly better
per-stream entropy coder, orthogonal to the table split).

### Matched-ratio quality (the same win expressed as quality), hires-256
| variant | 20× PSNR | MS-SSIM | GMSD | edgeMAE | seam16 | enc MB/s | dec MB/s |
|---|---|---|---|---|---|---|---|
| BASE DCT16+HF+RLGR | 40.96 | 0.9779 | 0.06808 | 5.547 | 2.846 | 55 | 137 |
| (a) qmat L2-radial | 41.01 | 0.9781 | 0.06762 | 5.506 | 2.835 | 53 | 138 |
| (a) qmat separable | 40.70 | 0.9767 | 0.06977 | 5.620 | 2.994 | 55 | 133 |
| (a) qmat energy-tuned | 40.85 | 0.9773 | 0.06916 | 5.604 | 2.905 | 45 | 135 |
| (b) lapped8+POT | 37.96 | 0.9595 | 0.09671 | 7.600 | 3.763 | 118 | 207 |
| **(c) context-coder** | **42.90** | **0.9852** | **0.05132** | **4.419** | **2.236** | 39 | 34 |

At matched ratio (c) lifts **PSNR +1.94 dB, MS-SSIM +0.0073, GMSD −25%, edge-MAE −20%,
seam −21%** — every quality/edge metric at once, on both volumes, at every ratio:

| | hires 10× | hires 20× | hires 50× | coarse 10× | coarse 20× | coarse 50× |
|---|---|---|---|---|---|---|
| ΔPSNR (c vs base) | +1.13 | +1.94 | +1.46 | +2.23 | +2.28 | +1.21 |
| ΔGMSD % | −21% | −25% | −12% | −27% | −20% | −8% |
| Δseam16 % | −12% | −21% | −16% | −24% | −24% | −12% |

(Full 6-variant × 2-volume × 3-ratio tables: `bench_transform_interior` raw output.)

---

## Per-variant verdicts

### (a) Tuned quant-matrix shape — NIX (baseline shape already near-optimal)
The existing **linear HF slope** (`1 − slope·(u+v+w)/45`) is already essentially optimal
among smooth monotone shapes. **L2-radial** is a statistical tie (≈+0.05 dB / GMSD −0.7%
at 20×, within noise, both volumes). **Separable per-axis** is *worse* (−0.3 dB, seam +5%):
the product over-coarsens off-axis high frequencies that carry ink/fiber. **Energy-tuned**
(per-position step ∝ 1/√mean-coef-energy, blended 0.6 toward flat) is ≈neutral-to-slightly
worse — the empirical energy profile is already close to the linear-slope shape, so
data-fitting it adds no headroom and a touch of overfit noise. **Conclusion: keep the
existing fixed linear HF matrix; no richer shape pays.** (Confirms HFDIST_RESULTS §4.)

### (b) Intra-atom lapped / POT transform — NIX (loses the 16³ compaction)
Splitting the 16³ atom into 8³ DCTs to gain an internal lapped seam is a **net loss**:
−3.0 dB @20× hires, GMSD +42%, seam +32%. The single internal seam the POT filter smooths
is far cheaper than the energy-compaction the full 16³ DCT gives at these high ratios — the
shorter 8-pt basis simply compacts worse, and the lapping cannot recover it. (It *is* ~2×
faster to transform, but that is irrelevant given the quality collapse.) This re-confirms
why DCT-16³ + mild post-deblock beat lapped/short-block transforms in the Phase-1 transform
bake-off: at ≤1 bpp the longer basis wins, and the 16³ atom already has *no* internal
sub-block seams to lap. **Conclusion: keep the single 16³ DCT.**

### (c) Coefficient context modeling — KEEP (large, free ratio/quality win)
A small table-free **binary range coder** with adaptive bit models, coding per atom in
increasing-frequency scan order:
- **significance** bit, context = freq-band(6) × #significant causal neighbors(0–3) — the
  key dependency: a coefficient adjacent to already-significant low-freq energy is itself
  far more likely significant;
- **magnitude** as a band-conditioned unary prefix (first bins split) + bypass mantissa,
  with an escape to a 16-bit bypass field for rare large levels;
- **sign** as a bypass bit.

Contexts **reset at each atom boundary** — no cross-atom dependency, so the 16³ random-
access / touched=1 property is fully preserved (round-trip-exact per atom, verified).

**Win:** −27 to −32% bytes @10–20×, −17 to −19% @50×, at identical quality (equivalently
+1.1 to +2.3 dB PSNR / −8 to −27% GMSD at matched ratio), on **both** volumes, **every**
ratio. **Cost:** the range coder is serial (like RLGR) but heavier — encode ~39 vs 55 MB/s
(−29%), decode ~34 vs 137 MB/s (**~4× slower**). It is *more* than free on ratio/quality;
the price is decode throughput.

---

## Decision

**(c) coefficient context modeling — KEEP.** It is the single biggest interior win found:
**~30% ratio at matched quality** (≈+2 dB) in the 10–20× operating band, ~18% at 50×, on
real data, on both volumes, with **16³ random access fully preserved** (per-atom contexts).
It stacks with the already-won EG2024 3-band split (orthogonal: better per-stream coder).

**(a) tuned matrix shape — NIX** (linear HF already optimal; L2 a tie, others worse).
**(b) intra-atom lapped/POT — NIX** (8³ tiling loses the 16³ compaction; −3 dB).

**Tradeoff to weigh at integration:** (c) costs ~4× decode and ~1.3× encode time vs RLGR.
For a **max-ratio / quality-first** profile the ~30% ratio (or +2 dB) is decisively worth
it. For a **max-decode-speed** profile RLGR stays. The model is deliberately small (6×4
significance + 6×3 magnitude contexts, ~50 probabilities) and table-free; an obvious
follow-up is to (i) reuse the bypass renorm to batch-decode signs/mantissas and (ii) tune
the adaptation shift / context split to claw back decode speed — but even untuned it is a
clear KEEP on ratio and quality. The decode-speed cost is the only reason this is not an
unconditional KEEP for *every* profile.

---

## Method / reproduce
```
cmake -S . -B build -DVC_TRANSFORM=dct16 -DVC_ENTROPY=rlgr && cmake --build build
./build/bench_transform_interior          # 6 variants x 2 volumes x 3 ratios + entropy-only
./build/bench_transform_interior 0        # hires only ; "1" = coarse only
ctest --test-dir build                    # incl. transform_interior round-trip/structural test
```
- New files: `src/entropy/ctxcoef.c` (context coder), `src/quant/qmatrix16_tuned.c`
  (tuned shapes), `src/transform/lapped16.c` (lapped), the bench + test. None touch codec.c.
- Pure C23, libc/libm, single-threaded reentrant, heap scratch. The range coder + RLGR are
  inherently serial; the vectorizable stages (transform/quant) are unchanged.
- 256³ = 4096 atoms of 16³; equal-ratio via global-step bisection (±1%).
```
```
