# Sign-bit modeling bake-off (branch r2-sign-model)

**Question.** Quantized DCT-16³ coefficient SIGNS are coded raw (1 bit each) by
RLGR. Do they carry exploitable structure (global skew, per-band bias, or
co-located neighbor-atom correlation, à la c3d sign-prediction) that a context
coder could turn into ~1-2% ratio gain?

**Answer: NO. NIX.** Signs are statistically indistinguishable from a fair coin
(49.8–50.0% negative at every q on both volumes). No context model beats raw
1-bit coding; all are marginally *worse* (adaptive-coder overhead at p≈0.5).
Best achievable saving = **0 bytes**, archive ratio gain = **+0.000%**.

## Method (clean substream isolation)

Self-contained throwaway bench `harness/bench_signmodel.c` (does NOT touch
codec.c). It runs the **won stack** per 16³ atom — integer DCT-16³
(`dct_int16.c`) + HF-protecting per-coefficient quant matrix (`qmatrix16.c`,
slope 0.60, dead-zone) — and links the **won RLGR** entropy coder for the
baseline archive size. Atom = 16³ = the fixed random-access unit (untouched).

The sign substream is isolated exactly: RLGR spends exactly **1 raw bit per
nonzero level** on its sign. The signs to code = signs of all nonzero quantized
levels in coef scan order (`z·256+y·16+x`). Everything else in the stream
(magnitudes, zero-runs, headers) is byte-identical, so

    new_archive = total_archive − raw_sign_bytes + model_coded_bytes

makes the measured delta the *exact* ratio impact. Model coded sizes are
produced by a real, round-trip-verified carry-propagating binary range coder
(Subbotin, 12-bit adaptive contexts, shift-update rate 5); `rt=ok` confirms each
model decodes its own stream bit-exactly.

### Sign models compared
- **M0 RAW** — 1 bit/sign (what RLGR does today).
- **M1 GLOBAL-AC** — one adaptive binary prob for all AC signs (catches global skew).
- **M2 PER-BAND** — adaptive prob keyed by frequency band b = u+v+w ∈ 0..45 (per-position bias).
- **M3 NEIGHBOR** — context = co-located previous-atom sign (same scan index, prev-neg flag) × band-class; the c3d-style "same basis function correlates atom-to-atom" hypothesis.

## Results — REAL data, q ∈ {16,32,64,128}, vs RLGR baseline

Bytes are the sign substream only. `best_save` = raw − best(M1,M2,M3) bytes
(coded). `arch_ratio_gain` = full-archive ratio change if we swapped raw signs
for the best model.

### PHerc Paris 4 hires-256 (ink/fiber-rich)
| q   | nz/atom | %neg | total_arch B | sign_raw B | M1 B | M2 B | M3 B | best_save B | arch ratio gain | enc MB/s | rt |
|-----|---------|------|--------------|-----------|--------|--------|--------|------------|-----------------|----------|----|
| 16  | 269.2   | 49.8 | 1,113,347    | 137,823   | 139,591| 138,961| 139,216| 0          | +0.000%         | 58       | ok |
| 32  | 139.1   | 49.8 | 607,335      | 71,242    | 72,174 | 71,783 | 71,904 | 0          | +0.000%         | 70       | ok |
| 64  | 69.3    | 49.8 | 325,059      | 35,494    | 35,949 | 35,712 | 35,756 | 0          | +0.000%         | 74       | ok |
| 128 | 31.9    | 49.9 | 171,830      | 16,346    | 16,541 | 16,435 | 16,395 | 0          | +0.000%         | 76       | ok |

### PHerc Paris 4 coarse-256
| q   | nz/atom | %neg | total_arch B | sign_raw B | M1 B | M2 B | M3 B | best_save B | arch ratio gain | enc MB/s | rt |
|-----|---------|------|--------------|-----------|--------|--------|--------|------------|-----------------|----------|----|
| 16  | 835.7   | 49.8 | 2,947,317    | 427,892   | 433,700| 432,060| 432,570| 0          | +0.000%         | 50       | ok |
| 32  | 438.4   | 49.9 | 1,664,214    | 224,477   | 227,696| 226,737| 227,016| 0          | +0.000%         | 61       | ok |
| 64  | 223.7   | 49.9 | 901,476      | 114,533   | 116,250| 115,734| 115,833| 0          | +0.000%         | 68       | ok |
| 128 | 102.0   | 50.0 | 447,716      | 52,207    | 53,025 | 52,784 | 52,792 | 0          | +0.000%         | 73       | ok |

## Interpretation

- **No global skew.** %neg sits at 49.8–50.0% everywhere → M1 is a fair-coin
  coder and pays pure overhead (≈ +1.3% over raw).
- **No per-band bias.** M2 (per-frequency-band adaptive prob) stays within
  0.1–0.8% *above* raw at every band group — DCT basis-function signs at a fixed
  position are not biased toward + or −.
- **No atom-to-atom correlation.** M3 (co-located previous-atom sign) is also
  worse than raw. The c3d sign-prediction premise (which works on *segmentation*
  curve labels and palette indices, not on transform-coefficient signs) does not
  transfer: a quantized DCT sign is essentially the sign of a near-zero-mean
  AC coefficient → memoryless Bernoulli(½).
- The sign substream is already ~12% of the whole archive, so even a *theoretical*
  1-2% saving on signs would be ≤0.25% on the archive — but the real saving here
  is **zero / negative**.

## Decision

**NIX.** Sign-bit context coding yields no ratio gain on PHerc Paris 4 (the only
plausible models all lose to raw 1-bit). Keeping signs raw is also the cheapest,
fastest, random-access-safe option.

### Decision numbers
- Ratio gain: **+0.000%** (best case; M1/M2/M3 all increase size by 0.1–1.3%).
- dB / quality: **0 dB** — sign coding is lossless side-info, distortion unchanged.
- Cost if adopted: an adaptive binary range coder is **serial** (like RLGR's bit
  IO, not autovectorizable) and would add a second entropy pass; enc throughput
  of the sign pass alone measured 50–76 MB/s. For zero benefit, that is pure cost.
- Random access: a per-atom sign model (M0/M1/M2) keeps the 16³ atom
  independently decodable. M3 (neighbor-atom context) would **break** strict
  per-atom random access (decode would depend on the previous atom's signs) — an
  extra reason to reject it.

## Notes / caveats
- Pre-existing failing test on this commit (70d0871): `ctest` target `ratectrl`
  fails because the Lagrangian allocator only reaches ~7.5× at high targets.
  This is unrelated to entropy/signs and was failing before any change here. The
  other 6 ctests (roundtrip, entropy, transforms, healing, chunkmodel,
  qmatrix16) remain green. This experiment adds no codec.c / config.h change and
  no new ctest is warranted (no new codec MODE was kept).
- Mandate compliance: pure C23, no hand intrinsics, heap scratch (volume-sized
  `sbuf`, not chunk-stack), single-threaded reentrant, round-trip verified.
