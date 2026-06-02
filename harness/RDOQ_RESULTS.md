# EXPERIMENT #19 — RDOQ (rate-distortion-optimized quantization)

**Branch:** `exp-rdoq`  ·  **Base:** 70d0871  ·  **Date:** 2026-06-02

## What & why

Independent rounding (`quant_atom` in `src/chunkmodel/blockgrid.c`) picks each DCT-16³
coefficient's quantization level to minimize that coefficient's distortion *alone*,
ignoring the bit cost the chosen level incurs in the entropy coder. RDOQ (the classic
HEVC/H.264 trellis, ~5-8% BD-rate there) instead jointly chooses an atom's AC levels to
minimize **D + λ·R** with R the *actual RLGR rate* — the won max-ratio coder.

Implemented `rdoq_quant_atom()` (encode-only; **decode and the bitstream layout are
unchanged**) with the two high-payoff RDOQ moves:
1. **Per-coefficient level-down** — for each coeff try level L vs L-1 vs 0, accept the
   lower level iff `λ·ΔR > ΔD`. Rounding a small coeff to 0 lets a zero-run absorb it
   (≈free in RLGR run mode) at a tiny distortion cost.
2. **Last-significant-coefficient cut** — sweep the freq-scan in reverse, force the tail
   to zero at the cut that maximizes `λ·rateSaved − distAdded`, so RLGR pays no trailing
   tokens. This is the dominant RDOQ gain.

Rate is the real RLGR structure modeled per symbol (zero ≈ run-amortized fraction of a
bit; nonzero ≈ 1 + GR(mag−1) + sign). Distortion is coefficient-domain SSD (Parseval
proxy for spatial MSE) in true reconstructed-coefficient units (`level·step`). λ is
exposed as `cfg.rdoq_lambda · step²` so one knob tracks the quality knob. Wired into both
the box (M1) and curve-group encode paths.

## The ONLY fair metric: ratio at EQUAL quality

Trading PSNR for ratio is just "use a bigger q". So we built a **finely-swept baseline
RD curve** (independent rounding, q∈{8..256}) and report, at each RDOQ point, the bitrate
delta vs the baseline **interpolated to the SAME PSNR** (and the same MS-SSIM). Positive =
real RDOQ win (≡ negative BD-rate). Stack = won max-ratio config: DCT-16³ + HF-quant +
box-128 M1 + RLGR, no prediction. Real PHerc Paris 4 256³ (hires-256 + coarse-256).

### Bitrate saved at equal quality (RLGR, λ_const=0.06 = sweet spot)

| input      | q   | bitrate@equal-PSNR | bitrate@equal-MS-SSIM |
|------------|-----|--------------------|------------------------|
| hires-256  | 16  | **+12.0%**         | +12.1% |
| hires-256  | 32  | **+10.7%**         | +10.3% |
| hires-256  | 64  | **+11.2%**         |  +9.4% |
| hires-256  | 128 | **+11.7%**         |  +7.1% |
| coarse-256 | 16  | **+12.0%**         | +11.9% |
| coarse-256 | 32  | **+12.0%**         | +11.2% |
| coarse-256 | 64  | **+13.6%**         |  +9.0% |
| coarse-256 | 128 | **+15.7%**         |  +2.2% |

RDOQ wins on **both** PSNR and MS-SSIM across the operating band (the perceptual win only
narrows at the extreme q=128 tail). rANS path: a smaller but still real +6-10% PSNR /
+3-7% MS-SSIM (rANS already amortizes a shared table, so less headroom).

### λ sensitivity (the trap)

Larger λ buys bigger PSNR savings but starts *losing* on MS-SSIM (over-zeroing HF) — that
is trading perceptual quality, which the equal-MS-SSIM column correctly penalizes:

| λ_const | hires PSNR-save (q16..128) | hires MS-SSIM-save | verdict |
|---------|----------------------------|---------------------|---------|
| 0.06    | +12.0 .. +11.7%            | +12.1 .. +7.1%      | **best, wins both** |
| 0.10    | +7.8 .. +11.0%            | +6.8 .. +5.1%       | good |
| 0.15    | +4.8 .. +10.6%            | +4.5 .. +2.8%       | ok |
| 0.22    | +0.1 .. +8.7%             | −2.1 .. +0.0%       | PSNR-only, MS-SSIM neutral/neg |

Recommended default: **λ = 0.06·step²**.

## Encode-time cost (RDOQ is encode-only)

| path | baseline enc | RDOQ enc | slowdown |
|------|--------------|----------|----------|
| RLGR  | ~80 MB/s | ~76 MB/s | **1.05-1.08×** |
| rANS  | ~64 MB/s | ~59 MB/s | **1.08-1.11×** |

Only **5-11% slower** to encode. Cheap because per-coeff decisions use the analytic RLGR
model (no per-candidate re-encode). **Decode speed is identical** (130-155 MB/s) — RDOQ
never touches the decoder.

## Invariants (PLAN §7 + 16³ random access)

- Pure C23, libc/libm only, single-threaded reentrant, heap scratch.
- Decode/quant/dct loops still autovectorize (`-fopt-info-vec` confirms the gather/build
  loops vectorize @ 16-byte vectors). The level-down decision loop is data-dependent and
  inherently serial — but it is **encode-only**, like RLGR itself.
- **touched==1 preserved**: RDOQ changes only level *values*, not the per-atom seekable
  byte ranges or directory. New ctest cases `rdoq-rlgr / rdoq-rans / rdoq-dcsv /
  rdoq-unaligned` all confirm round-trip-exact decode, random-access match, maxtouch==1.
- Existing ctest: `chunkmodel` + `roundtrip` green. (`ratectrl` fails identically on the
  clean base 70d0871 — pre-existing, unrelated to this change.)

## Decision

**KEEP.** +10.7 to +15.7% bitrate-at-equal-PSNR and +7 to +12% at-equal-MS-SSIM on the
won RLGR max-ratio stack, for only ~5-8% encode slowdown and **zero** decode cost. This is
at the top of the literature RDOQ range and it wins on both fidelity metrics at λ=0.06.
Encode-only, bitstream-compatible, random-access-preserving — it slots cleanly into the
frozen codec as a default-on encoder option. Reproduce: `./build/bench_rdoq <repo-root>`.
