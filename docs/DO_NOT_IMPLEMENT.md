# DO NOT IMPLEMENT — the frozen-codec ignore list

Authoritative list of things the shipped codec MUST NOT include, with the measured
rationale for each. These were settled by experiment. Do NOT re-add them, even if
they seem helpful, "more flexible", "safer", or "standard practice". The single
frozen pipeline is intentionally minimal.

## THE FROZEN PIPELINE (what IS in — final)
**32³ atom** · integer **DCT-32** (partial-butterfly, i32 MAC accumulators) ·
dead-zone quant (0.80 width / 0.40 dequant offset) + 1-parameter HF-boost curve ·
frequency scan + EOB · adaptive binary range coder (context coder, sole entropy,
table-free, per-atom reset) · per-chunk q via real measure-bisection on a ≤128³
calibration crop · chunk = I/O/index grouping (no shared tables) · raster (z,y,x)
order · 8 fixed LODs (0–7), each fully independent · uniform/absent fast-paths ·
self-describing archive (magic+version+atom+chunk, validated on open) · u64 offsets ·
positional index. Pure C23, CPU, no GPU, no hand intrinsics, coarse-multithreaded
(OpenMP, independent atoms/chunks), lossy-only, u8, 3D, 10–100×, **32³ random access**.

## TRANSFORM — DCT-32, final (measured)
- **9/7 (or any) WAVELET — REJECTED.** DCT-32 beats 9/7 by ~3.7–4.2 dB at 32³ (fair
  test: 9/7 lifting verified perfect-reconstruction, proper per-subband quant, weighting
  swept). Wavelets win for whole-image multi-resolution; DCT wins for fixed *block*
  transforms. Longer filters (13/9 etc.) are moot — a refinement within a family already
  losing by 3.7 dB, and they ring on ink edges.
- **DCT-64 / 64³ atom — REJECTED.** 64³ gains only ~+0.3 dB over 32³ (vs 16→32's +0.8–1.2;
  deep diminishing returns) and is smallest at the low ratios actually used; costs ~8×
  random-access latency and spills the atom scratch L2→L3. 32³ is the sweet spot.
- **DCT-16 — superseded.** 16³ trailed c3d by 1–2.8 dB; 32³ closed it. (A 32³ atom MUST use
  a true 32-point DCT, not tiled 16-pt DCTs — that would give 32³ latency with 16³ quality.)
- **LFNST / secondary transforms / per-atom adaptive transform mode** — need a per-atom
  flag → violate the single pipeline. Cut.

## QUANTIZATION / RATE CONTROL
- **per-block / per-atom q-field — REJECTED (re-measured).** It DID gain quality in
  isolation vs RLGR without an HF curve, but on the composed context-coder+HF-curve stack
  the equal-bytes gain is NOISE (±0.05 dB; threshold was +0.3). The HF curve already does
  the detail-vs-flat reallocation; scroll data has no activity spread to exploit. Add back
  ONLY for a pathological half-void/half-dense volume.
- **Lagrangian λ / closed-form q-from-λ / convex-hull R-D / multi-q-probe / equal-slope**
  — all cut. Rate control = q chosen by real measure_member_size bisection (on a ≤128³
  calibration crop, since q is spatially homogeneous, ~15% stdev). No allocation algorithm.
- **Cheap per-atom-sample q estimator — REJECTED.** A log-log fit over sampled atoms was
  fragile (diverged from true member size on very-compressible / thin volumes → clamped q
  → crushed quality). Use the real measure bisection. If a fast calibration is wanted later,
  validate it against measure_member_size across the edge cases first.
- **D4-lattice / VQ / any non-scalar quantizer; content-adaptive / perceptual-in-loop quant;
  z-anisotropy / per-axis anisotropic quant; learned/trained tables** — all cut.
- **Near-lossless / lossless tier, bit-plane / SNR-scalable coding** — lossy-only; the
  resolution pyramid serves progressive quality.

## ENTROPY — context coder, do NOT swap (measured retrade)
- **RLGR / static rANS / Rice / any alternate coder — do not substitute.** Entropy is
  lossless (swapping costs ZERO quality, only ratio). Context coder gives the best ratio
  (~30% over RLGR, 3.7× over Rice). RLGR would lose ~24% ratio for only ~+14% total decode
  (entropy is ~33% of decode, the DCT is ~37%). rANS ≈ −15% ratio, same speed class. The
  right decode lever is multithreading, not a worse coder. Rice can't code zero-runs (~6× cap).
- **EG2024 3-band split; shared/region entropy tables; sparse prepass** — all rANS-table
  artifacts, dead under the table-free context coder.

## OTHER CUTS
- **Deblock post-filter — DROPPED at 32³.** Measured gain ≤0.02 dB (the bigger atom has 8×
  fewer faces → no blocking to heal). Removing it also made vc_decode_atom EXACTLY match
  full decode (0 mismatch — deblock was the only thing differing) and sped decode.
- **Inter-LOD residual / cross-LOD prediction / anchor+residual chains** — NIXED. All 8 LODs
  are totally independent (each a standalone encode). No residual/upsample/anchor/cascade.
- **Curve-groups, cross-atom prediction (6/18/26-conn), DC sub-volume / DC-DPCM** — ratio-
  neutral or worse; cut.
- **Shared chunk-level low-frequency (base+residual / DC-field / downsampled-chunk) — REJECTED
  (measured).** Both shapes LOST at equal bytes (−0.1 to −1.3 dB): the DCT already represents
  its low frequencies well, and a side-channel LF is a worse basis that just steals residual
  bytes. The quality gap to a bigger-window codec is frequency RESOLUTION, which only a larger
  transform fixes (→ 32³), not a side channel.
- **Per-atom checksums / xxHash / integrity** — cut; rely on storage.
- **min/max / range / predicate-skip metadata** — keep ONLY the uniform/absent flags.

## CUT RATIONALES — benefits we explicitly DON'T want
- **NO determinism machinery.** Cross-CPU bit-exactness is a NON-GOAL. The integer DCT is for
  SPEED (SIMD lanes), not reproducibility. `-ffast-math` is fine.
- **NO selectable features / profiles / modes / config knobs.** Single frozen pipeline, one
  code path. (Build-time `VC_MARCH`/`VC_PORTABLE`/`VC_OPENMP`/`VC_LTO` are *build* options,
  not codec modes — they don't change the bitstream.)
- **NO GPU, NO hand intrinsics.** Compiler-autovectorizable C23 only; SIMD is reached via
  `-march` and the same arch-generic source serves AVX2 / AVX-512 / ARM NEON.
- **NO custom compiler pass.** Would break the "builds with any C compiler" drop-in contract.
  (The one place clang lowered the forward DCT poorly was fixed with a per-compiler `#if`
  source form, pure C.)
- **NO ink-detection / downstream-task gating.** Quality judged on proxy metrics
  (PSNR / MS-SSIM / GMSD / HaarPSI).

## PERFORMANCE NOTES (do not regress)
- MAC accumulators are **i32, not i64** — i64 forced emulated/narrow SIMD (~+61% decode lost).
  Range-safe (max |acc| ≈ 854M ≪ INT32_MAX). Quant levels are clamped to i16 (a too-small q
  could otherwise wrap the level and corrupt the atom — fuzz/test-caught).
- DCT: unit-stride transform passes + a separate streaming rotate (fusing the rotate into the
  transform = strided writes = ~5× slower). Coefficient pruning skips all-zero lines.
- SIMD is exhausted: every vectorizable loop is 512-bit-vectorized; the rest (serial range
  decoder, memory-bound transpose) is inherently un-vectorizable. Further speed = threads, not
  more SIMD or hand intrinsics.
- Decoder is bounds-hardened for untrusted input (fuzzed: libFuzzer ASan+UBSan, MSan, AFL++
  LAF+CmpLog — all clean after fixes; TSan-clean modulo libomp false-positives).
