# DO NOT IMPLEMENT — the frozen-codec ignore list

Authoritative list of things the implementation MUST NOT include. These were deliberately
NIX'd after ~26 experiments + audit. Do NOT re-add them, even if they seem helpful, "more
flexible", "safer", or "standard practice". The single frozen pipeline is intentionally minimal.

## CUT FEATURES — do not implement any of these
- **RDOQ / trellis quantization** — overlaps EOB (same trailing-coeff win), decode-neutral, complex. Use EOB only.
- **D4-lattice / vector quantization / any non-scalar quantizer** — cut; quant is HF-matrix + scalar dead-zone ONLY.
- **Morton / Hilbert / any space-filling-curve atom order** — use RASTER (z,y,x) only.
- **Shared entropy tables / region-shared frequency tables / rANS tables** — the context coder is table-free; there is NO table to share. Chunk = pure I/O/index grouping only.
- **Per-ATOM checksums** — checksums are PER-CHUNK only (xxHash).
- **min/max / range / predicate-skip metadata** — keep ONLY the per-atom `uniform`/all-same flag (memset fast-path). No min/max stats.
- **RLGR, static rANS, plain Rice, or ANY alternate entropy coder** — the context coder is the SOLE entropy stage. Do not keep RLGR even "as a fast path" or "for comparison" in the shipped codec.
- **EG2024 3-band (DC/low-AC/high-AC) table split** — was rANS-only; dead under table-free context coder.
- **LFNST / secondary transforms** — cut (needs a per-atom flag → violates single-pipeline).
- **Sparse prepass** — fit rANS tables; nothing to prepass under table-free coder.
- **Curve-groups, cross-atom prediction (6/18/26-conn), DC sub-volume / DC-DPCM** — all ratio-neutral or worse; cut.
- **9/7 wavelet, lapped/POT transform, directional/oriented basis, adaptive per-atom transform mode** — cut; transform is integer DCT-16³ ONLY.
- **CDEF, content-adaptive quant, perceptual-in-loop quant, sign-bit modeling, z-anisotropy / per-axis anisotropic quant, learned/trained tables** — all cut.
- **multi-q-probe / analytical-Laplacian / single-pass-feedback rate control** — rate control is closed-form q-from-λ ONLY.
- **Near-lossless tier, lossless tier, bit-plane / SNR-scalable coding** — lossy-only, no quality-layer scalability. (Resolution pyramid serves progressive quality.)
- **fast-rc as a separate optional profile** — closed-form q-from-λ is THE single rate controller, not an option.

## CUT RATIONALES — do not add machinery for benefits we explicitly DON'T want
- **NO determinism machinery.** Cross-CPU bit-exactness is a NON-GOAL. Integer DCT is for SPEED (16-bit SIMD lanes), not reproducibility. `-ffast-math` is fine; do not add careful rounding modes / fixed-point bookkeeping for determinism.
- **NO selectable features / profiles / modes / config knobs / runtime dispatch.** SINGLE frozen pipeline. No "toolkit", no compile-time-configurable blocks in the shipped codec, no user-pickable options. One code path.
- **NO 8³ sub-block logic.** The transform block = the atom = 16³. There are no sub-blocks inside an atom. Deblock handles atom/chunk faces only, not internal sub-block faces.
- **NO GPU, NO hand intrinsics.** Compiler-autovectorizable C23 only.
- **NO ink-detection / downstream-task gating.** Quality judged on proxy metrics (PSNR/MS-SSIM/GMSD/HaarPSI); do not build a task harness.
- **NO broader-corpus / robustness / multithread / SIMD-tuning work yet** — punted post-freeze; not part of the integration.

## THE ONLY THINGS THAT ARE IN (the frozen pipeline)
16³ atom · integer DCT-16³ · HF-matrix + scalar dead-zone(0.80/0.40) quant · frequency-scan + EOB ·
context coder (sole, table-free, per-atom reset) · closed-form q-from-λ per-16³ q-field rate control ·
chunk = region grouping (I/O/index) · raster order · 8 fixed LODs (0-7) · inter-LOD residual (always-on,
L7 independent anchor) · deblock (decode post-filter, 0.3-0.5) · per-chunk xxHash · uniform-flag fast-path ·
u64 offsets · positional index. Pure C23, CPU, no GPU, no hand intrinsics, single-threaded reentrant,
lossy-only, u8, 3D, 10-100×, 16³ random access (touched=1). Nothing else.
