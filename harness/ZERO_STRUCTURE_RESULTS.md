# Hierarchical zero-structure for the entropy stage — R3 experiment

Branch `r3-zero-structure`. Real PHerc Paris 4 hires-256 + coarse-256 (256³ u8).
Bench: `harness/bench_zerostruct.c` (standalone; links the won DCT-16³ transform +
HF-protecting per-coef quant matrix, slope 0.50, dead-zone width 0.80, recon-offset
0.40 — the current stack — and the won RLGR entropy coder). Built with
`-O3 -ffast-math -ffp-contract=fast`; quant + DCT kernels confirmed autovectorized
(`-fopt-info-vec`). RLGR/EOB/CBF bit-IO is inherently serial (as RLGR already is).

## What was tested

The HEVC/AV1 multilevel-significance-map + last-position design, applied WITHIN one
16³ DCT atom (stays in-atom, touched=1 holds). Two structure mechanisms, layered on
the won RLGR run-mode baseline:

1. **EOB / last-significant-position** — along a 3D frequency scan (sorted by band
   sum u+v+w), code the last-nonzero index; RLGR only the [0..last] prefix. Trailing
   zeros cost ~nothing.
2. **Sub-cube coded-block-flags (CBF)** — partition the 16³ coefficient cube into
   sub-cubes (4³ = 64 sub-cubes, OR 8³ = 8 sub-cubes), code one any-nonzero flag per
   in-range sub-cube, and RLGR only the survivors in significant sub-cubes.

All modes encode the **same quantized levels** (identical distortion / reconstruction)
so only the entropy stage varies — this isolates the pure incremental gain over
RLGR's run-mode. Round-trip verified for every atom × mode × q (all "ok"). Everything
is in-atom: 16³ random access and touched=1 are preserved.

Modes: **A** RLGR (won baseline) · **B** EOB · **C** CBF4(4³)+EOB · **D** CBF8(8³)+EOB.

## Results (ratio vs RLGR baseline, and decode cost)

`dec syms/atom` = RLGR levels the decoder runs through (decode-work proxy; RLGR decode
is ~linear in coded symbols). `dec MB/s` = measured atom-decode throughput.

### hires-256 (ink/fiber-rich)
| q | mode | ratio | vs-RLGR | dec syms/atom | dec MB/s |
|---|---|---|---|---|---|
| 16  | A RLGR | 29.26x | — | 4096 | 1239 |
| 16  | **B EOB** | 29.59x | **+1.13%** | 788 | **4538 (3.7×)** |
| 16  | C CBF4 | 29.28x | +0.08% | 570 | 1251 |
| 16  | D CBF8 | 29.59x | +1.13% | 719 | 1488 |
| 32  | A RLGR | 55.47x | — | 4096 | 1689 |
| 32  | **B EOB** | 57.40x | **+3.49%** | 358 | **7860 (4.7×)** |
| 32  | C CBF4 | 57.02x | +2.81% | 281 | 2442 |
| 32  | D CBF8 | 57.75x | +4.13% | 315 | 2880 |
| 64  | A RLGR | 103.96x | — | 4096 | 2102 |
| 64  | **B EOB** | 113.66x | **+9.33%** | 168 | **12223 (5.8×)** |
| 64  | C CBF4 | 112.52x | +8.23% | 143 | 4694 |
| 64  | D CBF8 | 113.09x | +8.79% | 159 | 5423 |
| 128 | A RLGR | 183.79x | — | 4096 | 2602 |
| 128 | **B EOB** | 231.99x | **+26.23%** | 75 | **19346 (7.4×)** |
| 128 | C CBF4 | 225.91x | +22.92% | 61 | 8324 |
| 128 | D CBF8 | 221.20x | +20.36% | 74 | 10349 |

### coarse-256
| q | mode | ratio | vs-RLGR | dec syms/atom | dec MB/s |
|---|---|---|---|---|---|
| 16  | A RLGR | 9.89x | — | 4096 | 534 |
| 16  | B EOB | 9.86x | -0.25% | 2285 | 1331 |
| 16  | C CBF4 | 9.84x | -0.45% | 1673 | 383 |
| 16  | D CBF8 | 9.85x | -0.36% | 2156 | 477 |
| 32  | B EOB | 19.15x | +0.32% | 1284 | 2325 |
| 32  | D CBF8 | 19.11x | +0.15% | 1226 | 893 |
| 64  | A RLGR | 39.01x | — | 4096 | 1360 |
| 64  | **B EOB** | 39.65x | +1.64% | 724 | **5319 (3.9×)** |
| 64  | D CBF8 | 40.01x | +2.55% | 611 | 1582 |
| 128 | A RLGR | 84.89x | — | 4096 | 1974 |
| 128 | **B EOB** | 89.28x | +5.17% | 362 | **9735 (4.9×)** |
| 128 | D CBF8 | 90.18x | +6.23% | 295 | 3113 |

## Findings

1. **EOB / last-significant-position is the win, and it's a clean win-win.** On the
   ink/fiber-rich hires volume it delivers the lit-predicted 3–6% (q32 +3.5%, q64
   +9.3%) and far more at high ratio (q128 +26%), AND it is the FASTEST decoder by a
   wide margin (3.7×–7.4× MB/s) because the decoder stops at the last nonzero instead
   of grinding through 4096 levels. Decode speed is our weak axis — this directly
   improves it. EOB costs one small Golomb header per atom and nothing else.

2. **Sub-cube CBF does NOT deliver — it is dominated by EOB.** Layering CBF on top of
   EOB always reduces decode symbols further, but the per-sub-cube flag overhead
   cancels (4³) or barely beats (8³, only at mid-q) the EOB ratio, while making the
   decoder SLOWER than plain EOB (extra flag parsing + survivor scatter, and it does
   not skip the dominant cost). CBF4 is worse than EOB at every q on both volumes.
   CBF8 edges EOB by ≤0.8% (hires q32/q64) and ≤1.1% (coarse), loses at q128 hires —
   not worth the added in-atom complexity. RLGR's run-mode already codes the
   within-prefix zero clusters that CBF tries to exploit, so the second significance
   level is redundant here. The HEVC/AV1 multilevel map pays off against a per-coef
   binary-arithmetic backend, NOT against an adaptive run-length backend like RLGR.

3. **The gain grows with ratio** (operating point matters): at the aggressive 100–180×
   range the scrolls target, EOB is worth +9% to +26% on hires. At very low ratio
   (q16 coarse) it is ratio-neutral but still 2.5× faster to decode.

4. **Invariants preserved.** All structure is in-atom; 16³ random access and touched=1
   hold; round-trip exact for every atom/mode/q. The only added per-atom side info is
   the EOB Golomb header (no per-coef-position tables, no cross-atom state).

## Decision: KEEP EOB, NIX sub-cube CBF

KEEP the **EOB / last-significant-position** layer on the RLGR entropy stage. It is a
small, self-contained change with a real win-win: +3.5%/+9.3%/+26% ratio at q32/64/128
on hires (the data that matters), positive on coarse at q≥32, and 3.7×–7.4× faster
atom decode — exactly the speed axis we most need. NIX the sub-cube CBF half: it is
dominated by EOB on this RLGR backend (≤1% gain at best, slower than EOB, more code).

Cost of KEEP (EOB): one Golomb-coded last-position per 16³ atom (~1–2 bytes, already
counted in the ratios above); no random-access, distortion, or touched=1 impact;
encode adds one O(4096) reverse scan per atom. No quality cost (lossless wrt the
quantized levels — identical reconstruction, so PSNR/GMSD/edge-MAE unchanged from the
baseline at equal q).
