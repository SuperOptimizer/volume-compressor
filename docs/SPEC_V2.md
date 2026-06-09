# volume-compressor v2 — design working-doc (NOT frozen)

A total rewrite of the v1 codec, designed from the masked micro-CT scroll data up.
This is a **living working-doc reflecting what the experiments have shown** — it is
NOT a locked spec. Numbers below are from `tools/v2lab` on real
`paris4_2um_processed` chunks (1024³ masked, 128³ tiles).

## Target compression
**10×–100× on real dense DCT data** (material/present blocks only — the v1-SPEC
convention). The zero-masking is *additive bonus* compression on top: total archive
ratio = (dense-DCT ratio in 10–100×) combined with mask savings from pruned air.
More mask savings = better, but the DCT ratio is judged on material only. Two
numbers always reported: MATERIAL-ratio (spec) and TOTAL-ratio (mask folded in).

## Goal
Beat v1 (which already strictly beats compress3d and compress4d) on ratio AND
quality, while staying fast (decode feeds a zoom viewer), by exploiting the one
thing the data has that a fixed-block codec can't use: **large, multi-scale,
interspersed zero (air) masking** — from ~1024³ outer padding down to tiny pockets
between papyrus sheets.

## Structure (what the bake-off settled on)
A **kd / octree** subdivides the volume coarse→fine. Each leaf is typed:
- **ZERO** — pruned, fills with 0. Free (~1 tag bit), and *inherited* from the
  coarse LOD where possible (a 0 at LOD k+1 proves the 2³ children are 0 —
  verified against the downscale kernel, `acc==0` ⟺ all-zero, lossless).
- **DCT** — a variable-size DCT cube (**4³/8³/16³**; 32³ optional, rarely chosen).
  Separable integer/float DCT-II + dead-zone quant + HF-boost curve, last-
  significant/EOB + context range coder (per-leaf).

The leaf size is chosen by a **cheap activity/occupancy proxy, NOT full RDO**
(H.266 research: partition RDO is >90% of encode cost for ~1% gain). Big leaves
use **HF-zeroing** (store only the low-frequency coefficient subset) to bound
latency on smooth data.

### Clean partition only (empirical)
Each region resolves to exactly ONE of {ZERO, DCT}. Overlap / layered modes
(DCT-base + deeper-ZERO-override, and the mirror ZERO-base + deeper-DCT-refine)
were implemented and measured — **they do not pay**: clean subdivision already
captures the structure; DCT-base overlap wastes bits, ZERO-base overlap drops
sparse signal and fails the quality bar. **Ship clean partition; no overlap.**

## LODs
8 fixed, **independent** encodes (no inter-LOD residual — measured not worth its
decode-speed cost). Objective = **summed-pyramid ratio** over all 8 LODs. The
pyramid is geometric: LOD0=87.5%, LOD1=11%, LOD2–7 <1.5% combined → all tuning
lives at LOD0(+LOD1); LOD2–7 get near-lossless quality essentially free. Coarse
LODs need lower q (they're dense — air dissolves under downscale) but cost ~nothing.

## Lossy everywhere, no exactness
Lossy-only (no lossless mode, no reversible-integer transform). The zero mask is
itself a **lossy lever** (drop a few faint near-boundary voxels) like the DCT
quantizer — but kept gentle: the metric basket showed aggressive zero-eps spikes
max-error. "Good not perfect" boundaries: push as loose as the metrics allow.

## Metric basket
Every experiment reports, over the **nonzero (true-signal) voxels** (air carries
no information): PSNR, SSIM, MAE, RMSE, **max-error, p90/p95/p99 absolute error**,
plus ratio, bpp, encode time. Worst-case (max/p99) matters most for ink and is
where v2 most beats v1.

## Measured result (v2 clean vs v1 fixed-32³, real chunks)
Ratio at equal PSNR over signal (higher = better), dense chunk / sparse chunk:
| target | v1 | v2 clean | v2 worst-case (max) |
|--------|----|----|----|
| PSNR≥45 | 3.5× | **7.0× / 10.4×** | 15–17 vs v1 35–47 |
| PSNR≥40 | 4.9–6.5× | **11.1× / 14.5×** | 22–28 vs v1 58–61 |
| PSNR≥36 | 7.5–18.8× | **17.7× / 20.0×** | 41–54 vs v1 71–106 |

v2 roughly **doubles** v1's ratio at the ink-relevant quality (PSNR 40–45) and
**halves worst-case error**, while encoding faster.

## Open / to confirm
- Bit-cost is currently an **estimate** (order-0 + EOB); build the real per-leaf
  range coder to confirm absolute ratios (quality/worst-case numbers are exact).
- Confirm size set 4..16 vs 4..32 on more chunks (32³ rarely chosen; likely drop).
- Transform precision (int vs f32 vs fp16/bf16) — measure only if DCT is the
  decode bottleneck; doesn't change the bitstream.

## Tools
`tools/v2lab/` — `v2lab.c` (codec experiment + sweep), `dct.h` (multi-size DCT),
`metrics.h` (the basket). Build: `cc -O3 -march=native -ffast-math v2lab.c -lm`.
