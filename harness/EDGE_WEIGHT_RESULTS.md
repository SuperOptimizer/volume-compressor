# Edge/Gradient Importance-Weight Allocation — RESULTS

Branch `r3-edge-weight` (worktree, explore-only). Round-3 lit-backed experiment.

## Hypothesis (lit)

Cheap precompute-once analog of Feature-Preserving RDO (~8% BD-rate / up to 10%
task-accuracy in the source literature). ONE Sobel/gradient pass over the volume
yields a per-16³-block importance weight from local edge energy. The rate
control / quantization then allocates MORE bits (finer dead-zone step) to
high-gradient atoms (ink strokes, sheet/fiber boundaries) and coarser steps to
flat regions, at a FIXED total ratio. Composes with the existing HF-protecting
per-coefficient quant matrix (qmatrix16, slope 0.6). Targets the binding ink
fidelity constraint: scroll ink is fine HF texture.

## Method (codec-faithful, touched=1 preserved)

`harness/bench_edgeweight.c`. Standalone codec-matched 16³ integer DCT (same Q14
matrix as `transform/dct_int16.c`) + qmatrix16 HF-protecting dead-zone quant
(QM_HF, slope 0.6) + RLGR rate. Per-16³-block step (the q-field) is already a
supported lever, so 16³ random access / touched=1 is preserved — each atom still
decodes from its own step independently.

ONE precompute: per-block mean 3D Sobel-gradient magnitude → importance weight
`w = 1 + (wmax−1)·(g/gmax)^0.7`. Three policies compared at EQUAL TOTAL BYTES
(= equal ratio), whole-volume reconstruction, on REAL PHerc Paris 4
hires-256 + coarse-256, q∈{16,32,64,128}:

* **UNIFORM** — every block the same base step (current-stack default; sets budget).
* **MSE-LAG** — per-block Lagrangian on plain MSE (existing rate control; control).
* **EDGEW w=2/4/8** — per-block Lagrangian on edge-WEIGHTED distortion `D' = w·D`.

Metrics: PSNR, plus the edge/perceptual proxies GMSD (down=better), edge-MAE
(down), HaarPSI (up), seam-step (down). HONEST: these are proxies for ink
fidelity — true ink-detection AUC needs a downstream harness we do not have.

## Headline result: NEGATIVE

At equal total ratio, edge-weighting does **not** improve edge/perceptual
fidelity — it slightly **worsens** GMSD and HaarPSI monotonically as the weight
strengthens, with edge-MAE flat (±0.05, noise level) and PSNR unchanged.

### hires-256 (representative, GMSD/HaarPSI deltas vs UNIFORM)

| q | ratio | policy | PSNR | ΔGMSD | ΔedgeMAE | ΔHaarPSI(×1e3) |
|---|---|---|---|---|---|---|
| 32 | 28.8× | EDGEW w=2 | 39.09 | +0.0012 | −0.004 | −0.12 |
| 32 | 28.8× | EDGEW w=4 | 39.08 | +0.0023 | −0.002 | −0.28 |
| 32 | 28.8× | EDGEW w=8 | 39.04 | +0.0034 | +0.016 | −0.54 |
| 64 | 55.8× | EDGEW w=4 | 35.57 | +0.0029 | −0.016 | −0.67 |
| 64 | 55.8× | EDGEW w=8 | 35.54 | +0.0041 | +0.009 | −1.08 |
| 128 | 114× | EDGEW w=8 | 32.20 | +0.0056 | −0.019 | −1.83 |

(All ΔGMSD positive = worse; all ΔHaarPSI negative = worse. coarse-256 same sign,
smaller magnitudes.) MSE-LAG control is within ±0.05 dB / ±0.001 GMSD of UNIFORM
(confirms the known "per-block MSE-Lagrangian ≈ uniform" result).

## Why it fails (mechanism, confirmed by diagnostic)

The allocation genuinely moves — at q=64 w=8, ~393/4096 blocks go FINER and
~672 go COARSER; the spread widens with w. So the weight bites; it just doesn't
help:

1. The **HF-protecting quant matrix already operates inside every block** —
   intra-block HF/edge protection is done. The edge weight only redistributes
   the *block-level* bit budget.
2. To give the top-gradient blocks more bits at fixed total ratio, bits are
   stolen from the **many mid-gradient blocks**. Those mid blocks also contain
   edges/fibers that the GLOBAL GMSD/HaarPSI/edge-MAE score. Net global edge
   fidelity drops slightly — the harder you weight, the worse it gets.
3. Scroll volumes are edge-dense (fibers/sheets everywhere); there is little
   "pure flat" budget to harvest, so there's no global slack a block-importance
   map can exploit. This mirrors the existing rate-control finding that
   MSE-optimal block allocation is already near-flat / matches uniform on this
   data.

## Cost

* Precompute: ONE Sobel pass, 1.7–2.5 ns/vox (~30–42 ms for 256³). Cheap, as
  advertised — but buys nothing here.
* Random access: per-block q already supported ⇒ **touched=1 holds, 1.00×**
  (no change to the atom or seek model).
* No build/ctest regression: existing 7 tests stay green; bench is self-contained.

## Decision: **NIX**

Equal-ratio edge-importance block weighting does not improve any edge/perceptual
proxy on real PHerc Paris 4; it mildly degrades GMSD/HaarPSI (up to +0.006 GMSD /
−1.8e-3 HaarPSI at q128 w8) for zero PSNR change. The intra-block HF-protecting
quant matrix already captures the edge-preservation win; a per-block importance
map has no global slack to add on edge-dense scroll data.

Caveat (honesty): the binding metric is downstream ink-detection AUC, which we
cannot measure here. It is *conceivable* edge-weighting helps true ink-AUC while
hurting global GMSD/HaarPSI (it concentrates fidelity on the strongest edges at
the expense of weak ones). But on every proxy we *can* measure it is neutral-to-
negative, and the lit's headline win did not materialize on these proxies — so
this is NIX unless/until an ink-detection harness exists to re-adjudicate. Park
the idea there, not in the codec.
