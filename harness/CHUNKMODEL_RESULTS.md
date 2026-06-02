# Phase-1 block-grid (chunk-model) bake-off — results

Engine: `src/chunkmodel/blockgrid.c` (self-contained DCT-16³ + dead-zone + the
three orthogonal axes A/B/C + chunk-size + entropy mode, all RUNTIME). Harness:
`harness/bench_chunkmodel.c`. Inputs: **hires-256** (8 cached 128³ PHerc Paris 4
center chunks, assembled 2×2×2) and **coarse-256** (256³ crop of the coarse 384³).
`q` = dead-zone base step. The 16³ block is the fixed decode atom throughout;
every config below round-trips and decodes any single 16³ atom in isolation
(`vc_test_chunkmodel`: RA_match=1, maxtouch=1 for all).

Mid-experiment guidance folded in (lit. review 2026-06-02): cross-atom **AC**
prediction is low-payoff (HEVC inter-block coeff pred ~1.5–1.8% BD-rate; JPEG
neighbor-block AC ~4.5%); the cheap win is **DC**, and the highest payoff-to-risk
form is the **JPEG-XL "DC frame" trick** — code all atoms' DC together as a
separate, globally-predicted DC sub-volume, decoupled from the atom decode order.
That variant is now implemented (`cfg.dc_subvol`) and measured first-class here.

## The variants compared (cols: ratio / PSNR / GMSD / amortized 16³-decode cost)

- **(a) no prediction** — `M1-rans none` / `M1-rlgr none`: shared per-chunk table,
  per-atom DC carried in the seek directory, AC coded vs the shared table.
- **M0-indep** — the cliff reference: each 16³ atom its own rANS table.
- **(b) 6-connected full-coefficient prediction** — `rans 6 raster self` etc.
- **DC-sub-volume variant (the requested first-class cell)** — `rans DCsv none` /
  `rlgr DCsv none`: **M1 shared rANS table + separate globally-predicted DC
  sub-volume + per-atom AC only** (AC stencil NONE).
- references **c3d / c4d** — from the entropy bake-off (`harness/BAKEOFF_RESULTS.md`),
  ratio-at-PSNR.

## Headline numbers (hires-256)

Ratio at each `q` (PSNR shown once — it is identical across coding models, as
expected: the model changes *coding*, not the transform/quant):

| config                         | q8 | q16 | q32 | q64 | q128 |
|--------------------------------|---:|----:|----:|----:|-----:|
| PSNR (dB)                      |45.7|42.5 |39.1 |35.5 |32.1 |
| **M0-indep none** (the cliff)  | 4.2| 5.2 | 6.1 | 6.8 | 7.3 |
| **(a) M1-rans none**           | 8.0|12.9 |21.9 |38.7 |70.9 |
| **(a) M1-rlgr none**           |10.8|20.5 |39.0 |72.1 |131.0|
| **(b) rans 6-conn**            | 8.0|12.9 |22.0 |39.0 |71.0 |
| rans 18-conn                   | 8.0|12.9 |22.0 |39.0 |71.0 |
| rans 26-conn                   | 8.0|12.9 |22.0 |39.0 |71.0 |
| **rans DCsv none** (DC-frame)  | 8.0|12.8 |21.9 |38.7 |70.9 |
| **rlgr DCsv none** (DC-frame)  |10.7|20.5 |39.0 |71.9 |131.0|
| M1-rans none, *noDC* shared    | 8.0|12.9 |21.9 |38.7 |70.9 |

coarse-256 tells the same story (M0→M1 jump, stencils/DCsv ratio-neutral): e.g.
q64 M0=5.5×, M1-rans=17.2×, rans 6-conn=17.3×, rans DCsv=17.2×, rlgr DCsv=25.4×.

All non-M0 rows above have **amortized 16³-decode = 1.00** (one AC entropy-decode
per atom) and single-atom **touched = 1**. Encode ~70–88 MB/s, decode ~95–162 MB/s
single-thread (RLGR decodes ~1.5× faster than rANS, consistent with the entropy
bake-off).

### vs references (ratio-at-PSNR, from BAKEOFF_RESULTS)

| input  | 38 dB | 40 dB |
|--------|------:|------:|
| ref-c3d (hires) | 51.3 | 32.4 |
| ref-c4d (hires) | 39.4 | 25.7 |
| **our M1-rlgr / rlgr-DCsv (hires)** | ≈30 (interp q≈36) | ≈22 |
| **our M1-rans / rans-DCsv (hires)** | ≈18 | ≈14 |

(Interpolated between the q grid above. The chunk-model's M1/DCsv ratios track the
main codec's `rans-hf`/`rlgr-hf` entropy bake-off numbers — same coders — so the
residual gap to c3d/c4d is the **transform** (CDF-9/7 wavelet vs DCT-16³), not the
chunk model.)

## Findings

1. **The 16³ cliff is a SHARED-TABLE problem, and M1 fixes ~all of it.** M0-indep
   → M1-rans is the dominant jump everywhere: hires q64 **6.8× → 38.7× (5.7×)**,
   coarse q64 5.5× → 17.2×. This is exactly c4d's region-table-sharing win,
   reproduced. Shared-static-table per chunk + per-atom seekable byte ranges
   recovers near-big-chunk ratio while keeping each 16³ atom independently
   decodable (touched=1).

2. **Cross-atom AC prediction buys ~0 — confirms the literature.** 6/18/26-conn
   are within ±0.3% of M1-none at every q on both inputs (e.g. q64 hires: M1-none
   38.7× vs 6-conn 39.0× vs 26-conn 39.0×). 18- and 26-connected add nothing over
   6-conn and *cost* decode work (their isolated `us/atom` is higher). Do not
   invest in AC stencils. Recommendation: AC stencil = **NONE**.

3. **The DC sub-volume (JPEG-XL DC-frame) is RATIO-NEUTRAL here but is the right
   architecture.** `rans DCsv none` matches `M1-rans none` to the third digit at
   every q (it neither closes nor opens a gap), because on this DCT-16³ pipeline
   the per-atom DC is already a single fine-quantized coefficient whose residual
   the shared rANS table codes near-optimally — there is little cross-atom DC
   redundancy *left* for a global DC predictor to capture once the shared table is
   in place. The expected "DC residual −75%" win from the lit. review is already
   absorbed by M1. **However**, the DC-frame variant is still preferable as the
   shipped design: it (i) keeps DC reconstruction a pure **O(1) lookup** into a
   small (Az·Ay·Ax) decoded sub-volume — no prediction-ancestor cone at all,
   strictly simpler random access than the directory-threaded DC path; (ii)
   guarantees no DC banding at 16³ faces by construction (global DC field); (iii)
   adds negligible bytes (`dc_subvol_bytes`, a single rANS-coded mini-volume).
   It does NOT, on its own, close the gap to c4d — that gap is the transform.

4. **"Shared DC" toggle is a no-op for ratio** (`M1-rans none ca8 noDC` ==
   `M1-rans none ca8`), consistent with (3): the DC handling is not where the bytes
   are at these operating points; HF AC zero-runs dominate the rate.

5. **Chunk size is ratio-flat** (ca4 ≈ ca8 ≈ ca16, all within ~1%), reconfirming
   the chunk-size sweep: compaction comes from the 16³ atom + shared table, not the
   bundle size. Pick chunk size on I/O grounds (≈64³–128³ = ca4–ca8).

6. **RLGR > rANS on ratio for this whole-chunk path** at the high-ratio end (q64
   hires: rlgr 72.1× vs rans 38.7×) and decodes faster — because the per-chunk
   rANS *table* side-cost is non-trivial relative to the very sparse high-q atom
   payloads, while RLGR is table-free. (At low q the table amortizes and they
   converge.) For the block-grid container, **RLGR + DC-sub-volume** is the
   balanced lead; **rANS + DC-sub-volume** only if a future table-sharing-across-
   chunks step removes the per-chunk table cost.

## Direct c4d comparison on the IDENTICAL hires-256 cube (more rigorous than the interpolated table above)

The §"vs references" table above reuses the 8³-era BAKEOFF_RESULTS interpolation,
whose PSNR scale differs and which predates the 16³ atom — it understates us. Re-running
**c4d on the exact same hires_256.u8** (`harness/refbuild`, c4d's own q scale, matched by
PSNR) gives the apples-to-apples curve:

c4d hires: q4 7.8×@46.6dB · q8 16.6×@42.1 · q16 38.9×@38.1 · q24 61.1×@35.8 · q32 83×@34.1.

| PSNR (dB) | c4d | **M1-rlgr-16³ (ours)** | M1-rans-16³ |
|----------:|----:|-----------------------:|------------:|
| ~43.9 | ~12 | **15.6** | 10.5 |
| ~40.5 | ~22 | **29.9** | 17.5 |
| ~38.1 | 38.9 | **~43** | ~22 |
| ~37.0 | ~50 | **55.9** | 30.3 |
| ~35.5 | ~62 | **72–73** | 38.7 |

coarse-256 ~40 dB: **M1-rlgr-16³ = 6.9× vs c4d = 5.2×** (c4d q8 = 40.2 dB).

**Conclusion correction: the ~2× gap to c4d the 8³ entropy bake-off left is CLOSED.**
M1-RLGR at the **16³ atom** matches or slightly beats c4d at equal PSNR on both inputs.
The win is the **atom size (16³, longer HF zero-runs) + table-free RLGR + shared-context
chunk** — NOT cross-atom prediction (ratio-neutral) and NOT a wavelet transform. The
"remaining gap is the transform" note above applied to the 8³ rANS path; at 16³ + RLGR
there is no remaining gap to c4d to speak of. (c3d, 256³ wavelet atom, still leads at the
very-high-quality end; the seam-y assembled hires cube understates it, per BAKEOFF notes.)

**Directory/table cost breakdown (probe, hires q32):** payload 96.8–97.9 % · seek directory
3.2 % (RLGR) / 2.1 % (rANS), → 1.6 % with prediction · shared rANS table 0.5 % at ca8 /
0.1 % at ca16 · DC sub-volume +1.1 %. M0's per-atom table is the cliff; M1's shared table
is ~free.

## Recommendation (chunk-model design)

**Shared static table per chunk (M1) + separate globally-predicted DC sub-volume
+ AC stencil NONE.** Chunk ≈ ca8 (128³). Coder: **RLGR** (balanced/max-ratio here)
or rANS (if/when tables are shared across chunks). This keeps the 16³ atom
independently decodable at touched=1 / amortized-decode=1.00, recovers essentially
all of the M0→big-chunk ratio cliff via the shared table, and gets banding-free DC
+ trivially simple random access from the DC frame. The remaining gap to c3d/c4d is
the transform (wavelet), not the chunk model — AC cross-atom prediction is
confirmed dead weight and is dropped.

## Reproduce

```
cmake --build build --target bench_chunkmodel vc_test_chunkmodel
build/vc_test_chunkmodel                       # round-trip + random-access invariants
for q in 4 8 16 32 64 128; do build/bench_chunkmodel . $q; done
```
