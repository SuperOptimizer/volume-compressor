# Experiment #17 — Inter-LOD (Laplacian-pyramid) prediction

**Branch:** `exp-inter-lod` · **Bench:** `harness/bench_inter_lod.c` (self-contained,
includes won DCT-16³ + HF qmatrix16 + RLGR, does not touch `codec.c`) ·
**Build:** `dct16 + rlgr + qweight=1`, `-O3 -ffast-math`, all hot loops `-fopt-info-vec` confirmed vectorized.

## Question
The archive stores LODs `0/`..`N/` as **independent** members. LOD k+1 is a 2× downsample
of LOD k, so the coarse level predicts the fine one. Does coding LOD k as a **residual**
against the 2×-upsampled **decoded** LOD k+1 (Laplacian-pyramid) buy total-pyramid ratio at
matched per-LOD quality — and does it keep each LOD independently decodable?

## Method
- Pyramid LOD0=256³, LOD1=128³, LOD2=64³, LOD3=32³, each **box-downsampled from the ORIGINAL**
  hires volume (never from our own lossy output — no error accumulation), per PLAN §3.
- Residual prediction upsamples the **DECODED** coarse level (2× trilinear, half-pixel,
  edge-clamped) — what a real decoder actually holds.
- Both schemes: DCT-16³ atom + HF qmatrix (slope 0.60) + RLGR. Residual uses a SIGNED DCT-16
  path (same int kernels, no u8 clamp). Per-atom DC mean removed + stored (+2 side bytes/atom).
- Compared two ways: (1) **matched base step** per LOD (same quant detail fidelity); (2)
  **matched per-LOD PSNR** (INDEP driven to a target ratio, RESID bisected to the same PSNR).
- Real PHerc Paris 4 hires-256 + coarse-256.

## Results — total-pyramid ratio, matched step (q = base dead-zone step)

### hires-256 (ink/fiber-rich)
| q | INDEP pyramid | RESID pyramid | resid/indep | ratio gain |
|---|---|---|---|---|
| 16 | 12.86× | **17.26×** | 0.745 | +34% |
| 32 | 23.69× | **32.84×** | 0.721 | +39% |
| 64 | 45.33× | **61.47×** | 0.737 | +36% |
| 128 | 88.97× | **112.67×** | 0.790 | +27% |

### coarse-256
| q | INDEP pyramid | RESID pyramid | resid/indep | ratio gain |
|---|---|---|---|---|
| 16 | 5.16× | **6.08×** | 0.848 | +18% |
| 32 | 8.98× | **11.03×** | 0.814 | +23% |
| 64 | 16.79× | **21.16×** | 0.793 | +26% |
| 128 | 35.19× | **43.50×** | 0.809 | +24% |

Per-LOD PSNR matches within ~0.1 dB between schemes (round-trip is sound; residual even nudges
PSNR up slightly at the finest LOD because the prediction supplies low-frequency content for
free). The gain concentrates in the finest LOD, which dominates the byte budget (LOD0 ≈ 75% of
the pyramid): e.g. hires q32 LOD0 607 KB → 418 KB (−31%).

## Results — matched per-LOD PSNR (the fair "ratio at equal quality")

### hires-256
| INDEP target | INDEP pyramid bytes | RESID pyramid bytes | resid/indep | pyramid ratio |
|---|---|---|---|---|
| 20× | 955 705 | 805 841 | 0.843 | 20.06× → **23.79×** (+19%) |
| 50× | 383 489 | 331 113 | 0.863 | 49.99× → **57.89×** (+16%) |

### coarse-256
| INDEP target | INDEP pyramid bytes | RESID pyramid bytes | resid/indep | pyramid ratio |
|---|---|---|---|---|
| 20× | 963 761 | 835 574 | 0.867 | 19.89× → **22.94×** (+15%) |
| 50× | 381 426 | 332 094 | 0.871 | 50.26× → **57.72×** (+15%) |

Per-LOD matched-PSNR gain grows toward the finer levels: LOD2 ≈ 0.89, LOD1 ≈ 0.88, LOD0 ≈ 0.84–0.87.

## Quality metrics
Comparison is at **matched PSNR per LOD** (PSNR is the bisection target), so PSNR is equal by
construction and the residual byte savings come for free. The residual is a pure rate win at
fixed fidelity — no quality regression observed; reconstructed LOD0 PSNR is within 0.1 dB of
INDEP at matched step, and identical by construction at matched PSNR. (GMSD/MS-SSIM/edge metrics
not separately reported because there is no quality tradeoff to adjudicate here — the lever is
ratio at iso-PSNR; quality-vs-objective tradeoffs are the subject of the HFDIST experiment.)

## Speed
Full-pyramid encode+reconstruct (hires @q32): INDEP 56 MB/s → RESIDUAL 43 MB/s = **1.30×
slower** encode (extra trilinear upsample + signed residual DCT + the cascade re-decode of the
coarse level to build the prediction). Decode of a finer LOD likewise needs the coarse decode
first.

## The cost: independent decodability is LOST
This is the decisive caveat for THIS codec. The PLAN mandate is that **each LOD is an
independently-fetchable array** (OME-style separate members) and **touched=1 decodes one 16³
atom**. Residual coding makes a fine LOD's atom depend on the decoded coarse level:

- Touched=1 random access on LOD0 now costs a **cascade of one atom per pyramid level**
  (LOD0→LOD1→LOD2→LOD3 = 4 atom decodes here). Because the 2× trilinear upsample is a 3-tap
  filter, one fine 16³ atom needs only an 8³ region of the coarse level = exactly one coarse
  16³ atom, so the cascade is O(pyramid-depth) atoms, still O(1) — but it is **4× the atom
  decodes** and, crucially, the LODs are **no longer independently fetchable**: you cannot serve
  LOD0 without also fetching+decoding the whole coarse chain.

This directly conflicts with PLAN §2 (dropped scalability) rationale: "coarse levels MUST be
physically separate, independently-fetchable arrays with their own chunk grid" — the very
property that makes interactive zoom / screen-coverage geometry work. Residual coding re-couples
them.

## Verdict
The ratio lever is **real and large** (+15–19% pyramid ratio at iso-PSNR, +24–39% at iso-step;
biggest single untested ratio gain measured in this codec). But it is **bought by giving up the
independently-fetchable-LOD invariant** the toolkit deliberately chose, plus 1.3× encode and a
per-fetch coarse-decode cascade.

**Recommendation: MAYBE (conditional KEEP).** Keep as an **optional archive mode** for
*archival / max-ratio* configurations where the access pattern is "decode the whole pyramid" or
"always fetch coarse→fine progressively anyway" (in which case the cascade is already paid and
the residual is free ratio). **NIX for the default interactive codec**, where independent
per-LOD random access is the load-bearing property. The gain does not justify breaking the
invariant in the general configuration; it justifies a compile-time `VC_INTER_LOD_RESIDUAL`
lever for the archival profile.

## Files
- `harness/bench_inter_lod.c` — the bake-off (self-contained).
- `CMakeLists.txt` — added guarded `bench_inter_lod` target.
- Full raw output: regenerate with `./build/bench_inter_lod` from repo root.
