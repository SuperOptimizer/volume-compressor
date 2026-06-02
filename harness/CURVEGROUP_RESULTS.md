# Curve-group experiment — results

**Question (user-proposed):** replace the per-128³-CHUNK-BOX shared-table/fetch unit
with a **"CURVE GROUP" = N consecutive 16³ atoms along a space-filling curve**
(Morton/Hilbert). Hypothesis: a curve-segment groups spatially-adjacent → statistically-
similar atoms more coherently than an arbitrary axis-aligned box, so a per-curve-group
shared table could be TIGHTER → better ratio at equal group size.

**Engine:** `src/chunkmodel/blockgrid.c`, curve path (`cfg.group_mode = VC_GROUP_CURVE`)
— a GLOBAL space-filling curve over the whole 16³-atom lattice, cut into runs of N
atoms; each run is a *group* with its own shared rANS table (or table-free RLGR) +
per-atom seek directory. NO chunk boxes. Random access: atom (z,y,x) → curve index →
`group_of[]` lookup → that group's header (cached) → decode that one atom's bytes.
DC level carried per-atom in the directory ⇒ **touched = 1** always.

**Stack held fixed (the winner from CHUNKMODEL_RESULTS):** DCT-16³ + HF-protecting
dead-zone quant + shared rANS table (and table-free RLGR), **NO prediction stencil,
NO DC sub-volume**. Only the GROUPING SHAPE varies. Bench: `bench_chunkmodel . <q> curve`.
Inputs: real PHerc Paris 4 **hires-256** + **coarse-256**. q ∈ {16,32,64,128}.

PSNR / MS-SSIM / GMSD are **identical across all grouping shapes** at each q (grouping
changes only *coding*, not transform/quant) — so ratio-at-equal-PSNR reduces to plain
ratio. All cells: **touched = 1**, **amortized 4³-box decode = 1.00** (no neighbor-atom
decode), verified in `vc_test_chunkmodel` (round-trip + random-access == full-decode).

## A. Headline: curve-group vs box-128³ ratio, equal group size (FIXED-N, FULL table)

Ratio (hires-256 / coarse-256). Box-128³ at ca8 = an 8³=512-atom axis-aligned box, so
N=512 is the equal-group-size comparison.

| q | box rans | curveHil N512 | curveMor N512 | box rlgr | curveHil N512 rlgr* |
|--:|---------:|--------------:|--------------:|---------:|--------------------:|
| 16  | 12.9 / 5.8  | 12.9 / 5.8  | 12.9 / 5.8  | 20.5 / 6.9  | 20.5 / 6.9 |
| 32  | 21.9 / 9.8  | 21.9 / 9.8  | 21.9 / 9.8  | 39.0 / 13.2 | 39.0 / 13.2 |
| 64  | 38.7 / 17.2 | 38.7 / 17.2 | 38.7 / 17.2 | 72.1 / 25.4 | 72.1 / 25.4 |
| 128 | 70.9 / 33.1 | 70.9 / 33.1 | 70.9 / 33.1 | 131.0 / 52.4| 131.0 / 52.4 |

\* RLGR rows used N=256 (table-free; N is immaterial — see below).

**ratio(curve-group N=512) / ratio(box-128³) = 1.000 at every q, both inputs, both
coders, Morton and Hilbert alike.** The hypothesis is NOT confirmed: at equal group
size a Hilbert/Morton curve-run produces a shared table of the same tightness as the
axis-aligned box. The DCT-16³ atom histograms are dominated by HF zero-runs whose
statistics are already near-uniform across the 256³ volume, so the box is *not* an
arbitrary-enough cut to lose anything to a curve run. Same outcome class as cross-atom
prediction in CHUNKMODEL_RESULTS: **~0**.

## B. Smaller groups (N<512): pure curve-grouping LOSES on header cost

Smaller groups → more group headers → more shared-table bytes. With a FULL table per
group this *costs* ratio (rANS), and the loss grows with q (sparser payloads, table
amortizes worse):

| q | box rans | curve N256 | curve N64 | N64 hdr% (vs box) |
|--:|---------:|-----------:|----------:|------------------:|
| 16  | 12.9 | 13.1 | 13.3 | 4.1% (1.6%) |
| 32  | 21.9 | 22.3 | 22.4 | 7.0% (2.8%) |
| 64  | 38.7 | 39.0 | 38.1 | 11.4% (4.0%) |
| 128 | 70.9 | 70.6 | 65.7 | 19.5% (5.8%) |

(hires; Hilbert. Morton identical to ±0.1×.) At low q the tiny per-table side-cost is
amortized and small N even edges +3% (more coherent local stats *do* help the payload);
but by q128 the FULL-table header is 19.5% of payload and N=64 loses **−7.3%**. The
"tighter local table" payload gain is real but small and is swamped by header cost.

**RLGR (table-free) is EXACTLY ratio-invariant to N and to box-vs-curve** at every q
(20.5/39.0/72.1/131.0 hires, identical to box) — there is no table to pay for, so the
grouping shape is irrelevant. For RLGR, curve-groups buy precisely 0.

## C. The redundancy hierarchy that makes coherent small groups affordable

To rescue the small-group payload gain, the experiment added (curve-mode only) three
mechanisms that cut the per-group *header* so small, coherent groups stop losing:
- **E1 table-DELTA** — store group g's freq table as a zig-zag-varint delta from g−1.
- **E2 base+delta** — one whole-lattice base table stored once; each group a delta off it.
- **I1 DRIFT boundaries** — variable-size groups that end when the running histogram
  diverges past `drift_thresh` (cap = group_n); internally-homogeneous groups.
- **B1 DC curve-predecessor prediction** (marginal-gain probe).

Ratio vs box-128³ rans (hires / coarse), best small-N variants:

| q | box | curve N64 FULL | **E1 N64 delta** | **E1 N128/256 delta** | I1+E1 drift.15 |
|--:|----:|---------------:|-----------------:|----------------------:|----------------:|
| 16  | 12.9/5.8  | 13.3/5.8  | 13.5/5.9 | 13.3/5.9 (N128) | 12.9/5.8 |
| 32  | 21.9/9.8  | 22.4/9.8  | 22.9/9.8 | 22.7/9.8 (N128) | 22.0/9.8 |
| 64  | 38.7/17.2 | 38.1/16.9 | 39.5/17.2 | **39.6/17.3 (N128)** | 38.9/17.2 |
| 128 | 70.9/33.1 | 65.7/31.7 | 70.1/32.7 | **71.7/33.2 (N256)** | 71.5/33.2 |

E1 delta-table coding converts the N=64 loss into a small **win**: at q64 hires the
plain-FULL N=64 was −1.6%, E1-delta N=64 is **+2.1%** (39.5 vs 38.7) and E1-delta N=128
is **+2.3%** (39.6); at q128 E1-delta N=256 is **+1.1%** (71.7). DELTA cuts the stored-
table cost ~3× (e.g. q128 N=64 table 15.4%→7.8% of payload). E2 base ≈ E1 (within 0.1×).
DRIFT alone is ratio-neutral (on this homogeneous data it rarely splits before the cap,
so ngroups ≈ box's 8); DRIFT+E1 lands at +0.8–1.0%. **B1 DC-curve-prediction buys ~0
ratio (≤+0.3×) and is dropped** — and it makes single-atom extraction ~4× slower (it
walks the intra-group DC chain at decode), see §D.

**Net ceiling of the whole hierarchy: ~+2–2.5% ratio over box (rANS only), at q≥64.**
Real but modest, and only for rANS. For RLGR (the CHUNKMODEL lead coder) it stays 0.

## D. Extraction cost: curve indexing vs box

Single-atom random-access decode, avg over 200 random atoms (µs/atom, touched, hires):

| variant                 | touched | µs/atom | vs box |
|-------------------------|:-------:|--------:|-------:|
| box-128 rans            | 1 | 38–44 | 1.00× |
| box-128 rlgr            | 1 | 27–35 | 1.00× |
| curve Morton/Hilbert N* | 1 | 37–47 | ≈1.0× |
| E1 / E2 / I1 / I1+E1     | 1 | 37–43 | ≈1.0× |
| **B1/B2 DC-curve-pred** | 1 | **147–161** | **~4×** |

Curve indexing was expected to cost ~2–3× the raster lookup (per the chunk-model note).
In practice the curve index (`morton3`/`hilbert_d`) + `group_of[]` lookup is computed once
per extraction and is **dwarfed by the inverse DCT-16³ + entropy decode**, so curve-group
extraction is **within noise of box** (≈1.0×). The ONE thing that is genuinely pricier is
**B1 DC-curve-predecessor prediction**: reconstructing an atom's absolute DC from the
intra-group residual chain re-walks the group's directory, ~4× the µs — for ~0 ratio. So
the "curve indexing is pricier" worry does not materialize for the grouping itself; only
the DC-chaining option is expensive, and it is not worth keeping.

## Decision numbers

- **ratio(curve-group)/ratio(box-128³) at equal group size + equal PSNR = 1.000**
  (every q, both inputs, both coders, Morton≈Hilbert). The core hypothesis — a curve
  run is a tighter shared-table scope than the box — is **not supported**.
- **Best ratio the curve path can buy** (small N + E1 delta-table, rANS only):
  **≈+2 to +2.5%** at q≥64, fading to ≈+1% by q128, ≈+1–4% at q16; **0% for RLGR**.
- **Extraction cost delta: ≈1.0× vs box** (curve indexing is negligible next to the
  iDCT+entropy decode). The only expensive option (B1 DC-curve-pred, ~4×) is ratio-dead.
- touched = 1 and amortized-4³-decode = 1.00 preserved by every curve variant.

## Recommendation

**Neutral-on-ratio → the box wins on simplicity/speed; BUT curve-group is the right
"no-box" unification if the user wants it, at ~equal numbers — and ship it with RLGR.**

1. The hypothesis is ~0 (like cross-atom prediction was): at equal group size a curve
   run and an axis-aligned box yield the same shared-table tightness on this DCT-16³
   pipeline. There is **no meaningful ratio reason to switch.**
2. If the conceptual unification ("one continuous lattice of 16³ atoms; groups are runs
   along a curve; no chunk-box concept") is preferred, **curve-group with RLGR** is the
   clean choice: it is **ratio-identical to box-RLGR at every q**, table-free (group size
   irrelevant, no header tuning), keeps touched=1 / amort=1.00, and extraction cost is
   within noise of box. This satisfies the user's "stop thinking in chunk-boxes" with
   zero ratio penalty.
3. If sticking with rANS and squeezing the last ~2% matters, use **small curve groups
   (N≈64–256) + E1 group-to-group table-DELTA** (the only thing that makes small coherent
   groups affordable). This is the maximum ratio the curve idea offers and it is small.
4. **Drop B1 DC-curve prediction** (~0 ratio, ~4× extraction cost). Drop the FULL-table
   small-N path (loses to header cost at high q).

Bottom line: curve-grouping does **not** beat the box on ratio (the box is already good
enough, as with prediction), so keep the box if ratio/speed is the only axis; adopt
curve-group + RLGR if you want the box concept gone — it costs nothing.

## Reproduce

```
cmake --build build --target bench_chunkmodel vc_test_chunkmodel
build/vc_test_chunkmodel                          # round-trip + random-access (curve cells incl.)
for q in 16 32 64 128; do build/bench_chunkmodel . $q curve; done
```
Full raw sweep used for this report: `harness/curvegroup_bench.txt`.
