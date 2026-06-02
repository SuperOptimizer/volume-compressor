# Experiment Spec: Curve-Groups — sharing scope = a run along a space-filling curve, not an axis-aligned box

**Status:** queued/launched after the chunk-model experiment (extends `src/chunkmodel/blockgrid.{c,h}`).
**One-line question:** Does defining the shared-table / fetch unit as a **"curve group" = N consecutive
16³ atoms along a space-filling curve** (Morton/Hilbert) beat the current **axis-aligned 128³ chunk box**
on ratio — while keeping cheap single-atom random access? Eliminates the "chunk box" concept (user
preference: stop thinking in chunk-boxes).

## Background (what was and wasn't tested)

The chunk-model experiment tested traversal order (raster/Morton/Hilbert) only as input to a *causal
predictor* — and prediction bought ~0, so traversal looked useless. **But traversal was NOT tested as
the thing that defines the SHARING SCOPE.** The shared rANS table was always per-128³-box. This
experiment tests traversal-as-sharing-scope: the table/directory is shared across a **curve-run** of
atoms instead of a box of atoms.

## The hypothesis

A 128³ axis-aligned box is an arbitrary cut. A Hilbert/Morton curve-segment of the same atom count is a
**spatially compact, statistically-coherent blob** (consecutive curve atoms are spatial neighbors → more
similar statistics). A tighter, more-coherent sharing group → a tighter shared table → potentially better
ratio than the box, at equal group size. Unknown whether the gain is real or (like prediction) ~0 because
the box is already good enough. **Measure it.**

## The design to implement

- **Curve group** = N consecutive 16³ atoms along the chosen curve (test N ∈ {64, 256, 512} atoms; 64 =
  user's suggestion ≈ a 4³-atom blob).
- **Group header** = shared rANS/RLGR table (or RLGR shared seed) + per-atom seek directory {offset, len,
  q, DC, ABSENT}.
- **Decode one atom (x,y,z):** compute its curve index → find its group (the group = a contiguous
  curve-index range; lookup via a small group-index table) → load that group's header (cached) →
  entropy-decode just that atom's bytes against the group's shared table → inverse-quant → inverse
  DCT-16³. Must keep **touched = 1** (no neighbor-atom decode).

## What to measure (real PHerc Paris 4 hires-256 + coarse-256, the winning stack)

Hold everything at the winning config — **DCT-16³ + HF-protecting quant + RLGR (and rANS) + shared table +
NO prediction stencil** — and vary ONLY the grouping shape:

| Grouping | sharing scope | curve |
|---|---|---|
| **box-128³ (baseline = current M1)** | axis-aligned 8³-atom box | raster within box |
| **curve-group Morton, N atoms** | N-atom Morton-curve run | Morton |
| **curve-group Hilbert, N atoms** | N-atom Hilbert-curve run | Hilbert |

For each, at q ∈ {16,32,64,128}: **ratio, PSNR, MS-SSIM, GMSD, single-atom extraction µs, blocks touched,
group-header byte overhead (table+directory as % of payload), amortized decode over a 4³-atom neighborhood
sweep.** Compare RLGR and rANS coders.

**Decision numbers:**
- `ratio(curve-group) / ratio(box-128³)` at equal group size + equal PSNR — does curve-grouping buy ratio? By how much?
- single-atom extraction µs vs box (curve indexing costs more — quantify; chunk-model saw Morton/Hilbert lookup ~2-3× the raster µs).
- the net: if curve-groups buy meaningful ratio (say >2-3%) at acceptable extraction cost, they replace the box; if ratio-neutral (like prediction was), keep the box (simpler, faster indexing) — but note curve-groups still satisfy "no box" if the user prefers the conceptual unification at ~equal numbers.

## Improvements to test — framed as INTER-group and INTRA-group (user 2026-06-02: "inter and intra group stuff"; "I like all those ideas to test")

The curve-group design is naturally TWO-LEVEL. Test all of these as variants (cheapest-first), each measured vs the fixed-N box and fixed-N curve-group baselines:

**INTRA-group (within one group):**
- **I1. Variable-size groups by statistical drift** — instead of fixed N atoms, END a group when its running symbol histogram diverges past a threshold (MDL / context-tree-style boundary). Each group becomes internally homogeneous → tighter shared model. Costs a group-boundary index. Measure adaptive-size vs fixed-N ratio.
- **I2. Nested sub-groups** — a group base model + small per-sub-group deltas (the intra half of hierarchy).

**INTER-group (across adjacent groups):**
- **E1. Group-to-group model/table PREDICTION (highest-value)** — code group N's shared table/param as a DELTA from group N-1's (adjacent curve-groups are spatially adjacent → similar stats → tiny delta). This shrinks the per-group HEADER cost, which is exactly what makes SMALL coherent groups affordable. Directly attacks the small-group penalty. Measure header-bytes saved + the smaller-group ratio it unlocks.
- **E2. Coarse super-group base model** — a global/coarse base table all groups correct against (the inter half of hierarchy). Two-level: base (inter) + group delta (intra) + atom decodes against base⊕delta.

**INTER-BLOCK / INTER-ATOM (between neighboring 16³ atoms — user 2026-06-02 "and inter block stuff"):**
Three-level hierarchy of redundancy: intra-ATOM (DCT-16³, done) → inter-ATOM (this) → intra/inter-GROUP (above).
- **B1. DC-only curve-causal prediction** — predict each atom's DC from its CURVE-PREDECESSOR (under Hilbert/Morton order the predecessor is a guaranteed spatial neighbor, so this is cheap + rides the ordering, stays touched=1-friendly via DC-in-directory). NOTE: the chunk-model experiment already found 6/18/26-connected FULL-coefficient prediction buys ~0 under BOX sharing — BUT this is different: (a) DC-ONLY isolated from the AC prediction that failed (lit says DC is the ~75%-residual-cut high-value piece), and (b) tested under the new CURVE-GROUP + shared-model design, which interacts. Re-measure because the context changed; expect possibly-still-~0 (shared model may absorb it) but confirm.
- **B2. Whether inter-block prediction is even needed once the curve-group shared model + table-prediction (E1/E2) are in place** — these may capture the cross-atom redundancy that prediction would, making explicit prediction redundant (as happened in the box case). Report the marginal contribution of B1 ON TOP OF the best group-model variant, not in isolation.

**Curve choice for GROUPING coherence (distinct from traversal-for-prediction, which bought 0):**
- **C1. Hilbert vs Morton for group compactness** — Hilbert N-runs are more spatially compact (no long jumps) → potentially more coherent groups → tighter shared model. Measure whether the locality advantage translates to ratio HERE (it may not, like prediction didn't — but it's a different mechanism: grouping-coherence, not prediction).

**Decision:** report each variant's ratio gain vs the fixed-N-box baseline + its extraction-µs and header-overhead cost. The likely winner is a TWO-LEVEL model: coarse base (inter, E2) + per-group delta (intra) with group boundaries either fixed or drift-adaptive (I1), tables predicted group-to-group (E1) to keep headers cheap, on a Hilbert ordering (C1) if its coherence pays. But MEASURE — any of these may be ~0 like cross-atom prediction was.

## ★ Literature findings (2026-06-02) — reshape the priors; test these FIRST

Near-exact precedent: **"Fast Compressed Segmentation Volumes" (Eurographics 2024)** — 16³/32³/64³ bricks + Morton + shared rANS tables + per-brick random access + ~10 GB/s GPU decode. Validates our architecture. Key results + columnar-format (ORC/Parquet) analogy give us:

**HIGH-VALUE, CHEAP, mostly UNTESTED — do these first:**
- **★ Split shared tables by SYMBOL ROLE (DC vs AC bands)** — "likely the biggest single ratio win." DC and HF-AC have very different distributions; pooling them wastes bits. We currently pool. Test per-band tables (e.g. DC / low-AC / high-AC). CHEAP.
- **★ Table-reuse-by-ID / delta (E1) done the RANDOM-ACCESS-SAFE way** — reference a prior group's table by ID (no chained reconstruction), with periodic key-frame resets, so any atom needs ≤1 back-ref. This is how ORC/Parquet keep O(1) random access. Makes small coherent groups affordable.
- **★ Sparse prepass** — build the shared table by sampling every Nth atom (they used 512th/4096th), not scanning all. Big encode-time cut, ~0 ratio loss (stats stationary). Directly addresses encode-speed concern.
- **★ Per-group min/max + uniform-flag skip metadata (3-tier index: volume→group→atom)** — skip near-constant air regions cheaply; ORC/Parquet-style. Synergizes with our ABSENT-chunk flag.
- **GLOBAL-table baseline** — they found ~2 global tables for the WHOLE dataset nearly optimal ("nibble freqs extremely similar volume-wide"). So the baseline to beat may be ONE GLOBAL TABLE, not the 128³ box. If stats are volume-stationary, fine curve-groups buy little. TEST global-table as a first-class baseline.

**TEMPERED expectations (lit counter-results):**
- **Hilbert ≈ Morton on ratio at 2-3× encode cost** — the most-analogous system ships Morton; Hilbert only monetizes IF cross-group table-delta or cross-atom prediction is active. Lower priority for C1.
- **Adaptive group-size (I1) likely low-single-digit %** on stationary X-ray data — measure, but the global-table result suggests fine grouping has little to gain.
- **Larger atoms compress better** (64³ > 16³: fewer edge voxels lose neighbor refs) — but 16³ is our random-access requirement; note the tradeoff.

## Mandate
Pure C23, no hand intrinsics, compiler-autovectorizable, single-threaded reentrant, heap scratch, keep
ctest green, add a curve-group round-trip + random-access test (decode one atom == full decode of that
atom; touched=1). Extend `src/chunkmodel/blockgrid.{c,h}` + `harness/bench_chunkmodel.c`; write
`harness/CURVEGROUP_RESULTS.md`. Do NOT touch `src/ratectrl/` (another agent owns it).
