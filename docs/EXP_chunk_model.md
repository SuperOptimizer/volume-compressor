# Experiment Spec: Block-Grid Model — recovering ratio while keeping 16³ random access

> **Foundational model (user 2026-06-02): the 16³ block is THE ONE TRUE ATOM; everything else
> is RELATIVE to it.** The atom is the decode = transform (DCT-16³) = cache = access = quant =
> prediction-node unit. Fixed/absolute. The data is ONE continuous volume; the codec operates
> on a **global 3D lattice of 16³ atoms**, with prediction & entropy context flowing across it
> by a **stencil**, INDEPENDENT of where chunk walls fall. A **chunk is purely "N atoms bundled
> for I/O"** (shared header + seek directory) — NOT a hard codec wall. "Do we need a neighbor
> atom's data to decode this one?" is a *data-availability* question (halo/fetch), NOT a coding
> boundary. Three orthogonal swept axes fall out, plus an I/O knob:
> - **(A) prediction stencil** — how much spatial neighborhood the coding uses: {none, 6-conn, 18-conn, 26-conn}.
> - **(B) atom traversal order** — the order atoms are coded across the grid: {raster ZYX, Morton/Z-order, Hilbert}.
> - **(C) edge data-availability** — how neighbor data is supplied at a storage boundary: {self-contained, halo, fetch-neighbor}.
> - **chunk size** = atoms-per-bundle (4³=64³, 8³=128³, …) — a separate I/O-amortization knob.
>
> **Two distinct "zigzags," both tested:** (1) coefficient scan WITHIN a 16³ atom — 3D-zigzag by
> (u+v+w) frequency so quantized-zero HF coeffs form long runs for the entropy coder (classic
> DCT ratio trick; entropy agent builds this). (2) atom TRAVERSAL order across the grid (axis B)
> — a space-filling curve (Morton/Hilbert) so consecutively-coded atoms are spatial neighbors →
> causal prediction/context lands on true neighbors (ratio↑) + neighborhood fetches contiguous
> (I/O↑). Morton recursion = "2×2×2 groups of atoms" maps a chunk to a contiguous curve run
> cleanly; seek directory still gives O(1) random access (index by Morton code). A,B interact:
> a space-filling order makes a *causal* stencil far more effective (more stencil neighbors are
> already-coded + near).

**Status:** queued (depends on the entropy bake-off landing first — see §1).
**Owner:** a Phase-1 agent, after the entropy experiment commits.
**One-line question:** Can a *large chunk of shared-context-but-individually-seekable 16³ blocks* recover near-large-chunk ratio (the ~10%-loss regime) while keeping each 16³ block independently decodable — instead of the ~50% ratio cliff that *fully-independent* 16³ blocks suffer?

---

## 0. Why this experiment exists (the settled facts it builds on)

- **16³ is the universal decode atom** (= cache block = the DCT-16³ transform block). Users extract 16³ blocks; we must be able to decode one 16³ block cheaply without decoding a whole chunk. This is an *invariant*, not under test.
- **Transform = integer DCT-16³** (won the transform bake-off: best PSNR/MS-SSIM/GMSD and *least* blocking at 50–100× on real PHerc Paris 4). Block-based, so 16³ decode is naturally possible.
- **Chunk size is ratio-neutral by itself** (chunk-size sweep: flat 32³→128³) — because compaction comes from the 16³ block, not the chunk. So a large chunk only earns its keep if it lets blocks **share entropy context** to recover ratio.
- **Prior c4d measurement (do NOT re-measure the cliff):** fully-*independent* 16³ chunks ≈ **−50% ratio** vs 64³; 256³→64³ only ≈ **−10%**. The cliff is per-unit overhead (own entropy table + own DC + no cross-block reuse) amortized over just 4096 voxels. **Independent-16³ is OFF THE TABLE.**

The crux tension: **more shared state → more ratio, but more you must "replay" to decode one 16³ block.** This experiment maps that curve and picks the sweet spot.

---

## 1. Dependency: the entropy coders define the sharing modes

"Shared context" *is* an entropy-coder property, so this experiment runs **on top of** the entropy bake-off's coders (`rice`, `rlgr`, `rans_static`) — wait for that to commit. The sharing modes map onto coder features:

| Sharing mode | Mechanism | 16³-decode cost | Expected ratio |
|---|---|---|---|
| **M0 independent-16³** (reference cliff, measure briefly to confirm) | each 16³ block self-contained: own table/param + own byte range | trivial (block bytes only) | ~−50% (the cliff) |
| **M1 shared-static-table** *(leading candidate)* | one entropy table/param set per *chunk*, stored once in chunk header; each 16³ block entropy-coded into its **own seekable byte range** against that shared table; per-block DC kept but cheap | cheap: load chunk header (table) + block byte range | target ≈ −10% (recovers the table-overhead part) |
| **M2 shared-adaptive-state** | coder adaptation carries across blocks in scan order | expensive: must replay blocks 0..N−1 to decode block N | best ratio, but **breaks cheap random access** (upper-bound reference only) |
| **M3a 6-connected prediction, INTRA-CHUNK only** *(strong candidate — user-proposed)* | each 16³ block predicted from its ≤6 face-neighbors *within the same chunk*; face-blocks at the chunk boundary fall back to no/intra prediction | decode ≤7 blocks per target; **cached + neighborhood access → ~1× amortized** (the 6 neighbors are blocks you'll want next anyway) | **YES — fully self-contained, no cross-chunk fetch, no halo** |
| **M3b 6-connected prediction, with HALO** | as M3a but face-blocks predict across the chunk edge using a thin stored boundary halo | ≤7 blocks + halo read (halo is in-chunk) | yes (halo stored in chunk) |
| **M3c 6-connected, FETCH neighbor chunks** | face/edge/corner blocks fetch 1–3 neighbor chunks' edges | ≤7 blocks + neighbor-chunk fetch | no (cross-chunk dependency) |

**Revised hypothesis (after user note 2026-06-02):** cross-block prediction is NOT off the table — it's a spectrum, and "decode ~7 blocks to get 1" is absorbed by the cache + neighborhood access pattern (≈1× amortized when sweeping a region; the 7× only bites for a truly isolated single-block fetch, which is rare). Key geometric fact: **the bigger the chunk, the larger the fraction of pure-INTERIOR blocks** (all 6 neighbors in-chunk → full prediction, zero cross-chunk dependency) — so prediction is *another* reason a large chunk earns its keep. **M3a is the likely sweet spot**: keeps chunks fully self-contained (no neighbor fetch, no halo storage), gains prediction on all interior blocks, decode cost absorbed by cache. M1 = the no-prediction baseline; M3b = recover face-block prediction via stored halo (small redundancy); M3c probably not worth the cross-chunk dependency; M2 = upper-bound reference only.
**Decision number additions:** ratio(M3a)/ratio(M1) — how much does intra-chunk prediction buy? And the *amortized* 16³ decode cost of M3a under a neighborhood-sweep access trace (not isolated single-block). If M3a buys meaningful ratio at ~1× amortized decode, it beats M1.

---

## 2. The chunk container layout to implement (M1)

A chunk = a header + N seekable 16³-block payloads:

```
chunk:
  [chunk header]
    - dims / n_blocks (Nx,Ny,Nz of 16³ blocks)
    - shared entropy table OR shared coder param(s)   (the thing amortized)
    - per-block directory: N × {byte offset, length, q, ABSENT flag}   (enables 16³ seek)
    - optional shared DC reference (see §4)
  [16³ block 0 payload]  ← DCT-16³ coeffs, quantized+scanned, entropy-coded vs the shared table
  [16³ block 1 payload]
  ...
```

Decode-one-block(bx,by,bz): read chunk header (cached after first touch) → look up block (bx,by,bz) in the directory → entropy-decode that byte range against the shared table → inverse-quant → inverse DCT-16³ → done. **No other block touched.** ABSENT blocks (all-zero) cost 0 payload bytes (directory flag only).

This is the concrete realization of "16³ random-access decode" + "shared context." The per-block directory is the seek index; the shared table is the ratio recovery.

---

## 3. What to measure (on real PHerc Paris 4)

Cross **{rice, rlgr, rans_static}** × **{M0, M1}** (and M2 once, for the upper-bound reference) × **chunk size {64³, 128³, 256³}**, on the standard corpus (8 hires 128³ center chunks + a coarse sub-region), at q targeting ~10×/20×/50×.

Report per cell:
- **Compression ratio** (the headline — how close does M1 get to the c4d ~−10% regime vs the −50% cliff?).
- **16³-decode cost**: time to decode ONE random 16³ block, and bytes that must be read to do so (block bytes + shared-header bytes). M1 should be ≪ full-chunk decode.
- PSNR / MS-SSIM / **GMSD** (blocking) — should be ~unchanged across models (model affects *coding*, not the transform), so this is a sanity check.
- enc / dec MB/s (full-volume).
- For M2 (reference only): ratio gain over M1, and the replay cost to decode one block (to quantify what random access costs us).

**The decision number:** `ratio(M1) / ratio(big-independent-chunk)`. If M1 lands within ~5–10% of a 64³/256³ chunk's ratio while keeping cheap 16³ seek, M1 is the design. If M1 still bleeds toward the −50% cliff (i.e. shared *tables* weren't the dominant overhead for *this* coder), escalate: try shared DC + a shared scan/context that's still seek-compatible, and report how much of the gap remains.

---

## 4. Sub-questions to settle inside the experiment

1. **Which coder benefits most from sharing?** `rans_static` has explicit per-block *tables* → M1 (one shared table/chunk) should help it a LOT (this is exactly c4d's region-table-sharing win). `rlgr`/`rice` are table-free (backward-adaptive params) → their per-16³-block penalty is mostly *adaptation warmup*, not tables → M1's benefit is smaller, but so was their independent-16³ penalty. **This determines whether the answer is "shared-table rANS" or "table-free RLGR with a shared seed param."** Report both.
2. **Per-block DC**: independent per-16³ DC subtraction causes DC steps at 16³ faces (banding). Test a **shared/predicted DC reference** in the chunk header (block stores a small delta) vs independent per-block DC — does it cut GMSD/seam-step at near-zero ratio cost?
3. **Chunk-size knee**: c4d saw 256³→64³ ≈ −10%. With M1, where does the amortization curve flatten? Pick the chunk size at the knee (likely 64³–128³) — big enough to amortize the shared header, not so big that fetch granularity / padding waste hurts.
4. **Directory overhead**: N × {offset,len,q,flags} per chunk. At 16³ blocks in a 128³ chunk that's 512 entries — quantify the directory's byte cost as a fraction of payload; compress it if needed (it's cheap: monotonic offsets → delta-code).
5. **DC sub-volume (JPEG-XL "DC frame" trick)** *(added 2026-06-02 from lit. review)*: code ALL atoms' DC together as a separate, globally-predicted DC SUB-VOLUME (one DC level/atom, an Az·Ay·Ax mini-volume, raster left/up/front-predicted + rANS), decoded ONCE; each atom then stores AC only and looks up its DC. Decouples DC prediction from atom decode order → preserves O(1) 16³ random access. **SETTLED** (`cfg.dc_subvol`, see `harness/CHUNKMODEL_RESULTS.md`): ratio-neutral vs M1 on this DCT-16³ pipeline (the shared table already captures the DC redundancy the DC-frame would), but it is the recommended design — touched=1 / amortized-decode=1.00 with strictly simpler random access (no prediction-ancestor cone) and banding-free DC by construction. Cross-atom **AC** stencils (6/18/26-conn) confirmed ~0 ratio gain — dropped.

---

## 5. Hard mandate (PLAN §7)
Pure C23, no hand intrinsics, compiler-autovectorizable, single-threaded reentrant, heap/arena scratch (never chunk-sized stack), lossy/u8/3D. Don't break existing round-trip tests; add tests for the chunk container (encode chunk → decode one random 16³ block → matches full decode of that block). Implement chunk models as swappable `VC_CHUNKMODEL` blocks; don't touch transform/metrics files.

---

## 6. Expected outcome / why it matters
This is the experiment that decides the **storage hierarchy**: 16³ decode atom ⊂ chunk (shared-context container) ⊂ member (LOD). It answers "how big should chunks be and what do blocks share," with the hard constraint that 16³ stays individually decodable. The likely result (to be confirmed): **M1 shared-static-table (or table-free-shared-param), chunk ≈ 64³–128³, shared/predicted DC** — recovering near-64³ ratio at cheap 16³ random access, giving up only cross-block prediction (which is incompatible with random access anyway).
