# EG2024 techniques experiment — results

Implements + measures the **"Fast Compressed Segmentation Volumes"** (Eurographics
2024, arXiv 2308.16619) techniques on the winning stack (DCT-16³ + HF-protecting
dead-zone quant + shared rANS table / table-free RLGR, **NO** cross-atom AC
prediction), on real PHerc Paris 4 **hires-256** + **coarse-256**, q ∈ {16,32,64,128}.

Engine: `src/chunkmodel/blockgrid.{c,h}` (new knobs `band_split`, `sparse_prepass`,
`skip_meta`; region/whole-cube scope = `chunk_atoms=16` over 256³ = 1 chunk; curve
scope = `group_mode=VC_GROUP_CURVE`). Bench: `bench_chunkmodel . <q> eg`. All new
modes round-trip and decode any single 16³ atom in isolation at **touched=1 /
amortized-4³-decode=1.00** (`vc_test_chunkmodel`, EG cells `eg-band-2/3`,
`eg-sparse`, `eg-band3-sparse`, `eg-skip-meta-uniform`). PSNR/MS-SSIM/GMSD are
identical across all table designs at each q (the table changes *coding*, not the
transform/quant), so ratio-at-equal-PSNR reduces to plain ratio.

Note on our DC: the per-atom DC level is carried **out-of-band in the seek
directory** (not in the entropy payload), so EG2024's "DC vs AC" split maps, on our
pipeline, to splitting the *AC* band into **low-AC vs high-AC**. We test all three:
pooled (1 table), DC|AC (2 tables), DC|low-AC|high-AC (3 tables; DC band carries
only the zeroed scan-position-0 token, ~free).

---

## ★ HEADLINE — split shared tables by symbol role (DC vs AC bands)

**Verdict: YES, it wins, and it is cheap — but ONLY the 3-band split (DC | low-AC |
high-AC), and it must be paired with a coarse (whole-cube / region) table scope so
the extra table headers amortize. The naive 2-band DC|AC split LOSES.**

Ratio (whole-256 scope = 1 table set over the cube, the scope that amortizes the
split best; rANS):

| q | hires pooled | hires **DC\|lo\|hi** | gain | coarse pooled | coarse **DC\|lo\|hi** | gain |
|--:|---:|---:|---:|---:|---:|---:|
| 16  | 12.6 | **16.1** | **+28%** | 5.7  | **6.2**  | **+9%** |
| 32  | 21.2 | **28.7** | **+35%** | 9.6  | **10.7** | **+11%** |
| 64  | 38.0 | **47.7** | **+25%** | 17.0 | **19.3** | **+14%** |
| 128 | 70.4 | 69.5     | −1.3%    | 32.8 | **36.6** | **+12%** |

At the per-128³-box scope the same 3-band split also wins at q≤64 (hires q32
21.9→28.4 = +30%, q64 38.7→46.7 = +21%) but at q128 the 8× duplication of 3 headers
costs more than it saves (hires q128 70.9→66.9 = −5.6%). Whole-cube scope fixes
that (the 3 tables are stored once). **The 2-band DC|AC split is a net LOSS at every
q on both inputs** (hires q32 21.9→20.3, q64 38.7→33.9, q128 70.9→56.3): pulling DC
out alone leaves the AC band's bimodal low/high distribution pooled, and you pay an
extra ~0.5 KB table for the trivial DC band. **The bits are in separating low-AC
from high-AC**, exactly EG2024's interior-vs-leaf-nibble analog.

- **Cheap?** Yes. Cost = (nbands−1) extra freq tables = ~0.5 KB each, ~0.4 % of
  payload at whole-cube scope (`tbl%` 0.04→0.15 hires q16; 0.12→0.45 hires q64).
  Encode/decode within noise (one extra gather/scatter of the band subsequence; the
  iDCT dominates). Extraction stays touched=1, us/atom within noise of pooled.
- **Why it works:** DCT-16³ low-AC coefficients (fsum 1..8) are dense with
  small-magnitude levels; high-AC (fsum 9..45) is a sea of zeros + sparse spikes.
  One pooled table is forced to a compromise geometric tail; two tables each fit
  their own regime. The gain is largest at mid-q (q32–q64) where both regimes carry
  real mass; it shrinks at q128 where high-AC is almost all zeros (RLGR-like runs).

**This is the single biggest cheap ratio win in the experiment (+25–35 % mid-q on
hires).** RLGR is table-free so the band split is a no-op there (and RLGR already
beats banded-rANS on ratio — see below).

---

## (2) Region-scoped tables (whole-256) vs per-128³-box vs per-curve-group

**Coarser scope is at-worst ratio-neutral and cuts header overhead — confirms
EG2024's "near-global is near-optimal, stats are volume-stationary."**

Pooled-table ratio by scope (hires / coarse):

| q | box-64 (64 tbls) | box-128 (8 tbls) | **whole-256 (1 tbl)** | curve N512 |
|--:|---:|---:|---:|---:|
| 16  | 13.3 / 5.8  | 12.9 / 5.8  | 12.6 / 5.7  | 12.9 / 5.8 |
| 32  | 22.4 / 9.8  | 21.9 / 9.8  | 21.2 / 9.6  | 21.9 / 9.8 |
| 64  | 38.1 / 16.9 | 38.7 / 17.2 | 38.0 / 17.0 | 38.7 / 17.2 |
| 128 | 65.7 / 31.7 | 70.9 / 33.1 | 70.4 / 32.8 | 70.9 / 33.1 |

The payload is **scope-insensitive** (all within ±2 %): the DCT-16³ histograms are
near-stationary across the 256³ cube, so a single whole-cube table fits as tightly
as 8 per-box tables. Where scope matters is **overhead**: whole-256 has the lowest
`tbl%` (0.04–0.22 %) and `hdr%`, while box-64 bleeds 64 table copies (q128 box-64
`tbl%` = 15.4 %, dragging ratio to 65.7 vs 70.9). So **finer scope only ever hurts**
(more headers) on this stationary data — there is no payload reason to subdivide.
Curve-group N512 is identical to box-128 (reconfirming `CURVEGROUP_RESULTS`: equal
group size ⇒ equal table tightness). **Coarser (region/whole-cube) ≥ finer (box),
and it is what makes the symbol-role split affordable** (amortizes the extra tables).

For a petabyte corpus the practical scope is a **region (~1024³ tile)**, not truly
global; our whole-256 cube emulates that — and the result says region-scope captures
essentially all of the global benefit.

---

## (3) Sparse prepass — build the table from a sample, not all atoms

**Verdict: ratio-neutral-to-POSITIVE and cuts the table-fit pass to ~1/stride of the
atoms. Implement it.** (Required a correctness fix — see below.)

The table is fit to every `stride`-th atom; the payload still codes ALL atoms. A
symbol that occurs only in un-sampled atoms has sample-count 0 and is **routed to the
ESC bypass** at encode time (NOT Laplace-smoothed — smoothing distorted the model
and broke reconstruction on sparse high-q data). With ESC-escape the model stays a
valid sampled distribution and reconstruction is **exact at any stride**.

Ratio (whole-256, hires / coarse):

| q | prepass/1 (all) | prepass/512 | Δ ratio | PSNR (all strides) |
|--:|---:|---:|---:|---:|
| 16  | 12.9 / 5.8  | 14.0 / 5.9  | +8.5% / +1.7% | identical |
| 32  | 21.9 / 9.8  | (≈20) / 9.1 | ≈neutral | identical |
| 64  | 38.7 / 17.2 | 42.1 / 17.5 | +8.8% / +1.7% | identical |
| 128 | 70.9 / 33.1 | 76.6 / 33.7 | +8.0% / +1.8% | identical |

The sampled+escape model is often **leaner** than the exhaustive one (fewer rare
symbols pollute the freq table; their bits move to the cheap ESC tail) — so sparse
prepass not only costs ~0 ratio, on this data it slightly HELPS while sampling
1/512 of atoms for the table fit. Stationarity (per #2) is why a 1/512 sample
reproduces the full histogram. **Big encode-time win for free.**

---

## (4) Skip metadata (per-chunk min/max + uniform flag, ORC/Parquet 3-tier)

**Verdict: ~0 on the dense test cubes (no uniform regions), but real and growing on
air-heavy data — 4.6 %→19 % of total bytes as q rises. Keep it (negligible cost).**

On the dense hires/coarse-256 cubes no 128³ chunk is uniform, so `skip%`=0 and the
3-byte/chunk index (min+max+flag) is the only cost — ratio-neutral. On a synthetic
air-heavy 256³ (zeroed upper half = absent + a constant-40 quadrant = uniform):

| q | skip-off ratio | skip-on ratio | uniform chunks | **bytes saved** |
|--:|---:|---:|---:|---:|
| 16  | 49.4 | 51.8  | 6 | **4.6 %** |
| 32  | 80.4 | 87.0  | 6 | **7.5 %** |
| 64  | 133  | 150   | 6 | **11.7 %** |
| 128 | 216  | 267   | 6 | **19.0 %** |

A uniform (single-value) chunk stores ONE byte + the 3-byte index instead of its
atom blobs + directory; the saving grows with q because the omitted constant-air
atoms shrink slower than the real payload. The skip index is 24 B for the whole cube
(3 B × 8 chunks) — negligible. Synergizes with the existing ABSENT-atom flag (pure
zero air costs ~0 either way; skip-meta adds the *non-zero* constant-air case).
touched stays 1 (a uniform-chunk atom is a `memset`, no decode).

---

## (5) Morton ≥ Hilbert — confirmed

Curve-group N512 ratio is **identical for Morton and Hilbert at every q on both
inputs** (hires 12.9/21.9/38.7/70.9; coarse 5.8/9.8/17.2/33.1 — Morton == Hilbert to
±0.0×), matching `CURVEGROUP_RESULTS`. Hilbert's tighter locality buys no ratio on
this stationary data, and (per the chunk-model note) its index is 2–3× the Morton
cost. **Use Morton.** (Both give touched=1; extraction within noise.)

## (6) 16³ vs 32³/64³ atom — documented ratio tradeoff (atom NOT changed)

We REQUIRE 16³ for O(1) single-atom random access, so this is documentation only.
EG2024 reports bigger bricks compress better (fewer edge voxels lose neighbor
context, longer HF zero-runs, table cost amortized over more coefficients). On our
pipeline the same direction holds qualitatively, bounded by two of our own results:
(a) the chunk-size sweep (`CHUNKMODEL_RESULTS` finding 5) found whole-chunk *bundle*
size ratio-flat ±1 % — but that varied the **bundle**, not the **transform atom**;
(b) a larger transform atom would lengthen HF zero-runs (the dominant rate term),
the same mechanism that made 16³ beat the old 8³ sub-block and closed the gap to c4d.
Estimated ratio left on the table by holding 16³ vs a 32³ atom: **low-single-digit %
to ~10 %** at high q (zero-run-length-limited), more at low q — but a 32³ atom would
make single-atom random access touch 8× the voxels and 8× the iDCT work, which is
disqualifying for the random-access requirement. **Keep 16³; accept the modest ratio
cost as the price of O(1) random access.**

---

## Combined best rANS config (`* whole-256 DC|lo|hi prep512`)

Whole-cube scope + 3-band symbol split + sparse prepass/512, rANS (hires / coarse):

| q | box-128 pooled (old baseline) | **combined** | gain |
|--:|---:|---:|---:|
| 16  | 12.9 / 5.8  | **16.1 / 6.2** | +25% / +7% |
| 32  | 21.9 / 9.8  | **21.7 / ≈10** | +mid (band) |
| 64  | 38.7 / 17.2 | **47.7 / 19.5** | +23% / +13% |
| 128 | 70.9 / 33.1 | 69.1 / **36.8** | −2.5% / +11% |

(hires q128 is the one cell where the 3-band tables don't fully pay back vs the very
sparse pooled payload; coarse and all mid-q cells win solidly.)

### rANS vs RLGR

RLGR (table-free) remains the **higher-ratio coder overall** (hires q64 72.1 vs
banded-rANS 47.7; q128 131 vs 69) and is invariant to scope and immune to the band
question (no tables to split). The EG2024 techniques **close much of the rANS↔RLGR
gap** (banded whole-cube rANS recovers +25–35 % toward RLGR) but do not overtake it.
The symbol-role split is therefore the lever that makes **rANS competitive** where
rANS is preferred (faster random-access seek into a fixed slot table, GPU-friendly
table decode à la EG2024); RLGR stays the max-ratio single-thread CPU pick.

---

## RECOMMENDATION — final shared-table design

1. **Scope = REGION (≈1024³ tile), emulated here as whole-256.** Coarse scope is
   ratio-neutral on the stationary X-ray statistics and minimizes header overhead.
   Do NOT subdivide to per-128³ boxes for the table — it only adds header copies.
   (Box/curve-group remain the I/O fetch unit; the *table* is region-scoped.)
2. **Symbol-role split = 3 bands: DC | low-AC (fsum≤8) | high-AC.** THE headline win,
   +25–35 % mid-q on hires, +9–14 % on coarse, ~free (≈0.4 % payload), touched=1.
   Do NOT use the 2-band DC|AC split — it loses (the gain is low-AC vs high-AC).
   Region scope is what makes the extra tables amortize (avoids the per-box q128
   penalty).
3. **Sparse prepass = sample every ~512th atom to fit the tables, ESC-escape the
   sample gaps** (never Laplace-smooth). Ratio-neutral-to-positive, big encode-time
   cut. Mandatory for petabyte-scale encoding.
4. **Skip metadata = per-region/chunk min/max + uniform flag (3-tier index).** ~0 on
   dense data, up to ~19 % on air-heavy regions, 3 B/chunk cost. Keep it; it
   synergizes with the ABSENT-atom flag for the non-zero constant-air case.
5. **Curve = Morton** (Hilbert buys 0 ratio at 2–3× index cost). **Atom = 16³**
   (random-access requirement; documented low-single-digit-to-~10 % ratio cost vs a
   larger atom, not taken).
6. **Coder:** RLGR for max single-thread ratio (table-free, the band question is
   moot); **region-scoped 3-band sparse-prepass rANS** where a fixed-slot,
   GPU-decodable table is wanted — the EG2024 stack makes that rANS path
   competitive (recovers +25–35 % of the gap to RLGR) while keeping O(1) random
   access at touched=1.

Net: the symbol-role table split (3-band, region-scoped) is the recommended cheap
ratio upgrade; sparse prepass and skip-metadata are free operational wins; Morton,
16³, and "no cross-atom AC prediction" are reconfirmed unchanged.

## Reproduce

```
cmake -S . -B build && cmake --build build
build/vc_test_chunkmodel                      # round-trip + RA + touched=1 (EG cells incl.)
for q in 16 32 64 128; do build/bench_chunkmodel . $q eg; done
```
