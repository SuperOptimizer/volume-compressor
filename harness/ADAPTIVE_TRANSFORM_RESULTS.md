# EXP#20 — Per-atom adaptive transform/mode selection

**Branch:** `exp-adaptive-transform`  ·  **Date:** 2026-06-02
**Verdict: NIX for the max-ratio (RLGR) codec. MAYBE/marginal for a rANS build.**

## What was built

A per-atom (2-bit) mode flag in the block-grid box path, stored in the seek
directory, letting each 16³ atom pick the cheapest of three coding modes by an
R-D trial (rate-bytes + λ·SSD, distortion measured on the *decoded* atom, not a
Parseval estimate):

- **DCT** (default) — the won HF-quant DCT-16³ AC levels via the shared coder.
- **SKIP** — DC-only: store no AC, reconstruct flat = the atom's DC value.
  (air / uniform escape)
- **RAW** — table-free zig-zag-LEB128 bypass of the AC levels, run-length-coded
  zeros, no shared table. (incompressible / outlier escape)

Code: `src/chunkmodel/blockgrid.{c,h}` (cfg fields `adaptive`, `adaptive_lambda`;
`raw_enc/raw_dec`, `atom_ssd`, mode dispatch in encode Pass 2 + `decode_one_atom_levels`).
`adaptive=0` is byte-for-byte the old DCT-only behaviour (flag costs nothing).
Bench: `harness/bench_adaptive.c` (real hires/coarse-256 + a synthetic mixed
air+smooth+noise 256³ to stress SKIP/RAW). `atom_ssd` autovectorizes
(`-fopt-info-vec`, 16-byte vectors). All round-trips exact (rt=1) and single-atom
random access stays **touched=1** in every cell — the invariant is preserved.

## Numbers vs the always-DCT baseline (box ca8 = 128³ chunk, dc_subvol)

Ratio (PSNR/MS-SSIM/GMSD are unchanged by adaptive on real data — same recon):

| q | input | RLGR base | RLGR adaptive(best λ) | rANS base | rANS adaptive(best λ) |
|---|---|---|---|---|---|
| 16 | hires | 20.49 | 20.50 (≈0) | 12.84 | 13.27 (+3.3%, −1.0 dB PSNR) |
| 16 | coarse | 6.94 | 6.93 (−0.1%) | 5.79 | 5.79 (0) |
| 32 | hires | 39.00 | 38.90 (−0.3%) | 21.94 | 22.00 (+0.3%) |
| 64 | hires | 71.92 | 71.61 (−0.4%) | 38.71 | 38.80 (+0.2%) |
| 64 | coarse | 25.38 | 25.34 (−0.2%) | 17.19 | 17.18 (0) |
| 128 | hires | 128.16 | 127.28 (−0.7%) | 70.11 | 70.79 (+1.0%) |
| 128 | coarse | 51.96 | 51.80 (−0.3%) | 32.93 | 32.86 (−0.2%) |

Mode histogram on real data: **DCT ≈ 99–100%** everywhere. SKIP fires <1%
(the `dc_subvol` + absent-atom path already absorbs air, leaving almost no
uniform-but-nonzero atoms). RAW fires only against the **rANS shared table**
(7–13% at high q on hires), where it rescues atoms the chunk-pooled table
mis-models — but those same atoms are already cheap under RLGR, which is the
codec winner and far ahead of rANS in absolute ratio.

### Synthetic mixed (air + smooth slab + pure-noise block) — the stress case

| q | RLGR base→adaptive | rANS base→adaptive (RAW%) |
|---|---|---|
| 16 | 20.46 → 20.44 | 19.34 → **22.50** (+16%, 72% RAW) |
| 64 | 39.38 → 39.28 | 33.04 → **39.57** (+20%, 75% RAW) |
| 128 | 56.07 → 55.94 | 51.82 → **61.36** (+18%, 72% RAW) |

The RAW escape is real and large — **but only for rANS**, and only because a
single shared table cannot model a noise block; RLGR (table-free) already handles
that block well, so RLGR+adaptive shows no gain (it just pays the flag).

## Cost / tradeoffs

- **Ratio:** net **loss** on RLGR (the 2-bit/atom directory flag, ~0.1–0.7%);
  ≤ +1% gain on rANS real data, large gain only on pathological noise.
- **Quality:** identical reconstruction (DCT/RAW lossless-equivalent; SKIP only
  chosen when it wins R-D). No PSNR/MS-SSIM/GMSD change on real data.
- **Encode speed:** roughly **halved** (≈74→40 MB/s on hires) — the R-D trial
  does an extra inverse-DCT per atom for the SSD. Decode unaffected (≈unchanged).
- **Random access:** preserved — touched=1, exact, O(1) (RAW/SKIP are
  self-contained per-atom byte ranges; no added dependency).

## Recommendation

**NIX** for the frozen codec, which is the **RLGR** max-ratio stack: adaptive
costs encode speed and a flag for ~zero (slightly negative) real-data benefit,
because RLGR is already per-atom table-free and `dc_subvol`/absent already cover
air. The JPEG-XL-style win (RAW escape on incompressible regions) materializes
**only against shared-table rANS**, and even then the RLGR baseline is the better
codec overall, so the comparison favours "just use RLGR" over "add adaptive modes
to rANS." If a future build is forced onto shared-table rANS for some reason, the
RAW escape (λ≈0.002–0.05, all equivalent) is a cheap +1% (real) / +20% (noisy)
safety valve worth keeping there — hence MAYBE, not a hard NIX, for that niche.
