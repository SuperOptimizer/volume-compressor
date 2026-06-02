# DC-DPCM experiment (branch r2-dc-dpcm)

**Question.** Our per-atom DC value is carried out-of-band in the per-chunk seek
directory, coded *raw* (zig-zag varint of the absolute quantized DC level). DC is
a smooth, spatially correlated field, so predictive coding (DPCM) of the DC field
across atoms should cut the DC residual a lot (lit: ~75%). Does that translate to
a measurable **total-stream** ratio gain, and does it keep 16^3 random access at
`touched==1`?

**Stack.** integer DCT-16^3 atom + HF-protecting 16^3 quant + dead-zone +
M1 box chunk (ca8 = 128^3 bundle) + per-16^3 rate control. **RLGR** entropy as the
measurable baseline. Real PHerc Paris 4: `hires-256`, `coarse-256` (256^3 each).
q (dead-zone base step) ∈ {16,32,64,128}.

## What "DC-DPCM" already is here

The winning box-M1 default carries the per-atom DC as an absolute level in the seek
directory (`dc_resid_q`, `was_intra=1`). The codebase already contains the exact
"directory-level DPCM the decoder reconstructs once" design the experiment calls
for: the **DC sub-volume** mode (`cfg.dc_subvol`, JPEG-XL "DC frame"). It gathers
every atom's quantized DC level into an `Az*Ay*Ax` mini-volume, predicts each from
its already-decoded **raster left/up/front** neighbours (causal-mean), and rANS-codes
the residual field **once** as a separate stream. Atom decode then does an O(1)
lookup into the decoded grid — **no ancestor cone, touched stays 1.**

So this experiment is (a) a head-to-head ratio measurement of that path vs the raw
baseline, and (b) a direct probe of the DC field to attribute the result. I added a
**planar predictor** variant (`L+U+F − UL−UF−FL + ULF`) and a per-predictor cost
probe (`vc_bg_get_dc_probe`, bench mode `dc`) to measure the residual reduction and
the DC fraction of the stream in both an entropy-coded and a directory-varint model.

## Result 1 — integrated total-stream ratio (RLGR), bench_chunkmodel

| q | input | RLGR raw DC (baseline) | RLGR DC sub-volume (DPCM) | Δ ratio | PSNR (both) |
|---|---|---|---|---|---|
| 16 | hires-256 | **20.5** | 20.5 | 0.0% | 42.52 |
| 16 | coarse-256 | **6.9** | 6.9 | 0.0% | 39.08 |
| 32 | hires-256 | **39.0** | 39.0 | 0.0% | 39.05 |
| 32 | coarse-256 | **13.2** | 13.2 | 0.0% | 35.13 |
| 64 | hires-256 | **72.1** | 71.9 | −0.3% | 35.52 |
| 64 | coarse-256 | **25.4** | 25.4 | 0.0% | 31.31 |
| 128 | hires-256 | **131.0** | 128.2 | −2.1% | 32.13 |
| 128 | coarse-256 | **52.4** | 52.0 | −0.8% | 32.13/27.67 |

Quality (PSNR / MS-SSIM / GMSD) is **bit-identical** between the two paths — DC
reconstruction is lossless level-residual coding in both, so this is a pure
rate question. The DPCM (DC sub-volume) path delivers **no ratio gain, and slight
regressions at high q.** Encode/decode speeds are within noise (dec ~90–135 MB/s,
enc ~45–78 MB/s) and `avgTouch == 1.00`, `amort == 1.00`, `us/atom` unchanged —
random access is preserved exactly.

## Result 2 — DC field probe (bench_chunkmodel `dc` mode): why

The residual-reduction claim *reproduces strongly*, but the field is tiny.

| q | input | DC raw-varint % of total | causal sum\|resid\| vs raw | best DPCM saves (% of total) |
|---|---|---|---|---|
| 16 | hires | 1.00% | **13.3%** (−86.7%) | 0.42% |
| 16 | coarse | 0.34% | **9.0%** (−91.0%) | 0.16% |
| 64 | hires | 3.10% | 13.3% | 1.57% |
| 64 | coarse | 1.21% | 9.0% | 0.72% |
| 128 | hires | 3.32% | 13.3% | 0.98% |
| 128 | coarse | 1.27% | 9.0% | 0.43% |

- **Causal-mean (raster left/up/front) DPCM cuts the DC residual magnitude to
  9–13% of raw (≈87–91% reduction)** — exceeds the cited ~75%. The smooth-field
  premise is correct.
- **The planar predictor is worse than causal-mean** here (sum\|resid\| 15.6–16.4%
  of raw, and higher order-0 entropy). The 3D cross-term adds noise; the simple
  causal mean wins. So no reason to add planar.
- **DC is only 0.3–3.3% of the total stream.** Even a perfect-free DC stream caps
  the total gain at that, and the realistic best-DPCM saving is **0.16–1.57% of
  total** in an idealized accounting that ignores stream framing.
- The integrated path (Result 1) realizes **none** of even that idealized saving and
  regresses at high q: the separate rANS DC stream pays a per-stream 12-bit
  frequency **table (~512 B)** and the directory still carries a 1-byte-per-atom
  varint floor, so the small win is eaten by framing overhead. The raw varint
  directory the box-M1 default already uses is, in practice, near-optimal for a
  field this small.

## Random access / touched=1

`vc_test_chunkmodel` passes (incl. the DC sub-volume path): `RA_match=1`,
`maxtouch=1`. The bench confirms `avgTouch=1.00`, `amort=1.00` for both raw and
DPCM modes. DC-DPCM as implemented (decoder reconstructs the DC grid once, O(1)
lookup per atom) does **not** make atom decode depend on neighbour-atom decode.
Invariant held.

## Tests

All ctest green except the **pre-existing** `ratectrl` failure (a rate-control
allocation-accuracy tolerance test on real data; present at base commit 70d0871,
untouched by this experiment which only adds DC analysis to `blockgrid.{c,h}` and a
`dc` mode to `bench_chunkmodel.c`). New analysis loops verified vectorized via
`-fopt-info-vec`. Pure C23, libc/libm only, heap scratch.

## Decision

**NIX (for the box-M1 default) / already-PARKED as `dc_subvol`.**
DC-DPCM is real (≈87–91% DC-residual reduction) but the DC field is 0.3–3.3% of the
stream, so the ceiling is ~1.5% of total and the implemented separate-stream path
captures ~0% (slight regression at high q from table framing). Not worth a default
change. The mechanism already lives behind `cfg.dc_subvol` for banding-free /
O(1)-DC use; keep it available there, do not enable it for ratio. The raw zig-zag
varint DC in the seek directory stays the default.

**Decision numbers.** Ratio Δ: 0.0% (q16/32) to −2.1% (q128 hires). PSNR/MS-SSIM/
GMSD: unchanged (bit-identical recon). Speed: within noise. Random access: preserved
(touched=1, amort=1.00). Cost to ship: negative (regression) — do not adopt.
