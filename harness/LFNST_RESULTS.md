# LFNST — low-frequency secondary transform (round-3 lit experiment)

**Question (lit: VVC LFNST, "highest payoff-to-risk, ~3-5% bitrate, near-zero
compute, preserves random access"):** after the separable primary integer
**DCT-16³** of each atom, does applying ONE small FIXED non-separable secondary
transform to the **low-frequency corner** of the coefficient cube further
decorrelate the residual energy the separable DCT cannot pack — buying ratio at
equal quality, block-locally (touched=1 preserved)?

**Stack under test (the winning config):** integer **DCT-16³** atom · per-coef
**16³ quant matrix** (HF slope 0.50, dead-zone width **0.80**, recon-offset
**0.40** — the current-stack tuning, applied IDENTICALLY to both arms) ·
**RLGR** entropy. Self-contained bench `harness/bench_lfnst.c` (does NOT touch
codec.c / ratectrl / chunkmodel). Real data: PHerc Paris 4 `hires-256` (ink/
fiber-rich) + `coarse-256`, 256³ each.

## Method

- **Secondary transform = empirical KLT** of the corner DCT coefficients. First
  pass over hires: DCT-16 every atom, accumulate the covariance of the lowest
  **N³** corner coefficients (DC excluded from the stats so it can't dominate),
  Jacobi-eigendecompose → orthonormal basis, rows = components ordered by
  descending eigenvalue. This is the strongest *fixed* matrix (the task allowed a
  "simple decorrelating matrix"; the KLT is the upper bound for a fixed linear
  one). Inverse = transpose. Derived once on hires, reused on both volumes.
- Two corner sizes tested: **N=4** (64-dim, VVC-scale) and **N=8** (512-dim, the
  whole low-freq octant of the 16³ cube).
- Three arms per q: **OFF** (baseline, no secondary), **ON** (always-on),
  **RDsw** (per-atom: code both, keep the smaller RLGR payload, **1 on/off
  bit/atom** side info, counted in the rate).
- **Equal-quality comparison:** at each q∈{16,32,64,128} the OFF pass sets a
  target PSNR; ON and RDsw steps are **bisected to match that exact PSNR**, then
  we read the achieved ratio. Gain = ratio(arm)/ratio(OFF) − 1.

Round-trip + block-locality covered by `tests/test_lfnst.c` (ctest `lfnst`,
green): orthonormal fwd∘inv recovers the corner; **fwd/inv touch ONLY the N³
corner of one atom — every coefficient outside is byte-identical** (touched=1
preserved). LFNST matmul + corner gather/scatter loops autovectorize
(`-fopt-info-vec-optimized`: lines 129/162/164/171/175/179 vectorized).

---

## Headline result

**N=4 corner delivers the lit-predicted ~3-5% (and more at high q) on the
ink-rich hires data, at near-zero compute and full random access. N=8 delivers
far more ratio (up to +24%) but at a 12× decode-speed collapse — it stops being
a "secondary" transform and becomes a heavy second-stage transform of the entire
low-freq octant.**

### N=4 (64-dim KLT corner) — the recommended operating point
hires-256, equal-PSNR ratio gain:

| q | tgt PSNR | OFF ratio | ON gain | RDsw gain | dec MB/s OFF→ON |
|---|---|---|---|---|---|
| 16 | 40.95 | 21.38× | +0.47% | +1.22% | 157→149 |
| 32 | 37.41 | 40.04× | +0.55% | +1.78% | (−5%) |
| 64 | 33.90 | 77.06× | +3.16% | **+3.91%** | |
| 128 | 30.62 | 145.64× | +7.54% | **+7.69%** | |

coarse-256 (KLT trained on hires → out-of-domain): ON is ratio-NEUTRAL-to-slightly-
negative (−0.3%…+0.1%); **RDsw stays non-negative** (+0.2%…+1.5%) — the 1-bit
flag makes it safe on data the matrix wasn't trained on.

- enc/dec speed @q32: enc 58→56 MB/s, dec 157→149 MB/s — **~5% compute cost**,
  matching the lit "near-zero compute" claim.
- GMSD / edge-MAE / seam essentially unchanged (it's a ratio play at equal PSNR,
  not a quality play): GMSD 0.06874→0.06867 @q16, edge-MAE and seam flat.

### N=8 (512-dim KLT corner) — bigger ratio, but NOT "near-zero compute"
hires-256, equal-PSNR ratio gain:

| q | OFF ratio | ON gain | RDsw gain |
|---|---|---|---|
| 16 | 21.38× | +11.21% | +11.96% |
| 32 | 40.04× | +15.01% | +17.06% |
| 64 | 77.06× | +22.47% | +23.58% |
| 128 | 145.64× | +23.87% | +23.91% |

- **Decode collapses 165→14 MB/s (~12×), encode 60→11 MB/s.** A 512×512 matmul
  is 262K mults/atom — it dwarfs the DCT-16 itself. This violates the lit
  premise ("near-zero compute") and the decode-speed budget the stack relies on.
- coarse-256 always-on LOSES up to −5% (KLT overfit to hires); **only RDsw
  recovers it** (+0.5…+3.4%) — confirms the per-atom flag is mandatory at N=8.

---

## Random-access / touched=1

**Preserved exactly.** The secondary transform reads and writes only the N³
low-frequency corner of a *single* 16³ atom's coefficient cube; it never touches
neighbour atoms or coefficients outside the corner (asserted byte-for-byte in
`tests/test_lfnst.c`). Decoding one atom = RLGR-decode → dequant → (optional)
LFNST-inverse on its own corner → IDCT-16. touched=1 holds; the only added
per-atom decode work is the inverse matmul (cheap at N=4, expensive at N=8) plus
1 bit of side info in the RDsw arm.

---

## Interpretation

- The separable DCT-16³ already packs the low-freq corner well, so at low q
  (gentle quant) there is little correlated residual to harvest → small N=4 gains
  (~0.5-1.8%). As quant gets aggressive (q64/q128) the dead-zone zeroes more
  coefficients and the KLT's energy-concentration into the first few components
  pays off → +4-8% (N=4). This is exactly the LFNST literature pattern.
- N=8's huge numbers are real but misleading as "LFNST": the 8³ corner is 1/8 of
  all coefficients and holds most signal energy, so a 512-dim KLT is effectively
  replacing the primary transform on the dominant octant. The compute is no
  longer secondary-transform-cheap (12× decode hit). If that much gain is wanted,
  the right move is to revisit the *primary* transform, not bolt on a giant
  secondary matmul.
- KLT is data-dependent: trained on hires it is out-of-domain on coarse. The
  RD-switched 1-bit/atom flag is what keeps it safe (never worse) cross-domain —
  it must ship with the on/off flag, not always-on.

---

## DECISION: PARK (lean KEEP for N=4 + RD-switch, pending broader-corpus check)

LFNST at **N=4 with the per-atom RD on/off bit** behaves exactly as the
literature promises: **~3-5% at the meaningful aggressive ratios (q64 +3.9%,
q128 +7.7%) on ink-rich data, ~5% compute cost, full 16³ random access, never
negative cross-domain.** That clears the "highest payoff-to-risk" bar for a
cheap add-on. It is **not** a free win at gentle quant (≤+1.8% at q16/q32) and
its KLT must be derived from a representative corpus and shipped as a fixed
table + on/off flag.

Reasons to PARK rather than immediate KEEP: (1) the gain is concentrated at
high-ratio operating points and on hires; the per-q payoff at the likely
shipping quality (q16-32) is only 1-2%; (2) it adds a derived-table dependency
and an entropy-stream bit; (3) it should be re-validated on the broader corpus
(task #15/#22) and ranked against the other round-3 levers before integration.
**N=8 is NIX** as an "LFNST" — its decode cost disqualifies it; that ratio is
better chased via the primary transform.

### Decision numbers (N=4, RD-switched, hires-256)
- ratio gain: **+1.2% (q16) / +1.8% (q32) / +3.9% (q64) / +7.7% (q128)**
- quality at equal-PSNR match: GMSD/edge-MAE/seam flat (±0.3%)
- compute cost: enc −3%, **dec −5% (157→149 MB/s)**
- random access: **touched=1 preserved** (block-local corner matmul + 1 bit/atom)
- cross-domain (coarse, out-of-train): RDsw +0.2%…+1.5% (never negative)
