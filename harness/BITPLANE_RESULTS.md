# Bit-plane / SNR-scalable coding — re-examination (branch r2-bitplane)

**Date:** 2026-06-02   **Base commit:** 70d0871   **Worktree:** vc-wt-r2-bitplane
**Status:** confirm-PARK measurement (the axis was DROPPED 2026-06-02; this quantifies the cost).

## Question

We dropped SNR/bit-plane scalability because the **resolution pyramid `0/`..`7/`
already serves progressive quality** (PLAN §2 dropped row, §6 last bullet). This
experiment quantifies the **ratio cost** of embedded, truncatable coding vs the
non-scalable table-free **RLGR** baseline so we are sure dropping it was right.

## What was built

- `src/entropy/bitplane.c` — a self-contained, table-free **embedded bit-plane
  coder** (EZW/SPIHT/EBCOT family, simplified): codes a 16³ atom's freq-scanned
  signed quantized levels MSB→LSB. Per plane: a **significance pass** (adaptive
  Golomb-Rice run code over the gaps to newly-significant coefficients + a sign
  bit each — the table-free analogue of EBCOT's context model) and a
  **refinement pass** (one bypass bit per already-significant coefficient).
  Truncating the byte stream at any point yields a valid coarser reconstruction
  (each kept plane halves the max quant error).
- `harness/bench_bitplane.c` — feeds the **identical** pipeline to both coders:
  per-axis integer **DCT-16³** → 16³ freq scan + **HF-protecting dead-zone
  qmatrix16** (slope 0.5) → entropy. Round-trips every atom (full stream),
  reports bytes / ratio / PSNR / enc+dec MB/s.
- `tests/test_bitplane.c` — full round-trip exact; SNR-scalable plane-drop error
  bound 2^b; **per-atom independence (touched=1)**. Added to ctest (passes).

Both coders use the **same quantizer**, so reconstruction PSNR is identical;
the only thing that moves is the **byte cost** of the entropy stage.

## Results — real PHerc Paris 4 (refbuild hires-256 + coarse-256, 256³ each)

HF slope 0.5, dead-zone baked into qmatrix16. RLGR = the measurable baseline.

| input      |   q | RLGR bytes | RLGR ratio | BP bytes | BP ratio | PSNR | **BP ratio penalty** |
|------------|----:|-----------:|-----------:|---------:|---------:|-----:|---------------------:|
| hires-256  |  16 |   793,687  |   21.1×    | 1,239,766|  13.5×   | 42.4 | **+56.2 %** |
| hires-256  |  32 |   413,521  |   40.6×    |   752,916|  22.3×   | 39.0 | **+82.1 %** |
| hires-256  |  64 |   219,308  |   76.5×    |   430,023|  39.0×   | 35.5 | **+96.1 %** |
| hires-256  | 128 |   118,284  |  141.8×    |   223,341|  75.1×   | 32.1 | **+88.8 %** |
| coarse-256 |  16 | 2,344,627  |    7.2×    | 2,837,754|   5.9×   | 38.9 | **+21.0 %** |
| coarse-256 |  32 | 1,240,114  |   13.5×    | 1,693,278|   9.9×   | 35.1 | **+36.5 %** |
| coarse-256 |  64 |   635,441  |   26.4×    |   984,333|  17.0×   | 31.2 | **+54.9 %** |
| coarse-256 | 128 |   306,965  |   54.7×    |   524,491|  32.0×   | 27.6 | **+70.9 %** |

**Speed (q=32):**
- hires-256: RLGR enc 938 / dec 1197 MB/s · BP enc 126 / dec 194 MB/s (≈7×/6× slower)
- coarse-256: RLGR enc 471 / dec 604 MB/s · BP enc 109 / dec 161 MB/s (≈4×/4× slower)

## Reading

- The embedded coder is **decisively worse on ratio**: +21 % to +96 % bytes, far
  beyond the "few %" embedded coding usually pays. **The penalty GROWS with the
  compression ratio** (sparser high-q payloads → the per-plane significance-map
  and per-coefficient refinement-bit overhead dominate). That is exactly the
  operating point (50–200×) that matters for scrolls.
- Why so large here: at scroll ratios the level array is overwhelmingly zeros
  with a few small magnitudes. RLGR codes that with **long adaptive zero-runs in
  one pass**. The embedded coder must instead **revisit every coefficient on
  every plane** (significance + refinement), so the same energy is spread across
  P passes — the structural cost of "make every bit independently truncatable".
  A full EBCOT/MQ-coder backend would narrow but not close this; embedded coding
  is fundamentally a few-%-to-tens-of-% ratio tax, and here it is tens of %.
- **PSNR is identical** (same quantizer) — the cost is pure ratio + speed, with
  **zero quality upside** over the baseline.
- **Random access is preserved** (touched=1): the stream is per-atom, so one 16³
  atom still decodes independently. Scalability would be *within* an atom's
  stream. So the cost is NOT a random-access regression; it is pure ratio/speed.

## What scalability would have bought (and why we don't need it)

A truncatable stream gives **encode-once, decode-to-any-quality** (JPEG2000
PCRD-style bitstream rate control / progressive-by-quality streaming). But we
**already have progressive quality** via the resolution pyramid (`5/`→`0/` zoom
refinement) and **per-level target-ratio** as the single quality knob. The
pyramid serves the actual interactive access pattern (screen-coverage geometry)
that in-bitstream quality LODs do *not* (PLAN §2 dropped row). So the +21–96 %
ratio tax + 4–7× slower codec buys a capability the resolution pyramid already
covers — no offsetting benefit on this data or access pattern.

## Decision: **PARK (confirmed)**

Dropping SNR/bit-plane scalability was correct. Quantified cost to re-add it:
- **Ratio:** +21 % (low-ratio coarse) to **+96 % (high-ratio hires)** bytes vs RLGR.
- **Quality:** 0 dB change (same quantizer).
- **Speed:** ~4–7× slower encode, ~4–6× slower decode.
- **Random access:** NOT broken (per-atom stream, touched=1 holds).
- **Benefit:** progressive-by-quality / bitstream truncation — **already covered**
  by the resolution pyramid for the real access pattern.

Keep `bitplane.c` + the bench in the tree as the documented negative result; do
not wire it into the codec or the default entropy path.
