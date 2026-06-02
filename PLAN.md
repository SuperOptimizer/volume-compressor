# volume-compressor — Implementation Plan

> A **compression toolkit** for dense 3D u8 volumetric X-ray data (Vesuvius/Herculaneum
> scrolls). You select compression building blocks at **compile time**; the build emits
> **one small, frozen, zero-dispatch custom codec** for that configuration. The toolkit is
> the menu + build system + benchmark harness; a configured build is the product.
>
> Greenfield, pure **C23**, CPU-only, **compiler-autovectorized** (no hand intrinsics, no
> GPU, no deps beyond libc/libm). Mines lessons from c3d/c2d/c4d/c2dv; shares no code.

---

## 0. Status & stance

This is a **planning document for an experimentation platform**, not a frozen spec. The
prior design analysis + literature review + a 3-agent adversarial red-team established that
several "obvious" choices (DCT vs wavelet, RLGR vs rANS, independent vs region chunks,
per-chunk vs per-block quality, MSE vs perceptual rate control) **rest on contested priors
that may be wrong for this specific data and access pattern.** The resolution is not more
argument — it is **measurement on real scroll data, head-to-head against c3d and c4d.**

Therefore: every contested axis is a **compile-time-selectable block** behind a common
contract. We build the skeleton + harness + one end-to-end path first, reproduce/branch
against the reference codecs, then add alternative blocks one at a time, each measured as it
lands. **Nothing here is locked; the data decides; we pivot freely.**

---

## 1. Hard constraints (these do not change)

- **Data:** dense **u8** grayscale, **3D only** (no 1D/2D/n-D; video = 3D volume).
- **Lossy only.** No true-lossless mode ("approaching lossless via high quality" suffices).
- **Pure C23.** Single concrete codec per build. No GPU. **No hand intrinsics** — code must
  be written to be *compiler-autovectorizable* (see §7). Deps: libc + libm only.
- **Compile-time configuration.** A config (header/build flags) selects one implementation
  per block. The build inlines them into one monomorphic, zero-dispatch codec. The
  experiment matrix is swept by building many configs.
- **Chunk = codec atom, u8.** Chunk side is a **compile-time-configurable constant** (not
  fixed): 32³/64³/128³/256³ all valid. 64³ is the c4d-measured leading candidate, but if
  ultra-fast de/compression makes larger chunks as cheap to touch, larger may win more
  ratio + quality (fewer seams) — to be measured (see §2 "Chunk size").
- **Bespoke single-file archive.** Zarr/napari/OME interop is a NON-goal (shim later if ever).
- **Single-threaded reentrant core; caller parallelizes** across chunks. No in-core OpenMP.
- **Measured against c3d & c4d on real Vesuvius data from day one.**

---

## 2. The contested axes (compile-time levers to test)

Each is an independent block with a fixed contract and ≥2 implementations. The red-team
findings that put each in play are noted.

| Axis | Implementations to build & measure | Why contested (red-team) |
|---|---|---|
| **Transform** | integer-DCT 8³, integer-DCT 16³, float-DCT, **CDF 9/7 wavelet**, **ZFP-style lifting**, lapped/POT | Scrolls compress at 50–200× = <1 bpp = the regime lit says wavelet beats DCT & DCT blocks. "ZFP uses DCT" was FALSE (it uses multiply-free lifting). Must verify transform choice at *real* scroll ratios, not assume. |
| **Quantizer** | uniform, dead-zone, per-subband-weighted, **HF-protecting weights** | HF-weighting changes the RC objective (see RC row). Dead-zone ≠ round() — affects distortion estimate. |
| **Entropy** | **RLGR**, plain Rice, **rANS (static)**, **rANS (interleaved)**, bypass/bit-pack | "Rice within a few % of rANS" is FALSE at high ratio (LOCO-ANS: 37.6% worse at aggressive quant). RLGR's "20%" is vs fixed-Rice, not rANS. SIMD-Rice decode is bit-serial/slow per lit. rANS may win both ratio AND speed at scroll ratios. |
| **Chunk/dependency model** | independent, **region-grouped (shared ctx, 256³)**, independent + **halo-decode**, cross-chunk-predicted | Access is *spatially correlated* (chunk+neighbors, not random single-chunk) → single-chunk independence is a property the user doesn't use; regions recover c4d's measured +5–8% and enable seam/DC healing. |
| **Quality granularity** | per-chunk scalar q, **per-sub-block q field** | A 64³ chunk mixes air + thin ink; one scalar q can't serve both. Per-block q field is how HEVC/JPEG-XL actually do "variable quality." |
| **Rate control** | fixed-q, multi-q-probe Lagrangian, **analytical Laplacian R-D model**, single-pass feedback | "1.5-pass" is really ~2–3× encode (4–5× on entropy). Laplacian R-D model gets the same allocation near 1-pass. |
| **Distortion metric (for RC)** | Parseval-MSE estimate, true-decode-MSE, edge/perceptual-weighted | Parseval "exact" is NOT exact for integer-DCT (non-orthonormal) + dead-zone; worst at 5–20×. Perceptual metrics need decoded spatial block → kills the "free distortion" trick. Verify estimate vs truth. |
| **Boundary healing** | none, decode-side deblock, within-chunk lapped, halo-blend | Independent 64³ + 8³ DCT = two-tier blocking + per-chunk DC steps deblocking can't fix. Couples to chunk-model choice. |
| **Chunk size** | 32³, **64³**, **128³**, 256³ (compile-time const) | c4d's "64³ beats 128³" was ENTROPY-TABLE-driven (tables amortized better at 64³), NOT transform-driven. With table-free entropy OR region-shared tables, that penalty disappears → bigger chunks may win: better energy compaction, **fewer chunk faces = less DC-banding/seam**, more DC continuity. Cost is coarser granularity/rate-control — but ultra-fast decode + correlated access make that cheap. Sweep JOINTLY with entropy + chunk-model + quality-granularity (entangled). |
| **Scalability mode (quality/SNR LOD)** | non-scalable (single fixed quality), **bit-plane / SNR-scalable** (embedded, truncatable) | A *quality* LOD distinct from the resolution pyramid: encode once, truncate the bitstream to any quality at decode/fetch time (~+6 dB per coefficient bit-plane added). This is the CORRECT version of c3d's failed §T9 — c3d truncated at whole-subband granularity (= resolution drop, done wrong); real SNR scalability truncates at **bit-plane granularity within each subband** (JPEG2000 quality layers + PCRD-opt). MUST be in the *coefficient/transform domain* — splitting raw u8 into nibbles does NOT form a quality ladder (raw intensity bits aren't importance-ordered; transform coeffs are). **Tension:** wants a bitplane-oriented entropy coder (binary AC / EBCOT-lite / MQ) → slower & more complex than RLGR/rANS, and embedded streams cost a few % ratio vs fixed-quality. Win for streaming + variable-quality + the correlated access pattern (fetch base for a neighborhood, refine the chunk in focus); measure the speed/ratio cost. |

**The central incoherence to resolve empirically:** MSE-optimal allocation *provably starves
HF* (the blur is the optimizer working as designed); "protect HF" silently swaps the
objective. We will measure both objectives against **human-relevant + ML-relevant metrics**
and let the scroll data + downstream-task deltas adjudicate, rather than asserting one.

---

## 3. Architecture

```
volume-compressor/
  include/vc/vc.h            # public API (encode/decode volume↔archive, config struct)
  src/
    config.h                 # COMPILE-TIME config: #defines select one impl per block
    core/                    # dimension-agnostic plumbing (no block logic)
      chunk.c                # gather/scatter (chunk side = compile-time const), padding, DC
      archive.c              # single-file container: trailer→dir→positional index→xxhash
      bitio.c, xxhash.c      # LE bit/byte IO, integrity
    transform/               # one .c per impl; all expose vc_transform_fwd/inv(block)
      dct_int.c  dct_float.c  dwt_97.c  lift_zfp.c  lapped.c
    quant/        quant_uniform.c  quant_deadzone.c  quant_subband.c
    entropy/      rlgr.c  rice.c  rans_static.c  rans_interleaved.c  bypass.c
    chunkmodel/   independent.c  region.c  halo.c  predicted.c
    ratectrl/     fixed.c  lagrangian_probe.c  laplacian_model.c  feedback.c
    healing/      none.c  deblock.c  lapped_heal.c  halo_blend.c
    metrics/      psnr.c  ssim.c  gmsd.c  haarpsi.c  percentile.c   # all C23, cheap
  harness/                   # the bake-off (not just tests)
    bench.c                  # sweep configs, emit ratio/quality/speed table
    compare_c3d_c4d.*        # run c3d & c4d on identical inputs, same metrics
    fetch_corpus.py          # pull Vesuvius 64³/256³ sub-volumes from AWS Open Data
  configs/                   # named build configs (one per experiment cell)
  PLAN.md  README.md  CMakeLists.txt
```

**Block contract (the key to compile-time swap with zero hot-loop cost).** Every block is a
plain C function with a fixed signature; `config.h` `#define`s map the generic name to the
chosen impl, so the compiler sees one concrete monomorphic call it can fully inline +
vectorize. Example:

```c
// config.h (one experiment cell)
#define VC_TRANSFORM     vc_dct_int8        // transform/dct_int.c
#define VC_ENTROPY_ENC   vc_rlgr_encode     // entropy/rlgr.c
#define VC_ENTROPY_DEC   vc_rlgr_decode
#define VC_CHUNKMODEL    vc_cm_independent
#define VC_RATECTRL      vc_rc_laplacian
#define VC_HEALING       vc_heal_deblock
#define VC_CHUNK_SIDE    64
```

No function pointers, no registry, no `switch` in the inner loop. Dispatch is *resolved by
the preprocessor*; the emitted codec is as small and fast as if hand-written for that config.
Sweeping the matrix = compiling N configs (scripted) and racing the binaries.

**Archive format** (rethought from c4d, key removed): EOF trailer (magic, version, dir
offset/len, xxhash) → member directory → **positional fixed-width chunk index** (offset,
length, per-chunk q, flags incl. ABSENT) — chunk coordinate is *implicit by array position*,
**no explicit chunk key** (the 64-bit key was a c3d-vs-c4d graft error). The **active config
is recorded in the archive header** so decode reconstructs the exact pipeline. **8 LODs** =
members `0/`..`7/`, each downsampled **from the original volume** (never from our own lossy
output — no error accumulation), independently encoded. No shards. No in-bitstream
progressive LODs (c3d's failed §T9).

---

## 4. Metrics & validation (first-class, from day one)

In-library (cheap, pure C23): **PSNR, MAE, L∞, p95/p99/p99.9, MS-SSIM (per-axis-slice + 3D),
GMSD, HaarPSI**. GMSD + HaarPSI are the cheap edge-sensitive perceptual metrics that catch
the blocking/seam artifacts PSNR hides — the shared currency of human readability *and*
ink/fiber ML.

Offline (Python harness): the real ML-readiness number — **downstream-task delta**
(segmentation Dice/IoU drop, ink-detection AUC drop) on decompressed vs original; optional
LPIPS/DISTS. Literature finding: task score is far more robust than pixel metrics
(segmentation Dice barely moves to ~20–50× even as SSIM falls) — **task delta is the metric
that matters; pixel metrics are screening proxies.**

Every harness run emits a table: per-config × per-ratio (5/10/20/50/100×) × all metrics +
encode MB/s + decode MB/s, **side-by-side with c3d and c4d on the identical inputs.**

---

## 5. Build sequence

**Phase 0 — skeleton + harness + one path (the day-one deliverable).**
Core plumbing (chunk gather/scatter, archive, bitio, xxhash), the metric bundle, the Vesuvius
loader, and **one simplest end-to-end pipeline** (leading candidate: integer-DCT 8³ + Rice +
independent chunks + fixed-q). Wire `compare_c3d_c4d` so the comparison table works
immediately. Goal: a real number on real data on day one.

**Phase 0.5 — reference reproduction (validate the harness).**
Confirm the harness reproduces c4d's published numbers (and c3d's) on the same volumes —
proves metrics/IO are correct before we trust any new-block comparison.

**Phase 1 — populate the levers.** Add alternative blocks one axis at a time, each measured
as it lands, in red-team-priority order (the choices most likely to be *wrong*):
1. **Transform bake-off** — int-DCT vs 9/7-wavelet vs ZFP-lifting at *scroll ratios*. (Settles the operating-point question first; it's upstream of everything.)
2. **Entropy bake-off** — RLGR vs rANS(static/interleaved) vs Rice on the winning transform's coefficients (ratio AND speed).
3. **Chunk-model + chunk-size bake-off (jointly)** — independent vs region vs halo vs predicted, crossed with 32³/64³/128³/256³ (ratio/quality/speed/seam-count vs the real correlated access pattern; size and model are entangled via the entropy choice).
4. **Quality-granularity** — per-chunk vs per-block q (within-chunk air+ink variance).
5. **Rate control** — Lagrangian-probe vs analytical-Laplacian (allocation quality vs encode cost).
6. **Healing** — deblock vs lapped vs halo-blend on the winning chunk model.

**Phase 2 — narrow in.** From the matrix, pick the winning configuration(s) per goal
(max-ratio / max-speed / balanced). Freeze the chosen config → that build *is* the shippable
custom codec. Document the frozen byte layout from the code.

**Phase 3 — polish.** LOD pyramid generation, variable-per-chunk-quality on the winning RC,
edge cases, fuzzing, the drop-in single-config build.

---

## 6. Open questions the data must answer (not us, now)

- At real scroll ratios (50–200×), does wavelet actually beat DCT enough to justify its worse
  SIMD profile? Or does a healed DCT win on the speed/quality balance?
- Does table-free entropy (RLGR/Rice) bleed the 20–40% the lit warns of vs rANS *on this
  data*? If so, is a per-chunk static rANS table (small, still independent) the better point?
- How big is the real independent-chunk penalty (ratio + DC banding) vs c4d regions, given
  the *correlated* access pattern makes region/halo decode cheap and natural?
- Does a **larger chunk (128³/256³)** win more ratio + quality (fewer seams, better
  compaction) once the per-chunk-table penalty is removed by table-free/region entropy — and
  can ultra-fast de/compression keep it as cheap to touch as 64³ was? (c4d's 64³-optimal
  result was table-driven and may not transfer.)
- Is per-chunk q good enough, or is per-block q field needed to deliver "variable quality"
  on mixed air+ink chunks?
- How wrong is the cheap Parseval distortion estimate vs true decode-MSE at 5–20×? Enough to
  mislead rate allocation?
- MSE-optimized vs edge/perceptual-optimized: which actually preserves ink/fiber for *both*
  human reading and downstream ML?
- Is **SNR/quality scalability** (encode-once, truncate-to-any-quality via coefficient
  bit-planes) worth its cost here? It needs a bitplane entropy coder (slower than RLGR/rANS)
  and gives up a few % ratio vs fixed-quality — but enables streaming + bitstream-truncation
  rate control (JPEG2000 PCRD-opt) that fits the correlated access pattern. Win for which use
  cases, lose for which? (Note: this is the *correct* form of c3d's failed §T9 — bit-plane,
  not whole-subband, truncation.)

---

## 7. Coding mandate for implementation (hammer this into every contributor/agent)

**All hot-path code MUST be written to be compiler-autovectorized — no hand intrinsics.**
- Straight-line loops over contiguous, aligned arrays; trip counts known/constant where
  possible (chunk dims are compile-time constants).
- No data-dependent branches in inner loops (use branchless select/masks); no function-pointer
  or virtual dispatch inside loops (compile-time block selection guarantees this).
- Prefer `restrict` pointers, `static const` tables (precompute what c4d did with `consteval`),
  structure-of-arrays layouts, integer math sized so the compiler picks wide lanes (e.g. 16-bit
  intermediates → 2× lane density).
- Verify vectorization actually happens: check `-fopt-info-vec` / `-Rpass=loop-vectorize`,
  inspect asm for the hot kernels, and benchmark — don't assume.
- Heap/arena for chunk-sized scratch (never tile-sized stack buffers — they explode in 3D).
- Build flags: `-O3 -ffp-contract=fast` (and `-ffast-math` acceptable; cross-platform
  byte-determinism is a non-goal). Target portable across x86-64/ARM via the *compiler*, not
  per-ISA code.
```
