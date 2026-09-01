# volcomp v1.0 — unified implementation brief

Sources unified here:
- `REVIEW_NOTES_A.md` (this team: 8 reviewer reports + synthesis + 30 owner decisions)
- `~/c5d/docs/REWRITE_CODE_REVIEW.md` (Codex antagonistic review, same commit `c649fcf`)

Both remain the record; this document is the decision-complete brief.
Everything below is settled unless marked **[design-once]** (an engineering
choice to make once, validated by a test gate — not an experiment campaign).

---

## 1. Scope in one paragraph

`volcomp` is a lossy compressor for **u8 volumetric CT data** organised as
**blocks** (16³, the transform unit), **chunks** (128³, the independently
decodable unit and the zarr chunk), and **shards** (1024³, one zarr v3
`sharding_indexed` file). It ships as a **stateless, single-threaded, Linux +
clang + x86-64 C23 library** with two caller inputs — `q` (quantizer step) and
`tau` (hard max absolute error) — plus a minimal CLI, a Python ctypes binding
and a zarr v3 codec plugin. Callers parallelise across chunks/shards
themselves. Quality is judged by mean / P90 / P95 / P99 / max abs error,
PSNR and SSIM; speed by 1-thread encode/decode MB/s and per-chunk latency.
No compatibility with c5d or anything else is required.

## 2. Owner decisions (normative for the port)

| # | decision |
|---|---|
| 1 | Terms: block 16³, chunk 128³, shard 1024³. `brick` retired everywhere. |
| 2 | `level` = OME-Zarr multiscale LOD only (level *n* = 2ⁿ× isotropic downscale, level 0 native). Quantized coefficients are `qcoef`, never "levels". |
| 3 | Only that downscaling scheme will ever exist. |
| 4 | Chunk-boundary seams must go; chunks stay decodable with zero neighbours. |
| 5 | Goals: encode speed, decode speed, latency, ratio, error percentiles/PSNR/SSIM. |
| 6 | u8 input only. No u16, no other dtypes. |
| 7 | No lossless mode, explicit or implicit. |
| 8 | Float math pipeline; no bit-exact determinism; agreement within a stated margin. |
| 9 | `q ≥ 1`, `tau ≥ 1` (never 0). |
| 10 | Label codec out of scope. |
| 11 | Block-local deblocking (a format change) is the seam fix; any format change is allowed now. |
| 12 | CPU-only v1; GPU deferred to v2; format stays GPU-friendly; SIMD-fast CPU. |
| 13 | No threading inside the library, implicit or explicit. |
| 14 | API thread-safe and re-entrant (no shared mutable state). |
| 15 | Shard = zarr v3 `sharding_indexed` layout; volcomp is the inner-chunk codec. |
| 16 | One baked-in configuration; no experiments; no tunables beyond `q`, `tau`. |
| 17 | `tau` mandatory; default `tau = 2·q`. |
| 18 | The codec never downsamples; LOD levels are supplied. |
| 19 | Entropy tables always transmitted, compactly coded; no embedded priors, no trainer. |
| 20 | Chunk dim fixed at 128 at compile time; no `dim` parameters. |
| 21 | 16³ block extraction API kept (non-zarr clients pull single blocks). |
| 22 | Namespace `volcomp_` / `VOLCOMP_`; lib `libvolcomp`; CLI `volcomp`; zarr codec id `"volcomp"`. |
| 23 | Deliverables: C library, CLI, Python ctypes binding, zarr v3 codec plugin. |
| 24 | Linux + clang + x86-64 (AVX2/AVX-512 + portable fallback). macOS/Windows/GCC/aarch64 → v2. |
| 25 | Real data: masked PHercParis4 zarr (`…/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/`), pinned + content-hashed, with a held-out region. |
| 26 | tau contract: exact against the encoder's own reconstruction; `tau + 1` LSB across differently built decoders. |
| 27 | `q`, `tau` stored as Q8.8 in `[1, 255]` (typical q 2–32, tau 4–64). |
| 28 | Edge-chunk padding is always 0. |
| 29 | No arenas, no context object; plain `malloc`/`free` per call. |
| 30 | What ships is v1.0; `version` byte starts at 1. |

## 3. The algorithm that is being kept (measured, do not re-litigate)

From c5d, carried as **kernels** (copied with rename, not redesigned):

- 16³ float DCT-II, separable, half-factored 8×8 even/odd banks; explicit
  AVX2/AVX-512 forward, autovectorised inverse with the `noinline` on
  `pass_axis_gs` and `C5D_DCT_CONTIGUOUS_X` guards **kept** (they prevented a
  measured −17 % LTO regression; keep the byte-identical cross-decode test
  that caught it). NEON kernels stay in-tree behind `#if`, unpromised.
- Dead-zone scalar quantizer: step `q·(1+r)^0.65` with `r = z+y+x`, DC step
  `q·0.125`, dead zone 0.2, reconstruction offset `(|l| + 0.26)·step`.
- Zigzag scan order `SCAN16`; run/level tokens with in-stream EOB; HybridUint
  (tokens 0–3 literal, else `2 + msb` + raw low bits in an LSB-first bypass
  stream); 32-token alphabet.
- 10 context models: runs/EOB by frequency band (3), levels by band × (run==0)
  (6), DC (1). DC delta-coded causally within a substream.
- rANS, 12-bit probabilities, `L = 2²³`, **2-way interleaved** (1-way −10 %
  decode, 4-way slower — both measured).
- `nsub = 32` substreams per chunk, each a contiguous run of 16 blocks in
  block-major (z, y, x) order, each separately flushed.
- Sparse-IDCT / flat-block fast paths (measured +17–23 % decode).
- Closed-loop encoder reconstruction (no self-decode) feeding a sparse
  correction layer that enforces `|err| ≤ tau` post-filter.
- `nz_extract` SIMD nonzero extraction, `gather/scatter_chunk` SIMD paths.
- Bakeoff metrics library (`metrics.c`): pooled MSE/PSNR, error histogram →
  MAE and nearest-rank P90/P95/P99/max, 3-D SSIM (11-tap Gaussian σ 1.5),
  blocking-amplification metric.

Measured-negative graveyard (from `docs/measured.md`; keep that file):
wavelet CDF 9/7, lapped analysis transforms, 8³ blocks, zero-tree / spatial
neighbour / prev-magnitude (ctx2) contexts, TCQ, per-block tables,
cross-chunk DC prediction, sign prediction, compressing the bypass stream,
Golomb-Rice, constant-rate RDO-lite, generic variance qmap, 4-way rANS,
substream sorting, one-sided outer-face deblock (1.7× worse), fp16 IDCT.

## 4. Stream format v1 (chunk) — `VOLC`

Byte-exact little-endian, written with explicit stores (never `memcpy` of
structs — Codex X7). One profile per version byte; no flag bits.

```
offset size field
0      4    magic        "VOLC"
4      1    version      1
5      1    reserved     0
6      2    nsub         u16, 1..128 (encoder always writes 32)
8      2    q            u16 Q8.8, 1.0..255.0
10     2    tau          u16 Q8.8, 1.0..255.0
12     4    corr_n       u32 bytes of correction stream
16     var  tables       10 models × 32 freqs, compact coding [design-once];
                         each model's freqs must sum to exactly 4096 → decoder
                         rejects otherwise (Codex X4, table half)
       12*nsub directory {u32 rans_n, u32 bypass_n, u32 ntok} per substream
       var  payload      per substream: rans bytes then bypass bytes
       corr_n corrections LEB128 gap (from previous corrected voxel, chunk
                         voxel index order z,y,x) + LEB128 zigzag(delta),
                         canonical LEB128, strictly increasing positions,
                         |delta| ≤ 255 (Codex X3)
       4    crc32c       over all preceding bytes (per-chunk integrity lives
                         here, not in the shard index)
```

Chunk geometry is implicit: 128³ voxels, 512 blocks, `blocks_per_sub =
512/nsub` (nsub must divide 512 or the last substream is short; encoder
writes 32). Decoder requirements: exact consumption of `rans_n`/`bypass_n`
per substream, `ntok` matches, all token/HybridUint classes within the
range implied by `q ≥ 1` (levels ≤ 2¹³ ⇒ HybridUint `k ≤ 13`; reject larger),
DC accumulator range-checked in 64-bit (C2), `nsub·blocks` bounds checked.

Reconstruction (normative, float, tolerance ±1 LSB across builds):
dequantize → sparse/flat-aware inverse DCT → **block-local deblock
[design-once, §6]** → clamp/round to u8 → apply corrections. Corrections are
therefore exact on the encoder's own build (decision 26).

## 5. Shard format = zarr v3 `sharding_indexed`

- Array `zarr.json`: `dtype uint8`, `fill_value 0`, `chunk_grid` regular
  `[1024,1024,1024]`, `codecs: [ {name:"sharding_indexed", configuration:
  {chunk_shape:[128,128,128], codecs:[{name:"volcomp", configuration:{q, tau}}],
  index_codecs:[{name:"bytes"},{name:"crc32c"}], index_location:"end"}} ]`,
  dimension order z, y, x.
- Shard file = inner chunk streams back-to-back, then the index: 512 ×
  `(offset u64, nbytes u64)` LE, then CRC32C of the index. Missing chunk =
  both fields `0xFFFF…`. **All-zero chunk = missing** (fill 0) — the writer
  detects all-zero source and writes nothing (replaces c5d's KNOWN_ZERO
  sentinel and skips encode/decode entirely, as before).
- Edge chunks are padded with 0 to 128³ before encoding (decision 28).
- Group-level OME-Zarr `multiscales` metadata with `coordinateTransformations`
  scale 2ⁿ per level; one `q`/`tau` per level (the caller supplies
  downsampled data and chooses per-level q; no bisection, no pyramid builder).
- No footer, no snapshots, no recovery scan, no per-entry CRC (all c5d shard
  findings C7–C9 vanish). Integrity: index CRC (zarr) + per-chunk CRC (§4).
- S3 use: fetch the index tail, then ranged-GET chunk `i` at
  `(offset, nbytes)`; the library exposes an index parser so nobody hand-rolls
  it (c5d's `remote.c` did).

## 6. Block-local deblocking [design-once]

Why: the c5d face filter skips chunk faces 0/128 (measured seam 3.08× at q8)
and over-flattens interior 16-planes 2× below ground truth; any cross-chunk
fix either costs 65–864 % bytes at q ≥ 4 or needs neighbours. The owner chose
the structural fix: a normative post-filter that is a function of the block's
**own** decoded data only, so every 16-plane and every 128-plane receives
identical treatment by construction, `deblock_pair` disappears, and a block
decoded alone equals the same block of a full decode.

Constraints from the measurements (report 08 in `REVIEW_NOTES_A.md`):
- "Apply today's gate + 3/8 delta one-sided from inside the block" is
  **exactly the prototype that measured 1.7× worse** — do not build that.
- Do not aim for maximal smoothing: interior planes today sit at 0.50 vs a
  true step of 1.10 at q8. Aim for uniform, truth-preserving treatment.
- Candidates: (i) symmetric in-block shell treatment applied to *both* sides
  of every internal 16-plane from within each block; (ii) coefficient-domain
  edge smoothing (attenuate the basis functions that create block-edge
  discontinuities before the IDCT, gated by the block's own q); (iii) a
  decode-side windowed synthesis (lapped *analysis* was negative; decode-only
  windowing was never tested).
- Applied before corrections so tau still holds exactly.

Gate (a test, run once on the held-out region, then frozen): 128-plane seam
metric equals the 16-plane metric within noise; blocking amplification
≤ 1.34× (q2) / 1.44× (q8); PSNR, SSIM, P99, max no worse than the c5d face
filter at matched bytes. Add the seam metric (boundary-vs-interior gradient
at 128-planes, same shape as the existing block metric) to `metrics.c`.

## 7. Public API — `include/volcomp/volcomp.h` (10 functions)

```c
#define VOLCOMP_ABI_VERSION     1u
#define VOLCOMP_FORMAT_VERSION  1u
#define VOLCOMP_BLOCK_DIM       16u
#define VOLCOMP_CHUNK_DIM       128u
#define VOLCOMP_CHUNK_VOXELS    (128u*128u*128u)
#define VOLCOMP_BLOCK_VOXELS    (16u*16u*16u)

typedef enum volcomp_status {
  VOLCOMP_OK = 0,
  VOLCOMP_ERR_ARG,        /* q/tau out of [1,255], null pointer, bad coordinate */
  VOLCOMP_ERR_CORRUPT,    /* malformed, non-canonical, CRC mismatch, truncated  */
  VOLCOMP_ERR_VERSION,    /* well-formed but unsupported version byte           */
  VOLCOMP_ERR_NOMEM,
  VOLCOMP_ERR_SHORT_BUF,  /* dst_cap too small; *needed says how much           */
} volcomp_status;

typedef struct volcomp_params { size_t struct_size; float q; float tau; } volcomp_params;

uint32_t        volcomp_abi_version(void);
const char     *volcomp_version_string(void);
const char     *volcomp_status_string(volcomp_status);
volcomp_params  volcomp_params_default(float q);          /* {sizeof, q, 2*q} */

size_t          volcomp_encode_bound(void);               /* worst case for one chunk */
volcomp_status  volcomp_encode(const volcomp_params *, const uint8_t *src_zyx /*128^3*/,
                               void *dst, size_t dst_cap, size_t *out_n);

typedef struct volcomp_info { size_t struct_size; float q; float tau; uint32_t nsub; } volcomp_info;
volcomp_status  volcomp_info(const void *enc, size_t enc_n, volcomp_info *out);

volcomp_status  volcomp_decode(const void *enc, size_t enc_n,
                               uint8_t *dst_zyx, size_t dst_cap /* >= 128^3 */);

/* 16^3 random access. Parses header + tables once; each block decode
 * entropy-decodes only the owning substream (<= 16 blocks of work) and
 * returns exactly the same bytes a full decode would place there. */
typedef struct volcomp_reader volcomp_reader;
volcomp_status  volcomp_open(const void *enc, size_t enc_n, volcomp_reader **out); /* enc must outlive */
void            volcomp_close(volcomp_reader *);
volcomp_status  volcomp_decode_block(volcomp_reader *, uint32_t bz, uint32_t by, uint32_t bx,
                                     uint8_t *dst_block /* 16^3 */, size_t dst_cap);
```

Conventions: every function returns `volcomp_status` decided at the point of
failure; caller owns all buffers; the library never returns memory the caller
must free except the opaque `volcomp_reader`; no globals, no env vars, no
static mutable state; `volcomp_reader` is not internally synchronised (one
per thread, or external locking); no `threads` anywhere; hidden visibility +
version script; `struct_size` on params/info for extensibility.

Optional second header `volcomp_shard.h` (separate target): zarr v3 index
parse from a buffer, shard writer (chunk streams + index + `zarr.json`),
thin mmap reader for local files.

## 8. Repository layout and build

```
volume-compressor/
  include/volcomp/volcomp.h        volcomp_shard.h
  src/volcomp.c                    api: validation + dispatch
  src/format.h                     internal normative constants (not installed)
  src/bitstream.h                  LSB-first bypass writer/reader (64-bit acc), HybridUint, zigzag, LEB128
  src/rans.c/.h                    model build/read (rejects non-normalised), 2-way encode/decode
  src/dct16.c/.h dct_tables.h      fwd/inv, SIMD + portable, SCAN16, EV/OD only
  src/encode.c  src/decode.c       chunk codec, corrections, block-local deblock
  src/shard/                       zarr v3 sharding writer/index/reader, crc32c (SSE4.2 + table)
  tools/volcomp.c                  CLI: encode decode info verify | shard-pack shard-unpack shard-stat
  tools/bench/                     bakeoff runner + metrics.c (+ seam metric) + ledger, codecs: volcomp, passthrough, zstd(opt)
  tools/fetch_corpus.py            pinned chunk list from the PHercParis4 zarr, sha256 recorded
  python/volcomp/                  ctypes binding + zarr v3 codec plugin ("volcomp")
  tests/                           quick, golden (regenerated), hostile, xdec (spec re-impl), seam, block-vs-full
  fuzz/                            d_chunk (hostile decode), rt_chunk (encode→decode asserting the tau bound)
  spec/format.md                   rewritten: §4–6 of this brief, normative
  docs/measured.md                 carried verbatim, appended to
```

CMake: `volcomp` (shared + static), `volcomp_shard`, `volcomp_cli`, tests,
`VOLCOMP_BENCH` (off), `VOLCOMP_FUZZ`, `VOLCOMP_SANITIZE` ("" | asan-ubsan |
tsan), `VOLCOMP_NATIVE` (one ISA switch), `VOLCOMP_WERROR` (on),
`VOLCOMP_HARDENING` (on in Release). Keep c5d's warning set
(`-Wall -Wextra -Wconversion -Wshadow -Wimplicit-fallthrough -Wformat=2 -Wvla
-Wdouble-promotion -Werror`), OpenSSF hardening block, hidden visibility,
`install(EXPORT)` + `.pc`. `-ffast-math` **only** on `dct16.c` and the
quant/dequant kernels, never on anything that parses bytes (Codex). No
`Threads`, no Vulkan, no ccache hard-requirement. CI legs: asan-ubsan
(Debug), Release, Release with the portable fallback forced
(`-DVOLCOMP_FORCE_PORTABLE=ON`), fuzz smoke (60 s each), clang-tidy.

## 9. Deleted from c5d (do not port)

`codec_v0.c`, `codec_wav.c`, `label.c/h`, `tifxyz.c/h`, `cache.c/h`,
`pool.c/h`, `stable.c/h`, `shard.c` footer/snapshot/recovery logic,
`priors.c` + `priors_v*.bin`, `train_priors.c`, `label_bench.c`,
`c5dc remote*/pack/slice/diff/tifxyz-*/label-*`, all of `src/gpu/` (→ v2
brief = report 02), `host_entropy.c`, `gputest.c`, `test_stable.c`,
`test_label.c`, `test_tifxyz.c`, `fuzz/d_label.c`, `ink_metric.py`,
`bdrate.py` (rewrite later with provenance), `PLAN.md`, `BACKLOG.md`
(archive), `.bak` manifests, `research/`.
Format features: flags 1/2/4/8/16/32/64/128/256 (all), u16 container, lossless
mode + `predict3d`, tau percentile, RDOQ, qmap, aniso, ctx2, rans_nway,
eprior, `encode_target`, `token_counts`, `encode_levels`, `decode` vs
`decode_par`, `deblock_pair`, `dim` parameters, `c5d_status`/`c5d_stable_status`
dual enums, custom allocators, cancel/progress callbacks, `_v1` suffixes.
Dead tables/functions: `DCT_M`, `EVT`, `ODT`, `SCAN8_TAB`, `M8`,
`c5d_dct_*_bs`, `rans_encode/decode/dec_*/encode_multi`, `c5d_mul_size`,
`c5d_add_size`, `c5d_alloc_aligned`, `enc_ctx.nsub`, `C5DF_RES_CTX*`.

## 10. Defects to turn into regression tests before comparing old vs new

| id | defect (c5d location) | test |
|---|---|---|
| C1 | portable `nz_extract` missing `}` (brick.c:280) | CI leg compiles the portable path |
| C2 / Codex 3 | DC accumulator signed overflow on hostile input (brick.c:1646) | hostile seed with maximal DC deltas; 64-bit accumulate + range check |
| X1 / Codex 1 | 32-bit bypass accumulator drops bits (brick.c:88-104) | 64-bit accumulator; property test over all token classes × pending-bit alignments |
| X2 / Codex 2 | qmap effective step → float→int32 UB | moot (qmap gone); keep a q=1 max-level assert |
| X3 / Codex 4 | correction gap wraps `size_t`; 64→32 truncation | bound check before add; canonical LEB128; monotonic positions |
| X4 / Codex 6 | non-normalised tables accepted; no exact consumption check | tables must sum to 4096; `rans_n`/`bypass_n`/`ntok` exactly consumed; CRC trailer |
| X6 / Codex | block decode ignored corrections → tau violated | block-vs-full bit-identity test (now includes deblock + corrections) |
| X7 / Codex | native-struct serialisation | explicit LE stores; static layout tests; golden vectors |
| X8 / Codex | unchecked `bpa³` | moot (dim fixed) |
| X9 / Codex | `rt_brick` asserts no invariant | `rt_chunk` asserts `max|err| ≤ tau` and P99 stats |
| C5 / Codex 7 | tests inside `assert()` vanish under NDEBUG | always-active `CHECK` macro; CI builds every preset from empty |
| C13 | empty substreams when nsub ∤ nblock | moot (32 ∣ 512) |
| C14 | unchecked token writes rely on worst-case sizing | checked emit with right-sized buffers |
| C16 | `dz_dq` code 0.26 vs doc 0.20 | baked 0.26; spec states it |
| seam | chunk faces unfiltered; per-chunk q | seam metric test; one q per level in the CLI |

## 11. Performance work that stays (all single-thread)

Baseline to beat (c5d, real 512-chunk corpus, 1 thread): **402 MB/s encode,
553 MB/s decode**, ≈ 3.8 ms per 2 MiB chunk decode. Profile: tokenize +
entropy ≈ 51 %, DCT ≈ 38 %, deblock ≈ 5 %.

1. Memoise the step table: 46 distinct radii, not 4096 `powf` per call (≈ 13 %
   of single-chunk decode latency; > 90 % of a block decode). With one baked
   `hf_exp`, the 46 powers are a `static const` table — zero runtime cost.
2. Right-size encode scratch: today ~21 MiB / ~100 mallocs per 2 MiB chunk
   (bypass sized at 43 bits/coeff). With `q ≥ 1` the bound is small; target
   one or a few `malloc`s per call.
3. Pack tokens + contexts into one `u16` stream; one rANS encoder.
4. Fold `RANS_MAX_SYMS` to 32 (freq/cum in 4 cache lines). `slot2sym` stays
   12-bit (40 KB for 10 models) — the L1 concern was ctx2's 78 KB.
5. Prefix substream offsets once (drop the O(nsub²) recompute).
6. `volcomp_reader`: parse + tables once; block decode cost = 1 substream.
7. Division-free rANS encode (reciprocals per symbol) — cheap, encoder-only.
8. SSE4.2 CRC32C in the shard module (c5d only had ARM).
9. Bench: pre-allocate SSIM buffers; report per-chunk latency P50/P90/P99
   and MB/s at 1 thread; a separate table for N caller threads.

Deliberately **not** done in v1 (measured or decided): x-face deblock SIMD
(≈1.3 % headroom, and the filter changes anyway), factored 16-point DCT
(Codex experiment), FSE/tANS (Codex experiment), any RD tuning.

## 12. Benchmark and acceptance

- Corpus: pinned 128³ chunk list from the decision-25 zarr, sha256 per
  chunk, split into *tune* (used for the one deblock design pass and golden
  vectors) and *held-out* (used for the acceptance numbers). Include unmasked
  interior chunks if that volume's mask leaves any; otherwise add a second
  region.
- Ledger rows carry: run id, full commit + dirty flag, compiler + flags,
  binary hash, CPU model, corpus hash, `q`, `tau`, reps, raw per-chunk sizes
  and timings (Codex methodology items adopted; no `--set=` remains to
  record).
- Acceptance vs c5d at matched `q` on held-out data: total bytes (incl.
  headers/tables/CRC) no worse than +1 %; PSNR ≥ −0.05 dB; SSIM ≥ −0.0005;
  MAE/P99 no worse; max ≤ tau exactly; 128-plane seam metric = 16-plane
  metric; 1-thread decode MB/s ≥ c5d's, encode ≥ c5d's; no sanitizer/fuzz/
  hostile failures; portable build passes goldens within ±1 LSB.

## 13. Order of work

1. Skeleton: header (§7), CMake (§8), CI legs, `bitstream.h` with property
   tests, `rans.c` with the normalised-table reader.
2. Port kernels (DCT, quant, gather/scatter, `nz_extract` fixed, tokenizer,
   2-way rANS) into `encode.c`/`decode.c` with the §4 stream layout, fixed
   constants, right-sized allocation, memoised steps, CRC trailer.
3. Corrections layer (mandatory tau, canonical LEB128, block-order indexing);
   `volcomp_reader` + `decode_block` exact.
4. Block-local deblock: implement candidates (i)/(ii), run the §6 gate once
   on the tune split, freeze, write `spec/format.md`.
5. Regression suite (§10), goldens regenerated, fuzzers, hostile suite.
6. Shard module (zarr v3 writer/index/reader), CLI, Python binding, zarr
   codec plugin; fetch script + hashed corpus.
7. Bench on held-out; publish v1.0 numbers (1-thread headline + caller-thread
   scaling table). Tag v1.0.
8. v2 backlog: GPU (report 02), aarch64/macOS/Windows/GCC, job-array API.

---

# As built (2026-09-01) — deviations from this brief

Decided with the owner during implementation; `volcomp.h` and
`spec/format.md` are authoritative.

- **Single header, all `static`**: `volcomp.h`; no library, no
  implementation TU. Consumers include it directly. Explicit AVX2+FMA
  kernels (transform, gather/scatter, quantise, post-filter) are required —
  the header `#error`s without `__AVX2__`/`__FMA__`; the entropy loops stay
  scalar (dependency-chain-bound; see `docs/measured.md`, "AVX2 pass").
- **API**: 4 codec functions — `volcomp_status_string`, `volcomp_encode`,
  `volcomp_decode`, `volcomp_decode_block` — plus `VOLCOMP_VERSION_STRING`
  and `VOLCOMP_ENCODE_BOUND`, and one optional post-process
  `volcomp_deblock(vol, nz, ny, nx, q)` for callers who assemble decoded
  regions (not part of the format). No reader object, no `info`, no params
  struct: the only input is `q`. Buffers passed to one call must not overlap.
- **Format**: 8-byte header (`VOLC`, version, reserved, q Q8.8); `nsub` is a
  constant 32; 8-byte directory entries; no CRC trailer; **no tau /
  correction layer**; **no in-format deblock** (block-local candidates
  measured worse than nothing — `docs/measured.md`); split reconstruction
  offsets 0.15 / 0.30; DC rounded to nearest; **entropy coder is 10-bit
  tANS (FSE)**, two lanes, one forward bit stream per substream (+1–21 %
  decode over rANS, 0.1–1.3 % smaller); DCT orthonormal scale folded into the
  quantiser tables.
- **Shard**: zarr v3 `sharding_indexed` written by the CLI only
  (`tools/cli/shard_pack.h`); no shard library, no zarr.json writers.
- **Tooling**: CLI `encode/decode/verify/shard-pack`; `volcomp-bench`
  prints a table (no ledger); corpus fetch via curl (dataset is zarr v2,
  raw chunks); no Python, no CI yaml.
- **Perf** (1T, `-march=native`, tune set, idle machine): encode ≈ 420 /
  650 / 780 MB/s, decode ≈ 590 / 1150 / 1650 MB/s at q2 / q8 / q32 — above
  c5d's filter-off numbers (296–499 / 377–760) without intrinsics.
- **Dropped from the brief**: GPU (entirely), NEON, label codec, tifxyz,
  u16, lossless, tau, threading, ctx, arenas/context object, install/pkg
  config, tsan, GitHub Actions, Python binding and zarr plugin (no Python
  installs allowed in this session).
