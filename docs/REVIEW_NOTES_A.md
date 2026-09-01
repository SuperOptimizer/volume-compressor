# volume-compressor — review notes, team A

Source reviewed: `~/c5d` (commit `c649fcf`, clean tree), read-only.
Method: one orchestrator, eight reviewers (3 Opus, 4 Sonnet, 1 Haiku), each
reading whole files and writing per-area reports; this document is the
synthesis, and the unedited per-area reports are appended in Part II so
nothing is lost when this is merged with the other team's notes.

No code in `~/c5d` or this repo was changed. One out-of-tree build was made in
the session scratchpad to collect baseline numbers (Part II, report 06).

---

# Part I — Synthesis

## 0. Decisions already taken by the owner (constraints for the port)

These came in during the review and override anything in the appendices that
says otherwise.

1. **Terminology.** `block` = 16³ (transform/coding unit), `chunk` = 128³
   (independently decodable unit), `shard` = 1024³ (file/object).
   The word **`brick` is retired everywhere** — symbols, files, format magic,
   spec, CLI, docs.
2. **`level` means OME-Zarr multiscale LOD level only.** Level 0 is native
   resolution, level *n* is 2ⁿ× isotropic downscale per axis. **Only that
   downscaling scheme will ever be used** (no anisotropic, no non-power-of-two
   pyramids). Quantized DCT coefficients must stop being called "levels".
3. **Chunk-boundary seams must be fixed** (visible today at 128³ boundaries),
   preferably encode-side only, at worst encode+decode — but a chunk must stay
   decodable with **zero neighbour chunks**. See §6.
4. No backwards compatibility with c5d streams is required.
5. Goals, in the owner's words: best encode speed, decode speed,
   bandwidth/latency, compression ratio, and quality measured as mean / P90 /
   P95 / P99 / max abs error, PSNR, SSIM.

6. **u8 only.** The library compresses and decompresses u8 volumes, full stop.
   No u16, no other input dtypes, ever. → drop the two-plane u16 wrapper,
   `deblock_u16`, the `VCU1` container, `dtype` from every struct, and the
   "native 16-bit" design question.
7. **No lossless mode, explicit or implicit.** → drop format flag 2, the 3-D
   MED predictor (`predict3d`), the residual token model/context slot, the
   lossless golden vector and fuzz variant, and the GPU "lossless unsupported"
   path. This also removes the only per-voxel serial (causal) kernel from the
   codec.
8. **Float math pipeline, no bit-exact determinism required.** Keep the f32
   DCT/quant/dequant/deblock; CPU and GPU only need to agree within a stated
   error margin (the current spec's "≤1 LSB cross-path, rare gate flips" is
   acceptable). → the fixed-point/deterministic-reconstruction experiment from
   the Codex review is dropped; "tau + tolerance" on GPU is an acceptable
   contract as long as the tolerance is published and tested.
9. **`q ≥ 1` and `tau ≥ 1` (never 0).** Parameter validation clamps to those
   floors. Consequences: quantized magnitudes are bounded by 8192/1 = 2¹³
   (2¹⁹ with a 1/64 qmap byte), so HybridUint classes stay ≤ 19 and the Codex
   bypass-writer / effective-step defects (X1, X2) become unreachable in valid
   streams — still fix the writer (64-bit accumulator) and validate the
   effective step, but they stop being a design constraint. It also lets the
   header carry `q` and `tau` as small fixed-point integers instead of floats.

10. **Label codec is out of scope — confirmed by the owner.** `C5L1`
    (`src/label.c`, 1503 lines), its CLI subcommands, `c5d-label-bench`,
    `test_label.c`, `fuzz/d_label.c`, the spec section and `LB_*` code all go.
    Report 04 in Part II remains the reference if it is ever wanted as a
    separate package.

11. **Block-local deblocking is the design (§6 option g2), and any format
    change is on the table right now.** The normative post-filter becomes a
    function of the block's own decoded data, never crossing a block face;
    16-planes and 128-planes are treated identically by construction, the
    seam disappears, `deblock_pair` ceases to exist, and single-block decode
    becomes exact. The 128³ seam metric (§6.3 item 4) is the acceptance gate.
    Level-uniform q in the LOD packer (g1) is still required — it fixes the
    other seam cause (different q per chunk) and is free.
12. **CPU-only for v1; all GPU work is deferred to v2.** No Vulkan sources,
    shaders, `host_entropy.c`, `gputest`, or GPU CMake in the v1 tree. The
    format must stay GPU-friendly (keep the `nsub` substream field and the
    block-independent structure; block-local deblock is *more* GPU-friendly
    than the face filter). Report 02 in Part II is the v2 starting brief.
    v1 must be SIMD-fast: keep the NEON / AVX2 / AVX-512 kernels, fix and
    CI-compile the portable fallback, dispatch at compile time (`VC_NATIVE`)
    or once per process via cpuid — never per call.
13. **No threading inside the library, implicit or explicit.** No thread
    pool, no `threads` parameter anywhere, no `vc_shutdown`, no `pthread`
    dependency. All work on one 128³ chunk is single-threaded. Callers
    parallelise across chunks and shards with their own threads.
14. **The API must be thread-safe and re-entrant** so callers *can* do that:
    no global or static mutable state (no lazily-built tables, no process
    caches, no env-var reads); per-call scratch is either stack/arena owned by
    a caller-created `vc_ctx` that is documented as "one per thread, not
    internally synchronised", or allocated per call. Every other function is
    pure with respect to its inputs.

Consequences of 11-14 folded into the sections below: `pool.c` is deleted
(not redesigned); `nsub` loses its CPU-parallelism rationale and its default
should be re-chosen for ratio (§5.1); the seam fix (a) via corrections is no
longer needed at any q; §7 becomes a v2 brief; the single-chunk latency
budget is the 1-thread number (≈3.8 ms per 2 MiB at the measured 553 MB/s
1T, 8.3 ms on the slower review host).

15. **Shard container = zarr v3 `sharding_indexed` layout.** One 1024³ shard
    file = 512 inner 128³ chunk streams back-to-back + trailing index of
    `(offset u64, nbytes u64)` LE per chunk + CRC32C of the index; missing
    chunk = both fields all-ones; known-zero chunk = missing + `fill_value 0`.
    Our stream is the inner-chunk codec. Consequences: `C5S1`/`VCS1`,
    the 40-byte footer, `OFFSET_ZERO` sentinel, per-entry CRC in the index,
    recovery snapshots and the backward magic scan all go (C7-C9 vanish).
    Per-chunk integrity moves to a 4-byte CRC32C trailer inside the chunk
    stream (cheap, and it covers the payload no matter how it was fetched).
    The packer also writes the `zarr.json` (array + OME-Zarr `multiscales`
    with scale 2ⁿ per level). `tools/c5dc/remote.c`'s hand-rolled index
    parsing is replaced by a `vc_shard_index_parse(buf, n)` helper for S3
    ranged-GET users; the mmap reader becomes a thin optional convenience.
16. **One baked-in configuration. No experiments, no tunables.** Everything
    below that says "sweep", "evaluate", "experiment", "profile byte", or
    "opt-in" is void for v1. Fixed by decision, using the values the headline
    numbers were actually measured with:
    - `hf_exp` 0.65, `dc_fine` 0.125, `dz_q` 0.2, `dz_dq` 0.26 (the value in
      the shipped code; the doc's 0.20 differed by +0.03 dB), 12-bit rANS
      probabilities, 2-way interleave, 10 context models, 32-token alphabet,
      `nsub` = 32 (the measured default; the header keeps the field so a v2
      GPU encoder may write 128 and the decoder accepts 1..128).
    - Deblock always on (no flag). Entropy tables always transmitted, compactly
      coded (item 19) — no priors blob, no flag.
    - **Dropped entirely**: `rdo`, `tau_pct` (tau is a hard max only),
      `block_qmap`/ROI (also removes X2 and X5), `eprior` tri-state, `ctx2`,
      aniso, `rans_nway`, `encode_target`, `token_counts` (the priors trainer
      becomes a one-off offline tool run once before freeze), `--set=` in
      the bench, every experiment listed in §5.5 and §5.4 items 2/4, and the
      pre-freeze experiment list in §12.
    - The only caller inputs are `q ≥ 1` and `tau ≥ 1` (§0 item 9). Stream
      header: `magic[4] "VCC1", u8 version, u8 log2(dim), u16 nsub, u16 q
      (Q8.8), u16 tau (Q8.8 or u8), u32 corr_n` = 16 bytes, followed by the
      compact 10×32 frequency tables (item 19), the `nsub` × 12-byte
      directory, payload, corrections, CRC32C trailer.
    - The block-local deblock filter (item 11) is *designed once*, validated
      once with the 128-plane seam metric as a **test**, and frozen. Choose
      the minimal departure from the measured face filter: the same
      gate/3-8 delta math applied to each block's own outer shell against its
      own inner neighbours is exactly the one-sided filter that was measured
      1.7× *worse* (§6 option e) — so it must instead be a symmetric
      in-block treatment (candidate ii, coefficient-domain edge smoothing, or
      candidate i applied to *both* faces of each internal 16-plane from
      within each block). Whoever implements it should read §6 in full first.

17. **`tau` is mandatory** (≥ 1, always a hard max-error bound; no "off").
    Normal use is `tau = 2·q`; `vc_params_default(q)` sets exactly that.
    The correction layer is therefore always present (possibly empty), and
    every decoded chunk carries a hard bound — the codec's stated contract.
18. **The codec does not downsample.** LOD levels are supplied already
    downsampled by the caller; the library/CLI/shard writer only encode what
    they are given, per level. No downsampling filter in the spec, no
    pyramid builder in the packer — only per-level shard writing and the
    OME-Zarr metadata (`multiscales` with scale 2ⁿ) describing what the
    caller produced. Level-uniform q (§6, g1) becomes simply "use one q per
    level", which is the natural CLI/API usage and needs no bisection.
19. **Entropy tables are always transmitted (option B, §12 item 8).** No
    embedded priors, no `priors_v*.bin`, no `c5d-train-priors`, no format
    flag 32, no `token_counts` hook. Each chunk stream carries its own 10
    model tables in a compact encoding (LEB128 delta-coded frequencies that
    sum to 4096; decoder rejects tables that don't — closes X4's table half).
    Header `version` byte therefore has no flag bits; the only format branch
    left is `nsub` in the header.

20. **Chunk dim is fixed at 128 at compile time.** No `dim` parameter on any
    call, no `log2dim` header byte (header is now 15 → pad to 16). Kernels
    may assume 512 blocks. Edge chunks are padded to 128³ by the shard writer.
21. **16³ block extraction stays in the API** (`open/close/decode_block`).
    Clarification of the current behaviour, since it was asked: the c5d
    format *does* support partial decode today, at **substream** granularity —
    `decode_chunk` parses the header, then entropy-decodes only the owning
    substream from its start (DC is delta-coded within a substream and rANS
    is serial), i.e. up to `512/nsub` = 16 blocks of work to deliver one, no
    full 128³ reconstruction. What it lacks today is deblocking (needs
    neighbours) and tau corrections — both fixed by decisions 11 and 17:
    with block-local deblock and corrections indexed per block, a block
    decoded alone is bit-identical to the same block of a full decode. So
    clients can pull a 16³ block without a 128³ round-trip through zarr.
    Cost knob is `nsub` (blocks decoded per fetch = 512/nsub): 32 → 16
    blocks, −0 % ratio (baseline); 64 → 8 blocks, ≈ −1 %; 128 → 4 blocks,
    −2.2 %. **Baked: `nsub` = 32** (single config, item 16); the header
    field remains so a future encoder could trade ratio for finer access.
    Also make the open handle cheap: parse + models once, step table
    memoised (§5.4 item 1) — today a block decode rebuilds 40 KB of tables
    and 4096 `powf`.
22. **Namespace is `volcomp`, not `vc`** (`vc` is Volume Cartographer).
    Everywhere this document says `vc_` / `VC_` / `vci_` / `vc` read
    `volcomp_` / `VOLCOMP_` / `volcomp_i_` (or file-static) / `volcomp`.
    Library `libvolcomp`, header `include/volcomp/volcomp.h`, CLI `volcomp`,
    CMake target/package `volcomp`, zarr v3 codec identifier `"volcomp"`
    with configuration `{"q": …, "tau": …}`. Magics stay `VCC1`-style ASCII
    (`"VOLC"` + version byte is an alternative; pick one, not a decision).
23. **Deliverables**: C library + minimal `volcomp` CLI (encode / decode /
    info / verify single chunks; shard pack / unpack / stat) + Python
    ctypes binding + a zarr v3 codec plugin so zarr-python / tensorstore
    read the shards directly. The plugin is what makes decision 15 useful.
24. **Platform: Linux + clang only for v1.** x86-64 with AVX2/AVX-512 paths
    plus the portable fallback compiled in CI. macOS, Windows/MSVC, GCC and
    aarch64/NEON are v2 (keep the NEON kernels in the tree, gated, but don't
    promise them).
25. **Real data**: fetch from
    `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/index.html#PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/`
    (a masked PHercParis4 volume, already zarr). Replace `fetch_corpus.py`'s
    manifest scheme with a fetcher that pulls a pinned set of 128³ chunks
    (plus one held-out region for the deblock/quality gate) and records
    content hashes. Note this volume is *masked* (air zeroed) — good for the
    known-zero path, but the gate corpus should include unmasked interior too
    if available.
26. **tau contract** (accepted): exact against the encoder's own float
    reconstruction; `tau + 1` LSB across differently-built decoders. Stated in
    the spec, not engineered around.
27. **`q`, `tau` in the header as Q8.8 fixed point**, range `[1, 255]`
    (typical use q 2..32, tau 4..64). Validation rejects anything else.

28. **Edge-chunk padding is always 0** (confirmed).
29. **No arenas, no context object. Plain `malloc`/`free` per call**; the
    user is expected to bring a good allocator. This deletes `volcomp_ctx`
    entirely — the API is fully stateless and re-entrant by construction
    (decision 14 satisfied for free). Independent of this, the port must
    still *right-size* the per-call allocations: today an encode mallocs
    ~21 MiB / ~100 blocks for a 2 MiB chunk because buffers are sized to a
    43-bits-per-coefficient worst case (§5.4 item 3); with q ≥ 1 the true
    bound is far smaller, and one allocation per call (or a handful) is the
    target. `volcomp_encode_bound()` stays so the *output* buffer is
    caller-owned.

30. **What ships is v1.0.** No backward or forward compatibility with c5d
    streams or any other codec is required; the `version` byte starts at 1
    and only matters for volcomp's own future.

Decided by default (say so if wrong): CI = GitHub Actions with asan/ubsan,
release, and portable-fallback legs.

### 0.1 Rename map (old → new)

| c5d term / symbol | new | notes |
|---|---|---|
| chunk (16³), `blk`, `BLKV`, `nchunk`, `chunks_per_sub`, `chunk_qmap` | **block**, `BLOCK_VOXELS`, `nblock`, `blocks_per_sub`, `block_qmap` | `blk`/`BLKV` in brick.c already meant this |
| brick (128³), `c5d_brick_*`, `C5D_BRICK_DIM`, `brick_idx`, "bricks per shard" | **chunk**, `vc_encode`/`vc_decode` (chunk is the default unit, no noun needed), `VC_CHUNK_DIM_DEFAULT 128`, `chunk_idx`, "chunks per shard" | |
| shard (1024³) | shard | unchanged; see §9.1 for zarr v3 alignment |
| `c5d_brick_decode_chunk` | `vc_decode_block` | 16³ random access |
| `c5d_brick_deblock_pair` | `vc_chunk_seam_filter` (only if a decode-side pair filter survives §6) | "deblock" now literally means the in-chunk block-seam filter |
| "levels" (quantized coefficients), `c5d_brick_encode_levels`, `levels_natural` | `qcoef` / "quantized coefficients", `vc_encode_qcoefs` (GPU-only internal) | frees `level` for LOD |
| `lod_level` (shard footer), pack.c "level" | `level` | OME-Zarr multiscale index |
| substream | substream | keep; "a contiguous group of blocks inside a chunk with its own rANS flush" |
| magics `C5B3` / `C5U1` / `C5L1` / `C5S1` / `C5SR` | `VCC1` (chunk) / `VCS1` (shard) / `VCSR` (shard snapshot); `C5U1` and `C5L1` have no successor (decisions 6-7) | write as 4-byte arrays, not `u32` literals — three of today's five serialise byte-reversed vs their comments (shard.h:11, shard.c:56) |
| `c5d_`/`C5D_`/`C5DF_` prefixes | `volcomp_`/`VOLCOMP_` public (owner decision 22; `vc` is Volume Cartographer) | the shorter `vc_` used in the API sketches below is to be read as `volcomp_` |
| library / header / CLI / zarr codec id | `libvolcomp`, `include/volcomp/volcomp.h`, `volcomp`, `"volcomp"` | |

The zarr v3 vocabulary is exactly *shard* (file) ⊃ *chunk* (unit of I/O and
decode) — the owner's chosen terms line up with it, which matters for §9.1.

---

## 1. Executive summary — the twenty things that matter

Correctness (fix during port, they exist today):

1. **Build-breaking**: `src/brick.c:280-286` portable `nz_extract` fallback is
   missing its closing `}` — any non-x86/non-aarch64 target fails to compile.
2. **Decoder UB on hostile input**: `brick.c:1646` `prev_dc + unzigzag(u)`
   accumulates unbounded across blocks in a substream → signed overflow. Encoder
   has `LEVEL_LIMIT`; decoder never re-imposes it. Needs a range check + fuzz seed.
3. **GPU encode race**: `codec.c:364` `token_stride`/`rans_stride` are ≡2 mod 4
   for odd `blocks_per_sub` (reachable at nsub=110); non-atomic word RMWs in
   `encode_tokenize.comp:42`/`encode_rans.comp:22` then collide across substreams.
4. **GPU decode writes caller buffer before checking shader status**
   (`codec.c:312-317`) — corrupt input yields garbage *and* an error.
5. **`tests/test_stable.c` does not build in Release** (`-Werror` unused
   variables: all control flow is inside `assert()`, stripped by `NDEBUG`).
   Reproduces on the repo's own `release`/`bench` presets.
6. `c5d_cache_destroy` frees pinned entries (use-after-free hazard) — moot,
   cache is dropped.
7. Shard crash-recovery backward magic scan (`shard.c:181-215`) is an unbounded
   cost path on hostile files; normal-path footer+index has **no CRC** while
   the snapshot path does (asymmetric trust).
7b. From the Codex review, verified here (§3 X1-X4): the 32-bit bypass bit
   writer can silently drop bits for valid extreme-q inputs; a qmap byte of 1
   can push the quantiser's float→`int32` cast out of range (UB); tau
   correction gaps can wrap the voxel index; non-normalised model tables and
   unconsumed payload bytes are accepted. All four become regression tests.

Scope cuts (everything below has zero production callers or measured-negative value):

8. Drop `codec_v0.c`, `codec_wav.c` (~840 lines; bakeoff-only baselines, wavelet
   measured −105..−190% vs DCT) and the rANS/DCT entry points only they use.
9. Drop `cache.c` (251 lines, only caller is its own smoke test).
10. Drop `tifxyz.c` (2-D float surfaces + a hand-rolled TIFF/LZW reader; not a
    volume codec; not in the spec despite README; no fuzz target).
11. Drop the `stable.h` façade *as a layer* (619 lines, called only by its own
    test): fold its conventions (status enum, validation, atomic publish) into
    the one public API. Note: commit `c649fcf` says the façade was built for a
    downstream consumer ("TSM") — the single new API must remain
    ctypes/dlopen-friendly, which is a constraint on the design, not a reason to
    keep two layers.
12. Drop format flag 64 / ctx2 (19-model context set): measured +0.4..1.2% ratio
    for −5..7% decode, default-off, halves the model tables (78→40 KB), removes a
    branch from the decode inner loop, deletes `priors_v2.bin`, a golden vector
    and a GLSL mirror.
13. Drop flag 128 / anisotropic axis weights (no measured RD result anywhere;
    sole reason the header has an overloaded `_pad`). Drop `rans_nway` 1 and 4
    (2 measured best; 4 measured *worse*); hardcode 2-way.
14. Drop from the public params: `hf_exp`, `dc_fine`, `dz_q`, `dz_dq`,
    `eprior` tri-state, `ctx2`, `rans_nway`, `axis_weight_*`. Freeze them.
15. Drop `c5dc remote*` (forks the `curl` binary), `tifxyz-*`, `slice`, `diff`.

Design (see §4, §9):

16. **One public header, ~17 entry points** (vs ~50 today), one status enum
    (three coexist now, plus raw `-1` at 60 sites), one ownership model
    (caller-owned buffers + `*_bound()`; deletes the façade's full-volume
    memcpy on every call), and **no threading at all** (owner decision 13;
    today five inconsistent `threads` conventions coexist).
17. **Thread pool lifecycle**: today `pool.c:95-103` spawns `ncpu−1` detached,
    never-exiting threads on first use regardless of requested width, with no
    shutdown and no `pthread_atfork`. **Resolved by deleting the pool**
    (decision 13); the port must also audit for any other static mutable
    state so the library is re-entrant (decision 14).
18. **Header freeze needs three experiments first** (all format-visible):
    `RANS_PROB_BITS` 12→11 (halves `slot2sym` to fit L1), `dz_dq` 0.20 (docs)
    vs 0.26 (code — unresolved drift), and DC round-to-nearest / magnitude-
    dependent reconstruction offset (cheap, untested, directly targets MAE/P99/
    blocking). Then shrink the 40 B header to ~16-20 B (it is 16% of a 249 B
    16³ LOD chunk).
19. **GPU**: the shipping `codec.c` is the *slow* path; every measured win
    (record-once, fences, HOST_CACHED readback, sliced dispatch) lives only in
    `gputest.c`. `host_entropy.c` (GPU-only) is compiled into the *core* lib.
    ~20 constants are hand-mirrored into GLSL with no CI gate (gputest has no
    `add_test`). Recommendation: port the GPU **after** the CPU format freezes,
    once, with generated `format.glsl`/`dct_tables.glsl` and embedded SPIR-V.
20. **Shard container should be the zarr v3 sharding layout** (or provably
    convertible to it) so OME-Zarr readers with a codec plugin can read our
    files directly; expose parse-index-from-buffer so S3 users do their own
    ranged GETs (`tools/c5dc/remote.c` hand-rolls exactly that today).

---

## 2. What the codec is (for the merge reader)

128³ chunk of u8 voxels → 512 blocks of 16³ → float DCT-16 (separable, NEON /
AVX2 / AVX-512 paths) → dead-zone quantization with radial power-law step
`q·(1+r)^0.65`, DC step `q·0.125`, DC delta-coded causally within a substream →
zigzag/HybridUint tokens (32-symbol alphabet, 10 context models by frequency
band × run state) → 2-way interleaved rANS, 12-bit probabilities, per-chunk
transmitted tables *or* embedded trained priors (exact-cost choice) → `nsub`
independent substreams (default 32, max 128) each separately flushed →
40-byte header + 12-byte directory per substream. Decode-side normative
quant-gated 4-tap deblock across internal block faces. Optional sparse
correction layer bounds max (or a percentile of) abs error to `tau`. Optional
lossless mode (3-D MED predictor + residual tokens). u16 handled by a two-plane
(coarse/residual) wrapper. Label volumes (`C5L1`) are a separate lossless
multi-channel palette + 405-context codec sharing the rANS backend. Shards
(`C5S1`) hold 512 chunk payloads + footer index with per-chunk CRC32C and
crash-recovery snapshots.

Measured headline (real 512-chunk PHercParis4 corpus, docs/BENCHMARKS.md):
24.16× @ 42.91 dB, P99 err 5, max 18, encode 2.48 GB/s, decode 3.02 GB/s
(Ryzen 7940HS auto-threads); 1T 402/553 MB/s. GPU (RTX 4060) decode 16.6 GB/s
compute / 4.05 GB/s streaming, encode ~1 GB/s.

The algorithm is in good shape and almost every knob has a number behind it
(docs/measured.md). The cruft is in default-off knobs, duplicated layers,
dead experiments, and lifecycle/ownership design.

---

## 3. Correctness findings (full list)

| # | Where | What | Severity |
|---|---|---|---|
| C1 | brick.c:280-286 | portable `nz_extract` missing `}` — never compiled | build break on non-x86/arm64 |
| C2 | brick.c:1646 | DC accumulator unbounded on hostile input → signed overflow | UB in decoder |
| C3 | gpu/codec.c:364 + encode_tokenize.comp:42, encode_rans.comp:22 | strides ≡2 mod 4 for odd blocks_per_sub → cross-substream word RMW race | GPU encode corruption |
| C4 | gpu/codec.c:312-317 | decode memcpys to `dst` before reading shader status | wrong-output-on-error |
| C5 | tests/test_stable.c | logic inside `assert()`; fails `-Werror` under NDEBUG | Release build break |
| C6 | cache.c:237-251 | destroy ignores `pins` | UAF (module dropped) |
| C7 | shard.c:181-215 | O(n·k) backward magic scan with CRC per false hit | hostile-input slow-open |
| C8 | shard.c:176-179 vs 207-211 | normal footer+index un-CRC'd; snapshot path CRC'd | integrity gap |
| C9 | shard.h:11, shard.c:56 | `C5S1`/`C5SR` magics serialise as `5CS1`/`5CSR`; scanner hardcodes reversed bytes | format wart |
| C10 | shard.c:220 vs 233/241 | `c5d_shard_brick` doesn't NULL-check reader; siblings do | API inconsistency |
| C11 | stable.c:502-506 | `finish` returns E_IO *after* successful publish if temp unlink fails | false failure |
| C12 | label.c:1318 | `npal`-sized alloc bounded only by header self-consistency (up to ~2 GB from a tiny header) | alloc-DoS on untrusted input |
| C13 | brick.c:873-874 | empty substreams emitted when nsub ∤ nblock (12 B dir + flush each) | waste |
| C14 | brick.c:632-636, 770-803 | unchecked token writes rely on worst-case sizing | latent overflow |
| C15 | brick.c:1249 | `malloc(sizeof *o * sizeof h)` works by accident | cosmetic |
| C16 | brick.h:34 vs brick.c:152; brick.c:142 vs measured.md | ctx2 "default on" comment vs off; `dz_dq` 0.26 vs documented 0.20 | doc/code drift, format-visible |

Additional defects reported by the Codex review (`~/c5d/docs/REWRITE_CODE_REVIEW.md`) and **independently verified against the source during this review** (see §13 for the full cross-reference):

| # | Where | What | Severity |
|---|---|---|---|
| X1 | brick.c:88-104, 124 | `bitw.acc` is 32-bit; `bw_put` does `acc \|= v << nbits` *before* draining. HybridUint can append k ≤ 28 low bits with up to 7 pending → 35 logical bits, top bits silently lost. Reachable with **valid** inputs: levels ≥ 2²⁶ need step ≤ 2⁻¹³, i.e. q ≲ 1e-4, or q ≲ 0.008 with a qmap byte of 1. Not reachable at production q, but it is a silent wrong-decode, and the same writer is copied in v0/wav. Fix: 64-bit accumulator + property test over every token class × pending-bit alignment. | valid-input miscompression |
| X2 | brick.c:187-194 vs 614, 657-660 | Validation guarantees `q > 2⁻¹⁴` so levels stay < 2²⁷, but a qmap byte of 1 divides the step by 64 → level up to 2³³ → `(int32_t)a` on an out-of-range float is UB, *before* `hyb_emit`'s `k ≥ 29` reject can fire. Fix: validate the minimum *effective* step after qmap scaling, or convert in a checked wider domain. | UB with valid inputs |
| X3 | brick.c:1559-1563 | `apply_corrections`: `vi += gap` with a 64-bit `gap` can wrap `size_t` back to a valid index; `(uint32_t)dz` truncates before validation → non-monotonic / non-canonical correction streams are accepted. Fix: `gap <= nvox-1-vi` before the add; require strictly increasing positions and canonical LEB128. | hostile-input canonicality |
| X4 | brick.c:1503-1508, rans.c:7-41 | Transmitted frequency tables that don't sum to 4096 are *renormalised* and accepted; no exact rANS/bypass consumption or final-state check. The label codec re-verifies (`label.c:1187-1189`); brick doesn't. Substituted/appended payload bytes can decode "successfully". Fix: reject non-normalised tables (shared `rans_model_read`), verify exact consumption, per-unit checksum. | canonicality / integrity |
| X5 | brick.c:1595, 1734 | qmap scales reconstruction per block, but deblock strength uses only the chunk-global header `q` (GPU same). Fine-ROI blocks are over-filtered, coarse ones under-filtered; the qmap cross-decode test disables deblock, so there is no gate. | quality defect in an opt-in feature |
| X6 | brick.c:1588-1595, brick.h:71 | `decode_chunk` (block decode) omits tau corrections → a stream advertised with a hard tau bound violates it through this API. | contract violation |
| X7 | brick.c:1010, 1465; shard.c:73 | All headers/tables/indexes are serialised by native-struct `memcpy`/`fwrite` with no endian conversion or static layout assert, despite the spec saying little-endian. | portability |
| X8 | brick.c:1483 vs 856 | decoder computes `bpa*bpa*bpa` in `uint32_t` unchecked (encoder checks in a wider type); `shard.c:87` unchecked `bpa³`. | overflow hygiene |
| X9 | fuzz/rt_brick.c | round-trip fuzzer claims a max-error invariant but doesn't check one; varies only q/deblock at dim 32. | test gap |
| X10 | gpu/host_entropy.c:299 (reported by Codex; consistent with report 02, not re-verified here) | GPU stream parse doesn't validate header floats `q/hf_exp/dc_fine/dz_dq` that the CPU parser rejects; they reach shader `pow` and float→int. | CPU/GPU acceptance mismatch |

---

## 4. Proposed public API (single header)

Merged from reports 01 and 03, with the owner's terminology applied. Rules:
every function returns `vc_status`; the library never allocates anything the
caller must free; no `threads` parameters exist; params
structs start with `struct_size`; no threading of any kind; no `_v1` suffixes (ABI carried by
`vc_abi_version()` + `struct_size` + hidden visibility + version script; the
real compatibility guarantee is the **format** version in the stream header).

```c
/* include/volcomp/volcomp.h */
#define VC_ABI_VERSION     1u
#define VC_FORMAT_VERSION  1u
#define VC_BLOCK_DIM       16u    /* transform + random-access unit         */
#define VC_CHUNK_DIM       128u   /* fixed at compile time (owner decision 20) — no dim params below; the `dim` arguments still shown are to be deleted */
#define VC_SHARD_DIM       1024u  /* only meaningful to volcomp_shard.h      */

typedef enum vc_status { VC_OK, VC_ERR_ARG, VC_ERR_CORRUPT, VC_ERR_NOMEM,
                         VC_ERR_FORMAT, VC_ERR_SHORT_BUF, VC_ERR_IO } vc_status;
/* u8 only (owner decision 6); no dtype anywhere. */

typedef struct vc_params {
  size_t   struct_size;
  float    q;           /* quantizer step in u8 units; >= 1               */
  float    tau;         /* hard max |err| bound in u8 units; >= 1; mandatory (default 2*q) */
} vc_params;            /* that is the whole configuration (§0 items 16-17) */
vc_params vc_params_default(float q);   /* {q, 2*q} */

/* No context object (owner decision 29): every call is stateless, uses
 * malloc/free for its own scratch, and is re-entrant. The `vc_ctx *` first
 * arguments still shown below are to be deleted. */
vc_params vc_params_default(void);

uint32_t    vc_abi_version(void);
const char *vc_version_string(void);
const char *vc_status_string(vc_status);

/* grayscale chunk codec */
size_t    vc_encode_bound(const vc_params *, uint32_t dim);
vc_status vc_encode(vc_ctx *, const vc_params *, const uint8_t *src_zyx, uint32_t dim,
                    void *dst, size_t dst_cap, size_t *out_n);
typedef struct vc_info { size_t struct_size; uint32_t dim; float q; float tau; } vc_info;
vc_status vc_info_read(const void *enc, size_t n, vc_info *out);
vc_status vc_decode(vc_ctx *, const void *enc, size_t n, uint32_t dim,
                    uint8_t *dst_zyx, size_t dst_cap);

/* 16^3 random access: open once (parses header, builds models + step table),
 * then decode blocks; touches only the owning substream. With block-local
 * deblock (§0 item 11) the result is bit-identical to the same block of a
 * full decode. */
typedef struct vc_reader vc_reader;
vc_status vc_open(const void *enc, size_t n, uint32_t dim, vc_reader **out);
void      vc_close(vc_reader *);
vc_status vc_decode_block(vc_reader *, uint32_t bx, uint32_t by, uint32_t bz,
                          void *dst_block, size_t dst_cap);

/* label codec — REMOVED (owner decision 10). No vc_dtype, no vc_channel,
 * no vc_label_* entry points. The public surface is the grayscale u8 chunk
 * codec above plus vc_shutdown; ~11 entry points total. */
/* No vc_shutdown, no threads anywhere: nothing global to release (§0 items 13-14). */
```

Optional headers, separate targets: `volcomp_shard.h` (§9.1) and
`volcomp_gpu.h` (§7). Deliberately absent: `hf_exp`, `dc_fine`, `dz_q`,
`dz_dq`, `eprior`, `ctx2`, `rans_nway`, `axis_weight_*`, `token_counts`
(internal header for the priors trainer), `encode_target` (moved to the CLI /
LOD packer; if kept, bisect on a tokenize-only cost estimate instead of 7 full
encodes), `encode_levels` (GPU-internal), `decode` vs `decode_par` (merged),
custom allocators, cancel/progress callbacks (they only fire *between* calls
today and report 0/1→1/1; batch callers can just stop calling), all of
`cache.h`, `tifxyz.h`, `codec_v0.h`, `codec_wav.h`, and — per §0 item 16 —
`rdo`, `tau_pct`, `deblock`, `nsub`, ROI/qmap. Public surface: `vc_params_default`, `vc_encode_bound`, `vc_encode`, `vc_info_read`,
`vc_decode`, `vc_open/close/decode_block`, version/status strings — 10 calls
(no ctx, no dim, per decisions 20 and 29).

(`deblock_pair` is gone: block-local deblock, §0 item 11.)

---

## 5. Core codec — detailed keep/drop and the port work list

### 5.1 Params and flags

| item | verdict | evidence |
|---|---|---|
| `q` | keep | |
| `hf_exp` | freeze 0.65, drop from API and header | 0.45/0.65/0.85 sweep RD-neutral ±5% (measured.md M1) |
| `dc_fine` | freeze 0.125, drop from API and header | never swept, never varied |
| `dz_q` | freeze 0.2 (encoder-only) | measured "keep 0.2" |
| `dz_dq` | keep normative, drop from API; **resolve 0.20 vs 0.26 before freeze** | brick.c:142 vs measured.md |
| `deblock` (flag 1) | keep | 3.9% of decode, materially reduces blocking |
| `lossless` (flag 2) | **drop** (owner decision 7) — with it `predict3d`, the residual model slot, the lossless golden/fuzz variants | it was also 1.5-2.5× worse than zstd-19 on masks (measured.md 2026-08-17) |
| `tau`, `tau_percentile` (flag 4) | keep, merge into `{tau, tau_pct}` | the scientific-use differentiator; percentile variant is ~25 encoder-only lines |
| `rdo` | keep, default 0 | q1.83+rdo.15: 24.18×/43.00 dB/max 17 vs 24.16×/42.91/max 18; blocking amp 1.34→1.28; encode 2.4× slower. Only lever that improves *max* error at iso-rate |
| `rans_nway` (flags 8/16) | hardcode 2, delete both flags and `c5df_flags_nway` | 2-way +10% decode vs 1; 4-way slower (633 vs 792 MB/s at q8) |
| `eprior` (flag 32) | **drop entirely — always transmit** (owner decision, §0 item 19) | 128³-only chunks under zarr make the 16³ win moot |
| `ctx2` (flag 64) | **drop** | see §1 item 12 |
| aniso (flag 128) | **drop** | unmeasured; frees `_pad` |
| `nsub` | keep the field; **re-choose the default for ratio** | its CPU-parallelism rationale is gone (item 13). Measured: nsub=8 +0.6% vs 32, nsub=128 −2.2%; a 48³ chunk is 15% smaller single-stream. GPU (v2) wants 128 and `decode_block` wants many; both are per-stream choices, not defaults. Sweep 1/4/8/16/32 on `corpus/dev` before freezing the default. |
| `chunk_qmap` (flag 256) → `block_qmap` | keep as `vc_encode_roi` | normative, opt-in; generic variance heuristic measured −0.1 dB — never ship as default |
| `nthreads` | **delete** (item 13) | |
| dims other than 128 | **drop — dim fixed at 128** (owner decision 20) | |

### 5.2 Functions

Keep: encode, decode (single, threaded), decode_block (with an open handle to
amortise `b_parse`: today each block decode rebuilds 40 KB of alias tables and
a 4096-`powf` step table and entropy-decodes up to 16 blocks to deliver one),
defaults. Delete u16 encode/decode and `deblock_u16` outright (decision 6).
Move to tools/internal: `encode_target`, `token_counts`, `encode_levels`.
Merge: `decode`+`decode_par`. Delete: `rans_encode`, `rans_decode`,
`rans_dec_*`, `rans_encode_multi`, `c5d_dct_*_bs`, `pass_axis_bs`, tables
`DCT_M`, `EVT`, `ODT`, `SCAN8_TAB`, `M8` (~2.5 KB dead), `c5d_mul_size`,
`c5d_add_size`, `c5d_alloc_aligned`, `c5d_version_string` (rewire),
`enc_ctx.nsub` (write-only), `C5DF_RES_CTX*` aliases, `c5df_dequant_mag` (declared
normative, called by nobody — either use it at the 4 open-coded sites or delete).
Merge duplicates: `rans_encode_multin` / `multin16` (36 identical lines — keep
the u16-ctx one), three transcriptions of the deblock kernel
(`deblock_face_task`, `deblock_pair_row_task`, `deblock_u16`) with three
different clamps, header float validation in 4 places, "flat block =
DC·0.015625" in 3 places.

### 5.3 Format header freeze — superseded by the 16-byte layout in §0 item 16; analysis retained

Today 40 B: `magic, dim, flags, nsub, q, hf_exp, dc_fine, dz_dq, corr_n, _pad`.
`dim` is redundant (every entry point takes it), `hf_exp`/`dc_fine`/`dz_dq` are
constants, `_pad` is the aniso field, `corr_n` forces a header rewrite +
`realloc` after the payload. Target ~16-20 B: `magic[4], u16 format_version,
u16 flags, u8 log2dim, u8 profile, u16 nsub, f32 q, u32 corr_n` (or drop
`corr_n` by putting the correction layer behind the directory). On a 249 B 16³
LOD chunk this is ~10% of the file. Decide `RANS_PROB_BITS` and `dz_dq` first.

### 5.4 Performance work list (CPU)

Baselines to beat: 1T 402/553 MB/s, auto 2.48/3.02 GB/s on the real corpus;
profile (report 06, software sampling): tokenize+entropy ≈ 51%, DCT ≈ 38%,
deblock ≈ 5%. Single-chunk decode floor on the review host: 8.3 ms serial,
3.4 ms auto.

1. `step_table` (brick.c:201-210): 4096 `powf` per encode *and* per decode
   *and* per block decode, for 46 distinct radii. Memoise a 46-entry table.
   ~80-110 µs ≈ 13% of the measured 700 µs P50 decode latency, >90% of a
   block decode. Cheapest win in the file.
2. `slot2sym` L1 pressure: drop ctx2 (78→40 KB), evaluate `RANS_PROB_BITS`
   11 (→20 KB), `RANS_MAX_SYMS` 64→32 (freq/cum in 4 lines not 8). Do **not**
   retry clz multi-byte refill or `freq|cum<<16` packing (measured −4..−9%).
3. Encode allocates ~21 MB per 2 MB chunk (brick.c:599-603: 393 KB bypass
   buffer per substream at "43 bits/coeff" worst case × 32, 96 mallocs) plus
   8 callocs + tau `realloc`. Use a per-thread arena / realistic bound with
   checked growth; collapse 8 parallel arrays into one `sub_state[]`.
4. Pack `toks[]`/`ctxs[]` into one `u16` stream (also unifies the two rANS
   encoders and matches the label codec's 16-bit contexts).
5. Pool: **deleted** (item 13). The `threads==1` fast path in every kernel is
   the only path that survives; the per-axis deblock barrier disappears with
   the block-local filter (item 11).
6. `b_decode_sub` O(nsub²) offset recomputation → prefix offsets in `b_ctx`.
7. Keep the load-bearing anti-optimiser guards (`noinline` on `pass_axis_gs`,
   `C5D_DCT_CONTIGUOUS_X`) and the byte-identical cross-decode test that
   caught the −17% LTO regression.
8. SIMD gaps: `deblock_u16` (fully scalar, single-threaded, all 3 axes);
   x-face deblock scalar (measured ~1.3% headroom, deliberately deferred);
   lossless `predict3d` serial by construction.
9. x86 CRC32C hardware path in the shard module (only ARM64 is accelerated).

### 5.5 Ratio / quality work list — VOID for v1 (§0 item 16: no experiments); retained for v2

Open (not in the measured graveyard), ranked by value/cost:
1. **DC round-to-nearest** (DC currently gets the same 0.2 dead zone as AC;
   DC error is a whole-block shift and the direct driver of blocking amp).
   Hours of work; likely the best MAE/P99/blocking lever left.
2. **Magnitude-dependent `dz_dq`** (separate offset for |l|=1 vs >1; classic
   0.1-0.3 dB in image coders). One header byte or a frozen pair.
3. Header shrink (§5.3): ~10% on the coarsest LOD levels.
4. Clamp emitted `nsub` to non-empty substreams (~20 B/chunk at odd nsub).
5. Full-candidate RDOQ beyond ±1 (gpudct got −2.8% BD with full candidates;
   ours is adjacent-only at +0.8-1.0%). Expensive.
6. Table-free rANS decode — the stated precondition for ctx2 ever paying.

Measured-negative, do not re-litigate: CDF 9/7 wavelet, lapped/overlap
transforms (seam energy ×2.2, MSE +54% — relevant to §6), 8³ blocks, zero-tree
and spatial-neighbour contexts, TCQ, per-block tables, cross-chunk DC-plane
prediction (0.1-0.5%), sign prediction, zstd/context-coding the bypass stream
(zstd-19 *expands* it), Golomb-Rice backend, constant-rate RDO-lite, generic
log-variance qmap, ctx2, 4-way rANS, substream sorting.

### 5.6 u16 — RESOLVED: out of scope (owner decision 6). Historical analysis kept below.

`brick.c:1168-1370` splits each sample into coarse/residual planes and runs two
full C5B3 encodes; the residual plane is white noise to a 16³ DCT. 2× transform
and entropy work, two extra planes, serial min/max and recombine loops, scalar
`deblock_u16`, **tau rejected outright**, no threads on decode, and **no
measured RD result** (corpus is u8). Native micro-CT is 16-bit. Recommendation:
one quantisation in source sample units on the existing f32 DCT (HybridUint
already reaches 2²⁸; only the `u8−128` block gather hardcodes 8-bit), keep the
`VCU1` container shape. Owner decision needed: is 16-bit input a first-class
target of this repo?

---

## 6. Chunk-boundary (128³) seam deblocking

Dedicated study (report 08 in Part II; measured in a scratchpad copy on a
smooth synthetic pair of adjacent 128³ chunks built from the real 48³ golden
fixture). Metric: flat-conditioned mean |step| across a plane, i.e. the
banding the eye sees in smooth regions.

### 6.1 Why the seams are there — three causes, in order of visibility

**D2 (dominant visually): neighbouring chunks are encoded at different q.**
`tools/c5dc/pack.c:62-66` calls `c5d_brick_encode_target` *per chunk*, so each
chunk bisects its own q to hit a per-chunk byte target. Adjacent chunks then
decode with different noise amplitude and texture *and* a different deblock
gate (`c = clamp(0.8q+1, 1, 24)`, brick.c:472). Measured qA=2 / qB=8: 54.4 vs
49.0 dB either side of the plane. No filter can remove a texture change.

**D1 (mechanical): the chunk's outer faces are the only 16-aligned planes the
normative filter skips.** `deblock_par` runs `dim/16 − 1` face tasks with
`f = (fi+1)·16` (brick.c:479, :421) → faces 16…112 are filtered, f = 0 and
f = 128 never are. Measured, same q both sides:

| q | interior 16-face (filtered) | chunk 128-face (unfiltered) | source truth | after `deblock_pair` |
|---|---|---|---|---|
| 2 | 1.183 | 1.230 (1.04×) | 1.10 | 1.207 |
| 4 | 0.818 | 1.408 (1.72×) | 1.10 | ~0.82 |
| 8 | 0.501 | **1.541 (3.08×)** | 1.10 | 0.503 |

At q = 2 there is essentially no seam; at q ≥ 4 the chunk face is 1.7-3.1×
rougher than every other 16-plane — consistent with seams being visible in the
aggressive part of the LOD ladder.

**The constraint that shapes every fix**: the filtered interior planes at q = 8
sit at 0.50 while the *true* field's step is 1.10. The filter over-flattens
interior faces ~2× below ground truth; the chunk face at 1.54 is only 1.4×
*above* truth. The artifact is a periodic *contrast in smoothness*, not a jump.
Spending bits at the boundary can only converge to 1.10 (verified: near-lossless
face blocks reach 1.175 and stop). Parity with 0.50 is only reachable by
reproducing the filter across the face, i.e. by sharing information.

**D3 (minor)**: x = 0 / x = 127 layers have a different noise statistic because
they are never pulled. Second order.

**Not a cause**: DC. `dc_fine = 0.125` on an orthonormal 16³ DCT gives a DC step
of 0.016 LSB at q = 8; DC delta-coding resets per substream and is entropy
context only; cross-chunk DC prediction is already measured-negative.

### 6.2 Options, measured

| option | change scope | decode indep. | measured cost / effect | verdict |
|---|---|---|---|---|
| **g1: level-uniform q** — bisect q once per LOD level on ~32 sampled chunks, then encode every chunk of the level at that fixed q | encode-only, `pack` only | yes | removes D2 entirely; RD-correct (constant λ, not constant bytes per chunk); `pack` ~7× cheaper (1 encode per chunk instead of ≤7) | **do first, unconditionally** |
| **(a) seam compensation via the existing sparse-correction layer** — 2-pass encode: encode all → barrier → decode all → run the pair-filter math on both reconstructions → append `(index, ±delta)` pairs to *each chunk's own* correction block | encode-only; **no format or decoder change** (corrections are an arbitrary per-voxel delta list applied post-deblock, brick.c:1736) | yes, exactly | bit-exact `deblock_pair` parity. Cost = 2 B per moved face voxel: **q2 1.75 %/face, 10.5 % all six; q4 65 %/face; q8 144 %/face** | **use at q ≤ 2** (L0 near-lossless); useless where seams are visible |
| (c) finer q on face blocks via `block_qmap` | encode-only, existing flag | yes | q8, one face layer at 0.25×: +18.3 % bytes, 1.541→1.230; at 0.125×: +26.8 %, →1.175; whole shell at 0.25×: **+63 %**. Saturates at ground truth, never reaches parity | poor |
| (d) RDOQ with a seam penalty | encode-only, needs neighbour recon (2-pass) | yes | est. 1-3 % bytes for 1.1-1.2× step reduction; a block's 4095 other voxels fight a one-plane objective; RDOQ is already 2.4× encode | poor |
| (e) decode-side one-sided outer-face filter (pull boundary voxel toward its inner neighbour), optionally with a transmitted per-face target | decode change / small format addition | yes | **prototyped: q8 1.541 → 2.640, 1.7× WORSE**; both sides' biases add because the field has a real gradient. A transmitted common target buys only 1/(1−α); matching the interior's 4× needs near-lossless face planes (~5 KB/face vs a 10 KB chunk). The seam is high-frequency (16×16-patch mean step rms 0.42 at the face vs 0.83 interior), so a low-order per-face parameter buys nothing | **rejected** |
| (b) halo encoding | format | yes | 132³ = +9.6 % voxels and breaks `dim % 16`; 160³ = +95 %; lapped transforms measured seam energy ×2.2 / MSE +54 % | rejected |
| (f) DC pre-compensation | — | — | DC is already exact to 0.016 LSB | dead |
| g3: `deblock=false` at high q | per-chunk flag, exists | yes | exact interior/boundary parity (1.497 vs 1.413 rms) for −0.5 dB and block artefacts everywhere | emergency lever only |
| **g2: make deblock block-local** — redefine the normative post-filter as a function of the block's *own* decoded coefficients (e.g. per-block taper / coefficient-domain smoothing before scatter), never crossing a block face | **format revision + quality campaign** | yes — and single-*block* decode becomes exact | 16-planes and 128-planes treated identically **by construction**; `deblock_pair` disappears; the deblock gate-flip clause leaves the determinism section | **the correct long-term design**; fallback if seams must be gone at high q |
| soften the filter at high q (scale the 3/8 delta or cap `c` below 24 as q rises) | decode-side normative tweak | yes | reduces the *contrast* by not over-flattening interiors 2× below truth; must be validated against blocking-amplification numbers | cheap interim mitigation for q ≥ 4 |

### 6.3 Recommendation — **owner chose g2 (block-local deblock) + g1**

Resolved (§0 item 11): the normative filter becomes block-local, so options
(a)/(c)/(d)/(e) and the interim softening are moot. What remains to do:

- **Design the block-local filter.** Candidates: (i) a per-block spatial
  taper on the 1-2 voxel shell of each 16³ block, gated by the block's own
  `q·qscale` and its own local gradient; (ii) coefficient-domain smoothing
  (attenuate the highest-frequency basis functions that produce edge
  discontinuities, i.e. a de-ringing/de-blocking window on the reconstructed
  block before scatter); (iii) a windowed/overlapped *synthesis* only (decode
  side) — note lapped *analysis* transforms measured negative, but a
  decode-only window is a different experiment. Evaluate each against the
  measured interior-face over-flattening (0.50 vs truth 1.10 at q8): the goal
  is uniform treatment, not maximal smoothing. Gate: the 128-plane seam
  metric equals the 16-plane metric within noise, blocking amplification
  ≤ today's 1.34×/1.44× (q2/q8), PSNR/SSIM/P99 no worse than the current
  face filter at matched rate.
- **g1**: level-uniform q in the LOD packer, exactly as below (still
  required; different-q neighbours are the dominant visible seam cause).
- Delete `deblock_pair`; `decode_block` becomes exact; drop the gate-flip
  determinism clause from the spec.

Pre-decision recommendation retained for the record:

1. **g1 now** (encode-only, negative cost): level-uniform q in the LOD packer.
   Implementation: `pack_ctx` carries `float q` instead of `double ratio`;
   a sampling loop before the parallel encode drives the existing bisection
   from the summed size of ~32 sampled chunks; `pack_brick_task` calls the plain
   encoder. Volume-edge and KNOWN_ZERO chunks unaffected; still fully parallel.
2. **(a) at q ≤ 2**: add `vc_append_corrections(blob, n, {index, delta}[], k)`
   next to the tau writer (factor brick.c:1080-1131 into a shared LEB128
   emitter that merges with an existing correction block, indices ascending).
   Packer: encode-all → barrier → decode-all → per internal face run the pair
   gate/delta on the two reconstructions, record `(vi, +δ)` for the negative
   chunk's `p0` and `(vi, −δ)` for the positive chunk's `q0` → append per
   chunk. No ownership rule (both tasks see both reconstructions, read-only);
   faces with no neighbour or a KNOWN_ZERO neighbour emit nothing; block-level
   random access still won't see corrections (already documented).
3. **q ≥ 4**: do not try to buy parity with bits. Either g2 (format change,
   the right answer) or the interim "soften at high q" tweak, and note that a
   fixed-point/deterministic reconstruction (Codex experiment, §13) would make
   g2's filter exact across CPU/GPU.
4. Add the **chunk-seam metric** (boundary-vs-interior step / step-error rms
   at 128-planes, same shape as the existing block metric in `metrics.c`) to
   `vc-bench` as the acceptance gate, and report SSIM on assembled volumes
   *with* the production seam treatment — today's whole-volume SSIM crosses
   independently decoded chunk faces with no pair filter (Codex, confirmed).

Owner decision: whether g2 (format revision) is in scope for v1 of the new
repo. If yes, it should be designed together with the header freeze (§5.3)
since it changes the normative decode. If no, ship g1 + (a) and the interim
softening, and document that seams above q≈4 are a contrast artefact, not a
discontinuity.

---

## 7. GPU path (Vulkan) — DEFERRED TO v2 (owner decision 12)

Nothing in this section is v1 work. It is retained as the v2 brief; the only
v1 obligations are (a) keep the format GPU-friendly (`nsub` field,
block-independent reconstruction, block-local deblock) and (b) don't carry
any GPU source into the v1 tree.

Verdicts: keep as an **optional component** (`VC_GPU=ON`, separate target,
zero GPU sources in the core lib — today `CMakeLists.txt:94` compiles
`src/gpu/host_entropy.c` into `libc5d` so a test can link it). Port it
**after** the CPU format is frozen: dropping ctx2/aniso/nway and shrinking the
header changes ~20 hand-mirrored GLSL constants (report 02 §2.5 lists them
all, including the whole 40-byte header layout in `encode_assemble.comp:46-52`).

Must-do in the port:
- Fix C3/C4 (§3).
- Move the fast path from `gputest.c` into the library: descriptor slicing,
  record-once + resubmit, fence streaming (not `vkQueueWaitIdle`), one
  barrier per *stage* not per chunk (today ~4n full barriers), HOST_CACHED
  readback on integrated GPUs, device-local encode output on discrete GPUs
  (today `encode_assemble` does per-byte `atomicOr` across PCIe).
- Persistent buffer arena + descriptor sets (today 12-14 buffer
  create/map/free per call; encode scratch ≈ 50 MiB per chunk, mostly
  zero-filled; 8 MiB output capacity for a ~100 KiB stream).
- Embed SPIR-V (C23 `#embed` or generated header); generate `format.glsl`
  and `dct_tables.glsl` from the C headers at build time; register a ctest
  with a graceful skip when no device exists.
- Split `gputest.c` (1702 lines, a *third* pipeline implementation) into a
  ~250-line API test and a ~350-line bench; delete `he_decode` + the dense
  shader branch (unreachable from the library), `vk_fill_range` (0 callers),
  the token-sort measurement residue, 12 env vars (`C5D_VK_DEVICE`/`VALIDATE`
  become create-options).
- API: opaque `vc_gpu`, options struct, status enum with `first_bad` for
  per-chunk CPU fallback, one caller arena for encode output (not
  `uint8_t ***`), internal batch splitting behind a limits query,
  `vc_gpu_encode_supported()` predicate. Decode should be at parity minus
  nothing (lossless no longer exists); encode is legitimately a profile (no tau/rdo/
  qmap — source-domain searches) but `dim != 128` is an unimplemented
  generalisation, not a profile choice.
- Perf: the docs blame reverse rANS, but the stage table shows
  `encode_tokenize` at 1.58 GB/s is 5× worse — it scans all 4096 scan
  positions per block with scattered loads; emit a compact nonzero list from
  `forward_quant.comp` instead. `encode_model`/`encode_prefix` run
  `local_size_x=1`; `encode_assemble` dispatches only `n` workgroups; `pow()`
  per coefficient instead of a step table; no subgroup intrinsics anywhere;
  no `VkPipelineCache`.

---

## 8. Label codec; other codecs

> **Superseded by owner decision 10 (§0)**: the label codec is out of scope
> and is removed. The paragraph below is the pre-decision assessment, retained
> only in case it is wanted as a separate package.

- **Label (`C5L1` → `VCL1`): keep**, minimised. Real, spec'd, fuzzed,
  measured (0.02-0.17 bpv masks, 0.001-0.007 bpv ids; 17-180× vs zstd-19).
  The 405-context candidate model (`lb_cands`, label.c:270-352) is
  load-bearing per its own bake-off — do not simplify it. Simplify the
  plumbing: 4th copy of HybridUint/bit-writer (label.c:392-424, 486-516),
  3rd zigzag, ~10 parallel encode buffers → arena, per-voxel type switch in
  the decode scatter (label.c:1283-1298) → hoisted dispatch, O(nsub²)
  `lb_sub_locate`. Drop `rans_nway` from its params (nway=2 measured +3.4%
  bytes for nothing). Add an absolute cap for C12.
- **tifxyz: drop from this repo** (contrib/separate tool if the segment
  pipeline still needs it; give it a fuzz target and a spec section there).
- **codec_v0 / codec_wav: delete**, plus their bakeoff glue
  (`tools/bakeoff/codecs.c:18-34,126-135,276`).
- Shared primitives to create once: `common/bitstream.h` (LSB-first bypass
  writer/reader, HybridUint emit/read, zigzag 32/64), `rans_model_read()` that
  bakes in the "reject non-pre-normalised tables" check that label and tifxyz
  hand-roll and v0/wav forgot, LEB128 helpers.

---

## 9. Infrastructure

### 9.1 Shard container — RESOLVED: zarr v3 sharding (§0 item 15); rationale retained

Keep the container, as a **separate optional target** (`volcomp_shard`), not
inside the codec library (it drags `mmap`, `fsync`, crash recovery and CRC
into a bytes→voxels library). Since the owner is implementing OME-Zarr:

- zarr v3 `sharding_indexed` is: a file holding N inner chunks back-to-back
  plus an index of `(offset u64, nbytes u64)` little-endian per chunk, with a
  CRC32C over the index, index at start or end, missing chunk = both fields
  all-ones. Our `C5S1` is the same idea with a 40-byte footer, per-chunk
  `{offset u64, nbytes u32, crc32c u32}`, sentinels for MISSING and
  KNOWN_ZERO, and recovery snapshots.
- **Recommendation**: make the shard *be* a zarr v3 shard (index layout,
  index checksum, index-at-end) with our stream as the inner-chunk codec,
  so `zarr-python`/`tensorstore` with a codec plugin read our data directly
  and OME-Zarr metadata (`multiscales`, `coordinateTransformations` with
  scale 2ⁿ) describes the pyramid. Things ours has that zarr's doesn't:
  per-chunk CRC (move it into the chunk stream trailer — also fixes C8 for
  the index), KNOWN_ZERO (zarr expresses this as a missing chunk +
  `fill_value = 0`, which is exactly the semantics `c5d_shard_put_zero`
  provides; drop the second sentinel), snapshots (keep as an *extension*
  placed before the final index, or drop — only `c5dc pack` and `stable.c`
  use the writer, and both are being reshaped). This is a **design decision
  for the owner**; the alternative is keeping `VCS1` bespoke and writing a
  converter.
- Regardless: expose `vc_shard_index_parse(buf, n, ...)` /
  `vc_shard_index_entry(...)` so S3 consumers do their own ranged GETs
  (`remote.c` hand-rolls this against raw footer structs today), make the
  reader opaque (today `c5d_shard_reader` is a public mmap struct embedded
  by value in the "stable" wrapper), add the x86 CRC32C path, bound the
  recovery scan (C7), CRC the index on the normal path (C8), make the
  fsync-every-64 MB cadence a parameter.
- LOD: with the owner's rule (level n = 2ⁿ isotropic), the pyramid is a fixed
  structure — 128³ chunks / 1024³ shards at every level, coarse levels using
  `dim` 64/32/16 chunks only at the tail. `pack.c`'s per-level "inverted
  quality ladder" and `encode_target` bisection (7 full encodes) should be
  re-evaluated against that fixed structure; the downsampling filter (2×2×2
  mean vs. something else) must be specified in the spec since level-n data
  is what gets compressed and viewed.

### 9.2 Cache — drop

251 lines, zero production callers, generic u64→bytes S3-FIFO, UAF on
destroy-while-pinned, O(1024) ghost scan per put. Every intended consumer
(VC3D, zarr, a renderer) already has its own cache.

### 9.3 Thread pool — DELETED (owner decision 13)

`pool.c`/`pool.h` and the `Threads::Threads` dependency go. No `threads`
parameters, no shutdown, no fork handlers needed. Thread safety is by
construction: no static mutable state anywhere in the library (audit the
port for `static` non-const data — today the codec core has none, `pool.c`
and `platform.h`'s cached CPU count are the only offenders, and the dropped
v0/wav codecs had mutable global scan state). Callers own parallelism across
chunks/shards. Original recommendation kept below for the record.

#### (superseded) keep the algorithm, fix the lifecycle

See §1 item 17 and §5.4 item 5. Concretely: size to the largest requested
width, joinable workers, `vc_shutdown()`, `pthread_atfork` (re-init in child),
atomic per-job ticket instead of a global mutex per item, `signal` not
`broadcast`, and document "callers parallelising across chunks should pass
`threads=1`". Keep the `threads==1 → no pool interaction` fast path (it is
consistent everywhere today). Default `threads` differs between
`c5d_brick_defaults` (1) and `c5d_label_defaults` (0) — pick one (moot once label is gone; recommend 0 = all).

### 9.4 `stable.c` — fold, don't keep

What it adds: argument validation (belongs in the core), status enum (becomes
*the* enum), atomic publish via temp file + link (the one real feature — moves
into the shard writer), allocator hooks (force a full extra memcpy of every
buffer on 100% of calls — drop), cancel/progress (fire only between calls —
drop). Its test's hostile/round-trip coverage carries over to the single API;
its allocator/cancel assertions go.

---

## 10. Build, tools, tests, docs

**CMake**: no `Threads`, no `VC_GPU` in v1. Keep the warning set (`-Wconversion`, `-Wdouble-promotion` are
load-bearing), the sanitizer enum, the OpenSSF hardening block, `PRIVATE
-ffast-math` on the codec only (never on anything that parses a bitstream),
hidden visibility. Targets: `volcomp` (shared, the codec), `volcomp_shard`
(opt), `volcomp_gpu` (opt), `vc` (CLI), tests, `VC_BENCH` (off by default;
zstd optional, drop the `C5D_ZFP_ROOT` prefix), `VC_TOOLS` (priors trainer).
Collapse `C5D_MARCH`/`C5D_MCPU` into `VC_NATIVE`; drop `msan` unless an
instrumented libc exists; drop `C5D_BRANCH_PROTECTION` or fold into hardening
on aarch64; make the ccache launcher conditional; add `install(EXPORT)` +
`.pc`. Put fuzz instrumentation on a separate object library instead of
leaking `-fsanitize` link flags to every consumer.

**CLI** (`vc`): `encode`, `decode`, `info`, `verify` (prints the goal metrics),
`shard-{stat,unpack,pack}` if the shard target is built. Drop `remote*`,
`tifxyz-*`, `slice`, `diff`, and `label-*` (label codec removed, §0 item 10). Dedupe the
`read_all`/`write_all`/`load_cube` helpers and the hand-rolled `--k=v` parsing.

**Benchmark**: `tools/bakeoff` is production quality and already computes
exactly the goal metrics — MAE, nearest-rank P90/P95/P99/max from an error
histogram, PSNR (per-chunk and pooled-MSE), 3-D SSIM (11-tap Gaussian σ=1.5,
standard constants, box-scoped variant), a blocking metric (boundary vs
interior RMSE amplification), enc/dec MB/s, latency P50/P90/P95/P99, JSON
ledger with git hash. Keep it as `vc-bench` with codecs `vc`, passthrough,
zstd only. Pre-allocate SSIM's seven 4 MB float buffers across chunks. Keep
`bench/bdrate.py`, `tools/prof/driver.c`, `tools/fetch_corpus.py` + lock,
corpus manifests (real data is not checked in; the `corpus/synth/*.json` stubs
have no payloads and nothing generates them — either add a generator or delete
them). `tools/ink_metric.py` is a downstream gate; keep or move to docs.
Add a **chunk-seam metric** (boundary-vs-interior gradient across 128³ faces,
same shape as the existing block metric) — it is the acceptance test for §6.

**Tests**: keep `test_quick`, `test_golden` (regenerate all vectors; drop the
ctx2 vector), `test_xdec` (shrinks to one C implementation + spec
re-implementation if `he_decode` leaves), `test_hostile` (port 1:1, add a seed
for C2); drop `test_label`; fold `test_stable` into the single-API test and fix C5;
drop `test_tifxyz`; register the GPU test. Fuzz: keep `d_brick`,
`rt_brick` (rename; make it actually assert the tau bound), don't check in corpora. Gaps: explicit
deblock edge cases (small dims, no-deblock),
chunk-seam bound, LOD downsample determinism, shard index integrity, a CI leg
that compiles the portable SIMD fallback (C1).

**Docs**: carry `spec/format.md` (rewritten in the new terms and without
ctx2/aniso/nway/TFX1; add the LOD downsample rule and the shard/zarr layout),
`docs/measured.md` (the decision graveyard — keep verbatim, append), a trimmed
BENCHMARKS with regenerated numbers. Drop PLAN.md history and BACKLOG's closed
items (or archive under `docs/history/`). `.clang-format` / `.clang-tidy` keep.

---

## 11. Baseline numbers collected during this review

(Report 06 in full. Host: WSL2, Core Ultra 9 275HX, 24 threads, clang 23,
no HW PMU. **No real corpus is present locally** — `corpus/dev|full` are
manifests only — so absolute numbers below are from a procedural 16-chunk
stand-in and are only useful as harness/relative signal.)

- Build: 0 warnings under the `bench` preset; `test_stable` fails to build (C5);
  6/6 remaining quick tests pass in 2.8 s. GPU target builds; `gputest` needs
  `corpus/dev`.
- Synthetic q2 auto-threads: 24.9× / 37.4 dB / MAE 2.31 / P90-95-99-max
  5-8-12-34 / enc 941 MB/s / dec 793 MB/s; 1T 357/466 MB/s. Lossless 2.07×.
- Toggles at q2/1T: `rdo=0.15` +8.4% ratio, −0.22 dB, enc −66%;
  `tau=4` max err 34→4 at −87% ratio (noisy synthetic content — real-corpus
  tau costs are far lower per BENCHMARKS.md); `nsub=8` +0.6%, `nsub=128` −2.2%;
  `eprior` auto already matched the better of forced on/off; `ctx2`/`nway`
  within noise at 1T (the toggle table's "ctx2=1 (default)" label is wrong —
  code default is off).
- Profile: tokenize+entropy ≈ 51%, DCT ≈ 38%, deblock ≈ 5%.
- Single 128³ decode latency: p50 8.27 ms serial / 3.39 ms auto; p99 12.4 / 4.9 ms.
- Sizes: `libc5d.a` 420 KiB; stripped binaries < 200 KiB; priors 640 B + 1216 B.
- A 48³ real fixture: the substream design *loses* to the single-stream
  baseline on ratio (20.7× vs 23.9×) — small chunks pay substream/header tax,
  which is what EPRIOR and the header shrink address for the LOD tail.

---

## 12. Decisions needed from the owner

1. ~~Chunk-seam design~~ — resolved: block-local deblock format change + level-uniform q (§0 item 11). Remaining sub-decision: *which* block-local filter (§6.3 candidates i-iii) — an experiment, not a policy call.
2. ~~16-bit / lossless / label codec~~ — all resolved (§0 items 6-10).
3. ~~GPU in v1~~ — resolved: CPU-only v1, GPU is v2 (§0 item 12).
4. ~~Shard layout / LOD downsampling~~ — resolved: zarr v3 `sharding_indexed`
   (§0 item 15); the codec does not downsample, levels are supplied (§0 item 18).
5. ~~Threading model~~ — resolved: none inside the library (§0 items 13-14).
6. ~~Pre-freeze experiments~~ — resolved: none; values baked (§0 item 16).
7. ~~`rdo` / `block_qmap`~~ — resolved: deleted (§0 item 16).
8. ~~Entropy tables ("priors")~~ — resolved (B). Background kept: Background: the rANS coder needs,
   for each of the 10 context models, a 32-entry symbol-frequency table so
   the decoder can invert the arithmetic (10 × 32 × u16 = **640 bytes**).
   c5d has two ways to get them to the decoder and today picks per chunk by
   exact bit cost (format flag 32):
   - *Transmitted*: the encoder histograms the chunk's own tokens and writes
     the 640 B into the stream. Perfectly adapted to that chunk's content;
     costs 640 B per chunk (≈0.6 % of a ~100 KB q2 stream, ≈6 % of a ~10 KB
     q8 stream).
   - *Embedded priors*: a fixed 640 B table compiled into the library
     (`priors_v1.bin`, trained by `c5d-train-priors` on the 512-chunk
     PHercParis4 corpus). Costs 0 bytes in the stream but is slightly
     mismatched to any given chunk, and is scroll-specific (Codex noted it
     was trained on the same corpus as the headline — in-sample).
   - Measured: at 128³ the two are *neutral* in total size (the auto choice
     mostly keeps transmitted tables); embedded wins big only on tiny chunks
     (16³: 876 → 249 B). **With zarr v3 every chunk is 128³ at every level
     (edge chunks are padded), so the tiny-chunk case no longer exists.**
   Options for a single baked config:
   - (A) keep the automatic per-chunk choice — most robust, but keeps one
     format branch, the trained blob, and the trainer tool.
   - (B) **always transmit** — simplest, content-agnostic, no trained data,
     no trainer, no flag; the only cost is 640 B/chunk, which can be cut to
     ~150-250 B with a compact table encoding (e.g. LEB128 delta-coded
     frequencies, as the label codec did) — a one-time format design
     choice, not an experiment.
   - (C) always embedded — zero bytes, but hard-codes one scroll's statistics
     into the format forever; degrades on unlike content and needs a
     training step whenever that changes.
   **Resolved by owner: (B), always transmit with compact table coding**
   (§0 item 19).
9. ~~tau optional?~~ — resolved: mandatory, default 2·q (§0 item 17).

---

## 13. Cross-reference with the Codex review (`~/c5d/docs/REWRITE_CODE_REVIEW.md`)

The Codex report (three specialist passes, same commit) was read after our
eight reports were written. Summary of the overlap:

**Independently found by both teams** (high confidence): DC-accumulator UB
(C2 = Codex #3); `test_stable` NDEBUG/-Werror break (C5 = #7); GPU public API
is the slow path while `gputest` owns the fast path; per-call GPU buffer
allocation and `vkQueueWaitIdle`; ~50 MiB GPU encode scratch + 11 clears;
`encode_tokenize` (1.58 GB/s) is the real GPU limiter, not rANS; per-chunk
GPU dispatch + barrier topology; `step_table` 4096 `powf` per call; ~21 MiB /
100+ heap ops per CPU encode; model tables rebuilt per decode and per block
decode; ctx2/aniso/rans4/generic-qmap/two-plane-u16 rejected; `encode_target`'s
seven full encodes; decode parallelism coupled to on-wire `nsub`; global
detached pool; `-ffast-math` on the whole library; byte-reversed shard magic;
shard footer/index un-CRC'd while snapshots are; backward recovery scan
cost; x86 CRC32C missing; stable façade's double copies and between-call-only
cancellation; `he_decode`/hybrid path is cruft; `gputest` not in ctest;
hand-mirrored GLSL constants; drop v0/wav/tifxyz/cache from the core;
tau has a catastrophic dense case (5.6 MiB for a 2 MiB chunk at tau2/q4 —
measured.md); `decode_chunk` amplification and lack of deblock.

**Found by Codex, missed by us, verified here**: X1-X9 in §3 (bypass writer
truncation, qmap effective-step UB, correction-index wrap, non-canonical
table/consumption acceptance, qmap-vs-deblock disagreement, block decode
violating tau, native-struct serialisation, unchecked decoder `bpa³`,
`rt_brick` checks no invariant). Also methodological points we did not
examine: the results ledger cannot identify an experiment (no `--set`
options, compiler, machine, threads, corpus hashes), `bdrate.py` silently
merges heterogeneous rows, `fetch_corpus.py` accepts any non-empty file, the
"full" corpus is one contiguous region of one scroll, and **the embedded
priors were trained on the same 512 chunks used for the headline** (in-sample).
Per-chunk clamped SSIM overstates by ~0.004 at q8; the cache caps each entry at
`budget/16` so a 2 MiB chunk cannot enter a 16 MiB cache.

**Found by us, not in Codex**: the build-breaking missing brace in the portable
`nz_extract` (C1); the odd-`blocks_per_sub` GPU word-RMW race (C3); the
`nsub`-doesn't-divide empty-substream waste (C13); doc/code drift on ctx2
default and `dz_dq` 0.20 vs 0.26 (C16); the DC-dead-zone and
magnitude-dependent `dz_dq` quality experiments; `RANS_PROB_BITS` 11 and
`RANS_MAX_SYMS` 32 as L1 levers; header shrink for LOD chunks (16% of a 249 B
16³ chunk); the specific dead-table inventory (`DCT_M`, `EVT`, `ODT`,
`SCAN8_TAB`, `M8`, ~2.5 KB); the four-way HybridUint/bitstream duplication
across brick/label/tifxyz/v0/wav; the label codec's `npal` alloc-DoS (C12) and
per-voxel type switch; the zarr v3 sharding alignment; the OME-Zarr LOD
terminology; baseline measurements on this host.

**Where the two reviews differ — owner decisions**:

| topic | Codex | this review | suggested resolution |
|---|---|---|---|
| Reuse of source | "start a new repository, copy no production source wholesale" | port with deletions | Carry the *measured kernels* verbatim-with-rename (SIMD DCT + its anti-optimiser guards, rANS decode loop, deblock kernel, label candidate model, hostile-parse checks, metrics.c); rewrite the *plumbing* (API, serialisation, allocation, pool, GPU host code, tools). Copying kernels preserves numbers that took campaigns to get; copying plumbing preserves the defects above. |
| API shape | context + arrays of jobs (`vc_encode(ctx, jobs, n)`), no process-global pool, async/GPU first-class | single-call functions + `threads` + `vc_shutdown` (§4) | **Not adopted (owner decisions 12-14, 29)**: no context, no arenas, no executor — stateless single-chunk synchronous calls with malloc/free; job arrays can appear in v2 with the GPU. |
| Random block decode | remove unless a workload is measured; if kept, true restart state + corrections | keep, with an open-handle to amortise parse | **Resolved by owner: kept** — non-zarr clients pull 16³ blocks directly (§0 item 21); block-local deblock + per-block corrections make it exact. |
| RDO | do not carry into the format | keep as opt-in encoder flag | RDO is encoder-only (no format impact); keep as an explicit encoder "effort" level after the port, off by default. |
| Lossless (grayscale) | decide whether it belongs in v1 | keep for CT | **Resolved by owner: dropped** (no lossless use case; q, tau ≥ 1). |
| u16 | delete the wrapper; add native later | redesign internals | **Resolved by owner: u8 only, wrapper deleted, no native u16.** |
| Deterministic integer reconstruction | experiment: fixed-point transform so CPU/GPU are bit-exact | not raised | **Resolved by owner: not needed** — float pipeline, agreement within a published margin is the contract. |
| Quantisation matrices | train static grouped 3-D quant tables per operating point, with SSIM/gradient/tail penalties | DC rounding + magnitude-dependent `dz_dq` first | Both; ours are hours, theirs is a campaign and needs a held-out corpus first. |
| Benchmark rigor | rebuild ledger/runner: full provenance, content-hashed scroll-level train/tune/test split, CIs, cold-cache modes, public-API timing only | keep bakeoff, add seam metric | Adopt Codex's provenance/split requirements into `vc-bench`; keep metrics.c. The in-sample priors must be retrained on a training split before the format freezes. |
| Seams | "define one production volume reconstruction; prefer treatment that does not require caller-driven mutation" | §6 (dedicated study) | Agreed in direction; §6 gives the concrete options and costs. |

---

## 14. Suggested port order

1. Decide §12 items 1-5.
2. New skeleton: header (§4), CMake (§10), naming (§0.1), CI with sanitizers,
   a portable-fallback compile leg, fuzz.
3. Port core codec with the deletions in §5.1/5.2 and §0 item 16, fixes
   C1/C2/C13-16 and X1/X3/X4, memoised step table, arena allocation, unified
   bitstream primitives; write the 16-byte header + CRC trailer; freeze spec.
4. Design and freeze the block-local deblock filter (§6.3 candidates) with
   the 128-plane seam metric as the gate; add level-uniform q to the packer.
5. zarr v3 shard writer/index parser + OME-Zarr `zarr.json` (levels supplied
   by the caller, one q per level); static-state audit for thread safety.
   (No label codec, no pool, no GPU, no ctx/arenas.)
6. CLI, bench (`vc-bench`), regenerated goldens, hostile suite, docs.
7. Re-measure on `corpus/full` (1-thread numbers are now the headline;
   report per-chunk latency and MB/s, and MB/s with N caller threads as a
   separate scaling table) and publish.
8. v2: GPU per §7 / report 02.

---

# Part II — Per-area reports (verbatim)



---

<!-- BEGIN 01-core-codec.md -->

# 01 — Core grayscale codec review (brick.c/h, format_internal.h, dct16, rans, priors, pool, platform)

Source read in full: `src/brick.c` (1807 L), `src/brick.h`, `src/format_internal.h`,
`src/transform/dct16.{c,h}`, `src/transform/dct_tables.h`, `src/entropy/rans.{c,h}`,
`src/entropy/priors.{c,h}`, `src/common/{platform.h,pool.c,pool.h,c5d_core.c}`,
`spec/format.md`, `docs/measured.md`, `docs/BACKLOG.md`, `docs/BENCHMARKS.md`.

Overall: the algorithm is in genuinely good shape and nearly all of it is
measurement-backed. The cruft is concentrated in (a) knobs that were measured
and left default-off, (b) an entire second context set that costs decode speed,
(c) three near-duplicate rANS entry points kept alive only by two dead
experimental codecs, (d) a header design that overloads a pad field and burns
40 B on every brick including 250-byte LOD bricks, and (e) two real
correctness/UB defects listed in §2.

---

## 1. Inventory and recommendations

### 1.1 `c5d_brick_params` (src/brick.h:19-38)

| Field | file:line | Rec | Reason |
|---|---|---|---|
| `q` | brick.h:20 | **KEEP** | The one knob users need. |
| `hf_exp` | brick.h:21 | **DROP from API, freeze at 0.65** | measured.md M1: 0.45/0.65/0.85 sweep is "near RD-neutral at iso-PSNR (±5%)". Four prior repos converged on 0.65. It is a per-brick f32 in the header (4 B) for a value nobody varies. |
| `axis_weight_z/y/x` | brick.h:22 | **DROP (with format flag 128)** | No measured RD result anywhere in measured.md/BENCHMARKS.md — only "implemented" in BACKLOG. It is the sole reason the header has an overloaded `_pad` field (format_internal.h:44). Costs validation code at brick.c:156-199, 878, 1392, 1473, 1527 + a GPU mirror. If anisotropic-resolution volumes appear later, add it as a v2 flag with its own field. |
| `dc_fine` | brick.h:23 | **DROP from API, freeze at 0.125** | Never swept in measured.md; never varied by any caller except the default. Another 4 header bytes. |
| `dz_q` | brick.h:24 | **DROP from API, freeze at 0.2** | Encoder-only (correctly not in the header). measured.md: "near-neutral 0.2..0.3 at iso-rate; 0.4 dominated. Keep 0.2." |
| `dz_dq` | brick.h:25 | **KEEP in header, DROP from API** | Normative (decoder needs it), but measured.md says "0.20 best by +0.03dB. Default 0.20" while `c5d_brick_defaults` sets **0.26** (brick.c:142). **Doc/code drift — resolve before freeze.** See §5.1. |
| `deblock` | brick.h:26 | **KEEP** | Normative flag 1; deblock is 3.9% of decode post-vectorization and materially reduces blocking (BENCHMARKS amplification table). |
| `lossless` | brick.h:27 | **KEEP** (see §3.6 caveat) | Flag 2. But note measured.md 2026-08-17: "c5d's grayscale lossless mode is 1.5-2.5x WORSE than zstd-19 on binary masks (it is a residual coder)". It is fine for CT, not for masks. |
| `tau` | brick.h:28 | **KEEP** | The hard-max-error bound is the codec's differentiator for scientific use. Flag 4. |
| `tau_percentile` | brick.h:29 | **KEEP, MERGE into tau** | Pure encoder policy, zero format/decoder cost, ~25 lines (brick.c:1053-1079) reusing the histogram that the hard case needs anyway. Express as `{float tau; float tau_pct;}`. |
| `rdo` | brick.h:30 | **KEEP, default 0** (or drop if minimizing hard) | measured.md 2026-08-06: matched rate q1.83+rdo.15 = 24.18x/43.00 dB/MAE 1.390/max 17 vs q2 24.16x/42.91/1.404/18; blocking amplification 1.3422→1.2758x. Encode ~2.4x. It is the only lever that improves *max* error and blocking at iso-rate. ~90 lines (brick.c:580-588, 665-731, 930-959). The old constant-rate "RDO-lite" (13%/dB vs an 18.9%/dB q-ladder) is already dead — do not resurrect. |
| `rans_nway` | brick.h:31 | **MERGE to a constant 2** | measured.md 2026-08-04: 2-way = +10% whole-decode 1T; 4-way "statistically identical speed, 3x the flush tax → rejected"; 2026-08-05 re-test: q8 dec 633 (nway4) vs 792 (nway2) — 4 is a *loss*. Keep flags 8/16 reserved in the format if you want, but drop 1 and 4 from the encoder API. Decoder should still accept nway=1 only if you care about v1.0 streams — you don't (no back-compat). **Recommend: format carries exactly 2-way, drop both flag bits and `c5df_flags_nway`.** |
| `eprior` | brick.h:32 | **KEEP the auto decision, DROP the tri-state** | The auto path is an exact per-brick entropy comparison (brick.c:971-997) that strictly cannot lose, and wins 3.5x on 16³ bricks (876→249 B). The `1`/`-1` overrides exist only because the GPU encoder can't emit priors (gputest.c:739, 1630) and for bakeoff sweeps. Replace with an internal flag, not a public param. |
| `nsub` | brick.h:33 | **KEEP** | Real measured tradeoff: nsub=128 gives full-GPU 557→1107 bricks/s but costs -0.7/-2.1/-6.1% ratio at q0.5/2/8. Default 32. Keep as a serving-profile knob. |
| `ctx2` | brick.h:34 | **DROP** — see §1.4 | Comment says "default on"; code sets `false` (brick.c:152). Another doc/code drift. |
| `chunk_qmap`, `chunk_qmap_n` | brick.h:35-36 | **KEEP** (flag 256) | Normative decoder input, cheap (1 B/chunk = 512 B/brick, only when present). The *generic variance heuristic* was measured RD-negative (-0.1 dB) and must stay unshipped; the map itself is for ROI/task allocation. Consider making it a separate `c5d_brick_encode_roi()` entry so the common struct stays 6 fields. |
| `nthreads` | brick.h:37 | **RENAME/MERGE** | Encode takes threads via the params struct, decode via a separate argument (`c5d_brick_decode_par`). Pick one. |

### 1.2 Public functions (src/brick.h)

| Function | file:line | Rec | Reason |
|---|---|---|---|
| `c5d_brick_defaults(q)` | brick.h:40 / brick.c:137 | **KEEP** | Needed with a designated-initializer struct. |
| `c5d_brick_encode` | brick.h:47 / brick.c:1158 | **KEEP** | Core. |
| `c5d_brick_decode` | brick.h:68 / brick.c:1742 | **MERGE** into `decode(…, nthreads)` | It is literally `decode_par(..., 1)` (brick.c:1743). |
| `c5d_brick_decode_par` | brick.h:69 / brick.c:1724 | **KEEP, RENAME** to `c5d_brick_decode` | Two names for one function is API noise. |
| `c5d_brick_decode_chunk` | brick.h:73 / brick.c:1746 | **KEEP** | The random-access story. But see §3.5/§4.6: it rebuilds all models + the 4096-entry `powf` step table and entropy-decodes up to `chunks_per_sub` chunks to yield 4 KB. |
| `c5d_brick_deblock_pair` | brick.h:79 / brick.c:524 | **KEEP** | Required for seam-correct multi-brick assembly; only test/harness callers today (test_quick.c:111) — make sure the new repo's assembly path actually uses it or it becomes dead. |
| `c5d_brick_encode_target` | brick.h:56 / brick.c:1767 | **MOVE to tools** | 7 full encodes for a ratio bisection (brick.c:1781-1801); only caller is `tools/c5dc/pack.c:66` (LOD ladder). Not core-codec surface. If kept, bisect on a tokenize-only cost estimate instead of 7 full encodes (≈3x faster). |
| `c5d_brick_encode_levels` | brick.h:52 / brick.c:1163 | **KEEP only if GPU encode ships** | Callers: `gputest.c:537/548/746`, `test_xdec.c:126`. It exists purely for the GPU forward-transform hand-off. If the new repo has no GPU path, DROP (removes `levels_natural` from `enc_ctx` and 12 lines in `tokenize_sub_task`). |
| `c5d_brick_token_counts` | brick.h:44 / brick.c:1372 | **DROP from public API** | Only caller is `tools/train_priors.c:63`. Duplicates 75 lines of `brick_encode_impl` setup (brick.c:1372-1448) purely to get histograms. Expose via an internal header used by the trainer, or add a `count_only` flag to the impl. The `uint64_t counts[608]` signature with a comment saying `counts[10*32]` (brick.h:43 vs :45) is itself confusing. |
| `c5d_brick_encode_u16` / `decode_u16` | brick.h:64,66 / brick.c:1231,1312 | **KEEP API, REDESIGN internals** | See §3.6 — correctness-tested but RD-unvalidated ("PHercParis4 in this corpus is u8, so no fabricated uint16 compression headline is reported", BENCHMARKS). |
| `c5d_version_string` / `c5d_status_string` | common/c5d_core.c:3,5 | **DROP or wire up** | `c5d_status` (c5d.h:22-29) is defined and never used by any codec function — every entry point returns bare `0/-1`. Either adopt the enum everywhere or delete both the enum and c5d_core.c. |

### 1.3 Format flags (src/format_internal.h:30-38 / spec/format.md)

| Flag | Rec | Reason |
|---|---|---|
| 1 DEBLOCK | KEEP | Normative, measured value. |
| 2 LOSSLESS | KEEP | |
| 4 TAU | KEEP | |
| 8 RANS2 / 16 RANS4 | **MERGE → remove both**; hardcode 2-way | 4-way measured worse (q8 633 vs 792 MB/s dec); 1-way measured 10% worse. Two flag bits + `c5df_flags_nway` + a mutual-exclusion check (brick.c:1471) + three decoder implementations of a runtime lane count, to express a constant. |
| 32 EPRIOR | KEEP | 3.5x on 16³ bricks; exact-cost auto decision cannot lose. *(Part I: overridden — dropped, §0 item 19.)* |
| 64 CTX2 | **DROP** | §1.4. |
| 128 ANISO | **DROP** | Unmeasured; forces the `_pad` overload. |
| 256 QMAP | KEEP | ROI/task allocation, normative, opt-in cost. |

### 1.4 legacy ctx (10 models) vs ctx2 (19 models) — **only legacy should survive**

Measured (measured.md 2026-08-04 "prev-magnitude context"; BACKLOG RD#2):
- ratio: **+0.4% (q2), +0.55% (q8), +1.2% (q0.5)** on the full corpus;
  +1.2–1.9% on small bricks. Offline estimate promised 1.7–4.1%; per-brick
  12-bit tables ate the rest.
- decode: **-5 to -7%** — "19 alias tables = 78 KB slot2sym, blows L1".
- Already shipped DEFAULT OFF (brick.c:152), and every golden/production
  stream uses the legacy set.

Cost of keeping it: `nm_of`/`dc_ctx_of`/`rctx`/`lctx` dispatch on a runtime
bool in the innermost decode loop (brick.c:70-81, 1657, 1669) and in the
tokenizer (brick.c:633, 766-799); a second 1216 B priors blob
(priors.c:336-341); doubled model arrays everywhere (`NMODELS = 19` sizes
`b_ctx.models` at ~82 KB of stack, brick.c:1453); ctx2-aware RDOQ rate code
(brick.c:675, 705, 728); a second golden vector; a mirrored branch in
host_entropy.c and in `entropy.comp`.

**Recommendation: DROP flag 64, `C5DF_NMODELS2`, `C5DF_DC_CTX2`,
`C5DF_RES_CTX2`, `c5df_run_ctx2`, `c5df_level_ctx2`, `priors_v2.bin`, and the
`ctx2` param.** Set `NMODELS = 10` everywhere; `rctx`/`lctx` become the plain
`c5df_run_ctx`/`c5df_level_ctx` with no branch. This is the single largest
simplification available in this area, it makes the decode inner loop
branch-free on context selection, and it costs ≤1.2% ratio on a feature that
was already never enabled. (If a future table-free/alias-free decoder lands,
re-add as v2 — measured.md already says exactly this.)

### 1.5 rANS surface (src/entropy/rans.h)

| Symbol | Rec | Reason |
|---|---|---|
| `rans_model_build` | KEEP | rans.c:109. |
| `rans_encode` / `rans_decode` | **DROP (dead)** | rans.c:145, 257. Zero callers anywhere in the tree (grep: definitions + header only). |
| `rans_dec` / `rans_dec_init` / `rans_dec_get` | **DROP** | rans.c:135-153. Only callers are `src/codec_v0.c:345,358,370` and `src/codec_wav.c:398,408,421` — the two dead experimental codecs (§2.1). |
| `rans_encode_multi` | **DROP** | rans.c:172. Same two dead callers only. Its body is `rans_encode_multin(..., nway=1)`. |
| `rans_encode_multin` | KEEP | rans.c:200. The production encoder. |
| `rans_encode_multin16` | **MERGE** | rans.c:275-310 is a **verbatim copy** of `rans_encode_multin` with `const uint16_t *ctxs` (only the label codec uses it, label.c:784). Make one function taking `const uint16_t*` and have the brick path widen its ctx array, or template it with a macro. 36 duplicated lines including the same comment twice. |
| `rans_decn` / `rans_decn_init` / `rans_decn_get` | KEEP | rans.h:66-100, the production hot loop. If nway is frozen to 2, `d->k`/`d->mask` become a single toggle. |
| `RANS_MAX_SYMS 64` | **CHANGE to 32** | rans.h:11. `NTOK` is 32 (format_internal.h:20); `freq[64]`+`cum[65]` wastes 130 B/model and pushes the hot `freq`/`cum` entries across cache lines. Label codec also uses 32 (`LB_NTOK`). |
| `RANS_PROB_BITS 12` | **evaluate 11** | §4.2 — halves `slot2sym` (the measured L1 problem) for a fraction of a percent of ratio. Must be decided *before* the format freezes. |
| `RANS_DEC_L` duplicated as `RANS_L` | fix | rans.h:64 vs rans.c:107 define the same constant twice with a comment admitting it. Single definition. |

### 1.6 DCT tables (src/transform/dct_tables.h)

| Symbol | Rec | Reason |
|---|---|---|
| `SCAN16_TAB` (line 53) | KEEP | Normative scan, 8 KB. |
| `EV` (23), `OD` (33) | KEEP | The even/odd butterfly tables actually used. |
| `DCT_M[16][16]` (line 5) | **DROP (dead)** | 1 KB, zero references outside the table file. |
| `EVT` (348), `ODT` (358) | **DROP (dead)** | 512 B, zero references. |
| `SCAN8_TAB` (311) | **DROP (dead)** | 1 KB, zero references. |
| `M8[8][8]` (43) | **DROP with codec_v0** | Only used by `pass_axis_bs` (dct16.c:324), which is only reachable through `c5d_dct_fwd_bs`/`c5d_dct_inv_bs`, whose only callers are `codec_v0.c:191,381`. |
| `c5d_dct_fwd_bs` / `c5d_dct_inv_bs` / `pass_axis_bs` | **DROP** | dct16.h:9-10, dct16.c:306-350. Lab-only generic path; dies with codec_v0. |

`dct_tables.h` is a header full of `static const` arrays included by 4 TUs
(brick.c:84, host_entropy.c:6, gputest.c:13, codec.c) — each gets its own copy
of the 8 KB scan table plus (today) the dead 2.5 KB. After the deletions,
consider making SCAN16 a real extern in one TU.

### 1.7 platform.h / pool.h / priors

| Symbol | Rec | Reason |
|---|---|---|
| `c5d_now_ns` | KEEP (tools only) | platform.h:10; only tools/bakeoff/prof use it. Belongs in the harness, not the codec lib. |
| `c5d_mul_size` / `c5d_add_size` | **DROP (dead)** | platform.h:17-18, zero callers. |
| `c5d_alloc_aligned` | **DROP (dead)** | platform.h:20, zero callers. After the three deletions `platform.h` has one function left → fold into the harness. |
| `c5d_parallel_for` | KEEP, redesign | §3.3. |
| `c5d_priors_v1_raw` | KEEP | priors.c:334. |
| `c5d_priors_v2_raw` | **DROP with ctx2** | priors.c:341 + `priors_v2.bin` (1216 B). |
| priors.h comment | fix | priors.h:11 says "640 bytes: 10 models" for a header that declares both v1 and v2; priors.c:322-326 header comment likewise only describes v1. |

---

## 2. Dead code, defects, drift

### 2.1 Two entire dead codecs still in the library target
`src/codec_v0.c` (392 L) and `src/codec_wav.c` (443 L) are compiled into
`libc5d` (CMakeLists.txt:88-89). `codec_wav` is the CDF 9/7 wavelet challenger
that measured.md records as **REJECTED** ("DCT-16 wins +105/+135/+190% ratio at
iso-PSNR"); `codec_v0` is the M1 baseline. They are the *only* users of
`rans_encode_multi`, `rans_dec_*`, `M8`, and `c5d_dct_*_bs`. **DROP both**;
that single deletion removes ~840 lines of codec plus ~120 lines of rANS API
plus 1.5 KB of dead tables. (They are outside my assigned files but they are
what keeps my assigned dead API alive — flagging for the merge.)

### 2.2 **BUG (build-breaking): `nz_extract` portable fallback is missing its closing brace**
`src/brick.c:280-286`:
```c
#else
static uint32_t nz_extract(const int32_t *lvl, uint16_t *pos) {
  uint32_t n = 0;
  for (uint32_t i = 1; i < BLKV; i++)
    if (lvl[i] != 0)
      pos[n++] = (uint16_t)i;
  return n;
#endif        /* <-- no '}' */
```
Any target that is neither `__aarch64__` nor x86 fails to compile. Nobody has
built one. Fix, and add a CI leg that compiles the generic path (e.g. with the
arch macros forced off) so the portable fallback stays real.

### 2.3 **UB: unbounded DC accumulator on hostile input**
`src/brick.c:1646`: `int32_t dc = prev_dc + unzigzag(u);`
`u` comes from `c5df_hyb_read` (format_internal.h:105-114), which permits
`tok = 31` → `k = 29` → `u < 2^30` → `unzigzag(u)` up to ±2^29. With
`chunks_per_sub` up to 512 (nsub=1 on a 128³ brick) a crafted stream drives
`prev_dc` past `INT32_MAX` — signed overflow UB, in the decoder, on attacker
data. The encoder is protected by `LEVEL_LIMIT` (brick.c:593) but the decoder
never re-imposes it. **Fix: reject `|dc| > LEVEL_LIMIT` after the add (or do
the add in `int64_t`/`uint32_t` and range-check).** The AC magnitude path is
fine (`mag + 1u` on a bounded `mag`, `pos += run` cannot wrap because
`run < 2^29`); the lossless residual path is already guarded (brick.c:1612)
and `apply_corrections` is guarded (brick.c:1563). This is the same UB class
the 2026-08-04 fuzz campaign found twice — it likely survives because it needs
hundreds of maximal DC tokens in one substream. Add a fuzz seed.

### 2.4 Doc/code drift
- `brick.h:34` — "prev-magnitude context set (format flag 64); **default on**"
  vs `brick.c:152` `.ctx2 = false` and measured.md "DEFAULT OFF".
- `brick.c:142` `.dz_dq = 0.26f` vs measured.md "dz_dq sweep: 0.20 best …
  Default 0.20."
- `brick.h:43` "counts[10*32]" vs `brick.h:45` "19 models x 32" vs the actual
  `uint64_t counts[608]` parameter.
- `brick.c:21-25` comment describes the model layout as "0..2 runs+EOB by band;
  3..8 levels; 9 DC" — correct for legacy, silently wrong once ctx2 is on.
- `priors.h:11` / `priors.c:322` describe only the 640 B v1 blob.
- `rans.h:1-2` "per-stream transmitted histogram" — no longer true with
  EPRIOR.
- `format_internal.h:3-8` claims three decoder implementations; if the GPU
  paths leave the new repo, this whole "cannot drift" apparatus (and
  tests/test_xdec.c) can shrink to one implementation + the spec re-impl.

### 2.5 Duplicated logic
- `rans_encode_multin` vs `rans_encode_multin16` — 36 identical lines
  (rans.c:200-235 vs 275-310).
- `deblock_face_task` (brick.c:419-469), `deblock_pair_row_task`
  (brick.c:489-522) and `deblock_u16` (brick.c:1184-1229) are three
  transcriptions of the same 4-tap gate+delta filter, with three separate
  copies of `c = clamp(0.8*q+1, 1, N)` (brick.c:472-476, 529-533, 1185-1189)
  and three different upper clamps (24, 24, 4096). Factor the kernel.
- `c5d_brick_token_counts` (brick.c:1372-1448) re-implements the parameter
  validation, nsub selection, step-table build and teardown of
  `brick_encode_impl` (brick.c:853-899, 1137-1155).
- Header float validation appears in `encode_params_valid` (brick.c:177-194),
  `b_parse` (brick.c:1478-1482), `c5d_brick_decode_u16` (brick.c:1319) and
  again in `src/gpu/codec.c:140-148`.
- The "flat chunk = DC*0.015625" reconstruction is written three times:
  encoder closed loop (brick.c:745), decoder (brick.c:1696), GPU dequant.

### 2.6 Unused / vestigial
- `enc_ctx.nsub` (brick.c:568) is assigned (brick.c:909, 1412) and never read.
- `b_hdr._pad` (format_internal.h:44) is not padding — it is the aniso weight
  field. After dropping flag 128 it becomes real padding that should be
  removed or repurposed explicitly.
- `c5df_dequant_mag` (format_internal.h:117) is declared as *the* normative
  reconstruction formula and is called by nobody — brick.c open-codes it at
  :1649 and :1677, the encoder at :743 and :758. Either use it everywhere or
  delete it.
- `C5DF_RES_CTX`/`C5DF_RES_CTX2` (format_internal.h:27-28) are aliases of
  `C5DF_DC_CTX`/`C5DF_DC_CTX2`; only `RES_CTX` is even `#define`d locally
  (brick.c:224) and then never used.
- `c5d.h` constants `C5D_SHARD_DIM`, `C5D_BRICKS_PER_SHARD`,
  `C5D_CHUNKS_PER_BRICK` are informational only.
- No `#if 0`, no debug env vars, no leftover printfs in the core files — this
  part is clean. (Env knobs `C5D_VK_*`, `C5D_GPU_STAGES` live only in
  `src/gpu/`.)

---

## 3. Bad designs / fragility

### 3.1 Header layout is the main obstacle to freezing the format
`c5df_hdr` (format_internal.h:41-45) is 40 B: `magic, dim, flags, nsub,
q, hf_exp, dc_fine, dz_dq, corr_n, _pad`.
- `dim` is redundant — every decode entry point already takes `dim` from the
  caller and rejects a mismatch (brick.c:1466).
- `hf_exp`, `dc_fine`, `dz_dq` are 12 B carrying values that are constant in
  every stream ever produced. `_pad` is an overloaded aniso field.
- `corr_n` forces the encoder to **rewrite the header after the payload**
  (brick.c:1131) and to `realloc` the whole blob (brick.c:1119).
- Net: after dropping aniso/hf_exp/dc_fine and folding dz_dq into a profile
  byte, a frozen header is ~16 B. At 128³ that is 0.03% (irrelevant); at a
  16³ LOD brick whose whole stream is **249 B** (measured.md, EPRIOR result)
  the 40 B header is **16% of the file**. Coarse LOD levels are exactly where
  the priors work already paid off — finish the job.
- Recommendation for the freeze: `u32 magic; u16 flags; u8 log2dim; u8
  profile; u32 nsub_and_corr…` or simply `magic, flags, nsub, q, corr_n`
  (20 B) with all other quant parameters implied by `profile`.

### 3.2 Error handling is untyped and inconsistent
Every function returns `0/-1`. `c5d_status` exists (c5d.h:22-29) with a
string table (c5d_core.c:5-13) and is used by **nothing**. Callers cannot
distinguish OOM from corrupt input from bad argument — which matters because
`brick_encode_impl` returns -1 for both (`goto out` at brick.c:899 for OOM and
at :860 for bad params). The `c5d_stable` façade (stable.h:30-40) has the
right enum; the core should use it directly rather than have a second
translation layer.

### 3.3 Threading model (`src/common/pool.c`)
- **Process-wide, permanent, unstoppable.** `pool_init` (pool.c:95-103)
  creates `min(available_cpus,256) - 1` detached threads on first use and
  `pool_worker` (pool.c:76-93) never returns. There is no shutdown, no
  `dlclose` safety, no fork safety, and a library user who asks for
  `nthreads=2` still pays for N-1 spawned threads.
- **Global mutex per work item.** Every item claim takes `g_pool.mu` and runs
  `find_runnable_job` (pool.c:63-67), a linear scan of a global job list.
  With `nsub=128` decode tasks that is 128 global-lock acquisitions per brick
  per job, plus a `pthread_cond_broadcast` on *every* completed item
  (`finish_item`, pool.c:72). Broadcast (not signal) with 100+ waiters is a
  thundering herd. Measured PMU says the codec is compute-bound at the thread
  counts tested, but 128-substream / many-core serving is exactly the case
  that isn't tested.
- **Parallelism is capped by nsub.** `c5d_parallel_for` clamps
  `nthreads > nitems` (pool.c:108); decode dispatches `nsub` items
  (brick.c:1731), so a 128³ brick at default nsub=32 can never use more than
  32 threads regardless of `nthreads`. Deblock dispatches `dim/16-1 = 7` items
  per axis (brick.c:479) with a hard barrier between axes → ≤7-way parallel
  for ~4% of decode.
- Recommendation: replace with a work-stealing or simple ticket pool that
  (a) is created lazily at the requested width, (b) can be destroyed,
  (c) uses an atomic index per job instead of a mutex per item, and
  (d) offers a "no pool" path when `nthreads==1` (already present, pool.c:110).
  Also consider letting the caller supply the pool — server embedders will
  want one pool for the whole process, not one per library.

### 3.4 Memory ownership
- `c5d_brick_encode*` return a `malloc`'d buffer the caller must free with
  libc `free()`. There is no `c5d_free()`, no allocator hook (the stable
  façade bolts one on by copying), and no way to encode into a caller buffer.
  For a server decoding/encoding thousands of bricks/s this forces a malloc +
  a full `realloc` copy on the tau path (brick.c:1119).
- Add `size_t c5d_brick_encode_bound(dim)` + `c5d_brick_encode_into(buf, cap,
  &n)` and a `c5d_free`. Decode already writes into a caller buffer (good).
- Encode leaks nothing (the `out:` teardown at brick.c:1137-1155 is correct and
  handles partial allocation), but it is 18 separate allocations per brick
  plus 3 per substream (§4.4).

### 3.5 Random access costs more than the spec implies
`c5d_brick_decode_chunk` (brick.c:1746) → `b_parse` (builds *all* models:
10 × 4096 B of `slot2sym` writes, plus a 4096-entry `powf` step table,
brick.c:1500-1508, 1527) → `b_decode_sub` entropy-decodes **every chunk from
the start of the substream** because DC is delta-coded within the substream
and rANS is serial. At nsub=32 that is up to 16 chunks decoded to deliver 1.
The spec's "any chunk subset decodes without touching other substreams"
(format.md §Substreams) is true but the intra-substream amplification is
undocumented. Either document it, or make DC prediction reset per chunk (costs
ratio), or recommend nsub=128 for random-access profiles (cps=4).

### 3.6 The uint16 wrapper is a workaround, not a 16-bit codec
`brick.c:1168-1370`. It splits each sample into `coarse = x/stride` and
`residual = x%stride`, then runs **two independent full C5B3 encodes**
(brick.c:1282, 1284). The residual plane is essentially white noise at a
16³-DCT — it gets a full forward DCT, quantization, tokenization and rANS pass
to compress something incompressible. Consequences:
- 2x transform + 2x entropy work on both encode and decode for stride>1.
- Two `malloc(n)` planes on each side (brick.c:1260, 1340).
- A serial single-threaded min/max scan over n voxels (brick.c:1240-1245) and
  a serial recombination loop (brick.c:1352-1362).
- `deblock_u16` (brick.c:1184-1229) is fully scalar, branchy, single-threaded,
  all three axes — the u8 path spent a whole measured campaign vectorizing
  exactly this.
- Tau is rejected outright (brick.c:1234), so the max-error guarantee — the
  headline feature — is unavailable on 16-bit data.
- **It has no measured RD result** (BENCHMARKS: "PHercParis4 in this corpus is
  u8, so no fabricated uint16 compression headline is reported").
Given that native micro-CT is 16-bit, this deserves a real design in the new
repo: carry the DCT in f32 (it already is), quantize once in original sample
units, widen the level alphabet (HybridUint already reaches 2^28), and let
`dz_dq`/steps do the work — the split exists only because the chunk gather
hardcodes `u8 - 128` (brick.c:350-409). Keep the C5U1 container API shape,
change what is under it. **Flag for cross-review: this is a design decision,
not a cleanup.**

### 3.7 Smaller fragility items
- `b_ctx` (brick.c:1451-1460) is a ~100 KB stack local (`models[19]` at
  ~4.4 KB each + `steps[4096]` + `dir[128]`), instantiated in
  `c5d_brick_decode_par` and again in `c5d_brick_decode_chunk`. It works, but
  it is a surprising frame size for a library, and it is why "decode allocates
  nothing" is true. Dropping ctx2 halves it.
- `b_decode_sub` recomputes the substream byte offset with an O(s) loop
  (brick.c:1577-1578) → O(nsub²) total. Harmless at 128, but store prefix
  offsets in `b_ctx` while parsing.
- Substreams can be *empty* when `nsub` does not divide `nchunk`: with
  `nchunk=8, nsub=5`, `chunks_per_sub=2` and substream 4 owns no chunks, yet
  still costs a 12 B directory entry + `4*nway` flush bytes. Clamp the emitted
  `nsub` to `ceil(nchunk / chunks_per_sub)` (brick.c:873-874).
- `axis_weights_pack` (brick.c:156-163) rounds `w*64+0.5` *after*
  `axis_weights_valid` checks the float range — the check and the pack use two
  different formulations of "fits in a nonzero byte". Moot if aniso is dropped.
- `deblock_par` is called on `recon` for tau (brick.c:1052) using
  `p->nthreads` while the surrounding code uses the local `nthreads`
  (brick.c:875) — same value today, but two spellings.
- `malloc(sizeof *o * sizeof h)` (brick.c:1249) — works out to 36 by accident
  (`sizeof *o == 1`); write `malloc(sizeof h)`.

---

## 4. Performance opportunities

Established baselines to beat (BENCHMARKS.md, full 512-brick corpus, q2):
1T 402 MB/s encode / 553 MB/s decode; auto-thread 2.48/3.02 GB/s.
Profile: DCT/IDCT ~43% of 1T wall before the x86 SIMD work, `pass_axis_gs`
16.8% after; deblock 3.9%; rANS stage ~25-30% of decode. PMU says
compute-bound, L1d miss 0.74%, no DRAM/TLB pressure.

### 4.1 `step_table`: 4096 `powf` calls per encode *and* per decode call
`src/brick.c:201-210`, called at brick.c:879 (encode), :1392 (token_counts),
:1527 (every `b_parse`, i.e. every decode *and* every `decode_chunk`).
In the isotropic case `radius = z+y+x` takes only **46 distinct integer
values (0..45)**; the loop computes `powf(1+radius, hf_exp)` 4096 times for 46
distinct results. Fix: build a 46-entry `powf` table then index it
(`steps[c] = q * pw[z+y+x]`). With aniso dropped this is unconditional.
Estimated: ~4096 scalar `powf` ≈ 250-350 kcycles ≈ 80-110 µs — **~2-3% of a
3.8 ms brick decode, ~13% of the measured 700 µs P50 auto-thread decode
latency, and >90% of a single-chunk decode.** Cheapest measurable win in the
file. Same for the `1.0f/steps[c]` reciprocal loop (brick.c:880-881, 1393-1394).

### 4.2 `slot2sym` is the known L1 problem — attack the table, not the models
measured.md already diagnosed it ("19 alias tables = 78 KB slot2sym, blows
L1"). Even the legacy set is **10 × 4096 = 40 KB** (`rans.h:16`), larger than
a 32 KB L1d. Three concrete levers, in order of value/risk:
1. **Drop ctx2** (§1.4): 78 KB → 40 KB, and removes a branch from the inner
   loop. Zero risk, already default-off.
2. **`RANS_PROB_BITS` 12 → 11**: `slot2sym` 40 KB → 20 KB, fits L1 alongside
   the working set. Cost is coarser probability quantization; for a 32-symbol
   alphabet with the measured token distributions this should be ≲0.2% ratio.
   **This is a format decision — measure it before freezing.**
3. **`RANS_MAX_SYMS` 64 → 32** (rans.h:11): -130 B/model and puts `freq[32]` +
   `cum[33]` in ~4 cache lines instead of 8.
Do **not** retry: `clz`-based multi-byte refill (-5..-9% on Oryon) or
`freq|cum<<16` packing (-4%) — both measured negative (measured.md 2026-08-05).

### 4.3 Encode: per-substream buffers are ~10x the input
`tokenize_sub_task` (brick.c:599-603) allocates, per substream, per call:
`toks` and `ctxs` at `chunks_per_sub*4096*2 + …` ≈ 131 KB each, and `byp` at
`chunks_per_sub*4096*6 + 64` ≈ **393 KB** ("worst case bypass ~43 bits/coeff",
brick.c:601). At nsub=32 that is **~21 MB of malloc per 2 MB brick encode**,
96 mallocs, all touched. Real bypass output is ~1 byte/coefficient. Fix:
size from a realistic bound with geometric growth on overflow (the writers
already return -1 on overflow: `bw_put` brick.c:100, and the token writes are
the unchecked ones — see §4.7), or use one per-thread arena reused across
bricks. Expect a measurable encode win on top of the 402 MB/s baseline and a
large RSS drop (measured max RSS 59 MB at 8T today).

### 4.4 Encode: allocation count per brick
`brick_encode_impl` does 8 `calloc`s (brick.c:883-890) + `recon` (895) +
output `malloc` (1027) + tau `malloc`/`realloc` (1081, 1093, 1119) + 3 per
substream. Collapse the 8 parallel arrays (`sub_rans/byp/toks/ctxs/ntok/bypn/
dir/counts`) into one `struct sub_state[nsub]` in a single allocation — also
better locality in the reduction loops (brick.c:925-928, 1008-1009).

### 4.5 Tokenize: two parallel byte arrays
`toks[]` and `ctxs[]` (brick.c:599-600) are written together and read together
(rans.c:211-216) but live in two streams. Pack into one `uint16_t` array
(ctx<<8|tok) — one cache stream instead of two in both the tokenizer and the
rANS encoder. (The label codec already needs 16-bit contexts, so this also
unifies `rans_encode_multin`/`multin16`, §1.5.)

### 4.6 `decode_chunk` latency
Per §3.5 + §4.1: a 4 KB chunk decode currently pays a full model build
(40 KB of table writes), a 4096-`powf` step table, and up to 16 chunks of
entropy decode. For a random-access serving path that is the dominant cost.
Fixes: cached/lazy step table (4.1), build only the models the substream
actually references, and expose a `c5d_brick_open()` handle that amortizes
`b_parse` across many `decode_chunk` calls on the same brick (today every call
re-parses).

### 4.7 Unchecked writes in the tokenizer
`toks[ntok] = t; ctxs[ntok] = …; ntok++` (brick.c:632-636, 770-773, 782-803)
has no bound check against the allocated size; correctness relies on the
worst-case sizing in brick.c:599-600 (2 tokens/coeff + 2/chunk). It is
provably sufficient today, but it is a silent-overflow hazard the moment the
token scheme changes. Add an assert or a checked emit; the buffer resize in
§4.3 needs one anyway.

### 4.8 SIMD coverage map (what is and isn't vectorized)
| Kernel | aarch64 | x86 | Gap |
|---|---|---|---|
| `gather_chunk` (brick.c:350) | explicit NEON | AVX2 | scalar fallback fine |
| `scatter_chunk` (brick.c:377) | explicit NEON | AVX2 + AVX-512BW | done (measured +43% end-to-end) |
| `nz_extract` (brick.c:232/252) | NEON vshrn | AVX2 | **broken generic path (§2.2)** |
| forward DCT (dct16.c:78, 201) | explicit NEON | autovec w/ cross-line vectorization enabled | done |
| inverse DCT (dct16.c:166, 269) | autovec | autovec, cross-line explicitly disabled | measured-optimal; explicit NEON inverse was slower in-app |
| encode quant loop (brick.c:657-662) | autovec | autovec | done (+13%) |
| deblock y/z (brick.c:450-468) | autovec width 16 | autovec | done |
| **deblock x (brick.c:429-441)** | scalar branchy | scalar branchy | ~1.3% whole-pipeline headroom; measured and deliberately deferred — leave it |
| **`deblock_u16` (brick.c:1184)** | scalar | scalar | 3 axes, no threads, no vectorization — the only completely un-optimized filter left |
| **`predict3d` lossless (brick.c:540)** | scalar per voxel | scalar per voxel | serial by construction (causal); a row-wise MED with the U-average lifted out is possible |
| RDOQ candidate loop (brick.c:679-730) | scalar | scalar | opt-in path, fine |
| rANS decode (rans.h:84) | scalar 2-lane | scalar 2-lane | load-latency bound; measured optimal in its naive form |

`pass_axis_gs` carries `__attribute__((noinline))` (dct16.c:253) and
`C5D_DCT_CONTIGUOUS_X` (dct16.c:25-29) — both are load-bearing anti-optimizer
guards with measured justification. **Keep them and keep the comments**; also
keep the byte-identical-blob cross-decode test that caught the -17% LTO
regression, it is the only thing that will catch it again.

### 4.9 Threading granularity
See §3.3. Concretely: decode parallelism ≤ nsub (brick.c:1731); deblock
parallelism ≤ dim/16-1 = 7 with a barrier per axis (brick.c:477-480). For
multi-brick workloads the right granularity is brick-level (the shard/cache
layer), which sidesteps both — make sure the new repo's batch API parallelises
over bricks and passes `nthreads=1` down, rather than nesting pools.

---

## 5. Ratio / quality opportunities

### 5.1 Reconstruction offset `dz_dq` — resolve the drift, then make it magnitude-aware (cheap, untested)
Code ships 0.26 (brick.c:142), docs say 0.20 (measured.md M1). Whichever is
right, the **untested** idea is a *different offset for |l|==1 vs |l|>1*: the
|1| bin is by far the most populated and its conditional mean sits closer to
the dead-zone edge than the higher bins'. This is one extra header byte (or a
frozen constant pair) and one `select` in the decoder inner loop
(brick.c:1677). Classic ~0.1-0.3 dB in still-image coders; not in the
graveyard, so it is genuinely open.

### 5.2 DC dead zone (cheap, untested)
The DC coefficient goes through the same `dz_q = 0.2` dead zone as ACs
(brick.c:657-662 applies `dzq` uniformly, including `c == 0`). DC error is a
whole-16³-chunk DC shift — the most visually and metrically expensive error,
and the direct driver of the blocking-amplification numbers
(1.34x at q2). Using `dz_q = 0.5` (round-to-nearest) for DC only costs almost
nothing in rate (DC is delta-coded and small) and should show up directly in
MAE/P99/blocking. Not in the graveyard. Try it first.

### 5.3 Header/table overhead on coarse LOD bricks
Per §3.1: 40 B header on a 249 B 16³ brick. With EPRIOR the freq tables are
already gone; the header is now the dominant fixed cost at the LOD tail, and
the LOD ladder generates 1/8 as many voxels per level but the same header per
brick. Shrinking to ~16 B is a straight ~10% win on the coarsest levels.

### 5.4 Things measured-NEGATIVE — do not re-litigate
- Wavelet (CDF 9/7): DCT wins +105/+135/+190% at iso-PSNR.
- Lapped/overlap transforms: seam energy ×2.2, MSE +54%.
- 8³ chunks: 16³ wins +27..+47% at iso-PSNR.
- Zero-tree parent context (-3..-15%), spatial-neighbour context (25-35% cost
  for ~5%), TCQ (-0.2 dB), per-chunk entropy tables (overfit).
- Cross-brick DC-plane prediction (0.1-0.5% of the stream); standalone
  DC-prediction and sign-prediction ~0 headroom (BACKLOG RD#5).
- Context-coding or zstd-ing the bypass stream: zstd-19 *expands* it
  (-0.01..-0.12%) — genuinely incompressible, closed.
- Golomb-Rice backend: +15.0/+41.4/+26.6% stream size at q0.5/2/8.
- Constant-rate "RDO-lite": 13%/dB vs an 18.9%/dB q-ladder — RD-negative.
- Generic log-variance qmap heuristic: -0.1 dB globally at matched rate.
- ctx2 / prev-magnitude context: +0.4-1.2% ratio for -5-7% decode (§1.4).
- 4-way rANS; substream sorting by token count (CPU-side ratio and GPU-side
  locality both lose).

### 5.5 Still open, ranked by (my estimate of) value/cost
1. DC round-to-nearest (§5.2) — hours, possibly the best MAE/blocking lever left.
2. Magnitude-dependent `dz_dq` (§5.1) — hours.
3. Header shrink for LOD (§5.3) — format work, ~10% on coarse levels.
4. `nsub` clamp to non-empty substreams (§3.7) — ~20 B/brick at odd nsub.
5. Full-candidate RDOQ beyond ±1 (gpudct's -2.8% BD was full-candidate; ours
   is adjacent-only and gets +0.8-1.0%) — expensive encode, already 2.4x.
6. A table-free/alias-free rANS decode path, which is the stated precondition
   for ctx2 ever being worth it.

---

## 6. Proposed minimal core API for the new repo

Design rules applied: one struct, six fields; one encode and one decode per
sample type; typed errors; explicit ownership; nothing that has not paid for
itself in `docs/measured.md`.

```c
/* volc/brick.h — core grayscale brick codec.
 * A brick is dim^3 voxels, dim % 16 == 0 (canonical 128; 64/32/16 for LOD).
 * Coding unit is a 16^3 chunk; the stream is split into `nsub` independent
 * substreams, each separately rANS-flushed, so any chunk subset decodes
 * without touching other substreams. */
#ifndef VOLC_BRICK_H
#define VOLC_BRICK_H
#include <stddef.h>
#include <stdint.h>

#define VOLC_CHUNK_DIM   16u
#define VOLC_NSUB_DEFAULT 32u   /* ratio-optimal */
#define VOLC_NSUB_MAX    128u   /* GPU-parallel serving profile */

typedef enum volc_status {
  VOLC_OK = 0,
  VOLC_E_ARG,       /* caller error: bad params, bad dim, null pointer   */
  VOLC_E_CORRUPT,   /* malformed or hostile bitstream                    */
  VOLC_E_NOMEM,
} volc_status;
const char *volc_status_str(volc_status);

typedef struct volc_params {
  float    q;         /* quantizer step in source sample units; >0        */
  float    tau;       /* 0 = off; else hard max |err| bound via sparse    */
                      /* corrections (full-brick decode only)             */
  float    tau_pct;   /* 100 = hard max; <100 bounds that percentile only */
  bool     lossless;  /* exact; q/tau ignored                             */
  bool     deblock;   /* normative decode-side seam filter (default true) */
  uint32_t nsub;      /* 0 = VOLC_NSUB_DEFAULT; else <= min(nchunk,128)   */
  unsigned nthreads;  /* 0 = all online CPUs, 1 = serial                  */
} volc_params;
volc_params volc_params_default(float q);   /* deblock=true, tau_pct=100 */

/* --- u8 --- */
size_t      volc_encode_bound(uint32_t dim);      /* worst-case byte count */
volc_status volc_encode(const volc_params *, const uint8_t *src_zyx,
                        uint32_t dim, uint8_t *dst, size_t dst_cap,
                        size_t *out_n);           /* no allocation         */
volc_status volc_encode_alloc(const volc_params *, const uint8_t *src_zyx,
                              uint32_t dim, uint8_t **out, size_t *out_n);
void        volc_free(void *p);                   /* frees volc_* output   */

volc_status volc_decode(const uint8_t *in, size_t in_n, uint32_t dim,
                        unsigned nthreads, uint8_t *dst_zyx);

/* --- u16 (dynamic-range container; see notes: internals need a redesign) */
volc_status volc_encode_u16(const volc_params *, const uint16_t *src_zyx,
                            uint32_t dim, uint8_t **out, size_t *out_n);
volc_status volc_decode_u16(const uint8_t *in, size_t in_n, uint32_t dim,
                            uint16_t *dst_zyx);

/* --- random access ---
 * volc_open parses the header, builds the entropy models and the step table
 * once; reuse it across many chunk decodes on the same brick. `in` must
 * outlive the handle. Decoding chunk (cx,cy,cz) entropy-decodes its
 * substream from that substream's first chunk (DC is delta-coded within a
 * substream) -- use a larger nsub for random-access-heavy profiles.
 * No deblock (needs neighbours); tau corrections are not applied. */
typedef struct volc_brick volc_brick;
volc_status volc_open(const uint8_t *in, size_t in_n, uint32_t dim,
                      volc_brick **out);
void        volc_close(volc_brick *);
volc_status volc_decode_chunk(volc_brick *, uint32_t cx, uint32_t cy,
                              uint32_t cz, uint8_t dst[4096]);

/* --- seam filter between two independently decoded adjacent bricks;
 *     `neg` precedes `pos` along axis 0=x,1=y,2=z. --- */
volc_status volc_deblock_pair(uint8_t *neg, uint8_t *pos, uint32_t dim,
                              uint32_t axis, float q, unsigned nthreads);

/* --- optional: per-chunk Q2.6 step multipliers for ROI/task-aware
 *     allocation (nonzero bytes, 64 == 1x, canonical chunk order).
 *     Deliberately NOT in volc_params: a generic activity heuristic
 *     measured RD-negative; this is for explicit application policy. --- */
volc_status volc_encode_roi(const volc_params *, const uint8_t *src_zyx,
                            uint32_t dim, const uint8_t *qmap, size_t qmap_n,
                            uint8_t **out, size_t *out_n);
#endif
```

Deliberately **not** in the public API (all justified above): `hf_exp`,
`dc_fine`, `dz_q`, `dz_dq`, `axis_weight_*`, `rans_nway`, `eprior`, `ctx2`,
`rdo` (make it a build-time/experimental entry point if kept),
`c5d_brick_token_counts` (internal header for the priors trainer),
`c5d_brick_encode_target` (tools), `c5d_brick_encode_levels` (GPU-only; keep
only if the GPU encoder ships), `c5d_brick_decode` vs `_par` (merged),
`rans_encode`/`rans_decode`/`rans_dec_*`/`rans_encode_multi` (dead),
`c5d_dct_*_bs` (dead), `c5d_mul_size`/`c5d_add_size`/`c5d_alloc_aligned`
(dead).

Internal headers to keep: `format_internal.h` (minus ctx2/aniso/nway),
`rans.h` (build + `encode_multin16` + `decn`), `priors.h` (v1 only),
`dct16.h` (fwd/inv only), `pool.h` (redesigned per §3.3).

---

## 7. Ordered work list for the port

**Must fix (correctness):** §2.2 missing brace; §2.3 DC accumulator UB.
**Delete:** codec_v0.c + codec_wav.c + their rANS/DCT dependents (§2.1, §1.5,
§1.6); ctx2 + priors_v2 (§1.4); aniso flag 128 (§1.1); rans_nway 1/4 (§1.1);
dead platform.h helpers; `c5df_dequant_mag` or its open-coded copies; the
`enc_ctx.nsub` field.
**Freeze:** header layout (§3.1) *after* deciding `RANS_PROB_BITS` (§4.2) and
`dz_dq` (§5.1) — both are format-visible.
**Cheap perf:** memoized step table (§4.1); `RANS_MAX_SYMS=32` (§1.5);
right-sized encode buffers (§4.3); packed tok/ctx (§4.5).
**Cheap quality experiments before freeze:** DC round-to-nearest (§5.2);
magnitude-dependent reconstruction offset (§5.1).
**Design decisions needing a human:** the u16 path (§3.6); whether the GPU
encode/decode paths come along (drives `encode_levels`, `format_internal.h`'s
raison d'être, and test_xdec).


<!-- END 01-core-codec.md -->


---

<!-- BEGIN 02-gpu.md -->

# 02 — GPU / Vulkan path review (c5d -> volume-compressor)

Scope: `src/gpu/{vk.c,vk.h,codec.c,codec.h,host_entropy.c,host_entropy.h,gputest.c,gpu.cmake}`,
`src/gpu/kernels/*.comp`, plus `src/format_internal.h`, skim of `src/brick.c`, and
`docs/{BACKLOG,BENCHMARKS,measured}.md`.

Line counts: vk.c 702, vk.h 110, codec.c 506, codec.h 37, host_entropy.c 475,
host_entropy.h 71, gputest.c 1702, gpu.cmake 56, kernels 1234 total (10 shaders).

Headline judgement: the **shipping API (`codec.c`) is the slow path** and the
**benchmark (`gputest.c`) is the fast path**. Every optimization the measured
docs credit (descriptor slicing, record-once + resubmit, multi-slot fences,
HOST_CACHED readback, cross-brick dispatch) lives only in `gputest.c` and is
never used by `c5d_gpu_decode_batch` / `c5d_gpu_encode_batch`. Porting those
into the library is the single highest-value action in this area.

---

## 1. Inventory + KEEP / DROP / MERGE / RENAME

### 1.1 Public GPU API (`src/gpu/codec.h`, 37 lines)

| symbol | verdict | note |
|---|---|---|
| `c5d_gpu_codec` opaque handle (codec.h:12) | KEEP | good shape |
| `c5d_gpu_codec_create(shader_dir, out)` (codec.h:15) | RENAME + RESHAPE | drop `shader_dir` (embed SPIR-V); take a `c5d_gpu_options` struct (device index/name, batch caps, flags) |
| `c5d_gpu_codec_destroy` (codec.h:16) | KEEP | |
| `c5d_gpu_codec_device_name` (codec.h:18) | MERGE | fold into one `c5d_gpu_info` query |
| `c5d_gpu_codec_subgroup_size` (codec.h:19) | MERGE | same; only consumer today is gputest's idle-estimate (gputest.c:1257) |
| `c5d_gpu_decode_batch` (codec.h:24) | KEEP (rework internals) | see §3/§4 |
| `c5d_gpu_encode_batch` (codec.h:33) | KEEP (rework internals) | see §3/§4 |

Missing from the API and needed: a capability/limits query (`max bricks per
batch`, supported flags), an error enum instead of bare `-1`, and a
device-resident output variant (BACKLOG.md:44 explicitly says device-resident
consumers can skip the readback ceiling, but the API cannot express it).

### 1.2 `vk.h` / `vk.c` — internal wrapper

Used-by matrix (counts of call sites, codec.c vs gputest.c):

- Used by both: `vk_init`, `vk_destroy`, `vk_buffer_create{,_cached,_device}`,
  `vk_buffer_destroy`, `vk_pipeline_create/destroy/bind_buffers`, `vk_begin`,
  `vk_dispatch`, `vk_barrier`, `vk_fill`, `vk_upload_range`,
  `vk_download_range`, `vk_submit_wait`.
- **gputest-only** (0 uses in the library): `vk_pipeline_alloc_sets` (vk.c:441),
  `vk_pipeline_bind_ranges` (vk.c:466), `vk_dispatch_set` (vk.c:537),
  `vk_timestamp_end` (vk.c:530), `vk_end` (vk.c:659), `vk_resubmit_wait`
  (vk.c:667), `vk_use` (vk.c:675), `vk_submit_async` (vk.c:680),
  `vk_wait_slot` (vk.c:688), `vk_last_gpu_ns` (vk.c:694).
- **Dead everywhere**: `vk_fill_range` (vk.c:547) — zero call sites in the repo.
  DROP.

Verdict: KEEP `vk.[ch]` as an internal (non-installed) helper, DROP
`vk_fill_range`, and **promote** the slicing/streaming/timestamp functions into
use by `codec.c` rather than deleting them (they encode the measured wins).
`vk_ctx` should stop being a public struct: `dev_index`, `dev_type`,
`timestamps_*`, `cur`, `cmds[4]` are implementation detail leaked to callers.

Redundancy inside vk.c: `vk_buffer_create` (vk.c:296) passes the *same* flags as
`want` and `fallback`, so the fallback machinery is a no-op there; and
`vk_buffer_create_device` (vk.c:302) duplicates ~40 lines of
`buffer_create_flags` (vk.c:251) instead of calling it with
`DEVICE_LOCAL` + no map. MERGE into one `vk_buffer_create(ctx, size, kind)` with
`kind ∈ {UPLOAD, READBACK, DEVICE}`.

`vk_pipeline` (vk.h:43-55) carries both `dset` and `dsets[8]` + two descriptor
pools — an artifact of bolting slicing on later. MERGE to a single pool + N sets
where N>=1.

### 1.3 `host_entropy.[ch]` — is the hybrid path cruft?

It contains **two separate things** that must be judged separately:

1. `he_decode` / `he_free` / `he_decoded` (host_entropy.h:15-39,
   host_entropy.c:110-297) — a full CPU run/level+rANS token machine producing
   **dense** `nchunk*4096` int32 levels. This is the "hybrid" front half.
   **CRUFT for the GPU path.** Its only remaining consumers are:
   - `gputest.c:275,345` (the legacy hybrid correctness/benchmark legs), and
   - `tests/test_xdec.c:102,147,177`, the CI cross-decoder gate.
   `codec.c` never calls it. The dense branch it feeds,
   `dequant_idct.comp:133-146`, is likewise unreachable from the library
   (`codec.c:280` always sets `.sparse = 1`).
   → **DROP `he_decode` and the dense shader branch.** If the cross-decoder gate
   is worth keeping (it is — it also checks the spec text, test_xdec.c:1-10),
   move the reference token machine into `tests/` as a deliberately independent
   spec reimplementation, next to the existing `spec_recon`
   (test_xdec.c:44-60) / `spec_steps` (test_xdec.c:31). It does not belong in a
   shipping library.

2. `he_gpu` / `he_gpu_setup` / `he_gpu_free` (host_entropy.h:49-69,
   host_entropy.c:299-475) — header parse, flag validation, rANS model build,
   subinfo/payload packing, tau LEB128 parse, exact `sum ceil(ntok/2)` pair
   bound. **NOT cruft**: `c5d_gpu_decode_batch` depends on it
   (codec.c:176). Model build on the host is a defensible design (it's O(19*32)
   and serial).
   → **KEEP, RENAME** to something honest, e.g. `c5d_gpu_stream_parse` /
   `c5d_gpu_stream_free`, and move it **into the GPU component**.

Does host_entropy belong in the core lib? **No.** `CMakeLists.txt:94` compiles
`src/gpu/host_entropy.c` into `libc5d` while `gpu.cmake:43` explicitly comments
"host_entropy.c lives in the core c5d lib". The only reason is `test_xdec`
linking `libc5d`. That is a test dependency dictating library layout — a bad
design. After splitting (1)/(2) above, the core lib keeps nothing GPU.

Duplication with `brick.c`: `he_decode`'s token loop (host_entropy.c:206-252) is
a near-line-for-line copy of `b_decode_sub`'s lossy branch (brick.c:1574-1717),
differing only in that it stops at integer levels. The header/dir/model parse in
`he_gpu_setup` (host_entropy.c:299-370) also duplicates `b_parse`
(brick.c:1462-1531). `format_internal.h` de-duplicates the *constants and
formulas* but not the *control flow*. In the new repo, factor a single
`brick_parse()` returning a `brick_view` used by CPU decode, GPU setup, and the
tests; then only the per-token loop differs.

### 1.4 `gputest.c` — test, benchmark, or both?

Both, badly mixed, and it is also a **third implementation** of the pipeline.
Structure:

- L165-240 corpus enumeration, device print, pipeline + buffer setup.
- L243-326 hybrid correctness pass (CPU oracle vs `he_decode` + dequant/deblock).
- L327-412 hybrid benchmark (50 reps, timestamps, transfer accounting).
- L413-572 forward-transform correctness + 3 benchmark legs + CPU-entropy hybrid rate.
- L573-872 full-GPU encode: correctness vs `c5d_brick_encode_levels`, benchmark,
  and an env-gated per-stage profiler (L813-851).
- L889-1618 "Milestone B": full-GPU decode correctness, batched benchmark with
  descriptor slicing (L1093-1426), streaming benchmark with fences (L1427-1578),
  plus a subgroup-imbalance estimator (L1255-1283) that is analysis, not a test.
- L1620-1687 public-API smoke incl. a hostile (zeroed rANS flush) case.

What should survive in the new repo:

- KEEP as a **test** (`ctest`-registered, currently it is *not* — `gpu.cmake`
  adds no `add_test`, so CI never runs it): the API-level correctness matrix —
  GPU decode vs CPU oracle within the documented tolerance, GPU encode
  byte-identical to CPU for the GPU profile, and the hostile smoke
  (gputest.c:1651-1674). That is ~200 lines against `codec.h` only, with no
  direct `vk.h`/kernel use.
- KEEP as a separate **benchmark** binary: the throughput legs, expressed
  through the public API (batched decode, streaming decode, batched encode).
- DROP: hybrid legs (L243-412) with `he_decode`; the forward-quant-only leg
  (L413-572) once full encode exists (it measures a stage, not a product — keep
  it only inside the env-gated stage profiler); the hand-rolled slicing/
  streaming drivers (L1093-1578) *after* their logic moves into `codec.c`; the
  subgroup-imbalance estimator (closed negative, see §2).
- DROP the duplicated push-constant structs (gputest.c:31-77) — they mirror
  codec.c:18-53 by hand and must be kept in sync with the GLSL `layout(push_constant)`
  blocks. MERGE into one `kernels/push.h` included by both C and (via generated
  defines) documented against each `.comp`.

Realistic size: ~250-line test + ~350-line bench, replacing 1702 lines.

### 1.5 Feature parity vs an explicit "GPU profile"

Current rejections:

- Decode: lossless rejected (host_entropy.c:311, and again at :131 in the dead
  `he_decode`); everything else supported (tau via `corrections.comp`, aniso
  weights, per-chunk qmap, arbitrary `nsub<=128`, nway 1/2/4, embedded priors,
  ctx2).
- Encode: `codec.c:353-355` rejects `lossless`, `tau>0`, `rdo>0`, `chunk_qmap`,
  and any `dim != 128`. `gpu_params_valid` (codec.c:137-151) re-validates floats
  by bit pattern because `-ffast-math` may fold `isfinite`.

Recommendation: **do not require full parity; define a documented GPU profile**,
but split it by direction because the reasons differ:

- **Decode must be at parity minus lossless.** A decoder that silently cannot
  read a valid stream is a footgun for a serving cache. Either implement the
  lossless 3D-predictor token machine as an 11th kernel, or make lossless
  streams a documented, *enumerated* rejection (`C5D_GPU_EUNSUPPORTED_LOSSLESS`)
  with an automatic CPU fallback in the caller-facing API. Prefer the fallback:
  `c5d_gpu_decode_batch` should never fail on a valid stream.
- **Encode is legitimately a profile.** `tau` and `rdo` are source-domain
  closed-loop searches, and `chunk_qmap` is application policy; these belong in
  the CPU encoder by construction (the code comment at codec.h:31 says exactly
  this and it is right). Publish it as `c5d_gpu_encode_profile` with a
  `c5d_gpu_encode_supports(params)` predicate so callers can branch without
  trial-and-error. `dim != 128` (codec.c:353) is *not* a profile decision though
  — it is an unimplemented generalization (`nchunk` is hardcoded 512 at
  codec.c:357); either generalize or name the constraint in the header.

---

## 2. Dead code, unused pipelines/descriptors, leftover experiments, env vars, duplication

### 2.1 Dead / unreachable

- `vk_fill_range` (vk.c:547-554): zero call sites. DROP.
- `he_decode`/`he_free`/`he_decoded` for the GPU path (§1.3.1): unreachable from
  `codec.c`.
- Dense-levels branch of `dequant_idct.comp:133-146`: only reachable with
  `pc.sparse == 0`, which the library never sets (codec.c:280). Along with it,
  the `pc_dq.levels_base` field (codec.c:278, dequant_idct.comp:23) is **dead in
  the sparse path** — `entropy.comp:222` writes absolute pair indices and
  `dequant_idct.comp:94` reads `pairinfo[...]` as absolute. Setting it is
  misleading.
- `pc_dq.bpa` (dequant_idct.comp:21) and `pc_fq.bpa` (forward_quant.comp:16) are
  always `dim/16` — derivable, DROP from push constants.
- `pc_ent.brick0` (entropy.comp:37) is always 0 from `codec.c` (never assigned in
  the `pc_ent` initializer at codec.c:256-266); only gputest's slicing sets it.
  Keep only if slicing moves into the library (it should).
- `pc_ent.tb_stride` is always `HE_NMODELS`=19 (codec.c:263); `pc_ent.st_stride`
  is always `nsub`; `pc_ent.sub_stride` always `nsub*7`; `pc_ent.pairinfo_stride`
  always `nchunk*2`. Four push constants that are functions of two others.
- `slot2sym` upload path: skipped for nsub∈{32,64,128} (host_entropy.c:376), and
  `codec.c:201` allocates a 4-byte dummy otherwise. The binding still exists in
  `entropy.comp:17` and the `else` branch at entropy.comp:127-132 is only for
  "weird" nsub. Since `nsub` is an encoder knob we control, consider **dropping
  slot2sym entirely** and always using the shared-memory cum binary search
  (entropy.comp:117-126) — that removes a binding, a 76 KB/brick host expansion
  (measured.md:309-311 already found it dead weight), and a whole code path.
  The only cost is that arbitrary nsub uses a 5-step search instead of a table.
- `counts_buf` in gputest (gputest.c:227) is bound as `pairinfo` but named
  "counts" — leftover naming from the pre-sparse design; the same buffer is
  called `info` in codec.c:215 and `PairInfo` in the GLSL. Pick one name.

### 2.2 Closed experiments still visible

- **Token-sort scheduling (closed negative).** The experiment itself is gone,
  but its measuring apparatus remains: `cmp_u32_desc` (gputest.c:79-82) and the
  subgroup-imbalance estimator (gputest.c:1255-1283) exist only to print the
  20.6% -> 1.7% idle figure from BACKLOG.md:40-46 / measured.md:445-448. DROP
  both; the conclusion is in the docs.
- **fp16 shared-storage IDCT** — rejected at 5 LSB (measured.md:304). No residue
  in the code. Good.
- **Milestone A/B labelling** — "Milestone B" appears in entropy.comp:2,
  host_entropy.h:41, gputest.c:889/1055. Historical project phases leaking into
  identifiers and output. RENAME/DROP.
- `record_pipeline` (gputest.c:134-163) is the hybrid-only recorder; dies with §2.1.

### 2.3 Debug/behaviour env vars (all undocumented outside the source)

In `vk.c`: `C5D_VK_VALIDATE` (vk.c:86), `C5D_VK_DEVICE` (vk.c:115).
In `gputest.c`: `C5D_GPU_TAU` (169), `C5D_GPU_WZ`/`WY`/`WX` (171),
`C5D_GPU_ENC_BRICKS` (419), `C5D_GPU_ENC_PROFILE` (813), `C5D_GPU_NSUB` (927),
`C5D_GPU_QMAP_TEST` (929), `C5D_GPU_REPL` (1074), `C5D_GPU_NS` (1127),
`C5D_GPU_STAGES` (1295).

Recommendation: **no env vars in the library.** `C5D_VK_DEVICE` and
`C5D_VK_VALIDATE` become fields of the create-options struct (a library that
changes device based on the environment is unpredictable for an embedder). The
gputest ones become CLI flags of the benchmark binary; most should just die with
the legs they gate.

### 2.4 Duplicated shader code

- **EV/OD DCT basis tables (64 floats each) are literally duplicated** between
  `dequant_idct.comp:42-61` and `forward_quant.comp:26-45`, and both are
  hand-transcribed from `src/transform/dct_tables.h` (comment at
  dequant_idct.comp:41 says "Bit-identical to src/transform/dct_tables.h" — with
  no mechanism enforcing it). Fix: generate a `kernels/dct_tables.glsl` from the
  C header at build time and `#include` it (glslc supports `#include` with `-I`).
- `frequency_radius()` duplicated verbatim: dequant_idct.comp:32-38 and
  forward_quant.comp:49-55. Same fix.
- `band_of`/`run_ctx`/`level_ctx`/`dc_ctx` duplicated: entropy.comp:82-90 and
  encode_tokenize.comp:33-39. Same fix (`kernels/format.glsl`).
- Byte-in-word accessor triplets duplicated across four shaders:
  `bytev/bytew/bytebw` (entropy.comp:58-80), `token_byte/context_byte/bypass_byte`
  (encode_tokenize.comp:42-56), `token_byte/context_byte/rans_byte`
  (encode_rans.comp:20-26), `rans_byte/bypass_byte/put_byte/put_u16/put_u32`
  (encode_assemble.comp:24-35).
- The C side duplicates every push-constant struct twice (codec.c:18-53 vs
  gputest.c:31-77) and the deblock strength formula three times
  (`deblock_c` codec.c:124, `deblock_strength` gputest.c:124, `deblock_par` in
  brick.c:471 — and `flip_bound` gputest.c:89 re-derives it again).

### 2.5 Hand-mirrored constants that must stay in sync with `format_internal.h`

`format_internal.h:1-11` states the contract: GLSL "mirrors these definitions BY
HAND". Full list of what is mirrored and where:

| constant/formula | C source | GLSL mirror(s) |
|---|---|---|
| `C5DF_MAGIC` 0x33423543 | format_internal.h:18 | encode_assemble.comp:20 `MAGIC` |
| `C5DF_BLKV` 4096 | :19 | entropy.comp:281,288 (literal 4096), encode_tokenize.comp:28, dequant_idct.comp:63/122/135, forward_quant.comp:47/103 |
| `C5DF_NTOK` 32 | :20 | entropy.comp (literal 32/33 strides), encode_tokenize.comp:29, encode_model.comp:14, encode_rans.comp:16, encode_assemble.comp:22 |
| `C5DF_TOK_EOB` 31 | :21 | entropy.comp:45, encode_tokenize.comp:30 |
| `C5DF_NMODELS` 10 / `C5DF_NMODELS2` 19 | :22-23 | entropy.comp:182 (`nm = ctx2?19:10`), entropy.comp:46 `NMX`, encode_tokenize.comp:31, encode_model.comp:15, encode_rans.comp:17, encode_assemble.comp:21 |
| `C5DF_DC_CTX` 9 / `C5DF_DC_CTX2` 18 | :25-26 | entropy.comp:90, encode_tokenize.comp:39 |
| flag bits DEBLOCK/RANS2/RANS4/CTX2/ANISO (1/8/16/64/128) | :30-38 | encode_assemble.comp:42-44 (flag assembly, literals) |
| 40-byte header layout + field order | :41-45 | encode_assemble.comp:46-52 (hand-written offsets 0,4,…,36) |
| 12-byte `sub_dir` order (rans_n, bypass_n, ntok) | :48-50 | encode_assemble.comp:60-62 |
| `c5df_band_of` thresholds 128/1024 | :57-59 | entropy.comp:82, encode_tokenize.comp:33 |
| `c5df_run_ctx2` / `c5df_level_ctx2` formulas | :69-74 | entropy.comp:83-89, encode_tokenize.comp:34-38 |
| `c5df_unzigzag` / `c5df_zigzag` | :76-81 | entropy.comp:91, encode_tokenize.comp:40 |
| LSB-first bit reader semantics | :84-102 | entropy.comp:149-164 (`bget`), encode_tokenize.comp:58-69 (`bw_put`) |
| HybridUint token mapping (`tok<4` direct, else `2+msb`, `k=tok-2`) | :105-114 | entropy.comp:166-171, encode_tokenize.comp:71-81 |
| dequant formula `(|l|+dz_dq)*step` | :117-119 | dequant_idct.comp:105/130/141 |
| step law `q*(1+radius)^hf_exp`, DC `q*dc_fine` | brick.c:201-210 (`step_table`) | dequant_idct.comp:128-129, forward_quant.comp:104-105 |
| `RANS_DEC_L` 1<<23, 12-bit probs | entropy/rans.h:9,64 | entropy.comp:44 `RANS_L`, encode_rans.comp:18, and the magic `524288u*f` (=(1<<19)*f) at encode_rans.comp:48 |
| model normalization (floor(c*4096/total), min 1, biggest-gets-remainder) | entropy/rans.c `rans_model_build` | encode_model.comp:24-64 (reimplemented with `umulExtended`) |
| deblock strength `clamp(0.8q+1,1,24)` and delta `(3d±4)/8` | brick.c:419-479 | deblock.comp:29 + `pc.c` supplied by host (codec.c:124) |
| SCAN16 table | transform/dct_tables.h | uploaded as an SSBO (codec.c:93-94, entropy.comp:19) — the one constant done right |
| EV/OD basis | transform/dct_tables.h | dequant_idct.comp:42-61, forward_quant.comp:26-45 |
| chunk side 16, `nsub<=128` | brick.h:13-18 | implicit everywhere (`>>4`, `&15u`, `*16u`) |

That is ~20 independent hand-sync points. The mitigation today is "run
c5d-gputest" — which CI does not run (no `add_test` in gpu.cmake) and which
requires a GPU. In the new repo: generate `kernels/format.glsl` and
`kernels/dct_tables.glsl` from the C headers at build time (a 60-line generator),
leaving only the header *byte layout* in encode_assemble as hand-written — and
cover that with a host-side assertion that a GPU-encoded header parses via
`c5df_hdr`.

---

## 3. Bad designs / fragility

### 3.1 Resource lifecycle — allocate-per-call

`c5d_gpu_decode_batch` creates **12** buffers (codec.c:217-224) and destroys them
all (codec.c:320-331) on every call; `c5d_gpu_encode_batch` creates **14**
(codec.c:379-387, destroyed :491-504). Each is `vkCreateBuffer` +
`vkAllocateMemory` + `vkMapMemory`. Per-call `vkAllocateMemory` is tens to
hundreds of microseconds and is also subject to `maxMemoryAllocationCount`
(4096 on many drivers). This makes small-batch latency dominated by allocation,
directly contradicting the PLAN.md:141 goal of a <50 µs cold path.
Fix: a persistent arena in `c5d_gpu_codec` sized on `create` from declared max
batch, with sub-allocation by offset (the descriptor-range machinery to bind
sub-ranges already exists: `vk_pipeline_bind_ranges`, vk.c:466).

Also: descriptor sets are re-written on every call
(`vk_pipeline_bind_buffers`, 10 calls per encode at codec.c:392-397) — with a
persistent arena they'd be written once.

No `VkPipelineCache` anywhere; `c5d_gpu_codec_create` compiles 10 pipelines from
scratch every process start (codec.c:79-88).

### 3.2 Synchronization

- `vk_submit_wait` uses `vkQueueWaitIdle` (vk.c:654, :671) — stalls the whole
  queue rather than waiting on this submission's fence, and prevents any
  overlap. The fences already exist (`vk_ctx.fences[4]`, vk.h:23) and the
  streaming path uses them (vk.c:680-691) — the library path just doesn't.
- Every dispatch in `codec.c` is followed by a **full global memory barrier**
  (`vk_barrier`, vk.c:566): decode issues one per brick for dequant
  (codec.c:286), one per deblock axis per brick (codec.c:297), one per
  corrections dispatch (codec.c:304). For n=24 bricks with deblock that is
  ~96 full pipeline drains where 5 would do (entropy | dequant | deblock x3).
  Bricks are independent; barriers between them are pure loss.
- `vk_begin` deliberately omits `ONE_TIME_SUBMIT` (vk.c:520-522) so buffers can
  be resubmitted — but `codec.c` never resubmits, so it pays the
  reusability cost for nothing.
- `vk_wait_slot` on a fence that was never submitted blocks forever; the ordering
  invariant is maintained only by convention in gputest.c:1520-1543. Fences
  should be created signalled, or the slot should carry an `in_flight` bit.
- The timestamp query pool has exactly 2 entries (vk.c:199) and per-context
  recording flags (`timestamps_recording`, `timestamp_closed`, vk.h:28-29) —
  stateful, non-reentrant, and forces the stage profiler to re-record the whole
  stream per stage (gputest.c:819-846).

### 3.3 Buffer sizing / `maxStorageBufferRange`

- `codec.c:205-211` validates 12 sizes against `g->vk.max_ssbo_range` and simply
  returns `-1` if the batch is too large. The caller has no way to know the
  limit, so the only strategy is guess-and-retry. Either (a) split internally
  (the library knows `max_ssbo_range` and the slicing primitives exist), or
  (b) expose `c5d_gpu_max_batch(gpu, dim)`. Prefer (a) with (b) as a query.
- The encode output capacity is `n * nvox * 4` (codec.c:370) = **8 MiB per
  128³ brick** for a stream that measures ~100 KiB (BENCHMARKS.md:53 ratio
  20.58x → ~102 KiB). Worse, `vk_fill(&outb, 0)` (codec.c:446) zeroes all of it
  every call because `encode_assemble.comp:27` composes bytes with `atomicOr` and
  therefore requires a zeroed destination. At n=8 that is 64 MiB of fill per
  encode. Fix: bound the capacity at something like `nvox/2 + header`, and/or
  change assembly to write whole words (each substream's payload is contiguous;
  only the 0-3 boundary bytes need atomics) so no clear is needed.
- **Latent race (real bug):** the byte writers in `encode_tokenize.comp:42-56`
  and `encode_rans.comp:22-26` do a *non-atomic* read-modify-write of the
  enclosing 32-bit word. That is safe only if each substream's region is
  4-byte aligned. `token_stride = cps*(4096*2+2) = cps*8194` (codec.c:364) and
  `rans_stride = ts*3+64`. `8194 ≡ 2 (mod 4)`, so for **odd `chunks_per_sub`**
  both strides are ≡2 (mod 4) and adjacent substreams share a word → two lanes
  RMW the same word concurrently → corrupted tokens/rANS bytes. Odd `cps` is
  reachable with valid parameters: `nsub=110` gives `cps=ceil(512/110)=5`,
  `ts=40970`. The gputest smoke happens to use `nsub=100` (`cps=6`) and the
  benchmark `nsub=128` (`cps=4`), which is why it has never fired. **Fix: round
  `token_stride`, `bypass_stride`, `rans_stride` up to a multiple of 4** (or
  better, 16). `bypass_stride = cps*4096*6+64` is always a multiple of 4, so only
  the token/context/rANS strides are affected.
- `pair_stride` is the max over the batch (codec.c:183-184), so one dense brick
  inflates scratch for all n. Acceptable, but worth a comment or a per-brick base
  array.

### 3.4 Error propagation from shaders

Three different mechanisms: a `status[]` SSBO per substream (entropy.comp:310),
`submeta[gsub*6+2]` err words (encode_tokenize.comp:149, encode_rans.comp:64),
and a `brickmeta[b*4+2]` bad flag (encode_prefix.comp:28). All are checked only
after a full submit+wait, all collapse to `return -1`.

Concrete problems:
- **Decode writes the caller's buffer before checking status.** `codec.c:312-314`
  memcpys every brick into `dst`, and only then (codec.c:315-317) inspects
  `status`. On a corrupt stream the caller receives garbage *and* an error; if
  they ignore the return, they render garbage. Swap the order.
- No error taxonomy: truncated substream, model-normalization failure, output
  overflow, and "unsupported flag" are indistinguishable.
- Encode: `enc_fail` (codec.c:485) frees `ov[b]` for all b including entries
  never allocated — they are `calloc`'d so `free(NULL)` is fine, but the
  `ov`/`on` allocation at codec.c:462 sits *after* `vk_submit_wait`, inside the
  same scope crossed by earlier `goto enc_done`, which is fragile.

### 3.5 Device selection

`vk_init` scores devices by type + shared-memory + workgroup size
(vk.c:52-71) and honours `C5D_VK_DEVICE` as index-or-substring (vk.c:115-135).
Problems: environment-driven behaviour in a library (§2.3); no way for an
embedder to pass an existing `VkDevice`/`VkInstance` (an application that already
has Vulkan up will create a second instance and second device); no queue sharing;
`VkDeviceCreateInfo` enables **no features and no extensions** (vk.c:175-178),
which forecloses subgroup ops, `shaderInt16`, `VK_KHR_shader_float16_int8`,
buffer-device-address, and timeline semaphores. At minimum the new API should
accept an optional "adopt this device/queue" path.

### 3.6 Shader loading from a build-directory path

`C5D_GPU_SPV_DIR` is baked as the build tree's `spv/` dir (gpu.cmake:46) and
`codec.c:78` falls back to `"."`. An installed `libc5d_gpu.a` therefore cannot
find its shaders — the SPIR-V is not installed, not versioned with the binary,
and a stale `spv/` silently produces wrong results. `gputest` has its own
duplicate define `C5D_SPV_DIR` (gpu.cmake:54).

**Fix: embed the SPIR-V.** The project is C23, so `#embed` is available (or
generate a `.h` of `static const uint32_t[]` at build time for portability).
Then `c5d_gpu_codec_create` takes no path at all, the shaders are guaranteed to
match the binary, and `read_file` (vk.c:352-375) plus the `snprintf` path
plumbing (codec.c:63-69) disappear. 10 shaders at ~5-20 KiB of SPIR-V each is
~100-200 KiB — nothing.

### 3.7 Scratch memory per brick

Decode scratch per brick (codec.c:196-204): pairs `pair_stride*4` (measured
~1305 KiB at q2, BACKLOG.md:41), vol 2 MiB, readback 2 MiB, payload, freq 2.4 KiB,
cum 2.5 KiB, slot 0-304 KiB, sub, status, pairinfo 4 KiB, qmap 2 KiB, corr.
Encode scratch per brick (codec.c:365-370) is far worse: levels `512*4096*4` =
**8 MiB**, tokens `nsub*ts` and contexts the same (nsub=32: 32*131104 ≈ 4 MiB
each), bypass `nsub*bs` ≈ 12 MiB, rANS ≈ 12.6 MiB, counts `nsub*19*32*4` = 78 KiB,
output 8 MiB → roughly **50 MiB of scratch per brick encoded**, all allocated and
freed per call, and mostly zero-filled (codec.c:436-446). This is the reason
"the full GPU encoder is underoccupied at small batches"
(BENCHMARKS.md:180-183) — you cannot batch deeply when each brick costs 50 MiB.
Tightening `token_stride`/`bypass_stride`/`rans_stride` (they are worst-case
"every coefficient nonzero" bounds; real q2 streams are ~100 KiB) is the single
biggest lever on encode batch depth.

Also: `readback` is allocated unconditionally (codec.c:218) but only used when
`discrete` (codec.c:308-314) — a wasted 2 MiB/brick on every integrated-GPU
decode.

### 3.8 Memory types — the shipped path misses a measured win

`measured.md:298-303` records that HOST_CACHED readback doubled end-to-end
throughput ("uncached mapped reads were ~1.5 GB/s and halved the loop").
`gputest` uses `vk_buffer_create_cached` for its readback targets
(gputest.c:1142, :1145). But in the library, for a **non-discrete** device
`vol` is created with `vk_buffer_create` (uncached HOST_VISIBLE|HOST_COHERENT,
codec.c:217) and read directly with `memcpy` at codec.c:314 — precisely the slow
uncached read the docs say to avoid. Symmetrically on **discrete** devices the
encode output `outb` is created host-visible (codec.c:387) so
`encode_assemble.comp` performs per-byte `atomicOr` **across PCIe**; gputest
instead used a device-local `outb` + `vk_download_range`
(gputest.c:628, :709). Both are straightforward fixes with measured upside.

---

## 4. Performance — concrete items with citations

Measured baseline (BENCHMARKS.md:44-62, measured.md:435-448): decode 16.60 GB/s
compute / 4.05 GB/s streaming (RTX 4060, 96-brick resident); compact 24-brick
8.44 / 4.68 GB/s. Encode 0.98 GB/s device / 0.94 GB/s end-to-end at 8 bricks.
Isolated encode stages: forward-quant 45.00, tokenize+hist **1.58**,
model-normalize 20.83, reverse rANS **8.11**, prefix 138.15, assemble 94.53 GB/s.

Note the docs name reverse rANS as "the compute limiter", but the stage table
itself shows **tokenize+hist at 1.58 GB/s is 5x slower than rANS** — that is the
actual limiter and the priority target.

### 4.1 Encode limiters

1. **`encode_tokenize.comp` (1.58 GB/s) — one lane per substream scanning all
   4096 scan positions per chunk.** `encode_tokenize.comp:118` loops
   `for (i = 1; i < BLKV; i++)` and reads `levels[lbase + ci*BLKV + scan[i]]`
   for every position, i.e. 4096 fully **scattered** global loads per chunk,
   discarding the ~90-99% that are zero. The CPU encoder does not do this — it
   uses `nz_extract` (brick.c:232/270/280, SIMD) to get the nonzero positions
   first. Fixes, in order of value:
   - Have `forward_quant.comp` emit a compact per-chunk nonzero list (it already
     has the whole chunk in shared memory at forward_quant.comp:103-110); one
     `atomicAdd` on a per-chunk counter in shared memory, then a coalesced
     write. Tokenize then walks ~50-500 entries instead of 4096.
   - Every token write is a non-atomic word RMW (`token_byte`,
     encode_tokenize.comp:42-46) — 3 words touched per token (token, context,
     bypass). Buffer 4 bytes in a register and write whole words.
   - `counts[...]++` (encode_tokenize.comp:90) is a global-memory increment per
     token with no atomics (safe only because each substream owns its row) —
     but it is a dependent global RMW in the inner loop. Keep the histogram in
     shared memory per workgroup, flush once.
2. **`encode_rans.comp` (8.11 GB/s)** — one lane per substream, inherently
   serial per substream, byte-at-a-time word RMW (`rans_byte`,
   encode_rans.comp:22-26), and a `x[lane] / f` + `x[lane] % f` **integer
   division pair per symbol** (encode_rans.comp:54). Fixes: precompute
   reciprocal/shift per (model,symbol) in `encode_model.comp` and use the
   standard multiply-shift division-free rANS encode; buffer output bytes in a
   register word. Parallelism only improves with more substreams (nsub=128 is
   already the GPU profile) — the per-symbol cost is what to attack.
3. **`encode_model.comp` runs `local_size_x = 1`** (encode_model.comp:6) with
   one invocation per brick doing `nmodels * NTOK * nsub` global reads
   (encode_model.comp:42-48 — 19*32*128 = 77,824 serial loads per brick) plus
   608 twelve-iteration binary searches (`scaled_freq`, :25-33). 20.83 GB/s only
   because it is small. Make it one workgroup per brick, one lane per token,
   subgroup-reduce the per-substream counts, and replace the binary search with
   a 64-bit multiply (or `umulExtended` once) — it is `floor(count*4096/total)`.
4. **`encode_prefix.comp` runs `local_size_x = 1` and a single global
   invocation** (encode_prefix.comp:12) serially over `nbrick*nsub`. Fast in
   absolute terms (138 GB/s), but it costs a full dispatch + barrier
   (codec.c:457-458) for a prefix sum. MERGE into the head of
   `encode_assemble.comp` (or do it host-side after a small readback) and drop a
   pipeline, a barrier, and 3 descriptors.
5. **`encode_assemble.comp` dispatches only `n` workgroups**
   (codec.c:459, `gl_WorkGroupID.x` = brick at encode_assemble.comp:38) and then
   loops **serially over substreams** (`for (s = 0; s < pc.nsub; s++)` at
   encode_assemble.comp:66) with 256 lanes cooperating inside each. At n=8 that
   is 8 workgroups on a device that wants thousands. Dispatch
   `n*nsub` workgroups, one per substream, and drop the serial loop; the
   `barrier()` at :64 then also disappears.
6. `pow()` per coefficient: `forward_quant.comp:104-105` evaluates
   `pow(1.0 + frequency_radius(c), hf_exp)` for **all 4096** coefficients of
   every chunk (the CPU builds a 4096-entry `step_table` once per brick,
   brick.c:201-210). Same in `dequant_idct.comp:129` (per nonzero) and :140.
   Fix: compute the 4096-entry step table once per workgroup into shared memory
   (16 `pow` per lane) or, better, upload it as an SSBO once per brick — it
   depends only on (q, hf_exp, dc_fine, qweights).
7. `vk_fill` of 11 buffers per encode (codec.c:436-446), ~50 MiB per brick (§3.7).
   Most are only needed because of non-atomic partial-word writes and `atomicOr`
   assembly. Removing the need to clear `tok`/`ctx`/`byp`/`rn`/`outb` alone
   should be visible end-to-end.

### 4.2 Decode limiters

1. **Per-brick dispatch + barrier in the library** (codec.c:271-307): n dequant
   dispatches each followed by `vk_barrier`, plus 3 deblock dispatches + barriers
   per brick. `gputest`'s batched leg instead loops slices inside one barrier
   region (gputest.c:1322-1365) — which is where 8.44 GB/s came from. Port that:
   put per-brick parameters (q, hf_exp, dc_fine, dz_dq, qweights, qmap flag,
   vol_base, pairinfo_base) into a small per-brick SSBO and dispatch
   `n*nchunk` workgroups once; likewise one deblock dispatch per axis over all
   bricks. Expected: 5 barriers instead of ~4n, and full occupancy at small n.
2. **`entropy.comp` is the decode limiter** and is one lane per substream with a
   serial rANS chain (entropy.comp:106-146). Subgroup idle from length imbalance
   was measured at 20.6% (BACKLOG.md:42) and sorting was rejected because it
   scattered payload reads. Untried alternatives: (a) interleave *within* a
   substream by using `rans_nway=4` and giving each lane one rANS lane
   (currently one thread runs all `nway` lanes round-robin at entropy.comp:144);
   (b) pack `nway` so a subgroup covers one brick's substreams contiguously and
   the payload stays contiguous — retaining locality while balancing.
3. **Readback is the streaming ceiling** (16.60 compute vs 4.05 end-to-end).
   Two fixes: use HOST_CACHED for the readback target in the library (§3.8), and
   add a device-resident output mode so a Vulkan consumer (texture atlas /
   renderer) never round-trips (BACKLOG.md:44 says this explicitly, the API
   cannot express it).
4. `dequant_idct.comp` flat-chunk fast path (dequant_idct.comp:99-120) is a good
   win already. Beyond it: `inv16` (dequant_idct.comp:66-86) keeps `ye/yo/E/O`
   as 8-element register arrays and does 128 MACs per line with `EV`/`OD` as
   `const float[64]` — on several drivers a `const` array indexed by a loop
   variable lands in scratch/global. Preloading EV/OD into `shared` (256 floats,
   1 KiB) alongside the existing `shared float s[4096]` (16 KiB) is cheap to try.
   Shared usage is 17 KiB at 256 threads — already limiting occupancy to 1-2
   workgroups/SM on 48-64 KiB parts; an fp32 16 KiB block is unavoidable, but
   the fp16 variant was measured and rejected at 5 LSB (measured.md:304).
5. **No subgroup intrinsics anywhere.** `vk_init` queries `subgroupSize`
   (vk.c:151-161) and it is used only for a printout (gputest.c:206) and the
   rejected idle estimate (gputest.c:1257). `VkDeviceCreateInfo` requests no
   features (vk.c:175). Natural candidates: `subgroupAdd` for the encode
   histogram merge (encode_model.comp:42-48), `subgroupExclusiveAdd` for the
   prefix stage, `subgroupBallot` for nonzero compaction in forward-quant.
6. Workgroup sizes are all fixed literals: entropy 64 (entropy.comp:11),
   dequant 256 (dequant_idct.comp:9), deblock 64 (deblock.comp:12),
   forward_quant 256 (forward_quant.comp:5), tokenize 64
   (encode_tokenize.comp:7), model 1, rans 64, prefix 1, assemble 256. Several
   are structurally required (256 = one lane per (z,y) row of a 16³ chunk); the
   64-wide ones and the two `local_size_x = 1` shaders are not. Use
   specialization constants so the size can follow `subgroupSize` (64 on AMD/
   Adreno, 32 on NVIDIA) without recompiling GLSL — `entropy.comp:186-199`
   hard-codes a "64 lanes = 2 bricks when nsub==32" assumption that a
   specialization constant would make explicit.
7. Recording cost: measured.md:254 notes host re-recording ~110 dispatches cost
   ~35 ms/submit on turnip (3x the GPU time), fixed in gputest by record-once +
   `vk_resubmit_wait`. The library re-records every call. With batched dispatch
   (§4.2.1) the dispatch count collapses anyway, which is the better fix.

---

## 5. Proposed minimal GPU API for the new repo

Single public header, no Vulkan types leaked, no env vars, no shader paths.

```c
/* gpu.h — optional Vulkan compute backend. Nothing here requires the caller
 * to link or include Vulkan. */
#ifndef VC_GPU_H
#define VC_GPU_H
#include <stddef.h>
#include <stdint.h>
#include "brick.h"   /* vc_brick_params */

typedef struct vc_gpu vc_gpu;

typedef enum {
  VC_GPU_OK = 0,
  VC_GPU_ENODEV,        /* no usable compute device                    */
  VC_GPU_EINVAL,        /* bad arguments                               */
  VC_GPU_EUNSUPPORTED,  /* valid stream/params outside the GPU profile */
  VC_GPU_ECORRUPT,      /* shader flagged a bad stream                 */
  VC_GPU_EOOM,
  VC_GPU_EDEVICE,       /* driver/submit failure                       */
} vc_gpu_status;

typedef struct {
  int32_t  device_index;   /* -1 = auto (best scoring)        */
  const char *device_name; /* NULL or substring match          */
  uint32_t max_batch;      /* bricks resident; 0 = pick from device limits */
  bool     validation;     /* Khronos validation layer, if present */
} vc_gpu_options;          /* {0}-initializable: all defaults are sane */

typedef struct {
  char     device_name[256];
  uint32_t subgroup_size;
  uint32_t max_decode_batch;  /* bricks per decode call, given dim */
  uint32_t max_encode_batch;
  bool     discrete;
} vc_gpu_info;

vc_gpu_status vc_gpu_create(const vc_gpu_options *opt, vc_gpu **out); /* opt may be NULL */
void          vc_gpu_destroy(vc_gpu *g);
void          vc_gpu_query(const vc_gpu *g, uint32_t dim, vc_gpu_info *out);

/* True iff params fall inside the GPU encode profile (lossy DCT, no tau/rdo/
 * qmap/lossless). Lets callers route to the CPU encoder without a failed try. */
bool vc_gpu_encode_supported(const vc_gpu *g, const vc_brick_params *p, uint32_t dim);

/* Decode n streams to dst[b*dst_stride]. Batches larger than the device limit
 * are split internally. Streams outside the GPU decode profile (lossless) are
 * reported via VC_GPU_EUNSUPPORTED with *first_bad set, so the caller can fall
 * back per brick. dst is untouched for any brick that failed. */
vc_gpu_status vc_gpu_decode(vc_gpu *g, const uint8_t *const streams[],
                            const size_t sizes[], uint32_t n, uint32_t dim,
                            uint8_t *dst, size_t dst_stride, uint32_t *first_bad);

/* Encode n bricks. One caller-owned arena receives every stream back to back;
 * offsets[]/sizes[] describe them. Pass out=NULL to size the arena first. */
vc_gpu_status vc_gpu_encode(vc_gpu *g, const vc_brick_params *p,
                            const uint8_t *src, size_t src_stride,
                            uint32_t n, uint32_t dim,
                            uint8_t *out, size_t out_cap,
                            size_t offsets[], size_t sizes[], size_t *out_used);
#endif
```

Deltas from today and why:

- **No `shader_dir`** — SPIR-V embedded (§3.6).
- **Status enum, not `-1`** (§3.4); `first_bad` enables per-brick CPU fallback
  instead of failing a whole batch.
- **One caller arena for encode output** instead of `uint8_t ***streams` +
  `size_t **sizes` with n+2 mallocs the caller must free individually
  (codec.h:33-35, codec.c:462-482). Fewer allocations, cache-friendlier, and no
  ownership puzzle.
- **Internal batch splitting** replaces the `maxStorageBufferRange` cliff
  (codec.c:205-211), with the limit exposed via `vc_gpu_query` for callers who
  want to size their own pipelining.
- **`vc_gpu_encode_supported`** makes the profile explicit rather than
  discovered by failure.
- Deliberately **not** in v1: async/streaming handles and device-resident
  output. Both are real (BACKLOG.md:44) but should be added as a second,
  clearly-separated surface (`vc_gpu_submit`/`vc_gpu_wait` + an exported
  `VkBuffer` accessor behind a `#ifdef VC_GPU_VULKAN_INTEROP` header) once a
  consumer exists — not baked into the simple synchronous calls.
- `host_entropy.h` becomes fully internal (`gpu/stream_parse.h`), `vk.h` becomes
  fully internal, and neither is installed.

### 5.1 Build / ship recommendation

- **Optional component**, off by default: keep `option(VC_GPU ...)`, but produce
  a *separate* target `vc_gpu` that links `vc` + `Vulkan::Vulkan`, and make sure
  the core library has **zero** GPU sources (today `CMakeLists.txt:94` compiles
  `src/gpu/host_entropy.c` into `libc5d` — fix by moving the test-only reference
  decoder into `tests/` per §1.3).
- **Embed SPIR-V**: `glslc -O --target-env=vulkan1.1` at build time
  (gpu.cmake:33 already does this), then generate a `.h` of
  `static const uint32_t vc_spv_<name>[]` and compile it into `vc_gpu`. Drops
  `C5D_GPU_SPV_DIR` / `C5D_SPV_DIR`, `read_file` (vk.c:352), and the whole class
  of stale-shader bugs. Ship the `.comp` sources for reference only.
- **Generate shared GLSL headers** (`format.glsl`, `dct_tables.glsl`) from
  `format_internal.h` / `dct_tables.h` at build time so the ~20 hand-mirrored
  constants in §2.5 stop being hand-mirrored. `glslc -I` handles `#include`.
- **glslc discovery**: `find_program` currently `FATAL_ERROR`s when absent
  (gpu.cmake:18-20). With embedded SPIR-V, also support a checked-in
  pre-generated SPIR-V header so `VC_GPU=ON` builds without a Vulkan SDK
  (regenerate only when a `.comp` changes, gated on `VC_GPU_REBUILD_SHADERS`).
- **Register the GPU test with ctest** with a `gpu` label and a graceful skip
  (exit 77) when no device is present — today nothing runs it
  (gpu.cmake has no `add_test`), which is why an on-device-only correctness
  contract has no CI teeth.
- Split the one 1702-line binary into `vc-gpu-test` (ctest-registered, public
  API only) and `vc-gpu-bench` (throughput legs, CLI flags not env vars).
- Add a `VkPipelineCache` persisted to a user cache dir (§3.1).

---

## 6. Prioritized action list

1. Fix the odd-`chunks_per_sub` word-RMW race by 4-aligning
   `token_stride`/`rans_stride` (codec.c:364, encode_tokenize.comp:42,
   encode_rans.comp:22) — correctness, one line.
2. Check `status` before writing `dst` in decode (codec.c:312-317) — correctness.
3. Move the batched/sliced dispatch + record-once + fence machinery from
   `gputest.c` into `codec.c`; per-brick params via SSBO; barriers per stage not
   per brick (codec.c:271-307) — the library then matches the published numbers.
4. Persistent buffer arena + descriptor sets in `c5d_gpu_codec`; kill 26
   allocate/free pairs per call (codec.c:217-224, :379-387) — latency.
5. HOST_CACHED readback on integrated decode (codec.c:217) and device-local
   encode output on discrete (codec.c:387) — measured 2x each.
6. Compact nonzero list out of `forward_quant.comp` to kill the 4096-position
   scatter scan in `encode_tokenize.comp:118` — the 1.58 GB/s stage.
7. Embed SPIR-V; generate `format.glsl` + `dct_tables.glsl`; delete
   `C5D_GPU_SPV_DIR`.
8. Delete `he_decode` + the dense shader branch + `vk_fill_range` + the
   token-sort measurement residue; relocate the reference decoder to `tests/`;
   take `host_entropy.c` out of `libc5d`.
9. Reshape the public header per §5; register a real ctest.
10. Then revisit: subgroup intrinsics, specialization-constant workgroup sizes,
    division-free rANS, one-workgroup-per-substream assembly.


<!-- END 02-gpu.md -->


---

<!-- BEGIN 03-api-architecture.md -->

# 03 — Cross-cutting architecture and public API surface (c5d → volume-compressor)

Scope: every public header in `/home/forrest/c5d/src`, the container/cache/façade
implementations, the thread pool, and the build system. All citations are
`path:line` in the READ-ONLY c5d tree.

---

## 0. Executive framing

c5d today ships **two parallel public APIs over the same codec**:

* the "native" API — `src/brick.h`, `src/label.h`, `src/shard.h`, `src/cache.h`,
  `src/tifxyz.h`, `src/c5d.h`, `src/gpu/codec.h` — raw `int` returns, `malloc`'d
  `**out`, params structs full of format internals;
* the "stable ABI" façade — `src/stable.h` — enum returns, custom allocators,
  cancel/progress callbacks, `_v1` suffixes, and a *second* copy of the label
  type enum and channel struct (`stable.h:42-51` vs `label.h:27-36`;
  `stable.h:72-76` vs `label.h:41-47`).

Only `src/stable.h` is installed (`CMakeLists.txt:115`); the native headers are
exported only via `target_include_directories(c5d PUBLIC src)`
(`CMakeLists.txt:97`), which also leaks `src/format_internal.h`,
`src/entropy/rans.h`, `src/transform/dct_tables.h` onto every consumer's include
path. So the *de facto* public surface is "whatever a downstream `#include`s",
which is the whole tree.

**Recommendation: exactly one public header for the new repo.** Neither existing
layer as-is: take the façade's *conventions* (enum status, explicit ownership,
no format internals) and the native layer's *shape* (small param struct,
caller-owned output buffers, no allocator indirection). Details in §5.

---

## 1. API surface table

Caller columns come from `grep -rIl` over `tools/ tests/ fuzz/ src/gpu bench/`,
excluding the defining `.c`.

### 1.1 `src/c5d.h` — "core" header

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `C5D_VERSION_MAJOR/MINOR/PATCH` | c5d.h:9-11 | none | DROP — duplicated by the string at `common/c5d_core.c:3`, and by CMake `project(... VERSION 0.0.1)` (CMakeLists.txt:2). Three sources of truth. |
| `C5D_CHUNK_DIM/BRICK_DIM/SHARD_DIM` + derived | c5d.h:14-20 | none | KEEP two (`CHUNK_DIM`, and a `BRICK_DIM` *default*), DROP `SHARD_DIM`/`BRICKS_PER_SHARD` with the shard layer (§3). Note the header claims a "Fixed spatial hierarchy" but the codec accepts any `dim%16==0` (spec/format.md:9, brick.c parse), so `C5D_BRICK_DIM` is a default, not a constant. Misleading as written. |
| `c5d_status` enum | c5d.h:22-29 | `tests/test_quick.c:49` only | MERGE — this is the *third* error vocabulary (see §2.1) and is used by nothing. Its values become the one public enum. |
| `c5d_version_string` | c5d.h:31 | nobody (defined `common/c5d_core.c:3`) | KEEP, RENAME. One entry point, hardcoded string. |
| `c5d_status_string` | c5d.h:32 | `tests/test_quick.c:49` | KEEP, RENAME. |

`src/c5d.h` is included by exactly two files in the whole repo
(`src/common/c5d_core.c:1`, `tests/test_quick.c:3`). It is a vestigial header.

### 1.2 `src/brick.h` — the actual codec

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `C5D_BRICK_NSUB`, `..._NSUB_MAX` | brick.h:13-17 | internal | DROP from public header — substream count is a format-internal encoder knob. |
| `c5d_brick_params` (17 fields) | brick.h:19-38 | everywhere | MERGE/SHRINK. See §2.5: `hf_exp`, `dc_fine`, `dz_q`, `dz_dq`, `eprior`, `ctx2`, `nsub`, `rans_nway`, `chunk_qmap` are *bitstream-tuning internals* published as API. |
| `c5d_brick_defaults(q)` | brick.h:40 | 13 files incl. `src/stable.c:273` | KEEP (renamed). Defaults live at `brick.c:140-153`. |
| `c5d_brick_token_counts` | brick.h:44 | `tools/train_priors.c`, `tests/test_hostile.c` | DROP from public API — it is a priors-training hook. Move to a private/internal header used only by the trainer. |
| `c5d_brick_encode` | brick.h:47 | 8 files | KEEP. |
| `c5d_brick_encode_levels` | brick.h:52 | `tests/test_xdec.c`, `test_hostile.c`, `src/gpu/gputest.c` | DROP from public API — GPU-front-half plumbing; belongs behind the GPU module's own internal header. No non-GPU consumer. |
| `c5d_brick_encode_target` | brick.h:56 | `tools/c5dc/pack.c`, `tests/test_hostile.c` | MERGE into encode via a `target_ratio` field in params (bisection is policy, not a second codec). |
| `c5d_brick_encode_u16` / `_decode_u16` | brick.h:64-66 | `tests/test_quick.c`, `src/stable.c` | MERGE — make sample type a params/`dtype` field rather than a parallel function family; see §5. Note `decode_u16` takes **no** `nthreads` while `decode_par` does — asymmetry propagates into the façade (`stable.h:126-129` has no threads param). |
| `c5d_brick_decode` | brick.h:68 | 8 files | MERGE with `_decode_par`. It is literally `return c5d_brick_decode_par(..., 1);` (`brick.c:1742-1744`). Pure redundancy; two names, one behaviour, and the header comment at brick.h:67 documents an `nthreads` argument the function does not have. |
| `c5d_brick_decode_par` | brick.h:69 | `tools/c5dc/pack.c`, `remote.c`, `bakeoff/codecs.c`, `prof/driver.c`, `src/stable.c:304` | KEEP as the single `decode`. |
| `c5d_brick_decode_chunk` | brick.h:73 | `fuzz/d_brick.c`, `fuzz/rt_brick.c` only | KEEP but justify: it is the random-access selling point (spec/format.md:36-41) yet **no tool or test outside the fuzzers uses it**. Either wire it into the CLI or it is dead weight. |
| `c5d_brick_deblock_pair` | brick.h:79 | `tests/test_quick.c` only | KEEP — required for correctness when a consumer decodes neighbouring bricks independently; but it must be documented as part of the *decode contract*, not an optional extra. |

### 1.3 `src/label.h` — label/segmentation codec

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `c5d_label_type` | label.h:27-36 | wide | KEEP, but this enum is duplicated verbatim as `c5d_stable_label_type` (stable.h:42-51) and cast between them (`stable.c:112`, `stable.c:229`) — a C-cast between two independently-declared enums is exactly the fragility the "stable ABI" was supposed to prevent. |
| `C5D_LABEL_NO_MASK` / `MAX_CHANNELS` | label.h:38-39 | wide | KEEP; also duplicated at stable.h:27-28. |
| `c5d_label_channel` | label.h:41-47 | wide | KEEP; duplicated at stable.h:72-76. |
| `c5d_label_params` | label.h:49-53 | wide | SHRINK — `nsub`, `rans_nway` are format internals. |
| `c5d_label_defaults` | label.h:55 | 3 files | KEEP. Note it defaults `nthreads = 0` (label.c:630) while `c5d_brick_defaults` defaults `nthreads = 1` (brick.c:153) — opposite meanings of "default" in two sibling APIs. |
| `c5d_label_encode` / `_decode` | label.h:57,74 | wide | KEEP. |
| `c5d_label_info` | label.h:63 | wide | KEEP. |
| `c5d_label_palette` | label.h:69 | `c5dc/label_cmd.c`, `test_label.c`, `fuzz/d_label.c` | KEEP — genuinely useful metadata-without-decode. |
| `c5d_label_decode_chunk` | label.h:80 | `test_label.c`, `fuzz/d_label.c` | KEEP (symmetry with brick chunk decode). |

### 1.4 `src/shard.h` — container

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `C5D_SHARD_MAGIC` | shard.h:11 | shard.c | **BUG in doc**: `0x31534335` byte-serialises to `"5CS1"`, not `"C5S1"` as the comment claims. `C5DF_MAGIC` (format_internal.h:18) and `LB_MAGIC` (label.c:12) and `TX_MAGIC` (tifxyz.c:348) *are* correct; shard and `SNAP_MAGIC` (shard.c:56, admitted at shard.c:183) are byte-reversed. Fix in the new format. |
| `C5D_SHARD_OFFSET_MISSING/ZERO` | shard.h:12-13 | shard.c, stable.c | KEEP if shard is kept; sentinel-in-offset is fine but should be a flags byte in the entry, not `UINT64_MAX-1` magic. |
| `c5d_shard_entry`, `c5d_shard_footer` | shard.h:15-28 | internal + stable.c:529 | DROP from public header — these are on-disk layout. `c5d_shard_footer` even carries `_pad` (shard.h:27) and an "informational" float `q` (shard.h:26) that `stable.c:529-535` then *depends on* for `brick_dim`. |
| `c5d_shard_create/put/put_zero/close` | shard.h:32-38 | `c5dc/pack.c`, `test_hostile.c`, `stable.c` | See §3 — split out. |
| `c5d_shard_snapshot_now` | shard.h:37 | `tests/test_hostile.c` only; source comment says "test hook" (shard.c:139) | DROP from public API. A test hook in a shipped header. |
| `c5d_shard_reader` **struct, fully exposed** | shard.h:44-49 | embedded *by value* in `struct c5d_stable_shard_reader` (stable.c:26) | DROP/OPAQUE. This is the single worst ABI smell: the "stable ABI" wrapper embeds a public, non-opaque, mmap-holding struct by value, so any change to the container layout breaks the shared library's ABI despite the `_v1` suffixes. |
| `c5d_shard_open/brick/is_zero/is_missing/close_reader` | shard.h:51-58 | pack.c, test_hostile.c, stable.c | See §3. |
| `c5d_crc32c`, `c5d_crc32c_cont` | shard.h:60-61 | `tools/c5dc/remote.c` (crc only) | DROP from the codec's public API — a checksum utility exported from a compressor header is scope creep; keep it internal to the container layer. |

### 1.5 `src/cache.h` — decoded-brick cache

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `c5d_cache*` (create/destroy/get/release/put/stats_get) | cache.h:19-29 | **`tests/test_quick.c` only** | DROP from this repo (§3). Zero production callers: not used by `c5dc`, not by the GPU path, not by `stable.h`, not by the bakeoff. PLAN.md:116-124 calls it a "first-class deliverable"; the code says otherwise. |

### 1.6 `src/tifxyz.h` — surface parameterisation

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `c5d_tifxyz` struct | tifxyz.h:25-30 | tests, c5dc | Not a volume compressor. This is a **TIFF reader/writer plus a 2.5D float-surface codec** — different data model (2D `w*h` float planes, not `dim^3` voxels), different magic (`TFX1`), single-threaded (no `nthreads` anywhere in tifxyz.c). |
| `c5d_tiff_read_f32` / `_write_f32` | tifxyz.h:35-36 | `tests/test_tifxyz.c` | DROP from public API — a bespoke partial TIFF implementation (tifxyz.c:143-212 handles classic+BigTIFF, LZW, predictors, strips/tiles) is a large hostile-input attack surface exported as a public utility. |
| `c5d_tifxyz_load_dir/save_dir/free/encode/decode` | tifxyz.h:38-43 | `c5dc/tifxyz_cmd.c`, `test_tifxyz.c` | SPLIT to a separate repo/optional module. Recommendation: **out of scope for "volume-compressor"**. |

### 1.7 `src/codec_v0.h`, `src/codec_wav.h` — lab baselines

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `c5d_v0_params`, `c5d_v0_defaults`, `c5d_v0_encode/decode` | codec_v0.h:10-24 | `tools/bakeoff/codecs.c` only | **DROP entirely.** Header self-describes as "M1 lab baseline… format NOT frozen" (codec_v0.h:3). 392 lines of a superseded codec compiled into the shipped library (CMakeLists.txt:89). |
| `c5d_wav_encode/decode` | codec_wav.h:9-10 | `tools/bakeoff/codecs.c` only | **DROP entirely.** "M1 challenger" (codec_wav.h:1). 443 lines. |

Together ~835 lines of dead codec in `libc5d.a` whose only consumer is the
bake-off harness. If the comparisons still matter, they belong in `bench/` as a
separate target, not in the library.

### 1.8 `src/gpu/codec.h`

| Symbol | Line | Callers | Verdict |
|---|---|---|---|
| `c5d_gpu_codec_create/destroy` | gpu/codec.h:15-16 | `src/gpu/gputest.c` only | KEEP as an *optional* module, but note the entire Vulkan path's only consumer in-tree is its own test binary (gpu.cmake:52-55). It is not reachable from `c5dc`, `stable.h`, or any test in `ctest`. |
| `c5d_gpu_codec_device_name`, `_subgroup_size` | gpu/codec.h:18-19 | gputest | DROP or MERGE into one `info` struct. |
| `c5d_gpu_decode_batch` / `_encode_batch` | gpu/codec.h:24,33 | gputest | KEEP (module-gated). |
| **`#include "brick.h"` at gpu/codec.h:7** | | | Design smell: the GPU public header pulls in the CPU codec's full params struct, so the GPU API cannot exist without the CPU API's ABI. |

Ownership smell: `c5d_gpu_encode_batch` returns `uint8_t ***streams` and
`size_t **stream_sizes`, "malloc-owned by the caller" (gpu/codec.h:30-31) —
a triple pointer whose free-loop the caller must open-code correctly. No
matching `c5d_gpu_free_batch`.

### 1.9 `src/stable.h` — the façade

All 18 entry points (stable.h:87-160) are called only by `tests/test_stable.c`.
No tool, no fuzzer, no GPU code uses the façade. It exists for hypothetical
ctypes consumers.

| Group | Lines | Verdict |
|---|---|---|
| `c5d_stable_abi_version`, `_status_string` | 87-88 | MERGE into the single public API. |
| `c5d_stable_label_encode/decode/info_v1`, `buffer_release_v1` | 92-109 | MERGE — these are `c5d_label_*` with an allocator and a memcpy. |
| `c5d_stable_brick_encode/decode_u8/u16_v1` (4 fns) | 114-129 | MERGE — these are `c5d_brick_*` with an allocator and a memcpy. |
| `c5d_stable_shard_writer_*_v1` (5 fns) | 133-144 | Container layer (§3). The atomic-publish (mkstemp+link+unlink, stable.c:429-506) is the *only* real functionality the façade adds over `shard.h` — it should live in the container layer itself, not in an ABI wrapper. |
| `c5d_stable_shard_reader_*_v1` (5 fns) | 146-160 | Container layer; each is `brick_index` + `shard_brick` + the corresponding decode (stable.c:547-618). Pure glue, ~70 lines to save the caller three calls. |

**Redundancy verdict:** the façade adds three things — (a) an enum instead of
`-1`, (b) a custom allocator, (c) cancel/progress callbacks — at the cost of a
duplicated type system, a full extra copy of every buffer, and 619 lines. In a
fresh repo with no back-compat obligation, fold (a) into the one API, make (b)
optional-and-global (§2.2), and reconsider (c) (§2.6). Delete the layer.

---

## 2. Design smells

### 2.1 Three error vocabularies + raw ints
1. `c5d_status` (c5d.h:22-29): 6 values, **used by one assert** (test_quick.c:49).
2. `c5d_stable_status` (stable.h:30-40): 9 values, used only inside `stable.c`.
3. Raw `int`: every real codec entry point. `brick.c` has 60 `return -1` sites
   (all indistinguishable); `label.c`, `tifxyz.c`, `shard.c` the same.
4. `c5d_cache_put` invents a *fourth* convention: `0` / `-1` alloc-or-size /
   `-2` backpressure (cache.h:25-28).
5. `c5d_shard_create` returns `NULL` on error (shard.h:32); `c5d_shard_open`
   returns `int` and fills a caller struct (shard.h:51) — two conventions inside
   one header.

Every failure in `brick.c` collapses to `-1`, so `stable.c` has to *guess*:
encode failure → `E_INTERNAL` (stable.c:280) even when the cause was a bad
argument; decode failure → `E_CORRUPT` (stable.c:305) even when it was OOM. The
façade's richer enum is therefore **fiction** — the information was destroyed
one layer down. Fix at the source: one enum, returned by the codec itself.

### 2.2 Ownership: four different models in one library
* **malloc'd `**out`**: `c5d_brick_encode(..., uint8_t **out, size_t *out_n)`
  (brick.h:47), `c5d_label_encode` (label.h:57), `c5d_tifxyz_encode`
  (tifxyz.h:42). Freed with plain `free()`; no library-provided free function,
  so a Windows/CRT-mismatched or allocator-overriding consumer cannot free it.
* **Caller buffers**: `c5d_brick_decode(..., uint8_t *dst, ...)` (brick.h:68),
  `c5d_label_decode` (label.h:74). Good — keep this.
* **Custom allocator + owning buffer struct**: `c5d_stable_buffer` carries its
  own allocator (stable.h:78-82) and must be freed via
  `c5d_stable_buffer_release_v1` (stable.h:109).
* **Library-owned pointer into an mmap**: `c5d_shard_brick` returns a borrowed
  pointer valid until `close_reader` (shard.h:52-53); `c5d_cache_get` returns a
  *pinned* pointer requiring `c5d_cache_release` (cache.h:22-24).

The custom-allocator model also costs real throughput: every façade encode
allocates via `malloc` inside the codec, then **copies the whole stream** into
allocator memory and frees the original (stable.c:193-199 and the shared helper
stable.c:244-261). Every façade decode allocates a full `dim^3` scratch buffer,
decodes into it, then `memcpy`s to the caller (stable.c:300-312 for u8,
stable.c:362-374 for u16) — purely to honour "outputs untouched on failure"
(stable.h:98-99). On a 2 MiB brick at the README's claimed 3.02 GB/s decode
(README.md:28), that is an extra full-volume read+write per brick on the hot
path. The label decode is worse: it allocates and copies **per channel**
(stable.c:111-138).

**Recommendation:** one model. Caller-provided output buffers for decode;
for encode, either (a) caller-provided buffer + a `bound()` function returning
the worst case, or (b) malloc'd out plus an explicit `vc_free()`. Prefer (a) —
it removes the allocator plumbing entirely and lets a Python/ctypes caller hand
in a NumPy array.

### 2.3 Thread parameters: five inconsistent forms
* `c5d_brick_params.nthreads` (brick.h:37) — in the struct, encode-side.
* `c5d_brick_decode_par(..., unsigned nthreads)` (brick.h:69) — trailing arg.
* `c5d_brick_decode(...)` (brick.h:68) — no arg, hardcoded 1.
* `c5d_brick_decode_u16(...)` (brick.h:66) — no arg at all, so the u16 path is
  permanently single-threaded and the façade's u16 decode has no threads
  parameter either (stable.h:126-129).
* `c5d_brick_deblock_pair(..., unsigned nthreads)` (brick.h:80) — trailing arg.
* `c5d_label_decode(..., unsigned nthreads)` (label.h:75) vs
  `c5d_label_params.nthreads` for encode (label.h:52).
* `c5d_label_decode_chunk` (label.h:80) — no threads.
* `tifxyz` — no threading anywhere.
* Façade uses `uint32_t threads` (stable.h:94) where native uses `unsigned`.

And the *meaning* of the value differs by call site: `c5d_brick_defaults` sets
`nthreads = 1` (brick.c:153) while `c5d_label_defaults` sets `nthreads = 0`
(label.c:630); `0` means "all CPUs" (pool.h:11).

### 2.4 Global thread pool lifecycle
`src/common/pool.c` holds a process-wide singleton (`g_pool`, pool.c:29-33)
initialised under `pthread_once` (pool.c:115) that spawns `ncpu-1` **detached,
never-terminating** worker threads (pool.c:98-102) on the first parallel call.
Consequences for a shipped library:

* **No shutdown.** `pool_worker` is `for(;;)` with no exit path (pool.c:79-91).
  A `dlclose()` of the shared library (`c5d_stable` is `SHARED`,
  CMakeLists.txt:106) while workers are parked in `pthread_cond_wait` on
  code that is about to be unmapped is a crash. This directly contradicts
  stable.h:4 ("suitable for dlopen/ctypes users").
* **No fork safety.** No `pthread_atfork` handler. A Python consumer using
  `multiprocessing` (fork start method, still the default on Linux for many
  stacks) after any encode/decode inherits a mutex that may be held by a thread
  that does not exist in the child → deadlock on the next parallel call.
* **Thread count is not caller-controlled.** `pool_init` spawns
  `available_cpu_count()-1` threads regardless of the `nthreads` the caller
  asked for (pool.c:96-102); `nthreads` only caps *concurrent claims per job*
  (`job.limit`, pool.c:121, pool.c:133). Asking for 2 threads on a 128-core box
  still creates 127 OS threads on first use.
* `c5d_parallel_for` returns `int` but can only ever return `0`
  (pool.c:106,112,150) — a `[[nodiscard]]`-shaped API with no failure mode;
  worker-creation failure is silently swallowed (pool.c:100 `break`).
* `pool.h` is a public-path header (reachable via `target_include_directories(c5d PUBLIC src)`)
  and `c5d_parallel_for` is called directly from `tools/c5dc/pack.c` and
  `tests/test_quick.c`.

**Recommendation:** explicit lifecycle. Either an opaque `vc_pool` created by
the caller and passed in the params struct, or keep a lazy global but add
`vc_shutdown()`, `pthread_atfork` handlers, joinable (not detached) workers, and
size the pool to the largest `nthreads` actually requested.

### 2.5 Header-with-internals leakage
* `c5d_brick_params` publishes nine fields that are bitstream-tuning constants:
  `hf_exp`, `dc_fine`, `dz_q`, `dz_dq` (brick.h:21-25), `eprior` (brick.h:32),
  `nsub` (brick.h:33), `ctx2` (brick.h:34), `rans_nway` (brick.h:31),
  `chunk_qmap` (brick.h:35). Several are documented in terms of *format flags*
  ("format flag 64", brick.h:34) — the public API is describing the bitstream.
  A caller who sets these wrongly produces worse-but-valid streams; a caller who
  sets them at all pins the format.
* `src/format_internal.h` is on every consumer's include path
  (CMakeLists.txt:97) despite its name and its `C5DF_` internal prefix. It is
  correctly *used* only by brick.c, gpu/host_entropy.c, gpu/gputest.c.
* `c5d_shard_reader` (shard.h:44-49) exposes `const uint8_t *map`, `map_n`,
  the on-disk `foot` and `index_off`. `tools/c5dc/pack.c` and `test_hostile.c`
  reach into it; `stable.c:26` embeds it by value.
* `c5d_shard_footer`/`c5d_shard_entry` (shard.h:15-28) are the on-disk records,
  including `_pad`.

### 2.6 Cancellation and progress callbacks
`c5d_stable_callbacks` (stable.h:66-70) is checked *between phases only* —
before starting (stable.c:106, 177, 272), and after the whole decode returns
(stable.c:132, 306). Progress is reported as `(0,1)` then `(1,1)`
(stable.c:127/139, 278/286, 302/310). So on a real 128³ brick these callbacks
convey **no information and cannot actually cancel anything** — a cancelled
decode still runs to completion, then discards the result. It is API surface
that pays for itself in neither latency nor usability. Either wire cancellation
into `c5d_parallel_for` at task granularity, or drop it. For 2 MiB bricks
decoding in sub-millisecond time, drop it; cancellation belongs at the
*batch/shard* level in the caller's own loop.

### 2.7 The `_v1` "stable ABI" versioning scheme
`C5D_STABLE_ABI_VERSION 1u` (stable.h:26) + `_v1` on 16 of 18 functions.

Arguments against carrying it into the new repo:
* **It does not achieve ABI stability.** `c5d_stable_shard_reader` embeds the
  non-opaque `c5d_shard_reader` by value (stable.c:26); the label enum is
  C-cast across two independent declarations (stable.c:112, 229). A layout
  change in `shard.h` silently changes the shared library's behaviour.
* **The format is explicitly unstable** — c5d.h:2, spec title "v1.5",
  README.md:10. A stable *ABI* over an unstable *format* is the wrong axis of
  stability: a ctypes caller's real risk is that last month's `.c5s` no longer
  decodes, not that a struct grew a field.
* **Symbol-level versioning is the most expensive way to do this.** The cheap
  alternatives are a `size_t struct_size` first field in the params struct
  (extensible without renaming), plus a runtime `vc_abi_version()`, plus real
  ELF symbol versioning (a version script) if it ever matters.
* Every `_v1` name costs 3 characters at every call site forever, and creates
  the "when do I add `_v2`?" question the repo will never resolve.

**Recommendation for a fresh repo with no back-compat requirement: drop `_v1`.**
Keep `vc_abi_version()` and a `struct_size`-prefixed params struct. Add a
version script and `-fvisibility=hidden` (already done at CMakeLists.txt:110)
so the export set is explicit. Revisit named symbol versions only after 1.0.

### 2.8 Magic numbers and constants
* Shard magic comment is wrong: `C5D_SHARD_MAGIC 0x31534335 /* "C5S1" */`
  (shard.h:11) actually serialises `"5CS1"`; `SNAP_MAGIC 0x52534335 /* "C5SR" */`
  (shard.c:56) serialises `"5CSR"` — the recovery scanner has to hardcode the
  reversed bytes with an apologetic comment (shard.c:183). `C5DF_MAGIC`
  (format_internal.h:18), `LB_MAGIC` (label.c:12) and `TX_MAGIC` (tifxyz.c:348)
  are byte-order-correct. Pick one convention (byte array, not `u32` literal).
* `NSHARDS 16`, `GHOSTS 1024` (cache.c:7-8), `nbuckets = 4096` (cache.c:64),
  `guard = 4096` (cache.c:110), `SNAP_INTERVAL_BYTES 64<<20` (shard.c:57),
  `nbricks <= (1u<<20)` (shard.c:177, repeated shard.c:202), `256` thread cap
  (pool.c:97, 109), `path_size + 32u` for the temp suffix (stable.c:418) — all
  unexplained literals, several duplicated at two sites.
* `ghost_has` is an O(1024) linear scan per `c5d_cache_put` (cache.c:85-89) — a
  1024-element sequential search on every insert.

### 2.9 Other correctness/design notes found while reading
* `c5d_shard_open` validates the *snapshot* index with a CRC (shard.c:207-211)
  but trusts the normal end-of-file footer + index with **no CRC**
  (shard.c:176-179). Per-brick CRCs (shard.c:228) cover payloads, not the index,
  so a corrupt index can only be caught by luck. Asymmetric trust between the
  common path and the recovery path.
* `c5d_stable_shard_writer_finish_v1` returns `C5D_STABLE_E_IO` **after the
  destination is fully published** if the temp `unlink` fails (stable.c:502-506),
  with a comment acknowledging the state is fine. Callers will treat a success
  as a failure.
* `writer_free` calls `c5d_shard_close` and discards its return (stable.c:387),
  so abort-path I/O errors vanish.
* `c5d_shard_close` frees the writer even when the final write fails
  (shard.c:146-157) — no way to retry.
* `c5d_cache_stats_get` casts away `const` to lock (cache.c:107) — the `const`
  in the signature (cache.h:29) is a lie.
* `c5d_shard_close_reader` has a triple cast to strip `const` from `map`
  (shard.c:248) — a symptom of the struct being public and `map` being `const`.

---

## 3. Are shard and cache in scope?

**Intended consumers** (PLAN.md:21-26, README.md:22-24): VC3D / Volume
Cartographer-class tooling, out-of-core renderers, ML pipelines reading random
sub-volumes, and S3 ranged-GET access (`tools/c5dc/remote.c:1-6` demonstrates
footer + index-slice + brick = 3 ranged reads).

### Cache — **DROP.**
* Zero production callers: only `tests/test_quick.c` (§1.5).
* It is a generic `uint64 key → bytes` LRU-ish store (cache.h:12-29). Nothing in
  it is volume- or codec-specific; keys are "caller-defined u64" (cache.h:3).
  It is a general-purpose cache that happens to live in a codec repo.
* The features that would make it codec-specific — parent-LOD fallback, GPU
  texture atlas mirror, tick-phase lock-free reads (PLAN.md:118-124) — are **not
  implemented**. What exists is the generic 20%.
* Every plausible consumer already has a cache: VC3D has its own chunk cache,
  Python users have `functools`/an LRU dict/`zarr` caching, a renderer has a
  texture pool. Shipping ours forces them to adopt our pin/release protocol
  (cache.h:22-24) and our `-2` backpressure convention (cache.h:25-28).
* Cost of keeping: 251 lines, a pthread dependency, a mutex-per-shard threading
  contract, and a public API that is 6 of the library's ~50 entry points.

### Shard — **SPLIT into a second, optional target; do not put it in the core codec library.**
Reasoning:
* It *is* load-bearing for the stated use case — one ranged GET per brick is a
  format-level design goal (PLAN.md:110-113) and the thing `c5dc remote`
  demonstrates. Dropping it entirely would strand the S3 story.
* But it is a **file/object container**, not a compressor: `fopen`, `mmap`,
  `fsync`, crash-recovery snapshots, CRC32C (shard.c). It drags POSIX file I/O,
  `sys/mman.h`, and a crash-consistency protocol into a library whose core job
  is `bytes → voxels`. A Python/ctypes user doing S3 ranged GETs **does not want
  our file reader at all** — they have `boto3`, they want to hand us
  `(index_bytes)` and `(brick_bytes)` and get voxels back.
* The right decomposition:
  - **`libvolcomp`** (core): brick encode/decode, label encode/decode,
    chunk-level random access. No file I/O. No mmap. No CRC. Pure buffers in,
    buffers out. This is what ctypes users bind, what the GPU module extends,
    and what fuzzers target.
  - **`libvolcomp_shard`** (optional, `VC_SHARD=ON`): the container — writer,
    reader, atomic publish (which today only exists in `stable.c:429-506`,
    the one genuinely valuable thing the façade added), crash recovery.
    Depends on core, not the other way round.
  - Crucially, expose a **parse-the-index-from-a-buffer** entry point so an S3
    consumer can do its own ranged GETs and never touch our file reader:
    `vc_shard_index_parse(const void *footer_and_index, size_t n, vc_shard_index *out)`
    and `vc_shard_index_entry(const vc_shard_index *, uint32_t brick, uint64_t *off, uint32_t *n, uint32_t *crc)`.
    `tools/c5dc/remote.c` currently reimplements exactly this by hand against the
    raw footer layout — that hand-rolled code is the evidence the API is missing.
* `c5d_crc32c` stays inside the shard module (it exists only to serve the index
  and per-brick checks, shard.c:27-54).

**tifxyz — DROP from this repo** (§1.6): different data model, different magic,
its own TIFF parser. It belongs in the Vesuvius segment tooling, not in a volume
compressor. Same for `codec_v0`/`codec_wav` (bench-only) and
`c5d_brick_token_counts` (trainer-only).

---

## 4. Build system

### 4.1 Cruft in the current CMake
| Item | Line | Verdict |
|---|---|---|
| `c5d-bakeoff` target (5 sources) + zstd + zfp detection | CMakeLists.txt:118-137 | MOVE to `bench/` behind `VC_BENCH=ON`, default OFF. It pulls two optional external deps into the default configure. |
| `C5D_ZFP_ROOT` cache path | CMakeLists.txt:131 | DROP — a hardcoded per-developer prefix for one baseline. |
| `c5d-label-bench` | CMakeLists.txt:140-145 | MERGE into the bench target. |
| `c5d-train-priors` | CMakeLists.txt:148-149 | Keep, but `VC_TOOLS=ON`, default OFF — it is a one-shot format-maintenance tool. |
| `c5dc` (5 sources: main/pack/remote/tifxyz_cmd/label_cmd) | CMakeLists.txt:152-154 | SHRINK to one CLI: `pack`, `unpack`, `info`. Drop `remote` (it shells out to the `curl` *binary* via fork/execvp, `tools/c5dc/remote.c:33-45` — a demo, not a shippable feature) and `tifxyz_cmd` (§3). |
| `c5d_stable` SHARED target + install | CMakeLists.txt:106-115 | The core library should *be* the shared library. Do not ship two libraries where the static one is the real code and the shared one is a 619-line copy layer. |
| 7 test executables, all `LABELS "quick"` | CMakeLists.txt:170-201 | The label is meaningless if every test has it; `ctest --preset quick` (README.md:67) runs everything. Either add a `slow` tier or drop the label. |
| `C5D_FUZZ` recompiling the *library* with fuzzer flags | CMakeLists.txt:156-167 | Works, but `target_link_options(c5d PUBLIC -fsanitize=...)` at line 160 leaks sanitizer link flags to every consumer of `c5d` in that configure. Prefer a separate `vc_fuzz` object library. |
| `msan` preset | CMakePresets.json:52-60 | MSan needs an instrumented libc++/libc to be meaningful; without one it produces false positives. Keep only if the toolchain is actually set up; otherwise cruft. |
| `bench` and `gpu` presets both set `C5D_MCPU=native` + hardening OFF | CMakePresets.json:38-40, 68-71 | Duplication; `gpu` should inherit `bench`. |
| `CMAKE_C_COMPILER_LAUNCHER: ccache` in the *base* preset | CMakePresets.json:10 | Hard-fails configure on a machine without ccache. Should be conditional. |
| `add_compile_definitions(_GNU_SOURCE)` globally | CMakeLists.txt:10 | Fine on Linux, but combined with `CMAKE_C_EXTENSIONS OFF` (line 8) it is contradictory intent. |
| `message(WARNING)` on non-clang | CMakeLists.txt:13-15 | Keep — honest. |

### 4.2 Worth keeping
* `-Wall -Wextra -Wconversion -Wshadow -Wimplicit-fallthrough -Wformat=2 -Wvla -Wdouble-promotion`
  + `-Werror` (CMakeLists.txt:22-25). `-Wconversion` and `-Wdouble-promotion` in
  particular are load-bearing for a codec.
* Sanitizer interface library with a validated enum (CMakeLists.txt:18-42),
  including the `FATAL_ERROR` on an unknown value (line 41).
* Hardening set: `-D_FORTIFY_SOURCE=3 -fstack-protector-strong
  -fstack-clash-protection -fstrict-flex-arrays=3` and
  `-Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack` (CMakeLists.txt:51-55), correctly
  gated off under sanitizers and non-Release. Keep verbatim.
* `C_VISIBILITY_PRESET hidden` (CMakeLists.txt:110) — apply to the *core* library.
* `-ffast-math -fno-math-errno` scoped `PRIVATE` to the codec only
  (CMakeLists.txt:98), with the bakeoff deliberately excluded (comment at
  CMakeLists.txt:117). Good discipline; preserve it, and keep it off any code
  that parses a bitstream (the reason for the integer-domain float validation at
  brick.c:32-38).
* `CMAKE_EXPORT_COMPILE_COMMANDS` (line 9), `POSITION_INDEPENDENT_CODE` (line 11).
* `.clang-tidy` with `bugprone-*`/`clang-analyzer-*`/`performance-*` and
  narrowing-conversions as errors.

### 4.3 Minimal CMake for the new repo
```
volume-compressor/
  CMakeLists.txt          # ~90 lines
  targets:
    volcomp        SHARED (default) — the codec. src/*.c, no I/O.
    volcomp_shard  optional, VC_SHARD=ON  (default ON on POSIX)
    volcomp_gpu    optional, VC_GPU=ON    (default OFF; Vulkan + glslc)
    vc             one CLI: pack / unpack / info
    tests          VC_TESTS=ON (default ON when top-level project)
  options:
    VC_SHARD, VC_GPU, VC_TESTS, VC_BENCH, VC_FUZZ,
    VC_WERROR (ON), VC_HARDENING (ON), VC_SANITIZE ("" | asan-ubsan | tsan),
    VC_NATIVE (OFF)   # replaces the C5D_MARCH/C5D_MCPU pair
  presets: dev (asan-ubsan) | release | bench (native, hardening off) | gpu
  install: libvolcomp + include/volcomp/volcomp.h + a CMake package config
           (volcompConfig.cmake) + volcomp.pc  — currently neither exists
```
Notable simplifications: one ISA option instead of the `C5D_MARCH`/`C5D_MCPU`
pair with its x86 remapping (CMakeLists.txt:66-80); drop the `msan` preset
unless the toolchain supports it; drop `C5D_BRANCH_PROTECTION` (AArch64-specific,
default OFF, never exercised — CMakeLists.txt:48, 57-64) or fold it into
`VC_HARDENING` on aarch64. Add an `install(EXPORT)` — the current install
(CMakeLists.txt:113-115) ships a `.so` and one header with no CMake package and
no pkg-config, so downstream integration is manual.

---

## 5. Proposed public header

One header, `include/volcomp/volcomp.h`. Optional second header
`include/volcomp/volcomp_shard.h` only if `VC_SHARD=ON`; optional
`volcomp_gpu.h` only if `VC_GPU=ON`. Nothing else installed.

```c
/* volcomp — 3D volume compression. One header, one library. */
#ifndef VOLCOMP_H
#define VOLCOMP_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define VC_API __declspec(dllimport)   /* dllexport when building */
#else
#  define VC_API __attribute__((visibility("default")))
#endif

/* ---- versioning -------------------------------------------------------- */
#define VC_ABI_VERSION    1u   /* bumped only on a source-incompatible change */
#define VC_FORMAT_VERSION 1u   /* bitstream revision; see spec/format.md      */
VC_API uint32_t     vc_abi_version(void);       /* runtime check for ctypes  */
VC_API const char  *vc_version_string(void);    /* "volcomp 1.0.0 (fmt 1)"   */

/* ---- status ------------------------------------------------------------ */
typedef enum vc_status {
  VC_OK = 0,
  VC_ERR_ARG,       /* caller passed something invalid                       */
  VC_ERR_CORRUPT,   /* malformed or hostile bitstream                        */
  VC_ERR_NOMEM,     /* allocation failed                                     */
  VC_ERR_FORMAT,    /* well-formed but from an incompatible format version   */
  VC_ERR_SHORT_BUF, /* caller's output buffer too small (see *needed)        */
  VC_ERR_IO,        /* only ever returned by volcomp_shard.h                 */
} vc_status;
VC_API const char *vc_status_string(vc_status s);

/* ---- geometry ---------------------------------------------------------- */
#define VC_CHUNK_DIM         16u   /* transform + random-access granularity  */
#define VC_BRICK_DIM_DEFAULT 128u  /* canonical brick edge; any dim%16==0 ok */

/* ---- sample types ------------------------------------------------------ */
typedef enum vc_dtype {
  VC_U8 = 0, VC_U16 = 1, VC_U32 = 2, VC_U64 = 3,
  VC_I8 = 4, VC_I16 = 5, VC_I32 = 6, VC_I64 = 7,
} vc_dtype;

/* ---- encode parameters -------------------------------------------------
 * Zero-initialise and set what you need; vc_params_default() fills the rest.
 * `struct_size` makes this struct extensible without an ABI break: pass
 * sizeof(vc_params); the library reads only the fields it knows.           */
typedef struct vc_params {
  size_t   struct_size;
  vc_dtype dtype;        /* VC_U8 (default) or VC_U16                       */
  float    quality;      /* quantizer step in source sample units; >0       */
  float    tau;          /* 0 = off; else hard max abs error bound          */
  double   target_ratio; /* 0 = off; else bisect `quality` to hit this      */
  bool     lossless;     /* exact; `quality` ignored                        */
  bool     deblock;      /* normative decode-side seam filter (default on)  */
  unsigned threads;      /* 0 = library default, 1 = caller's thread only   */
} vc_params;
VC_API vc_params vc_params_default(void);

/* ---- grayscale volume codec -------------------------------------------- */
/* Upper bound on the encoded size for these params — call once, allocate,
 * encode. Removes every allocator hook and every **out from the API.       */
VC_API size_t vc_encode_bound(const vc_params *p, uint32_t dim);

/* src is dim^3 samples of p->dtype, contiguous ZYX. Writes into caller
 * memory; *out_n receives the byte count.  VC_ERR_SHORT_BUF if dst_cap is
 * below what vc_encode_bound reported.                                     */
VC_API vc_status vc_encode(const vc_params *p, const void *src, uint32_t dim,
                           void *dst, size_t dst_cap, size_t *out_n);

/* Header-only inspection: dim, dtype, lossless/tau flags, encoded quality.
 * Lets a caller size its buffer and pick a dtype without decoding.         */
typedef struct vc_info {
  size_t   struct_size;
  uint32_t dim;
  vc_dtype dtype;
  float    quality;
  bool     lossless;
  bool     has_tau;
} vc_info;
VC_API vc_status vc_info_read(const void *enc, size_t enc_n, vc_info *out);

/* Whole-brick decode into caller memory (dim^3 samples of the stream's
 * dtype). threads: 0 = library default, 1 = this thread only.              */
VC_API vc_status vc_decode(const void *enc, size_t enc_n, uint32_t dim,
                           void *dst, size_t dst_cap, unsigned threads);

/* Single 16^3 chunk decode — touches only the owning substream. This is the
 * random-access primitive the format exists for; ~4 KB out.                */
VC_API vc_status vc_decode_chunk(const void *enc, size_t enc_n, uint32_t dim,
                                 uint32_t cz, uint32_t cy, uint32_t cx,
                                 void *dst4096, size_t dst_cap);

/* Filter the shared face of two independently decoded, adjacent bricks.
 * Required for seam-free output whenever bricks are decoded separately.    */
VC_API vc_status vc_deblock_pair(void *neg, void *pos, uint32_t dim,
                                 uint32_t axis, float quality, unsigned threads);

/* ---- label / segmentation codec (lossless, multi-channel integers) ------ */
#define VC_LABEL_NO_MASK      UINT32_MAX
#define VC_LABEL_MAX_CHANNELS 64u
typedef struct vc_channel {
  vc_dtype type;
  uint32_t mask_channel;  /* index of a LOWER channel, or VC_LABEL_NO_MASK  */
  void    *data;          /* dim^3 samples, ZYX; NULL on decode = skip      */
} vc_channel;

VC_API size_t    vc_label_encode_bound(uint32_t dim, uint32_t nchan, const vc_channel *ch);
VC_API vc_status vc_label_encode(const vc_params *p, const vc_channel *ch, uint32_t nchan,
                                 uint32_t dim, void *dst, size_t dst_cap, size_t *out_n);
VC_API vc_status vc_label_info(const void *enc, size_t enc_n, uint32_t *dim,
                               uint32_t *nchan, vc_channel *ch, uint32_t ch_cap);
/* Distinct values in one channel, from the header, without decoding voxels. */
VC_API vc_status vc_label_palette(const void *enc, size_t enc_n, uint32_t chan,
                                  int64_t *out, size_t cap, size_t *count);
VC_API vc_status vc_label_decode(const void *enc, size_t enc_n, uint32_t dim,
                                 vc_channel *ch, uint32_t nchan, unsigned threads);
VC_API vc_status vc_label_decode_chunk(const void *enc, size_t enc_n, uint32_t dim,
                                       uint32_t cz, uint32_t cy, uint32_t cx,
                                       vc_channel *ch, uint32_t nchan);

/* ---- threading --------------------------------------------------------- */
/* Release the library's worker threads. Optional; call before dlclose() or
 * before fork() in a consumer that forks. Idempotent.                      */
VC_API void vc_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif /* VOLCOMP_H */
```

**One line of justification per entry point**

| Entry point | Why it exists |
|---|---|
| `vc_abi_version` | The only way a ctypes/dlopen caller can detect a mismatched `.so` before crashing. |
| `vc_version_string` | Human-readable identification in logs and bug reports. |
| `vc_status_string` | Turns the enum into a message without every caller writing a switch. |
| `vc_params_default` | Sane quality/deblock/thread defaults so the common call is two lines. |
| `vc_encode_bound` | Lets the caller own the output buffer, which deletes the entire allocator/`**out`/`buffer_release` apparatus. |
| `vc_encode` | The compressor. |
| `vc_info_read` | Size and type a buffer from a stream header without decoding it — needed by every container and every S3 consumer. |
| `vc_decode` | The decompressor. |
| `vc_decode_chunk` | 4 KB random access is the reason for the substream format; without it the brick layout has no payoff. |
| `vc_deblock_pair` | Independently decoded neighbours have visible seams otherwise; part of the decode contract, not a nicety. |
| `vc_label_*` (5) | Label volumes are a distinct data model (integer, lossless, masked, multi-channel) that the grayscale path cannot express; they share the entropy backend, not the API. |
| `vc_shutdown` | Makes the thread-pool lifecycle explicit — the one thing §2.4 says is missing. |

That is **17 entry points** plus 2 optional modules, against roughly 50 today.

**Conventions, stated once in the header and honoured everywhere**
* *Errors*: every function returns `vc_status`; nothing returns `int` or `NULL`
  for failure. `VC_ERR_ARG` vs `VC_ERR_CORRUPT` vs `VC_ERR_NOMEM` is decided at
  the *point of failure*, not guessed by a wrapper (fixes §2.1).
* *Ownership*: the library never allocates anything the caller must free.
  Outputs go into caller memory; `*_bound()` sizes it. Scratch is internal.
  This makes NumPy/ctypes binding trivial and deletes `c5d_stable_allocator`,
  `c5d_stable_buffer`, and all six copy sites in `stable.c`.
* *Failure atomicity*: outputs are unspecified on failure (documented). The
  façade's "untouched on failure" guarantee costs a full extra volume copy per
  call (§2.2) and buys almost nothing — callers discard the buffer anyway.
* *Threading*: one convention. `threads` is `unsigned`; `0` = library default
  (all available CPUs), `1` = run on the calling thread only, `n` = at most `n`.
  It appears in `vc_params` for encode and as the last argument for decode —
  and it appears on *every* multi-threadable entry point, including the u16 and
  chunk paths that lack it today.
* *Coordinates*: voxel buffers are contiguous ZYX (`z*dim*dim + y*dim + x`);
  any brick-address triple is XYZ. Stated once, matching stable.h:6-7.
* *Reentrancy*: all functions are thread-safe and may be called concurrently;
  the only global state is the worker pool, whose lifetime is `vc_shutdown`.
* *Versioning*: no `_v1` suffixes. Compatibility is carried by
  `vc_abi_version()`, `struct_size` on every params/info struct, `-fvisibility=hidden`
  plus an explicit version script, and a documented policy: the format version
  in the stream header is what actually gates decodability, and any format
  change bumps `VC_FORMAT_VERSION` and appears in `spec/format.md`. Pre-1.0 the
  format may change; post-1.0 decoders must read every prior version.

---

## 6. Naming

Repo is `volume-compressor`. Recommendation: **prefix `vc_` / `VC_`**.

| Layer | Scheme | Example |
|---|---|---|
| Public C symbols | `vc_<noun>_<verb>` | `vc_encode`, `vc_label_decode_chunk`, `vc_shard_open` |
| Public macros/enums | `VC_` | `VC_OK`, `VC_CHUNK_DIM`, `VC_U16` |
| Public types | `vc_<noun>` | `vc_params`, `vc_status`, `vc_channel` |
| Internal (non-static, cross-TU) | `vci_` | `vci_rans_decode`, `vci_dct16_fwd` |
| File-static | no prefix | `parse_header`, `tokenize_sub` |
| Library / SONAME | `libvolcomp.so.1` | |
| Installed include dir | `include/volcomp/` | `#include <volcomp/volcomp.h>` |
| CMake target / package | `volcomp`, `volcomp::volcomp` | |
| CMake options | `VC_` | `VC_GPU`, `VC_SANITIZE`, `VC_WERROR` |
| Env vars | `VC_` | `VC_THREADS`, `VC_SPV_DIR` |
| CLI | `vc` with subcommands | `vc pack`, `vc unpack`, `vc info` |

Caveat: `vc_` is short and slightly collision-prone (it also reads as "Vesuvius
Challenge" / "Volume Cartographer", which in this ecosystem is arguably a
feature). If a longer prefix is preferred, `vol_`/`VOL_` is the next best; avoid
`vcomp_` (clashes with MSVC's OpenMP runtime `vcomp`).

**Source layout**
```
include/volcomp/volcomp.h            # the public API (§5)
include/volcomp/volcomp_shard.h      # optional container
include/volcomp/volcomp_gpu.h        # optional Vulkan module
src/api.c                            # the 17 entry points; validation + dispatch
src/brick.c  src/label.c             # codecs
src/entropy/  src/transform/         # kernels (vci_ prefix)
src/format.h                         # NOT installed — the internal format contract
src/pool.c src/pool.h                # threading
src/shard/                           # container module
src/gpu/                             # Vulkan module
```
`src/format.h` replaces `format_internal.h` — the "_internal" suffix is a
symptom of the header being installable in the first place; a header simply not
under `include/` cannot leak.

**Format magic**

Write magics as 4-byte arrays, not `u32` literals, so the byte order in the file
is what the comment says (fixes §2.8). Proposal, all four bytes ASCII and all
sharing a `VC` stem so a hexdump identifies the family instantly:

| Stream | Magic (file bytes) | Replaces |
|---|---|---|
| Grayscale brick | `"VCB1"` | `C5B3` (format_internal.h:18) — the `3` was a revision counter in the magic, which is why it is already at v1.5 with a `3` in the name. Keep the revision in a header field, not the magic. |
| Label brick | `"VCL1"` | `C5L1` (label.c:12) |
| uint16 container | `"VCU1"` | `C5U1` (spec/format.md:85) |
| Shard container | `"VCS1"` | `C5S1` (shard.h:11, actually `5CS1`) |
| Shard snapshot | `"VCSR"` | `C5SR` (shard.c:56, actually `5CSR`) |

Trailing digit = format generation, bumped only on a break; within a generation
the revision lives in a `u16 format_version` header field so a decoder can
report "too new" (`VC_ERR_FORMAT`) instead of "corrupt".

---

## Summary (10 lines)

1. There are two public APIs over one codec — the native headers and the
   `stable.h` façade (619 lines, called only by `tests/test_stable.c`). The new
   repo should ship exactly one; take the façade's conventions and the native
   layer's shape.
2. Three error vocabularies coexist (`c5d_status`, c5d.h:22-29, used by one
   assert; `c5d_stable_status`, stable.h:30-40; raw `-1` at 60 sites in
   brick.c), so the façade's rich enum is guesswork — encode failure becomes
   `E_INTERNAL` (stable.c:280) whatever the cause. Fix at the source.
3. Four ownership models coexist (malloc'd `**out`, caller buffers, custom
   allocator + owning buffer, borrowed mmap/pinned pointers). Standardise on
   caller-owned buffers plus a `*_bound()` sizer; that deletes the allocator
   plumbing and the façade's per-call full-volume memcpy (stable.c:193-199,
   300-312, 362-374) from the hot path.
4. Threading is specified five different ways, and `c5d_brick_decode` is
   literally `decode_par(...,1)` (brick.c:1742) while `decode_u16` cannot thread
   at all (brick.h:66) — that limitation reaches the façade (stable.h:126).
5. The global pool spawns `ncpu-1` detached, never-exiting threads on first use
   (pool.c:96-102) regardless of the requested count, with no shutdown and no
   `pthread_atfork` — directly contradicting stable.h:4's dlopen/ctypes claim.
   Add `vc_shutdown()` and fork handlers.
6. `c5d_brick_params` publishes nine bitstream-tuning internals (brick.h:21-35,
   one documented as "format flag 64"), `format_internal.h` is on every
   consumer's include path (CMakeLists.txt:97), and `c5d_shard_reader` is a
   public non-opaque struct embedded by value in the "stable ABI" wrapper
   (stable.c:26) — which is why the `_v1` scheme does not achieve ABI stability.
7. Drop `_v1` suffixes for a fresh repo; use `vc_abi_version()`, a `struct_size`
   first field, hidden visibility plus a version script, and put the real
   stability guarantee on the *format* version, not the symbol names.
8. Cache: DROP — 251 lines, zero production callers (only test_quick.c), and it
   is a generic u64→bytes store; every intended consumer already has one.
   Shard: SPLIT into an optional module, and add a parse-index-from-a-buffer
   entry point so S3 users can do their own ranged GETs (which
   `tools/c5dc/remote.c` currently hand-rolls). tifxyz, codec_v0, codec_wav:
   drop (~1300 lines, bench/CLI-only).
9. Build: keep the warning set, the sanitizer enum, the OpenSSF hardening block
   (CMakeLists.txt:51-55) and `PRIVATE -ffast-math`; drop the bakeoff/zfp/
   label-bench/remote targets from the default configure, collapse
   `C5D_MARCH`/`C5D_MCPU` into one `VC_NATIVE`, ship one library instead of
   static+shared, and add the missing CMake package config and `.pc` file.
10. Naming: `vc_`/`VC_` for public symbols, `vci_` for cross-TU internals,
    `include/volcomp/volcomp.h` as the single installed header, `vc` as the
    single CLI, and 4-byte ASCII magics `VCB1/VCL1/VCU1/VCS1/VCSR` — three of
    today's five magics (shard.h:11, shard.c:56) serialise byte-reversed
    relative to their own comments.


<!-- END 03-api-architecture.md -->


---

<!-- BEGIN 04-secondary-codecs.md -->

# Secondary codecs review: label.c, tifxyz.c, codec_v0.c, codec_wav.c

Repo: /home/forrest/c5d (read-only). All line numbers below refer to that repo.

## 1. codec_v0.c / codec_wav.c — bakeoff-only baselines

**Confirmed DROP.** Both are wired into exactly one place: the bakeoff
harness registry.

- `tools/bakeoff/codecs.c:18-34` (`v0_encode`/`v0_decode`, `bake_codec v0`,
  name `"c5d0"`) and `:126-135` (`wav_encode`/`wav_decode`, `bake_codec wav`,
  name `"c5dwav"`); both included in `registry[]` at
  `tools/bakeoff/codecs.c:275-283`.
- `codec_v0.h:1-3` self-describes as "M1 lab baseline... Bakeoff-grade:
  single-threaded, scalar, format NOT frozen."
- `codec_wav.h:1-2` / `codec_wav.c:1-2` self-describes as "M1 challenger"
  and literally comments "lab duplication accepted" (`codec_wav.c:171`)
  about re-deriving brick.c's bitstream helpers.
- Neither has a spec/format.md entry, no CLI subcommand, no test file, no
  fuzz target. grep across `tools/`, `tests/`, `fuzz/` finds zero other
  callers of `c5d_v0_*` or `c5d_wav_*` (`codec_v0.h`/`codec_wav.h` are only
  included from `tools/bakeoff/codecs.c` and the .c files themselves).
- Their entire purpose was answering "does the winning brick codec
  (`c5d1`/brick.c) actually beat a plain DCT+dead-zone baseline and a
  3-level CDF-9/7 DWT baseline" during architecture selection — a decision
  already made (brick.c won; see docs/measured.md and README headline
  24.16x @ 42.91 dB for brick). They carry zero information for a
  minimized production API: no unique format, no unique feature (label
  masking, tifxyz surfaces), no external caller.
- **Both duplicate the run/level HybridUint + bitstream logic already in
  brick.c/format_internal.h nearly verbatim**: `codec_v0.c:52-130`
  (`bitw`/`bitr`/`hyb_emit`/`hyb_read`) is byte-for-byte the same shape as
  `codec_wav.c:171-241`, and both are pre-extraction duplicates of what
  became `format_internal.h`'s `c5df_bitr`/`c5df_hyb_read` shared by
  brick.c (`src/brick.c:90-129`). Keeping v0/wav means maintaining a third
  and fourth copy of that logic outside the shared header.

**What goes with them if dropped:**
- `src/codec_v0.c`, `src/codec_v0.h`, `src/codec_wav.c`, `src/codec_wav.h`
  (full files, not exported).
- `tools/bakeoff/codecs.c:1-5` (`#include "../../src/codec_v0.h"`,
  `"../../src/codec_wav.h"`), the `v0_encode`/`v0_decode`/`bake_codec v0`
  block (`:18-34`), the `wav_encode`/`wav_decode`/`bake_codec wav` block
  (`:126-135`), and their two `registry[]` entries (`:276`).
  `activity_qmap()` (`:39-79`) stays — it is brick-only (`b1_encode`
  `bp.chunk_qmap`), unrelated to v0/wav.
- No test/fuzz files reference either, so nothing else to remove there.
- If the new repo keeps *some* bakeoff/benchmark harness for the primary
  brick codec (recommended, to keep external baselines like zstd/zfp for
  future comparisons), it should register only the shipped codec(s)
  (`c5d1`-equivalent, passthrough, and optionally zstd/zfp if those deps
  are kept) — not v0/wav.
- Net: dropping removes 392 + 26 + 443 + 12 = 873 lines of engine code plus
  ~100 lines of bakeoff glue, with zero loss of shipped functionality.

## 2. label.c — full API/format inventory

`src/label.c` is 1503 lines, `src/label.h` is 83 lines. Format: `spec/format.md`
"Label bricks" section (lines 142-211), magic `C5L1` (`label.c:12`).

### API surface (label.h)
- `c5d_label_type` enum: u8/u16/u32/u64/i8/i16/i32/i64 (8 types,
  `label.h:27-36`).
- `c5d_label_channel {type, mask_chan, data}` — up to 64 channels
  (`C5D_LABEL_MAX_CHANNELS`, `label.h:39`), each optionally masked by a
  strictly-lower-numbered channel (`C5D_LABEL_NO_MASK` sentinel,
  `label.h:38,43`).
- `c5d_label_params {nsub, rans_nway, nthreads}` + `c5d_label_defaults()`.
- `c5d_label_encode` / `c5d_label_decode` (full-volume).
- `c5d_label_info` — header-only introspection (dims, per-channel type +
  mask) without touching the payload.
- `c5d_label_palette` — per-channel distinct-value list read from the
  header without decoding voxels.
- `c5d_label_decode_chunk` — single 16^3 chunk, touching only the owning
  substream (random access).

### Format features
- **Per-channel typed integer storage**, 8 fixed-width types, arbitrary
  per-channel type (not a uniform volume type like brick.c).
- **Masking / transitive validity**: any channel may reference a strictly
  lower-indexed channel as its mask; masked-out voxels are omitted from
  coding and forced to 0 on decode. Validity propagates through mask
  chains (`lb_plan` at `label.c:1387-1399`, `lb_mask_valid` at
  `label.c:664-667`).
- **Palette**: per-channel sorted list of distinct values, delta+zigzag
  LEB128-coded in the header (`label.c:963-965`), independently readable
  via `c5d_label_palette` without decoding (this is a real, used feature —
  `c5dc label-info` and downstream tooling read palettes cheaply).
- **Two chunk modes**: SAME (repeats running `prev`) / NEW-constant
  (whole 16^3 chunk one value) / CODED (per-voxel).
- **CODED voxel model**: causal 9-neighbor + running-previous candidate
  set, vote-ranked, rank-coded symbol or escape-to-palette-index-rank; 405
  voxel contexts (27 face-agreement states x 3 in-plane x 5 z-plane
  agreement counts) — see `lb_cands` (`label.c:270-347`) and
  `lb_vox_ctx` (`label.c:350-352`). This is genuinely bespoke to
  label-shaped (highly redundant, few distinct values) data; nothing like
  it exists in brick.c.
- **Sparse model-table transmission**: only used (model, token) entries
  are written, LEB128-coded (`lb_tables_put`, `label.c:608-625`) — matters
  because 405 contexts x 32 tokens would otherwise dominate small
  (1-4 KB) sparse-channel streams (documented rationale in
  docs/measured.md 2026-08-17).
- **nsub independent substreams per channel** (default `min(16,
  nchunks)`, max 128) for parallel encode/decode and chunk-level random
  access (`lb_geo_init`, `label.c:584-598`).
- **rANS interleave lanes** (1/2/4-way, `rans_nway` param) — measured
  nway=2 costs +3.4% bytes for no decode-speed win on label data
  (docs/measured.md), so default is 1; the knob exists but the measured
  data says never use it for this codec.
- **Threading**: `c5d_parallel_for` (shared pool, `src/common/pool.c`)
  drives three phases — per-substream distinct-value scan (phase A,
  `lb_scan_task`), per-substream tokenize (phase B, `lb_tokenize_task`),
  per-substream rANS encode (`lb_rans_task`) — plus parallel decode
  (`lb_decode_sub_task`).

### Dead / over-general / redundant code
- **Own open-addressing hash map** (`lb_map`, `label.c:185-249`) — used
  for palette construction (distinct-value sets per substream, merged
  per-channel value->index map). This is genuinely needed (nothing
  equivalent exists in brick.c since brick.c has no palette concept) but
  it's a second hand-rolled hash-map implementation in the codebase;
  worth checking whether `src/common/` has (or should have) one shared
  generic open-addressing map instead of this being label-private. Not
  found in `src/common/` today — grep shows this is the only hash map in
  the reviewed files.
- **Duplicated LEB128 varint codec**: `lb_leb_put`/`lb_leb_get`/
  `lb_leb_get32` (`label.c:139-170`) is label-private; brick.c's C5B3
  format uses fixed-width header fields instead (no varints), so there is
  no direct brick.c duplicate to point at, but this is exactly the kind of
  primitive (LEB128 varint read/write) that belongs in a shared
  `common/` header if tifxyz or a future format also wants variable-length
  ints — tifxyz.c currently does NOT use LEB128 (fixed u32 fields,
  `tx_wr32`/`tx_rd32`), so today there is no duplication, just an
  isolated instance.
- **Duplicated HybridUint bit-writer/reader**: `lb_bits_put`
  (`label.c:392-404`), `lb_tok_put`/`lb_hyb_put` (`label.c:405-424`), and
  the decode side `lb_hyb_get`/`lb_tok_get` (`label.c:486-516`) are a
  third independent reimplementation of the same HybridUint bitstream
  idiom already shared between brick.c and format_internal.h
  (`c5df_bitr`/`c5df_hyb_read`, `format_internal.h:84-105`) and separately
  reimplemented a fourth time in tifxyz.c (`tx_bits_put`/`tx_put`/
  `tx_get`, `tifxyz.c:367-430`). All four are functionally identical
  (`u<4` direct token, else `2+msb(u)` token with low bits in an LSB-first
  byte-aligned-at-end bypass stream) but none share code. This is the
  single clearest "should be unified" finding across the whole area (see
  §4).
- **rANS/model code is NOT duplicated** — label.c correctly reuses
  `entropy/rans.h`'s shared `rans_model`, `rans_model_build`,
  `rans_encode_multin16`, `rans_decn_init`/`rans_decn_get`
  (`label.c:10, 784-785, 1267, 487`). Good: this is the one piece brick.c,
  tifxyz.c, codec_v0.c, codec_wav.c, and label.c all correctly share.
- **Threading pool is NOT duplicated** — correctly reuses
  `common/pool.c`'s `c5d_parallel_for` (`label.c:9`, used at
  `label.c:856, 895, 920, 1373`). Good.
- **`lb_st`/`lb_gather` typed dispatch macros** (`label.c:35-108`) are
  label-specific (per-channel type dispatch into a shared `int64_t`
  working buffer) — no equivalent elsewhere since brick.c is
  single-typed (u8 grayscale). Not redundant, but the macro-generated
  switch-per-type is verbose; a function-pointer table (load/store per
  type) would cut ~60 lines with no measured cost (this is header/lane
  code, not the per-voxel hot loop).

### Is 1.5k lines justified?
Mostly yes, given the feature set actually in use (multi-channel typed
storage, transitive masking, palette-based lossless coding, chunk random
access, threaded phased encode). The bulk (`lb_cands`/`lb_enc_chunk`/
`lb_dec_chunk`, ~250 lines) is the genuinely novel per-voxel entropy model
that gives the measured 17-180x-vs-zstd-19 ratios on ids/winding fields
(docs/measured.md 2026-08-17) — this is not boilerplate, it's the payload.
The plumbing that padding it out to 1503 lines:
- Duplicated bitstream/HybridUint (~120 lines) — collapsible via §4.
- The stream-parsing layer (`lb_parse`, `lb_read_palette`,
  `lb_read_models`, `lb_sub_locate`, `label.c:1051-1218`, ~170 lines) is
  necessarily verbose because the format supports partial/chunk decode
  and per-channel/per-substream directories; this is not obviously
  simplifiable without losing chunk random access, which is a real,
  spec'd, tested feature (`c5d_label_decode_chunk`, used by
  `tools/label_bench.c` and `fuzz/d_label.c`).
- Encode-side buffer bookkeeping (`c5d_label_encode`, `label.c:791-1024`,
  ~230 lines) is mostly malloc/goto-cleanup ceremony for ~10 parallel
  arrays (`valid`, `sets`, `pmap`, `npal`, `pal`, `toks`, `sub_rans`,
  `sub_rans_n`, `models`, `mused`) — a small "encode context" arena
  allocator (single block, offsets) would shrink this without behavior
  change, but it's not incorrect as-is.

### Could ratios be preserved with simplification?
Yes for the plumbing above (bitstream unification, arena allocation), no
for the core model (candidate ranking + 405-context split is exactly what
docs/measured.md's context-model bake-off (2026-08-17) shows is load-
bearing: dropping to W/N/U/P-only loses ~30% on fg and ~5x on recto;
dropping face-neighbor state loses another large chunk). Do not touch the
core `lb_cands`/context formula when simplifying.

### Correctness / robustness concerns with hostile input
Overall label.c is careful — `lb_parse` (`label.c:1051-1149`) validates
magic, flags, nchan bounds, per-channel type byte + reserved-zero bytes,
mask_chan < c, palette strict-increasing + type-fit, `npal <=
nchunks*4096`, model index bounds (`< LB_NMODELS`), freq `<=
RANS_PROB_SCALE`, and total size == in_n. `fuzz/d_label.c` exercises
`c5d_label_info` -> `c5d_label_decode` -> `c5d_label_decode_chunk` on
arbitrary bytes; docs/measured.md 2026-08-17 claims "300 random bit flips
+ all truncations... never crash/hang", libFuzzer clean, TSan clean. A few
finer points worth flagging to whoever ports this:
- `lb_read_models` (`label.c:1164-1192`) re-derives `rans_model_build`
  from transmitted counts and then re-checks
  `models[m].freq[t] != counts[t]` to reject non-pre-normalized tables
  (`label.c:1187-1189`) — good defense (rejects a stream whose declared
  freqs don't already sum to `RANS_PROB_SCALE`), but note
  `rans_model_build` (`entropy/rans.c:7-41`) can still return 0 for
  freq-sum-not-4096 inputs that happen to normalize back to themselves
  only when the input was already exactly a valid distribution — the
  double-check exists precisely because `rans_model_build` alone doesn't
  enforce "already normalized," so this pattern (build, then compare) is
  the correctness-load-bearing part; it's duplicated in tifxyz.c
  (`tifxyz.c:610-612`) and codec_v0.c/codec_wav.c do NOT do this check
  (`codec_v0.c:328-336`, `codec_wav.c:379-388` just call
  `rans_model_build` and trust it) — those two are legacy-baseline code
  being dropped anyway (§1), but if any shared entropy-table-read helper
  is factored out for the new repo, it should bake in this stricter
  check by default rather than leaving it opt-in per caller.
- `lb_map_get`/`lb_map_grow` (`label.c:211-249`) grow-on-2x-load without
  an explicit cap; with `nchan` up to 64 and per-channel palette up to
  `nchunks*4096` (dim=1024 -> ~256M distinct values theoretically),
  encode-side memory is bounded by the *actual source data* (trusted
  input, not attacker-controlled) so this is fine for encode; decode-side
  never builds `lb_map`, only `pal[]` arrays sized by the (validated)
  `npal` header field, which is capped at `nchunk*4096` — bounded, but
  worth noting `npal` up to `1024^3/16^3*4096 = 268M` entries x 8 bytes =
  ~2GB allocation is permitted by a small (tens-of-bytes) crafted header
  before any payload validation. A denial-of-service-via-allocation-size
  vector (`lb_decode_channel`, `label.c:1318` `malloc((ch->npal ?:1) *
  8)`) — not a memory-safety bug, but a hostile-input sizing concern the
  new repo's minimized API should consider (e.g., a caller-supplied max
  palette / max size-in-bytes guard) if inputs can be untrusted network
  data rather than local files.

## 3. tifxyz.c — full treatment

`src/tifxyz.c` is 701 lines, `src/tifxyz.h` is 45 lines. Magic `TFX1`
(`tifxyz.c:348`).

### Is it in scope for a "volume compressor"?
Marginal fit. It compresses **2D float32 raster surfaces** (a (u,v) grid of
3D coordinates, i.e. a parameterized mesh sheet used by the Vesuvius
Challenge scroll-segment pipeline), not volumetric grayscale/label data.
It shares the entropy backend (HybridUint + rANS) conceptually but not the
16^3/128^3/1024^3 chunk/brick/shard geometry that defines the rest of the
format (spec/format.md's "Geometry" section, brick.c, label.c all key off
`dim % 16 == 0`). tifxyz operates on arbitrary `w x h` 2D grids with no
chunking, no substreams, no threading, no random access. It is a
**distinct format bolted onto the same repo for a specific downstream
workflow** (surface parameterization for ink-detection/segmentation),
not a "volume compressor" feature in the sense of brick.c/label.c.

Notably, **`spec/format.md` does not actually document tifxyz** — grep
finds zero occurrences of "tifxyz" or "TFX1" in `spec/format.md` despite
`README.md:11` claiming the spec covers "TFX1 tifxyz surfaces". This is
either a doc bug in c5d or evidence tifxyz was deliberately kept out of
the normative bitstream spec because it's considered a side format. Worth
flagging either way: it doesn't currently meet the same "spec'd, normative
format" bar the label codec meets (label has a full spec/format.md
section, `spec/format.md:142-211`).

### Who uses it?
- `tools/c5dc/tifxyz_cmd.c` (`tifxyz-pack`/`tifxyz-unpack`/`tifxyz-verify`
  subcommands) — CLI only.
- `tests/test_tifxyz.c` — roundtrip test.
- No fuzz target (`fuzz/` has only `d_brick.c`, `d_label.c`, `rt_brick.c`
  — no `d_tifxyz.c`/`f_tifxyz.c`). Hostile-input coverage is limited to
  the 200-trial inline bit-flip loop inside `tests/test_tifxyz.c:76-85`,
  not continuous/corpus-driven fuzzing like label.c gets. This is a real
  gap versus label.c's dedicated `fuzz/d_label.c` + documented libFuzzer
  runs.
- Not referenced from `tools/bakeoff/`, not in README beyond the format
  banner line.

### Lossless 2.75x / quant 5-12x — design soundness
- **Own minimal TIFF reader/writer** (`tifxyz.c:13-281`, ~270 lines):
  classic+BigTIFF, uncompressed or LZW, predictor 1/3, strips or tiles,
  1 sample/pixel float32. This is a real, non-trivial, self-contained
  parser (including a hand-rolled TIFF-variant LZW decoder,
  `tf_lzw`, `tifxyz.c:48-103`) whose only job is reading the specific
  TIFF flavor Vesuvius Challenge tooling (tifffile) emits. This is
  arguably out of scope for a "volume compressor" core library — it's
  file-format I/O glue, not compression. If tifxyz ships at all, this
  TIFF reader/writer should live in a CLI/tool-only layer, not the core
  codec library, since a caller integrating the compressor into another
  pipeline has no reason to want a bundled TIFF parser.
- **Masked parallelogram prediction** (`tx_pred`, `tifxyz.c:448-458`):
  W+N-NW / (W+N)/2 / W / N / NW / running-`last`, chosen by which causal
  neighbors are valid (mask-aware). Sound, standard technique (same
  family as PNG's Paeth / lossless JPEG predictors), appropriately
  adapted for masked/irregular-boundary grids (real segments have ragged
  edges and holes, per docs/measured.md 2026-08-10).
- **Two coding modes**: lossless via `tx_fmap`/`tx_funmap`
  (`tifxyz.c:439-444`) — a monotonic float32-bit-pattern remapping
  (`(u&0x80000000)?~u:(u|0x80000000)`) so residuals of nearby floats
  stay small and zigzag-friendly even across the sign boundary; this is
  a well-known trick (order-preserving float bit-cast) and is sound.
  Quantized mode: `llround(v * 2^log2q)`, dequant `(float)(dv /
  2^log2q)`, verified exact/idempotent in `tests/test_tifxyz.c:104-116`
  ("stab" re-encode-idempotence check) — sound, and the `|v|*2^log2q <
  2^36` guard (`tifxyz.h:14-15`, checked implicitly via the encode/decode
  int64 domain) avoids silent overflow.
- **Mask RLE + per-pixel residual tokens, single rANS pass, 7 models**
  (mask runs; per-component calm/rough context via
  `prev_tok[c] >= TX_ROUGH_TOK`, `tifxyz.c:353,522`) — much simpler than
  label.c's 405-context model, appropriate for smooth per-pixel
  prediction residuals rather than label.c's near-constant/palette-index
  data.
- Ratios are measured and documented (docs/measured.md 2026-08-10):
  2.74-2.78x lossless, 5.0-11.8x quantized depending on step, versus
  0.99-1.9x for LZW-tif/zstd-19/xz-6 baselines on the same 3 real
  segments — a real, substantial win over the naive alternatives, so the
  design is doing its job.

### Duplicated code
- **HybridUint + bitstream**: `tx_bits_put`/`tx_put`/`tx_get`
  (`tifxyz.c:367-430`) is the fourth independent copy of the same idiom
  (see §2 and §4) — distinct in one way: tifxyz's tokens are 64-bit-value
  capable (`63u - clzll`, `tifxyz.c:386`) vs label's/brick's 32-bit
  (`31u - clz`), because tifxyz needs to represent full float32-bit-
  pattern-mapped values (up to 2^32) as zigzag deltas which can span the
  full 33-bit range — this is the one place the four copies aren't
  byte-identical, so any unification needs a 64-bit-capable variant (see
  §4 recommendation).
- **Zigzag**: `tx_zig`/`tx_unzig` (`tifxyz.c:432-437`) is a third zigzag
  implementation (`c5df_zigzag`/`unzigzag`, 32-bit, in
  `format_internal.h:76-81`; `lb_zig64`/`lb_unzig64`, 64-bit, in
  `label.c:131-136`). tifxyz's is 64-bit like label's but written
  independently (`v>=0 ? v<<1 : (-(v+1)<<1)+1` vs label's
  `(v<<1)^(v>>63)` — same result, different formula, both correct branch-
  vs-branchless implementations of the same operation).
- rANS/model code correctly reuses `entropy/rans.h` (`tifxyz.c:11`,
  `rans_model_build`/`rans_encode_multin`/`rans_decn_init`/
  `rans_decn_get` at `tifxyz.c:539,543,628,410`) — same good pattern as
  label.c.
- No threading (`c5d_parallel_for` not used) — tifxyz is single-threaded
  only; given typical segment sizes (docs/measured.md: ~194/246 MB/s 1T
  on a 26 MB test file) this is probably fine for the format's actual
  scale (2D surfaces, not 1024^3 volumes), but it's an inconsistency
  worth deciding on purpose rather than by omission if tifxyz ships.

## 4. Shared-code opportunities (brick / label / tifxyz)

Concrete unification candidates, ranked by duplication severity:

1. **HybridUint token codec + LSB-first bypass bitstream** (highest
   value). Four independent implementations of the identical algorithm:
   - `format_internal.h:84-105` (`c5df_bitr`, `c5df_hyb_read`) +
     `brick.c:90-129` (`bitw`, `hyb_emit`) — shared already between
     encode/decode of brick.c via format_internal.h, this is the
     reference implementation.
   - `label.c:392-424` (`lb_bits_put`, `lb_tok_put`, `lb_hyb_put`) +
     `label.c:486-516` (`lb_hyb_get`) — 32-bit value range.
   - `tifxyz.c:367-397` (`tx_bits_put`, `tx_put`) + `tifxyz.c:409-430`
     (`tx_get`) — 64-bit value range (the one real behavioral variant).
   - `codec_v0.c:52-130`, `codec_wav.c:171-241` — dropped per §1, but
     were two more copies until now.
   Recommendation: move a single `bitw`/`bitr` (LSB-first byte-aligned
   bypass stream) + `hyb_emit`/`hyb_read` pair into `common/` (e.g.
   `common/bitstream.h`), parameterized or provided in both 32-bit and
   64-bit-value flavors (or just always 64-bit internally — the 32-bit
   users pay nothing extra since `u < 4` and small-k paths are identical
   cost). Every consumer (brick, label, tifxyz) keeps its own token
   model / context selection, only the mechanical bit-packing unifies.

2. **Zigzag encode/decode** (32-bit in format_internal.h, 64-bit
   reimplemented twice differently in label.c and tifxyz.c) — trivial 4-
   line functions, but three copies for one operation. Fold into the same
   `common/bitstream.h` (or a `common/varint.h`) as `zigzag64`/
   `unzigzag64`, with the 32-bit brick.c version becoming a thin cast or
   direct alias.

3. **rANS model table serialization + the "re-verify pre-normalized"
   pattern** (§2 correctness note): label.c's `lb_read_models`
   (`label.c:1164-1192`) and tifxyz's inline table read
   (`tifxyz.c:599-613`) both (a) read raw counts, (b) call
   `rans_model_build`, (c) re-compare `model.freq[t] != counts[t]` to
   reject non-pre-normalized streams. brick.c likely has its own version
   too (not fully reviewed here, but format_internal.h is shared with the
   GPU decoder so it may already be centralized — worth checking during
   implementation). This "build + verify" pattern belongs as a single
   `rans_model_read` helper in `entropy/rans.h`/`.c` rather than being
   re-derived per caller, closing the gap where codec_v0/codec_wav
   silently skipped the check (moot once those are dropped, but a good
   habit to bake into the shared primitive so no future codec forgets
   it).

4. **LEB128 varint** — currently label-only (`label.c:139-170`). Not
   strictly duplicated today, but if the new minimized format wants
   variable-length header fields anywhere else (e.g. a future compact
   tifxyz header, or brick.c's shard-level metadata), this should live in
   `common/` from the start rather than being copy-pasted a second time
   under a new name.

5. **Threading pool** (`common/pool.c`) and **rANS core**
   (`entropy/rans.c`) — already correctly shared, no action needed. This
   is the good pattern to replicate for items 1-4.

6. **Not** worth unifying: label.c's open-addressing `lb_map` (palette
   construction) and tifxyz's TIFF reader — both are single-consumer,
   purpose-specific, and forcing them into `common/` would add an
   abstraction layer for one caller each.

## 5. Performance and quality opportunities (concrete file:line)

Label:
- `label.c:270-347` (`lb_cands`) is the documented hot spot
  (docs/measured.md 2026-08-17: "~12 ns per coded voxel... SIMD/
  branchless candidate build is the obvious next step" on dense/noisy
  data). The insertion-sort-by-count (`label.c:320-333`) and the O(nd)
  linear scans in `lb_esc_rank`/`lb_esc_unrank` (`label.c:355-376`) are
  fine given `nd <= 10` (LB_MAXCAND) but the neighbor-gather branching
  (`label.c:276-296`, six `LB_ADD` conditional loads) is a straightforward
  branchless/SIMD target already identified by the project's own notes —
  no new finding here, just confirming the doc's claim by reading the
  code: the branches are genuinely per-voxel and genuinely on the hot
  path (`lb_enc_chunk`/`lb_dec_chunk` inner loops, `label.c:454-470`,
  `544-575`).
- `label.c:35-108` (`lb_st`/`lb_gather`): per-call type-switch macros
  invoked once per chunk (gather) or per voxel (store, inside the decode
  scatter loop at `label.c:1283-1298`) — the store path
  (`lb_st` called per-voxel in the scatter loop) re-dispatches the type
  switch every voxel instead of hoisting it out via a function pointer or
  a per-type specialized scatter loop; for dense channels (docs/measured
  s1 L1: 100-120ms/brick 1T) this switch-per-voxel is pure overhead that
  a hoisted-dispatch scatter (mirroring `LB_GATHER_T`'s hoist-out-of-loop
  pattern already used on the encode/gather side) would remove.
- `label.c:1049-1050` comment: "per-substream directory and tables are
  decoded on demand" — `lb_sub_locate` (`label.c:1195-1218`) re-walks the
  directory from the start for every substream lookup (`for (k=0; k<=s;
  k++)`), i.e. O(nsub) per substream located, O(nsub^2) total if all
  substreams of a channel are decoded sequentially without caching
  offsets. With `nsub` capped at 128 this is bounded and cheap in
  absolute terms, but it's an easy linear-scan-to-prefix-sum fix if
  channel decode ever needs to scale nsub up.
- Threading: encode is 3 sequential parallel-for phases (scan, tokenize,
  rANS) with a serial palette-merge in between per channel
  (`c5d_label_encode`, `label.c:854-919`) — channels are processed one at
  a time in the phase-A/palette loop (`label.c:854-892`), not overlapped
  across channels; for `nchan` up to 64 (rare in practice — measured
  usage is 5 channels) this leaves some parallelism on the table between
  channels, but given typical nchan=5 and nsub=16 there's already
  plenty of substream-level parallelism to saturate cores, so this is low
  priority.

Tifxyz:
- `tifxyz.c` is entirely single-threaded — no `c5d_parallel_for` use.
  Given surfaces are 2D grids (not 128^3 bricks) the absolute per-call
  cost is presumably lower, but the raster-order residual loop
  (`tifxyz.c:511-528` encode, `tifxyz.c:650-669` decode) is a serial
  causal-dependency chain by construction (parallelogram prediction needs
  W/N/NW), so it's not embarrassingly parallel row-by-row without a
  wavefront/diagonal scheduling scheme — not a quick win, more of an
  architectural question if tifxyz ships and large segments become a
  bottleneck.
- `tx_pred`'s per-pixel branch cascade (`tifxyz.c:451-457`, up to 5
  branches per pixel per component = up to 15 branches/pixel) is the
  per-pixel hot path; a lookup-table-indexed dispatch (8 W/N/NW presence
  combinations -> function pointer or precomputed coefficient pair) would
  cut branching, mirroring label.c's face-context precomputation
  approach.
- `tf_lzw` (`tifxyz.c:48-103`) allocates 3 `_Thread_local` 4096-entry
  tables (`prefix`/`suffix`/`stack`) that persist for the process
  lifetime per thread — fine for a CLI tool, questionable if tifxyz's
  TIFF reader is linked into a long-lived multi-threaded server process
  (thread-local static growth per thread that never shrinks). Not a bug,
  a note for embedding contexts.

## 6. Recommendation for the new repo

**Ship: label codec only, as a well-scoped secondary API.**
- Keep `c5d_label_encode`/`decode`/`decode_chunk`/`info`/`palette` — this
  is a real, spec'd, fuzzed, measured, externally-motivated feature
  (multi-channel lossless label volumes with masking) that's a natural
  companion to a volume compressor's primary grayscale/lossy path, not a
  bolt-on. It has actual measured wins (17-180x vs zstd-19 on ids/
  winding) that justify shipping it.
- Minimal params to expose: keep `nsub` (0=auto is a fine default,
  rarely needs override) and `nthreads`; **drop `rans_nway` from the
  public param struct** — docs/measured.md 2026-08-17 explicitly measured
  nway=2 as a pure loss (+3.4% bytes, no speed win) for this codec's data
  shape, and nway=4 is presumably worse still; keep it as an internal
  constant (1-way) rather than a knob nobody should turn. This shrinks
  `c5d_label_params` from 3 fields to 2.
- Simplify before porting: unify the HybridUint/bitstream/zigzag code per
  §4 item 1-2 (saves ~150 lines and removes a maintenance hazard);
  consider the arena-allocation cleanup for `c5d_label_encode`'s ~10
  parallel buffers (§2); keep `lb_cands`/context model untouched (it's
  the actual product).
- Address the hostile-input allocation-size gap (§2, `npal`-driven
  unbounded-looking allocation) if the new repo's threat model includes
  untrusted/network input, by adding an explicit max-palette-bytes or
  max-total-size guard in the public decode/info entry points.

**Drop: tifxyz codec**, or at minimum **do not ship it as part of the
core "volume compressor" library**. Reasoning:
- It compresses a fundamentally different data shape (2D parameterized
  float surfaces, not volumes) for one specific downstream pipeline
  (Vesuvius Challenge surface/ink workflow) — out of charter for a
  general "volume compressor" with a "small stable API."
- It is not part of the normative spec (spec/format.md has no tifxyz
  section despite the README claiming otherwise) and has no fuzz-corpus
  coverage (only an inline 200-trial bit-flip smoke test), i.e. it does
  not meet the same robustness bar as label.c or brick.c.
- Its TIFF reader/writer (~270 lines) is file-I/O glue for a specific
  external tool's output format, not compression logic — even if the
  *codec* (masked-parallelogram-predict + rANS) were kept, the TIFF layer
  should not live in a core codec library.
- If the consuming team still needs this functionality, the pragmatic
  path is to keep tifxyz.c as a **separate, optional module/tool**
  (own repo or a clearly-labeled `contrib/`/`extras/` directory) rather
  than part of the minimized core API surface — it can still reuse the
  unified bitstream/zigzag primitives from §4 if kept anywhere.
- If it is kept in-tree despite this recommendation: give it a real
  fuzz target (`fuzz/f_tifxyz.c` mirroring `fuzz/d_label.c`'s pattern)
  and add its format to spec/format.md before calling it a stable API,
  and switch its 64-bit-value HybridUint to the shared primitive from
  §4.

**Drop: codec_v0 and codec_wav** entirely (§1) — no ambiguity here, they
are historical bakeoff baselines with zero unique shipped functionality
and duplicated bitstream code that the shared-primitives cleanup (§4)
would otherwise need to account for a third and fourth time.

**Net effect**: of the four files reviewed, one (label.c) ships as a
minimized, unified-primitives version of itself; one (tifxyz.c) is
recommended out of the core repo (or demoted to an optional extra); two
(codec_v0.c, codec_wav.c) are deleted outright along with their bakeoff
glue.


<!-- END 04-secondary-codecs.md -->


---

<!-- BEGIN 05-infra.md -->

# Infra review: shard.c/h, cache.c/h, pool.c/h, platform.h, c5d_core.c, stable.c/h, tests

All paths under /home/forrest/c5d (read-only). Line numbers refer to that tree.

## 0. Caller map (ground truth before opinions)

- `c5d_shard_*` (shard.h:32-58): called from `tools/c5dc/pack.c` (create/put/put_zero/close/open/brick/brick_is_zero/close_reader),
  `src/stable.c` (create/put/put_zero/close/open/brick/is_zero/is_missing/close_reader),
  `tests/test_hostile.c`, `tests/test_stable.c` (indirectly via stable.c). `c5d_crc32c`/`c5d_crc32c_cont` also used directly by
  `tools/c5dc/remote.c:154,362` for verifying blobs fetched over HTTP — remote.c does **not** call `c5d_shard_open`/`c5d_shard_brick`;
  it re-implements footer/index parsing itself against `c5d_shard_footer`/`c5d_shard_entry` structs (remote.c:94-120,184,247-310) to
  avoid mmap'ing a whole remote object. So the mmap-based reader half of shard.c (open/brick/brick_is_zero/brick_is_missing/close_reader)
  has exactly one real production caller: stable.c (i.e., only reachable through the stable façade + `c5dc pack`'s local `unpack`/`stat` paths).
- `c5d_cache_*` (cache.h): **zero production callers**. `grep -rl cache.h` → only `src/cache.c` and `tests/test_quick.c`. Not referenced
  from `tools/c5dc/remote.c` (the doc comment in cache.h explicitly frames it as for "the S3 remote client") nor from `pack.c`, fuzz, or
  stable.c. It is fully dead code outside its own smoke test.
- `c5d_parallel_for` (pool.h:12): called from `src/brick.c` (5 call sites), `src/label.c` (4 call sites), `tools/c5dc/pack.c` (2), and
  `tests/test_quick.c`. This is a real, load-bearing shared dependency.
- `c5d_version_string`/`c5d_status_string` (c5d_core.c): `c5d_status_string` used once, in `tests/test_quick.c:49`. `c5d_version_string`
  has **no callers anywhere** in src/tools/tests/fuzz.
- No fuzz target touches shard.c, cache.c, pool.c, or stable.c (`fuzz/` only has d_brick.c, d_label.c, rt_brick.c). The hostile-input
  coverage for shard.c lives entirely in `tests/test_hostile.c` as hand-written regression cases, not continuous fuzzing.

---

## 1. Line-by-line review

### src/shard.c

- **crc32c dispatch (13-49)**: ARM64 hardware path vs. software fallback table built via `pthread_once`. Fine, standard. No x86 SSE4.2
  `_mm_crc32_*` path at all — on x86-64 (the realistic deployment target for a desktop/server codec) this always takes the byte-at-a-time
  software table (shard.c:40-45), which is measurably slower than either ARM CRC or x86 CRC32 intrinsics on GBs of brick data flowing
  through `c5d_shard_put`/snapshotting. This is a real speed gap for "best encode/decode speed" if shard files are written/verified on
  x86 (crc is computed on every `c5d_shard_put` payload at shard.c:122 and re-verified on every read at shard.c:228).
- **`write_snapshot` (73-85)**: writes index+footer+crc+magic and fsyncs *every call*, unconditionally, even when called from
  `c5d_shard_put`'s automatic 64 MB cadence. fsync on every ~64 MB chunk is a real throughput cost for local NVMe writes and a much bigger
  one on network/S3-backed filesystems; there is no way to disable snapshotting for callers who don't need crash recovery (e.g. batch
  pack jobs writing to scratch space that get retried wholesale on failure). No count of failed writes distinguishes "disk full" from
  "transient" — a single `fwrite`/`fsync` failure aborts the whole put chain (see `c5d_shard_put` returns -1, caller then has a half-open
  writer with no way to know how many snapshots actually landed).
- **`c5d_shard_put` (118-130)**: `n > UINT32_MAX` check (119) is correct overflow guard for `e.nbytes`. But `w->cursor += n` (123) is
  unchecked `uint64_t` addition — with adversarial/very large cumulative writes this can theoretically wrap after exabytes; low priority
  but the codebase elsewhere is careful about wraparound (e.g. shard.c:225-226), so this is an inconsistency, not a real-world risk at
  current brick/shard scales (max 512 bricks/shard per shard.h:1-2).
- **`c5d_shard_close` (146-158)**: on `fwrite`/`fflush`/`fsync` failure, `rc` stays -1 but the code still `fclose`s and frees everything
  and returns -1 — correct cleanup, but the **caller has no way to know if a partial footer got written to disk** (fwrite of index could
  succeed, footer fwrite fail, file left with a stray half index and no valid footer — this is actually what crash-recovery scanning in
  `c5d_shard_open` is for, so it's self-healing, but it's fragile: recovery depends on some *earlier* full snapshot having landed. If
  `close` is the very first flush point (e.g. shard smaller than 64 MB, no snapshot ever fired) and it fails midway, the shard is
  unrecoverable. `c5d_shard_snapshot_now` (139-144) exists as a manual escape hatch but nothing calls it except the test.)
- **`c5d_shard_open` (160-218)**: `st.st_size < (off_t)sizeof(c5d_shard_footer)` guard (165) is correct. `r->foot.nbricks <= (1u << 20)`
  (177) is an arbitrary sanity cap — reasonable but undocumented relationship to `shard.h`'s stated "up to 512 bricks" design comment
  (shard.h:1-2); a hostile footer claiming `nbricks == 1<<20` with a shard sized just enough to satisfy `map_n >= footer+idx_bytes` would
  pass the check and produce a 1M-entry `c5d_shard_entry` index read via `c5d_shard_brick`, which is still bounds/crc-checked per-entry
  (shard.c:224-228) so no OOB, just wasted validation time for a bogus file — low severity.
- **Crash-recovery backward scan (181-215)**: scans byte-by-byte backward for a 4-byte magic (183-194), which is O(file size) per byte in
  the worst case (`while (pos--) { memcmp 4 bytes }` — no skip-ahead via Boyer-Moore/memmem), then for every accidental 4-byte magic
  collision does a full CRC recompute over up to the whole index+footer before giving up and continuing the scan (196: `end = pos`, retry
  from just before the false match). On a large adversarial or simply unlucky file (magic bytes are common — 0x35,0x43,0x53,0x52 is not a
  rare 4-byte sequence) this is quadratic-ish in the worst case and is a **hostile-input DoS vector**: an attacker who controls shard
  bytes can force many spurious magic hits near the end of a huge file, each triggering a `c5d_crc32c` pass over a large index, before
  `c5d_shard_open` ever returns -1. There's no time/iteration budget beyond the natural shrinking of `end`. Not exploitable for memory
  safety (bounds are all checked), but plausible for a slow-open DoS on untrusted shard files (e.g. downloaded-and-opened locally, or a
  fuzzer feeding it — though no fuzz target exercises this path today, see caller map above).
- **`c5d_shard_brick` (220-231)**: `e.offset >= C5D_SHARD_OFFSET_ZERO` (224) correctly treats both sentinels (MISSING = UINT64_MAX, ZERO =
  UINT64_MAX-1) as "no blob" since ZERO is the smaller of the two — relies on the specific values chosen in shard.h:12-13; fragile if
  someone ever adds a third sentinel without re-checking this comparison, but currently correct and commented (222-226 have good
  overflow-safety comments). CRC checked on every access (228) — no caching of "this brick already validated", meaning repeated
  `c5d_shard_brick` calls on the same reader re-hash the blob every time. For a decoded-brick cache built *on top of* shard access (the
  stated purpose of cache.c, which is unused — see below) this asymmetry (crc-per-read forever, no memoization) is one more reason a
  brick-level LRU in front of the shard reader would pay for itself, except nothing wires that up today.
- **`c5d_shard_brick_is_zero`/`_is_missing` (233-245)**: both null-check `r`/`r->map` defensively (234, 241) — inconsistent with
  `c5d_shard_brick` itself (220), which does *not* null-check `r`, and dereferences `r->foot.nbricks` unconditionally. Any caller that
  passes a `NULL` reader to `c5d_shard_brick` (vs. the is_zero/is_missing siblings) segfaults instead of returning gracefully. Minor API
  inconsistency worth fixing if this ships in the new repo.
- **`c5d_shard_close_reader` (247-249)**: double-close is silently safe (only unmaps if `r->map` non-null, then nulls it) — good.

### src/shard.h

- Header doc (39-41) says "every 64 MB... c5d_shard_open recovers from the newest snapshot" — accurate and matches implementation.
- `c5d_shard_reader` struct exposes raw `map`/`map_n`/`index_off` fields directly (shard.h:44-49) rather than opaque handle — fine for an
  internal module but means `stable.c` and `pack.c` reach into `reader->foot.brick_dim` etc. directly (see stable.c:529-535, 577, 601),
  coupling those callers to shard.c's internal footer layout. Not a bug, but it's the reason stable.c can't treat shard as a black box —
  contributes to stable.c boilerplate (see §3).

### src/cache.c / cache.h

- **Dead code**: per caller map, `c5d_cache_*` has no production caller. The header's own comment (cache.h:1) and the design doc says
  it exists for "the S3 remote client" (per your framing), but `tools/c5dc/remote.c` does its own ranged-GET + per-brick decode with no
  cache layer at all (grep for `c5d_cache` in remote.c: zero hits). This is speculative infrastructure — a full S3-FIFO-lite
  implementation (ghost ring, probation/main queues, CLOCK, pinning, per-shard mutexes) that nothing in the shipped tools uses.
- **S3-FIFO correctness review** (in case it's kept): `make_room` (109-150) has a `guard = 4096` iteration cap (110) — a defensive bound
  against a queue that can't make room (e.g. everything pinned); good practice, but it means under pathological pin patterns
  `make_room` can silently give up *before* the caller-facing backpressure check at cache.c:186 fires, so the -2 return isn't a hard
  guarantee, just "usually accurate." With `guard` exhausted, `c5d_cache_put` still re-checks total bytes after the loop (186) so
  correctness (never exceeding budget by inserting) holds — it just might reject (-2) slightly more eagerly than "every eviction
  candidate is pinned" would suggest, since the loop can bail before literally checking everything. Comment on line 186-188 slightly
  overstates the guarantee.
- **Ghost ring** (`ghost[GHOSTS]` per shard, 1024 entries, cache.c:27,85-93): linear scan `ghost_has` is O(1024) *per put* (85-88) — for
  16 shards × 1024 = fine at small scale, but it's an O(n) membership test where a Bloom filter or hash set would be O(1); given this
  whole cache is unused, the cost is moot, but if resurrected for a hot decode path this scan matters.
- **`c5d_cache_stats_get` (222-235)**: takes `const c5d_cache *c` but casts away const to lock the mutex (226: `(shard *)&c->shards[s]`)
  — legal C but a code smell; `pthread_mutex_t` fields should either not be under `const` or the API should not claim const in the first
  place. Locking is otherwise correct (locks each shard once, no double-locking, no missed unlocks on any path I can see).
- **Pinning semantics**: `c5d_cache_get` increments `pins` and never expires/times out a forgotten `release` (168-174) — a caller that
  gets-and-forgets to release leaks a permanent eviction-exemption for that entry (not a memory leak, but a policy leak: entry becomes
  unevictable forever). No debug/assert mode counts outstanding pins or warns on `c5d_cache_destroy` with nonzero pins (237-251 frees
  everything unconditionally regardless of `pins` — actually **this is a latent use-after-free hazard for any real caller**: if a
  pinned pointer returned by `c5d_cache_get` is still in use by another thread when `c5d_cache_destroy` runs, `free(e->data)` yanks it
  out from under that thread. There's no assertion or wait against `pins != 0` in destroy.) Since nothing calls this in production this
  is currently theoretical, but it's a real correctness gap in the design that would need fixing before resurrecting the cache.
- **`put_zero` sentinel**: cache.c has no such concept — that's a *shard.c* concept (`C5D_SHARD_OFFSET_ZERO`). Confirmed by reading both
  files; the task brief's "put_zero sentinel" question maps to shard.c:132-137 (`c5d_shard_put_zero`), not cache.c. It is used (pack.c
  and stable.c both call it, see caller map) and is a legitimate small optimization: bricks that are provably all-zero (common in
  micro-CT air/background regions) skip writing/reading/decoding a payload entirely (shard.c:224 short-circuits `c5d_shard_brick` to
  NULL for zero entries; stable.c:555-564,584-588,609-613 synthesize zeros with `memset` instead of decoding). This is justified and
  cheap — keep it.

### src/common/pool.c / pool.h

- **Global singleton pool** (`g_pool`, pool.c:29-33) with lazy `pthread_once` spin-up (pool_init, 95-103) that creates
  `available_cpu_count()-1` threads (98, main thread is worker #1) and **never joins/destroys them** — they park forever in
  `pool_worker`'s `pthread_cond_wait` (79-83) for the lifetime of the process. No `atexit` handler, no shutdown API at all. For a
  process that embeds this as a library and unloads it (dlclose, Python module reload, etc.) this leaks OS threads referencing freed
  code — real risk if this becomes a Python-embedded library per the stable.h doc comment ("suitable for dlopen/ctypes users",
  stable.h:4-5). Worker threads never check any "shutdown requested" flag; the only way they stop is process exit killing them.
- **Fork-unsafety**: no `pthread_atfork` handling. If the host process forks after the pool has spun up workers, the child inherits a
  frozen `g_pool.mu`/`g_pool.work` (potentially locked) with zero live worker threads (fork only preserves the calling thread) — any
  subsequent `c5d_parallel_for` call in the child with `nthreads > 1` will submit a job and then block forever in `pthread_cond_wait`
  waiting for workers that don't exist in that process image. This is a classic fork+pthread hazard and is completely unhandled.
- **`default_thread_count` cache (51-61)**: caches `available_cpu_count()` in a static atomic on first use and never re-samples — if the
  process's CPU affinity mask changes at runtime (cgroup update, `taskset` from another tool, etc.) this pool won't adapt; low-impact
  but worth noting for a long-lived server process.
- **nthreads 0/1 semantics**: `c5d_parallel_for(nitems, fn, ctx, nthreads)`: `nthreads==0` → “use all CPUs” (pool.h:11, pool.c:107);
  `nthreads==1` (or nitems==1 after clamping, 108-109) → **runs the loop synchronously in the calling thread with zero pool
  interaction** (110-113), never touching `g_pool` or blocking on its mutex at all. This is a good fast path (no thread-pool overhead
  for degenerate single-thread cases) and is consistently honored by every caller I checked (brick.c, label.c, pack.c) — I did not find
  a caller passing an out-of-range nthreads value that would violate the `nthreads>256` clamp (109) or the `nthreads>nitems` clamp
  (108). One thing to flag: **the meaning of `threads` differs at the `stable.c` boundary** — `c5d_stable_brick_encode_u8_v1`'s
  `threads` parameter is passed straight through as `parameters.nthreads` (stable.c:275) with **no validation at all** (no 0/1/N check
  in `channels_valid`/`dimensions`/anywhere in stable.c) — a caller passing `UINT32_MAX` reaches `c5d_parallel_for` which itself clamps
  to 256 (pool.c:109), so it's safe, just silently different behavior than what the caller asked for with zero feedback.
- **Concurrent-job scheduling** (`find_runnable_job`, 63-67; `pool_worker`, 76-93): simple linked-list of jobs scanned linearly by every
  idle worker on every wakeup — O(jobs) per dispatch, fine at the scale of "a handful of concurrent parallel_for calls" but would not
  scale to many small concurrent jobs. The `limit` field (per-job thread cap, set to the caller's requested `nthreads`, pool.c:121) is
  enforced correctly under the single global mutex, so independent callers submitting jobs concurrently do get bounded parallelism per
  job — this is the design intent stated in pool.h:1-2 ("Multiple callers and nested jobs may submit concurrently").
- **Submitting-thread participation** (129-143): the calling thread runs the same claim/execute loop as pool workers, which is correct
  and avoids one wasted thread, but note it means **the calling thread is not returned to the caller until the whole job finishes** even
  though it's spending some of that time helping — fine for this codec's synchronous encode/decode API shape, just noting there's no
  async/non-blocking submission mode (not needed here).

### src/common/platform.h

- Small, clean. `c5d_alloc_aligned` (20-24): `size + align - 1` can overflow `size_t` for adversarial huge `size` near `SIZE_MAX` — no
  overflow check, unlike the careful `ckd_add`/`ckd_mul` helpers defined two lines above it in the same file (17-18) that this function
  conspicuously does *not* use for its own rounding math. Given brick/label sizes are bounded (dim ≤ a few hundred, channels ≤ 64), this
  is not reachable from real inputs today, but it's an inconsistency in a file whose whole purpose is "checked arithmetic helpers."
  Grep shows **`c5d_alloc_aligned` itself has no callers** in the codebase (only `c5d_now_ns`, `c5d_mul_size`, `c5d_add_size` are used
  transitively via callers of platform.h) — worth confirming at export time whether it's needed at all.
- `clock_gettime(CLOCK_MONOTONIC_RAW, ...)` (12): Linux-specific clock id: not available on macOS pre-some-versions or other POSIX
  systems without fallback — `platform.h` has no `#ifdef` guarding this, so it is Linux/BSD-specific despite being named "shared
  platform helpers." If the new repo targets non-Linux, this needs a fallback to `CLOCK_MONOTONIC`.

### src/common/c5d_core.c

- 15 lines, two functions. `c5d_version_string` (3): **zero callers anywhere in src/tools/tests/fuzz** — pure dead weight, presumably
  meant for a CLI `--version` flag that doesn't exist yet, or for embedders. `c5d_status_string` (5-14): one caller, a single assert in
  `test_quick.c:49`. Both are trivial and harmless to keep (near-zero cost), but neither is presently pulling weight; `c5d_status_string`
  duplicates `c5d_stable_status_string` in stable.c (149-162) as a second, differently-enumerated status-to-string mapping — two
  parallel status enums (`c5d_status` in c5d.h vs. `c5d_stable_status` in stable.h) each with their own stringify function is exactly
  the kind of duplication a minimized single-API repo should collapse into one.

---

## 2. Dead code / over-engineering summary

| Component | Verdict | Evidence |
|---|---|---|
| `c5d_cache_*` (cache.c, 251 lines + cache.h) | **Fully dead in production.** Only consumer is its own smoke test (`tests/test_quick.c:124-137`). | grep of `cache.h` across repo: 2 files, both are the cache's own source/test. |
| Snapshot recovery every 64 MB (shard.c:56-85,125-128) | **Justified, keep**, but the fsync-every-64MB cost should be tunable/optional, and the crash-recovery backward scan (shard.c:181-215) should be hardened against the DoS pattern noted above (skip-scan or magic + immediate CRC-fail early exit tuning) before trusting it on untrusted input. | Only real caller path is `c5d_shard_close`/`c5d_shard_open` used by `pack.c` and `stable.c`; `test_hostile.c:147-184` exercises exactly the crash-recovery scenario this exists for, so the feature earns its keep functionally — it's the implementation's robustness under adversarial magic collisions that's unproven (no fuzz coverage). |
| Ghost ring in cache (cache.c:27,85-93) | Moot — the whole cache is dead. If cache is dropped, this goes with it. If cache is kept, ghost ring is a legitimate part of S3-FIFO and not itself over-engineered, just implemented with an O(n) scan that should become a small hash/Bloom structure. | — |
| `put_zero` sentinel (shard.c:132-137, shard.h:13,35-36) | **Justified, keep.** Real production use in `pack.c:261` and `stable.c:470-482,555-564,584-588,609-613`; avoids encoding/storing/decoding known-empty micro-CT regions, a real win for CT data with large air/background volumes. | caller map above |
| Global pool with no shutdown (pool.c) | Over-simple for a library target: fine for a CLI process, a real hazard for a library embedded in Python/other long-lived runtimes (leaked threads on unload, fork-unsafety). Not "dead code" but a design gap. | pool.c:95-103 (no join), no `pthread_atfork`, no shutdown API in pool.h |
| `c5d_version_string` (c5d_core.c:3) | Dead — no callers. | grep, above |
| `c5d_alloc_aligned` (platform.h:20-24) | Dead — no callers found. | grep |

---

## 3. stable.c: adapter boilerplate audit

Every `_v1` function in stable.c does some mix of these five things over the "raw" call it wraps (`c5d_brick_encode`/`c5d_brick_decode_par`/
`c5d_brick_encode_u16`/`c5d_brick_decode_u16`/`c5d_label_encode`/`c5d_label_decode`/`c5d_label_info`/`c5d_shard_*`):

1. **Argument validation that the raw call doesn't do itself** (dim%16==0 check via `dimensions()` stable.c:76-82; channel validation
   via `channels_valid()` 84-95; float `isfinite`/range checks e.g. stable.c:269,330). This is genuinely useful hardening that the
   underlying `brick.c`/`label.c` calls apparently rely on the caller for — worth keeping, but it belongs *in* the core API, not bolted
   onto a separate façade layer, per the recommendation below.
2. **Allocator indirection** (`c5d_stable_allocator`, stable.h:53-60; `allocator_value()` stable.c:41-49; used in every encode/decode
   path to allocate scratch/output buffers instead of calling `malloc` directly). Every encode function does raw-call → `malloc`
   internally (inside brick.c/label.c) then **copies the result into the caller's allocator** (`copy_encoded_buffer`, stable.c:244-261;
   also inlined in `label_decode_impl`/`decode_u8_impl`/`decode_u16_impl` for temporaries, e.g. stable.c:120,300,362) — i.e. this is not
   just a hook, it forces **one extra full memcpy of every encoded/decoded buffer** on top of what the core codec already allocates,
   purely so the stable ABI can hand back memory from a caller-chosen allocator. That's a real throughput cost (a brick/label buffer
   copy on every single encode and decode call) paid on 100% of calls to support a feature (custom allocator) that only matters to
   embedders with unusual allocation regimes (e.g. Python capsule/arena allocators).
3. **Cancellation checks** (`cancelled()` stable.c:51-54, called at entry and again after the potentially-long codec call in every
   function, e.g. 106,132,177,189,272,281,299,306,333,342,361,368,459,474,487,559,585,610). This is cooperative-only (checked at 2-3
   fixed points, not truly interruptible mid-encode) — for the single-threaded/short-lived brick/label codec workloads here (bricks are
   ≤ a few hundred^3 voxels, calls complete in milliseconds-to-low-seconds) this buys very little: cancellation can only take effect
   *before* or *after* the (uninterruptible) core call, not during it, so it can't actually abort a slow encode in progress — it can
   only skip a call that hasn't started yet or discard a result that already finished. That's a much weaker guarantee than the API
   surface (`callbacks->cancelled`) implies.
4. **Progress callbacks** (`progress()` stable.c:56-60, called with hardcoded phase-name strings and 0/1 or N/total granularity — e.g.
   `progress(callbacks, "c5l1-decode", 0u, 1u)` then `1u,1u` at stable.c:127,139 — i.e. **not real incremental progress**, just
   "started"/"finished" markers dressed up as a progress API). For per-brick shard writes it does report real incremental counts
   (`writer->written`/`writer->total`, stable.c:466,480,507) which is the one place this callback carries real information.
5. **Status-code translation**: each core function returns a bare `int` (0/-1) or writes into out-params; stable.c maps every failure
   mode to one of 9 `c5d_stable_status` values (stable.h:30-40) — mostly a flat "0 → OK, -1 → E_CORRUPT/E_INTERNAL depending on which
   call failed" translation with little real information gained (e.g. `c5d_label_decode` failing always becomes `C5D_STABLE_E_CORRUPT`
   regardless of *why* it failed, stable.c:129; `c5d_label_encode` failing always becomes `C5D_STABLE_E_INTERNAL`, stable.c:188). This
   is a real usability win for FFI consumers (Python/ctypes can't easily inspect errno-style out-params) — worth keeping in some form.

**Recommendation on which conveniences deserve the ONE public API:**

- **Status codes: keep**, but make the core API itself return a single small status enum instead of `int`/0/-1 — fold `c5d_status`
  (c5d.h) and `c5d_stable_status` (stable.h) into one enum. This removes stable.c's translation layer (§5) entirely rather than
  preserving it as an adapter.
- **Custom allocator: drop from the hot encode/decode path.** The forced extra memcpy on every call (item 2 above) is a real,
  measurable cost paid unconditionally to support a rare need. If Python/embedder integration needs allocator control, do it at the
  *buffer ownership* level (return a buffer with a matching `_free` function, like `c5d_stable_buffer_release_v1` already does) without
  requiring the *scratch/intermediate* allocations to also go through a caller hook — that's what's driving the extra copies in
  `label_decode_impl`/`decode_u8_impl`/`decode_u16_impl` (stable.c:97-145,290-314,351-376). A single `malloc`+`free` default with an
  opt-out via one global override (or none at all) covers the realistic embedding need at a fraction of the complexity.
- **Cancellation: drop, or make it genuinely cheap/no-op by default.** As implemented it only ever fires between whole codec calls, not
  during them, so it doesn't deliver what a caller would reasonably expect from "cancellation" for anything but a batch loop over many
  bricks (where the *caller* can just stop calling the next brick — no API support needed at all). Recommend removing the callback
  struct entirely and letting batch-level callers (shard packing loops) implement "stop calling" themselves, which they can already do
  today for free.
- **Progress: drop for single-brick/single-label calls** (where it's just start/end noise); **keep only** the per-item counter that
  `c5d_stable_shard_writer_put_v1`/`put_zero_v1` already provide naturally via return values — a caller writing N bricks in a loop
  already knows N and can count success returns itself, so even this doesn't need a callback, just accurate return codes.
- Net effect: `c5d_stable_allocator` and `c5d_stable_callbacks` (stable.h:53-70), and every `(void)voxels`/cancelled()/progress() call
  site they cause (roughly half of stable.c's line count — the pattern repeats near-identically 8 times across the encode/decode/shard
  functions) can be deleted if the new repo's ONE API adopts plain `malloc`/`free` and synchronous, all-or-nothing calls. What's left of
  stable.c after removing those two concerns is essentially: dim/channel/argument validation (genuinely worth keeping, but push it down
  into brick.c/label.c so it isn't duplicated at a separate layer), the shard-writer atomic-publish-via-temp-file-and-link dance
  (stable.c:385-514, a real, non-boilerplate feature worth keeping as-is), and status translation (subsumed if brick.c/label.c/shard.c
  return the unified enum directly).

---

## 4. Threading model assessment

- **Design**: one process-wide lazily-initialized pool (`g_pool`, pool.c:29) sized to `available_cpu_count()` (respects
  `sched_getaffinity` on Linux, pool.c:40-49) minus one (main/submitting thread participates, pool.c:98). Every `c5d_parallel_for` call
  from any thread, any codec (brick.c, label.c) submits into the same shared pool and gets bounded to `nthreads` workers via the
  job's `limit` field.
- **Is a global persistent pool right for a library embedded in Python/other runtimes?** No, not as implemented, for two concrete
  reasons already noted in §1: (a) **no shutdown path** — a Python module that `dlclose()`s the shared object (or a `.so` reload in a
  long-running Jupyter/notebook process) leaves N-1 detached threads spinning in `pthread_cond_wait` against a `.text` segment that may
  have just been unmapped, which is a straightforward crash/UB source, not a hypothetical; (b) **no fork-safety** — Python's
  `multiprocessing` (fork start method, still the default on Linux) or any `os.fork()`-based worker pool will silently break: a forked
  child that calls into this library gets a pool with zero live workers and a job-submission path that blocks forever waiting on
  `job.done`/`g_pool.work` condvars that nothing will ever signal (pool.c:141, no timeout). Both are the standard, well-known failure
  modes of "global lazy pthread pool" designs when used from an embedding runtime rather than a standalone CLI process, and this
  implementation has no mitigations for either (no `pthread_atfork`, no `c5d_pool_shutdown()`/refcounted teardown API).
- **Oversubscription when callers already parallelize per brick**: this is a real risk today. `tools/c5dc/pack.c:247,291` calls
  `c5d_parallel_for(nb, pack_brick_task, ...)` to parallelize *across bricks*, and each `pack_brick_task` in turn calls
  `c5d_brick_encode`/`c5d_brick_encode_u16` which **internally** call `c5d_parallel_for` again for their own sub-stream tokenize/rANS
  work (brick.c:922,952,1003,1423,1731; label.c:856,895,920,1373). Because all of these route through the *same* global pool and the
  same mutex/job-list, a caller doing `parallel_for(N_bricks, ...)` with per-brick work that itself calls `parallel_for(nsub, ...)`
  creates nested jobs on the shared pool: outer job claims some workers, each of those workers (while executing `pack_brick_task`)
  becomes a *submitter* for an inner job on the same pool, competing for the same worker set. The pool's mutex-protected job list and
  per-job `limit` field do prevent unbounded oversubscription of OS threads (there are still only ~NCPU total workers), but the nested
  submission pattern means a worker thread executing one brick's inner `parallel_for` blocks (pool.c:132-142, `pthread_cond_wait`) while
  *other* idle-looking workers from the outer job's pool might be sitting idle waiting for the outer job's next un-claimed item — the
  design handles this correctly in principle (any worker can pick up any runnable job, `find_runnable_job` scans all jobs, pool.c:63-67)
  but it does mean **deep call stacks of blocked worker threads each waiting on their own inner job's condvar**, with no priority between
  outer/inner jobs. Under this model, if `pack.c` is invoked with `threads = N = NCPU`, and each per-brick task also requests
  `nsub`-way parallelism internally, you get up to `N` outer workers each trying to additionally recruit from the same `N`-worker pool
  for their inner job — the pool correctly caps total concurrent work at ~N threads (good, no explosion), but the *effective*
  parallelism per inner call can degrade to 1 (every worker already busy in an outer task, so `find_runnable_job` picks up items
  serially as workers free up) without any signal to the caller that nesting is happening. In short: no correctness bug, no thread
  explosion, but a nested-parallelism scheduling design that's opaque and can silently serialize inner loops under contention, with no
  metrics exposed to observe it.
- **Concrete recommendation for the new repo**: replace the always-on global background pool with either (a) an explicit,
  caller-created/destroyed pool handle (`c5d_pool_create(nthreads)` / `c5d_pool_destroy()`, passed explicitly into encode/decode calls)
  so an embedding runtime controls its lifetime and can tear it down before unload/fork, or (b) drop the internal thread pool from the
  library entirely and make the **single public API single-threaded by default, with the caller responsible for parallelizing across
  bricks** (which `pack.c`'s own usage pattern shows is the natural unit of parallelism for this workload anyway — bricks are
  independent). Given that per-brick parallelism (across bricks, via the caller's own thread pool or Python's
  `concurrent.futures`/`multiprocessing`) already covers the embarrassingly-parallel structure of this workload, the *intra-brick*
  `nsub`-way parallelism in brick.c/label.c (which is what actually needs a low-latency pool) is the only piece that plausibly still
  benefits from an internal pool — and even that could be sized/owned explicitly per top-level call rather than globally. Recommend (b)
  as the default for a minimized library API (simplest, safest for embedding, no fork/unload hazards, no oversubscription risk) with
  `nthreads` on the public encode/decode calls used only for the intra-brick tokenize/rANS parallel sections, and document clearly that
  callers parallelizing across many bricks should pass `nthreads=1` and do their own outer-loop parallelism.

---

## 5. Minimal-repo proposal: what survives, in what form

| Module | Verdict | Form in new repo |
|---|---|---|
| **shard.c/h** | **Keep, trim.** Real, load-bearing container format with genuine crash-recovery value and two real callers (`pack.c`, `stable.c`). | Keep as its own small module. Fixes to make while porting: (1) add x86 CRC32 hardware path alongside the ARM path (shard.c:13-49) since x86 is a realistic target; (2) bound/short-circuit the crash-recovery backward scan (shard.c:181-215) so a hostile/huge file with many false magic hits can't cause slow-open; (3) make the fsync-per-64MB snapshot cadence a parameter instead of a hardcoded constant (shard.c:57) so batch/scratch callers can disable it; (4) fix the `c5d_shard_brick` NULL-reader inconsistency (shard.c:220 vs 233,241); (5) fold its status reporting into whatever single status enum the new repo settles on instead of bare `int`. |
| **cache.c/h** | **Drop.** Zero production callers found anywhere (grep confirmed above); it duplicates policy that belongs at the *application* layer (an S3-backed remote client would want a cache, but `remote.c` doesn't use this one and implements its own on-demand fetch instead), and as written has a real correctness gap (destroy-while-pinned use-after-free, §1) that would need fixing before it's trustworthy. If a future S3/remote client genuinely needs a decoded-brick cache, it should be rebuilt as part of *that* client (or reconsidered as a much simpler bounded LRU without the ghost-ring/S3-FIFO machinery) rather than carried forward as unused library surface. | Delete. |
| **pool.c/h** | **Keep the parallel-for primitive, redesign the lifetime model.** The core work-stealing-ish parallel_for loop (claim-next-item under a mutex, participate-while-waiting) is simple and correct and is genuinely used by brick.c/label.c's intra-brick parallelism. | Keep the algorithm; replace the implicit global `atexit`-less singleton with an explicit pool object the top-level API owns (created once per process by the caller, or created/destroyed per top-level call if simplicity is preferred over pool-reuse overhead) so there is a real shutdown point and so `pthread_atfork` handling (or "forking with this library loaded is unsupported, document it") can be reasoned about. Given today's usage, a **fully explicit, caller-visible pool handle** matches "minimize API + best speed" better than either the current global singleton or a fresh-thread-per-call design (which would add creation overhead per brick). |
| **platform.h** | **Keep, trim.** `c5d_now_ns`, `c5d_mul_size`, `c5d_add_size` are used and cheap. | Keep those three. Drop `c5d_alloc_aligned` (no callers found) unless a specific SIMD-alignment need is identified during the brick/label port; if kept, make it overflow-checked like its neighbors. Add a non-Linux fallback for `CLOCK_MONOTONIC_RAW` if portability beyond Linux is a goal, else document Linux-only. |
| **c5d_core.c** | **Drop or fold in.** Two trivial functions; `c5d_version_string` is unused, `c5d_status_string` has one caller (a test). | Fold `c5d_status_string`-equivalent into the single unified status enum's stringify function (replacing both this and `c5d_stable_status_string`); add a version string only if the new repo actually wants a `--version`/`c5d_version()` surface — don't carry the current one forward unchanged since it's dead. |
| **stable.c/h** | **Keep the *shape*, gut the implementation.** The shard-writer atomic-publish-via-tempfile+link logic (stable.c:385-514) and the input-validation logic (`dimensions`, `channels_valid`, brick_index) are real value and should become the *actual* public API's validation, not a wrapper's. | Collapse `stable.c` and the "core" brick/label/shard APIs into one layer: the single public API validates its own arguments, returns one status enum, uses plain malloc/free (see §3), and drops the allocator-hook/cancellation/progress-callback machinery. This eliminates the separate "internal API" + "stable façade" duplication entirely — there is only ever one API in the new repo, which was the whole point of building `stable.c` as a façade over an unstable internal API that no longer needs to exist once there's nothing to keep stable *against*. |
| **tests/test_stable.c, tests/test_hostile.c** | **Keep, adapt.** Both are good-quality regression tests: `test_hostile.c` specifically encodes known adversarial-input findings (NaN/Inf quality, hostile nsub, LEB128 shift-overflow, shard offset overflow, truncated-shard crash recovery, unaligned brick blobs) and should be preserved as-is (just re-pointed at the new unified API) since they're testing real hardening work, not façade plumbing. `test_stable.c` mostly exercises the façade itself (allocator/cancel plumbing) — its allocator/cancel-specific assertions (e.g. lines 74-78 testing `cancel_now`) can be dropped along with those features per §3, but its round-trip/shard/finish-abort coverage should carry forward. | Port hostile-input cases 1:1 onto the merged API; trim cancellation-callback assertions from the stable test; keep everything else. |

---

## 10-line summary

1. `cache.c`/`cache.h` (282 lines total) have **zero production callers** anywhere in src/tools/tests/fuzz — confirmed by exhaustive grep; only their own smoke test in `test_quick.c` uses them. Drop entirely from the new repo.
2. `shard.c` is real and load-bearing (used by `pack.c` and `stable.c`); its 64 MB fsync'd crash-recovery snapshots are functionally justified and tested (`test_hostile.c:147-184`), but the backward magic-scan on open (shard.c:181-215) is an unbounded-cost path on hostile/huge files and should get a cost bound before shipping against untrusted input; crc32c also lacks an x86 hardware path (only ARM64 is accelerated, shard.c:13-49).
3. `c5d_cache_destroy` frees entries even if `pins != 0` — a real destroy-while-pinned use-after-free hazard, currently unreachable only because nothing calls the cache in production.
4. `stable.c` (619 lines) is roughly half genuine validation/atomic-publish logic worth keeping and half boilerplate (allocator indirection, cancellation, progress) that should be deleted per §3 — the allocator hook alone forces an extra memcpy of every encoded/decoded buffer on 100% of calls to support a rarely-needed feature.
5. Cancellation callbacks (stable.c `cancelled()`) only ever fire *between* whole codec calls, never during one — they don't deliver real mid-encode interruption and can be replaced by "just stop calling the next item" at the batch-loop level for free.
6. `pool.c`'s global lazily-spun-up thread pool has no shutdown path (threads never join, no atexit) and no `pthread_atfork` handling — both are real hazards for a library meant to be embedded in Python/other long-lived or fork-using runtimes, per stable.h's own stated dlopen/ctypes goal.
7. Nested `c5d_parallel_for` calls (outer per-brick loop in `pack.c`, inner per-substream loop inside `brick.c`/`label.c`, both hitting the same global pool) don't oversubscribe OS threads but can silently serialize inner-loop parallelism under contention with no visibility into it.
8. `nthreads==0` means "all CPUs" and `nthreads<=1` means "run synchronously, no pool interaction" consistently across pool.c and all its callers — this convention is solid and worth keeping unchanged.
9. `c5d_version_string` (c5d_core.c) and `c5d_alloc_aligned` (platform.h) both have zero callers anywhere; two parallel status-string functions (`c5d_status_string` vs `c5d_stable_status_string`) exist for two parallel status enums that should collapse into one.
10. Recommended shape for the new repo: keep shard.c (trimmed/hardened) and pool.c's algorithm (with an explicit, owned pool handle instead of a global singleton), drop cache.c entirely, and merge stable.c's validation/shard-writer logic directly into the single public API rather than keeping it as a separate façade over an "unstable" inner layer that no longer needs to exist.


<!-- END 05-infra.md -->


---

<!-- BEGIN 06-baseline-measurements.md -->

# c5d baseline measurements (2026-09-01)

Environment: WSL2, Linux 6.18.33.2-microsoft-standard-WSL2, Intel Core Ultra 9
275HX, 24 logical CPUs. clang 23.0.0 (Ubuntu build), cmake 4.2.3, ninja,
ccache, glslc present, Vulkan 1.4.341 (llvmpipe/lavapipe software + host
drivers registered). `perf` present but hardware PMU counters are NOT
available in this environment (`perf stat -e cycles` -> "No supported
events"); software `cpu-clock` sampling works and was used instead.

Repo: `/home/forrest/c5d` @ commit-less checkout (git status snapshot from
the calling session showed clean `main`, single "Initial commit"). Nothing
under `/home/forrest/c5d` or `/home/forrest/volume-compressor` was modified.
Build tree: `/tmp/.../scratchpad/build-bench` (CPU/bench config) and
`/tmp/.../scratchpad/build-gpu` (GPU build attempt), both out-of-tree,
neither touches `/home/forrest/c5d/build`.

## 1. Build and ctest

Configured to replicate `CMakePresets.json`'s `bench` preset exactly, with
`-B` redirected to the scratchpad:

```
cmake -S /home/forrest/c5d -B <scratch>/build-bench -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DC5D_MCPU=native -DC5D_HARDENING=OFF -DC5D_BRANCH_PROTECTION=OFF
cmake --build <scratch>/build-bench -j24 -- -k 999
```

**Result: 13/14 targets built clean with 0 compiler warnings.** `grep -c
warning: build.log` = 0.

**`test_stable` fails to build** under this Release/-Werror/-DNDEBUG
configuration — 14 `-Werror` unused-variable/unused-but-set-variable errors
in `tests/test_stable.c`. Root cause: the test's control flow lives entirely
inside `assert(...)` calls (`assert(c5d_stable_brick_encode_u8_v1(...) ==
C5D_STABLE_OK)`, etc.); Release defines `NDEBUG`, which makes `assert()` a
no-op, so every local only referenced inside an assert becomes a genuinely
unused variable and `-Werror -Wunused-variable` (from the project's own
`c5d_warnings` flags: `-Wall -Wextra -Wconversion -Wshadow
-Wimplicit-fallthrough -Wformat=2 -Wvla -Wdouble-promotion -Werror`) kills the
build. This is **not** a sanitizer/hardening-only test — it reproduces on the
plain `bench`/`release` preset combination (Release + `-Werror`), so anyone
building the `release` preset hits the same failure. It is an
existing, unmodified-by-us bug in the checked-out tree — flagged here per
instructions not to touch repo files. Everything else (including the 13
other targets, the full `c5d` static lib, `c5d-bakeoff`, `c5dc`,
`libc5d_stable.so`) built with zero warnings and zero other errors.

`ctest -L quick` (excluding the unbuildable `stable` target — CTest can't
run a test whose binary doesn't exist):

```
1/6 quick     Passed   0.01 sec
2/6 tifxyz    Passed   0.16 sec
3/6 label     Passed   2.61 sec
4/6 golden    Passed   0.01 sec
5/6 hostile   Passed   0.00 sec
6/6 xdec      Passed   0.01 sec
100% tests passed, 0 tests failed out of 6   (total 2.81 sec)
```

`stable` (7th "quick"-labeled test) could not run because it does not build
— see above.

### GPU build (C5D_GPU=ON)

Vulkan headers/loader (`libvulkan1`, `libvulkan-dev`, `mesa-vulkan-drivers`,
`vulkan-tools` 1.4.341) and `glslc` are present. Configuring and building
with `-DC5D_GPU=ON` (otherwise identical bench flags) **succeeded trivially**
— all 10 compute shaders compiled to SPIR-V, `libc5d_gpu.a`,
`c5d-gputest` and every other target linked with 0 warnings, except the same
pre-existing `test_stable.c` failure (identical 14 errors, unrelated to GPU).
`c5d-gputest` itself could not be exercised: it hard-codes `corpus/dev` as
its input and that corpus is not fetched in this checkout (see below), so it
immediately exits with `cannot open corpus dir corpus/dev`. GPU
encode/decode numbers were therefore **not** collected — the build is fine,
there is simply no local data to feed it, and writing a GPU-path driver was
out of scope for this pass.

## 2. Corpora available locally

- **`corpus/manifest_dev.json` / `manifest_full.json`**: manifests only —
  list of Vesuvius Challenge scroll CT `.zarr` chunk coordinates on S3
  (`s3://vesuvius-challenge-open-data/...`), with per-chunk `class`,
  `mean`, `std`, `air_frac`, `entropy_bits` metadata. **No voxel payloads are
  present** — `corpus/dev` and `corpus/full` directories that `c5d-bakeoff`
  expects (`.u8` + `.json` sidecar pairs) do not exist on disk, and nothing
  was downloaded (as instructed).
- **`corpus/synth/*.json`** (4 files: `synth_air_0`, `synth_dense_0`,
  `synth_smooth_0`, `synth_surface_0`): also metadata stubs only
  (`{"id":..., "class":..., "synthetic": true}`), **no matching `.u8`
  files**. `tools/bakeoff/corpus.c` only ever looks for `<id>.u8` files and
  reads `<id>.json` purely for the `class` string — with no `.u8` files,
  `corpus_load("corpus/synth")` returns "no bricks found" and the harness
  cannot run against this directory as-is. There is no generator script in
  the repo (`tools/fetch_corpus.py` has no reference to `synth`) that
  materializes these stubs into voxel data.
- **`tests/golden/src48.u8`**: a single 48^3 (110592-byte) u8 volume used by
  the golden regression test (`tests/test_golden.c`), plus its reference
  encoded/decoded artifacts (`lossless.c5b`, `lossy_q2*.c5b`, `tau4_q4.c5b`,
  etc.). This is real fixed test content (not synthetic noise) but is a
  single small (48^3, i.e. 3x3x3 chunks) sample — not representative of
  production 128^3 bricks and too small alone for throughput/latency
  statistics.
- **`docs/BENCHMARKS.md` / `docs/measured.md`**: contain **prior** benchmark
  numbers against the real downloaded corpora (`corpus/dev` = 24 bricks,
  class-balanced but not scroll-representative; `corpus/full` = 512 bricks
  from PHercParis4, described there as "the scroll-representative
  headline"), run on different hardware (Snapdragon X Elite / Ryzen 7940HS).
  These are **not reproduced here** (no local data) but are cited below for
  context/sanity-checking our synthetic numbers.

**Because no real corpus is present, a small procedural stand-in corpus was
generated *outside the repo*** (`<scratch>/gen_synth.c` ->
`<scratch>/corpus_synth128/`, 16 bricks: 4 each of `air`, `dense_interior`,
`smooth_papyrus`, `surface_boundary`, 128^3 u8, ~33 MiB total) to exercise
the harness meaningfully. It is a deliberately simple stand-in: smooth
trilinearly-upsampled control-point fields plus class-appropriate noise/
periodic "fiber" texture and a soft air/material interface for
`surface_boundary`, tuned only to land in the right ballpark of the
manifest's own classifier thresholds (air_frac, std, entropy). **This is
explicitly not real scroll CT data** — no fiber microstructure, no scanner
noise correlations, no real material physics — so absolute ratio/PSNR/SSIM
numbers below should be read as *harness-and-relative-option* signal only,
not as production quality numbers. The golden 48^3 real fixture was also run
through the harness separately as one authentic (if small and single-sample)
data point. Generator source kept at `<scratch>/gen_synth.c` for
reproducibility; not part of the deliverable repo.

## 3. Bakeoff sweep — synthetic 128^3 corpus (16 bricks), codecs x quality x threads

Command pattern:
```
c5d-bakeoff --corpus=<scratch>/corpus_synth128 --codec=<c5d1|c5d0|c5dwav> \
  --quality=<Q> --threads=<0|1> --reps=3 --no-ledger
```
`c5d1` = production brick codec (`src/brick.c`, NSUB substreams, deblock,
tau/RDO/ctx2/eprior/rans_nway knobs). `c5d0` = the older/simpler `codec_v0.c`
whole-brick DCT-16+dead-zone+rANS path exposed for comparison. `c5dwav` =
a wavelet baseline. `zstd` = generic lossless baseline (linked; `zfp` is not
linked in this build, so it's absent from the registry). "ALL" rows are
brick-count-weighted aggregates across the 4 classes (see full per-class
breakdown in `<scratch>/results/sweep_main.txt`).

### threads=0 (auto, all 24 cores)

| codec | q | ratio | PSNR dB | SSIM | MAE | P90/P95/P99/max err | enc MB/s | dec MB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c5d1 | 0.5 | 6.13 | 45.66 | .9807 | 1.015 | 2/3/3/8 | 594 | 603 |
| c5d1 | 1 | 11.80 | 41.66 | .9639 | 1.605 | 3/4/6/16 | 782 | 702 |
| c5d1 | 2 | 24.93 | 37.42 | .9373 | 2.309 | 5/8/12/34 | 941 | 793 |
| c5d1 | 4 | 87.95 | 34.42 | .8869 | 3.012 | 8/12/17/36 | 1080 | 909 |
| c5d1 | 8 | 241.5 | 33.81 | .8642 | 3.244 | 10/13/17/40 | 1062 | 885 |
| c5d1 | lossless | 2.07 | inf | 1.0000 | 0 | 0/0/0/0 | 393 | 521 |
| c5d0 | 0.5 | 6.14 | 45.67 | .9807 | 1.014 | 2/3/3/8 | 132 | 176 |
| c5d0 | 1 | 11.82 | 41.66 | .9639 | 1.605 | 3/4/6/16 | 198 | 291 |
| c5d0 | 2 | 25.02 | 37.42 | .9373 | 2.309 | 5/8/12/34 | 313 | 457 |
| c5d0 | 4 | 89.16 | 34.43 | .8871 | 3.010 | 8/12/17/36 | 529 | 770 |
| c5d0 | 8 | 252.0 | 33.80 | .8648 | 3.249 | 10/13/17/40 | 596 | 925 |
| c5dwav | 0.5 | 1.73 | 56.86 | .9982 | 0.134 | 1/1/1/2 | 19 | 17 |
| c5dwav | 1 | 2.27 | 51.11 | .9930 | 0.454 | 1/1/2/5 | 20 | 20 |
| c5dwav | 2 | 3.32 | 46.34 | .9842 | 0.899 | 2/2/3/9 | 21 | 21 |
| c5dwav | 4 | 5.61 | 40.94 | .9304 | 1.799 | 4/4/6/18 | 23 | 24 |
| c5dwav | 8 | 10.73 | 35.92 | .8588 | 3.105 | 7/8/11/35 | 26 | 28 |
| zstd (lossless, level 3) | - | 1.83 | inf | 1.0000 | 0 | 0/0/0/0 | 236 | 1135 |

**Observations (synthetic data):**
- `c5d1` and `c5d0` are essentially bit-identical in ratio/PSNR/error stats
  at every quality (same math, same entropy coder), as expected — `c5d1`'s
  win is purely operational: parallel decode (`nthreads`), substreams,
  deblock, tau/RDO knobs. At `threads=0` `c5d1`'s multi-substream design
  gives it 1.6-3.5x the encode throughput and ~1.0-1.3x the decode
  throughput of `c5d0`'s single-stream path — the gap narrows at high Q
  because entropy coding dominates less relative to transform cost.
- `c5d1` scales from ~600 MB/s (q0.5) to ~1.0-1.1 GB/s (q4-8) encode and
  ~600-900 MB/s decode with auto-threading over 24 cores on this synthetic
  set. This is well below the 2-3+ GB/s auto-thread numbers in
  `docs/BENCHMARKS.md` for the real 512-brick corpus on a 8-core 7940HS —
  expected, since 16 small bricks give poor thread-pool amortization
  compared to hundreds of bricks, and this is a different CPU/microarch.
- `c5dwav` is dramatically slower (17-28 MB/s, i.e. ~30-60x slower than
  `c5d1`) but noticeably higher quality per bit at matched `--quality`
  value (e.g. q2: c5dwav PSNR 46.3 dB / ratio 3.3x vs c5d1 PSNR 37.4 dB /
  ratio 24.9x) — the `--quality` knob is not calibrated the same way across
  codecs, so this is not an apples-to-apples RD comparison, just a
  throughput/latency data point. `c5dwav` is far too slow (~0.1-0.15s per
  128^3 brick) to be a serious production candidate at these settings.
  zstd (byte-lossless, no transform) gets only 1.83x on this content and is
  much faster to decode (1.1 GB/s) than to encode (236 MB/s) — sanity
  baseline, not a fair comparison to the lossy codecs.
- c5d1 lossless mode: 2.07x ratio (this synthetic content has real per-voxel
  noise so lossless headroom is limited — real scroll CT with more spatial
  coherence would compress further), 393/521 MB/s enc/dec auto-thread.
- Ratio grows faster than error metrics degrade from q1->q8 (11.8x->241x
  ratio, MAE only 1.6->3.2, PSNR 41.7->33.8 dB) — the dead-zone quantizer is
  doing its job on this smooth-ish synthetic content; real scroll interior
  texture (higher local entropy) would show a shallower ratio/quality
  tradeoff, consistent with the real-corpus numbers in `docs/BENCHMARKS.md`
  showing q2 at 24.16x/42.9dB and q8 at 72.7x/37.7dB (i.e. real content's
  ratio grows much more slowly with q than our synthetic set's does —
  confirms our synthetic corpus is smoother/more compressible than real
  scroll interior on average, as expected from a simple procedural
  generator).

### threads=1 (serial) vs threads=0 (auto), c5d1

| q | enc MB/s (1T) | enc MB/s (auto) | speedup | dec MB/s (1T) | dec MB/s (auto) | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 170 | 594 | 3.5x | 189 | 603 | 3.2x |
| 1 | 254 | 782 | 3.1x | 305 | 702 | 2.3x |
| 2 | 357 | 941 | 2.6x | 466 | 793 | 1.7x |
| 4 | 467 | 1080 | 2.3x | 619 | 909 | 1.5x |
| 8 | 591 | 1062 | 1.8x | 1017 | 885 | 0.9x |
| lossless | 73 | 393 | 5.4x | 83 | 521 | 6.3x |

Speedup from auto-threading (24 cores available, but only 16 bricks and
`nthreads` also parallelizes *within* a brick across substreams) tops out
around 3-6x, well short of 24x — expected given only 16 bricks total and
per-brick thread-pool spin-up overhead dominating at this small a working
set; a production run over hundreds of bricks would show better scaling
(the `docs/BENCHMARKS.md` GB/s-class real-corpus numbers corroborate this).
At q8 decode, single-thread is actually *faster* than "auto" median (likely
noise/threading overhead crossover on nearly-degenerate 34x-ratio streams —
not a reliable signal, see reps note below).

**Noise caveat:** throughput numbers above have run-to-run variance in the
±10-20% range on this WSL2/shared-host environment (background load,
frequency scaling, no isolated core pinning) — treat single-run MB/s deltas
smaller than ~15% as not meaningfully different. Latency percentiles (below)
were more stable since they pool `reps x bricks` samples.

Full per-class breakdown, latency percentiles, and blocking-gradient rows
for every (codec, q, threads) combination above:
`<scratch>/results/sweep_main.txt`.

## 4. Option-toggle sweep (`--set=k=v`), c5d1 @ q=2, threads=1, synthetic corpus

Baseline (defaults, q2, 1T): ratio 24.93, PSNR 37.42 dB, SSIM .9373,
MAE 2.309, P90/95/99/max 5/8/12/34, enc 341 MB/s, dec 452 MB/s.

| toggle | ratio | PSNR | SSIM | MAE | max err | enc MB/s | dec MB/s | delta vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `ctx2=1` (default) | 24.90 | 37.42 | .9373 | 2.309 | 34 | 343 | 417 | ~identical (already default) |
| `ctx2=0` | 24.93 | 37.42 | .9373 | 2.309 | 34 | 323 | 431 | no ratio/quality change; within noise on speed |
| `rans_nway=1` | 24.97 | 37.42 | .9373 | 2.309 | 34 | 327 | 404 | no measurable effect at 1T on this corpus |
| `rans_nway=2` | 24.93 | 37.42 | .9373 | 2.309 | 34 | 362 | 419 | no measurable effect at 1T |
| `rans_nway=4` | 24.87 | 37.42 | .9373 | 2.309 | 34 | 362 | 409 | no measurable effect at 1T (nway lanes help SIMD/ILP more at higher thread/data volume than tested here) |
| `eprior=1` (force on) | 23.43 | 37.42 | .9373 | 2.309 | 34 | 333 | 455 | **ratio -6.0%** (embedded priors cost bits on this synthetic content's statistics; same quality) |
| `eprior=-1` (force off) | 24.93 | 37.42 | .9373 | 2.309 | 34 | 359 | 440 | == default (auto already chose off here) |
| `rdo=0.15` | 27.03 | 37.20 | .9343 | 2.354 | 31 | 116 | 415 | **ratio +8.4%**, PSNR -0.22 dB, **enc MB/s -66%** (RDO zero-out trades a lot of encode time for a modest ratio gain here) |
| `deblock=0` | 24.93 | 37.42 | .9373 | 2.309 | 34 | 303 | 404 | ratio/PSNR unchanged (deblock is decode-side only, doesn't touch the bitstream); blocking-gradient MAE unchanged too on this synthetic set (0.8822x amp both ways) — the deblock filter isn't doing much visible work on this smooth synthetic content |
| `nsub=8` | 25.08 | 37.42 | .9373 | 2.309 | 34 | 269 | 350 | ratio +0.6% (fewer substream flush overheads), throughput down (fewer substreams -> less within-brick parallel decode, though at 1T that shouldn't matter — likely noise) |
| `nsub=32` (default) | 24.93 | 37.42 | .9373 | 2.309 | 34 | 271 | 315 | baseline |
| `nsub=128` | 24.38 | 37.42 | .9373 | 2.309 | 34 | 300 | 347 | ratio **-2.2%** (more substream flush overhead: ~20 B/substream x 96 extra substreams on a 2 MB brick ≈ 2 KB, matches the observed ratio loss) |
| `tau=4` | 3.36 | 43.32 | .9667 | 1.314 | **4** | 100 | 319 | hard max-error bound: caps worst-case error at 4 (vs 34 default) via sparse correction layer, at a **large ratio cost (24.9x -> 3.4x, -87%)** and **enc MB/s -71%** — correction layer is expensive to build and bulky on this noisy synthetic content; PSNR/SSIM actually improve since large outliers are corrected |

Full raw output: `<scratch>/results/sweep_toggles.txt`.

**Interpretation of the toggles:** on this synthetic corpus the biggest
levers are `tau` (guarantees a max-error bound at real cost — worth keeping
as an opt-in correctness knob, not a default), `rdo` (real ratio/PSNR
tradeoff, meaningfully slower encode — worth it only where encode time is
cheap relative to storage), `eprior` (content-dependent; the harness's
auto/cost-based selection already made the right call here, so `eprior`
auto is validated at least on this content), and `nsub` (default 32 looks
like a reasonable knee — 8 doesn't meaningfully help ratio, 128 measurably
hurts it, consistent with the per-substream flush overhead documented in
`src/brick.h`). `ctx2` and `rans_nway` showed no measurable effect in this
single-threaded microbenchmark; they're primarily encode-parallelism /
context-modeling knobs whose benefit would show up more at higher thread
counts or on more compressible content — not falsified here, just not
exercised meaningfully by this test.

## 5. Real (golden) 48^3 fixture, single brick

`c5d-bakeoff --corpus=<scratch>/corpus_golden --codec=c5d1,c5d0,c5dwav,zstd --quality=2 --threads=0 --reps=5`
(`corpus_golden` = `tests/golden/src48.u8` copied out of the repo tree with a
sidecar json; 1 brick, 48^3 = 110592 bytes):

| codec | ratio | PSNR | SSIM | MAE | max err | enc MB/s | dec MB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| c5d1 | 20.73 | 34.89 | .9880 | 3.810 | 16 | 100 | 122 |
| c5d0 | 23.88 | 34.90 | .9880 | 3.807 | 16 | 309 | 476 |
| c5dwav | 2.08 | 45.69 | .9992 | 1.001 | 7 | 28 | 28 |
| zstd | 1.17 | inf | 1.0000 | 0 | 0 | 1223 | 1217 |

Single small (48^3, 27 chunks) brick: `c5d0`'s single-stream simplicity beats
`c5d1`'s substream-parallel design on both ratio (less substream overhead:
23.88x vs 20.73x, ~15% more compact) and throughput here, because there
isn't enough work to amortize `c5d1`'s parallelism machinery — the opposite
of what the 128^3 sweep showed. This is a useful reminder that `c5d1`'s
substream/thread design is tuned for 128^3-scale bricks, not small ones.

## 6. Hot-path profile (`perf record -e cpu-clock -F 999 -g`, software sampling — no HW PMU available)

Combined encode+decode loop, c5d1 @ q2, 1 thread, 30 reps over the 16-brick
synthetic corpus (7373 samples, `--no-ssim` to keep SSIM's own cost out of
the profile):

| % time | symbol | phase |
|---:|---|---|
| 22.9% | `tokenize_sub_task` | encode: per-substream tokenization (levels -> symbols) |
| 18.4% | `decode_sub_task` | decode: per-substream rANS decode + detokenize |
| 14.0% | `pass_axis_gs.llvm...` | encode: forward transform pass (axis sweep) |
| 13.2% | `inv16_vert` | decode: inverse DCT-16 vertical pass |
| 10.6% | `fwd16_vert` | encode: forward DCT-16 vertical pass |
| 9.9% | `rans_encode_multin` | encode: multi-lane rANS entropy coding |
| 5.2% | `deblock_face_task` | decode: seam deblock filter |
| 1.8% | `scatter_chunk` | decode: chunk scatter/reassembly |
| 1.3% | `__memset_avx2_unaligned_erms` | misc buffer zeroing |
| 0.9% | `metric_blocking_u8` | harness metrics (not codec) |
| 0.8% | `main` | harness loop overhead |
| 0.3% | `rans_model_build` | encode: entropy model construction |
| <0.3% each | `__powf_fma`, `_int_malloc`, `__memmove_avx_unaligned_erms`, `brick_encode_impl`, `step_table`, `__log2_fma`, `c5d_brick_decode_par`, `malloc` | negligible |

Top-20 raw: `<scratch>/results/perf_top.txt`. **Interpretation:** tokenization
and entropy (de)coding (`tokenize_sub_task` + `decode_sub_task` +
`rans_encode_multin` + `rans_model_build` ≈ 51% combined) dominate over the
DCT-16 transform itself (`pass_axis_gs` + `inv16_vert` + `fwd16_vert` ≈ 38%
combined) — the entropy path is the bigger optimization target by raw
sample count, followed closely by the transform. Deblock is a modest but
non-trivial 5% of decode-side time. This profile mixes encode and decode
samples in one run (a quick isolated driver wasn't built for the profiler
pass given time budget) — symbol names make the encode/decode split visible
per-line above, so treat the percentages as "share of combined enc+dec
wall-clock," not "share of decode alone."

## 7. Single-128^3-brick decode latency (dedicated driver, not corpus-bound)

Wrote `<scratch>/brick_latency.c`, linked directly against the bench build's
`libc5d.a` (`clang -O3 -march=native -flto=thin ... libc5d.a` — the archive
is ThinLTO bitcode, so `-flto=thin` is required at link time too). One
synthetic 128^3 brick (independently generated, same style as the corpus
generator), encoded once at q=2 (ratio 8.83x), then decoded 200 times per
thread setting with `c5d_brick_decode_par`:

| threads | dec latency p50 | p90 | p95 | p99 | max | throughput (p50) |
|---|---:|---:|---:|---:|---:|---:|
| 1 (serial) | 8265.5 us | 10925.0 us | 11355.7 us | 12409.9 us | 12926.8 us | 253.7 MB/s |
| 0 (auto, 24 cores) | 3394.7 us | 4195.7 us | 4495.5 us | 4901.5 us | 6136.1 us | 617.8 MB/s |

Auto-threading a single brick's internal substreams gives ~2.4x median
throughput here (24 cores available but only ~32 substreams to spread
across, plus thread-pool wake/join overhead per call — consistent with the
sub-linear scaling seen in section 3). Single-brick decode latency floor on
this host is ~3.4 ms with auto threads, ~8.3 ms serial, for a 2 MiB brick.

## 8. Binary / library / priors sizes

| artifact | size |
|---|---:|
| `libc5d.a` (static lib, all codecs) | 429,714 B (420 KiB) |
| `c5d-bakeoff` (unstripped) | 167,808 B |
| `c5d-bakeoff` (stripped) | 158,504 B |
| `c5dc` CLI (unstripped) | 198,712 B |
| `c5dc` CLI (stripped) | 187,296 B |
| `libc5d_stable.so` (stable ABI shared lib) | 179,024 B |
| `c5d-label-bench` | 139,160 B |
| `c5d-train-priors` | 63,784 B |
| `src/entropy/priors_v1.bin` | 640 B |
| `src/entropy/priors_v2.bin` | 1,216 B |

Priors blobs are tiny (under 2 KiB combined) — embedded-priors mode
(`eprior`) is not a meaningful binary-size cost, only a per-stream bit-cost
tradeoff (see section 4).

## 10-line summary

1. Build is clean: 0 compiler warnings across the whole tree under the exact
   `bench` preset flags (`-Werror` with `-Wall -Wextra -Wconversion -Wshadow
   ... `).
2. One pre-existing, unrelated bug blocks a full build/ctest: `tests/test_stable.c`
   fails 14x under `-Werror -Wunused-variable` in Release (`NDEBUG` strips
   its `assert()`-only control flow) — reproduces on plain `release`/`bench`
   presets, not just this rig; not fixed since repo files must stay
   untouched.
3. All other quick tests pass: `quick`, `tifxyz`, `label`, `golden`,
   `hostile`, `xdec` — 6/6, 2.81 s total.
4. GPU build (`-DC5D_GPU=ON`) compiles cleanly (Vulkan + glslc both present)
   but could not be exercised: `c5d-gputest` hard-requires `corpus/dev`,
   which isn't downloaded here.
5. No real corpus data exists locally: `corpus/dev`/`corpus/full` were never
   fetched, and `corpus/synth/*.json` are metadata-only stubs with no
   matching `.u8` payloads — the bakeoff harness cannot run against either
   as shipped.
6. A procedural 16-brick (4 classes x 4, 128^3) stand-in corpus was
   generated out-of-tree to exercise the pipeline; treat its absolute
   ratio/PSNR/SSIM numbers as harness-and-relative-option signal only, not
   as production quality numbers — it's visibly more compressible than the
   real corpus numbers recorded in `docs/BENCHMARKS.md` at matched quality.
7. `c5d1` vs `c5d0` are ratio/quality-identical at every setting (same
   math); `c5d1` wins on throughput for 128^3 bricks (up to ~3.5x encode)
   via substreams+threading, but *loses* to `c5d0` on a single small 48^3
   brick (more substream overhead than payoff at that scale).
8. `c5dwav` is far higher quality per `--quality` unit but 30-60x slower
   than `c5d1` — not currently competitive on throughput.
9. Option-toggle sweep at q2/1T: `tau` and `rdo` have real, large effects
   (max-error bound at -87% ratio cost; +8.4% ratio at -66% encode speed,
   respectively); `nsub` default (32) looks like a sane knee (128 measurably
   hurts ratio via substream flush overhead); `ctx2`/`rans_nway` showed no
   measurable effect at 1 thread on this small corpus (needs more
   threads/data to show their benefit); `eprior` auto already matches the
   better of forced on/off on this content.
10. Hot path (combined enc+dec, software `cpu-clock` sampling — no HW PMU
    available here): tokenization+entropy coding (~51%) dominates over the
    DCT-16 transform (~38%); deblock is ~5%. Single 128^3-brick decode
    latency floor: ~8.3 ms serial / ~3.4 ms with auto-threading (median).
    Priors blobs are trivially small (<2 KiB); stripped binaries are all
    well under 200 KiB.


<!-- END 06-baseline-measurements.md -->


---

<!-- BEGIN 07-tools-tests-docs.md -->

# c5d Audit: Tools, Tests, Docs — Triage for volume-compressor

**Task**: Audit everything outside `src/` in the c5d repo (READ-ONLY) to decide what carries forward to the cleaned-up `volume-compressor` repo. Scope: tools/, tests/, fuzz/, bench/, corpus/, docs/, spec/, config files.

---

## 1. FILE-BY-FILE TABLE

| Path | Size/Lines | Purpose | Dependencies | Status | Reason |
|------|-----------|---------|---|--------|--------|
| **tools/c5dc/main.c** | 173L | Subcommand dispatcher; slice/diff (PGM export) | brick, shard, tifxyz, label | **REWRITE** | Keep encode/decode/info/verify only; drop remote/remote-batch, pack, slice, diff, tifxyz-*, label-* (move label to separate tool or cut) |
| **tools/c5dc/pack.c** | 398L | Build LOD pyramids from 8×8×8 brick grids | brick, shard, pool | **DROP** | Niche tool for assembling full-scale regioned volumes; assume users have data pre-split into shards |
| **tools/c5dc/remote.c** | 415L | S3-remote single/batch brick decode via curl | brick, shard, curl | **DROP** | S3 batch fetch is site-specific; users will integrate into their workflows |
| **tools/c5dc/tifxyz_cmd.c** | 148L | Surface parameterization (3D point cloud) pack/unpack/verify | tifxyz | **DROP** | Scope creep; not a volume codec operation |
| **tools/c5dc/label_cmd.c** | 290L | Multi-channel lossless label brick pack/unpack/info/verify | label | **KEEP** | Core feature; good self-contained CLI; clean API |
| **tools/bakeoff/main.c** | 560L | Benchmark harness: run codecs over corpus, measure error metrics, emit ledger | corpus, codecs, metrics, ledger | **KEEP/REFACTOR** | Essential benchmarking tool; rename to c5d-bench or integrate into CMake |
| **tools/bakeoff/codecs.c** | 290L | Codec wrappers (c5d0, c5d1, wav, zstd, zfp, passthrough) | brick, v0, wav, zstd, zfp | **KEEP** | Needed for bench comparisons; strip to c5d1, zstd, passthrough only |
| **tools/bakeoff/metrics.c** | 247L | Quality metrics: SSE, PSNR, SSIM-3D, max-error, percentile, blocking | none | **KEEP** | Core measurement library; no external deps beyond math |
| **tools/bakeoff/corpus.c** | 127L | Corpus loader: walk dir, load u8 bricks + sidecars | none | **KEEP** | Clean; reusable for other tools |
| **tools/bakeoff/ledger.c** | 43L | Append JSON results to ledger file | none | **KEEP** | Minimal; good hygiene (per-row ledger vs monolithic output) |
| **tools/prof/driver.c** | 193L | Focused profiling: encode-only / decode-only loops | brick, corpus | **KEEP** | Simple, useful for perf measurement; no external deps |
| **tools/train_priors.c** | 114L | Retrain entropy priors from corpus .u8 files | brick | **KEEP** | Format-critical; must be available when priors change |
| **tools/label_bench.c** | 319L | Label codec bakeoff: per-channel bytes vs zstd, encode/decode speed | brick, label, zstd | **KEEP** | Validates label codec; good benchmarking companion |
| **tools/fetch_corpus.py** | 47L | Download pinned zarr bricks from S3 (reproducible) | zarr, fsspec | **KEEP** | Reproducibility; users must be able to regenerate corpus |
| **tools/ink_metric.py** | 127L | Downstream task gate: ink-detector AUROC/Dice/F1 on decoded vs reference | numpy | **KEEP** | Release gate pattern; educational for task-aware metrics |
| **tools/requirements-corpus.lock** | – | pip lock file (zarr, fsspec pinned) | – | **KEEP** | Reproducibility; regenerate if deps change |
| **tests/test_quick.c** | 247L | <5s sanity: metrics correctness, pool, SSIM, percentiles, blocking | brick, cache, pool, metrics | **KEEP** | Fast CI gate; validates metrics code |
| **tests/test_golden.c** | 122L | Conformance: decode golden bitstreams, verify ≤1 LSB tolerance + tau/lossless invariants | brick | **KEEP** | Decoder freeze; golden files must be regenerated but tests carry forward |
| **tests/test_stable.c** | 129L | Stable-ABI tests: u8/u16/label encode/decode, shard read/write with cancel | stable ABI, label | **KEEP** | ABI compatibility gate; critical for long-term support |
| **tests/test_xdec.c** | 199L | Cross-path decode fuzzing: CPU scalar vs rANS interleave vs chunk decode | brick | **KEEP** | Determinism validation |
| **tests/test_hostile.c** | 255L | Hostile-input harness + adversarial review fixes | brick, label, shard | **KEEP** | Security gate; must run every build |
| **tests/test_label.c** | 279L | Label codec: encode/decode roundtrip, mask chains, palette | label | **KEEP** | Feature validation |
| **tests/test_tifxyz.c** | 134L | tifxyz pack/unpack/verify | tifxyz | **DROP** | Scope creep; if tifxyz is dropped, this goes too |
| **tests/golden/** | ~600 KB | src48.u8 source + 7 golden bitstreams (.c5b) + expected outputs (.out.u8) | – | **KEEP/REGENERATE** | Frozen format conformance; regenerate all files in the new repo's first build |
| **fuzz/d_brick.c** | 26L | Hostile-input fuzzer for c5d_brick_decode | brick | **KEEP** | Part of CI; corpus on disk negligible |
| **fuzz/d_label.c** | 40L | Hostile-input fuzzer for label decoder | label | **KEEP** | Part of CI |
| **fuzz/rt_brick.c** | 53L | Round-trip fuzzer: encode params + random voxels, verify error bounds | brick | **KEEP** | Part of CI |
| **bench/profile.sh** | 9L | `perf record` + `perf report` wrapper | perf (system tool) | **KEEP** | User documentation; low friction |
| **bench/bdrate.py** | 73L | Ledger analysis: list runs, compute BD-rate curves | numpy, scipy | **KEEP** | Measurement tool; requires stable ledger format |
| **bench/baselines/** | varies | Pre-built zstd/zfp/gpudct binaries or build scripts | external codecs | **DROP** | Site-specific; users will regenerate if needed |
| **corpus/manifest_dev.json** | small | Pinned 512-brick PHercParis4 subset (real scroll data) | – | **KEEP** | Locked corpus = reproducible benchmarks |
| **corpus/manifest_full.json** | small | Larger development corpus manifest | – | **KEEP** | Available but optional |
| **corpus/synth/synth_*.json** | 4×70B | Synthetic test case descriptions (air, dense, smooth, surface) | – | **KEEP** | Light-weight reference corpus; data files generated on-the-fly |
| **corpus/dev/** | ∅ (remote) | Actual brick data (fetched by fetch_corpus.py, not in repo) | – | **KEEP (manifest only)** | Manifests curate the selection; binaries download on-demand |
| **spec/format.md** | 200L | Normative bitstream spec: C5B3/C5U1/C5S1/C5L1/TFX1 | – | **KEEP/TRIM** | Essential documentation; drop TFX1 (tifxyz) section if tifxyz is dropped |
| **docs/measured.md** | 200+L | Measured decisions graveyard: every rejected idea with numbers | – | **KEEP** | Invaluable design rationale; prevents re-litigating decisions |
| **docs/BACKLOG.md** | 100+L | Confirmed findings + validated improvement candidates (2026-08-06 review) | – | **KEEP** | Roadmap; informs future work |
| **docs/BENCHMARKS.md** | varies | Current results, honest gaps, per-class breakdowns | – | **KEEP** | Reference; update with first new runs |
| **PLAN.md** | 300+L | Implementation plan v2: design posture, requirements, architecture (§0–3), discipline rails | – | **KEEP/UPDATE** | Historical context + architecture guide; update for simplified scope |
| **README.md** | 80L | Quick-start, headline numbers, use-case overview | – | **KEEP/TRIM** | Update quick-start commands to drop remote/pack; update headline (regenerated) |
| **.clang-format** | 8L | Style: LLVM base, 100-char column, 2-space indent | – | **KEEP** | Non-controversial; consistency matters |
| **.clang-tidy** | 10L | Static checks: bugprone-*, clang-analyzer-*, performance-*; warnings-as-errors on 4 checks | – | **KEEP** | Quality gate; maintain or tighten |
| **.gitignore** | 16L | Excludes: build/, corpus/dev+full, fuzz corpora, .venv, pycache, research repos, tifxyz data | – | **KEEP** | Standard; add volume-compressor-specific paths if needed |
| **CMakePresets.json** | 90L | Presets: dev (asan+ubsan), release (hardened), bench (native), tsan, msan, gpu | – | **KEEP** | Simplify: drop msan (rare use), gpu (out of scope for M1), keep dev/release/bench/tsan |
| **requirements-corpus.lock** | 1 KB | pip freeze output for zarr/fsspec (reproducible deps) | – | **KEEP** | Reproducibility; regenerate on update |

---

## 2. TOOLS/C5DC SUBCOMMANDS AUDIT

**Main.c dispatch table (lines 77–101):**

| Subcommand | Argc | Purpose | Status | Notes |
|---|---|---|---|---|
| `pack` | 5+ | Assemble LOD pyramid from brick grid + quality ladder | **DROP** | Niche; assumes pre-prepared 8×8×8 brick grid in zarr coords; users have data pipelines |
| `stat` | 3+ | Print shard stats (brick count, zero/present, bytes, ratio) | **KEEP** | Diagnostic; 30L in pack.c; useful for verification |
| `unpack` | 5+ | Extract brick from shard to raw u8 file | **KEEP** | Debugging; part of core CLI; supports threading |
| `slice` | 6+ | Dump a 2D slice of a raw u8 cube as PGM | **REWRITE** | Retain as example/debugging tool; not essential for library use |
| `diff` | 7+ | Compute |a-b| error heatmap between two u8 cubes | **REWRITE** | Retain as example; not essential |
| `tifxyz-pack` | 4+ | Pack surface parameterization dir → .tfx | **DROP** | Scope creep |
| `tifxyz-unpack` | 4+ | Unpack .tfx → dir of .tif files | **DROP** | Scope creep |
| `tifxyz-verify` | 3+ | In-memory roundtrip: load dir, encode, decode, verify | **DROP** | Scope creep |
| `label-pack` | 5+ | Multi-channel label brick: raw .u8/.u16/.u32 → .c5l | **KEEP** | Core feature; clean args |
| `label-unpack` | 4+ | .c5l → per-channel raw files | **KEEP** | Core feature |
| `label-info` | 3+ | Dump .c5l header: dim, types, masks, palettes | **KEEP** | Core feature; light-weight |
| `label-verify` | 4+ | Encode/decode roundtrip; measure stats | **KEEP** | Validation; matches brick subcommand pattern |
| `remote` | 5+ | Fetch single brick from S3 shard via ranged curl GET | **DROP** | S3 integration is site-specific; move to separate tool/docs |
| `remote-batch` | 5+ | Parallel batch fetch + decode from S3 shard | **DROP** | S3 integration; users will integrate into their workflows |

**Subcommand grouping in new repo:**

```
c5d encode <input.u8> <dim> [--quality=Q] [--lossless] [--tau=T] [...] -o <out.c5b>
c5d decode <in.c5b> -o <out.u8>
c5d info <in.c5b>
c5d verify <in.c5b> <ref.u8> [--tolerance=TOLERANCE]

c5d-label-pack <out.c5l> <dim> [opts] <chan>...
c5d-label-unpack <in.c5l> <out_prefix>
c5d-label-info <in.c5l>
c5d-label-verify <dim> [opts] <chan>...

c5d-bench [--corpus=DIR] [--codec=all|c5d1] [--quality=Q] ...
c5d-prof [--mode=encode|decode|lossless] [CORPUS_DIR] [QUALITY] [THREADS]
```

**Arg-parsing duplication:**  
- All subcommands re-implement option parsing (--option=value style).
- Centralize in a single parser or delegate to a getopt-style helper.
- c5dc main.c uses simple linear scan; acceptable for small CLI, but label/brick CLIs should share a pattern if both survive.

**IO helpers:**  
- `load_cube()` / `write_pgm()` (main.c, lines 13–62) — single-purpose, OK.
- `read_all()` / `write_all()` (label_cmd.c, lines 28–58) — duplicates above; dedupe or move to a shared util header.

---

## 3. TOOLS/BAKEOFF METRICS & BENCHMARK TOOL

### Metrics library (`metrics.c`):

**Implemented metrics:**
- `metric_sse_u8()` — sum of squared error (lines 7–14)
- `metric_psnr_u8()` — 10·log₁₀(255²/MSE) (lines 16–25)
- `metric_maxerr_u8()` — max |a[i] - b[i]| (lines 27–34)
- **3D SSIM** (`metric_ssim3d_u8()`, `metric_ssim3d_u8_box()`, lines 36–150):
  - Separable Gaussian blur (11-tap, σ=1.5), 3 axes, clamped edges
  - Per-voxel SSIM using blurred means + variances + covariance
  - Constant c1, c2 (0.01×255, 0.03×255)² — SSIM standard
  - Returns average SSIM over a box or full brick
  - **Speed concern**: 7 float buffers @ 4 MB each (128³ brick) + 3 separate malloc/free for intermediate m_xx/m_yy/m_xy. Optimization: allocate once, reuse via workspace manager.
- `metric_errhist_u8()` — error distribution histogram (line 152–154)
- `errhist_mae()` — mean absolute error from histogram (lines 156–163)
- `errhist_percentile()` — nearest-rank P_k from histogram (lines 165–176); used for P90, P95, P99, P100(max)
- **Blocking metrics** (`metric_blocking_u8()`, lines 215–224; post-deblock artifact detection):
  - Measure gradients across block boundaries vs interior
  - Separate boundary (chunk face) and interior (mid-chunk) error statistics
  - Outputs: boundary_abs/sse/n, interior_abs/sse/n (for MAE/RMSE)
  - Amplification = boundary_rmse / interior_rmse (detects deblock ringing)

**Quality**: Metrics are precise and well-designed. SSIM memory usage can be optimized for batch processing (allocate large buffer upfront).

**Coverage vs goals:**
- ✓ Mean/P90/P95/P99/max error (errhist functions)
- ✓ PSNR (both point + aggregate pooled-MSE)
- ✓ SSIM (3D, full-brick + box-scoped)
- ✓ Blocking artifact detection
- ✗ Frequency-domain metrics (perceptual weighting, DCT energy distribution) — not in scope for M1
- ✗ Temporal metrics (LOD coherence) — future
- ✗ Task-specific loss (ink detection, segmentation) — delegated to downstream tools (ink_metric.py)

### Benchmark tool (`main.c` + supporting files):

**Scope**:
- Load corpus (corpus.c)
- For each codec in selection, for each brick in corpus:
  - Encode N times (reps), collect latency
  - Decode N times
  - Measure error metrics (SSE, PSNR, SSIM, histogram, blocking)
  - Per-class aggregation (dense_interior, smooth_papyrus, surface_boundary, air, unknown)
- Output: per-codec/class row with ratio, PSNR, SSIM, MAE, P90/95/99, max error, enc/dec MB/s, percentile latencies
- Append to ledger (JSON one-per-line, timestamped git hash)

**Ledger format** (ledger.c, lines 19–42):
```json
{
  "ts": "2026-09-01T12:34:56Z",
  "git": "a1b2c3d",
  "codec": "c5d1",
  "quality": 2.0,
  "corpus": "dev",
  "class": "dense_interior",
  "bricks": 128,
  "ratio": 24.5,
  "psnr_db": 42.3,
  "ssim": 0.9876,
  "maxerr": 5.0,
  "mae": 0.8,
  "p90": 2.0, "p95": 3.0, "p99": 4.0,
  "enc_mbps": 650.0, "dec_mbps": 800.0,
  "timing": "median",
  "enc_us_p50": 15.3, "enc_us_p90": 18.5, ...,
  "block_boundary_rmse": 0.15, "block_interior_rmse": 0.02, "block_amplification": 7.5
}
```

**Features in main.c (lines 113–400+):**
- Codec selection: `--codec=all|NAME[,NAME...]`
- Quality sweep: `--quality=Q` (single quality per run)
- Reps: `--reps=N` for latency distribution
- Timing mode: `--timing=median|best` (representative sample method)
- Corpus source: `--corpus=DIR`
- Vol-SSIM mode: `--vol-ssim` (assemble full volume from bricks, compute SSIM on interior boxes) — complex, mainly for reference; can be dropped if memory is tight
- `--seed=N` for reproducible corpus shuffle
- `--set=k=v` for per-codec parameters (e.g., `--set=deblock=1`)
- Ledger control: `--no-ledger` (skip append), `--no-ssim` (skip SSIM, slow), `--vol-ssim` (slow but exact for assembled volume)

**Codecs in registry** (codecs.c, lines 270–279):
- `b1` — c5d1 (modern, primary)
- `v0` — c5d0 (legacy reference)
- `wav` — wavelet codec (experimental)
- `pt` — passthrough (1x, for harness sanity check)
- `zstd_c` — zstd-3 (lossless reference, optional `#ifdef C5D_HAVE_ZSTD`)
- `zfp_c` — zfp (lossy reference, optional `#ifdef C5D_HAVE_ZFP`)

**For new repo:**
- Keep b1 (c5d1), pt (passthrough)
- Keep zstd (external baseline reference)
- Drop v0, wav, zfp (experimental/old)
- Update README: "Current codec is c5d1; zstd lossless baseline for comparison."

### Corpus loader (`corpus.c`):

**Mechanics**:
- Walk directory for `*.u8` files (cubic volumes, size = dim³)
- Load optional `*.json` sidecar (class field extracted)
- Sort by brick ID for determinism
- `corpus_shuffle()` uses LCG for reproducible randomization
- Supports up to `CORPUS_MAX_BRICKS` = 1024

**For new repo:**
- Keep as-is; supports both dev (512 real) and full (1K+) corpus
- Manifests are locked (checked in); binaries fetched on-demand by fetch_corpus.py

---

## 4. TESTS AUDIT

### Test coverage summary:

| Test | Lines | Runtime | Coverage | Keep? | Notes |
|---|---|---|---|---|---|
| test_quick.c | 247 | <1s | Metrics (PSNR/SSIM/maxerr/percentile/blocking), pool, version strings | **KEEP** | CI gate; fast |
| test_golden.c | 122 | <1s | Decoder conformance: 7 golden bitstreams, ≤1 LSB tolerance, tau/lossless invariants | **KEEP** | Freeze spec; regenerate files |
| test_stable.c | 129 | <5s | Stable ABI: u8/u16 brick, label, shard read/write, cancel callbacks | **KEEP** | Long-term API contract |
| test_xdec.c | 199 | <5s | Cross-path: scalar vs rANS-interleave vs chunk decode, determinism | **KEEP** | Format invariant |
| test_hostile.c | 255 | <10s | Hostile input: brick/label/shard decoders don't crash on garbage; specific adversarial cases from 2026-08-03 review | **KEEP** | Security gate |
| test_label.c | 279 | <2s | Label codec: roundtrip u8/u16/u32/u64/i8/i16/i32/i64, masks, palettes | **KEEP** | Feature validation |
| test_tifxyz.c | 134 | <1s | tifxyz pack/unpack/verify | **DROP** | Scope creep if tifxyz is dropped |

### Golden vectors (tests/golden/):

**Current files** (48³ = 110 KB each):
- `src48.u8` — source reference
- `lossless.c5b` + `lossless.out.u8` — lossless roundtrip
- `lossy_q2*.c5b` + `.out.u8` — 5 variant configs (IL2, EP, C2, S8)
- `tau4_q4.c5b` + `.out.u8` — tau=4, percentile-bounded

**Action**:
- Retain format/structure
- **Regenerate all files in new repo** (first build with full CMake)
- Rationale: backward compat not required; golden files freeze the new repo's codec behavior, not the old one's

### Coverage gaps vs goals:

**Existing coverage:**
- ✓ Metrics (mean, percentile, SSIM, blocking)
- ✓ Lossless exactness
- ✓ Tau bounds (P99 ≤ tau)
- ✓ Determinism (cross-path, same-build)
- ✓ Hostile input (fuzz)

**Gaps:**
- ✗ u16 encode/decode (in test_stable.c but not golden; add u16 golden variants if that mode is core)
- ✗ Chunk-level decode consistency (test_xdec.c touches this; expand if chunk API is user-facing)
- ✗ Multi-threading determinism (--threads=1 vs N; test_stable and bakeoff touch this; add explicit gate if needed)
- ✗ Deblock edge cases (small bricks, no-deblock mode, boundary mismatch) — covered in adversarial review but not pinned in gold
- ✗ LOD level consistency (if pyramid generation is kept; closed-loop decimation should be tested)
- ✗ Shard index integrity (CRC checks, missing brick handling) — covered in test_stable.c via shard writer, but not in golden

**Action**:
- Add 1–2 u16 golden tests if u16 is a primary mode
- Add deblock-edge-case coverage to test_quick or adversarial suite
- Do NOT expand golden test count beyond current 7; golden is a freeze, not a feature showcase

---

## 5. FUZZ AUDIT

**Harnesses**:

| File | Lines | Target | Corpus | Status |
|---|---|---|---|---|
| d_brick.c | 26 | c5d_brick_decode (hostile input) | generated (libfuzzer) | **KEEP** |
| d_label.c | 40 | c5d_label_decode + header readers (hostile) | generated | **KEEP** |
| rt_brick.c | 53 | Encode→decode roundtrip (random voxels + params) | generated | **KEEP** |

**Corpus on disk**:
- `build/fuzz/` contains compiled harnesses
- No seed corpus checked in (generated on first run)
- Total fuzz campaign: 4.5M execs (d_brick + earlier), 2.0M (rt_brick), zero findings post-2026-08-03 fixes
- Coverage: d_brick 561 (hostile parse), rt_brick 835 (roundtrip + params)

**Action**:
- Keep all three harnesses
- Do NOT copy existing seed corpus; fuzzer regenerates on-demand
- CI: run `ctest --preset fuzz` with time limit (e.g., 60s per harness)
- Update docs: "Fuzz corpus generated dynamically; seed corpus grows across runs."

---

## 6. DOCS & SPEC AUDIT

### spec/format.md (200+ lines):

**Scope**: v1.5 bitstream spec — normative, parse-exact, float semantics documented.

**Sections** (all relevant to library):
- Geometry (chunk/brick/shard hierarchy)
- Brick header (40 B: magic, dim, flags, params, substream table)
- Substreams (rANS independence, chunk partitioning)
- Lossy pipeline (DCT-II, dead-zone quant, scanning, entropy)
- Context models (band-based, ctx2 variant)
- Entropy coder (rANS, interleave, normalization)
- Lossless mode (3D predictor, residual)
- uint16 wrapper (C5U1 magic)
- Deblock (normative, post-decode)
- Tau corrections (sparse, percentile-bounded)
- Determinism (float tolerance, UB boundary)
- Shard container (C5S1 magic, index, footer, LOD)
- Label bricks (C5L1 magic, multi-channel, masked channels, palette)
- **Tifxyz surfaces (TFX1 magic)** — OPTIONAL; drop if scope excludes surfaces

**Action**:
- Keep all C5B3/C5U1/C5S1/C5L1 sections
- Drop TFX1 section if tifxyz is excluded
- No changes to format itself

### docs/measured.md (decision graveyard):

**Content**: Every rejected idea with numbers; inherited assumptions tested; measured outcomes 2026-08-03, 2026-08-06.

**Covered decisions**:
- CDF 9/7 wavelet vs DCT-16 ✗ (DCT wins +105–190%)
- Lapped transforms ✗
- hf_exp sweep ✓ (0.65 default)
- SSIM pooling vs mean ✓ (pooled better for lossless air)
- Frequency-band context +5–7%
- 8³ vs 16³ chunk size (+27–47% ratio for 16³)
- RDO-lite ✗ (RD-negative)
- Fuzz results (793k+ execs, clean)
- Adversarial review 2026-08-03: 4 critical + 1 major fixes, deferred items
- 2026-08-06 x86/GPU review: GPU encode done, Vulkan scheduling closed

**Action**:
- **KEEP as-is** — invaluable for future decisions and understanding the codec
- Append new measured sections as they accumulate in the new repo
- Link from PLAN.md

### docs/BACKLOG.md (improvement candidates):

**Confirmed findings** (2026-08-03 review):
- Shard crash-safety weaker than PLAN claimed (footer only at close)
- Decode logic triplicated (brick.c, host_entropy.c, GLSL) — no CI equivalence gate
- Cache budget unenforced under heavy pinning

**Validated improvements** (magnitude known):
- Deblock NEON: +30% to lossy decode (26–32% of current decode time)
- Interleaved 2–4 rANS states: +1.6–1.8x rANS decode
- Trained #embed priors: kills 640 B/brick table tax; unlocks prev-magnitude context

**Action**:
- Keep as-is for roadmap visibility
- Use as guidance for M2+ optimization work

### docs/BENCHMARKS.md:

**Content**: Current measured results, honest gaps, per-class breakdowns.

**Action**:
- Keep structure; update numbers after first new run
- Reference this from README headline

### PLAN.md (implementation plan v2):

**Sections**:
1. Design posture (ideas ≠ authorities; measure everything)
2. Requirements (use case, speed targets, hierarchy, GPU native, tolerance model)
3. Architecture (per-chunk pipeline, deblocking, LOD, format, cache, execution, targets)
4. Discipline (C23, CMake, sanitizers, fuzzing, CI, determinism, cross-path verification)
5. Status (pre-2026-08 roadmap; now superseded by measured/backlog)

**Action**:
- Keep §0–2 (design posture + requirements unchanged)
- Update §3 (architecture) if scope changes (e.g., drop remote, drop tifxyz)
- Archive or reference §5 (old roadmap) in measured.md
- Add "Recent status (2026-08)" section linking to measured/backlog

### README.md (quick-start guide):

**Current headline** (real 1 GiB PHercParis4):
> 24.16× @ 42.91 dB, P99 error 5, encode 2.48 GB/s, decode 3.02 GB/s.

**Quick-start commands**:
```sh
cmake --preset release && cmake --build --preset release
python3 -m venv .venv-corpus
.venv-corpus/bin/pip install -r tools/requirements-corpus.lock
.venv-corpus/bin/python tools/fetch_corpus.py dev
./build/release/c5d-bakeoff --corpus=corpus/dev --codec=c5d1 --quality=2
./build/release/c5dc pack corpus/full out 50 8   # DROP
./build/release/c5dc remote-batch https://bucket/object.L0.c5s 0,1,2 bricks 2  # DROP
```

**Action**:
- Update headline numbers (regenerate from first run)
- Remove `c5dc pack` and `c5dc remote-batch` examples
- Add `c5d encode`, `c5d decode` examples
- Add `c5d-label-*` commands if label codec is kept
- Trim tifxyz examples if tifxyz is dropped

---

## 7. CONFIG FILES AUDIT

### .clang-format:

```yaml
BasedOnStyle: LLVM
IndentWidth: 2
ColumnLimit: 100
AlignConsecutiveMacros: true
AllowShortFunctionsOnASingleLine: Inline
PointerAlignment: Right
SortIncludes: true
```

**Action**: **KEEP**. Non-controversial; consistency matters. No changes needed.

### .clang-tidy:

```yaml
Checks: >
  bugprone-*,
  clang-analyzer-*,
  performance-*,
  readability-misplaced-array-index,
  readability-enum-initial-value,
  misc-include-cleaner,
  -bugprone-easily-swappable-parameters,
  -misc-include-cleaner
WarningsAsErrors: 'narrowing-conversions, implicit-widening-of-multiplication-result, BitwiseShift, DivideZero'
HeaderFilterRegex: '(src|tools|tests)/.*'
```

**Action**: **KEEP**. Quality gate; tighten further if time allows (add more checks). No issues with current set.

### .gitignore:

```
build*/
corpus/dev/ corpus/full/
bench/baselines/*/
bench/results/*.png
.cache/
compile_commands.json
*.o
*.u8
!tests/golden/*.u8  # keep golden test data
__pycache__/
.venv*/
research/repos/
data/tifxyz/
```

**Action**: **KEEP**. Add `bench/results/ledger.jsonl` if ledger grows large, or keep it version-controlled for reproducibility.

### CMakePresets.json:

**Current presets**:
- `dev` (asan+ubsan, debug)
- `release` (hardened, ThinLTO)
- `bench` (native ISA, no branch protection/hardening)
- `tsan`, `msan` (thread/memory sanitizers)
- `gpu` (Vulkan, native ISA)

**Action**: **SIMPLIFY**:
- Keep `dev`, `release`, `bench`, `tsan`
- Drop `msan` (rarely used, slow)
- Drop `gpu` if out of scope for M1
- Add `fuzz` preset if not present (or keep in CMakeLists, no preset needed)

---

## BUILD DIRECTORY PRESETS

**build/* listing** (ls -la):

```
asan/  fuzz/  release/  tsan/  tsm/
```

- `asan/` — dev preset (AddressSanitizer + UBSan)
- `fuzz/` — fuzz preset with instrumentation
- `release/` — release preset (hardened)
- `tsan/` — ThreadSanitizer preset
- `tsm/` — Unknown; check CMakeLists if this is a legacy preset (possibly "thick-single-main"?)

**Action**: Document the `tsm` preset or remove it if no longer used. Simplify to 4–5 active presets in CMake.

---

## SUMMARY: KEEP/DROP/REWRITE DECISIONS

### KEEP (core to new repo):

**Tools**:
- `c5d encode/decode/info/verify` (rewritten main.c dispatcher)
- `c5d-label-{pack,unpack,info,verify}` (label codec)
- `c5d-bench` (bakeoff harness, refactored)
- `c5d-prof` (profiling driver)
- `c5d-label-bench` (label codec benchmarking)
- `train-priors` (entropy model retraining)
- `fetch_corpus.py`, `ink_metric.py` (reproducibility, downstream gate)

**Tests**:
- test_quick.c, test_golden.c, test_stable.c, test_xdec.c, test_hostile.c, test_label.c
- tests/golden/* (regenerated)
- Fuzz harnesses (d_brick, d_label, rt_brick)

**Docs**:
- spec/format.md (C5B3/C5U1/C5S1/C5L1, trim TFX1 if tifxyz dropped)
- docs/measured.md, docs/BACKLOG.md
- PLAN.md (updated)
- README.md (updated)

**Config**:
- .clang-format, .clang-tidy, .gitignore, CMakePresets.json (simplified)

### DROP (out of scope):

- `c5dc pack` — LOD pyramid assembly (site-specific; users pre-split data)
- `c5dc remote/remote-batch` — S3 batch fetch (site-specific; users integrate into workflows)
- `c5dc tifxyz-*` — Surface parameterization (scope creep)
- `c5dc slice/diff` — Visualization (examples, not essential; can be examples in docs if needed)
- test_tifxyz.c — Tifxyz validation (scope creep)
- bench/baselines/* — Pre-built zstd/zfp binaries (users regenerate if needed)
- CMakePresets: `msan`, `gpu` (if out of scope)

### REWRITE/REFACTOR:

- c5dc/main.c → new `c5d` CLI (encode/decode/info/verify only; keep label subcommands or split into `c5d-label`)
- bakeoff/main.c → rename to `c5d-bench`, integrate into CMake as standard build target
- codecs.c → strip to c5d1 + zstd + passthrough (drop v0, wav, zfp)
- README.md — update commands, headline numbers, trim scope
- PLAN.md — §3 architecture, update for simplified CLI and format

---

## 10-LINE SUMMARY

The c5d repo is clean, well-measured, and ready to clone into volume-compressor with surgical cuts. **Keep all core codec/test/measurement infrastructure**; the bakeoff harness and metrics library are production-quality and should be the new repo's benchmarking foundation. **Drop S3/remote operations, surface parameterization (tifxyz), and the LOD pyramid packing tool**—users will integrate those into their own pipelines. **Regenerate golden vectors** in the new repo (no backward compat required); carry forward the measured decisions log and spec as canonical design record. **Simplify the CLI** to encode/decode/info/verify plus label operations; update CMake presets to 4–5 active targets (dev/release/bench/tsan, drop msan/gpu if out-of-scope). **Keep the corpus manifests** (locked), fetch script (reproducible), and ledger format (bench comparison); add a 10-line README explaining how to generate/regenerate the full benchmark. The refactored codebase will be smaller, focused, and easier to maintain than the current multi-tool sprawl.

---

## DETAILED NOTES FOR INTEGRATION

### Golden Files Generation

Create a `tests/golden/regenerate.sh` script in the new repo:

```bash
#!/usr/bin/env bash
# Regenerate golden test vectors for the current codec version.
# Run after any format change that invalidates old files.
cd "$(dirname "$0")/.."
cmake --build --preset release
DIM=48
# Generate source
head -c $((DIM*DIM*DIM)) /dev/urandom > tests/golden/src48.u8
# Generate golden bitstreams via encode API or test tool
./build/release/test_golden  # Modify test_golden to write files instead of comparing
```

### Ledger Format Stability

The ledger format (ledger.c) is the foundation for benchmarking reproducibility. **Lock it now**; breaking changes require a version field (e.g., `"ledger_version": 1`). Current fields:

```
ts, git, codec, quality, corpus, class, bricks, ratio, psnr_db, ssim,
maxerr, mae, p90, p95, p99, enc_mbps, dec_mbps, timing,
enc_us_p50/p90/p95/p99, dec_us_p50/p90/p95/p99,
block_boundary_rmse, block_interior_rmse, block_amplification
```

### Codecs to Bundle

For the new repo's bakeoff, keep only:
1. **c5d1** (primary; modern encoder)
2. **passthrough** (1x; harness sanity check)
3. **zstd** (lossless reference; optional `#ifdef C5D_HAVE_ZSTD`)

Rationale: v0, wav, zfp are research variants; remove clutter. Users wanting other baselines can write their own codec wrappers.

### Corpus Reproducibility

- **Manifests** (corpus/manifest_dev.json, manifest_full.json) stay version-controlled
- **Binaries** (corpus/dev/*.u8, corpus/full/*.u8) never checked in; fetched on-demand by `fetch_corpus.py`
- **Lock file** (tools/requirements-corpus.lock) pinned; regenerate with `pip freeze` if zarr/fsspec versions change
- **CI note**: First `ctest` run will trigger corpus download; plan for ~500 MB download + 1–2 min setup

### Proposed New Repo Structure

```
volume-compressor/
├── src/                     # C23 codec library (from c5d/src)
├── tools/
│   ├── c5d.c               # Main CLI: encode/decode/info/verify
│   ├── c5d-label.c         # Label codec CLI
│   ├── c5d-bench/          # Benchmark harness
│   │   ├── main.c, metrics.c, corpus.c, codecs.c, ledger.c
│   │   └── bench.h
│   ├── c5d-prof.c          # Profiling driver
│   ├── c5d-label-bench.c   # Label benchmark
│   ├── train-priors.c      # Entropy model retraining
│   ├── fetch_corpus.py     # Corpus download
│   ├── ink_metric.py       # Downstream task gate
│   └── requirements-corpus.lock
├── tests/
│   ├── test_quick.c, test_golden.c, test_stable.c, test_xdec.c
│   ├── test_hostile.c, test_label.c
│   └── golden/             # Regenerated golden vectors
├── fuzz/
│   ├── d_brick.c, d_label.c, rt_brick.c
├── bench/
│   ├── profile.sh, bdrate.py
│   └── results/            # Ledger accumulated over time
├── corpus/
│   ├── manifest_dev.json, manifest_full.json
│   ├── synth/              # Synthetic test cases
│   └── (dev/, full/ fetched on-demand)
├── spec/
│   └── format.md           # Normative bitstream spec (C5B3/C5U1/C5S1/C5L1)
├── docs/
│   ├── measured.md         # Decision graveyard
│   ├── BACKLOG.md          # Future work candidates
│   └── BENCHMARKS.md       # Current results
├── .clang-format, .clang-tidy, .gitignore
├── CMakePresets.json       # Simplified to 4–5 presets
├── CMakeLists.txt          # (from c5d, minimal updates)
├── PLAN.md                 # Architecture + updated roadmap
└── README.md               # Quick-start, trimmed CLI examples, updated headline
```

### CI/CD Pipeline (Recommended)

```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build (dev)
        run: cmake --preset dev && cmake --build --preset dev
      - name: Test quick
        run: ctest --preset quick
      - name: Fuzz (60s)
        run: |
          timeout 60 ctest --preset fuzz || true
      - name: Lint
        run: clang-tidy-17 --checks=* src/** tools/** (check exit code vs warnings-as-errors)
  bench:
    runs-on: linux-optimized (if available)
    steps:
      - name: Build (release)
        run: cmake --preset release && cmake --build --preset release
      - name: Corpus (fetch dev)
        run: python3 tools/fetch_corpus.py dev  # ~500 MB, 1–2 min
      - name: Benchmark
        run: ./build/release/c5d-bench --corpus=corpus/dev --codec=all --reps=3 | tee bench_results.txt
      - name: Ledger append
        run: cat bench_results.txt >> bench/results/ledger.jsonl
      - name: Upload
        uses: actions/upload-artifact@v3
        with:
          name: ledger
          path: bench/results/ledger.jsonl
```



<!-- END 07-tools-tests-docs.md -->


---

<!-- BEGIN 08-brick-seam-deblocking.md -->

# 08 — 128^3 brick-boundary seams: diagnosis and design options

Repo studied read-only at /home/forrest/c5d. Experiments in
/tmp/.../scratchpad/{c5d-exp,build,seam*.c,qm.c,prof.c}.

## 1. What the code actually does

### Internal deblock filter
`deblock_face_task` /home/forrest/c5d/src/brick.c:419-469, driver
`deblock_par` :471-481.
- gate strength `c = clamp((int)(0.8f*q + 1.0f), 1, 24)` (src/brick.c:472-476);
  q is the *header* q (src/brick.c:1735), i.e. the brick's own quantizer.
- for a tap quad `p1 p0 | q0 q1` across a face: skip when `|q0-p0| == 0`, or
  `>= 4c`, or `|p1-p0| >= c`, or `|q1-q0| >= c`; else
  `delta = sign(d)*((3|d|+4)>>3)`, `p0 += delta`, `q0 -= delta`
  (src/brick.c:432-439 scalar x-axis form; :456-467 branchless y/z form).
  It moves *both* sides ~3/8 of the jump, so the step is cut to ~1/4.
- **Which faces:** `c5d_parallel_for(dim/16 - 1, deblock_face_task, ...)`
  (src/brick.c:479) with `f = (fi+1)*16` (src/brick.c:421). For dim=128 that is
  f in {16,32,...,112}. **f = 0 and f = 128 (the brick's outer faces) are never
  filtered.** Same for all three axes.
- Applied inside the decoder at src/brick.c:1734-1735, before corrections.

### c5d_brick_deblock_pair
src/brick.c:489-537 (decl src/brick.h:79). Identical math, taps taken from the
two neighbour buffers. Requires both bricks decoded => violates the
independence requirement if made mandatory.

### Sparse correction ("tau") layer — the important lever
- Encoder: src/brick.c:1047-1132. Builds `recon` closed-loop
  (src/brick.c:738-770), runs `deblock_par(recon,...)` (src/brick.c:1050-1052),
  histograms |src-recon|, selects the largest errors, emits
  `LEB128 gap, LEB128 zigzag(delta)` pairs appended after the substreams,
  `hdr.corr_n` rewritten (src/brick.c:1126-1131).
- Decoder: `apply_corrections` src/brick.c:1533-1569, called at
  src/brick.c:1736-1738 **after deblock**, delta clamped to +-255.
- **Format-wise this is an arbitrary per-voxel delta list** (spec/format.md
  "Tau corrections"). Only the *encoder's selection policy* is error-magnitude
  based. Carrying deliberately-chosen face deltas needs no format or decoder
  change at all. Caveat already in spec: single-chunk decode does not apply
  corrections (src/brick.c:1745-1764 has no correction step).

### chunk_qmap
Per-chunk Q2.6 multiplier on every step incl. DC: `qscale = qmap[ci]/64`
(src/brick.c:614), used at src/brick.c:655 and :745/:757; flag 256
(src/brick.c:1017, spec/format.md "With flag 256"). Encode-only decision,
normative decoder input, already shipped. No neighbour knowledge needed.

### DC prediction — NOT a seam source
`prev_dc = 0` at the head of each substream (encoder src/brick.c:610, decoder
src/brick.c:1591): DC is delta-coded within a substream only, never across
bricks. That is entropy context, not reconstruction. And DC is effectively
lossless: `dc_fine = 0.125` (src/brick.c:140) with an orthonormal 16^3 DCT
(gain 64), so the DC step in voxel units is `q*0.125/64` = 0.016 LSB at q=8.
Measured per-column signed bias in a decoded brick is +-0.2..0.5 LSB and shows
no boundary anomaly (scratchpad/prof.c output). Cross-brick DC-plane
prediction is already listed as rejected in docs/measured.md:17.

### RDO
Two-pass RDOQ, encoder-only, src/brick.c:668-722 + :931-956. Distortion is
per-coefficient squared error only; no spatial/seam term. Default off
(docs/measured.md:462-468).

### Encoder path / neighbour availability
`tools/c5dc/pack.c` loads the **whole 1024^3 volume into `vol`**
(pack.c:146-218) and encodes bricks in parallel from it (pack.c:35-68). So the
encoder trivially has all 26 neighbours' *source*; getting a neighbour's
*reconstruction* needs an extra decode pass (already done for LOD:
pack.c:271-284). LOD levels are separate shards, each with its own ratio
(pack.c:219, spec/format.md "LOD").

## 2. Diagnosis — why seams appear

Three independent contributors, in decreasing order of visible impact.

**(D1) The outer faces are the only 16-aligned faces the normative filter
skips** (src/brick.c:479). Measured on a smooth synthetic (mirror-tiled
src48.u8, smoothstep-upsampled x8), two adjacent 128^3 bricks, same q,
flat-conditioned mean |step| across a plane (only voxel pairs whose *source*
step is <= 3 LSB, i.e. the region where the eye sees banding):

| q | interior 16-face (filtered) | brick 128-face (unfiltered) | source truth | after deblock_pair |
|---|---|---|---|---|
| 2 | 1.183 | 1.230 (1.04x) | 1.10 | 1.207 |
| 4 | 0.818 | 1.408 (1.72x) | 1.10 | ~0.82 |
| 8 | 0.501 | 1.541 (3.08x) | 1.10 | 0.503 |

So at q>=4 the brick face is 1.7-3.1x rougher than every other 16-plane, and
`deblock_pair` restores exact parity. **At q=2 there is essentially no seam** —
consistent with the user seeing them only in the aggressive part of the ladder.

Note the second half of the story, which constrains every fix: the filtered
interior planes at q=8 sit at 0.50 while the *true* field's step there is 1.05.
**The filter over-flattens interior faces by ~2x below ground truth.** The
brick face at 1.54 is only 1.4x above ground truth. The artifact is therefore a
periodic *contrast in smoothness*, not a jump discontinuity — and no amount of
extra bits spent at the boundary can reach 0.50, because 0.50 is not a property
of the data. Confirmed: making the face chunk layer near-lossless
(qmap = 0.125 => q_eff 1) only takes the brick face from 1.541 to 1.175, i.e.
it converges to ground truth 1.10 and stops. Any "parity with the interior"
requirement can only be met by reproducing the filter, i.e. by sharing
information across the face.

**(D2) Every brick gets its own quantizer.** `pack_brick_task` calls
`c5d_brick_encode_target` per brick (tools/c5dc/pack.c:62-66), which bisects q
7 times to hit a *per-brick* byte target (src/brick.c:1766-1804). Adjacent
bricks with different content therefore decode at different q. Measured with
qA=2, qB=8: PSNR 54.40 vs 48.95 across the plane, brick-face step-error rms
1.079 vs 0.757 for interior faces. This produces a visible change of *texture
and noise amplitude* at the plane, which no deblock filter can remove. It also
desynchronises the gate: `c` differs between the two sides (src/brick.c:472).

**(D3) No filter contribution to the outer voxel layers.** x=0 is the only
"q0" column in the brick that is never pulled, x=127 the only "p0" column, so
those two layers have a systematically different noise statistic from
x=16,31,32,... Second-order relative to D1/D2 (measured per-column MAE at q=8:
0.759 at x=0, 0.840 at x=127, vs 0.905 at x=16 and 0.46-0.72 mid-chunk — the
filter actually *raises* per-voxel MAE at the columns it touches while lowering
the step).

## 3. Options

Ratio-cost anchors from the measurements: brick sizes on this synthetic at
UP=8 are 20548 B (q2), 14588 (q4), 10393 (q8); face chunks = 296/512 chunks
(shell), the single chunk layer on one face = 64/512; one face plane =
16384 of 2097152 voxels (0.78%); a correction pair for a strided face plane
costs exactly 2 bytes (gap 127 < 128 => 1 byte LEB, |delta| <= 3 => 1 byte).

### (a) Encoder-side seam compensation through the sparse correction layer
Mechanism: 2-pass encode. Pass 1: encode every brick (parallel, unchanged).
Barrier. Decode every brick (parallel). Pass 2: for each internal face, run the
`deblock_pair` math on the two reconstructions, and *append* the resulting
per-voxel deltas to each brick's own correction block (payload untouched, only
`corr_n` and the appended bytes change, src/brick.c:1126-1131).
- encode-only; **no decoder change, no format change** (syntax already exists,
  applied post-deblock at src/brick.c:1736).
- decode independence: preserved exactly.
- Measured cost (fraction of face voxels the filter actually moves, x2 bytes,
  x6 faces, vs brick size): q2 1.1% of voxels -> 364 B/face = **1.75% per face,
  10.5% for all 6**; q4 29.3% -> **65% per face, 391% for 6**; q8 46.3% ->
  **144% per face, 864% for 6**. Verdict: **viable at q<=2 only**, useless
  exactly where the seam is visible.
- Seam reduction: exact — reproduces `deblock_pair` bit-for-bit.
- Encode cost: +1 full decode pass over the level (~30% of encode time) plus a
  barrier; both passes stay embarrassingly parallel.
- Also note the encoder must decide face ownership or compute both sides'
  reconstructions (it has both), and single-chunk decode still won't see it.

### (b) Halo encoding (128^3 + 2-voxel halo = 132^3)
132^3/128^3 = 1.096 => **+9.6% raw voxels**, and the halo is 6 slabs of
128^2 x 2 that are not 16-aligned, so it breaks the chunk grid entirely
(dim % 16 == 0 is enforced at src/brick.c:857). A 16-aligned halo (160^3) is
+95%. Rejected on cost and on the geometry invariant. docs/measured.md:13
already records lapped/overlap transforms as "seam energy x2.2, MSE +54%,
rejected, do not revisit".

### (c) Finer q for boundary chunks via chunk_qmap  — MEASURED
Encode-only, existing format, no ordering constraint, no neighbour needed.
At q=8, applying the qmap only to the 64-chunk layer touching one face:

| qmap on face layer | bytes vs baseline | brick-face step |
|---|---|---|
| 1.00 (none) | 10393 (1.00x) | 1.541 |
| 0.50 | 11484 (+10.5%) | 1.408 |
| 0.25 | 12295 (+18.3%) | 1.230 |
| 0.125 | 13174 (+26.8%) | 1.175 |
Whole shell (296 chunks) at 0.25: 16929 B, **+63%**, face step 1.230.
Interior-face reference stays 0.50. **Saturates at ground truth (1.10-1.18);
never reaches parity.** Bad trade: +27% bytes for a 1.31x step reduction.

### (d) RDO with a seam-discontinuity penalty
Encode-only. Add a term to the RDOQ objective (src/brick.c:668-722) penalising
the reconstructed step across the outer face against the neighbour's
reconstruction. Requires the neighbour's recon => same 2-pass/barrier structure
as (a), and RDOQ already costs ~2.4x encode (docs/measured.md:467). Coefficient
choices are global to a 16^3 chunk, so a boundary-plane objective is fought by
4095 other voxels; expected step reduction is small (the RDOQ candidate set is
+-1 level). Estimated ratio cost 1-3%, seam reduction maybe 1.1-1.2x. Poor
value per unit of complexity.

### (e) Decode-side one-sided outer-face filter — MEASURED, REJECTED
Any in-brick estimate of the missing `p0` reduces to "pull the boundary voxel
toward its own inner neighbour". Prototyped exactly that (same gate, same 3/8
delta, taps `v0,v1,v2` from inside): brick-face step at q=8 went **1.541 ->
2.640, i.e. 1.7x WORSE**; q4 1.408 -> 1.645; q2 1.230 -> 1.233. Reason: the
true field has a real gradient across the face, so A's last column is pulled in
-x and B's first column in +x — the two biases *add*. Replicate-halo (p0=q0)
is a no-op by construction (dd == 0 fails the gate).
Adding a transmitted per-face target `t` does not save it: with
`v0' = v0 + a(t - v0)` and `t` common to both sides, the residual step is
`(1-a)(errB - errA)`, so a=3/8 buys only 1.6x, and a=3/4 (the 4x the interior
gets) forces `err(t) <~ err(v0) ~ 0.76 LSB` at full face resolution — i.e.
near-lossless coding of six 128^2 planes, ~5 KB/face against a 10 KB brick.
A coarse per-face DC/gradient parameter removes only the coherent low-frequency
component, and the measurement says the seam is **not** coherent: 16x16-patch
mean step-error rms at the brick face is 0.42 vs 0.83 at interior faces at q=8.
The seam is high-frequency noise contrast, so a cheap low-order face parameter
buys nothing.

### (f) Encoder DC pre-compensation of boundary chunks
Dead on arrival: DC is already accurate to 0.016 LSB (see D-section above) and
the measured seam is a high-frequency phenomenon.

### (g) Others
- **g1 (recommended): stop giving neighbouring bricks different q.** Fixes D2
  outright for zero bits and *less* encode work. See §4.
- **g2: uniform-treatment fix — make deblock a within-chunk operation.** If the
  normative post-process never crossed a chunk face (e.g. a per-chunk taper /
  coefficient-domain smoothing applied to the 16^3 block before scatter), every
  16-face and the 128-face would be treated identically and the seam would
  vanish *by construction*, decode independence would be trivially preserved,
  and `c5d_brick_decode_chunk` (src/brick.c:1745, today documented "No deblock
  (needs neighbors)", src/brick.h:72) would become exact. This is the correct
  long-term design but is a real format revision plus a quality campaign.
- **g3: turn the filter off at high q.** Measured: with `deblock=false` at q=8,
  interior 16-face step-error rms 1.4965 vs brick face 1.4130 — parity, no
  seam at all, at the cost of -0.5 dB (48.97 -> 48.48) and chunk blockiness
  everywhere. Proves the mechanism; a usable emergency lever
  (`p.deblock` is per-brick, src/brick.c:1012) but a bad trade.
- **g4: shift the brick grid per LOD level** — does not help, the seam is
  per-level.

## 4. Recommendation

**Primary (encode-only, no format change, no decode change, negative cost):**
Replace the per-brick rate target in `tools/c5dc/pack.c:62-66` with a
**level-uniform q**. Bisect q once per level on a sampled subset of bricks
(reuse the `c5d_brick_encode_target` bisection at src/brick.c:1766 but drive it
from the summed size of ~32 sampled bricks), then encode all bricks of the
level with `c5d_brick_encode` at that fixed q. This removes contributor D2
entirely — the strongest visible cue, a plane where noise amplitude and
texture change — and it is also the RD-correct allocation (constant lambda,
not constant rate per brick). It makes `pack` ~7x cheaper (one encode per
brick instead of up to 7). Changes: `pack_ctx` carries `float q` instead of
`double ratio`; `pack_brick_task` calls `c5d_brick_encode`; `cmd_pack` gains a
sampling loop before `c5d_parallel_for(nb, pack_brick_task, ...)`
(pack.c:243). Volume-edge/KNOWN_ZERO bricks (pack.c:52-60) are unaffected.
Ordering: none — still fully parallel.

Then, for the residual D1 contrast, pick by operating point:
- **q <= 2 (L0 near-lossless): option (a).** Exact `deblock_pair` parity for a
  measured **1.75%/face, ~10% for all six faces**, and it needs *zero* decoder
  or format change. Implementation: add
  `int c5d_brick_append_corrections(uint8_t **blob, size_t *n, const struct
  {uint64_t vi; int16_t d;} *fix, size_t nfix)` next to the tau writer
  (factor out src/brick.c:1080-1131 into a shared LEB128 emitter that merges
  with any existing corr block, keeping voxel indices ascending). `cmd_pack`
  gains: encode-all -> barrier -> decode-all -> for each of the 3 axis
  directions and each internal face, run the `deblock_pair` gate/delta on the
  two recons and record `(vi, +delta)` for the negative brick's `p0` and
  `(vi, -delta)` for the positive brick's `q0` -> append per brick. Both
  encoders already have both reconstructions, so no ownership rule is needed
  and both sides stay consistent. Faces with no neighbour (volume edge) or a
  KNOWN_ZERO neighbour (shard.h `UINT64_MAX-1`) emit nothing, matching today's
  behaviour. Parallel-safe: pass 1 and pass 2 are each fully parallel with one
  barrier between; each brick's correction list is written by its own task
  after reading two read-only recon buffers. Memory: needs the level's recon
  resident, which `cmd_pack` already allocates for the LOD loop
  (pack.c:271-275).
- **q >= 4 (the visible-seam regime): do NOT try to buy parity.** Option (a)
  costs 65-864%, (c) costs 27-63% and saturates short of parity, (e) is worse
  than nothing. The honest fix is g2 (make deblock chunk-local) or, as an
  immediate mitigation, reduce the *contrast* rather than the boundary: keep
  the filter but soften it at high q so interior planes stop sitting 2x below
  ground truth. That is a decode-side normative change (scale the 3/8 factor
  down as q rises, or cap `c` lower than 24 at src/brick.c:474-476) and should
  be validated against blocking-amplification numbers in
  docs/measured.md:468.

**Fallback (encode+decode, if a seam must be gone at high q and a format bump
is acceptable):** g2 — redefine the normative post-filter as a function of the
chunk's own decoded coefficients, applied inside the 16^3 block. Uniform by
construction at 16- and 128-planes, keeps every brick and every *chunk*
independently decodable, removes the `deblock_pair` API's reason to exist, and
removes the deblock gate-flip clause from spec/format.md Determinism.

## 5. Reproduction
- `scratchpad/seam2.c` flat-conditioned step metric (the table in D1)
- `scratchpad/seam3.c` step-error rms + 16x16-patch coherence
- `scratchpad/qm.c` qmap sweep + one-sided filter prototype
- `scratchpad/prof.c` per-column MAE/bias profile
Build: `cmake -S c5d-exp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
-DCMAKE_C_COMPILER=clang && ninja -C build c5d` (the `test_stable` target does
not compile in this tree; the library does).
Source: mirror-tiled `tests/golden/src48.u8`, smoothstep-upsampled x8, two
128^3 bricks adjacent along x.


<!-- END 08-brick-seam-deblocking.md -->
