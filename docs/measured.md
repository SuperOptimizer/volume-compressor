# Measured decisions graveyard

Every idea we try gets a verdict here, with numbers, so nothing is
re-litigated from memory. Format: date, idea, config/experiment ref,
result, verdict.

Inherited (from prior repos, pending re-verification on our harness — see
PLAN.md §0):

| Idea | Prior result | Status |
|---|---|---|
| CDF 9/7 wavelet vs DCT-16 | DCT +62–88% ratio iso-quality (fenix ADR 0005, float) | re-test M1 |
| Lapped/overlap transforms | seam energy ×2.2, MSE +54% (3ddct) | rejected, do not revisit |
| Zero-tree parent context | −3–15% (fenix) | rejected |
| DC-plane prediction across bricks | 0.1–0.5% of bitstream (gpudct) | rejected |
| Spatial-neighbor context | 25–35% cost for ~5% ratio (c4d) | rejected |
| TCQ | −0.2 dB (c4d) | rejected |
| Per-chunk entropy tables | overfit; brick/tile-global +73% (fenix) | adopt tile-global, re-verify M1 |
| hf_exp quant exponent | 0.65 converged across 4 repos | re-sweep M1 |
| K=32 rANS interleave | unused; K=4 sufficient on CPU (gpudct) | re-test with substream design |

## 2026-08-03 — M1 lab (synthetic corpus, 4 bricks)

- **c5d0 v0 baseline** (DCT-16 + dead-zone + order-0 rANS, per-coeff tokens +
  u16 EOB table): 5.19x@39.85dB (q1) .. 71.9x@32.9dB (q8). ~2x better ratio
  than zfp-accuracy at iso-PSNR. 103/122 MB/s enc/dec scalar.
- **Even/odd DCT butterfly** (256->128 mults): identical RD, +40-60% speed.
  ADOPTED.
- **Zero-run + in-stream EOB, 2-model rANS** (runs+EOB / levels): +5.4% ratio
  at q1, +12.6% at q4, +24% at q8, decode 1.3-4x faster (161-596 MB/s).
  ADOPTED — confirms c4d's 10-30% HF zero-run lever under our constraints.
- **hf_exp sweep on real corpus** (0.45/0.65/0.85, 5-point RD curves):
  near RD-neutral at iso-PSNR (±5%); 0.45 marginally better >49dB, 0.85
  marginally better <41dB. KEEP 0.65 default; revisit with SSIM/task metric.
- **Pooled-MSE aggregate PSNR** replaces mean-of-PSNR (999s from lossless
  air bricks poisoned the mean). Harness fix, not codec.
- **Frequency-band context split** (3 bands, scan-pos tertiles): level model
  split +5.1/+4.6/+2.2% (q0.5/2/8, real corpus); run+EOB model split adds
  +1.7/+2.6/+2.5%. Total +5..7% at identical PSNR, zero speed cost. ADOPTED.
  (c4d's +13% context gain included prev-token class — candidate next.)
- **dz_dq sweep**: 0.20 best by +0.03dB (noise-level). Default 0.20.
- **dz_q sweep**: near-neutral 0.2..0.3 at iso-rate; 0.4 dominated. Keep 0.2.
- **prev-run level context** (9 models): +0.3..0.6%, free. ADOPTED.
- **CDF 9/7 wavelet challenger, FAIR re-test** (3-level 3D lifting on 128^3
  brick, SAME run/level+band rANS backend): DCT-16 wins +105% (@51.5dB),
  +135% (@43.6dB), +190% (@36.4dB) ratio at iso-PSNR on real corpus.
  WAVELET REJECTED — fenix ADR 0005 confirmed under our constraints.
- **16^3 block size**: mandated by requirements (4KB = page/cache granularity,
  = GPU warp segment). 8^3/32^3 sanity check dropped per Forrest 2026-08-03.
- **Per-brick entropy tables**: 9 models x 26 syms x 2B = 468B per 2MB brick
  (0.02%) — overhead negligible at brick scope; region-sharing unnecessary.
- **8^3 vs 16^3 chunk size** (c5d0, real corpus, 5-point curves): 16^3 wins
  +27% ratio at ~50dB, +31% at ~46dB, +35% at ~42dB, +47% at ~38dB iso-PSNR.
  The 3D DCT needs the longer basis to capture papyrus fiber periodicity;
  8^3 also 8x's per-chunk overhead (DC+EOB per 512 vox). GPU parallelism
  does NOT improve: a subgroup's work granule is ~4KB either way (DietGPU
  segment), so 8^3 just means 8 sub-chunks per subgroup with worse ratio,
  and the 4KB=page=chunk alignment property is lost. 8^3 REJECTED.
- **RDO-lite (isolated-|l|=1 thresholding, constant rate est.)**: rdo=1 gives
  +5.1% ratio for -0.39dB = 13%/dB, but the q-ladder curve slope here is
  ~18.9%/dB — RD-NEGATIVE vs simply raising q. REJECTED (knob ships default
  0). Full RDOQ with a real per-context rate model + run coupling (gpudct
  gained -2.8% BD) remains future work.
- **PGO (clang -fprofile-generate/use, 2-run profile)**: 2x SLOWER than the
  stock ThinLTO release build (166/335 vs 362/380 MB/s enc/dec). Profile too
  thin + lost ThinLTO in the experiment config. REJECTED for now; revisit
  with IR-level PGO + ThinLTO combined if last-mile speed is needed.
- **Fuzz campaigns (coverage-instrumented lib)**: d_brick 793k execs cov 561,
  rt_brick 95k+ roundtrips cov 835 — zero findings. Plus 4.5M/2.0M execs
  from earlier (pre-instrumentation) runs, also clean.

## 2026-08-03 — Adversarial review (24-agent workflow) outcomes

Confirmed + FIXED (commit 8542433):
- CRITICAL brick.c b_parse: NaN/Inf/OOB header floats -> float->int cast UB.
- CRITICAL shard.c: offset+nbytes u64 overflow -> OOB read on hostile index.
- CRITICAL c5dc remote.c: popen shell-string -> command injection.
- MAJOR brick.c apply_corrections: unbounded LEB128 -> shift-exponent UB.
- CONFIRMED gpu host_entropy: silently dropped tau -> up to 85 LSB divergence
  (now rejects FLAG_TAU). Regressions in tests/test_hostile.c.

Confirmed doc-honesty fixes (BENCHMARKS.md):
- zfp advantage was ~10x (mixed corpora); real matched-PSNR value ~4.2x.
- dev "ALL" aggregate inflated ~12%/~0.8dB by air bricks -> full corpus is
  now the headline; per-class rows added; dev labeled non-representative.
- full-region throughput refreshed with actual c5d1 numbers + MAE/P99.

Confirmed but DEFERRED (backlog, docs/BACKLOG.md):
- Shard crash-safety weaker than PLAN claimed (footer only at close) —
  known limitation; double-buffered superblock is future work.
- Decode logic triplicated (brick.c x2 + host_entropy.c + GLSL), no CI
  equivalence gate — maintenance hazard; shared header + gate is future work.
- Cache budget unenforced under heavy pinning (prototype, not yet wired).

Improvement candidates (validated magnitudes, docs/BACKLOG.md):
- Deblock NEON: deblock is 26-32% of lossy decode, 100% scalar -> ~1.3x decode.
- Interleaved 2-4 rANS states/substream: ~1.6-1.8x on rANS decode.
- Trained #embed priors: kills 640B/brick table tax, big win on coarse-LOD
  small bricks; enables prev-magnitude context (~2.5% entropy) that's
  currently table-cost-vetoed.
- Real per-context RDOQ: ~2.8% BD (gpudct evidence); RDO-lite failed only on
  a constant rate model.

## 2026-08-03 (later): deblock vectorization + fast-math validation hole
- Deblock vectorized: y/z-axis faces are contiguous rows -> branchless
  gate-mask form with restrict row pointers autovectorizes (width 16);
  x-axis faces stride by dim (no NEON gather) -> kept branchy scalar.
  Whole-decode 8T: q2 955->1391, q8 882->1460 MB/s (+46-65%). Golden
  bit-exact (trunc((3d+/-4)/8) == sign(d)*((3|d|+4)>>3) by odd symmetry).
- REJECTED intermediate: single all-branchless loop with axis branch inside
  did not vectorize (uncountable loop / unknown bounds) and was SLOWER at
  low q than branchy scalar (lost the early-out; q2 955->890).
- CRITICAL fast-math lesson #2: the bit-level f_finite() guard (itself added
  because isfinite() dies under -ffast-math) was FOLDED AWAY under
  ThinLTO+fast-math — LLVM's known-FP-class analysis used nnan/ninf flags on
  the adjacent float range compares to "prove" q finite, then deleted the
  exponent test; a NaN q header was accepted in release (test_hostile only
  ran under dev). Fix: validate header floats entirely in the integer domain
  (positive IEEE floats order like their bits; sign/NaN/Inf patterns exceed
  any positive bound for free). Added a release testPreset so hostile/golden
  now gate the fast-math build too.

## 2026-08-04: interleaved rANS (format v1.1)
- 2-way interleaved rANS states per substream (flags bit 8; bit 16 = 4-way):
  +10% whole-decode 1T (412->453 MB/s, q2 full corpus), ~+5% at 8T. The
  backlog's 1.6-1.8x figure was the rANS stage alone; the stage is ~25-30%
  of decode wall post-deblock-vectorization, so +10% is the whole-pipeline
  truth. Encoder cost ~0, ratio cost +4 B/substream/lane (~0.02%).
- 4-way: statistically identical speed to 2-way, 3x the flush tax ->
  rejected; 2-way is the default (c5d_brick_defaults.rans_nway = 2).
- Decoded output is byte-identical across nway 1/2/4 (same models, same
  symbol sequence; interleave only reorders state renorms).
- entropy.comp not ported; he_gpu_setup rejects interleaved streams, gputest
  full-GPU leg pins rans_nway=1. Hybrid GPU path (host_entropy.c) supports
  all three. New golden vector lossy_q2_il2 + fuzz seed seed_il2.bin.
- Fuzzer (with new interleaved seeds) exposed a latent alignment bug: b_parse
  and host_entropy aliased `const sub_dir *` directly onto the input buffer,
  which has no alignment guarantee (fuzz inputs; brick blobs at arbitrary
  offsets in mmap'd shards). UBSan flagged misaligned u32 loads; exit code
  was still 0 (UBSan non-halting) — the earlier "clean" campaigns only
  checked exit status. Fix: copy the <=384 B directory into the parse ctx;
  regression: test_hostile decodes from a deliberately misaligned buffer.
  Lesson: grep fuzz logs for "runtime error", never trust exit code alone.
- Encode quant loop vectorized: natural-order reciprocal-multiply dead-zone
  quant (w4 x4 interleave) + int gather to scan order, replacing the
  scan-order fdiv loop. Encode 1T q2: 340->384 MB/s (+13%); ratio/PSNR
  unchanged to 4 significant digits (recip rounding flips ~0 levels).

## 2026-08-04 (later): GLSL interleave port + decode-logic dedup
- entropy.comp ported to N-way interleave (nway via push constant); full-GPU
  path re-validated on 24 real bricks with 2-way default streams: max
  |CPU-GPU| = 1 LSB. gputest no longer pins nway=1.
- Decode triplication addressed: src/format_internal.h is now the single
  source for header/directory structs, flags, context formulas, bit reader,
  HybridUint (brick.c + host_entropy.c alias it; GLSL mirrors by hand with
  a pointer to the header). host_entropy.c moved into the core lib.
- New CI gate tests/test_xdec.c (label quick): decodes the same streams
  through c5d_brick_decode AND he_decode + spec-reimplemented
  dequant/IDCT recon; levels must be identical across nway 1/2/4, voxels
  within 1 LSB. Catches C-path drift and spec-text drift; GLSL drift is
  caught on-device by c5d-gputest.

## 2026-08-04 (later still): trained #embed priors (format v1.2)
- 640 B/brick table tax replaced by globally-trained priors when cheaper
  (exact per-brick entropy cost decision; flag 32). Trained on all 512 full
  bricks x q{0.5,2,8}. Small-brick wins (mean of 20): 16^3 876->249 B
  (3.5x), 32^3 2262->1696 (+33%), 64^3 +0.8%, 128^3 neutral. Priors blob is
  normative (#embed, sha256 pinned in spec); retraining = format revision.
- Fuzzer (with EPRIOR seed) found pre-existing signed-overflow UB in
  apply_corrections and lossless residual add (hostile ~2^30 deltas overflow
  before the range check); both guarded with explicit +-255 residual checks.

## 2026-08-04: robustness sweep (BACKLOG items 1, 3, 4)
- Shard crash-safety: writers append a self-validating snapshot
  [index][footer][crc32c][C5SR] + fsync every 64 MB (0.01% overhead);
  c5d_shard_open falls back to the newest valid snapshot when the final
  footer is missing (crash mid-pack loses only post-snapshot bricks).
  Regression: truncation test in test_hostile. Also fixed misaligned
  c5d_shard_entry loads from the mmap (same UB class as the sub_dir bug).
- Cache backpressure: c5d_cache_put returns -2 when the shard budget cannot
  be met because all eviction candidates are pinned (was: silently exceeded
  budget). cache_stats_get now locks each shard (no torn counter reads);
  TSan clean.

## 2026-08-04: harness methodology (BACKLOG)
- Bakeoff A/B interleave: bricks outer, codecs inner — comparative numbers
  now share thermal envelopes (1T drift was +-15%).
- Whole-volume SSIM (--vol-ssim): brick+halo crops, interior-box means ==
  exact whole-volume per-voxel SSIM at 74 MB peak (vs 28 GB naive at
  1024^3). Full region: q0.5 0.9965, q2 0.9846, q8 0.9511 — the per-brick
  padded SSIM had overstated q8 by ~0.004.

## 2026-08-04: per-context RDOQ + bypass-bit ceiling (BACKLOG RD #3, #4)
- Per-context RDOQ (two-pass encode; pass-1 stats give real per-(model,
  token) bit costs; rate-aware zeroing of isolated |l|=1 incl. run-merge
  accounting): dev q2 sweep -> at lambda 0.1-0.2, +0.8-1.0% ratio at
  iso-PSNR with SSIM equal-or-better (0.9888 vs 0.9885); lambda >=0.4 is
  RD-negative. Cost: ~1.6x encode (two tokenize passes). Far below the
  2.8% gpudct figure (that was full-candidate RDOQ; ours zeroes isolated
  ones only). Default OFF; validated knob --set=rdo=0.15. The old
  constant-rate RDO-lite stays dead.
- Bypass bits: zstd-19 on concatenated bypass bytes EXPANDS them
  (-0.01..-0.12%) at q0.5/2/8 on real bricks -> genuinely incompressible;
  context-coding them (RD #4) is closed as a negative result.

## 2026-08-04: prev-magnitude context (format v1.3, flag 64) — default OFF
- Offline entropy analysis promised 1.7-1.8% (q0.5-2) to 4.1% (q8) on
  run+level tokens. Implemented (19 models, all three decoders, priors_v2,
  golden+fuzz seeds, xdec/gputest green): real gains are only +0.4-1.2%
  whole-stream on 128^3, +1.2-1.9% on small bricks — per-brick 4096-grain
  tables and thinner per-model statistics eat the difference. Decode is
  5-7% SLOWER (19 alias tables = 78 KB slot2sym, blows L1).
- Verdict: fully supported in the format, DEFAULT OFF. For the caching /
  decode-speed-first use case the trade is negative; revisit only if a
  table-free (priors-only) decode path with smaller alias tables lands.

## 2026-08-04: GPU minors (BACKLOG robustness #5)
- C5D_VK_VALIDATE=1 opt-in Khronos validation layer (graceful when absent).
- entropy.comp: per-substream status SSBO (binding 7) flags truncated
  streams (renorm past rans_n, or rans_n too small for the flush words)
  instead of silently zero-padding; gputest checks it after every brick.
- Device-local staging: left open intentionally — meaningless on this
  unified-memory iGPU; first task for any desktop-GPU port.

## 2026-08-04: Golomb-Rice entropy backend — REJECTED (measured)
- Question: GR is "perfectly parallelizable" — should we switch? Premise is
  off: GR decode is serial within a stream exactly like rANS; parallelism
  comes from stream partitioning, which the format already provides (32
  substreams/brick x batched bricks). GR would inherit, not improve, it.
- Cost, measured on 24 real bricks with per-context ORACLE k (GR best case,
  EOB as run-to-end): +15.0% / +41.4% / +26.6% stream size at q0.5/2/8 vs
  contexted rANS + bypass. The 1-bit/symbol floor destroys the sub-bit
  symbols (EOB, run=0, mag-1) that dominate at our target ratios.
- The cheap-raw-bits advantage GR offers already exists in the format: the
  33-35% bypass stream is uncoded (and measured incompressible).

## 2026-08-04: full-GPU decode pipeline 0.17 -> 1.0 GB/s (Adreno X1-85)
Step-by-step, each validated <=1 LSB vs CPU oracle on 24 real bricks:
- Batched mega-buffers, per-brick dispatches in one submit: 83->133 br/s.
- Flat cross-brick substream dispatch (all lanes live): ->429 br/s.
- Shared-mem freq/cum + 5-step in-shared binary search replacing global
  slot2sym (76 KB/brick of cold reads): ->482 br/s.
- Register-window byte fetch: ->494 br/s (+2.5%).
- 128-wide workgroups: SLOWER (460) -> stayed at 64.
- maxStorageBufferRange TRAP: turnip caps any SSBO *binding* at 128 MB and
  larger bindings silently read garbage — the early 24-brick batch (192 MB)
  was out-of-spec and only worked by luck; repl=4 exposed it (maxdiff 128
  everywhere). Fix: big buffers + per-slice 128 MB descriptor sets
  (vk_pipeline_alloc_sets/bind_ranges), per-stage dispatches per slice.
- vkCmdFillBuffer levels-clear was eating 30-60 ms/superbatch (~4-8 GB/s
  fill rate): replaced dense levels with SPARSE packed (pos12|val20) pairs
  + per-chunk counts; dequant zeroes its shared block and scatters. No
  clear at all, and levels traffic drops from 16 MB/brick to ~nnz words.
- Host re-recording ~110 dispatches cost ~35 ms/submit on turnip (3x GPU
  time): record each superbatch once, resubmit per rep (vk_end/
  vk_resubmit_wait). Wall 268->427 br/s.
- Final: 24-brick batch 427 br/s wall (0.90 GB/s); 96-brick 482 br/s
  (1.01 GB/s). Hybrid (CPU entropy) leg: 913 br/s kernel (1.91 GB/s).
- Segfault find: vk_pipeline_bind_buffers had info[8]/w[8] stack arrays;
  9-binding entropy pipeline overflowed them -> bounds now 16 + guard.
- Wave-imbalance attack REJECTED (both variants, measured): substream
  lengths leave 64-lane waves 47% idle (sorted-lane layout would be 13.6%),
  but (a) lane-sorting by ntok is NET SLOWER (2.72 vs 2.24 ms/brick — it
  forfeits shared freq/cum tables and scatters each wave's table/payload
  locality over up to 64 bricks) and (b) brick-total sorting is a no-op
  (all 32 substreams of a brick share one wave; the imbalance is
  intra-brick). The idle is structural at nsub=32 on 64-wide waves; the
  real fix is more substreams per brick (format knob) — desktop-GPU work.
- Per-stage split after all optimizations (24-brick batch): entropy 1.29
  ms/brick (57%), deblock 0.41 (18%), dequant+IDCT 0.32 (14%), barriers
  the rest. C5D_GPU_STAGES=e|d|b knob added for stage timing.
- Packed-u8 volume (4 voxels/uint word): dq writes words; deblock rewritten
  word-safe (axis 0 owns its two straddling words naturally; axes 1/2 one
  thread per 4-voxel word column, dispatch/4). Deblock 0.41->0.16 ms/brick
  (2.6x), dq 0.32->0.27, and output is now NATIVE u8 (no host conversion).
  Full-GPU: 24-brick 557 br/s kernel (1.17 GB/s); 96-brick 615 br/s wall
  (1.32 GB/s). Hybrid leg kernel: 913->1708 br/s (3.6 GB/s). <=1 LSB kept.
- Final stage split: entropy 1.30 ms/brick (76%), dq+IDCT 0.27, deblock
  0.16. Session total: full-GPU wall 74->615 br/s (8.3x).

## 2026-08-04: nsub as a GPU knob (format v1.4) — full-GPU 2.3 GB/s
- Root cause of the entropy ceiling: nsub=32 serial substreams/brick vs
  64-wide waves. v1.4 relaxes nsub to [1, min(nchunk,128)] (header already
  carried it); encoder knob params.nsub / --set=nsub.
- nsub=128: full-GPU pipeline 557->1107 bricks/s kernel (2.32 GB/s wall at
  96-brick depth; 15x the session-start naive kernel). Entropy stage 1.30->
  0.41 ms/brick. CPU decode unhurt (slightly faster, more parallelism).
- Ratio cost: -0.7% (q0.5), -2.1% (q2), -6.1% (q8) — real tradeoff, so the
  DEFAULT stays 32; nsub=128 is the GPU-serving profile chosen per shard.
- Golden lossy_q2_s8 (non-default nsub) + fuzz seed + hostile nsub checks
  (0, >nchunk, >128 all reject); kernel shared-table preload generalized
  (nsub 64/128 = one brick per workgroup).

## 2026-08-04: streaming loop + memory-type findings; fp16 rejected
- HOST_CACHED readback: turnip's plain VISIBLE|COHERENT mapping is uncached
  for CPU reads (~1.5 GB/s memcpy) — the 32 MB/round volume download was
  half the streaming loop. vk_buffer_create_cached (VISIBLE|COHERENT|CACHED,
  device-local on this part) doubled end-to-end AND sped up GPU-side access
  (batched kernel 0.91->0.83 ms/brick).
- Overlapped streaming leg (double-buffered cmd buffers + fences; host
  stages round N+1 and drains round N-1 while GPU decodes N): end-to-end
  incl all transfers = 1.95 GB/s at nsub=128 (2.53 kernel), 1.30 GB/s at
  default nsub=32 — equal to kernel rate, transfers fully hidden.
- fp16 shared-storage IDCT: REJECTED — 2.6 GB/s but max |CPU-GPU| = 5 LSB
  (18k voxels >=2) breaks the normative <=1 LSB contract.
- 128-wide entropy workgroups: still slower even at nsub=128 (2.14 vs 1.79
  ms/brick at nsub=32; 1.23 vs 0.90 at 128) — wave64 is fine, 4-slab shared
  footprint hurts residency. Wide bypass refill: -3%, reverted.
- slot2sym is dead weight on the GPU path (kernel binary-searches cum in
  shared mem for nsub 32/64/128): he_gpu_setup now skips the 78 KB/brick
  expansion and staging entirely on those paths.
- 3-deep streaming pipeline (was 2): hides fence/submit latency; end-to-end
  1.97 -> 2.35 GB/s at nsub=128 = 93% of kernel rate. Default format:
  1.35 GB/s end-to-end == kernel rate.

## 2026-08-04: dq flat-chunk fast path; two more micro-rejections
- DC-only/empty chunks skip the whole 3-pass IDCT (constant = DC/64,
  written as splatted words): kernel 0.83->0.77 ms/brick (2.53->2.73 GB/s),
  end-to-end 2.35->2.41 GB/s on dev corpus, <=1 LSB kept. Bigger win at
  high q / coarse LODs where flat chunks dominate.
- REJECTED: padded shared layout for the IDCT (stride 17/273 anti-bank-
  conflict) — 9% SLOWER; Adreno LDS is not conflict-bound and the larger
  footprint costs residency. Batched findMSB renorm — no change.

## 2026-08-04: GPU tau support + gate-flip contract amendment
- Tau on GPU: host parses the LEB128 correction stream into (voxel,
  delta+512) pairs (mirrors apply_corrections incl. hostile guards);
  corrections.comp scatters post-deblock via per-byte CAS on the packed
  volume (sparse -> no contention). Validated: hybrid + full-GPU legs pass
  at q4/tau2 on dev corpus; CPU-side pair application reproduces the
  oracle EXACTLY (698k pairs on a dense brick).
- Found while validating: q4 exposes deblock GATE FLIPS — a 1-LSB recon
  difference at a filter threshold flips the gate, diverging up to the
  filter delta (~1.5c; observed 5 at q4, bound 7). PRE-EXISTING since M5
  (q2/8/40 happened to have none); first suspected the DC fast path —
  eliminated by narrowing it (still failed). Contract amended honestly in
  spec Determinism: <=1 LSB except <=8 gate-flip voxels/brick within the
  delta bound; gputest enforces exactly that. GPU tau bound: tau + that
  tolerance.
- Bug chain fixed en route: corr_buf sized 4 MB but tau2@q4 emits up to
  5.6 MB of pairs on one brick (silent overflow -> 255-level garbage);
  now worst-case nvox*8. Batched/streaming legs skip tau runs (would need
  a corr mega-buffer; leg A/B validate the path).

## 2026-08-05: CPU speed campaign part 1 (Graviton-focused NEON work)
Baseline (dev q2, 1T, thermally-managed alternating A/B with cool-downs —
single runs drift up to -17% under sustained load; ALL numbers below are
paired): enc 383, dec 472 MB/s. After: enc 438 (+14.5%), dec 484 (+2.5%).
- -mcpu knob (C5D_MCPU cmake option): native on Oryon = +1.3/+1.7% (noise
  level; keep portable default). Real target: -mcpu=neoverse-v2 on Graviton 4
  (untested here; also unlocks SVE2 autovec).
- Tokenize NEON nonzero-extraction (vshrn nibble-mask -> position list;
  emission walks nnz not scan positions): +7.5% enc, stream-identical.
- NEON forward DCT (register-blocked fwd16_vert + 4-line transposed x-pass):
  encoder streams stay BYTE-IDENTICAL to scalar (verified: same blob bytes).
  Explicit-NEON INVERSE variants (both vert and x) measured SLOWER in-app
  despite +42% hot-loop microbench -> inverse stays autovec/scalar. Microbench
  lied: L1-hot + warm predictors; in-app the autovec form wins.
- TRAP FOUND: changing dct16.c flipped ThinLTO into scalarizing the
  float->u8 clamp in scatter_chunk (fadd/fcmp/csel per voxel, ~+1G instrs,
  -17% whole-brick decode) — in a file the edit never touched. Diagnosed by
  byte-identical-blob cross-decode (same blob, two libs: 273 vs 242 MB/s) +
  per-symbol instruction attribution. Fix: explicit NEON gather/scatter
  (vqmovun/vqmovn saturating narrows ARE the clamp; vcvta matches the
  +0.5-trunc rounding). Robust against optimizer mood + faster than the
  original SLP form.

## 2026-08-05: CPU speed campaign part 2 (micro-opts, PGO — mostly negative)
- rANS decn refill via clz (single multi-byte BE refill replacing the
  byte-loop): -5..-9% decode on Oryon (controlled same-blob decbench,
  paired). The 0-3-iteration byte loop predicts well; the clz+assemble adds
  latency to the serial chain. REJECTED.
- rANS freq|cum<<16 packed u32 (one load instead of two u16 loads): -4%
  decode. The unpack AND sits on the multiply's critical path; two
  independent zero-extending u16 loads are effectively free on Oryon's load
  ports. REJECTED. The rANS inner loop is load-latency-bound and already
  optimal in its naive form on this core.
- PGO retry, properly this time (IR PGO + ThinLTO both on, trained on dev
  q0.5/2/8 enc+dec 1T and 8T): +0.5-1% enc, +-0% dec vs same-flags non-PGO.
  The 2026-08-03 rejection stands, now for the right reason: hot paths are
  already well-shaped; PGO isn't worth the two-phase build. (The old
  experiment's 2x slowdown was the lost-ThinLTO config, confirmed.)
- Final dev-corpus ladder after part 1 (1T, ratios/PSNR bit-identical):
  q0.5 291/340, q2 440/485, q4 532/565, q8 630/642 MB/s enc/dec
  (was 241/300, 383/472 paired, 462/545, 543/617). 8T q2: 1553/1363.
- Gates: release+dev(ASan/UBSan)+TSan all green; encoder blobs byte-identical
  pre/post campaign (verified with cmp on dense-brick blobs).

## 2026-08-05: PMU characterization (q2, release build, Oryon)
1T dev enc+dec: IPC 3.83 @ ~3.4 GHz; stalls: frontend 12.3%, backend 17.4%,
backend-mem only 2.1%. L1d miss 0.74% (4.78G acc / 35M refill); L2->DRAM
refill 0.63M lines (~38 MB/s); dTLB walks 103K, i-cache misses 27K (nil).
Branch misses 2.0/kinstr. 8T dev: IPC 3.36, mem stalls 3.4%. Full 1GB 8T:
IPC 3.33, DRAM traffic ~(refill+wb)x64B = 10.7 GB over 6.1 s = ~1.8 GB/s
(bus_access proxy upper bound ~11 GB/s incl cache-to-cache). Decode-only
dense brick: IPC 3.01, L2 refills ~zero (cache-resident), mem stalls 0.16%.
Max RSS 59 MB (8T, dev corpus). VERDICT: codec is compute-bound with a
near-ideal memory profile — no bandwidth, TLB, or i-cache pressure at any
thread count measured; DRAM use is ~1-2% of platform capability. Scaling on
many-core Graviton should be linear until well past 64 cores; remaining
1T upside is SIMD width (SVE2) and the serial rANS chain, not memory.

## 2026-08-05: CPU DC-only IDCT skip (+17-23% decode); nway=4 evaluated
- Flat-chunk fast path in b_decode_sub (nnz_ac==0 -> constant DC*0.015625,
  same expression as the GPU dequant fast path; memset instead of 3-pass
  IDCT + scatter): dev q2 dec 485->566 (+17%), q8 642->790 (+23%), enc and
  streams untouched, ratio/PSNR bit-identical. All gates green; GPU
  cross-validation re-run: both legs + batched PASS <=1 LSB at q2 (CPU
  oracle now agrees with the GPU constant exactly on flat chunks).
- rans_nway=4 vs 2 (existing v1.1 flag, decode-compatible everywhere incl
  GPU): q2 dec 564 vs 567 (tie), q8 dec 633 vs 792 (loss; small streams
  pay 4x flush overhead), ratio -0.1..-1%. Default stays 2. Re-test once
  on Graviton (deeper OoO) before serving decisions there.

## 2026-08-06: Zen 4 SIMD audit — inverse gather rejection + packed scatter
- `perf` on dev q2 1T put DCT/IDCT at 42.8% of cycles. Disassembly showed
  Clang 21 still vectorizing the x-line loop across lines into AVX-512
  `vgatherqps`/`vscatterqps`; the existing pragma was attached to the vertical
  helper, not the loop that triggered the bad transform.
- Disabling cross-line vectorization for both directions was mixed: decode
  467->559 MB/s, but encode 398->335. Split policy kept the beneficial forward
  transform and suppressed only inverse: 398/467 -> 400/562 MB/s.
- Explicit saturating float-to-u8 reconstruction packing then raised decode
  to 656 MB/s with AVX2 and 670 MB/s with AVX-512BW (+43% end-to-end versus
  the original), while encode remained 402 MB/s. Encoder streams stayed
  byte-identical; golden/xdec remain within the normative cross-path tolerance
  (rare inverse-rounding deblock gate flips are covered by that contract).
  Full-corpus q2 1T is 402/553 MB/s; an auto-thread thermally variable run
  measured 2.48/3.02 GB/s.
- Post-change profile: x-axis `pass_axis_gs` fell from 22.3% to 16.8%; total
  deblock is 3.9%, so its strided x face offers about 1.3% whole-pipeline
  headroom and would require gather plus scalar/packed scatter. Deferred on
  measured value/cost, not assumed benefit.

## 2026-08-06: full GPU encode, compact decode, and schedule rejection
- Full GPU C5B3 encode is byte-identical to CPU for the implemented profile.
  RTX q2, eight bricks: 0.98 GB/s device and 0.94 GB/s including u8 upload and
  compact stream readback, ratio 20.58x. Isolated stage rates (GB/s): forward
  quant 45.00, tokenize+hist 1.58, model normalize 20.83, reverse rANS 8.11,
  prefix 138.15, assemble 94.53. The full small-batch rate is dominated by
  synchronization/transfer; reverse rANS is the kernel compute limiter.
- Exact `sum ceil(ntok/2)` compact pair storage replaced 4096 dense int32
  levels/chunk. q2 scratch fell 8192->1305 KiB/brick. Current 24-brick RTX run:
  8.44 GB/s compute, 4.68 GB/s end-to-end streaming, <=1 LSB. A nonuniform
  qmap run used 1194 KiB and reached 7.50/4.35 GB/s, also <=1 LSB.
- REJECTED: sorting substreams by ntok reduced projected subgroup-32 idle from
  20.6% to 1.7%, but entropy-only throughput fell 13.05->10.09 GB/s (-23%) from
  scattered payload/table access. Preserve on-disk order.
- Hostile audit fixed three latent GPU hazards: bounded compact writes by
  ceil(ntok/2), terminate rANS/bypass reads on truncation instead of allowing
  an infinite zero-renormalization loop, and require exact token consumption.
  The reusable API now supplies global slot tables for arbitrary valid nsub;
  32/64/128 retain shared-table fast paths.

## 2026-08-06: qmaps and full adjacent-candidate RDOQ
- Format v1.5 adds one nonzero Q2.6 step multiplier per chunk. CPU encode,
  CPU decode, hybrid parsing, compact GPU decode, and hostile validation agree.
- A generic log-variance activity map was tested rather than promoted by
  intuition. At near-matched dev rate, it lost about 0.1 dB global PSNR while
  shifting substantial quality toward surface bricks; uniform q remains the
  default. Explicit qmaps remain valuable for ROI/task objectives.
- RDOQ now greedily evaluates magnitude-1/current/magnitude+1 for every AC,
  including rate changes in the coupled following run/EOB and ctx2 previous-
  magnitude state. Full 512-brick matched rate: baseline q2 24.16x/42.91 dB/
  .9846 SSIM/1.404 MAE/P99 5/max18; q1.83+rdo.15 24.18x/43.00 dB/.9846/
  1.390/P99 5/max17. Blocking amplification 1.3422->1.2758x. Encode cost is
  ~2.4x, so default OFF.

## 2026-08-06: out-of-core and downstream gates
- Fresh full-corpus pack: 512 L0 entries, 122 KNOWN_ZERO, 26.9 MB payload,
  39.9x including zeros. `remote-batch` decoded requested bricks 0..7 with
  three total ranged GETs (footer, index tail, one coalesced payload); results
  matched local shard decode byte-for-byte. A four-job pthread ring overlaps
  network fetch with decode/write.
- `tools/ink_metric.py` compares the same detector on original and decoded
  inputs using AUROC, Dice/F1, confusion counts, probability MAE/P99/max and
  threshold-flip rate. The gate is implemented and synthetic-tested. No real
  ink number is claimed without a paired label projection and checkpoint.

## 2026-08-10: turnip miscompile in GPU encode tokenize (fixed on-device)
- The merged full-GPU-encode pipeline passed on llvmpipe but failed on
  Adreno X1-85/turnip: every substream containing a nonzero AC flagged err=1
  at its first level token (deterministic; flat substreams unaffected, which
  masqueraded as an even/odd-sub parity pattern on the dev brick).
- Root cause: turnip miscompiles `if (!f(...) || !g(...))` where both calls
  carry side effects through inout params (bypass bit-writer state). glslc -O
  vs no -O made no difference; llvmpipe executes the same SPIR-V correctly.
- Fix: sequentialize the two calls with explicit bools in encode_tokenize
  (`okm` then conditional `oks`), no stream semantics change. After the fix
  the full GPU encoder is 8/8 byte-identical to the CPU encoder on-device;
  decode legs and public-API smoke all PASS at <=1 LSB.
- gputest now dumps per-substream submeta (ntok/bpos/err/ptr) and
  model_status when a brick's encode status is bad on a mapped-memory iGPU.

## 2026-08-10: tifxyz surface-parameterization codec (new feature)
- New `src/tifxyz.c` + `c5dc tifxyz-{pack,unpack,verify}`: compresses the
  3-plane float32 tifxyz format (x/y/z.tif mapping (u,v) -> volume coords,
  -1.0 = invalid, meta.json carried verbatim). Own minimal TIFF reader
  (classic+BigTIFF, LZW, floating-point predictor 3, tiles+strips) and
  uncompressed BigTIFF writer — no libtiff dependency; output re-read
  bit-exact by tifffile.
- Measured on 3 real segments (PHercParis4, 7.91um meshes): first-order u/v
  steps are ~20 voxels (scale 0.05 grid); a masked parallelogram predictor
  drops residuals to ~0.15-0.4 voxel. Cross-component residual correlation
  is negligible (|r| <= 0.2), so planes code independently — measured, not
  assumed.
- Entropy stack reused: HybridUint tokens (NTOK=40) + static rANS
  (rans_encode_multin, nway=2), 7 context models (mask runs + per-component
  calm/rough). Mask coded as alternating RLE.
- Ratios (vs 12 B/px raw): lossless bit-exact 2.74-2.78x; quantized
  q=1/64 voxel (maxerr 0.0078) 5.0-5.3x; q=1/16 6.9-7.5x; q=1/4 10.6-11.8x.
  Baselines on the same data: original LZW tifs 0.99-1.6x, zstd-19 1.3-1.4x,
  xz -6 1.8-1.9x. Speed 1T: ~194 MB/s encode, ~246 MB/s decode (jordi 26MB).
- Quant mode is exact power-of-two scaling: dequant (float)(dv/2^k) is
  exact, re-encoding decoded output is idempotent (tested). Quantized
  |v|*2^k must stay < 2^36 or encode fails cleanly.
- Hostile: 200-trial random bit-flip fuzz on decode must not crash/hang;
  zero-length mask runs rejected except leading-invalid case. Gates: release
  + dev ASan/UBSan + TSan all green (5/5 tests incl. new tifxyz suite).

## 2026-08-17: lossless label-volume codec (new feature, "C5L1")
- New `src/label.{h,c}` + `c5dc label-{pack,unpack,info,verify}` +
  `c5d-label-bench` + `tests/test_label.c` + `fuzz/d_label.c`. A label brick
  is dim^3 x nchan typed integer channels (u8..i64; 14-digit segment-id
  timestamps fit); a channel may be MASKED by a lower channel (stored only
  where the mask is nonzero, decodes 0 elsewhere): fg / recto-on-fg /
  winding-on-recto / ink-on-recto / segment-id-on-fg is the intended shape.
  Per channel: sorted brick palette in the header (distinct labels are
  readable without decoding), per 16^3 chunk SAME/NEW-constant tokens or
  per-voxel causal-neighbour match symbols + static rANS, nsub independent
  substreams, chunk-level random access (`c5d_label_decode_chunk`), decode
  of a channel subset. Spec: spec/format.md "Label bricks".
- Corpus for the bake-off (no public label volumes on hand): labels DERIVED
  from real c5d-encoded prediction bricks in ~/r3d-data — fg = surface
  prediction >= 128, recto = fg && !fg[x-1] (mask fg), comp = 26-connected
  component id (u32, mask fg), seg = 20230827161847 + comp*1000003 (u64,
  mask fg), wind = floor((.7x+.5y+.3z)/40)-2 on recto (i16, mask recto);
  ink = ink3d prediction >= 128 (u8). 12 random 128^3 bricks each of
  paris4-surf L0 (clean surfaces), s1-surf L1 (dense, noisy), paris4-ink3d
  L0. Baselines: zstd-1/-19 on the raw channel bytes, c5d lossless (u8).
- Context-model bake-off (s1 L1, 3 bricks; total bytes fg/recto):
  candidates = W,N,U,P only, ctx = nd x prev-symbol class: 160K/108K.
  9 causal nbrs + majority-vote ranking, same ctx: 183K/98K (votes alone
  HURT fg: ranking without configuration context loses information).
  + face-neighbour state (W,N,U each equal/differs/absent = 27) x nd3: 132K/34K.
  x diag-agree count (7): 123K/26K.  x in-plane(3) x z-plane(5) agree
  counts = 405 ctx (ADOPTED): 123K/22K.  Full 9-bit equality template
  (512..3072 ctx) is WORSE (133K-137K / 27K-43K): table cost + dilution.
  Prev-symbol context: neutral once configuration contexts exist (dropped).
  Model tables are transmitted sparsely (only used models, only tokens up to
  the last used one, LEB128) — with 405 contexts this matters for sparse
  bricks (comp/seg channels are 1-4 KB per brick, ~1/3 of it tables).
- Palette-index escapes coded as rank among non-candidate entries: binary
  channels never spend bits on the escape value; segment ids cost log2(npal)
  bits per escape instead of ~30-bit deltas.
- Defaults measured (paris4 L0 12 bricks, joint bytes): nsub 8/16/32 =
  98.1/101.7/108.7 KB (~3.5% per doubling; multi-thread decode 188/240/306
  MB/s) -> nsub 16. rans nway 1 vs 2: +3.4% bytes for no measurable decode
  speed here -> nway 1 (params allow 2/4).
- Results (bits per voxel per channel; ratio vs zstd-19 / zstd-1):
  paris4-surf L0: fg 0.0223 (2.9x/5.6x), recto 0.0058 (6.6x/12x), comp
  0.0011 (99x/173x), seg 0.0011 (118x/242x), wind 0.0020 (17x/33x); joint
  0.032 bpv for all five channels = 496x vs the 16 B/voxel raw set.
  s1-surf L1 (dense, noisy threshold): fg 0.167 (1.5x/3.7x), recto 0.033
  (5.2x/13x), comp 0.0031 (139x), seg 0.0033 (182x), wind 0.0073 (24x).
  paris4-ink3d L0: ink 0.0059 (2.25x/3.4x). c5d's grayscale lossless mode
  is 1.5-2.5x WORSE than zstd-19 on binary masks (it is a residual coder),
  so it is not the answer for labels.
- Speed (Core Ultra 9 275HX, WSL2, release): per 128^3 brick x 5 channels
  encode 14 ms / decode 8 ms with all threads on paris4 L0 (1T 52/40 ms);
  dense s1 L1 1T ~120/100 ms (hot spot: candidate gather + ranking, ~12 ns
  per coded voxel; SIMD/branchless candidate build is the obvious next
  step). Encode phases: parallel per-substream distinct-value scan ->
  merged sorted palette -> parallel tokenize straight from the typed source
  (no index volume) -> tables -> parallel rANS. Serial palette pass was 3x
  slower multi-threaded before this restructuring (54 -> 146 MB/s).
- Hostile: 300 random bit flips + all truncations of a 3-channel stream
  never crash/hang (test), libFuzzer d_label 4 min clean, decoders check
  ntok exactness, palette monotonicity/type fit, escape/new indices < npal,
  directory/table sizes. Gates: release quick suite 6/6, dev ASan+UBSan
  6/6, TSan test_label clean, clang-tidy 0 errors on new files.

---

# volcomp v1 (2026-09-01) — measured decisions during the port

Corpus: 54 tune + 54 held-out real PHercParis4 128³ chunks (masked volume,
levels 0–2, disjoint z-ranges; one contiguous 2×2×2 octet per split for the
chunk-seam metric). 1 thread, clang 23, `-O3 -march=native`, Core Ultra 9
275HX under WSL2 (no hardware PMU; timings are medians, ±10% noise).

## Block-local deblock — no candidate beats "no filter" (gate failed)

Face filter removed (decision 11). Candidates evaluated on the tune set at
q ∈ {2,4,8} against c5d `c649fcf` with its neighbour-aware face filter:

| candidate | q8 PSNR | SSIM | MAE | blocking amp | 128/16-plane ratio |
|---|---|---|---|---|---|
| c5d face filter (needs neighbours) | 35.41 | .9465 | 3.022 | — | 3.08× (unfiltered chunk faces) |
| none | 35.24 | .9433 | 3.098 | 1.548 | 1.00 |
| A: taper toward the block's own LS line fit (7 variants: A∈{.375,.5,.625,.75}, windows 2..13/2..9/1..14, 1–2 layers) | 35.16 (best) … 34.87 | .9421 … .9361 | 3.130 … 3.260 | 1.579 … 1.683 | 1.00 |
| B: Wiener shrink of AC coefficients, β=0.083 / 0.25 / 0.5 | 35.27 / 35.25 / 35.12 | .9437 / .9438 / .9427 | 3.085 / 3.086 / 3.128 | 1.526 / 1.492 / 1.462 | 1.00 |
| C = A∘B (β=.25, A=.5, 1–2 layers) | 35.08 / 35.06 | .9406 / .9404 | 3.160 / 3.170 | 1.574 / 1.572 | 1.00 |

Interpretation: a block-local operator cannot know which side of a block
edge carries the quantisation noise; every spatial candidate (A) raises MAE
and blocking. B is a coefficient-domain reconstruction tweak, not a seam
fix, and its gain is the same degree of freedom as the reconstruction
offset. The chunk-face seam ratio is 1.00 for every variant by construction
(c5d: 1.04× at q2, 1.72× at q4, 3.08× at q8 with its filter skipping chunk
faces). Outcome reported to the owner; no filter frozen at the time of writing.

## Reconstruction offsets: split |level|=1 / ≥2 (adopted)

`dz_dq` single 0.26 → (|1| 0.15, ≥2 0.30). Tune set, q2/4/8/16: PSNR
+0.03 dB, SSIM +0.0002…+0.0008, MAE −0.4%, blocking amplification
1.548→1.527 (q8), max error +1…+2, bytes unchanged (decoder-only). Pairs
(0.20,0.26), (0.20,0.30), (0.15,0.30) were within 0.01 dB of each other.

## DC round-to-nearest (adopted, no measurable effect)

DC dead zone 0.2 → 0.5. Identical ratio/PSNR to 4 digits: the DC step is
`q/8`, i.e. sub-LSB at every q ≥ 1, so DC was already effectively lossless.
Kept because it is the correct rule and costs nothing.

## Error tails are high-contrast interior, not the air mask

Voxels with |err| > 3q at q = 2 / 8 / 32: 1.24% / 0.10% / 0.0004% of all
voxels; of those, only 1.2% / 0.7% / 0.2% lie in blocks touching the mask
(mask-touching blocks hold 7.2% of voxels), while 86% / 93% / 99.6% lie in
blocks whose source range exceeds 128. A min/max clamp per block would not
help; the tail is DCT ringing on genuine high-contrast structure. No format
change.

## Compact frequency tables (adopted)

Per model: u32 presence bitmap + LEB128 per nonzero frequency, sum verified
== 4096 by the decoder. 640 B raw → ~130 B per chunk (measured on the 512
c5d production tables). With the 8-byte header and 8-byte directory
entries, volcomp streams are 0.6% (q2) … 5.6% (q32) smaller than c5d's at
identical quantisation.

## Speed (plain C, no intrinsics)

Baseline c5d 1T native, filter off, tune set: encode 296–499 MB/s, decode
377–760 MB/s (q2…q32). volcomp after the items below, same host: encode
361 / 543 / 648 MB/s, decode 410 / 767 / 1276 MB/s at q2 / q8 / q32.

- Factored 16-point DCT (Lee recursion, generated straight-line code, 32
  multiplies) replacing two 8×8 dot-product banks: equivalent to 1.1e-4.
- Sparse inverse: x and y passes skip coefficient planes with no nonzero
  (z-plane mask from dequant). Flat blocks skip the transform entirely
  (encoder and decoder).
- Division-free rANS encode (ryg reciprocal symbols).
- Nonzero extraction via vectorised 32-wide compare masks + ctz, mapped
  through an inverse scan table and sorted, instead of a 4096-entry
  permutation gather.
- Codegen fragility found and fixed: the float→u8 scatter compiled to
  per-byte `vpextrb` extracts in most translation units (2× slower decode:
  ~570 vs ~1000 MB/s at q8 for the *same* source). Splitting clamp/convert
  from the i32→u8 narrowing copy makes the vectoriser emit packed
  truncations everywhere. Also: the transform must stay out-of-line
  (`noinline`) and the entropy helpers force-inlined (+9% decode); `flatten`
  on the public functions made things slower.
- 46-entry reciprocal step table indexed by radius inside the quantise
  loop was *slower* than the 16 KB natural-order table (−15% encode);
  reverted.
- 10-bit coarse symbol table (+ ≤3-step `cum` fix-up) vs 12-bit direct
  table: throughput within noise (±5%); model memory 42 KB → 11 KB and model
  build 4× cheaper, which is the fixed cost of every `volcomp_decode_block`
  call. Adopted.
- No hardware PMU in this VM: cache behaviour inferred from working sets
  (decoder ≈ 11 KB models + 16 KB block + 5 KB scatter staging; encoder
  ≈ 16 KB block + 16 KB quantised + 16 KB reciprocals + 8 KB positions).

## tANS replaces rANS (adopted, format v1)

10-bit tANS (FSE table build, zstd spread step, two interleaved lanes sharing
one forward LSB-first bit stream) vs the 12-bit 2-lane byte rANS, tune set,
pinned, decode-only harness best of 2: q2 581 vs 573 MB/s, q8 973 vs 890,
q32 1772 vs 1467 (+1 / +9 / +21 %); bytes −0.1 / −0.4 / −1.3 % (tables cost
fewer bytes at 10 bits; precision loss is smaller than that saving). Without
the bulk 7-byte bit refill tANS was *slower* than rANS (490 / 925 / 1356):
the per-symbol bit read dominates once the multiply is gone. 11-bit tables
were not better. Decode table memory: 10 × 4 KB.

## Optional post-decode deblock (`volcomp_deblock`, not in the format)

c5d's quant-gated 4-tap face filter applied to every 16-plane of an assembled
decoded region. Tune set, per-chunk application: q8 PSNR 35.27 → 35.43 (c5d
in-format filter: 35.41), SSIM .9439 → .9468, MAE 3.085 → 3.012, blocking
amplification 1.53 → 1.32; q32 29.99 → 30.43 dB, amplification 1.93 → 1.13;
q2 unchanged. Cost ≈ 10 % of decode time (scalar). Applied per chunk it
leaves chunk faces unfiltered (octet 128/16-plane ratio 1.2–2.0); applied to
the assembled region all planes are treated alike.

## DCT orthonormal scale folded into the quantiser tables (adopted)

The generated Lee kernels drop their 16 output/input scale multiplies; the
per-coefficient scale (1/4 per zero index, √2/4 otherwise, per axis — four
distinct values) is multiplied into the encoder's reciprocal step table and
the decoder's dequantised coefficient. Bit-identical streams and
reconstructions; decode ≈ +5 % (within run-to-run noise on this host).

## Encoder-side RDOQ — not adopted (v1)

c5d measured full adjacent-candidate RDOQ at +0.09 dB / −1.0 % MAE at
matched rate for 2.4× encode time. A restriction to the |level| = 1 bin
within a 1.3× encode budget is expected to keep roughly half of that (~1 %
bytes). Left out of v1 as too small for its complexity; the research survey
(docs in plan) ranks it as the first encoder-only item if more ratio is
wanted later. Sign-data hiding (~0.5 %) and significance-map coefficient
coding (2–5 % claimed, decode cost) were likewise deferred.

## AVX2 pass (adopted, 2026-09-01; no format change)

Explicit intrinsics, AVX2+FMA required (`#error` otherwise). Every item was
measured with a paired in-process harness (both headers linked into one
binary, alternated per chunk, best of 7) because wall-clock A/B on this WSL2
host varies ±15 % run to run. All adopted items are bit-identical to the
plain-C output (stream and voxel hashes over the tune set).

- One 8-lane `__m256` Lee DCT kernel (the scalar straight-line body with the
  type swapped) for all three axes: y/z as lane passes, x through two 8×8
  register transposes per eight lines. Replaced the SLP-vectorised per-line
  x pass and the compiler's lane loop: encode +9/+18/+18 %, decode
  +12/+19/+27 % (q2/q8/q32).
- AVX2 gather (`cvtepu8_epi32`, −128) with the constant-block test fused in
  (byte min/max) and AVX2 scatter (+128.5, clamp, `cvttps`, two `packus`, one
  lane permute per row pair); explicit quantise + nonzero-mask loop (fmadd,
  `cvttps`, sign fold, `cmpeq`+`movemask`): encode +11/+17/+30 %, decode
  +7/+14/+18 %.
- Decoder: z pass out of place (coefficient planes → voxel buffer, zero
  planes not loaded), so only the block's dirty planes are re-zeroed instead
  of a 16 KB memset per block; x pass skips a plane's upper eight lines when
  no coefficient has y ≥ 8: decode +1/+12/+16 %.
- `volcomp_deblock`: 16 lanes of 16-bit arithmetic for the two axes whose
  lines are contiguous (gates as masks, `sign_epi16`, `packus`); the scalar
  clamps were dead (results stay within [P0, Q0]). Filter cost 10 % → 2–8 %
  of decode. Exactness vs the scalar reference is a unit test.
- Rejected: 8-byte bulk flush in the bit writer (encode −0.4…−2.8 %, noise
  or slightly worse) and a 4 KB scan-ordered step×scale index table replacing
  the radius / zero-count arithmetic in the dequant loop (decode +0…+5 %,
  changes rounding). Neither helps because the token loops are bound by the
  tANS dependency chain (state → `dt[state]` → bits → next state, with the
  two lanes coupled through the shared bit stream and the run→context→level
  dependence), not by ALU work. The remaining decode lever is a format change
  that gives each lane its own bit stream (estimated +15–20 % decode); not
  done.
- Net, tune set, final vs plain-C: encode +22/+36/+45/+60/+57 %, decode
  +19/+18/+31/+39/+50 % at q 2/4/8/16/32.
