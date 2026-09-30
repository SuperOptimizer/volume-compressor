# volcomp fact sheet (2026-09-30)

Compiled read-only from the repo at /home/forrest/volume-compressor (paths below relative to it unless absolute), the memory notes in /home/forrest/.claude/projects/-home-forrest-usrm2/memory/ (abbreviated MEM/), the villa-volcomp checkout, `gh pr view 1704`, and a few live HTTP listings of the mirror on 2026-09-30. Every number carries `file:line`. Where sources disagree both are listed (section 8).

## 1. What volcomp is

volcomp is a single-header C compressor for uint8 volumetric data (micro-CT), built for the Vesuvius scrolls (README.md:3-5). It has one quality knob `q` and one stream format, no threads and no dependencies beyond libc/libm (README.md:3-5). `q = 0` is exact lossless; `q = 1..255` is a 3-D DCT with a dead-zone quantiser that trades a controlled error for 20-200x smaller greyscale CT and probability maps (README.md:9-12, 25-27). Extra modes store binary masks and thresholded surface predictions (README.md:14-20). Data is cut into 128^3 chunks (one zarr chunk each), packed into 1024^3 zarr v3 `sharding_indexed` shards, so a viewer can HTTP-Range-fetch just the chunks it needs (README.md:29-30; spec/format.md:166-177). Current version 1.3.0, format revision 4 (volcomp.h:59, 71; spec/format.md:1, 51).

## 2. The format

### Geometry (README.md:29-30; spec/format.md:8-19)
- block 16^3 (transform + random-access unit), chunk 128^3 = 8x8x8 blocks (independently decodable), shard 1024^3 = 8x8x8 chunks (zarr v3 sharding_indexed). A chunk has 32 substreams of 16 blocks each.
- Decoding one 16^3 block touches one substream (<= 16 blocks) and equals the same region of the full decode (README.md:52-53). Not true for surface chunks (surface_mode.md:323-324).

### Lossy DCT (q >= 1) (spec/format.md:57-64, 66-150)
- Per block: 3-D orthonormal DCT-II (separable 16-point), dead-zone quantisation, zigzag scan tokenisation, 2-lane tANS (FSE-style, TABLE_LOG 10, per-chunk frequency tables, 10 context models) with a raw bypass bit stream (spec:59-64, 95-105).
- Step law: DC step = q/8; AC step = q*(1+r)^0.65 with r = x+y+z; encoder AC dead zone floor(|X|/step+0.2), DC rounds to nearest; decoder reconstruction offsets 0.15 (|level|=1) / 0.30 (>=2) (spec:131-139).
- Arithmetic is IEEE single; AVX2 and C kernels may differ by at most +-1 in a voxel (spec:148-151). README says with clang/GCC on x86-64 they produce identical bytes (README.md:106-107).
- No in-format post-filter, so no chunk seams (spec:153-164).
- q is stored as q_raw = q*256 (Q8.8) in the 8-byte header (spec:30). API range VOLCOMP_Q_MIN 1, VOLCOMP_Q_MAX 255, lossless 0 (volcomp.h:76-81). README: error percentiles scale with q, P99 ~ 2.5q on scroll data (README.md:110-111).

### What q means (README.md:9-12, 109-111)
- q = 0: exact (class maps, masks, anything where values are the data).
- q 1..255: quantiser step in voxel units, typical 2..32. Production CT store uses q 8 at level 0, then 4 / 2 / 1 for levels 1 / 2 / >=3 (tools/export/README.md:7-8; PR body says level 0 Q=8, level 1 Q=4, level 2 Q=2, levels 3-5 Q=1, from `gh pr view 1704`).

### Modes (header byte 5) (spec/format.md:27-29, 44-55)
| mode | meaning | revision/version added |
|---|---|---|
| 0 | lossy DCT | 1.0 (rev 1) |
| 1-3 | lossless (CODED / RAW / CONST) | rev 1 slots; implemented 1.0.x lossless work (spec:222-229) |
| 4 | binary mask, 2x2x2 majority pool (64^3 grid) range-coded, decoded trilinearly | rev 2, volcomp 1.1.0 |
| 5 | spatially lossless binary mask (full 128^3) | rev 3, volcomp 1.2.0 |
| 6 | surface chunk (exact threshold) | rev 4, volcomp 1.3.0 |
The version byte stays 1; the mode byte is the compatibility mechanism, older readers reject newer modes cleanly (spec:40-55).

### Lossless mode (spec/format.md:220-276; README.md:35-38)
- Per-block prediction one step back along z (y then x at edges), zigzag residuals, per block CONST / SPARSE / DENSE / RAW, same tANS. All-flat chunk = 9 bytes. Never larger than raw + 8 bytes (spec:274-276). Plain C, no float math, identical streams on every build (README.md:110-112).

### Mask modes (spec/format.md:278-460; README.md:39-48)
- Mode 4: 2x majority pool to 64^3, adaptive binary range coder over 12 causal neighbours (4096 contexts), decode trilinearly interpolated to 128^3; max chunk 32,777 bytes (spec:300-303).
- Mode 5: same coder over the full 128^3; decodes to exact 0/255; max 262,153 bytes (spec:420-421).
- Probability adaptation shift schedule {1,1,2,2,3,...} (spec:~570 region; docs/measured.md:789-790 adoption note "shift schedule").

### Mode 6 exact-threshold surface mode (spec/format.md:461-587; docs/surface_mode.md:169-233)
- Chunk = 16-byte header (thr, law, margin M, base length) + DCT base with a FLAT step law (every AC step = q, DC q/8) + refinement (spec:461-490; surface_mode.md:171-177).
- Refinement: one context-coded bit for each voxel whose base value lies within margin M of the threshold saying whether the base put it on the wrong side; a flipped voxel decodes to thr or thr-1 (surface_mode.md:179-186). 3,072 contexts (surface_mode.md:188-195). Far candidates coded only in flagged 8^3 blocks (surface_mode.md:196-204).
- The base is reconstructed bit-exactly on every build with strict IEEE kernels (no FMA, fixed butterfly order, table steps) so encoder and decoder agree (surface_mode.md:206-218).
- Guarantee: for every voxel `(source >= thr) == (decoded >= thr)`; mask, skeleton, distance field are exact (spec:461-470; surface_mode.md:220-222). Default thr 128. q range 2..255 (README.md:145; python/README.md:65-66). A surface chunk at q is finer than mode-0 at the same q: surface q48 is about the size of mode-0 q16 (spec:578-584).
- A revision-3 reader refuses mode-6 chunks cleanly (surface_mode.md:225-227).

### zarr v3 / sharding integration (spec/format.md:166-177; python/README.md:13-16)
- Shard = up to 512 chunk streams back to back followed by an index of 512 x {u64 offset, u64 nbytes} plus a CRC-32C of the 8192 index bytes; a missing chunk (including all-zero source chunks) is offset = nbytes = 0xFFFF...FFFF and reads as fill_value 0 (spec:168-176).
- Byte-compatible with `sharding_indexed`, `index_location = "end"`, `index_codecs = [bytes(little), crc32c]`, chunk shape 128^3, shard 1024^3, inner codec `"volcomp"` with `{"q": q}` (spec:173-177). Codec chain is volcomp only, no zstd (surface_mode.md:229-230; MEM/usrm2-store-format.md:15-18 area).
- Verified live on the mirror: PHerc0343P CT level-0 `zarr.json` has `sharding_indexed`, inner `volcomp` `{"q": 8.0}`, index codecs `bytes` little + `crc32c`, shape [5398, 5057, 5057] (http listing 2026-09-30).
- Level directory names: CT levels are `0..5` (live listing); m7 prediction stores name levels by pitch (`8.64/ 17.28/ ... 1105.92/`, live listing; MEM/m7-wholevol.md convention line).

### Decode-only deblocking / smoothing (docs/deblocking.md; README.md:150-177)
Not part of the format; a reader opts in (`volcomp_decode_smooth`, Python `set_read_smoothing(2)` / `VOLCOMP_SMOOTH=2`, CLI `decode --smooth`) (README.md:174-177; deblocking.md:24-31).
1. Smooth the chunk's interior block faces; 2. project every block's coefficients back into the quantisation cells they were decoded from, so the result is still a reconstruction the stream allows (README.md:158-165; deblocking.md:259-275).
- CT recommendation: gated filter at strength 2 with zero guard for q >= 4; off below q4 (deblocking.md:15-17, 275).
- Probability maps: Gaussian seam blur sigma 0.6 with zero guard; surface chunks keep the exact threshold (deblocking.md:19; 272-273).
- Exact zeros (masked air) stay 0 with the guard; `volcomp_deblock` now skips face positions that touch an exact zero by default (README.md:164, 171-172; deblocking.md:20, 254-258).
- Encode-only error feedback and per-chunk signalled strength were not adopted (deblocking.md:34-48).

## 3. Measured numbers

### 3a. Ratio / quality / speed vs q on real CT, PHercParis4 2.4 um masked, 128^3 chunks
Host: Core Ultra 9 275HX under WSL2, clang 23, -O3 -march=native, 1 thread, medians of 3, +-10% noise (docs/BENCHMARKS.md:10-14). Held-out set: 54 chunks z in [350,451] (BENCHMARKS.md:3-6). Table = BENCHMARKS.md:22-33 (volcomp rows; also quoted in the villa PR body):

| q | ratio | PSNR dB | SSIM | MAE | P90 | P95 | P99 | max | enc MB/s | dec MB/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 2 | 17.56 | 41.33 | .9840 | 1.573 | 3 | 4 | 7 | 49 | 471 | 702 |
| 4 | 30.05 | 38.04 | .9700 | 2.282 | 5 | 6 | 11 | 77 | 574 | 906 |
| 8 | 52.92 | 35.00 | .9463 | 3.217 | 7 | 9 | 15 | 121 | 784 | 1375 |
| 16 | 95.58 | 32.22 | .9088 | 4.416 | 10 | 13 | 21 | 143 | 1033 | 2135 |
| 32 | 176.5 | 29.67 | .8532 | 5.911 | 13 | 17 | 28 | 155 | 1113 | 2484 |

Same held-out streams with per-chunk `volcomp_deblock` (BENCHMARKS.md:55-61): PSNR 41.34 / 38.10 / 35.17 / 32.54 / 30.09 dB at q 2/4/8/16/32; SSIM 0.9840 / 0.9709 / 0.9494 / 0.9159 / 0.8656; decode+filter 646 / 957 / 1323 / 1804 / 2425 MB/s (BENCHMARKS.md:50-53).
Tune set (54 chunks used for design), BENCHMARKS.md:65-71: ratio 18.58 / 31.96 / 56.47 / 102.4 / 190.1 and PSNR 41.48 / 38.25 / 35.27 / 32.52 / 29.99 at q 2/4/8/16/32.
Error tails: voxels with |err| > 3q: 1.24% (q2), 0.10% (q8), 0.0004% (q32), sitting in high-contrast interior blocks (BENCHMARKS.md:83-85).
Kernel sets (BENCHMARKS.md:94-101): AVX2 enc/dec 480/665 (q2), 868/1406 (q8), 1222/2424 (q32) MB/s; plain C 295/368, 514/610, 642/761. README says C kernels run at 40-60% of AVX2 speed (README.md:108).
Time split at q8: decode ~1.4 ms per 2 MiB chunk, encode ~2.4 ms (BENCHMARKS.md:108-113).

### 3b. Per-scan CT tables (512^3 cubes of ORIGINAL uint8 CT, 128 MiB raw, chunks 128^3, plain decode / deblocked; docs/comparison/README.md)
Cube sizes and ratios (size / ratio / PSNR dB / MAE / P99 / max; deblocked PSNR in parentheses):

PHerc0500P2 0.55 um (README.md lines 9-16; volume [12320,10813,3891]):
| q | size | ratio | PSNR (db) | MAE | P99 | max |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 4.50 MiB | 28.4x | 48.3 (48.3) | 0.71 | 2 | 10 |
| 2 | 2.58 | 49.6x | 46.2 (46.3) | 0.93 | 3 | 15 |
| 4 | 1.50 | 85.2x | 44.0 (44.2) | 1.22 | 4 | 26 |
| 8 | 0.88 | 145.7x | 41.6 (42.1) | 1.59 | 6 | 34 |
| 16 | 0.51 | 250.3x | 39.2 (39.8) | 2.08 | 8 | 41 |
| 32 | 0.30 | 428.0x | 36.7 (37.5) | 2.72 | 12 | 66 |

PHerc1667 1.13 um (README.md lines 36-43; [42209,22122,20276]):
| q | size | ratio | PSNR (db) | MAE | P99 | max |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 3.87 MiB | 33.1x | 50.7 (50.7) | 0.49 | 2 | 17 |
| 2 | 2.46 | 52.0x | 48.3 (48.5) | 0.69 | 3 | 21 |
| 4 | 1.57 | 81.7x | 45.6 (46.0) | 0.99 | 4 | 27 |
| 8 | 0.97 | 131.7x | 42.7 (43.4) | 1.40 | 5 | 29 |
| 16 | 0.59 | 217.6x | 39.8 (40.7) | 1.99 | 7 | 41 |
| 32 | 0.35 | 361.5x | 36.9 (37.9) | 2.77 | 11 | 67 |

PHercParis4 2.40 um (README.md lines 63-70; [75784,32693,32693]):
| q | size | ratio | PSNR (db) | MAE | P99 | max |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 10.24 MiB | 12.5x | 46.5 (46.5) | 0.90 | 3 | 16 |
| 2 | 6.50 | 19.7x | 43.3 (43.3) | 1.35 | 5 | 20 |
| 4 | 4.04 | 31.7x | 39.9 (40.0) | 2.01 | 7 | 26 |
| 8 | 2.45 | 52.3x | 36.6 (36.9) | 2.97 | 10 | 41 |
| 16 | 1.42 | 90.1x | 33.3 (33.9) | 4.32 | 15 | 60 |
| 32 | 0.78 | 163.4x | 30.3 (31.0) | 6.12 | 21 | 92 |

PHerc0191 9.36 um (README.md lines 90-97; [18977,8387,8387]):
| q | size | ratio | PSNR (db) | MAE | P99 | max |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 16.29 MiB | 7.9x | 45.1 (45.1) | 1.06 | 4 | 28 |
| 2 | 10.84 | 11.8x | 41.1 (41.1) | 1.73 | 6 | 41 |
| 4 | 6.89 | 18.6x | 37.2 (37.2) | 2.75 | 9 | 47 |
| 8 | 4.13 | 31.0x | 33.4 (33.5) | 4.27 | 15 | 69 |
| 16 | 2.30 | 55.7x | 29.8 (30.1) | 6.41 | 23 | 100 |
| 32 | 1.17 | 109.3x | 26.7 (27.1) | 9.20 | 33 | 142 |
(All in docs/comparison/README.md; raw numbers in docs/comparison/report.json; line numbers above are approximate table positions in that file: PHerc0500P2 9-16, PHerc1667 36-43, PHercParis4 63-70, PHerc0191 90-97.)

### 3c. Deblocking gains (docs/deblocking.md)
- Masked CT q>=4: +0.1 to +0.6 dB PSNR, no chunk worse; about 5 ms per chunk (README.md:167-168; deblocking.md:17). PSNR gain table deblocking.md:114-123 (e.g. smooth gated2 zg: interior q8 +0.36, q16 +0.56; tune q16 +0.61).
- Unguarded old filter at q16 on the tune octet: edge MAE 3.18 -> 4.89, chunk faces 2.8x rougher, 4/8 chunks worse; guarded +0.52 dB, none worse (deblocking.md:20; table lines 159-161).
- Speed (bench_smooth.c): CT q8 plain decode 2,192 MB/s vs gated2 zg 383 MB/s (5.5 ms per chunk), Gaussian 155 MB/s (13.5 ms) (deblocking.md:292-299).

### 3d. Probability / surface stores (q8 and surface mode)
- Existing q8 stores cost 0.06-0.08 bits/voxel, 100-125x smaller than u8 (surface_mode.md:79-82). Per store table surface_mode.md:67-74 (e.g. recto eval.zarr 0.0689 bpv, 17,735 B per stored chunk; m7 p4val256 0.0775 bpv, 19,919 B).
- Same probabilities in other forms, bits/voxel recto / m7 (surface_mode.md:87-94): float32+zstd-19 20.2 / 21.6; float16+zstd-19 12.9 / 11.2; u8+zstd-19 3.94 / 1.70; volcomp lossless 2.50 / 1.03; volcomp q8 0.065 / 0.052; threshold mask zstd-19 / mode 5 0.076 / 0.025 and 0.047 / 0.012.
- q8 loses the threshold: recto 0.66% voxels on the wrong side of 0.5, dice 0.981; m7 0.18%, dice 0.992; skeleton F1 0.52 (recto), 0.64 (m7) (surface_mode.md:99-104).
- Surface mode results (surface_mode.md:244-254 recto, 258-268 m7; 12 cubes of 256^3, 201 M voxels per teacher): recto q8 0.0647 bpv, dice 0.9807, band MAE 2.61; surface q64 0.0503 bpv = 0.777x q8, dice 1, band MAE 3.51, skeleton F1 1. m7 q8 0.0523 bpv, dice 0.9917, band MAE 2.21; surface q64 0.0370 bpv = 0.708x q8, band MAE 3.64. Surface q32: recto 1.174x, m7 1.014x q8 size with q8-like soft error. Surface q96 0.665x / 0.592x; q128 0.614x / 0.529x.
- Cost of exactness: at q64 the refinement is 26-51% of a recto chunk and 12-27% of an m7 chunk (surface_mode.md:298-307).
- Throughput, single thread (surface_mode.md:316-319): mode 0 q8 recto enc/dec 1,672 / 3,455 MB/s, m7 1,904 / 4,184; surface q64 recto 406 / 958, m7 619 / 1,519. About 1 GB/s per core, a 1024^3 shard decodes in about a second (surface_mode.md:321-323).
- Mask modes on the PHercParis4 recto (19.6% foreground): mode 4 0.0057 bits/voxel (1,500 B per chunk), mode 5 0.0255 bpv (6,682 B), packbits+zstd-19 0.062, ideal 12-neighbour model 0.022 (spec/format.md:387-389, 451-456; measured.md:761-767, 803-813). Mode 4 is 11x smaller than packbits+zstd-19 (README.md:42-44); mode 5 is 2.4x smaller (README.md:47-48). Thresholding mode 4 decode changes 1.7% of voxels, mean 1.06 voxels from the published boundary (spec:390-392).
- Mask speed: mode 4 encode 1.2-2.5 G, decode 0.8-1.9 G voxels/s (measured.md:783-786); mode 5 encode 247-289 M, decode 227-269 M voxels/s (measured.md:~820).
- Label planes (ink 3d prediction, 24 planes, q8): 9.5 KB per plane, 220x, 35x smaller than the blosc-zstd chunks it ships in; MAE 0.5, P99 6, PSNR 43.7 dB (README.md:207-210).

### 3e. Lossless (BENCHMARKS.md:116-170)
Synthetic: 2-class 5% mask 169x, 8-class Voronoi 53x, smooth SDF-like 3.8x, noise 1.00x (raw+8), synthetic CT 1.8x (BENCHMARKS.md:126-133). Real TSM channels vs zstd (BENCHMARKS.md:144-148): faces_valid 8,346 B (251x) vs zstd-19 10,692; rv_class 31,069 B (67x) vs zstd-19 20,482; sdf_in 663,690 B (3.2x) vs zstd-19 510,196. volcomp is 22% smaller than zstd-19 on the mask and 44% smaller than zstd-3 on the class map but loses to zstd-19 by 34% and 23% on the two busier channels (BENCHMARKS.md:150-157).

### 3f. Comparison against other codecs (what exists in the docs)
- Against its predecessor c5d at the same q (BENCHMARKS.md:16-46): volcomp bytes 0.6% (q2) to 6.9% (q32) smaller; decode 2.0-4.0x, encode 1.4-2.2x faster; volcomp has no in-format filter, c5d has one.
- c5d0 baseline was ~2x better ratio than zfp-accuracy at iso-PSNR (measured.md:26); a later doc-honesty correction says the zfp advantage was ~10x on mixed corpora but ~4.2x on matched PSNR (measured.md:82-83). These are c5d-lineage numbers, not volcomp-on-scroll-CT numbers.
- DCT-16 beat a CDF 9/7 wavelet on real corpus by +105% / +135% / +190% ratio at iso-PSNR (measured.md:~46-52); 16^3 beat 8^3 by +27% to +47% (measured.md:~53-58).
- Against blosc/zstd: labels (README.md:208), masks and lossless above. No JPEG-XL, HEVC or AV1 measurement exists in the docs (grep of README.md, docs/*.md found none; HEVC/AV1 appear only as literature in deblocking.md:324-331).

### 3g. Published whole-volume stores (live checks 2026-09-30 by summing the HTTP directory listings; sizes rounded to 0.1 KiB per file)
PHerc0343P 8.640 um CT `20250521134555-8.640um-1.2m-116keV-masked.zarr` (shape [5398,5057,5057] = 138.04 Gvox):
| level | shape | size on mirror |
|--:|---|--:|
| 0 | [5398,5057,5057] | 239.3 MiB (67 files) = 550x vs raw u8 |
| 1 | [2699,2529,2529] | 95.4 MiB |
| 2 | [1350,1265,1265] | 30.6 MiB |
| 3 | [675,633,633] | 6.3 MiB |
| 4 | [338,317,317] | 0.9 MiB |
| 5 | [169,159,159] | 0.1 MiB |
Whole CT pyramid 372.6 MiB. The m7 probability store `20250521134555-surface-20260925170000-surface-m7-L0-prob.zarr` is 252.3 MiB in 300 files (MEM/m7-wholevol.md says 252 MiB for 0343P; the request said 253 MiB, see section 8).
Other sizes from notes: PHerc0009B m7 store 1.33 GiB, 971 files (MEM/m7-wholevol.md, 20:15 update); 41 upstream published masks exported mask-lossless: 112 GB, 39,461 units (MEM/usrm2-unified-plan.md line "2026-09-20 15:40 CDT"); projection 32.7 GB vs 7.8 GB for the 2x mode (measured.md:~840); recto four fleet levels ~19 GB (MEM/usrm2-unified-plan.md, 2026-09-19 evening entry); whole-bucket claim from PR 1704: ~300 TB uncompressed to ~6 TB, ~44x (`gh pr view 1704` body); export README says 64 volumes, ~760 TB logical at level 0 plus pyramids (tools/export/README.md:3-4). These two totals disagree (section 8).

## 4. Downstream validation (all measured, as documented)
- Recto teacher on decoded CT vs on raw CT, interior 512^3, central 256^3 scored (deblocking.md:178-196): dice at 0.5 = 0.9777 (q4), 0.9511 (q8), 0.9273 (q16) with plain decode; mean abs diff 2.40 / 5.85 / 8.59 (u8); corr 0.99775 / 0.98812 / 0.97752. Deblocking moves this by ~0.2% dice either way at q4/q8 (plain decode best at q8: 0.951 vs 0.948-0.949); at q16 every filter helps (0.927 -> 0.931-0.933) (deblocking.md:50-61). Recommendation: keep plain decode for teacher inference on q8 stores (deblocking.md:58-59).
- Earlier check: teacher on volcomp q8 input vs published raw-data prediction, dice 0.87-0.91 on the eval box; deblocking changes nothing (MEM/usrm2-findings.md:12). Different box/metric from the 0.951 above.
- Thresholded probability stores: q8 costs 0.66% / 0.18% flipped voxels (recto/m7); mode 6 makes it 0 (surface_mode.md:99-104). Deblocking on m7 q16: dice 0.9913 -> 0.9931, iso-surface shift 0.095 -> 0.075 voxel (deblocking.md:19, 232-235).
- m7 whole-volume production: deblocked CT input (smoothing 2) changed pred MAE 4.10 -> 3.70 with dice unchanged (MEM/m7-wholevol.md, 20:20 update). m7 dice vs stride: s96 vs s144 0.82 (0343P L0) / 0.946 (Paris4 L2); stride 128 vs 96 dice 0.946 on the Paris 4 box: the mask moves with window placement, not a compression effect (MEM/m7-wholevol.md). TRT vs torch m7 corr 0.99996; store == direct blend after q8, only q8 DCT blocking at 128-faces (same file).
- Segmentation/tracing effect: no measurement of compression on vc3d tracing or segment quality is recorded in any listed source. The PR itself says models not finetuned on compressed data are sensitive to it, artifacts visible at q8 above 2x zoom at native resolution, finetuning on compressed data is in progress and unfinished (`gh pr view 1704` body). No measured recovery number exists in the sources.

## 5. Ecosystem
- **C library**: `volcomp.h` single header, all `static`, AVX2+FMA chosen at runtime on x86-64, plain C elsewhere (README.md:102-108). `volcomp_label.h` for multi-plane label chunks (README.md:179-202; spec/format.md:188-218). CLI `volcomp` (encode/decode/verify/shard-pack/label-*/surface-pack/occupancy, README.md:220-230; tools/cli/). Requires C23 clang/GCC (README.md:233-236).
- **Python**: package `volcomp-zarr` 1.0.0 (python/pyproject.toml), module `volcomp_zarr`, requires python >= 3.10, optional `zarr>=3`, numpy; registers entry point `zarr.codecs` `volcomp = volcomp_zarr:VolcompCodec` (python/pyproject.toml). Build: `cmake --preset release && cmake --build --preset release --target volcomp_shim`, copy `libvolcomp.so` into `python/volcomp_zarr/`, `pip install ./python[zarr]` (python/README.md:36-40). Use `zarr.open(...)` on a mirror store, or `VolcompCodec(q=8)`, `VolcompCodec(q=0)`, `mode="mask"|"mask-lossless"|"surface"` as serializer (python/README.md:13-16, 27-31, 61-71). Chain must be volcomp only: pass `compressors=None` (python/README.md:69-70; MEM/usrm2-store-format.md). Also ctypes API encode/decode/decode_block/deblock/decode_smooth, `set_read_smoothing` (deblocking.md:276-280).
- **villa / vc3d PR**: https://github.com/ScrollPrize/villa/pull/1704 "3D DCT based compression backend for VC3D", state OPEN, mergedAt null, +5,692/-192, 21 files (`gh pr view 1704`). Adds vendored volcomp (`volume-cartographer/libs/volcomp`), C++ shim `utils/volcomp_codec.{hpp,cpp}`, `VcDataset`/`Volume`/`ZarrChunkFetcher` support, C++ zarr reader/writer changes for `index_location` end and crc32c index, tests `test_vcdataset_volcomp` and `test_zarr_shard_index_location` (file list from `gh pr view 1704`). Branch commits (villa-volcomp checkout): vendor 1.3.0 rev 4 (878732f1a, 2026-09-25), "zarr: write shards with the index at either location" (de22d9382, 2026-09-25: fixes writers that assumed a front index, honours trailing crc32c, refuses unproducible index codecs), dropped recompress option, portability re-vendor. Vendored README: zarr v3 published exports use 1024^3 shards, `index_location: "end"`, crc32c index; ZarrArray reads/writes both locations (villa-volcomp/volume-cartographer/libs/volcomp/README.md:44-49). Decode smoothing exposed but not called by any VC reader (same README:34-40).
- **Browser viewer**: `web/dist/volcomp_viewer.html`, one ~170 KB self-contained page (HTML, JS, worker, WASM decoder inlined base64), works from `file://`, streams zarr v3 stores from the mirror by HTTP Range (needs CORS `*`, Range, 206) (web/README.md:3-11). Shows three orthogonal slices with prediction overlay; sample/volume/overlay pickers, auto level by pixel size, deblock toggle (gated 2xq for CT, Gaussian 0.6 for predictions), paste box for any URL, URL hash for links, keyboard/mouse table (web/README.md:13-65). Reads the trailing shard index by Range and checks crc32c; merges ranges with gap <= 64 KB, reads <= 8 MB; worker pool; 1.5 GB decoded LRU (web/README.md:76-99). Build with emcc via `web/build.sh` (web/README.md:101-117). Verified: wasm decoder bit-identical to native C kernels on 7 chunks (web/README.md:160-166).
- **Mirror**: https://dl.ash2txt.org/community-uploads/forrest/volcomp/ mirrors the AWS bucket layout: `<Scroll>/volumes/<vol>.zarr/<level>/...` and `<Scroll>/representations/predictions/surfaces/<name>.zarr/<pitch>/...` (PR body; tools/export/README.md:8-9; web/README.md:16-18). CT levels are numbered; prediction levels are named by voxel size in um. Export pipeline: coordinator + workers, S3 to SFTP (tools/export/README.md:14-28).
- **Published scans** (live directory listing 2026-09-30, 39 sample directories): PHerc0009B, 0125, 0139, 0172, 0175A, 0175B, 0191, 0211, 0257, 0268, 0306B, 0332, 0343, 0343P, 0358, 0483A, 0483B, 0490A, 0490B, 0500P2, 0800, 0813, 0814, 0826, 0841, 0846A, 0846B, 1203, 1218, 1299, 1447, 1451, 1545, 1667, PHercMAN5, PHercMANB, PHercMANBp, PHercParis3, PHercParis4. The PR says all 39 scrolls and ~65 volumes; the notes list 27 fine-pitch volumes across 16 samples (MEM/vesuvius-volume-inventory.md).
- **Prediction stores**: (a) the 41 upstream published masks (+ Paris 4) re-exported as mode-5 lossless (Paris 4 recto as mode 4), 8 levels each, complete 2026-09-20 (MEM/usrm2-unified-plan.md). (b) Our m7 whole-volume probability stores (`...-surface-20260925170000-surface-m7-L<k>-prob.zarr`, uint8 p*255, q8, shards 1024>>k min 128, 8 levels by pitch): MEM/m7-wholevol.md. The live listing shows 58 prob directories (one `.part` in flight): m7-L0 for 0009B, 0125, 0172, 0175A, 0175B, 0191, 0211, 0257, 0268, 0306B, 0343, 0343P, 0358, 0483A, 0483B, 0490A, 0490B, 0500P2, 0800, 0813, 0814, 0826, 0841, 0846A, 0846B, 1203, 1218, 1447, 1451, 1545; m7-L1 to L4 fine-pitch stores for 0009B, 0139 (x6), 0332, 0343P, 0500P2 (L1/L2/L4), 0814, 0841, 0846A, 1203, 1299, 1451, 1667 (L2, L3), MAN5, MANBp (L2, L3), Paris4 (L2, L2, L3). The full list is from a live query on 2026-09-30 (58 lines) and can change; in the notes, tier 4 was 57/61 uploaded and PHercParis3 L2 was the last (MEM/m7-wholevol.md, 2026-09-30 incident). Scope: all 31 8-9 um volumes (27 already had upstream m7 L0 masks; only 0343P, 0009B, 1451, 0172 were originally in scope) plus 30 fine-pitch scans at the level nearest 9 um (MEM/m7-wholevol.md). 0343P and 0009B: stride 96, plain CT decode; later volumes stride 128 with CT smoothing 2.

## 6. Limitations and caveats stated in the docs
- Lossy: no error bound guarantee; hostile content can ring arbitrarily (fuzz/rt_chunk.c header). P99 ~ 2.5q is an empirical scroll-data statement (README.md:110-111).
- Artifacts visible at q8 above 2x zoom at native resolution; models not fine-tuned on compressed data are sensitive to it (PR body).
- q8 CT is fine for viewing; q >= 16 has visible blocking; blocking amplification 1.19 / 1.36 / 1.53 / 1.76 / 1.93 at q 2/4/8/16/32 (BENCHMARKS.md:78-79). Chunk faces are only smoothed by decode_smooth from one side; region deblock needed for them (deblocking.md:105-110; web/README.md:178-180).
- q8 probability stores lose the 0.5 threshold; use mode 6 for anything that thresholds (surface_mode.md:96-104, 292-296).
- Mode 6: encode ~4x slower than q8 (0.4-0.6 GB/s); decode 2.8-3.6x slower; block decode decodes whole chunk; existing q8 stores were not re-encoded; no rate control; no second threshold (surface_mode.md:321-324, 359-368). External readers (vc3d) needed to learn mode 6; a revision-3 reader rejects it (surface_mode.md:359-361). The vendored vc3d copy is now 1.3.0 (villa-volcomp commit 878732f1a).
- Mask mode 4 is not exact (1.7% of thresholded voxels differ, up to 2 voxels); mode 5 is (spec:390-392, 420-456).
- Lossless loses to zstd-19 on busy channels (BENCHMARKS.md:150-157).
- Decode smoothing is opt-in; not in the format; the Gaussian prediction smooth can differ by +-1 LSB between AVX2 and C builds (web/README.md:181-183). Encode-only and signalled deblocking rejected (deblocking.md:34-48).
- Measured only on Linux x86-64 (WSL2, +-10% timing noise); arm64 compiles and uses C kernels (BENCHMARKS.md:10-12; README.md:233-236). Viewer tested only in headless Chromium; slices nearest-neighbour; overlay aligned by pitch only, OME translations ignored (web/README.md:175-199).
- Corpus: quality tables use PHercParis4 chunks and 4 512^3 cubes; generalisation to other scans is shown only in docs/comparison (per-scan PSNR varies from 26.7 dB (9.36 um, q32) to 50.7 dB (1.13 um, q1)).
- Encoding q per level is an owner choice (8/4/2/1); "downscaling averages noise away, so coarser levels keep more" (tools/export/README.md:8).

## 7. Existing comparison images: /home/forrest/volume-compressor/docs/comparison/
Four cube directories (each 153 files), plus README.md and report.json (per-q bytes, ratio, PSNR, MAE, P99, max, deblocked variants). Directories: `PHerc0500P2_0.55um/`, `PHerc1667_1.13um/`, `PHercParis4_2.40um/`, `PHerc0191_9.36um/` (docs/comparison/README.md:3-5). Every image is a lossless 512x512 PNG of a 512^3 cube of ORIGINAL uint8 CT.
Naming scheme `<view>_<version>[_db].png`:
- views: six cube faces `z0 z511 y0 y511 x0 x511` and three centre planes `zmid ymid xmid`;
- versions: `raw`, `q1 q2 q4 q8 q16 q32` (plain decode, what a zarr reader gets), and `_db` = decoded with the optional `volcomp_deblock()` post-filter (README.md:3);
- `<view>_all.png` = raw and six q levels side by side, no deblock; `<view>_all_db.png` = the same with deblocking (README.md:3);
- `diff_<zmid|ymid|xmid>_q<N>.png` = error overlays: raw slice in grey with a red overlay whose opacity is min(1, 4|raw-decoded|/255) (error 32 = 50% red, >= 64 solid), 2x magnified to 1024^2, plain decode only (README.md:113-115); generated by tools/diff_images.py. Per cube: 9 views x (raw + 6 q + 6 db + all + all_db) = 135 plus 18 diffs = 153 (verified by `ls`).
Generator: tools/compare_images.py; git: 018bef0, 61ff40f, 36daf54, e565113 (2026-09-18).

## 8. Discrepancies and unverified claims
- **Requested numbers not found in the listed sources**: "upstream binary 1.05 GB / 363 Gvox" and "dice 0.978" and "PSNR 32.3-32.5 dB deblocked" for PHerc0343P appear nowhere in the memory notes (rvsm-project.md contains no volcomp, compression or 0343P entries) or repo docs. Nearest documented values: recto teacher dice at q4 0.9777 (deblocking.md:182, not 0343P, not m7); recto q8 threshold dice 0.9807 (surface_mode.md:99); Paris4 held-out q16 PSNR 32.22 plain / 32.54 deblocked (BENCHMARKS.md:30, 60). Do not quote them for 0343P without a fresh measurement.
- **Verified by live check (not in notes)**: 0343P CT L0 = 138.04 Gvox raw -> 239.3 MiB (550x, includes shard indexes); m7 L0 prob store 252.3 MiB (note says 252 MiB; the request said 253 MiB, a rounding of 252.3 MiB would be 252).
- Ratio at q8: 52.92x on held-out Paris4 chunks (BENCHMARKS.md:28) and 52.3x on the Paris4 cube (comparison/README.md:68) vs "~44x" whole-bucket blended over q 8/4/2/1 levels (PR body) vs "~40x at q=8 (~40 dB PSNR)" in the vendored villa README (libs/volcomp/README.md:4-5) which conflicts with BENCHMARKS' 35.00 dB at q8 (BENCHMARKS.md:28). Prefer the BENCHMARKS values.
- "P99 ~ 2.5q" (README.md:13) vs measured P99 15 at q8 (1.9q) and 28 at q32 (0.9q) (BENCHMARKS.md:28, 33).
- Recto q8 dice vs float source: 0.9807 (12 cubes, surface_mode.md:246-247) vs 0.981 (surface_mode.md:99-100) vs 0.9872 (evalbox + r32k only, deblocking.md:202).
- Recto q8 bits/voxel: 0.0647 (12 cubes, surface_mode.md:247) vs 0.0689 (eval.zarr store, surface_mode.md:69) vs 0.065 (rawcost, surface_mode.md:93).
- Whole-bucket size: PR body ~300 TB uncompressed, 39 scrolls / ~65 volumes; tools/export/README.md:3-4 says 64 volumes, ~760 TB logical at level 0 plus pyramids; unresolved.
- Teacher dice with q8 CT input: 0.87-0.91 vs published raw-data prediction (MEM/usrm2-findings.md:12) vs 0.951 vs raw-CT prediction (deblocking.md:187); different boxes and references.
- Frequency table sums: spec says 1024 (spec:117-118); measured.md:637-638 describes an older 4096 (superseded by tANS switch, measured.md:676-685).
- README compares q8 P99 "P99 ~ 2.5q" and label store "35x smaller than blosc-zstd" (README.md:208) only for label planes, not CT.

## 9. What is verified by tests and fuzzing
- ctest (`cmake --preset release && ctest --preset release`, README.md:223) runs: test_bitstream (bit writer/reader), test_tans, test_tables, test_codec (layout, round trip quality, block == full, encode bound, hostile inputs), test_golden (frozen v1 stream and decoder output; also runs with C kernels as `test_golden_c`), test_lossless (1000 fuzzed chunks byte-exact), test_mask (modes 4/5), test_surfchunk (mode 6, exact threshold, golden stream bit-identical on every build; `test_surfchunk_c` too), test_deblock (masked-edge regression, zero guard), test_label, test_shardpack, test_surface (surface-pack). Files: tests/test_*.c. Python: python/test_ctypes.py, python/test_zarr_roundtrip.py. Web: web/test (wasm_check.mjs, ref_checksums.py, browser_check.cjs). Export: tools/export/test_export.py (offline end-to-end with a synthetic bucket).
- Fuzz (libFuzzer, preset `fuzz`): fuzz_d_chunk (decoder on hostile input), fuzz_rt_chunk (round trip incl. q=0, block decode == full decode), fuzz_rt_mask, fuzz_rt_mask_ll (README.md:232-233; fuzz/*.c; CMakeLists.txt:105-111). Presets also include asan+ubsan `dev` (README.md:232). The suite passes under asan+ubsan (BENCHMARKS.md:168-170; last recorded on the lossless work of 2026-09-07). I did not re-run the tests in this research pass.

## 10. Git history (63 commits; `git log`)
- 2026-09-01 d948fec Initial commit; b6435e9 volcomp v1.0 single-header AVX2 u8 codec; a656753 export pipeline, Python ctypes binding, zarr v3 codec.
- 2026-09-03 label codec `volcomp_label.h`.
- 2026-09-04 70320c5 AVX2 at runtime, plain C kernels elsewhere.
- 2026-09-07 da3d1f9 lossless mode q=0; 456ff7d label q per plane (label format version 2).
- 2026-09-18 comparison images and error overlays (018bef0, 61ff40f, 36daf54, e565113).
- 2026-09-19 180abdd mask chunks (mode 4, format revision 2, volcomp 1.1.0); ramp export (4e3ad54) then replaced by mask export.
- 2026-09-20 fc01075 mode 5 lossless mask (revision 3, 1.2.0); pool-levels / upload-tree tooling.
- 2026-09-24 18a711e mode 6 surface chunks; 7c14567 spec revision 4 section 13; 069a4bd deblock zero guard and `volcomp_decode_smooth`; f31b0e2 docs/deblocking.md (this is the commit vendored into villa).
- 2026-09-25 3e549a3 web viewer (HEAD of volume-compressor main). VOLCOMP_VERSION_STRING "1.3.0", VOLCOMP_FORMAT_VERSION 1, VOLCOMP_FORMAT_REVISION 4 (volcomp.h:59-71).
