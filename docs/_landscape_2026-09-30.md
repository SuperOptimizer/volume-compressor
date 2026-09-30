# Landscape of 3-D grayscale / volumetric compression, and where volcomp sits (2026-09-30)

Scope: web survey done 2026-09-30 for the contest writeup. volcomp facts come from `docs/_factsheet_2026-09-30.md` (abbrev. FS) and are the only numbers measured on papyrus CT. Everything else is quoted from a cited source, on different data (volumetric floats from simulations, clinical CT, ultrasound video, 34 um mouse micro-CT), and is NOT comparable to volcomp's numbers. No source below benchmarks any codec on scroll CT. Items marked [recollection] are from background knowledge and were not re-verified in this pass; treat as unsourced.

Tooling caveat: the pages were read through a summarising fetcher, so numbers are what the summary reported from the page; a few (noted) come from search-result snippets only. Verify against the papers before printing any of them in the writeup.

## 0. volcomp in one paragraph (measured, FS section 3)

16^3 3-D DCT, dead-zone quantiser, tANS entropy coder, 128^3 chunks, 1024^3 zarr v3 shards, one knob q. On PHercParis4 2.4 um held-out chunks: q4 30.1x / 38.0 dB, q8 52.9x / 35.0 dB, q16 95.6x / 32.2 dB (FS 3a). Per-cube, same scan: q8 52.3x / 36.6 dB (FS 3b). Other scans at q8 (cube tables): 0.55 um 145.7x / 41.6 dB, 1.13 um 131.7x / 42.7 dB, 9.36 um 31.0x / 33.4 dB. So "about 50x at 35 dB" holds for 2.4 um; at finer pitch the same q gives more ratio and more dB, at coarser pitch less of both. Decode 1.4 GB/s at q8 (single thread), encode 0.78 GB/s. Lossless q=0 about 1.8x on synthetic CT (FS 3e; no real-CT lossless ratio is recorded).

## 1. General-purpose lossless (zstd / blosc / lz4 in zarr, OME-zarr)

- Blosc is the default compressor of zarr-python (v2), and Blosc-zstd is the usual recommendation for microscopy volumes. A bioRxiv benchmark of lossless codecs for microscopy data recommends BLOSC+ZSTD as best ratio over its test conditions: "A Comparison of Lossless Compression Methods in Microscopy Data Storage Applications", 2023, https://www.biorxiv.org/content/10.1101/2023.01.24.525380.full.pdf (fetch returned HTTP 429; claim is from search-result text, per-dataset ratios not retrieved). A 2026 holotomography OME-Zarr study likewise found Blosc-zstd and zstd among the best across bandwidths (https://ideas.repec.org/a/plo/pone00/0351560.html, snippet only).
- Typical ratio on real CT-like noisy 8-bit data: about 1.3-2x. This is the range given in the task brief; no source retrieved in this pass states it, so treat as folklore until the bioRxiv table is read. It is consistent with volcomp's lossless mode on synthetic CT (1.8x, FS 3e), where noise is the floor.
- Why they cap out: byte-oriented LZ+entropy coders exploit repeated byte strings and skewed histograms, not spatial smoothness. Sensor noise makes the low bits incompressible, so lossless ratios are bounded by noise entropy whatever the codec. Lossy is the only way to reach 20-500x.
- Strength: universal, zero surprise, every zarr reader supports it, decode is fast (lz4 and zstd are GB/s class [recollection]). Random access: per zarr chunk, and with zarr v3 sharding per inner chunk.
- Where volcomp stands (measured): lossless mode beat zstd-19 by 22% on a sparse mask and 44% over zstd-3 on a class map, but lost by 23-34% on busier channels (FS 3e). Label planes 35x smaller than the blosc-zstd chunks they shipped in (FS 3d).

## 2. 2-D image codecs per slice

- Baseline facts [recollection]: JPEG (8x8 DCT), JPEG 2000 (wavelet, ISO/IEC 15444-1, mandated in DICOM as a transfer syntax), JPEG XL (VarDCT + Modular), WebP, AVIF/HEIF (AV1 intra). All treat one slice as an image. imagecodecs exposes JPEG, JPEG 2000, JPEG XL, JPEG-LS, AVIF, WebP, ZFP etc. as numcodecs, so they can be dropped into zarr: https://pypi.org/project/imagecodecs/ (2023-2025 releases; the release page also notes Zarr 3 fixes) and https://zarr.dev/codecs-registry/ (lists ZFPY, JPEG2K, JPEGXL, AVIF).
- What they lose by ignoring z: the cleanest quantitative evidence is the 3-D vs 2-D JPEG 2000 comparison on thoracic CT (16-detector, 0.75 mm collimation, ratios 4:1 to 64:1) presented by J. Johnson et al. at RSNA 2004: for 0.75 mm slices at 8:1 and 16:1, 3-D beat 2-D by 0.5 and 1.0 JND, i.e. 1-3 dB PSNR; the gap grows with thinner slices and higher ratios; thin-slice 3-D matched 10 mm slice quality (https://www.auntminnie.com/imaging-informatics/enterprise-imaging/pacs-vna/article/15571809/3d-jpeg-2000-compression-shines-in-multislice-ct-data ; also https://diagnosticimaging.com/view/3d-outperforms-2d-compressing-thinner-ct-slices). Clinical CT, 4-64:1, not comparable to volcomp; direction of the effect is what transfers. Papyrus at 2.4 um is isotropic and highly correlated across z, so the loss should be at least this large. Not measured by us.
- A micro-CT study (34 um mouse vasculature, 16-bit, 579x355x681) compared JPEG 2000 (ratios 8, 9, 12:1) with HEVC (16, 34, 58:1) and concluded up to ~34x for HEVC and ~12x for JPEG 2000 preserved vascular morphology (PMC10406799, J Digit Imaging 2023, https://pmc.ncbi.nlm.nih.gov/articles/PMC10406799 ; https://link.springer.com/article/10.1007/s10278-023-00800-5). It notes JPEG 2000 encodes slices independently (parallel, no inter-slice dependency) while HEVC's motion estimation makes random access depend on GOP configuration.
- Random access: a per-slice codec gives 2-D tile access only; a 128^3 chunk read means decoding 128 slices' tiles. Aligned with an x-y viewer, hostile to xz/yz slicing and to 3-D chunk tracers.
- No JPEG XL / WebP / AVIF measurement on scroll CT exists in the volcomp docs (FS 3f). This is a gap: the writeup should not claim a comparison it did not run.

## 3. Video codecs with z as time (H.264, HEVC, AV1, VVC)

- Mechanism: inter-frame prediction with motion compensation exploits correlation between adjacent slices, in effect a 2.5-D predictor. Costs: sequential dependency within a GOP, so reading slice k needs decoding from the previous intra frame; a decode of a scattered 128^3 chunk means decoding whole GOPs of full-resolution frames (x-y extent is the full slice unless tiled). ERCIM's healthcare codec benchmark used an intra update every 32 frames aligned with the GOP for random access (https://ercim-news.ercim.eu/en119/research-and-innovation/the-battle-of-the-video-codecs-in-healthcare, 2019; ultrasound video, not CT).
- Results on medical volumes:
  - Same ERCIM study (10 ultrasound videos, 560x448): VVC saves 36% bitrate vs AV1 at equal quality; AV1 needs 18% more than HM and 4% more than x265 (2019). Not volumetric CT.
  - Lossless HEVC for volumetric medical images with least-squares inter-slice prediction: >44% bitrate reduction vs DICOM-recommended encoders and 13.8% vs standard lossless HEVC for 8 bpp volumes (search-result summary of "A method to improve HEVC lossless coding of volumetric medical images", Signal Processing: Image Communication; https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/229/paper-detail/40213 ; year not confirmed).
  - The 34 um micro-CT study above: HEVC ~34x usable vs JPEG 2000 ~12x (2023).
  - VVMIC (learned volumetric coder) reports 7.67% / 17.23% / 8.84% BD-rate savings vs VVC on MRNet axial/coronal/sagittal, evaluating JP2K, JPEG-XL, HEVC, VVC as baselines on MRI and abdominal CT (https://arxiv.org/html/2412.09231v1, 2024).
- Structural limits for this use case [recollection, not sourced]: video codecs are 8-bit-luma oriented with hardware/library decoders that do not decode arbitrary 128^3 sub-volumes; no zarr codec exists for HEVC/AV1 in the numcodecs registry (registry lists AVIF, an intra/still profile: https://zarr.dev/codecs-registry/). Browser support is via WebCodecs for whole frames, not sub-volumes. Encoders are slow (AV1/VVC), decoders are fast only in hardware.
- Honest caveat: video codecs are probably the strongest competitor on ratio (they have RDO, adaptive block sizes, in-loop filters, and decades of tuning). No head-to-head on papyrus exists. volcomp's docs contain only literature mentions of HEVC/AV1 (FS 3f); a tiled-HEVC vs volcomp experiment at iso-PSNR is the obvious missing measurement.

## 4. 3-D native codecs

### JPEG 2000 Part 10 (JP3D)
ISO/IEC 15444-10, first 2008, current edition 2011: lossless and lossy coding of volumetric images, extending Parts 1 and 2, with extensions listed as variable DC offset, arbitrary wavelet kernels, multi-component transforms, non-linear transforms and region-of-interest (https://www.iso.org/standard/40024.html). It is a 3-D wavelet plus EBCOT design; codestream random access by precinct/code-block is built in. Few implementations exist [recollection]; no numcodecs/zarr codec. The 2004 RSNA study above is the 3-D vs 2-D evidence (1-3 dB).

### 3-D DCT / 3-D wavelet coders (3D-SPIHT, 3-D SPECK, etc.)
[recollection] A large 1990s-2000s literature; embedded bitplane coders (SPIHT, SPECK) over 3-D wavelet trees give progressive streams with good rate-distortion but whole-volume or large-subband coding; block-based variants give random access at some cost. No source was retrieved for specific numbers in this pass. volcomp's own doc says its 16^3 DCT beat a CDF 9/7 wavelet by +105% / +135% / +190% ratio at iso-PSNR on its real corpus (FS 3f, measured.md), measured with the c5d lineage on a different corpus than the scroll table; internal measurement, not a peer-reviewed comparison.

### ZFP (Lindstrom, IEEE TVCG 2014)
- Blocks of 4^d values (4x4x4 in 3-D), fixed-rate mode maps each block to a fixed number of bits, giving read/write random access at block granularity; also fixed-precision, fixed-accuracy and lossless modes. Up to ~2 GB/s throughput on the project page; 4 bits per double is 16x and visually indistinguishable from 64-bit (https://computing.llnl.gov/projects/zfp ; https://github.com/LLNL/zfp ; paper: https://www.academia.edu/63125780/Fixed_Rate_Compressed_Floating_Point_Arrays). BSD-3, C/C++/Python/Fortran bindings; OpenMP and CUDA.
- ZFP is the closest architectural cousin to volcomp (small independent transform blocks, embedded coding, fast). Differences: 4^3 vs 16^3 blocks (larger blocks capture more of the smooth low-frequency structure; volcomp measured 16^3 beating 8^3 by 27-47% ratio, FS 3f, internal), bit-plane coding vs context-coded tANS with a dead-zone quantiser, floating-point-first design. In the fixed-rate mode a block is fixed size, so no entropy gain from flat regions; accuracy mode is variable-rate.
- Numbers on other data: TTHRESH paper Figure 12 (Density volume, float): at 47.1 dB TTHRESH 1,855:1 vs ZFP 107:1; at 38.6 dB TTHRESH 3,494:1 vs ZFP 160:1 (https://ar5iv.labs.arxiv.org/html/1806.05952, TVCG 2019). Simulation floats, very smooth: not comparable to 8-bit noisy CT.
- volcomp's own docs: c5d0 (predecessor) about 2x better than zfp-accuracy at iso-PSNR, later corrected to ~4.2x on matched PSNR vs ~10x on mixed corpora (FS 3f). Predecessor lineage, not volcomp on scroll CT; no direct ZFP-on-scroll run is recorded.
- Integration: numcodecs has zfpy; imagecodecs has ZFP; both appear in the zarr codecs registry (https://zarr.dev/codecs-registry/). Zarr v3 status not confirmed by that page.

### SZ / SZ3
Error-bounded prediction-based compressors (Lorenzo / interpolation predictor, linear quantiser, Huffman + zstd). SZ3 (Liang et al., IEEE Trans. Big Data 2023; arXiv 2111.02925, https://arxiv.org/abs/2111.02925) is a modular framework; abstract claims up to 20% ratio improvement at equal distortion vs prior state of the art. Snippet-level figures: on Miranda at PSNR 90, SZ3-Interp ratio 47 vs SZ3-LR 30; NeurLZ paper reports bit-rate reductions vs ZFP of 91.3/37.4/36.8% (SZ3) on Nyx/Miranda/Hurricane (https://arxiv.org/pdf/2409.05785). Float HPC data. Strength for our use: a hard per-voxel error bound (volcomp has none: P99 ~ 2.5q empirical only, FS 6). Weakness: serial-dependency predictors give slow decode relative to block transforms [recollection], and SZ is chunk-level, not block random access. In serial mode ZFP is faster than SZ3 (STZ paper snippet, https://arxiv.org/pdf/2509.01626).

### MGARD (Gong, Chen, Whitney et al., SoftwareX 2023; https://arxiv.org/pdf/2401.05994, https://arxiv.org/abs/2401.05994)
Multigrid hierarchical decomposition plus quantisation with finite-element-theory error guarantees, GPU kernels; targets structured/unstructured floating-point grids. Also does progressive refactoring. Numbers not retrieved (fetch exceeded size limit). Relevance: error-controlled (guaranteed) and progressive, but GPU/HPC oriented and float-first.

### TTHRESH (Ballester-Ripoll, Lindstrom, Pajarola, IEEE TVCG 2019, 26(9):2891-2903; https://arxiv.org/abs/1806.05952, https://github.com/rballester/tthresh)
Tucker/HOSVD tensor decomposition + bit-plane, run-length, arithmetic coding. Best ratio at low-to-medium bit rates and very smooth degradation, no block artifacts, resampling in compressed domain. Tests included 8-bit volumes (Foot, Engine, Teapot at 256^3) and 32/64-bit float volumes. Speed: Teapot (11.1 MB) 2.4 s compress / 1.0 s decompress; Isotropic-fine (512 MB) 61.5 s / 25.8 s; "between 0.5 and 2 orders of magnitude slower than the fastest, zfp" (ar5iv page above). That is roughly 20 MB/s decode versus volcomp's 1,400 MB/s single thread, about 70x. Global decomposition: no useful random access (a subvolume needs the whole core tensor and factor matrices), so it suits archiving, not chunk streaming. LGPL-3.0.

## 5. Neural / learned volumetric compression

- Coordinate-network (INR) approaches: Lu, Jiang, Levine, Berger, "Compressive Neural Representations of Volumetric Scalar Fields", EuroVis 2021 (https://arxiv.org/abs/2104.04523): an MLP maps (x,y,z) to a value, weights quantised; random-access evaluation without full decompression. Devkota & Pattanaik, "Efficient Neural Representation of Volumetric Data using Coordinate-Based Networks", Computer Graphics Forum 2023 (https://arxiv.org/abs/2401.08840): hash encoding + Reptile meta-learning for faster convergence. A 2025 Fourier-feature INR paper (https://arxiv.org/abs/2508.08937) benchmarks against TTHRESH, SZ3, ZFP; a search summary of the INR literature says INR frameworks reach 9.6x to 102.2x ratios beyond traditional compressors at comparable quality, but traditional compressors are far faster to encode; a sparse-training variant cut training from 30 to 11 minutes at compression rate 14 (search snippets, not verified in the papers). A snippet from a 2026 arXiv paper (https://arxiv.org/pdf/2607.20970) quotes TTHRESH 235 vs ZFP 148 vs SZ3 84 ratio on a combustion field at 42.25 dB; that paper could not be fetched (size), so treat as unverified.
- Video INR (NeRV, Chen et al., NeurIPS 2021, https://arxiv.org/abs/2110.13903): 38-132x faster decode than pixel-wise INRs; NeRV-Dec paper reports 11x faster loading than conventional codecs and 65x smaller (ECCV 2024, https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5618_ECCV_2024_paper.php). Small clips, RGB video, per-video training.
- Learned 3-D transform coders: VVMIC (arXiv 2412.09231, 2024) beats VVC by 7.7-17.2% BD-rate on MRI, using a volumetric autoencoder that models inter-slice redundancy.
- Costs for this use case: (a) encoding is training: minutes to hours per volume for MB-scale data, versus volcomp 0.8 GB/s encode; a 138 Gvox scan (FS 3g) is out of range by orders of magnitude. (b) Decoding is a network forward pass per voxel or per patch, needing a GPU for throughput; no source gave GB/s figures on CPU. (c) No zarr codec, no browser-friendly runtime beyond WebGL/WebGPU inference. (d) No hard error bound, fine-detail loss, per-dataset tuning. (e) The reported gains are on smooth simulation fields, not noisy CT where the noise floor dominates.
- Judgement: neural methods win reported ratios on smooth scientific fields and lose on decode cost, encode cost, and scalability to TB-PB. They are not a candidate for a 300 TB archive; they could matter for small hot regions.

## 6. Vesuvius Challenge / scroll-CT efforts

- Formats and hosting: OME-Zarr is the primary distribution format "because it is cloud-optimized (chunked, multi-resolution) and supports streaming / partial reads"; TIFF stacks sometimes; open data at s3://vesuvius-challenge-open-data/ and HTTP mirror https://dl.ash2txt.org/ (https://scrollprize.org/llms-full.txt, retrieved 2026-09-30). The llms-full text does not state the compressor; the "blosc" claim in the task brief is consistent with zarr-python defaults but is not confirmed by a Vesuvius source in this pass. (Check with the ash2txt store's .zarray / zarr.json before stating it.)
- Sizes reported in Vesuvius text: multi-terabyte CT scans (hiring post, https://scrollprize.substack.com/p/vesuvius-challenge-is-hiring); the first 3.24 um volume of PHerc 332 at 53 keV is 4.1 TB, and 36 volumes were planned in that release across 3.24 and 7.91 um, four energies (https://scrollprize.substack.com/p/new-scans-of-herculaneum-papyri-at); the 2023 Grand Prize region is 77 GB as zarr vs 389.9 GB as TIFF stack (llms-full.txt); Paris4 spiral annotations ~49.6 GB. Whole-bucket totals: the volcomp export PR says ~300 TB uncompressed (about 44x to ~6 TB with per-level q 8/4/2/1), while tools/export/README says 64 volumes and ~760 TB logical at level 0 plus pyramids; unresolved (FS sections 3g, 8). No official total was found publicly; do not print "hundreds of TB" without qualifying which number and source.
- Community compression: search found Forrest McDonald's vesuvius-c single-header C library that streams scroll data, and a mention of Oliver Daubney working on masking and compressing scroll data to improve accessibility; no published lossy-compression study or benchmark for scroll volumes was found. scroll2zarr (KhartesViewer, https://github.com/KhartesViewer/scroll2zarr) converts scroll data to zarr (compression details not checked). Also volcomp's PR into villa (ScrollPrize/villa#1704, open, +5,692/-192) is the in-ecosystem integration path (FS 5).
- Measured on papyrus: the 0343P 8.64 um CT L0 store is 239.3 MiB for 138 Gvox (550x, masked volume, mostly zeros; FS 3g). That ratio includes large masked-air regions and is not a codec ratio; quote the per-cube numbers (FS 3b) for codec behaviour and the store size only as a data point on masked scans.

## 7. Axes that matter for this use case

| Axis | Why it matters here | Best fit in the landscape |
|---|---|---|
| Random access at 128^3-chunk granularity | Tracers and viewers read scattered chunks | Per-chunk-independent designs: volcomp, ZFP fixed-rate (4^3), zarr chunk codecs (blosc, jpeg2k per chunk), JP3D precincts. Weak: video (GOP), TTHRESH/INR (global), SZ (chunk-level only) |
| Decode speed | Viewer latency; teacher inference on the fly | volcomp 0.7-2.5 GB/s single thread (FS 3a); ZFP up to ~2 GB/s (project page); TTHRESH ~20 MB/s from its own numbers; INR needs GPU |
| Encode speed | 300 TB to re-encode | volcomp 0.5-1.1 GB/s; ZFP fast; TTHRESH 0.5-2 orders slower than ZFP; INR training; AV1/VVC slow [recollection] |
| HTTP range streaming | Static hosting on S3/SFTP | zarr v3 sharding_indexed with trailing index: volcomp streams per-chunk by Range and its viewer merges gaps <= 64 KB (FS 5). Any codec inside sharding_indexed can do the same, so this is a zarr property, not a codec property |
| zarr / OME-zarr integration | Existing readers, vc3d | zstd/blosc native; zfpy, jpeg2k, jpegxl, avif via numcodecs/imagecodecs registry; volcomp via `zarr.codecs` entry point plus a C++ shim in the villa PR (open). Video, TTHRESH, INR: none |
| Browser decoding | Zero-install viewer | volcomp WASM decoder, ~170 KB single page, bit-identical to native on 7 chunks (FS 5). JPEG/WebP/AVIF native in browsers; JPEG XL not universal [recollection]; zstd/blosc via WASM builds [recollection] |
| Lossless / near-lossless for labels | Masks and class maps must round trip | volcomp q=0 and mask modes 5/6 (exact threshold); zstd/blosc lossless; lossy DCT loses the 0.5 threshold on 0.66% (recto) / 0.18% (m7) of voxels at q8 until mode 6 (FS 3d) |
| Error guarantee | Segmentation sensitivity | SZ/MGARD/ZFP-accuracy give bounds; volcomp does not (empirical P99); PR states models not fine-tuned on compressed data are sensitive at q8 (FS 6) |

## 8. Comparison table

"Ratio at ~35 dB" is filled only where a source gives it on CT-like data; everything else is "no source". Where a number is given on other data it is labelled.

| Codec | 3-D aware? | Random-access granularity | Ratio at ~35 dB on CT-like data | Decode speed | zarr integration | Browser support |
|---|---|---|---|---|---|---|
| volcomp q=8 (this work) | Yes, 16^3 DCT | 16^3 block (one substream <=16 blocks); chunk 128^3 independent | 52.9x at 35.0 dB, 2.4 um papyrus (MEASURED, FS 3a) | ~1.4 GB/s single thread at q8 (MEASURED) | Native codec `volcomp` in sharding_indexed; Python entry point; C++ in villa PR (open) | WASM, ~170 KB page (tested in headless Chromium only) |
| volcomp q=0 lossless | Prediction along z | chunk | ~1.8x synthetic CT (MEASURED; no real-CT value recorded) | plain C, no float | same | same |
| zstd / blosc-zstd / lz4 | No (byte-oriented; shuffle helps) | zarr chunk | not applicable (lossless), ~1.3-2x brief only, unsourced | GB/s class [recollection] | Native (zarr default) | WASM builds [recollection] |
| JPEG / WebP (2-D, per slice) | No | slice tile | no source | fast [recollection] | via imagecodecs (jpeg, webp) | native |
| JPEG 2000 per slice (DICOM) | No | tile | micro-CT 34 um: 8-12x at diagnostic-quality (PMC10406799, different data) | slow relative to DCT [recollection] | numcodecs jpeg2k | no native; WASM ports exist [recollection] |
| JPEG XL, AVIF/HEIF (2-D) | No | tile | no source | AVIF decode moderate [recollection] | imagecodecs jpegxl, avif (registry) | AVIF native; JXL partial [recollection] |
| HEVC / AV1 / VVC (z as time) | Partly (motion compensation between slices) | GOP (e.g. intra every 32 frames in ERCIM study); full-slice frames | micro-CT 34 um: 34x (HEVC) vs 12x (J2K) at diagnostic quality, PMC10406799, different data; not PSNR-matched to 35 dB | HW decode fast; sub-volume decode not supported | none found | WebCodecs, whole frames |
| JP3D (JPEG 2000 Part 10) | Yes (3-D wavelet) | precinct/code-block | thoracic CT: 1-3 dB better than 2-D J2K at 8:1-16:1 (RSNA 2004, different data) | no source | none found | none found |
| 3D-SPIHT / 3-D SPECK / 3-D DCT coders | Yes | whole volume or block-coded variants | no source retrieved | no source | none | none |
| ZFP (4^3, fixed-rate / accuracy) | Yes | 4^3 block (fixed-rate) | Float simulation Density: 107x at 47.1 dB, 160x at 38.6 dB (TTHRESH paper, different data) | up to ~2 GB/s (project page) | zfpy / imagecodecs ZFP in registry | WASM ports exist [recollection] |
| SZ3 | Yes (Lorenzo/interpolation) | chunk | Float HPC, e.g. Miranda PSNR 90 ratio 47 (SZ3 paper snippet, different data) | slower than ZFP | no zarr codec confirmed | none found |
| MGARD | Yes (multigrid) | chunk / refactored levels | no source retrieved | GPU-oriented | none found | none |
| TTHRESH | Yes (Tucker/HOSVD, global) | none (whole tensor) | Float Density: 1,855x at 47.1 dB, 3,494x at 38.6 dB (different data) | ~20 MB/s implied (Teapot 11.1 MB in 1.0 s; Isotropic 512 MB in 25.8 s) | none | none |
| INR (SIREN/hash-grid/NeRV-style) | Yes (function of x,y,z) | point query, but needs the whole net | 9.6-102x on smooth fields (search snippet, not verified) | GPU forward passes; no CPU GB/s given | none | WebGL/WebGPU possible [recollection] |
| Learned 3-D transform coder (e.g. VVMIC) | Yes (inter-slice latents) | whole volume / slab | BD-rate -7.7% to -17.2% vs VVC on MRI (arXiv 2412.09231); no PSNR-matched CT ratio | GPU | none | none |

## 9. Strongest and weakest points of volcomp relative to the field

Strongest
1. Throughput at ratio: 1.4 GB/s decode and 0.8 GB/s encode single-thread at q8 with 53x on 2.4 um CT (measured). Only ZFP in this survey is comparable in speed, and ZFP has no entropy-coded 16^3 transform.
2. Chunk-level random access with real streaming: independent 128^3 chunks in zarr v3 shards, verified end to end in a WASM viewer with HTTP Range reads and a trailing CRC-checked index. Video codecs, TTHRESH, INR and JP3D-in-practice cannot offer this.
3. Design fit: one knob, lossless, mask and exact-threshold surface modes in a single format; no other surveyed codec covers grayscale CT and thresholded prediction stores with an exactness guarantee.
4. Deployment footprint: single C header, no dependencies, 170 KB browser decoder.

Weakest
1. No head-to-head measurement against HEVC/AV1/JPEG XL/JPEG 2000/ZFP/SZ3/TTHRESH on scroll CT. Every competitor number in this document is on other data. Video codecs and TTHRESH probably beat volcomp on ratio at equal PSNR; this is unmeasured.
2. No error bound: quality is empirical (P99 ~ 2.5q, actually 1.9q at q8, FS 8); SZ/MGARD/ZFP-accuracy offer bounds. PR text says models not fine-tuned on compressed data are sensitive at q8, and there is no measured tracing or segmentation impact.
3. Blocking artifacts from 16^3 blocks (visible above 2x zoom at q8; blocking amplification 1.19 to 1.93); mitigated by opt-in decode-time deblocking, not by the format.
4. Ecosystem: integration into vc3d is an open PR, and mode 6 needs reader updates; the zarr codec is not in any registry. Measured on Linux x86-64 only.
5. Some headline totals in the docs disagree (whole bucket 300 TB vs 760 TB logical; ~40 dB vs measured 35 dB at q8 in the vendored README), so the writeup must use the BENCHMARKS values.

## 10. Source list (URL, year)

- volcomp fact sheet: /home/forrest/volume-compressor/docs/_factsheet_2026-09-30.md (2026)
- TTHRESH: https://arxiv.org/abs/1806.05952 ; ar5iv https://ar5iv.labs.arxiv.org/html/1806.05952 ; https://github.com/rballester/tthresh (TVCG 2019)
- ZFP: https://github.com/LLNL/zfp ; https://computing.llnl.gov/projects/zfp ; Lindstrom, TVCG 2014 (mirror https://www.academia.edu/63125780/Fixed_Rate_Compressed_Floating_Point_Arrays)
- SZ3: https://arxiv.org/abs/2111.02925 (2021 preprint, IEEE Trans. Big Data 2023); comparisons via https://arxiv.org/pdf/2409.05785 and https://arxiv.org/pdf/2509.01626 (snippets)
- MGARD: https://arxiv.org/abs/2401.05994 (SoftwareX 2023)
- JP3D: https://www.iso.org/standard/40024.html (ISO/IEC 15444-10:2011)
- 3-D vs 2-D JPEG 2000 CT: https://www.auntminnie.com/imaging-informatics/enterprise-imaging/pacs-vna/article/15571809/3d-jpeg-2000-compression-shines-in-multislice-ct-data (RSNA 2004)
- Micro-CT HEVC vs JPEG 2000: https://pmc.ncbi.nlm.nih.gov/articles/PMC10406799 (2023)
- Video codecs in healthcare: https://ercim-news.ercim.eu/en119/research-and-innovation/the-battle-of-the-video-codecs-in-healthcare (2019)
- HEVC lossless volumetric: https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/229/paper-detail/40213 (year unconfirmed)
- VVMIC: https://arxiv.org/html/2412.09231v1 (2024)
- INR: https://arxiv.org/abs/2104.04523 (2021); https://arxiv.org/abs/2401.08840 (2023/2024); https://arxiv.org/abs/2508.08937 (2025); NeRV https://arxiv.org/abs/2110.13903 (2021); NeRV-Dec https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5618_ECCV_2024_paper.php (2024); https://arxiv.org/pdf/2607.20970 (unverified snippet)
- Microscopy lossless: https://www.biorxiv.org/content/10.1101/2023.01.24.525380.full.pdf (2023); https://ideas.repec.org/a/plo/pone00/0351560.html (2026)
- Codec registries: https://zarr.dev/codecs-registry/ ; https://pypi.org/project/imagecodecs/
- Vesuvius: https://scrollprize.org/llms-full.txt (2026); https://scrollprize.substack.com/p/new-scans-of-herculaneum-papyri-at ; https://scrollprize.substack.com/p/vesuvius-challenge-is-hiring ; https://github.com/KhartesViewer/scroll2zarr ; https://github.com/ScrollPrize/villa/pull/1704

## 11. Suggested follow-up measurements before publishing (not done)

1. Same 512^3 cubes (FS 3b) through x265/SVT-AV1 (z as time, GOP 8 and 32), JPEG XL lossy per slice, JPEG 2000 per slice, ZFP accuracy mode, and (offline) TTHRESH, matched at 35 dB, with decode timing.
2. Real-CT lossless ratio for zstd-19 / blosc-zstd / volcomp q=0 on the same cubes, to replace the unsourced 1.3-2x.
3. Read the ash2txt zarr metadata to confirm the compressor actually in use.
