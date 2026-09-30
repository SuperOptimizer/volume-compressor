# volcomp: September submission (short version)

volcomp is a single-header C codec for `uint8` scroll CT and predictions. It uses a 16³ 3-D DCT and tANS,
with independently decodable 128³ chunks in zarr v3 shards. The open-data bucket holds 761 TB of logical
level-0 CT, of which 275 TB is stored.

| data (one core) | q | ratio | PSNR |
|---|--:|--:|--:|
| PHercParis4 2.4 µm, 54 held-out chunks | 8 | 52.9x | 35.0 dB |
| 0.55 / 1.13 / 2.40 / 9.36 µm cubes | 8 | 146x / 132x / 52x / 31x | 41.6 / 42.7 / 36.6 / 33.4 dB |
| decode / encode speed | 8 | 1.4 / 0.8 GB/s | |
| recto teacher on q 8 CT vs raw CT | 8 | dice 0.951 | |

![PHercParis4 raw and q1-q32](comparison/PHercParis4_2.40um/zmid_all.png)

**Released**

- C library, format spec and a Python zarr v3 codec
- A single-file browser viewer that streams the mirror by HTTP Range
- The mirror, <https://dl.ash2txt.org/community-uploads/forrest/volcomp/>: CT of all 39 samples, the published masks as exact mask stores, and 61 new whole-volume m7 probability stores
- VC3D integration: ScrollPrize/villa PR #1704 (open)

| our m7 (8-bit prob, q 8) vs published (binary) | 0343P | 0009B |
|---|--:|--:|
| size ratio, level 0 | 2.34x smaller | 1.63x smaller |
| box dice p >= 0.5 | 0.637 | 0.655 |

Most of the dice gap comes from inference, not compression. The two are separate runs, and m7 moves with
window placement (stride 128 vs 96 gives dice 0.946). The thresholds differ (0.2 vs 0.5), and the
published masks mark blocks of masked air. Against the same run's float output, q 8 costs dice 0.978 and
MAE 4 of 255.

![PHerc0343P: CT, published, ours, diff](comparison/surfaces/PHerc0343P/z1493_zoom.png)

![viewer: CT with m7 probabilities](img/viewer_ct_m7L2.png)

**Caveats**

- No error bound
- Blocks are visible at high zoom
- The m7 stores are q 8, which flips 0.2-0.7 % of voxels at 0.5
- The viewer is tested in Chromium only
- The benchmark against JPEG XL, AV1, x265, ZFP and others is **[pending: docs/bench/]**

Full writeup: [SUBMISSION_2026-09.md](SUBMISSION_2026-09.md).
