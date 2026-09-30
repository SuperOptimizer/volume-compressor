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

- C library, spec, Python zarr v3 codec, single-file browser viewer
- The mirror, <https://dl.ash2txt.org/community-uploads/forrest/volcomp/>: 5.86 TB in all (52.6x smaller than the bucket's stored bytes, 233x smaller than its logical level 0). It holds the CT of 39 samples, the 41 published m7 masks as volcomp stores, and whole-volume m7 probability stores (59 up, 61 by tonight)
- VC3D integration: ScrollPrize/villa PR #1704 (open)

| our m7 (8-bit prob, q 8) vs published (binary) | 0343P | 0009B |
|---|--:|--:|
| size ratio, level 0 | 2.34x smaller | 1.63x smaller |
| box dice p >= 0.5 | 0.637 | 0.655 |

The dice gap is mostly inference, not compression: separate runs (stride 128 vs 96 alone gives dice
0.946), thresholds 0.2 vs 0.5, and masked-air blocks in the published masks. Against the same run's float output, q 8 costs dice 0.978 and
MAE 4 of 255.

![PHerc0343P: CT, published, ours, diff](comparison/surfaces/PHerc0343P/z1493_zoom.png)

![viewer: CT with m7 probabilities](img/viewer_ct_m7L2.png)

**Caveats**

- No error bound
- The m7 stores are q 8, which flips 0.2-0.7 % of voxels at 0.5
- The viewer is tested in Chromium only
- Interim benchmark: SVT-AV1/x265 need 0.53-0.99x volcomp's bits at 35 dB, all others 1.4-31x; per chunk, volcomp reads 6-21x fewer bytes than them and decodes 14-179x faster

Full writeup: [SUBMISSION_2026-09.md](SUBMISSION_2026-09.md).
