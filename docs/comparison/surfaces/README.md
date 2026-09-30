# m7 surface predictions: our volcomp store vs the published upstream mask

For each scroll, our m7 surface prediction (volcomp q8, **8-bit probability**) is compared with the **published** upstream m7 prediction (**binary mask, threshold 0.2**, blosc) of the **same scan at the same level**. Made by `tools/surface_compare.py`; per-scroll images, tables and captions are in the scroll folders, every number in their report.json. A thumbnail index is in [../GALLERY.md](../GALLERY.md).

PHerc0343P and PHerc0009B compare at level 2 of 2.2/2.4 µm scans (8.86/9.60 µm); the other six at level 0 of 8.64/9.36 µm scans, where upstream publishes m7 level-0 masks. Our stores are separate inference runs of the same checkpoint (see each README for stride and CT input), so the disagreement mixes inference differences (stride, blending, fp16 TensorRT vs the upstream predictor, q8 CT input) with compression. On the one box where an uncompressed float prediction was kept, compression alone costs about 0.02 dice ([PHerc0343P](PHerc0343P/README.md#compression-only)).

## Summary

| scroll | scan | level (pitch) | voxels | published L0 | ours L0 | ratio | stride | dice p>=0.2 box | dice p>=0.2 whole | dice p>=0.5 box | dice p>=0.5 whole | published in masked air |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [PHerc0343P](PHerc0343P/README.md) | 20260304131111-2.215um-0.4m-111keV | 2 (8.86 µm) | 56 Gvox | 370 MiB | 158 MiB | 2.34x | 128 | 0.735 | 0.603 | 0.637 | 0.459 | 81 % |
| [PHerc0009B](PHerc0009B/README.md) | 20260319104112-2.401um-0.3m-77keV | 2 (9.604 µm) | 363 Gvox | 999 MiB | 613 MiB | 1.63x | 128 | 0.744 | 0.757 | 0.655 | 0.686 | 71 % |
| PHerc0800 | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |
| PHerc1218 | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |
| PHerc0175A | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |
| PHerc0268 | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |
| PHerc1447 | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |
| PHerc0125 | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |
| **all 2 with numbers** | | | | **1.34 GiB** | **771 MiB** | **1.78x** | | | | | | |

Sizes are level 0 only (published: S3 listing of the blosc chunks; ours: HEAD of every shard). Dice is our plain decode at p>=0.2 (the published threshold) or p>=0.5 against the published mask, counted inside the CT mask (CT > 0); `box` is the 1024³ window with the most published surface inside the CT mask, `whole` every level-0 voxel. The last column is the share of published surface voxels (whole volume) that lie where the CT is 0; they are left out of the scores. Decode smoothing moves the scores by at most 0.0005 (the first two scrolls' READMEs list gated2 and gauss0.6 too).

**Reading the images.** Amber = surface over the CT: solid for the published binary mask, opacity 0.6·p for our probability. In the diff panels red = published only, blue = ours only (p >= 0.5), grey = both, dark red = published surface in masked air. Since the published threshold (0.2) is below ours (0.5), a thin red rim along every sheet is expected even from identical predictions.
