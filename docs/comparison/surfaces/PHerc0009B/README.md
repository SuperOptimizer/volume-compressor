# PHerc0009B: our volcomp m7 surface prediction vs the published m7 prediction

Scan `20260319104112-2.401um-0.3m-77keV-masked.zarr`, CT level 2 (9.604 µm), prediction grid 7278 x 7065 x 7065 (z, y, x) = 363.3 Gvox. Made by `tools/surface_compare.py`; every number is in [report.json](report.json).

- **Published**: `s3://vesuvius-challenge-open-data/PHerc0009B/representations/predictions/surfaces/20260319104112-surface-20260413222639-surface-m7-L2-th0.2.zarr/` level 0, OME-zarr v2, blosc-zstd 192³ chunks. A **binary mask**: the m7 probability thresholded at **0.2**.
- **Ours**: `https://dl.ash2txt.org/community-uploads/forrest/volcomp/PHerc0009B/representations/predictions/surfaces/20260319104112-surface-20260925170000-surface-m7-L2-prob.zarr` level `9.604`, zarr v3 sharded, volcomp q8. An **8-bit probability** (uint8 = round(p · 255)), decoded plain (what any zarr reader gets) or with volcomp's decode-only smoothing: `gated2` is what `volcomp_zarr.set_read_smoothing(2)` gives (`decode_smooth(…, 2, GATED | ZERO_GUARD)`), `gauss0.6` is `decode_smooth(…, 0.6, ZERO_GUARD)`, the setting docs/deblocking.md recommends for probability maps. The images use plain and gated2.
- **CT** under the images: the original uncompressed level 2 (`s3://vesuvius-challenge-open-data/PHerc0009B/volumes/20260319104112-2.401um-0.3m-77keV-masked.zarr/2`).

**What differs besides compression.** Both are the same m7 checkpoint (surface_m7_nnunet) on the same scan at the same level, but they are separate inference runs. Ours: 192³ windows at **stride 128**, separable Gaussian blend, TensorRT fp16, no TTA, run on our volcomp q8 CT (volcomp_decode_smooth strength 2, gated, zero guard). Published: the upstream pipeline (nnU-Net predictor, TTA disabled per its metadata.json) on the original CT. m7's mask moves with window placement (stride 128 against 96 on the same input gave dice 0.946; no stride pair reaches 0.99), so a large part of the disagreement below is inference, not compression. The compression-only loss is measured separately at the end.

## Table

| | |
|---|---|
| voxels (level 0) | 363,275,729,550 (363.3 Gvox) |
| published store, level 0 | 999 MiB (1,047,330,582 B) |
| our store, level 0 | 613 MiB (642,517,004 B) |
| ratio published / ours, level 0 | 1.63x |
| published, all 6 levels / ours, all 8 levels | 1.23 GiB / 764 MiB (1.65x) |

Agreement with the published mask in the comparison box z 2368-3392, y 3200-4224, x 3840-4864 (1.07 Gvox, the window with the most published surface inside the CT mask; 100 % of it is inside the CT mask), counted **inside the CT mask** (CT > 0). Precision = fraction of our voxels that the published mask also has; recall = fraction of the published voxels we also have. `p>=0.2` compares at the published threshold, `p>=0.5` at the usual one.

**The published mask also marks masked air.** In this box 70,582 published surface voxels lie where the published CT is 0 (outside the scroll mask), mostly in solid blocks; our store has 21,688 voxels >= 0.2 there (it is 0 wherever our volcomp copy of the CT is 0; the few left are at mask edges where the two copies differ). They are drawn dark red in the diff panels and left out of the scores below; counted in, the plain p>=0.5 dice would be 0.6551.

| ours decoded | threshold | dice | precision | recall | our voxels | published voxels |
|---|---|---|---|---|---|---|
| plain | p>=0.5 | 0.6552 | 0.8590 | 0.5295 | 167,005,122 | 270,908,686 |
| plain | p>=0.2 | 0.7443 | 0.7041 | 0.7894 | 303,708,920 | 270,908,686 |
| gated2 | p>=0.5 | 0.6551 | 0.8592 | 0.5293 | 166,897,793 | 270,908,686 |
| gated2 | p>=0.2 | 0.7444 | 0.7043 | 0.7893 | 303,585,436 | 270,908,686 |
| gauss0.6 | p>=0.5 | 0.6540 | 0.8606 | 0.5274 | 166,013,861 | 270,908,686 |
| gauss0.6 | p>=0.2 | 0.7445 | 0.6997 | 0.7955 | 308,012,851 | 270,908,686 |

Whole volume (every level-0 voxel inside the CT mask; the mask is CT level 4 > 0, nearest-upsampled 4x). Published surface voxels outside the mask: 11,359,222,565; ours >= 0.2 outside it: 4,064,737 (mask edges: the 4x-coarser mask is approximate there). The whole volume includes the sparse outer wraps and mask edges, so it can score below the dense box.

| ours decoded | threshold | dice | precision | recall | our voxels | published voxels |
|---|---|---|---|---|---|---|
| plain | p>=0.5 | 0.6860 | 0.8675 | 0.5673 | 3,067,929,223 | 4,691,838,419 |
| plain | p>=0.2 | 0.7575 | 0.7147 | 0.8057 | 5,288,940,082 | 4,691,838,419 |
| gated2 | p>=0.5 | 0.6858 | 0.8678 | 0.5669 | 3,065,155,246 | 4,691,838,419 |
| gated2 | p>=0.2 | 0.7576 | 0.7152 | 0.8054 | 5,284,011,880 | 4,691,838,419 |
| all_voxels_plain | p>=0.5 | 0.2784 | 0.8673 | 0.1658 | 3,068,965,388 | 16,051,060,984 |
| all_voxels_plain | p>=0.2 | 0.3543 | 0.7143 | 0.2356 | 5,293,004,819 | 16,051,060,984 |

No uncompressed float m7 prediction exists for this scroll; see [../PHerc0343P/README.md](../PHerc0343P/README.md#compression-only) for the compression-only loss measured on the kept PHerc0343P box.


## Images

Slices at 1/3 and 2/3 of the box along each axis (fixed positions, not picked), named by axis and level-0 index. Full resolution: 1 image pixel = 1 voxel, 1024² unless the volume is smaller.

- `<view>_ct.png`: the CT slice.
- `<view>_published.png`: the published **binary threshold-0.2 mask** in amber at 60 % opacity over the CT.
- `<view>_ours.png`: our **8-bit probability**, plain decode, in amber at opacity 0.6 · p over the CT (faint amber = low probability; the published mask has no such gradation).
- `<view>_ours_smooth.png`: the same with decode smoothing (`gated2`, what `volcomp_zarr.set_read_smoothing(2)` gives; on the kept float box it has the lowest error of the three decodes).
- `<view>_diff.png` (plain decode): darkened CT; **red** = published only, **blue** = ours only (plain decode, p ≥ 0.5), **grey** = both, **dark red** = published surface where the CT is 0 (masked air). Because the published threshold (0.2) is lower than ours (0.5), a thin red rim along surfaces is expected even from identical models.
- `<view>_all.png`: the five panels side by side at half resolution, labelled.
- `<view>_zoom.png`: the 256² crop of the slice with the most surface, five panels at 2x (nearest), labelled.

| view | slice dice (p>=0.5, plain) | precision | recall | images |
|---|---|---|---|---|
| z2709 | 0.6468 | 0.8631 | 0.5172 | [all](z2709_all.png) · [zoom](z2709_zoom.png) · [CT](z2709_ct.png) · [published](z2709_published.png) · [ours](z2709_ours.png) · [ours smooth](z2709_ours_smooth.png) · [diff](z2709_diff.png) |
| z3050 | 0.6479 | 0.8490 | 0.5238 | [all](z3050_all.png) · [zoom](z3050_zoom.png) · [CT](z3050_ct.png) · [published](z3050_published.png) · [ours](z3050_ours.png) · [ours smooth](z3050_ours_smooth.png) · [diff](z3050_diff.png) |
| y3541 | 0.7822 | 0.9397 | 0.6699 | [all](y3541_all.png) · [zoom](y3541_zoom.png) · [CT](y3541_ct.png) · [published](y3541_published.png) · [ours](y3541_ours.png) · [ours smooth](y3541_ours_smooth.png) · [diff](y3541_diff.png) |
| y3882 | 0.6240 | 0.9006 | 0.4774 | [all](y3882_all.png) · [zoom](y3882_zoom.png) · [CT](y3882_ct.png) · [published](y3882_published.png) · [ours](y3882_ours.png) · [ours smooth](y3882_ours_smooth.png) · [diff](y3882_diff.png) |
| x4181 | 0.6563 | 0.8601 | 0.5306 | [all](x4181_all.png) · [zoom](x4181_zoom.png) · [CT](x4181_ct.png) · [published](x4181_published.png) · [ours](x4181_ours.png) · [ours smooth](x4181_ours_smooth.png) · [diff](x4181_diff.png) |
| x4522 | 0.6395 | 0.8317 | 0.5195 | [all](x4522_all.png) · [zoom](x4522_zoom.png) · [CT](x4522_ct.png) · [published](x4522_published.png) · [ours](x4522_ours.png) · [ours smooth](x4522_ours_smooth.png) · [diff](x4522_diff.png) |

![z2709 all](z2709_all.png)

![z2709 zoom](z2709_zoom.png)
