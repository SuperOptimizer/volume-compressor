# m7 surface predictions: our volcomp store vs the published upstream mask

For each scroll, our m7 surface prediction (volcomp q8, **8-bit probability**) is compared with the **published**
upstream m7 prediction (**binary mask, threshold 0.2**, blosc) of the **same scan at the same level**. Made by
`tools/surface_compare.py`; per-scroll images, tables and captions are in the scroll folders.

- [PHerc0343P](PHerc0343P/README.md): scan 20260304131111 (2.215 µm), level 2 = 8.86 µm, 56.0 Gvox
- [PHerc0009B](PHerc0009B/README.md): scan 20260319104112 (2.401 µm), level 2 = 9.60 µm, 363.3 Gvox

Upstream publishes m7 only on these fine-pitch scans (level 2), not on the 8.64 µm scans, so our 8.64 µm level-0
stores (stride 96) have no published counterpart. The stores compared here are our tier-4 runs: **stride 128**,
CT input = our volcomp q8 CT with decode smoothing 2. Both are separate inference runs of the same checkpoint, so
the disagreement mixes inference differences (stride, blending, fp16 TensorRT vs the upstream predictor, q8 CT
input) with compression. On the one box where an uncompressed float prediction was kept, the compression-only
loss is small (last table).

| | PHerc0343P | PHerc0009B |
|---|---|---|
| voxels (level 0) | 56.0 Gvox | 363.3 Gvox |
| published store, level 0 (binary, blosc) | 370 MiB | 999 MiB |
| our store, level 0 (8-bit probability, volcomp q8) | 158 MiB | 613 MiB |
| ratio published / ours, level 0 | 2.34x | 1.63x |
| all levels (published 6 / ours 8) | 468 / 202 MiB (2.32x) | 1.23 GiB / 764 MiB (1.65x) |
| dice, ours p>=0.5 vs published, dense 1024³ box | 0.637 | 0.655 |
| precision / recall, same | 0.877 / 0.501 | 0.859 / 0.530 |
| dice, ours p>=0.2 (the published threshold), box | 0.736 | 0.744 |
| dice p>=0.5 / p>=0.2, whole volume | 0.459 / 0.603 | 0.686 / 0.758 |
| precision / recall p>=0.5, whole volume | 0.779 / 0.325 | 0.868 / 0.567 |
| published surface voxels where the published CT is 0 (masked air) | 6.11 Gvox | 11.36 Gvox |

All dice/precision/recall are counted inside the CT mask (CT > 0), plain decode. Decode smoothing (`gated2`,
`set_read_smoothing(2)`) moves every score by at most 0.0005.

**Compression only** (PHerc0343P 8.64 µm scan, 384×768² box, our q8 store vs the uncompressed fp32 blend of the
same run; no float prediction exists for the level-2 scans above):

| decode | MAE /255 | dice p>=0.5 | dice p>=0.2 |
|---|---|---|---|
| plain | 4.10 | 0.978 | 0.968 |
| gated2 | 3.70 | 0.978 | 0.969 |
| gauss0.6 | 4.67 | 0.977 | 0.955 |

**Reading the images.** Amber = surface over the CT: solid for the published binary mask, opacity 0.6·p for our
probability. In the diff panels red = published only, blue = ours only (p >= 0.5), grey = both, dark red =
published surface in masked air. Since the published threshold (0.2) is below ours (0.5), a thin red rim along
every sheet is expected even from identical predictions.

**Caveat on the published masks.** 80 % (0343P) and 71 % (0009B) of the published surface voxels lie outside the
published CT mask, in solid blocks of masked air (visible as staircase blocks at the scroll edge). Counted in,
dice drops to 0.12-0.28; the tables leave them out.
