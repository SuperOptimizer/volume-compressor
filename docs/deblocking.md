# Deblocking: encode-only, decode-only and both

volcomp reconstructs every 16³ block, and every 128³ chunk, independently. That
is what gives random access and chunk independence, but at higher q it leaves
seams at block faces. This note surveys the three families of fixes, measures
the most promising candidate of each on real data, and records what was
implemented and what the defaults are. The data are:

- **raw masked CT**, the codec's main use: PHercParis4 2.4 µm level 0, fetched
  uncompressed from the open-data bucket;
- the surface-teacher probability maps of `docs/surface_mode.md`.

## Summary and recommendation

| data | recommended reader setting | what it buys (measured below) | cost |
|---|---|---|---|
| masked CT, q >= 4 (q 8 is the production CT store) | `volcomp_decode_smooth(…, 2.0, VOLCOMP_SMOOTH_GATED \| VOLCOMP_DEBLOCK_ZERO_GUARD)` per chunk. For assembled regions, add `volcomp_deblock` over the region for the chunk faces | PSNR **+0.11 to +0.61 dB** (+0.36-0.42 at q8) and SSIM up at every q ≥ 4, blocking amplification 1.9-2.3 → 1.4-1.5. **No chunk worse** than no filter, on all 4 volumes at q 4, 8 and 16 | 5-6 ms per chunk: 380-420 MB/s single-thread, against 2.2-5 GB/s plain |
| masked CT, q < 4 | plain `volcomp_decode` (decode_smooth does this by itself below q 4) | at q 2 the gated filters are within ±0.03 dB of no filter, and the Gaussian loses up to 0.42 dB | — |
| surface probability maps, mode 0 or surface mode | `volcomp_decode_smooth(…, 0.6, VOLCOMP_DEBLOCK_ZERO_GUARD)`, the Gaussian seam blur, projected | m7 q16: dice 0.9913 → 0.9931 and iso-surface shift 0.095 → 0.075 voxel. recto q16: 0.182 → 0.170. Band MAE −3 to −10 %. **Surface chunks keep their exact threshold** | 13-17 ms per chunk: 120-155 MB/s |
| any volume through `volcomp_deblock` | the zero guard is now **on by default** | masked-CT air edges are no longer blurred. At q16 on the tune octet the unguarded filter gave edge MAE 3.18 → 4.89, chunk faces 2.8× rougher and 4 of 8 chunks worse than no filter; guarded it gives +0.52 dB and none worse | none |

**Defaults.**

- `volcomp_decode` is unchanged: no reader-side filtering unless it is asked for,
  as the task requires.
- The only default that changed is `volcomp_deblock`, and only by the zero
  guard. On every measured volume that is at least as good as the unguarded filter,
  and much better on masked edges.
- Python readers opt in with `volcomp_zarr.set_read_smoothing(2)` (or
  `VOLCOMP_SMOOTH=2`). The CLI opts in with `volcomp decode … --smooth[=S]`.

**Not adopted.**

- **Encode-only error feedback:**
  - It costs +10-23 % bytes and gives −0.1 to −0.8 dB (tune octet: q8 0.0669
    → 0.0744 bpv, 39.91 → 39.84 dB).
  - The encode-only survey's remaining candidates are RDOQ and boundary-aware RD
    costs. docs/measured.md already records RDOQ as at most ~1 % for 1.6-2.4×
    encode time, so none of them was worth a format-neutral encoder change here.
- **Encoder + decoder (per-chunk signalled strength, 2 bits per chunk):**
  - It **equals the best fixed strength** (gated 3) to within 0.01 dB on every
    CT volume and q.
  - The fixed filter is never worse than no filter at q ≥ 4, so the encoder has
    nothing to choose.
  - On probability maps it cannot beat the Gaussian variant.
  - It would need a format change for nothing. The briefs' other both-sides
    options (lapped transforms, adaptive block size) break chunk independence or
    the block geometry, and lapping was already measured and rejected.

**Downstream: does the teacher care?** The recto teacher was run on the raw
interior 512³ CT and on the decoded CT, and the predictions were compared on
the central 256³.

- At q 4 and q 8, deblocking moves the prediction by about 0.2 % dice either way.
  At q 8, plain decode agrees best with the raw-CT prediction: dice 0.951 against
  0.948-0.949 with any filter.
- At q 16 every filter brings the prediction closer (0.927 → 0.931-0.933).
- For teacher inference on the q 8 CT stores, keep plain decoding. The PSNR and
  seam gains are for viewing and for q ≥ 16.
- On the mask-edge region the teacher predicts no surface on any input, so it
  is not scored.

## 2. Measurement

**Data.**

- **Raw CT:** PHercParis4 2.4 µm masked level 0, uncompressed zarr v2 chunks from
  the open-data bucket. The desk mirror is itself volcomp q 8, so it cannot be
  the ground truth.
  - **interior**: 512³ at chunks z 269-272, y 118-121, x 144-147. Dense papyrus,
    around the teacher evaluation box.
  - **mask-edge**: 512³ at chunks z 269-272, y 158-161, x 226-229. The scroll's
    outer boundary: 28 of the 64 chunks are masked air, and air edges fall on
    chunk faces.
  - The **tune** and **heldout** octets (256³) of `corpus/`.
- **Probability maps:** the evalbox and r32k cubes of both teachers, from
  `docs/surface_mode.md`.

**Encoding.** Mode 0 at the CT store's q (2, 4, 8, 16); mode 0 q8/q16 and surface
q48/q64 for the probability maps.

**Metrics** (script `tools/surface_study/dbk4.py`):

| metric | definition |
|---|---|
| PSNR | over the whole volume, and over tissue only (source > 0) |
| SSIM | on z slices |
| block amplification | RMSE of the error's first difference across 16-planes, divided by the same across mid-block planes |
| chunk-face amplification | the same, at the 128-planes |
| mask-edge MAE | on voxels within 2 of the air boundary |
| chunks worse | 128³ chunks with a higher MSE than no filter |
| dice, disp, band MAE | probability maps only, as in docs/surface_mode.md |

**Candidates.**

| candidate | class | what it is |
|---|---|---|
| deblock (existing) | decode-only, region | `volcomp_deblock` as it was: the gated 4-tap face filter over the assembled region |
| deblock zg | decode-only, region | the same with the zero guard |
| smooth gatedS zg | decode-only, per chunk | `volcomp_decode_smooth` with the gated filter at S×q as the smoothing step, then the projection |
| smooth gauss0.6 zg | decode-only, per chunk | the same with a Gaussian seam blur (σ 0.6, only within 2 voxels of interior block faces) as the smoothing step |
| signalled per chunk | encoder + decoder | the encoder picks, per chunk, none / gated1 / gated2 / gated3 by MSE against the source |
| encode-only feedback | encode-only | re-encode with α × the decoded error at block faces added to the source (`dbk2.py`) |

**Chunk faces.** The projection works inside one chunk, from that chunk's stream
alone, so a chunk face is only seen from one side. `decode_smooth` leaves it as
decoded; it smooths only the 7 interior block faces per axis. That is why its
chunk-face amplification stays near the unfiltered value. Running
`volcomp_deblock` over the assembled region afterwards fixes those faces (the
"+ deblock zg" rows), at no PSNR cost.

#### CT: PSNR gain over no filter (dB) / chunks worse than no filter

| candidate | interior q2 | interior q4 | interior q8 | interior q16 | mask-edge q2 | mask-edge q4 | mask-edge q8 | mask-edge q16 | tune q2 | tune q4 | tune q8 | tune q16 | heldout q2 | heldout q4 | heldout q8 | heldout q16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 | +0.00 / 0 |
| deblock (existing) | +0.01 / 10 | +0.09 / 0 | +0.26 / 0 | +0.48 / 0 | +0.09 / 4 | +0.24 / 5 | +0.35 / 5 | +0.42 / 6 | +0.01 / 0 | +0.11 / 0 | +0.30 / 4 | +0.06 / 4 | +0.02 / 0 | +0.13 / 0 | +0.29 / 4 | +0.20 / 4 |
| deblock zg | +0.01 / 10 | +0.09 / 0 | +0.26 / 0 | +0.48 / 0 | +0.08 / 0 | +0.23 / 0 | +0.33 / 0 | +0.41 / 0 | +0.01 / 0 | +0.11 / 0 | +0.30 / 0 | +0.52 / 0 | +0.02 / 0 | +0.13 / 0 | +0.29 / 0 | +0.51 / 0 |
| smooth gated2 zg | -0.00 / 33 | +0.11 / 0 | +0.36 / 0 | +0.56 / 0 | +0.11 / 0 | +0.24 / 0 | +0.37 / 0 | +0.48 / 0 | +0.02 / 0 | +0.14 / 0 | +0.41 / 0 | +0.61 / 0 | +0.03 / 0 | +0.15 / 0 | +0.40 / 0 | +0.59 / 0 |
| smooth gated3 zg | -0.03 / 54 | +0.11 / 0 | +0.41 / 0 | +0.56 / 0 | +0.11 / 0 | +0.25 / 0 | +0.38 / 0 | +0.48 / 0 | -0.00 / 1 | +0.15 / 0 | +0.43 / 0 | +0.61 / 0 | +0.01 / 1 | +0.16 / 0 | +0.42 / 0 | +0.59 / 0 |
| smooth gauss0.6 zg | -0.37 / 64 | +0.05 / 12 | +0.32 / 0 | +0.45 / 0 | -0.42 / 30 | -0.14 / 22 | +0.05 / 18 | +0.13 / 18 | -0.29 / 4 | +0.10 / 0 | +0.36 / 0 | +0.49 / 0 | -0.22 / 4 | +0.11 / 0 | +0.33 / 0 | +0.46 / 0 |
| smooth gated2 zg + deblock zg | -0.00 / 34 | +0.10 / 0 | +0.35 / 0 | +0.52 / 0 | +0.12 / 0 | +0.25 / 0 | +0.38 / 0 | +0.48 / 0 | +0.02 / 0 | +0.14 / 0 | +0.40 / 0 | +0.55 / 0 | +0.03 / 0 | +0.15 / 0 | +0.38 / 0 | +0.54 / 0 |
| signalled per chunk (2 bits) | +0.01 / 0 | +0.11 / 0 | +0.41 / 0 | +0.56 / 0 | +0.11 / 0 | +0.25 / 0 | +0.38 / 0 | +0.48 / 0 | +0.02 / 0 | +0.15 / 0 | +0.43 / 0 | +0.61 / 0 | +0.03 / 0 | +0.16 / 0 | +0.42 / 0 | +0.59 / 0 |

#### CT at q8 and q16: all metrics

| volume | q | candidate | PSNR | tissue PSNR | SSIM (z slices) | block amp | chunk-face amp | mask-edge MAE | ms/chunk |
|---|---|---|---|---|---|---|---|---|---|
| interior 512^3 | 8 | none | 36.34 | 36.34 | 0.9536 | 1.92 | 1.92 | 3.24 | 2 |
| interior 512^3 | 8 | deblock (existing) | 36.60 | 36.60 | 0.9555 | 1.57 | 1.57 | 3.17 |  |
| interior 512^3 | 8 | deblock zg | 36.60 | 36.60 | 0.9555 | 1.57 | 1.57 | 3.17 |  |
| interior 512^3 | 8 | smooth gated2 zg | 36.71 | 36.71 | 0.9559 | 1.48 | 1.84 | 3.15 | 12 |
| interior 512^3 | 8 | smooth gauss0.6 zg | 36.67 | 36.67 | 0.9558 | 1.13 | 1.83 | 3.15 | 197 |
| interior 512^3 | 8 | smooth gated2 zg + deblock zg | 36.69 | 36.69 | 0.9558 | 1.49 | 1.55 | 3.15 |  |
| interior 512^3 | 16 | none | 33.37 | 33.37 | 0.9200 | 2.18 | 2.17 | 4.56 | 2 |
| interior 512^3 | 16 | deblock (existing) | 33.85 | 33.85 | 0.9248 | 1.42 | 1.40 | 4.37 |  |
| interior 512^3 | 16 | deblock zg | 33.85 | 33.85 | 0.9248 | 1.42 | 1.40 | 4.37 |  |
| interior 512^3 | 16 | smooth gated2 zg | 33.93 | 33.93 | 0.9253 | 1.35 | 2.09 | 4.36 | 11 |
| interior 512^3 | 16 | smooth gauss0.6 zg | 33.82 | 33.82 | 0.9244 | 1.29 | 2.07 | 4.38 | 201 |
| interior 512^3 | 16 | smooth gated2 zg + deblock zg | 33.88 | 33.88 | 0.9248 | 1.43 | 1.40 | 4.36 |  |
| mask-edge 512^3 | 8 | none | 44.49 | 40.36 | 0.9781 | 1.47 | 1.34 | 2.05 | 2 |
| mask-edge 512^3 | 8 | deblock (existing) | 44.84 | 40.71 | 0.9790 | 1.05 | 0.86 | 2.02 |  |
| mask-edge 512^3 | 8 | deblock zg | 44.82 | 40.71 | 0.9790 | 1.07 | 0.91 | 2.02 |  |
| mask-edge 512^3 | 8 | smooth gated2 zg | 44.86 | 40.72 | 0.9795 | 1.07 | 1.29 | 2.00 | 9 |
| mask-edge 512^3 | 8 | smooth gauss0.6 zg | 44.54 | 40.57 | 0.9791 | 0.93 | 1.16 | 2.43 | 193 |
| mask-edge 512^3 | 8 | smooth gated2 zg + deblock zg | 44.87 | 40.73 | 0.9796 | 1.05 | 0.90 | 1.99 |  |
| mask-edge 512^3 | 16 | none | 42.18 | 38.13 | 0.9668 | 1.59 | 1.44 | 3.14 | 1 |
| mask-edge 512^3 | 16 | deblock (existing) | 42.60 | 38.56 | 0.9683 | 1.01 | 0.76 | 3.12 |  |
| mask-edge 512^3 | 16 | deblock zg | 42.60 | 38.57 | 0.9682 | 1.03 | 0.84 | 3.09 |  |
| mask-edge 512^3 | 16 | smooth gated2 zg | 42.66 | 38.58 | 0.9698 | 1.02 | 1.41 | 3.04 | 9 |
| mask-edge 512^3 | 16 | smooth gauss0.6 zg | 42.31 | 38.38 | 0.9689 | 0.97 | 1.24 | 3.65 | 194 |
| mask-edge 512^3 | 16 | smooth gated2 zg + deblock zg | 42.66 | 38.58 | 0.9699 | 1.01 | 0.84 | 3.04 |  |
| tune octet | 8 | none | 39.91 | 36.90 | 0.9716 | 2.01 | 2.01 | 2.29 | 2 |
| tune octet | 8 | deblock (existing) | 40.21 | 37.20 | 0.9734 | 1.60 | 1.66 | 2.23 |  |
| tune octet | 8 | deblock zg | 40.21 | 37.20 | 0.9734 | 1.60 | 1.65 | 2.23 |  |
| tune octet | 8 | smooth gated2 zg | 40.32 | 37.31 | 0.9736 | 1.50 | 1.94 | 2.22 | 8 |
| tune octet | 8 | smooth gauss0.6 zg | 40.27 | 37.26 | 0.9732 | 1.15 | 1.93 | 2.22 | 255 |
| tune octet | 8 | smooth gated2 zg + deblock zg | 40.31 | 37.30 | 0.9736 | 1.52 | 1.64 | 2.22 |  |
| tune octet | 16 | none | 36.94 | 33.93 | 0.9521 | 2.30 | 2.31 | 3.18 | 2 |
| tune octet | 16 | deblock (existing) | 37.00 | 34.22 | 0.9564 | 2.17 | 6.43 | 4.89 |  |
| tune octet | 16 | deblock zg | 37.46 | 34.45 | 0.9564 | 1.45 | 1.56 | 3.05 |  |
| tune octet | 16 | smooth gated2 zg | 37.56 | 34.55 | 0.9563 | 1.38 | 2.24 | 3.04 | 12 |
| tune octet | 16 | smooth gauss0.6 zg | 37.44 | 34.43 | 0.9555 | 1.32 | 2.21 | 3.06 | 89 |
| tune octet | 16 | smooth gated2 zg + deblock zg | 37.50 | 34.49 | 0.9561 | 1.49 | 1.55 | 3.05 |  |
| heldout octet | 8 | none | 40.53 | 37.51 | 0.9701 | 1.97 | 2.18 | 2.33 | 2 |
| heldout octet | 8 | deblock (existing) | 40.81 | 37.80 | 0.9717 | 1.56 | 1.95 | 2.28 |  |
| heldout octet | 8 | deblock zg | 40.82 | 37.81 | 0.9717 | 1.56 | 1.89 | 2.27 |  |
| heldout octet | 8 | smooth gated2 zg | 40.92 | 37.91 | 0.9718 | 1.46 | 2.10 | 2.26 | 8 |
| heldout octet | 8 | smooth gauss0.6 zg | 40.86 | 37.85 | 0.9716 | 1.18 | 2.12 | 2.27 | 253 |
| heldout octet | 8 | smooth gated2 zg + deblock zg | 40.91 | 37.90 | 0.9718 | 1.48 | 1.88 | 2.26 |  |
| heldout octet | 16 | none | 37.76 | 34.75 | 0.9545 | 2.29 | 2.55 | 3.19 | 2 |
| heldout octet | 16 | deblock (existing) | 37.96 | 35.09 | 0.9576 | 1.95 | 5.45 | 4.10 |  |
| heldout octet | 16 | deblock zg | 38.27 | 35.26 | 0.9576 | 1.45 | 1.95 | 3.06 |  |
| heldout octet | 16 | smooth gated2 zg | 38.35 | 35.34 | 0.9576 | 1.38 | 2.46 | 3.06 | 10 |
| heldout octet | 16 | smooth gauss0.6 zg | 38.22 | 35.21 | 0.9570 | 1.36 | 2.46 | 3.08 | 86 |
| heldout octet | 16 | smooth gated2 zg + deblock zg | 38.30 | 35.29 | 0.9575 | 1.48 | 1.95 | 3.07 |  |

#### Downstream: recto teacher on decoded CT vs on the raw CT (interior 512^3, central 256^3 scored)

| q | candidate | dice at 0.5 | mean abs diff (u8) | corr |
|---|---|---|---|---|
| 4 | none | 0.9777 | 2.40 | 0.99775 |
| 4 | deblock (existing) | 0.9779 | 2.31 | 0.99793 |
| 4 | deblock zg | 0.9779 | 2.31 | 0.99793 |
| 4 | smooth gated2 zg | 0.9772 | 2.42 | 0.99772 |
| 4 | smooth gated2 zg + deblock zg | 0.9775 | 2.37 | 0.99781 |
| 8 | none | 0.9511 | 5.85 | 0.98812 |
| 8 | deblock (existing) | 0.9492 | 5.90 | 0.98826 |
| 8 | deblock zg | 0.9492 | 5.90 | 0.98827 |
| 8 | smooth gated2 zg | 0.9480 | 6.09 | 0.98780 |
| 8 | smooth gated2 zg + deblock zg | 0.9478 | 6.03 | 0.98802 |
| 16 | none | 0.9273 | 8.59 | 0.97752 |
| 16 | deblock (existing) | 0.9310 | 8.04 | 0.97913 |
| 16 | deblock zg | 0.9311 | 8.04 | 0.97913 |
| 16 | smooth gated2 zg | 0.9322 | 8.08 | 0.97981 |
| 16 | smooth gated2 zg + deblock zg | 0.9325 | 7.89 | 0.97999 |

#### recto probability maps (evalbox + r32k cubes, mean)

| setting | candidate | PSNR | block amp | dice | disp mean | band MAE | chunks worse |
|---|---|---|---|---|---|---|---|
| dct q8 | none | 40.00 | 1.89 | 0.9872 | 0.143 | 2.65 | 0/16 |
| dct q8 | deblock (existing) | 40.19 | 1.60 | 0.9873 | 0.141 | 2.61 | 0/16 |
| dct q8 | smooth gated2 zg | 40.11 | 1.69 | 0.9872 | 0.143 | 2.65 | 1/16 |
| dct q8 | smooth gauss0.6 zg | 40.31 | 1.34 | 0.9877 | 0.137 | 2.57 | 0/16 |
| dct q8 | signalled per chunk (2 bits) | 40.18 | 1.61 | 0.9874 | 0.141 | 2.61 | 0/16 |
| dct q16 | none | 37.90 | 2.39 | 0.9835 | 0.182 | 3.45 | 0/16 |
| dct q16 | deblock (existing) | 38.16 | 1.96 | 0.9837 | 0.180 | 3.40 | 0/16 |
| dct q16 | smooth gated2 zg | 38.28 | 1.87 | 0.9840 | 0.176 | 3.32 | 0/16 |
| dct q16 | smooth gauss0.6 zg | 38.43 | 1.58 | 0.9846 | 0.170 | 3.24 | 0/16 |
| dct q16 | signalled per chunk (2 bits) | 38.29 | 1.85 | 0.9841 | 0.175 | 3.31 | 0/16 |
| surface q48 | none | 38.22 | 2.39 | 1.0000 | 0.157 | 3.19 | 0/16 |
| surface q48 | deblock (existing) | 38.50 | 1.91 | 0.9970 | 0.158 | 3.15 | 0/16 |
| surface q48 | smooth gated2 zg | 38.45 | 2.01 | 1.0000 | 0.156 | 3.18 | 0/16 |
| surface q48 | smooth gauss0.6 zg | 38.72 | 1.61 | 1.0000 | 0.145 | 3.02 | 0/16 |
| surface q48 | signalled per chunk (2 bits) | 38.45 | 2.01 | 1.0000 | 0.156 | 3.18 | 0/16 |
| surface q64 | none | 37.13 | 2.69 | 1.0000 | 0.177 | 3.65 | 0/16 |
| surface q64 | deblock (existing) | 37.46 | 2.13 | 0.9956 | 0.179 | 3.63 | 0/16 |
| surface q64 | smooth gated2 zg | 37.56 | 2.02 | 1.0000 | 0.170 | 3.52 | 0/16 |
| surface q64 | smooth gauss0.6 zg | 37.74 | 1.77 | 1.0000 | 0.161 | 3.40 | 0/16 |
| surface q64 | signalled per chunk (2 bits) | 37.56 | 2.02 | 1.0000 | 0.170 | 3.52 | 0/16 |

#### m7 probability maps (evalbox + r32k cubes, mean)

| setting | candidate | PSNR | block amp | dice | disp mean | band MAE | chunks worse |
|---|---|---|---|---|---|---|---|
| dct q8 | none | 43.55 | 3.08 | 0.9944 | 0.062 | 2.21 | 0/16 |
| dct q8 | deblock (existing) | 44.00 | 2.59 | 0.9947 | 0.059 | 2.13 | 0/16 |
| dct q8 | smooth gated2 zg | 44.02 | 2.62 | 0.9947 | 0.059 | 2.14 | 0/16 |
| dct q8 | smooth gauss0.6 zg | 44.11 | 1.93 | 0.9954 | 0.051 | 2.11 | 0/16 |
| dct q8 | signalled per chunk (2 bits) | 44.05 | 2.61 | 0.9947 | 0.059 | 2.13 | 0/16 |
| dct q16 | none | 39.88 | 3.54 | 0.9913 | 0.095 | 3.51 | 0/16 |
| dct q16 | deblock (existing) | 40.43 | 2.82 | 0.9918 | 0.090 | 3.37 | 0/16 |
| dct q16 | smooth gated2 zg | 40.55 | 2.92 | 0.9919 | 0.090 | 3.40 | 0/16 |
| dct q16 | smooth gauss0.6 zg | 40.94 | 2.29 | 0.9931 | 0.075 | 3.14 | 0/16 |
| dct q16 | signalled per chunk (2 bits) | 40.57 | 2.91 | 0.9919 | 0.089 | 3.36 | 0/16 |
| surface q48 | none | 40.92 | 3.52 | 1.0000 | 0.080 | 2.96 | 0/16 |
| surface q48 | deblock (existing) | 41.54 | 2.70 | 0.9991 | 0.076 | 2.83 | 0/16 |
| surface q48 | smooth gated2 zg | 41.36 | 3.03 | 1.0000 | 0.079 | 2.98 | 0/16 |
| surface q48 | smooth gauss0.6 zg | 41.66 | 2.41 | 1.0000 | 0.066 | 2.78 | 0/16 |
| surface q48 | signalled per chunk (2 bits) | 41.36 | 3.03 | 1.0000 | 0.079 | 2.98 | 0/16 |
| surface q64 | none | 39.24 | 3.74 | 1.0000 | 0.097 | 3.64 | 0/16 |
| surface q64 | deblock (existing) | 39.89 | 2.77 | 0.9986 | 0.092 | 3.51 | 0/16 |
| surface q64 | smooth gated2 zg | 39.88 | 3.03 | 1.0000 | 0.093 | 3.55 | 0/16 |
| surface q64 | smooth gauss0.6 zg | 40.15 | 2.59 | 1.0000 | 0.078 | 3.33 | 0/16 |
| surface q64 | signalled per chunk (2 bits) | 39.88 | 3.03 | 1.0000 | 0.093 | 3.55 | 0/16 |


## 3. Implementation

`volcomp.h`. None of this is part of the format: streams are unchanged, and only
the reader's reconstruction differs.

- **`volcomp_deblock_ex(vol, nz, ny, nx, q, flags)`**
  - `VOLCOMP_DEBLOCK_ZERO_GUARD` skips every face position where one of the four
    voxels the filter reads is exactly 0. AVX2 and C both do it.
  - `volcomp_deblock` now calls it with the guard; `flags = 0` is the old
    filter.
- **`volcomp_decode_smooth(enc, n, dst, cap, strength, flags)`**
  1. Decode the chunk.
  2. Smooth it: with `VOLCOMP_SMOOTH_GATED`, the gated face filter at
     strength×q on this chunk; otherwise a separable Gaussian of σ = strength,
     taken only within 2 voxels of interior block faces. With the zero guard,
     the Gaussian is normalised over nonzero voxels, so air neither spreads nor
     receives.
  3. Re-read every coefficient's level from the stream.
  4. For each block, forward-transform the smoothed block and clamp each
     coefficient into its quantisation cell:
     - level L ≠ 0: |X|/step in [|L| − 0.2, |L| + 0.8] for AC, [|L| − 0.5, |L| + 0.5] for DC;
     - L = 0: |X| < 0.8·step (0.5 for DC).
  5. Inverse-transform. With the zero guard, exact zeros are restored.
  - Mode-6 chunks project onto the base's flat-law cells, then clamp to the
    refined sides, so the threshold stays exact.
  - Lossless and mask chunks decode unchanged.
  - The gated mode does nothing on mode-0 chunks below q 4.
- **Python:**
  - `decode_smooth(enc, sigma, zero_guard=, gated=)`;
  - `deblock(…, zero_guard=True)`;
  - `set_read_smoothing(strength)` for zarr readers. It is a reader property and
    is never written into the array metadata.
- **CLI:** `volcomp decode in.volc out.u8 --smooth[=S]` runs gated S (default
  2) with the zero guard.
- **Tests:**
  - `tests/test_deblock.c` is the masked-edge regression: tissue next to exact-0
    air with the edge on a chunk face, a block face and an oblique plane,
    q 2..32. It checks that no exact zero moves under either filter, that the
    filtered MSE and the edge-band error do not exceed no filter by more than
    2 % (the synthetic texture is white noise, the worst case), and the mode
    handling of `decode_smooth`.
  - `python/test_zarr_roundtrip.py` checks `set_read_smoothing` end to end.

**Speed.** Single thread on forlindesk2, 16 real chunks (`bench_smooth.c`):

| stream | plain decode | smooth gated2 zg | smooth gauss0.6 zg |
|---|---|---|---|
| CT mode 0 q8 | 2 192 MB/s | 383 MB/s (5.5 ms per chunk) | 155 MB/s (13.5 ms) |
| m7 mode 0 q16 | 5 089 MB/s | 388 MB/s | 153 MB/s |
| m7 surface q64 | 1 572 MB/s | 264 MB/s | 129 MB/s |
| recto surface q64 | 1 054 MB/s | 237 MB/s | 123 MB/s |


## 1. Survey

Three research briefs, one per family. In short:

### (a) Encode-only (every existing decoder benefits)

The candidates are:

- rate-distortion-optimised or trellis quantisation (H.264/HEVC RDOQ; Kamaci &
  Shah 2005);
- a boundary-aware term in the RD cost;
- pre-filtering before the transform;
- dithered or noise-shaped quantisation;
- iterative encode–decode–correct loops (POCS at the encoder).

`docs/measured.md` already records RDOQ (+0.8-1.0 % ratio at 1.6-2.4× encode
time, not adopted) and every spatial block-local filter as rejected. The cheapest
new candidate, **error feedback**, was prototyped here: re-encode with the
decoded error at the block faces pushed back into the source.

### (b) Decode-only (works on existing streams)

The candidates are:

- H.264/HEVC deblocking: boundary strength and tc/β tables (List et al. 2003;
  Norkin et al. 2012). `volcomp_deblock` is a simplified version of this.
- AV1 CDEF (Midtskogen & Valin, ICASSP 2018). It targets ringing, which the
  dead-zone quantiser already suppresses, and it is expensive.
- AV1 loop restoration: Wiener and self-guided filters (Mukherjee et al.
  2017).
- Blind SAO. With no transmitted offsets there is nothing to estimate.
- Bilateral / NLM filters and small CNNs such as ARCNN (Dong et al. 2015): 10×
  or more over the decode budget.
- **Quantisation-constraint projection**, POCS: Zakhor 1992; Yang, Galatsanos &
  Katsaggelos 1994. Smooth the decode, transform each block, clamp every
  coefficient back into the quantisation cell it was decoded from, and invert.

The brief ranked POCS first. volcomp's decoder knows every coefficient's cell
exactly: the step law, the dead-zone rounding (0.2 AC, 0.5 DC) and the
reconstruction offsets. A projected result is still a reconstruction that the
stream allows. That constraint is what the rejected free-form spatial filters
lacked.

### (c) Encoder and decoder (a new mode or flag)

The candidates are:

- normative in-loop filters. In an intra-only codec "in-loop" buys nothing
  beyond what a post-filter gives;
- signalled side information: per-chunk filter strength or on/off flags,
  SAO-style offsets, ALF/Wiener taps;
- lapped / overlapped transforms (Malvar LOT; JPEG XR POT; Daala TDLT);
- adaptive block size.

`docs/measured.md` already rejects lapping ("seam energy ×2.2, MSE +54 %"), and
lapping would break chunk independence and single-substream `decode_block`.
Adaptive block size is a format-geometry rewrite. The brief ranked **per-chunk
signalled filter strength** first, and it was prototyped here: the encoder picks
one of four strengths per chunk (2 bits) by MSE against the source.
