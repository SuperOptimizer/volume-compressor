# Operating points: bits per voxel at 30 / 35 / 40 dB PSNR

Log-linear interpolation along each codec's sweep; `-` = outside the swept range. Lower is better; **bold** = best in the column.

## PHerc0500P2_0.55um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | - | 0.137 | 0.269 | 10.42 |
| jpeg2000 | 2d | 0.021 | 0.053 | 0.129 | 4.05 |
| webp | 2d | - | 0.092 | 0.210 | 7.02 |
| x264 | video | - | 0.019 | 0.058 | 1.43 |
| x264-intra | video-intra | - | 0.058 | 0.150 | 4.43 |
| zfp-acc | 3d | - | - | - | - |
| zfp-rate | 3d | - | - | 0.573 | - |
| sz3 | 3d | **0.016** | 0.039 | 0.103 | 2.94 |
| volcomp | volcomp | - | 0.013 | 0.039 | 1.00 |
| volcomp-smooth | volcomp | - | **0.011** | **0.033** | 0.86 |

Lossless: blosc-zstd5 4.638 bpv, lz4 7.346 bpv.

## PHerc1667_1.13um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | - | - | - | - |

Lossless: blosc-zstd5 4.592 bpv, lz4 7.453 bpv.

## PHercParis4_2.40um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | - | 0.405 | 0.746 | 3.43 |
| jpeg2000 | 2d | 0.104 | 0.241 | 0.469 | 2.05 |
| jpegxl | 2d | - | 0.232 | 0.418 | 1.97 |
| webp | 2d | - | 0.360 | 0.741 | 3.05 |
| avif | 2d | - | 0.222 | 0.442 | 1.88 |
| x264 | video | 0.069 | 0.193 | 0.455 | 1.64 |
| x265 | video | **0.031** | 0.097 | 0.246 | 0.83 |
| svtav1 | video | - | **0.082** | **0.203** | 0.69 |
| x264-intra | video-intra | 0.120 | 0.298 | 0.636 | 2.53 |
| x265-intra | video-intra | 0.166 | 0.289 | 0.499 | 2.45 |
| svtav1-intra | video-intra | - | 0.219 | 0.420 | 1.85 |
| zfp-acc | 3d | - | - | - | - |
| zfp-rate | 3d | - | 0.742 | 1.149 | 6.30 |
| sz3 | 3d | 0.089 | 0.219 | 0.549 | 1.86 |
| volcomp | volcomp | 0.046 | 0.118 | 0.256 | 1.00 |
| volcomp-smooth | volcomp | 0.039 | 0.107 | 0.251 | 0.90 |

Lossless: zstd19 5.336 bpv, blosc-zstd5 5.987 bpv, lz4 8.025 bpv.

