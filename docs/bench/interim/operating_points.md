# Operating points: bits per voxel at 30 / 35 / 40 dB PSNR

Log-linear interpolation along each codec's sweep; `-` = outside the swept range. Lower is better; **bold** = best in the column.

## PHerc0500P2_0.55um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | 0.075 | 0.137 | 0.269 | 10.42 |
| jpeg2000 | 2d | 0.021 | 0.053 | 0.129 | 4.05 |
| jpegxl | 2d | - | - | 0.141 | - |
| webp | 2d | - | 0.092 | 0.210 | 7.02 |
| avif | 2d | - | 0.054 | 0.124 | 4.09 |
| x264 | video | - | 0.019 | 0.058 | 1.43 |
| x265 | video | - | 0.009 | 0.030 | 0.71 |
| svtav1 | video | - | **0.007** | 0.025 | 0.53 |
| aom | video | - | - | **0.025** | - |
| x264-intra | video-intra | - | 0.058 | 0.150 | 4.43 |
| x265-intra | video-intra | - | 0.119 | 0.190 | 9.05 |
| svtav1-intra | video-intra | - | 0.042 | 0.115 | 3.16 |
| aom-intra | video-intra | - | 0.045 | 0.110 | 3.41 |
| zfp-acc | 3d | - | - | 0.467 | - |
| zfp-rate | 3d | 0.301 | 0.406 | 0.573 | 30.85 |
| sz3 | 3d | **0.016** | 0.039 | 0.103 | 2.94 |
| volcomp | volcomp | - | 0.013 | 0.039 | 1.00 |
| volcomp-smooth | volcomp | - | 0.011 | 0.033 | 0.86 |

Lossless: zstd19 4.134 bpv, blosc-zstd5 4.638 bpv, lz4 7.346 bpv.

## PHerc1667_1.13um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | 0.088 | 0.162 | 0.299 | 10.22 |
| jpeg2000 | 2d | 0.026 | 0.059 | 0.119 | 3.72 |
| jpegxl | 2d | - | - | 0.127 | - |
| webp | 2d | - | 0.107 | 0.243 | 6.76 |
| avif | 2d | - | 0.063 | 0.123 | 3.95 |
| x264 | video | **0.012** | 0.030 | 0.075 | 1.91 |
| x265 | video | - | 0.011 | 0.029 | 0.72 |
| svtav1 | video | - | **0.009** | **0.024** | 0.55 |
| aom | video | - | - | 0.026 | - |
| x264-intra | video-intra | - | 0.075 | 0.169 | 4.75 |
| x265-intra | video-intra | - | 0.127 | 0.189 | 8.03 |
| svtav1-intra | video-intra | - | 0.049 | 0.112 | 3.11 |
| aom-intra | video-intra | - | 0.055 | 0.113 | 3.45 |
| zfp-acc | 3d | - | - | 0.467 | - |
| zfp-rate | 3d | 0.316 | 0.401 | 0.513 | 25.28 |
| sz3 | 3d | 0.022 | 0.046 | 0.081 | 2.90 |
| volcomp | volcomp | - | 0.016 | 0.038 | 1.00 |
| volcomp-smooth | volcomp | - | 0.013 | 0.032 | 0.85 |

Lossless: zstd19 3.924 bpv, blosc-zstd5 4.592 bpv, lz4 7.453 bpv.

## PHercParis4_2.40um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | 0.197 | 0.405 | 0.746 | 3.43 |
| jpeg2000 | 2d | 0.104 | 0.241 | 0.469 | 2.05 |
| jpegxl | 2d | 0.121 | 0.232 | 0.418 | 1.97 |
| webp | 2d | 0.150 | 0.360 | 0.741 | 3.05 |
| avif | 2d | 0.100 | 0.222 | 0.442 | 1.88 |
| x264 | video | 0.069 | 0.193 | 0.455 | 1.64 |
| x265 | video | 0.031 | 0.097 | 0.246 | 0.83 |
| svtav1 | video | **0.027** | **0.082** | **0.203** | 0.69 |
| aom | video | 0.029 | 0.088 | 0.225 | 0.75 |
| x264-intra | video-intra | 0.120 | 0.298 | 0.636 | 2.53 |
| x265-intra | video-intra | 0.166 | 0.289 | 0.499 | 2.45 |
| svtav1-intra | video-intra | 0.088 | 0.219 | 0.420 | 1.85 |
| aom-intra | video-intra | - | 0.207 | 0.411 | 1.76 |
| zfp-acc | 3d | - | 0.689 | 1.095 | 5.84 |
| zfp-rate | 3d | 0.480 | 0.742 | 1.149 | 6.30 |
| sz3 | 3d | 0.089 | 0.219 | 0.549 | 1.86 |
| volcomp | volcomp | 0.046 | 0.118 | 0.256 | 1.00 |
| volcomp-smooth | volcomp | 0.039 | 0.107 | 0.251 | 0.90 |

Lossless: zstd19 5.336 bpv, blosc-zstd5 5.987 bpv, lz4 8.025 bpv.

## PHerc0191_9.36um

| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |
|---|---|--:|--:|--:|--:|
| jpeg | 2d | 0.428 | 0.818 | 1.394 | 2.55 |
| jpeg2000 | 2d | 0.302 | 0.592 | 1.079 | 1.85 |
| jpegxl | 2d | 0.322 | 0.528 | 0.861 | 1.65 |
| webp | 2d | 0.410 | 0.831 | 1.389 | 2.59 |
| avif | 2d | 0.267 | 0.518 | 0.870 | 1.62 |
| x264 | video | 0.229 | 0.519 | 0.957 | 1.62 |
| x265 | video | 0.132 | 0.318 | 0.618 | 0.99 |
| svtav1 | video | **0.113** | **0.277** | **0.542** | 0.86 |
| aom | video | 0.119 | 0.287 | 0.589 | 0.90 |
| x264-intra | video-intra | 0.333 | 0.696 | 1.230 | 2.17 |
| x265-intra | video-intra | 0.328 | 0.581 | 0.936 | 1.81 |
| svtav1-intra | video-intra | 0.260 | 0.485 | 0.813 | 1.51 |
| aom-intra | video-intra | 0.248 | 0.498 | 0.837 | 1.55 |
| zfp-acc | 3d | - | 1.200 | 1.843 | 3.74 |
| zfp-rate | 3d | 0.812 | 1.318 | 2.004 | 4.11 |
| sz3 | 3d | 0.306 | 0.740 | 1.348 | 2.31 |
| volcomp | volcomp | 0.148 | 0.321 | 0.594 | 1.00 |
| volcomp-smooth | volcomp | 0.136 | 0.315 | 0.594 | 0.98 |

Lossless: zstd19 6.212 bpv, blosc-zstd5 6.683 bpv, lz4 8.027 bpv.

