# Speed and random access at the cheapest setting reaching 35 dB PSNR

One thread. `quiet` = the single-worker timing pass (`run --timing --workers 1`); `par` = the 12-worker quality pass (contended; shown only where no quiet number exists). Random access = bytes read and wall time to reconstruct the 128^3 chunk at the cube centre (best of 3).

## PHerc0500P2_0.55um

| codec | setting | PSNR dB | bpv | enc MB/s | dec MB/s | RA bytes | RA ms | units read |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| zstd19 | level 19 | inf | 4.1335 | 1.2 (par) | 498.6 (par) | 1,106,365 | 2.4 (par) | 1 |
| blosc-zstd5 | clevel 5 | inf | 4.6385 | 40.9 (par) | 599.6 (par) | 1,241,519 | 2.6 (par) | 1 |
| lz4 | level 0 | inf | 7.3456 | 603.5 (par) | 1984.4 (par) | 2,000,602 | 1.5 (par) | 1 |
| jpeg | quality 25 | 39.42 | 0.2481 | 146.1 (par) | 621.6 (par) | 1,011,485 | 41.7 (par) | 128 |
| jpeg2000 | target PSNR 37 | 36.17 | 0.0645 | 26.0 (par) | 171.4 (par) | 250,517 | 112.3 (par) | 128 |
| jpegxl | distance 25 | 35.09 | 0.0669 | 3.4 (par) | 53.4 (par) | 275,346 | 360.7 (par) | 128 |
| webp | quality 20 | 36.91 | 0.1322 | 19.0 (par) | 107.4 (par) | 534,938 | 100.4 (par) | 128 |
| avif | quality 10 | 35.79 | 0.0608 | 7.4 (par) | 225.5 (par) | 248,037 | 117.5 (par) | 128 |
| x264 | crf 32 | 37.50 | 0.0324 | 46.1 (par) | 376.3 (par) | 121,033 | 66.3 (par) | 1 |
| x265 | crf 32 | 37.72 | 0.0171 | 8.1 (par) | 175.3 (par) | 67,136 | 133.6 (par) | 1 |
| svtav1 | crf 58 | 36.25 | 0.0095 | 13.4 (par) | 294.2 (par) | 38,183 | 74.9 (par) | 1 |
| aom | crf 58 | 35.36 | 0.0080 | 12.8 (par) | 893.5 (par) | 31,874 | 128.7 (par) | 1 |
| x264-intra | crf 32 | 36.55 | 0.0777 | 77.3 (par) | 230.1 (par) | 310,049 | 108.4 (par) | 1 |
| x265-intra | crf 32 | 37.38 | 0.1426 | 5.2 (par) | 344.0 (par) | 597,883 | 154.6 (par) | 1 |
| svtav1-intra | crf 53 | 36.34 | 0.0548 | 5.8 (par) | 297.7 (par) | 219,757 | 86.1 (par) | 1 |
| aom-intra | crf 38 | 39.69 | 0.1035 | 7.5 (par) | 229.2 (par) | 411,433 | 116.8 (par) | 1 |
| zfp-acc | tolerance 256 | 38.24 | 0.4190 | 125.9 (par) | 84.9 (par) | 112,168 | 33.2 (par) | 1 |
| zfp-rate | bits/value 0.5 | 38.52 | 0.5001 | 145.3 (par) | 97.3 (par) | 131,088 | 16.8 (par) | 1 |
| sz3 | abs error 8 | 39.77 | 0.0973 | 26.4 (par) | 31.5 (par) | 26,024 | 69.5 (par) | 1 |
| volcomp | q 32 | 36.68 | 0.0187 | 124.3 (par) | 744.3 (par) | 5,159 | 2.3 (par) | 1 |
| volcomp-smooth | q 32 | 37.41 | 0.0187 | 59.9 (par) | 68.3 (par) | 5,159 | 17.7 (par) | 1 |

## PHerc1667_1.13um

| codec | setting | PSNR dB | bpv | enc MB/s | dec MB/s | RA bytes | RA ms | units read |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| zstd19 | level 19 | inf | 3.9241 | 1.1 (par) | 310.0 (par) | 1,052,790 | 2.3 (par) | 1 |
| blosc-zstd5 | clevel 5 | inf | 4.5924 | 45.2 (par) | 397.5 (par) | 1,219,597 | 2.0 (par) | 1 |
| lz4 | level 0 | inf | 7.4535 | 204.9 (par) | 411.8 (par) | 2,010,896 | 6.1 (par) | 1 |
| jpeg | quality 25 | 39.59 | 0.2857 | 133.4 (par) | 411.0 (par) | 1,224,477 | 64.4 (par) | 128 |
| jpeg2000 | target PSNR 37 | 36.05 | 0.0696 | 29.2 (par) | 197.0 (par) | 315,583 | 204.8 (par) | 128 |
| jpegxl | distance 25 | 35.43 | 0.0685 | 2.9 (par) | 46.1 (par) | 298,595 | 411.4 (par) | 128 |
| webp | quality 5 | 35.31 | 0.1134 | 12.5 (par) | 150.6 (par) | 487,574 | 202.6 (par) | 128 |
| avif | quality 10 | 35.71 | 0.0686 | 7.6 (par) | 224.6 (par) | 296,177 | 130.7 (par) | 128 |
| x264 | crf 32 | 37.57 | 0.0485 | 13.7 (par) | 156.0 (par) | 212,133 | 496.7 (par) | 1 |
| x265 | crf 32 | 38.17 | 0.0206 | 15.0 (par) | 205.5 (par) | 89,971 | 87.0 (par) | 1 |
| svtav1 | crf 58 | 36.26 | 0.0113 | 12.8 (par) | 205.8 (par) | 49,580 | 154.3 (par) | 1 |
| aom | crf 58 | 35.25 | 0.0100 | 9.9 (par) | 236.9 (par) | 43,712 | 52.5 (par) | 1 |
| x264-intra | crf 32 | 36.07 | 0.0905 | 43.8 (par) | 108.5 (par) | 383,971 | 178.5 (par) | 1 |
| x265-intra | crf 32 | 37.23 | 0.1481 | 6.2 (par) | 240.4 (par) | 629,851 | 181.2 (par) | 1 |
| svtav1-intra | crf 53 | 36.45 | 0.0635 | 4.9 (par) | 159.3 (par) | 277,241 | 109.9 (par) | 1 |
| aom-intra | crf 38 | 40.22 | 0.1162 | 6.9 (par) | 219.9 (par) | 506,042 | 110.4 (par) | 1 |
| zfp-acc | tolerance 256 | 38.10 | 0.4219 | 76.2 (par) | 44.3 (par) | 114,536 | 33.9 (par) | 1 |
| zfp-rate | bits/value 0.5 | 39.66 | 0.5001 | 76.1 (par) | 40.1 (par) | 131,088 | 37.7 (par) | 1 |
| sz3 | abs error 8 | 39.75 | 0.0776 | 15.9 (par) | 15.7 (par) | 25,050 | 90.2 (par) | 1 |
| volcomp | q 32 | 36.87 | 0.0221 | 25.0 (par) | 81.1 (par) | 5,942 | 3.7 (par) | 1 |
| volcomp-smooth | q 64 | 35.02 | 0.0134 | 56.5 (par) | 46.9 (par) | 3,536 | 43.1 (par) | 1 |

## PHercParis4_2.40um

| codec | setting | PSNR dB | bpv | enc MB/s | dec MB/s | RA bytes | RA ms | units read |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| zstd19 | level 19 | inf | 5.3363 | 1.0 (par) | 84.2 (par) | 1,374,996 | 4.8 (par) | 1 |
| blosc-zstd5 | clevel 5 | inf | 5.9872 | 36.6 (par) | 218.4 (par) | 1,533,810 | 1.9 (par) | 1 |
| lz4 | level 0 | inf | 8.0254 | 1509.3 (par) | 2611.9 (par) | 2,101,552 | 1.5 (par) | 1 |
| jpeg | quality 25 | 36.27 | 0.4799 | 129.7 (par) | 454.3 (par) | 2,000,292 | 61.4 (par) | 128 |
| jpeg2000 | target PSNR 37 | 36.37 | 0.2976 | 18.7 (par) | 103.9 (par) | 1,243,628 | 284.8 (par) | 128 |
| jpegxl | distance 8 | 35.31 | 0.2389 | 1.8 (par) | 25.9 (par) | 996,758 | 535.4 (par) | 128 |
| webp | quality 40 | 36.18 | 0.4295 | 16.0 (par) | 168.1 (par) | 1,793,522 | 169.8 (par) | 128 |
| avif | quality 40 | 36.94 | 0.2944 | 2.4 (par) | 97.3 (par) | 1,234,321 | 188.5 (par) | 128 |
| x264 | crf 26 | 37.40 | 0.2994 | 16.3 (par) | 144.1 (par) | 1,247,242 | 215.8 (par) | 1 |
| x265 | crf 26 | 36.84 | 0.1414 | 6.5 (par) | 105.3 (par) | 595,107 | 502.4 (par) | 1 |
| svtav1 | crf 38 | 36.94 | 0.1185 | 11.5 (par) | 219.4 (par) | 491,789 | 144.3 (par) | 1 |
| aom | crf 38 | 36.04 | 0.1085 | 3.7 (par) | 71.0 (par) | 447,732 | 331.4 (par) | 1 |
| x264-intra | crf 24 | 35.93 | 0.3515 | 45.1 (par) | 185.9 (par) | 1,467,778 | 156.8 (par) | 1 |
| x265-intra | crf 24 | 36.92 | 0.3590 | 3.3 (par) | 131.0 (par) | 1,497,105 | 194.0 (par) | 1 |
| svtav1-intra | crf 38 | 37.58 | 0.3248 | 5.6 (par) | 150.3 (par) | 1,357,850 | 127.9 (par) | 1 |
| aom-intra | crf 38 | 36.45 | 0.2597 | 3.9 (par) | 298.6 (par) | 1,083,864 | 138.1 (par) | 1 |
| zfp-acc | tolerance 128 | 37.85 | 0.8947 | 69.0 (par) | 62.7 (par) | 227,776 | 20.9 (par) | 1 |
| zfp-rate | bits/value 1 | 38.38 | 1.0001 | 96.8 (par) | 57.7 (par) | 262,160 | 34.5 (par) | 1 |
| sz3 | abs error 8 | 38.09 | 0.3532 | 25.0 (par) | 22.9 (par) | 95,078 | 82.3 (par) | 1 |
| volcomp | q 8 | 36.55 | 0.1528 | 52.9 (par) | 390.7 (par) | 38,504 | 2.8 (par) | 1 |
| volcomp-smooth | q 8 | 37.04 | 0.1528 | 54.4 (par) | 66.9 (par) | 38,504 | 20.3 (par) | 1 |

## PHerc0191_9.36um

| codec | setting | PSNR dB | bpv | enc MB/s | dec MB/s | RA bytes | RA ms | units read |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| zstd19 | level 19 | inf | 6.2120 | 2.3 (par) | 147.7 (par) | 1,559,150 | 3.9 (par) | 1 |
| blosc-zstd5 | clevel 5 | inf | 6.6831 | 17.9 (par) | 127.8 (par) | 1,689,456 | 4.0 (par) | 1 |
| lz4 | level 0 | inf | 8.0267 | 144.1 (par) | 167.6 (par) | 2,105,376 | 6.7 (par) | 1 |
| jpeg | quality 50 | 37.12 | 1.0376 | 88.3 (par) | 195.6 (par) | 4,285,952 | 89.9 (par) | 128 |
| jpeg2000 | target PSNR 37 | 36.42 | 0.7072 | 9.4 (par) | 21.7 (par) | 2,821,974 | 548.0 (par) | 128 |
| jpegxl | distance 3 | 37.99 | 0.7089 | 3.1 (par) | 65.4 (par) | 2,920,681 | 477.3 (par) | 128 |
| webp | quality 60 | 35.70 | 0.9024 | 8.8 (par) | 80.7 (par) | 3,721,264 | 236.2 (par) | 128 |
| avif | quality 55 | 37.32 | 0.6770 | 2.5 (par) | 133.2 (par) | 2,821,473 | 247.0 (par) | 128 |
| x264 | crf 26 | 35.58 | 0.5666 | 13.6 (par) | 119.8 (par) | 2,367,158 | 221.1 (par) | 1 |
| x265 | crf 20 | 38.85 | 0.5410 | 3.3 (par) | 81.9 (par) | 2,253,142 | 395.0 (par) | 1 |
| svtav1 | crf 28 | 38.98 | 0.4793 | 8.2 (par) | 130.0 (par) | 1,995,248 | 239.9 (par) | 1 |
| aom | crf 28 | 37.88 | 0.4472 | 1.4 (par) | 66.4 (par) | 1,876,397 | 292.6 (par) | 1 |
| x264-intra | crf 16 | 38.78 | 1.0932 | 26.1 (par) | 79.4 (par) | 4,563,026 | 382.7 (par) | 1 |
| x265-intra | crf 16 | 40.10 | 0.9452 | 2.5 (par) | 96.5 (par) | 3,936,592 | 449.4 (par) | 1 |
| svtav1-intra | crf 38 | 36.36 | 0.5751 | 4.7 (par) | 152.0 (par) | 2,354,282 | 218.5 (par) | 1 |
| aom-intra | crf 23 | 40.77 | 0.9067 | 2.8 (par) | 106.2 (par) | 3,716,950 | 230.0 (par) | 1 |
| zfp-acc | tolerance 128 | 36.06 | 1.3304 | 66.9 (par) | 47.1 (par) | 335,664 | 12.4 (par) | 1 |
| zfp-rate | bits/value 1.5 | 36.41 | 1.5001 | 82.9 (par) | 78.4 (par) | 393,232 | 39.1 (par) | 1 |
| sz3 | abs error 8 | 35.94 | 0.8727 | 19.7 (par) | 16.7 (par) | 198,003 | 72.5 (par) | 1 |
| volcomp | q 4 | 37.19 | 0.4307 | 34.7 (par) | 132.6 (par) | 107,187 | 3.7 (par) | 1 |
| volcomp-smooth | q 4 | 37.20 | 0.4307 | 53.1 (par) | 67.7 (par) | 107,187 | 24.2 (par) | 1 |

