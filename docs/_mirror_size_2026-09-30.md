# Volcomp mirror size (2026-09-30)

Measured by crawling the read-only nginx autoindex of https://dl.ash2txt.org/community-uploads/forrest/volcomp/ . Bucket reference: `_bucket_size_2026-09-30.md` (761.1 TB logical level 0, 275.1 TB stored all levels, 239.7 TB stored level 0; 67 volumes).

## Totals

| what | bytes | TB (1e12) |
|---|---|---|
| CT volumes, level 0 | 3,272,277,474,988 | 3.272 |
| CT volumes, all levels (0-5) | 5,234,944,285,012 | 5.235 |
| surface prediction stores (`representations/predictions/surfaces`: m7 prob, m7 th0.2, recto) | 506,179,327,132 | 0.506 |
| teacher_regions stores (`representations/predictions/teacher_regions`, recto+verso 2.4um) | 109,008,405,686 | 0.109 |
| model checkpoints (`representations/models/rvsm-paris4`) | 9,087,418,083 | 0.009 |
| mask stores under representations/ | 0 | none exist (the volumes are already `-masked` scans) |
| everything | 5,859,219,435,913 | 5.859 |

831,103 files, 167,585 directories listed. 64 CT volumes in the mirror (bucket has 67). Level directories are indices 0..5 (not pitches); the pitch is in the scan directory name (`<scan>-<pitch>um-...-masked.zarr`). Every volume is zarr v3, 1024^3 shards of 128^3 volcomp chunks (q=8).

## Ratios against the whole bucket

- Bucket logical level 0 / our CT level 0: 761.1 TB / 3.272 TB = **233x** (but see caveat: our mirror lacks 3 bucket volumes)
- Bucket stored, all levels / our CT all levels: 275.1 TB / 5.235 TB = **52.6x**
- Bucket stored, level 0 / our CT level 0: 239.7 TB / 3.272 TB = **73.3x**
- Bucket logical with full pyramids (869.9 TB) / our all levels: 166x

Caveat: the mirror holds 64 of the bucket's 67 volumes, so the whole-bucket ratios above are very slightly biased by the missing ones (listed below; all three are small, 0.3-0.6 TB stored each, about 1.2 TB together, 0.2 percent of the bucket). Like-for-like on the 64 volumes present in both tables (bucket stored figures are estimates with a few percent error): logical L0 758.0 TB, bucket stored L0 238.6 TB / total 273.9 TB, ours L0 3.272 TB / total 5.235 TB. So on matched volumes **logical/ours = 231.7x (L0)**, **bucket-stored/ours = 72.9x (L0) and 52.3x (all levels)**.

Volumes in the bucket table but not in the mirror:

- PHerc0172 20241024131838 (7.91 um, shape [21000, 6700, 9100], bucket stored total 0.626 TB)
- PHerc0139 20251107132835 (9.362 um, shape [20961, 6621, 6621], bucket stored total 0.302 TB)
- PHerc0139 20251107135911 (9.362 um, shape [20961, 6621, 6621], bucket stored total 0.302 TB)

## Predictions, masks, other

- `representations/predictions/surfaces`: 59 m7 prob stores (`*-surface-m7-L?-prob.zarr`, 0.364 TB), 41 m7 th0.2 stores (0.115 TB), 2 other (recto-090 on PHerc0139, recto-2um-ps256 on PHercParis4; 0.026 TB). Total 102 stores, 0.506 TB. **m7 prob store count now: 59** (expected 61 when the last 3 finish tonight, so 2 more than that are expected beyond what is listed; none of the listed stores is a `.part`, so the 2 missing are not yet uploaded at all).
- `*.part` stores under surfaces: 0.
- `teacher_regions`: 15719 region stores (8386 recto, 7333 verso; recto 0.063 TB, verso 0.046 TB), all under PHercParis4. **595 are still `.part` (uploading), 0.005 TB in `.part` directories so far.** Totals above include the partial bytes.
- Masks: no separate mask stores exist anywhere under representations/ (the volumes are `-masked` CT scans).
- models: rvsm-paris4 checkpoints only (0.009 TB, 26 files).
- No `segments/` directories in the mirror.

## Per-volume

ratio columns: logical L0 bytes / our bytes. Bucket columns come from the bucket doc (estimates): stored L0 in TB, then bucket-stored / ours for L0 and all levels.

| sample | scan | pitch um | shape (z,y,x) | logical Gvox | ours L0 GB | ours all-levels GB | logical/ours L0 | logical/ours total | bucket L0 TB | b/o L0 | b/o total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PHerc0500P2 | 20250821110041 | 0.500 | [12320, 10813, 3891] | 518.34 | 2.25 | 3.52 | 230.7 | 147.2 | 0.536 | 238.6 | 175.2 |
| PHerc0500P2 | 20250821110041 | 0.550 | [12320, 10813, 3891] | 518.34 | 2.25 | 3.52 | 230.7 | 147.2 | 0.536 | 238.6 | 175.2 |
| PHerc0139 | 20260413113053 | 1.129 | [19297, 35971, 32318] | 22432.97 | 60.38 | 97.54 | 371.5 | 230.0 | 13.3 | 220.3 | 156.3 |
| PHerc0814 | 20260521123630 | 1.129 | [46721, 22078, 20284] | 20923.07 | 121.31 | 198.65 | 172.5 | 105.3 | 13.876 | 114.4 | 80.0 |
| PHerc1667 | 20260323082859 | 1.129 | [42209, 22122, 20276] | 18932.66 | 57.93 | 94.30 | 326.8 | 200.8 | 11.015 | 190.2 | 133.9 |
| PHercMANBp | 20260427100434 | 1.129 | [35024, 22137, 20286] | 15728.27 | 8.19 | 13.61 | 1921.3 | 1155.8 | 0.734 | 89.7 | 62.5 |
| PHercParis4 | 20260608103018 | 1.129 | [59969, 36006, 32354] | 69860.17 | 547.82 | 873.97 | 127.5 | 79.9 | 43.178 | 78.8 | 56.6 |
| PHerc0343P | 20260304131111 | 2.215 | [20714, 13155, 13155] | 3584.64 | 4.36 | 7.07 | 822.5 | 507.1 | 0.448 | 102.8 | 74.3 |
| PHerc0500P2 | 20250526151718 | 2.215 | [28096, 18209, 18209] | 9315.73 | 4.46 | 7.30 | 2086.4 | 1276.4 | 0.479 | 107.3 | 76.3 |
| PHerc0139 | 20260102150214 | 2.399 | [76953, 26511, 26511] | 54085.12 | 199.99 | 324.75 | 270.4 | 166.5 | 13.377 | 66.9 | 47.3 |
| PHerc0332 | 20251211183505 | 2.399 | [33592, 15761, 15761] | 8344.56 | 36.17 | 59.15 | 230.7 | 141.1 | 2.034 | 56.2 | 39.6 |
| PHerc0814 | 20260309142202 | 2.399 | [74568, 23891, 23891] | 42561.91 | 196.38 | 316.74 | 216.7 | 134.4 | 10.435 | 53.1 | 37.8 |
| PHerc1299 | 20260309130042 | 2.399 | [49233, 20579, 20579] | 20849.94 | 48.23 | 77.29 | 432.3 | 269.8 | 2.765 | 57.3 | 41.2 |
| PHerc1451 | 20260319101107 | 2.399 | [59944, 20811, 20811] | 25961.61 | 96.65 | 156.99 | 268.6 | 165.4 | 5.417 | 56.0 | 39.6 |
| PHerc1667 | 20251217075048 | 2.399 | [37076, 15229, 15229] | 8598.76 | 28.37 | 46.50 | 303.1 | 184.9 | 2.137 | 75.3 | 53.0 |
| PHercMAN5 | 20260311104824 | 2.399 | [31640, 17495, 17495] | 9684.21 | 22.94 | 36.71 | 422.2 | 263.8 | 1.486 | 64.8 | 46.6 |
| PHercMANB | 20260323091048 | 2.399 | [72681, 26723, 26723] | 51902.86 | 112.42 | 178.31 | 461.7 | 291.1 | 16.586 | 147.5 | 107.6 |
| PHercMANBp | 20251216152116 | 2.399 | [17148, 12577, 12577] | 2712.49 | 1.15 | 1.88 | 2359.7 | 1442.3 | 0.086 | 74.8 | 54.2 |
| PHercParis3 | 20260427095331 | 2.400 | [68417, 42403, 42403] | 123014.75 | 650.03 | 1042.75 | 189.2 | 118.0 | 42.125 | 64.8 | 46.3 |
| PHercParis4 | 20260323153942 | 2.400 | [6625, 8431, 8431] | 470.92 | 8.13 | 12.39 | 57.9 | 38.0 | 0.385 | 47.4 | 35.6 |
| PHercParis4 | 20260411134726 | 2.400 | [75784, 32693, 32693] | 81000.38 | 320.20 | 509.01 | 253.0 | 159.1 | 20.91 | 65.3 | 47.1 |
| PHerc0009B | 20250820154339 | 2.401 | [29112, 28259, 28259] | 23248.00 | 30.57 | 49.19 | 760.4 | 472.7 | 1.683 | 55.0 | 39.5 |
| PHerc0009B | 20260319104112 | 2.401 | [29112, 28259, 28259] | 23248.00 | 23.63 | 37.52 | 984.0 | 619.7 | 1.493 | 63.2 | 45.8 |
| PHerc0139 | 20250820105138 | 2.403 | [19393, 26105, 26105] | 13215.77 | 69.25 | 111.19 | 190.8 | 118.9 | 3.581 | 51.7 | 36.9 |
| PHerc0139 | 20250822062710 | 2.403 | [15137, 26061, 26061] | 10280.68 | 72.16 | 116.93 | 142.5 | 87.9 | 3.763 | 52.1 | 36.9 |
| PHerc0139 | 20260319133050 | 2.403 | [15137, 26061, 26061] | 10280.68 | 61.53 | 99.37 | 167.1 | 103.5 | 3.825 | 62.2 | 44.2 |
| PHerc0139 | 20260319133554 | 2.403 | [19368, 26105, 26105] | 13198.73 | 58.70 | 93.99 | 224.8 | 140.4 | 3.628 | 61.8 | 44.3 |
| PHerc0841 | 20260319124803 | 2.403 | [15121, 30469, 30469] | 14037.73 | 52.34 | 84.92 | 268.2 | 165.3 | 2.985 | 57.0 | 40.4 |
| PHerc0846A | 20260319102732 | 2.403 | [15137, 29587, 29587] | 13250.79 | 68.15 | 110.15 | 194.4 | 120.3 | 3.949 | 57.9 | 41.2 |
| PHerc1203 | 20260319130212 | 2.403 | [15137, 26493, 26493] | 10624.34 | 45.62 | 73.15 | 232.9 | 145.2 | 2.664 | 58.4 | 41.8 |
| PHerc0500P2 | 20250528085330 | 4.317 | [15838, 9423, 9423] | 1406.30 | 0.89 | 1.49 | 1580.4 | 946.8 | 0.089 | 100.0 | 72.7 |
| PHerc0172 | 20241024131839 | 7.910 | [20820, 6700, 9100] | 1269.40 | 25.45 | 36.35 | 49.9 | 34.9 | 0.63 | 24.8 | 20.0 |
| PHerc0009B | 20250521125136 | 8.640 | [9598, 7837, 7837] | 589.50 | 1.30 | 2.00 | 452.0 | 295.4 | 0.064 | 49.1 | 39.1 |
| PHerc0175A | 20250521115057 | 8.640 | [12748, 9363, 9363] | 1117.56 | 7.56 | 11.70 | 147.7 | 95.5 | 0.284 | 37.5 | 28.2 |
| PHerc0175B | 20250521125822 | 8.640 | [15897, 9613, 9613] | 1469.04 | 12.21 | 18.91 | 120.3 | 77.7 | 0.495 | 40.5 | 30.3 |
| PHerc0268 | 20251110183117 | 8.640 | [14833, 12145, 12145] | 2187.88 | 15.74 | 24.27 | 139.0 | 90.1 | 0.753 | 47.8 | 35.8 |
| PHerc0306B | 20250521133212 | 8.640 | [15898, 8849, 8849] | 1244.89 | 9.72 | 14.96 | 128.0 | 83.2 | 0.327 | 33.6 | 25.3 |
| PHerc0343 | 20250521140437 | 8.640 | [17998, 8595, 8595] | 1329.58 | 9.30 | 14.31 | 142.9 | 92.9 | 0.345 | 37.1 | 27.9 |
| PHerc0343P | 20250521134555 | 8.640 | [5398, 5057, 5057] | 138.04 | 0.25 | 0.39 | 550.2 | 353.3 | 0.033 | 131.5 | 94.7 |
| PHerc0483A | 20250521140913 | 8.640 | [15898, 8849, 8849] | 1244.89 | 7.61 | 11.67 | 163.5 | 106.7 | 0.28 | 36.8 | 27.9 |
| PHerc0483B | 20251124083638 | 8.640 | [11880, 8343, 8343] | 826.92 | 6.67 | 10.29 | 124.1 | 80.3 | 0.246 | 36.9 | 27.6 |
| PHerc0490A | 20250521151210 | 8.640 | [11698, 9101, 9101] | 968.92 | 7.93 | 12.22 | 122.2 | 79.3 | 0.292 | 36.8 | 27.6 |
| PHerc0490B | 20250521151215 | 8.640 | [10648, 8343, 8343] | 741.16 | 5.52 | 8.44 | 134.2 | 87.8 | 0.19 | 34.4 | 26.2 |
| PHerc0800 | 20250521135224 | 8.640 | [24298, 9867, 9867] | 2365.60 | 16.76 | 25.79 | 141.2 | 91.7 | 0.646 | 38.5 | 29.0 |
| PHerc1218 | 20250521120456 | 8.640 | [23247, 7593, 7593] | 1340.27 | 7.09 | 10.79 | 189.1 | 124.3 | 0.253 | 35.7 | 27.4 |
| PHerc1447 | 20250521151220 | 8.640 | [24297, 8343, 8343] | 1691.21 | 10.17 | 15.57 | 166.3 | 108.6 | 0.365 | 35.9 | 27.2 |
| PHerc1451 | 20250521151225 | 8.640 | [17998, 5815, 5815] | 608.59 | 4.10 | 6.35 | 148.4 | 95.9 | 0.145 | 35.4 | 26.5 |
| PHerc0125 | 20250821151825 | 9.362 | [20840, 8387, 8387] | 1465.92 | 9.75 | 14.77 | 150.4 | 99.2 | 0.351 | 36.0 | 27.5 |
| PHerc0139 | 20250728140407 | 9.362 | [20974, 6621, 6621] | 919.45 | 8.45 | 12.93 | 108.8 | 71.1 | 0.245 | 29.0 | 22.0 |
| PHerc0191 | 20250821151635 | 9.362 | [18977, 8387, 8387] | 1334.88 | 11.25 | 17.17 | 118.7 | 77.8 | 0.391 | 34.8 | 26.4 |
| PHerc0211 | 20250821151803 | 9.362 | [19416, 7948, 7948] | 1226.52 | 8.09 | 12.31 | 151.7 | 99.6 | 0.296 | 36.6 | 28.0 |
| PHerc0257 | 20250821151750 | 9.362 | [18872, 8388, 8388] | 1327.81 | 7.73 | 11.72 | 171.8 | 113.3 | 0.266 | 34.4 | 26.3 |
| PHerc0358 | 20250821151737 | 9.362 | [14744, 7783, 7783] | 893.12 | 7.26 | 11.04 | 123.0 | 80.9 | 0.244 | 33.6 | 25.5 |
| PHerc0500P2 | 20250820143440 | 9.362 | [7057, 4196, 4196] | 124.25 | 0.20 | 0.31 | 613.0 | 402.0 | 0.012 | 59.2 | 48.5 |
| PHerc0813 | 20250821151723 | 9.362 | [16993, 7947, 7947] | 1073.19 | 8.28 | 12.66 | 129.7 | 84.7 | 0.28 | 33.8 | 25.6 |
| PHerc0814 | 20250804134230 | 9.362 | [19390, 5784, 5784] | 648.69 | 7.18 | 10.94 | 90.3 | 59.3 | 0.19 | 26.5 | 20.2 |
| PHerc0826 | 20250821151701 | 9.362 | [16920, 8169, 8169] | 1129.11 | 6.26 | 9.53 | 180.3 | 118.4 | 0.204 | 32.6 | 25.7 |
| PHerc0846A | 20250728152254 | 9.362 | [14019, 7726, 7726] | 836.81 | 6.42 | 9.78 | 130.2 | 85.6 | 0.225 | 35.0 | 26.7 |
| PHerc0846B | 20250804142305 | 9.362 | [13926, 7859, 7859] | 860.12 | 6.45 | 9.79 | 133.4 | 87.9 | 0.222 | 34.4 | 26.6 |
| PHerc1203 | 20250820131727 | 9.362 | [18977, 6844, 6844] | 888.89 | 7.78 | 11.87 | 114.3 | 74.9 | 0.27 | 34.7 | 26.5 |
| PHerc1545 | 20250821151648 | 9.362 | [20961, 7506, 7506] | 1180.94 | 8.06 | 12.19 | 146.4 | 96.9 | 0.253 | 31.4 | 24.1 |
| PHerc0841 | 20250821151531 | 9.366 | [19400, 7703, 7703] | 1151.12 | 8.97 | 13.63 | 128.3 | 84.5 | 0.269 | 30.0 | 23.0 |
| PHercParis4 | 20260310170716 | 45.532 | [4071, 2264, 2264] | 20.87 | 0.12 | 0.18 | 170.0 | 114.0 | 0.004 | 32.6 | 27.3 |
| PHercParis4 | 20260310173927 | 45.532 | [4066, 2264, 2264] | 20.84 | 0.19 | 0.27 | 110.1 | 77.5 | 0.004 | 21.1 | 18.6 |
| **total** | | | | 758029 | 3272.3 | 5234.9 | 231.7 | 144.8 | | | |

### By pitch group

| pitch um | volumes | logical L0 TB | ours L0 TB | ours total TB | logical/ours L0 |
|---|---|---|---|---|---|
| <=1.2 | 7 | 148.914 | 0.800 | 1.285 | 186.1 |
| 2.2-2.4 | 23 | 573.473 | 2.211 | 3.553 | 259.3 |
| 4.3 | 1 | 1.406 | 0.001 | 0.001 | 1580.4 |
| 7.9-9.4 | 33 | 34.236 | 0.260 | 0.395 | 131.8 |

## Crawl method

Python 3 crawler run from the laptop (8 worker threads, so at most 8 requests in flight, 50 ms sleep per request, User-Agent `volcomp-size-audit/1.0`, retry with exponential backoff on 429/5xx or connection errors; 167,585 GET requests, 0 failed, 56 retries). Each directory listing was parsed from the HTML table (`<td class="size">`) and recursed to the leaf files. Wall time 88 min on 2026-09-30 (about 11:15 to 12:45 CDT). No pagination or truncation was seen (the crawl closed with zero pending directories). Level-0 `zarr.json` of each volume was fetched separately for the shape (64 small GETs). An earlier first attempt stalled and was killed; the numbers here are from the complete second run.

Precision: the autoindex prints sizes rounded to 0.1 of a unit (B/KiB/MiB/GiB), so per-file error is at most ~0.05 of a unit (about 0.05 MiB on a few-hundred-MiB shard, under 0.05 percent). Errors are unbiased and average out, so totals are good to well under 0.1 percent but are not byte-exact. Uploads were in progress during the crawl (verso teacher_regions `.part` stores), so this is a snapshot.
