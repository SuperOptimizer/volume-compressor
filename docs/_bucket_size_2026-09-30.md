# Vesuvius Challenge open-data bucket size (2026-09-30)

Bucket s3://vesuvius-challenge-open-data, listed anonymously via aws CLI on forlindesk2. Every volume is zarr v2, uint8, uncompressed, 128^3 chunks; only non-empty chunks are stored (masked volumes are mostly empty), so stored << logical.

Logical = prod(level-0 .zarray shape) bytes. Stored = bytes under the whole volume prefix (all pyramid levels); 'exact' = full `ls --summarize --recursive`, 'est' = 12 evenly spaced z-planes of level 0 (and of level 1, x8/7 for deeper levels), scaled to the volume. Stored L0 is level 0 only.

| sample | volume | um | shape (z,y,x) | logical L0 TB | stored L0 TB | stored total TB | exact/est | stored L0 / logical |
|---|---|---|---|---|---|---|---|---|
| PHerc0500P2 | 20250821110041 | 0.5 | [12320, 10813, 3891] | 0.518 | 0.536 | 0.617 | est | 1.034 |
| PHerc0500P2 | 20250821110041 | 0.55 | [12320, 10813, 3891] | 0.518 | 0.536 | 0.617 | est | 1.034 |
| PHerc0139 | 20260413113053 | 1.129 | [19297, 35971, 32318] | 22.433 | 13.300 | 15.241 | est | 0.593 |
| PHerc0814 | 20260521123630 | 1.129 | [46721, 22078, 20284] | 20.923 | 13.876 | 15.897 | est | 0.663 |
| PHerc1667 | 20260323082859 | 1.129 | [42209, 22122, 20276] | 18.933 | 11.015 | 12.627 | est | 0.582 |
| PHercMANBp | 20260427100434 | 1.129 | [35024, 22137, 20286] | 15.728 | 0.734 | 0.851 | est | 0.047 |
| PHercParis4 | 20260608103018 | 1.129 | [59969, 36006, 32354] | 69.860 | 43.178 | 49.444 | est | 0.618 |
| PHerc0343P | 20260304131111 | 2.215 | [20714, 13155, 13155] | 3.585 | 0.448 | 0.525 | est | 0.125 |
| PHerc0500P2 | 20250526151718 | 2.215 | [28096, 18209, 18209] | 9.316 | 0.479 | 0.557 | est | 0.051 |
| PHerc0139 | 20260102150214 | 2.399 | [76953, 26511, 26511] | 54.085 | 13.377 | 15.365 | est | 0.247 |
| PHerc0332 | 20251211183505 | 2.399 | [33592, 15761, 15761] | 8.345 | 2.034 | 2.340 | est | 0.244 |
| PHerc0814 | 20260309142202 | 2.399 | [74568, 23891, 23891] | 42.562 | 10.435 | 11.984 | est | 0.245 |
| PHerc1299 | 20260309130042 | 2.399 | [49233, 20579, 20579] | 20.850 | 2.765 | 3.185 | est | 0.133 |
| PHerc1451 | 20260319101107 | 2.399 | [59944, 20811, 20811] | 25.962 | 5.417 | 6.219 | est | 0.209 |
| PHerc1667 | 20251217075048 | 2.399 | [37076, 15229, 15229] | 8.599 | 2.137 | 2.463 | est | 0.249 |
| PHercMAN5 | 20260311104824 | 2.399 | [31640, 17495, 17495] | 9.684 | 1.486 | 1.712 | est | 0.153 |
| PHercMANB | 20260323091048 | 2.399 | [72681, 26723, 26723] | 51.903 | 16.586 | 19.182 | est | 0.320 |
| PHercMANBp | 20251216152116 | 2.399 | [17148, 12577, 12577] | 2.712 | 0.086 | 0.102 | exact | 0.032 |
| PHercParis3 | 20260427095331 | 2.4 | [68417, 42403, 42403] | 123.015 | 42.125 | 48.251 | est | 0.342 |
| PHercParis4 | 20260323153942 | 2.4 | [6625, 8431, 8431] | 0.471 | 0.385 | 0.441 | est | 0.817 |
| PHercParis4 | 20260411134726 | 2.4 | [75784, 32693, 32693] | 81.000 | 20.910 | 23.978 | est | 0.258 |
| PHerc0009B | 20250820154339 | 2.401 | [29112, 28259, 28259] | 23.248 | 1.683 | 1.943 | est | 0.072 |
| PHerc0009B | 20260319104112 | 2.401 | [29112, 28259, 28259] | 23.248 | 1.493 | 1.720 | est | 0.064 |
| PHerc0139 | 20250820105138 | 2.403 | [19393, 26105, 26105] | 13.216 | 3.581 | 4.107 | est | 0.271 |
| PHerc0139 | 20250822062710 | 2.403 | [15137, 26061, 26061] | 10.281 | 3.763 | 4.319 | est | 0.366 |
| PHerc0139 | 20260319133050 | 2.403 | [15137, 26061, 26061] | 10.281 | 3.825 | 4.389 | est | 0.372 |
| PHerc0139 | 20260319133554 | 2.403 | [19368, 26105, 26105] | 13.199 | 3.628 | 4.160 | est | 0.275 |
| PHerc0841 | 20260319124803 | 2.403 | [15121, 30469, 30469] | 14.038 | 2.985 | 3.428 | est | 0.213 |
| PHerc0846A | 20260319102732 | 2.403 | [15137, 29587, 29587] | 13.251 | 3.949 | 4.534 | est | 0.298 |
| PHerc1203 | 20260319130212 | 2.403 | [15137, 26493, 26493] | 10.624 | 2.664 | 3.059 | est | 0.251 |
| PHerc0500P2 | 20250528085330 | 4.317 | [15838, 9423, 9423] | 1.406 | 0.089 | 0.108 | exact | 0.063 |
| PHerc0172 | 20241024131838 | 7.91 | [21000, 6700, 9100] | 1.280 | 0.547 | 0.626 | est | 0.427 |
| PHerc0172 | 20241024131839 | 7.91 | [20820, 6700, 9100] | 1.269 | 0.630 | 0.727 | est | 0.497 |
| PHerc0009B | 20250521125136 | 8.64 | [9598, 7837, 7837] | 0.589 | 0.064 | 0.078 | exact | 0.109 |
| PHerc0175A | 20250521115057 | 8.64 | [12748, 9363, 9363] | 1.118 | 0.284 | 0.330 | est | 0.254 |
| PHerc0175B | 20250521125822 | 8.64 | [15897, 9613, 9613] | 1.469 | 0.495 | 0.573 | est | 0.337 |
| PHerc0268 | 20251110183117 | 8.64 | [14833, 12145, 12145] | 2.188 | 0.753 | 0.868 | est | 0.344 |
| PHerc0306B | 20250521133212 | 8.64 | [15898, 8849, 8849] | 1.245 | 0.327 | 0.379 | est | 0.263 |
| PHerc0343 | 20250521140437 | 8.64 | [17998, 8595, 8595] | 1.330 | 0.345 | 0.399 | est | 0.259 |
| PHerc0343P | 20250521134555 | 8.64 | [5398, 5057, 5057] | 0.138 | 0.033 | 0.037 | exact | 0.236 |
| PHerc0483A | 20250521140913 | 8.64 | [15898, 8849, 8849] | 1.245 | 0.280 | 0.325 | est | 0.225 |
| PHerc0483B | 20251124083638 | 8.64 | [11880, 8343, 8343] | 0.827 | 0.246 | 0.284 | est | 0.298 |
| PHerc0490A | 20250521151210 | 8.64 | [11698, 9101, 9101] | 0.969 | 0.292 | 0.338 | est | 0.301 |
| PHerc0490B | 20250521151215 | 8.64 | [10648, 8343, 8343] | 0.741 | 0.190 | 0.221 | est | 0.257 |
| PHerc0800 | 20250521135224 | 8.64 | [24298, 9867, 9867] | 2.366 | 0.646 | 0.748 | est | 0.273 |
| PHerc1218 | 20250521120456 | 8.64 | [23247, 7593, 7593] | 1.340 | 0.253 | 0.295 | est | 0.189 |
| PHerc1447 | 20250521151220 | 8.64 | [24297, 8343, 8343] | 1.691 | 0.365 | 0.424 | est | 0.216 |
| PHerc1451 | 20250521151225 | 8.64 | [17998, 5815, 5815] | 0.609 | 0.145 | 0.168 | est | 0.237 |
| PHerc0125 | 20250821151825 | 9.362 | [20840, 8387, 8387] | 1.466 | 0.351 | 0.407 | est | 0.240 |
| PHerc0139 | 20250728140407 | 9.362 | [20974, 6621, 6621] | 0.919 | 0.245 | 0.284 | est | 0.266 |
| PHerc0139 | 20251107132835 | 9.362 | [20961, 6621, 6621] | 0.919 | 0.261 | 0.302 | est | 0.284 |
| PHerc0139 | 20251107135911 | 9.362 | [20961, 6621, 6621] | 0.919 | 0.261 | 0.302 | est | 0.284 |
| PHerc0191 | 20250821151635 | 9.362 | [18977, 8387, 8387] | 1.335 | 0.391 | 0.454 | est | 0.293 |
| PHerc0211 | 20250821151803 | 9.362 | [19416, 7948, 7948] | 1.227 | 0.296 | 0.345 | est | 0.241 |
| PHerc0257 | 20250821151750 | 9.362 | [18872, 8388, 8388] | 1.328 | 0.266 | 0.308 | est | 0.200 |
| PHerc0358 | 20250821151737 | 9.362 | [14744, 7783, 7783] | 0.893 | 0.244 | 0.282 | est | 0.273 |
| PHerc0500P2 | 20250820143440 | 9.362 | [7057, 4196, 4196] | 0.124 | 0.012 | 0.015 | exact | 0.098 |
| PHerc0813 | 20250821151723 | 9.362 | [16993, 7947, 7947] | 1.073 | 0.280 | 0.324 | est | 0.261 |
| PHerc0814 | 20250804134230 | 9.362 | [19390, 5784, 5784] | 0.649 | 0.190 | 0.221 | est | 0.293 |
| PHerc0826 | 20250821151701 | 9.362 | [16920, 8169, 8169] | 1.129 | 0.204 | 0.245 | est | 0.180 |
| PHerc0846A | 20250728152254 | 9.362 | [14019, 7726, 7726] | 0.837 | 0.225 | 0.261 | est | 0.268 |
| PHerc0846B | 20250804142305 | 9.362 | [13926, 7859, 7859] | 0.860 | 0.222 | 0.260 | est | 0.258 |
| PHerc1203 | 20250820131727 | 9.362 | [18977, 6844, 6844] | 0.889 | 0.270 | 0.315 | est | 0.304 |
| PHerc1545 | 20250821151648 | 9.362 | [20961, 7506, 7506] | 1.181 | 0.253 | 0.294 | est | 0.214 |
| PHerc0841 | 20250821151531 | 9.366 | [19400, 7703, 7703] | 1.151 | 0.269 | 0.313 | est | 0.234 |
| PHercParis4 | 20260310170716 | 45.532 | [4071, 2264, 2264] | 0.021 | 0.004 | 0.005 | exact | 0.213 |
| PHercParis4 | 20260310173927 | 45.532 | [4066, 2264, 2264] | 0.021 | 0.004 | 0.005 | exact | 0.215 |

## Totals

- 45 sample prefixes at bucket top level (plus _thumbnails); 39 have volumes in this bucket, 6 (PHerc1667Cr1Fr3, PHerc51Cr4Fr8, PHercParis1Fr34, PHercParis1Fr39, PHercParis2Fr143, PHercParis2Fr47) have none, volumes: 67
- logical level 0: 761.1 TB
- pyramid overhead (+1/7, 6 levels): 108.7 TB; logical with pyramids: 869.9 TB
- stored, level 0 only: 239.7 TB
- stored, all levels: 275.1 TB (exact part 0.4 TB, estimated part 274.8 TB)

## By pitch group

| pitch um | volumes | logical L0 TB | stored total TB |
|---|---|---|---|
| <=1.2 | 7 | 148.9 | 95.3 |
| 2.2-2.4 | 23 | 573.5 | 168.0 |
| 4.3 | 1 | 1.4 | 0.1 |
| 8.6-9.4 | 34 | 37.3 | 11.8 |
| 45 | 2 | 0.0 | 0.0 |

## Notes and caveats

- Only 0.4 TB of the stored total was measured exactly (volumes with < 50k level-0 objects); the rest is a 12-plane systematic estimate over z, so expect a few percent error on stored numbers. Logical numbers are exact (from level-0 .zarray).
- PHerc0500P2 has two volume prefixes (0.500um-masked and 0.550um-0.1m-65keV-masked) with identical shape and identical sampled bytes; they look like duplicate copies, so both are counted (both are billed). Its stored L0 slightly exceeds logical (ratio 1.03): some level-0 objects are larger than a 128^3 chunk (e.g. 16 MiB and 128 MiB objects in PHerc0343P), so the store is not a clean 128^3 chunk grid and byte counts include those.
- Level-0 zarr arrays are uncompressed; stored/logical ratio reflects mostly empty (masked-out) chunks that are not written. Ratio ~0.2-0.3 for 8.6-9.4 um and 2.4 um scrolls, 0.6-0.66 for 1.1 um.
- The old 3.24 and 7.91 um scans in metadata.json are hosted at data.aws.ash2txt.org, not in this bucket, and are excluded.
- Sampling was not reduced to 6 planes; all sampled volumes used 12 level-0 and 12 level-1 planes.
- Cross-check: 34 volumes at 8.6-9.4 um total 37.3 TB logical vs 34.2 TB for the 31 native m7 volumes (extra: pag0/pag50 variants, duplicates, PHerc0500P2).

## Surface prediction stores (<sample>/representations/predictions/surfaces/), all exact

| kind | stores | objects | TB |
|---|---|---|---|
| zarr | 43 | 3,883,682 | 1.43 |
| normal-grids | 41 | 1,306,838 | 0.87 |

- total 84 stores, 2.30 TB. The recto-2um-ps256-L0-th0.45 store for PHercParis4 20260411134726 alone is 0.72 TB (+0.41 TB normal-grids).
- Segments: 327 segment directories across samples (top-level count of <sample>/segments/, capped at 2000 per sample; 45 samples listed). Sizes not measured.

## Answer: is the bucket ~300 TB?

Logical level-0 voxels are about 761 TB (870 TB with full pyramids), so "~300 TB" is far below the logical size. What the bucket actually stores (and bills) is much smaller because empty masked chunks are not written: about 240 TB for level 0 and about 275 TB for volumes with all pyramid levels, plus about 2.3 TB of surface predictions, about 277 TB in total. So ~300 TB matches stored bytes (volumes plus surface predictions), not the logical size. The fine-pitch scans (<= 2.4 um) hold about 720 TB logical and about 260 TB stored, i.e. nearly all of it.
