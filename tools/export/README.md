# Bulk export: S3 (raw zarr v2) → SFTP (zarr v3 + volcomp)

Re-encodes every uncompressed volume of the Vesuvius Challenge open-data bucket
(`vesuvius-challenge-open-data`, 64 volumes, ~760 TB logical at level 0 plus the
LOD pyramids) — and its 43 published surface predictions, as binary **mask
chunks** on the exact resolution ladder (see **Surface predictions**) — into zarr v3
`sharding_indexed` arrays — 1024³ shards of 128³
`volcomp` chunks (q = 8 at native resolution, 4 / 2 / 1 at LOD levels 1 / 2 / ≥3: downscaling averages noise away, so coarser levels keep more) — under `sftp://dl.ash2txt.org:9238/volcomp/`, mirroring the
bucket keys: `volcomp/<scroll>/volumes/<volume>.zarr/<level>/c/<sz>/<sy>/<sx>`.

Everything is Python 3 standard library + the `volcomp` CLI + OpenSSH `sftp` (+`sshpass`); nothing to
pip-install on the VMs.

## Pieces

| file | runs on | does |
|---|---|---|
| `coordinator.py` | coordinator VM | sqlite queue of shard units; HTTP claim/done/fail/status; writes the zarr v3 metadata tree |
| `worker.py` | every compute VM | claim → download 512 source chunks (threads) → `volcomp shard-pack` → `volcomp shard-verify` → SFTP upload (`.part` + rename) → done; a surface unit downloads its source footprint and calls `volcomp surface-pack` once |
| `test_export.py` | your machine | offline end-to-end test: a synthetic bucket, the real coordinator and worker, decoded back and compared with a scipy-computed ramp |
| `fleet.py` | your machine | Blue Lobster VMs: launch / wait / bootstrap / list / ssh / destroy |
| `bootstrap.sh` | pushed by `fleet.py` | installs clang+cmake, builds `volcomp` at a pinned commit, installs systemd services |

A **unit** is one (volume, level, shard) triple: up to 512 source chunks (1 GiB
raw), ~20 MB out at q8. Units are leased for 15 minutes; a worker that dies
simply lets its leases expire. `/done` is idempotent, uploads are atomic
(`<key>.part` → rename), and the manifest can be rebuilt at any time without
losing progress, so the whole thing is safe to stop and resume.

## Occupancy masks (`coordinator.py occupancy`)

Most of every volume is masked air. Before (or while) the workers run, the
coordinator downloads each volume's coarsest level (≤ 5; a level-5 voxel is a
32³ block of native voxels, so the whole level is 1/32768 of the data) and runs
`volcomp occupancy --chunks` on it once per level: a chunk of level *k* is
"occupied" when any coarse voxel within its footprint, dilated by one coarse
voxel, is nonzero. Every unit gets a 512-bit chunk mask and `est`, the number
of occupied chunks; units with `est = 0` are marked done on the spot (a missing
shard key is the fill value) and workers request only masked-in chunks (a 404
is simply "absent"; no per-row listing). `report` uses Σ`est` of the remaining
units for the ETA. Checked on a real volume: every chunk stored in S3 was
masked in. The one thing the mask can drop is an isolated speck whose 32³ mean
rounds to 0 with nothing nonzero in the neighbouring coarse voxels.

```sh
sudo systemd-run --unit volcomp-occupancy volcomp-coordinator occupancy --db /var/lib/volcomp/export.db --volcomp /usr/local/bin/volcomp --tmp /var/tmp/volcomp-occ
journalctl -u volcomp-occupancy -f
```

## Surface predictions (`manifest-surfaces`, `volcomp surface-pack`)

The bucket also publishes 43 **surface predictions**
(`<Scroll>/representations/predictions/surfaces/<name>.zarr`): thresholded binary
masks (uint8, only 0 and 255) in blosc/zstd zarr v2, chunked 128³ / 192³ / 256³.
They are exported as a second kind of unit, in their own shape:

- **What is read.** Only level `0` of each prediction; its own pyramid is a binary
  copy and never becomes output data — ours is built from the mask itself. The
  coarsest published level is still used, as an occupancy oracle, but only where
  it is provably an exact max pool (see below).
- **The mask, at half the rung's resolution** (`encoding: "mask"`, the default).
  Each 128³ output chunk holds the published binary mask; what is *stored* is its
  2×2×2 majority pool — a 64³ grid — coded exactly by `volcomp.h`'s mask mode
  (spec §11), and what a reader gets back from `volcomp_decode` is that grid
  **trilinearly interpolated** to 128³: a continuous `u8` field, 0 or 255 exactly
  at every stored block centre and a two-voxel ramp across the boundary. It is the
  training target directly, at **0.0057 bits per voxel** on the PHercParis4 recto —
  30× less than the ramp below at q 8 and 11× less than packbits + zstd-19 of the
  mask. There is no quantiser on a mask level: `q` is recorded as 0 everywhere.
  Air far from any mask is 0, so those chunks (and shards) are simply absent.
  The resample onto the ladder is trilinear **on the 0/1 mask, thresholded at 0.5**.
- **Every rung is the majority pool of the one below** — which is exactly the rung
  below's own stored grid, so the pyramid is built by reading grids, not by pooling
  pixels, and nothing is lost on the way up. A thin sheet does eventually vanish
  under repeated majority pooling; those top rungs are then simply absent.
- **The ramp** (`--encoding ramp`) is the older form, kept for re-exports that ask
  for it. The mask becomes a continuous *signed-distance ramp*: with `s`
  the signed Euclidean distance to the mask boundary in source voxels, positive
  inside, clipped to ±3, the stored value is `round(127.5 + 42.5 * s)` — inside
  1, 2, 3 voxels deep: **170, 213, 255**; outside: **85, 43, 0**; and the **128
  crossing lies exactly on the published edge**. Air far from any mask is 0, so
  those chunks (and shards) are simply absent, as in the CT export. The distance
  is exact, not an approximation: clipped at 3, the nearest opposite-class voxel
  has every offset within ±3, so three min-plus passes with a ±3 window over
  squared distances give the exact Euclidean value. Each output chunk is computed
  with a 3-voxel halo, so the result never depends on the tiling.
- **The exact ladder.** Every exported array sits on the power-of-two grid
  `rung k = 0.6 * 2^k um`, not on the source's own grid: a prediction native at
  9.362 or 8.640 or 9.596 um is resampled onto exactly 9.6 um, 2.215 onto 2.4,
  and even a 2.399 um scan is resampled onto exactly 2.400 (0.1 % is ~90 voxels
  of drift across PHercParis4). Output voxel `i` samples source coordinate
  `i * rung_um / native_um`, output shape is `round(shape * native_um / rung_um)`,
  and origins are aligned at voxel 0. The ramp is computed first, at the source
  resolution, and only then trilinearly resampled: the binary mask itself is never
  resampled. Where the source is already exactly on the rung the resample is the
  identity and is skipped. The native voxel size comes from the source volume the
  prediction names (its timestamp prefix) and the level it was run on (`-L2-`, or
  the level whose shape matches).
- **q per level** (ramp encoding only) is a function of the voxel size, not of the level index
  (`rung_q()` in `coordinator.py`, the one table for the whole export):

  | um | 0.6 | 1.2 | 2.4 | 4.8 | 9.6 | 19.2 | 38.4 and coarser |
  |---|---|---|---|---|---|---|---|
  | q | 32 | 16 | 8 | 4 | 2 | 1 | 0 (lossless) |

  So the PHercParis4 2.4 um recto prediction is q 8 / 4 / 2 / 1 and lossless above,
  and every m7 prediction on a 2.4 um scan (`-L2-`, i.e. 9.6 um natively) is q 2, then
  1 at 19.2 um, then lossless. The
  top rungs of a prediction fit in a single 128³ chunk and hold a handful of small
  values that a dead-zone quantiser would wipe out, so they are stored losslessly
  (`q = 0`); it costs a few hundred bytes.
- **Level names are the voxel size.** The arrays are `9.6/`, `19.2/`, `38.4/` ...
  up to `1228.8/` (the whole scroll in one chunk), not `0/ 1/ 2/`, and the group's
  OME `multiscales` states the exact voxel size in micrometres. Read the pyramid
  by iterating `datasets[].path`; never assume integer level names. (The CT export
  keeps integer names until its next re-export.)
- **One unit = one 1024³ shard at the native rung, and the three levels above
  it.** A unit writes whole shard files at all four levels (shard edge 1024 /
  512 / 256 / 128, always 128³ volcomp chunks inside), aligned to the same
  footprint, so nothing crosses units. Levels 1..3 are exact 2× mean pooling of
  the *exact* ramp, before it is quantised.
- **Levels above the fourth** are built offline by `coordinator.py pool-levels`,
  one 128³ shard at a time (`volcomp shard-pool`: for a mask level, the eight
  stored grids below assembled into the coarser chunk; for a ramp level, 2× mean
  pooling): streaming,
  resumable (an output shard that exists is left alone), and it can read the level
  below from a local tree or over HTTPS from the published one.

### Occupancy masks, and the pyramid that is not a max pool

The published prediction pyramids are cheap occupancy oracles **when they are exact
max pools**: a zero coarse voxel then guarantees a whole block of zeros at level 0,
so a unit's 512 output chunks can be masked before any worker touches them, exactly
as the CT `occupancy` pass does it. `manifest-surfaces` builds those masks per
prediction: it downloads the coarsest published level, reduces it separably to one
bit per 128³ output chunk (the chunk's resampled source extent, the ramp halo and
the interpolation voxel, dilated by one coarse voxel), stores the 512-bit mask and
`est` per unit, and marks `est = 0` units done on the spot.

**It verifies the pyramid first, and not every prediction passes.** `volcomp
surface-maxpool-check` compares a sampled chunk of each level with the eight chunks
below it; only a prediction whose whole chain is an exact max pool gets masks.

- PHercParis4 `recto-2um-ps256` (76 800 units, 69 % of the export): levels 5..0
  verified as exact max pools on 15 sampled chunks. The mask marks **50 509 of the
  76 800 units empty** (65.8 %, done without a worker round-trip) and leaves
  **11 424 267 of 39 321 600 chunks** occupied — 71 % of the chunks never touched.
  The whole pass costs ~2.5 minutes, most of it the verification downloads.
- The m7 predictions (the other 42): **not max pools**. Their coarse levels are
  downsampled probabilities re-thresholded, so they *lose* structure going up:
  level 1 of the PHercMANBp m7 has 279 598 voxels (1.7 % of a chunk) that are zero
  over a nonzero 2³ block, and a coarse voxel at level 5 reads zero where its
  level-0 block holds 11 781 nonzero voxels. Using them as an oracle silently drops
  published data — it dropped exactly one 128³ chunk in a test unit before the
  check was added. They are exported with every chunk occupied; the exact skips
  below cost them nothing.

Use `--no-occupancy` to skip the pass entirely, `--occupancy-samples N` to widen the
verification.

### The hot path is C

`volcomp surface-pack` does everything numeric in one invocation per unit, with
one thread per output chunk:

```sh
volcomp surface-pack SRCDIR OUTDIR --csize=192 --src-shape=4287,3145,3145 \
    --out-shape=4285,3144,3144 --shard=2,1,1 --scale=1.0004168403418091 \
    --mask [--dmax=3] [--threads=0] [--samples=8]      # or --q=2,1,1,1 for the ramp
# ok src_chunks=252 src_nonzero=1 out_voxels=1073741824 compared=11 psnr_min=44.17
#    max_err=66 t_decode=0.33 t_ramp=4.91 t_encode=0.25 vox_per_s=2.18e+08 ...
#    present0=489 bytes0=19157548 present1=64 ... bytes=27765986
```

`--mask` selects the mask encoding (no argument, no `--q`); `--occupancy=HEX` is
the unit's 512-bit occupancy mask (128 hex digits) when the prediction has one.
(It was spelled `--mask=HEX` before the mask encoding existed.) `SRCDIR` holds the raw source chunks the worker downloaded, named
`<cz>_<cy>_<cx>.blosc` by their absolute source chunk index (a missing file is an
absent, all-zero chunk); `OUTDIR` gets `0.shard` .. `3.shard`. The tool decodes
the blosc1/zstd containers itself (`tools/cli/blosc1.h`, ~80 lines against
libzstd: the whole format is a 16-byte header, a block offset table and one zstd
frame per block; byte unshuffle is implemented but is the identity at typesize 1),
re-tiles source chunks of any size onto the 128³ output grid, resamples, pools,
encodes, and verifies — every stored chunk is decoded again, and for a sampled
subset the mask encoding checks its own invariant (every stored block centre comes
back as the 2×2×2 majority of the source, 0 or 255 exactly) while the ramp encoding
compares against the ramp it came from (`psnr_min`, `max_err`, reported back to the
coordinator). A unit whose source is nonzero but whose level-0 shard came out empty
fails.

The Python worker only orchestrates: claim, list + download the source chunks
covering its footprint plus the halo (threads, standard library), run the C tool,
upload one shard per level, report. **The VMs stay standard-library only** — no
numpy, no scipy; `bootstrap.sh` adds `libzstd-dev` for the build.

**What a unit costs.** The transform runs one thread per output chunk and skips
what it can prove is uniform:

- an output chunk the occupancy mask clears is never touched (`skip_mask`);
- a chunk whose haloed source box meets only all-air source chunks is absent, and
  one that meets only all-mask chunks is the constant 255 — no distance transform,
  no resample (`skip_zero`, `skip_full`). Source chunks are classified as air /
  mask / mixed as they are decoded, so this costs nothing to find out. **A sparse
  unit now costs what its data costs**, not what its footprint costs;
- the coarse levels only pool and encode the 128³ blocks whose level-0 chunk
  produced data.

The rest is the ramp itself, which walks its sub-box **once**: for each z plane it
builds both seed planes straight out of the assembled region, runs the x and y
min-plus passes into two 7-plane rings (~300 KB per thread, L2 resident) and, as
soon as a plane has its full ±3 neighbourhood, runs the z pass and writes that ramp
plane. Both min-plus passes, the table lookup, the trilinear resample and the 2×
pooling have AVX2 kernels (32 uint8 lanes for the distance passes, four output
voxels at a time for the resample, `maddubs` for the pooling) with the scalar
kernels kept and selected at runtime, exactly as `volcomp.h` does it. The AVX2 and
scalar kernels are byte-identical, and so is the whole optimised pipeline against
the straightforward one: `tests/test_surface.c` pins 16 golden hashes of a dense and
a sparse unit (four levels each, lossless and at the real quantisers), generated
from the build before any of this.

Measured on a 24-thread laptop, one 1024³ unit of the PHercMANBp m7 prediction
(9.6 µm, 192³ source chunks) straight from S3, best of three:

| | ramp + resample + pool + level-0 encode | |
|---|---|---|
| dense unit (203 source chunks, 489 chunks out) | before | after |
| 1 thread | 37.6 s | **7.0 s** |
| 4 threads (the VM class) | 10.7 s | **2.0 s** |
| 24 threads | 4.9 s | **1.3 s** |
| sparse unit (2 source chunks of 216, 6 chunks out) | | |
| 4 threads, whole unit wall clock | 9.5 s | **0.07 s** |

That stage still contains the level-0 encode and verify (2.6 CPU-seconds, untouched
by any of this); the transform alone went from 40.2 to 6.2 CPU-seconds on the dense
unit, 6.4×. Peak RSS is 1.27 GB dense, 58 MB sparse. The rest of the unit: 1.4 s to
download its 203 stored source chunks (18.6 MB), 0.7 s to blosc-decode them, 0.3 s
to encode levels 1..3. Output for that unit **in the ramp encoding**: 19.2 MB at
9.6 µm (q 2), 6.7 MB at 19.2, 1.6 MB at 38.4, 0.34 MB at 76.8 — 27.8 MB.

### What the mask encoding costs (measured, S3, whole units)

The same PHercMANBp unit [2, 1, 1], `--mask`, 24 threads: 203 of 252 source chunks
downloaded in 2.2 s, packed in 1.2 s (0.7 s of transform), **502 chunks out**:

| level | 9.6 µm | 19.2 | 38.4 | 76.8 | total |
|---|---|---|---|---|---|
| mask | 367 kB | 87 kB | 22.8 kB | 5.3 kB | **482 kB** |
| ramp (q 2/1/1/1) | 19.2 MB | 6.7 MB | 1.6 MB | 0.34 MB | 27.8 MB |
| ratio | 52× | 77× | 70× | 64× | **58×** |

Four whole units of the **PHercParis4 recto** (2.4 µm, 256³ source chunks), sampled
across the occupancy distribution — 1704 occupied chunks in total, 1.4 s to 2.0 s of
download and 0.3 s to 1.5 s of packing each:

| level | 2.4 µm | 4.8 | 9.6 | 19.2 |
|---|---|---|---|---|
| bytes per occupied chunk | 1285 | 286 | 68.5 | 16.6 |

The manifest counts **11 424 267 occupied chunks** over the recto's 26 291 non-empty
units, so the four levels the fleet writes come to **14.7 + 3.3 + 0.8 + 0.2 = 19 GB**
— against the ~700 GB the ramp at q 8 would have cost for level 0 alone.

### Run it

```sh
sudo volcomp-coordinator manifest-surfaces --db /var/lib/volcomp/export.db \
     --volcomp /usr/local/bin/volcomp --tmp /var/tmp/volcomp-occ    # + the occupancy masks
sudo volcomp-coordinator metadata --db /var/lib/volcomp/export.db --out /var/lib/volcomp/meta
volcomp-worker upload-tree /var/lib/volcomp/meta --sftp ... --netrc ...
tools/export/fleet.py bootstrap --role worker --kind surface ...     # PARALLEL=1, all cores in C
# afterwards, on any machine with the tree (or over HTTPS):
volcomp-coordinator pool-levels --db export.db --volcomp volcomp --src /tree --out /tree
```

A surface unit lists the source rows it needs (`<pred>0/<cz>/<cy>/`) before
downloading, so absent chunks cost no GETs and a 404 can never be mistaken for a
transient failure; with an occupancy mask it only lists and fetches the chunks an
occupied output chunk actually reads.

Local dry run, exactly as for CT but with `manifest-surfaces`; the offline
end-to-end test does precisely this against a synthetic bucket:

```sh
python3 -m pytest tools/export/test_export.py     # both encodings; needs numpy/scipy/numcodecs, no network
ctest --preset release -R 'test_surface|test_mask'  # ramp values, tiling, pooling, blosc, mask chunks
```

### Reading a prediction back

```python
import zarr, volcomp_zarr
g = zarr.open_group("volcomp/PHercMANBp/.../<name>.zarr", mode="r")
ms = g.attrs["ome"]["multiscales"][0]
paths = [d["path"] for d in ms["datasets"]]          # ["9.6", "19.2", ...]: never assume ints
a = zarr.open(f"volcomp/.../<name>.zarr/{paths[0]}", mode="r")
block = a[2048:2176, 1024:1152, 1024:1152]           # mask: 0 = air, 255 = inside, 128 = the edge
```

`g.attrs["volcomp"]` records the source key, the encoding and its formula, the
threshold, the native and rung voxel sizes, the resample scale, both shapes, and
the q and shard edge of every level.

## Verification per shard (`volcomp shard-verify`)

- index size and CRC-32C parse;
- every present chunk decodes without error;
- N sampled chunks (default 8) are compared with their source: PSNR and max
  error are reported back to the coordinator (`psnr_min`, `max_err` per unit);
- a source chunk that is nonzero but missing from the shard fails the unit;
- a download failure that is not a clean 404 is retried and then fails the unit
  — a transient error can never be mistaken for a masked (absent) chunk.

## Run it

```sh
# 0. one-time: SFTP credentials for the workers (kept out of the repo)
printf 'machine dl.ash2txt.org login forrest password ...\n' > ~/.volcomp-netrc && chmod 600 ~/.volcomp-netrc

# 1. VMs (Blue Lobster; API key in ~/bluelobster.txt)
tools/export/fleet.py launch --role coordinator
tools/export/fleet.py launch --role worker --count 40
tools/export/fleet.py wait
tools/export/fleet.py bootstrap --commit <sha> --sftp sftp://dl.ash2txt.org:9238/volcomp --netrc ~/.volcomp-netrc

# 2. queue + metadata (on the coordinator)
tools/export/fleet.py ssh volcomp-coordinator
  sudo volcomp-coordinator manifest --db /var/lib/volcomp/export.db           # all 64 volumes, all levels
  sudo volcomp-coordinator occupancy --db /var/lib/volcomp/export.db --volcomp /usr/local/bin/volcomp --tmp /var/tmp/volcomp-occ  # see below
  sudo volcomp-coordinator metadata --db /var/lib/volcomp/export.db --out /var/lib/volcomp/meta
  volcomp-worker upload-tree /var/lib/volcomp/meta --sftp sftp://dl.ash2txt.org:9238/volcomp --netrc ~/.volcomp-netrc
  volcomp-coordinator report --db /var/lib/volcomp/export.db                  # or curl :8765/status

# 3. watch; add or remove workers at any time
tools/export/fleet.py launch --role worker --count 20 && tools/export/fleet.py wait && tools/export/fleet.py bootstrap --role worker ...
tools/export/fleet.py destroy --role worker
```

Restrict the manifest while testing: `manifest --volume PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr --levels 3,4,5`.

Local dry run without any VM or upload (what `tests` of this directory amount to):

```sh
python3 tools/export/coordinator.py manifest --db /tmp/e.db --volume <volume-prefix> --levels 3,4,5
python3 tools/export/coordinator.py serve --db /tmp/e.db --bind 127.0.0.1 &
python3 tools/export/worker.py run --coordinator http://127.0.0.1:8765 --volcomp build/release/volcomp \
        --tmp /tmp/vc --local-out /tmp/vc-out --exit-when-idle
build/release/volcomp shard-verify /tmp/vc-out/<...>/3/c/0/0/0 <tmpdir-with-source-chunks>
```

## Sizing

Per worker the encode is ~1 s of one core per shard; a 4-vCPU VM is bound by its
S3 download rate (1 GiB per shard). With `--parallel 4` and 32 connections per
shard a VM keeps ~128 GETs in flight. Total output at q8 is roughly 5–10 TB, so
the SFTP server's inbound bandwidth is the other ceiling: at 100 MB/s that is
1–2 days regardless of worker count. Coordinator load is negligible (a few
requests per second at hundreds of workers).

## Reading the result

`python/volcomp_zarr` registers the `volcomp` codec with zarr ≥ 3
(`import volcomp_zarr` before `zarr.open`). The group `zarr.json` carries
OME-Zarr 0.5 multiscales copied from the source `.zattrs`; array shapes are the
true source shapes (edge shards are partial; chunks beyond the shape or in
masked air are "missing" index entries and read back as the fill value 0;
shards with no data at all are not written — a missing shard key is the fill
value in zarr v3).
