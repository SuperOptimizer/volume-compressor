# Bulk export: S3 (raw zarr v2) → SFTP (zarr v3 + volcomp)

Re-encodes every uncompressed volume of the Vesuvius Challenge open-data bucket
(`vesuvius-challenge-open-data`, 64 volumes, ~760 TB logical at level 0 plus the
LOD pyramids) into zarr v3 `sharding_indexed` arrays — 1024³ shards of 128³
`volcomp` chunks (q = 8 at native resolution, 4 / 2 / 1 at LOD levels 1 / 2 / ≥3: downscaling averages noise away, so coarser levels keep more) — under `sftp://dl.ash2txt.org:9238/volcomp/`, mirroring the
bucket keys: `volcomp/<scroll>/volumes/<volume>.zarr/<level>/c/<sz>/<sy>/<sx>`.

Everything is Python 3 standard library + the `volcomp` CLI + OpenSSH `sftp` (+`sshpass`); nothing to
pip-install on the VMs.

## Pieces

| file | runs on | does |
|---|---|---|
| `coordinator.py` | coordinator VM | sqlite queue of shard units; HTTP claim/done/fail/status; writes the zarr v3 metadata tree |
| `worker.py` | every compute VM | claim → download 512 source chunks (threads) → `volcomp shard-pack` → `volcomp shard-verify` → SFTP upload (`.part` + rename) → done |
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
