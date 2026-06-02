#!/usr/bin/env python3
"""Fetch real PHerc Paris 4 scroll data from the PUBLIC Vesuvius Challenge open
data bucket (no credentials; --no-sign-request, us-east-1) for the Phase-0
benchmark (PLAN §3 item 5).

Two modes, both implemented:

  (a) coarse  -- pull a sub-region of the WHOLE coarse volume
        20260310170716-45.532um-11.0m-74keV-masked.zarr  (level 0, 4071x2264x2264,
        u8, 128^3 chunks, dimension_separator "/", compressor null = raw).
        Reassembles the requested chunk-grid sub-region into one raw u8 volume on
        local disk. The full volume is ~21 GB -- by default we pull a small
        centered sub-region, NOT the whole thing.

  (b) chunks  -- pull individual verified-present high-res CENTER 128^3 chunks
        20260411134726-2.400um-0.2m-78keV-masked.zarr  (level 0), coords
        0/{296,297}/{127,128}/{127,128}. Each chunk file is raw u8, 128^3 =
        2097152 bytes, no header. Writes each chunk as a *.u8 file.

Everything is cached under the gitignored data/ dir.

Examples:
  python3 fetch_corpus.py chunks
  python3 fetch_corpus.py coarse --cz 15 --cy 8 --cx 8 --nz 2 --ny 2 --nx 2
"""
import argparse, os, subprocess, sys, pathlib
import numpy as np

BUCKET = "s3://vesuvius-challenge-open-data"
COARSE = f"{BUCKET}/PHercParis4/volumes/20260310170716-45.532um-11.0m-74keV-masked.zarr/0"
HIRES  = f"{BUCKET}/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"
CHUNK  = 128
COARSE_SHAPE = (4071, 2264, 2264)

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def s3_cp(src, dst):
    """Copy one S3 object to dst. Returns True if present, False if 404
    (absent zarr chunk = all-zero fill)."""
    r = subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", "--region", "us-east-1", src, str(dst)],
        capture_output=True, text=True)
    return r.returncode == 0


def fetch_chunk(base, cz, cy, cx, cache):
    """Fetch one 128^3 chunk (base/cz/cy/cx). Returns a 128^3 u8 ndarray
    (zeros if the chunk is absent in zarr)."""
    local = cache / f"{cz}_{cy}_{cx}.u8"
    if not (local.exists() and local.stat().st_size == CHUNK**3):
        if not s3_cp(f"{base}/{cz}/{cy}/{cx}", local):
            np.zeros(CHUNK**3, dtype=np.uint8).tofile(local)
    return np.fromfile(local, dtype=np.uint8).reshape(CHUNK, CHUNK, CHUNK)


def mode_chunks(args):
    out = DATA / "hires_chunks"
    cache = DATA / "cache_hires"
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    coords = [(cz, cy, cx)
              for cz in (296, 297) for cy in (127, 128) for cx in (127, 128)]
    print(f"Fetching {len(coords)} high-res center 128^3 chunks...")
    for (cz, cy, cx) in coords:
        arr = fetch_chunk(HIRES, cz, cy, cx, cache)
        name = out / f"hr_{cz}_{cy}_{cx}.u8"
        arr.tofile(name)
        print(f"  {name.name}  nonzero={ (arr!=0).mean():5.1%}  mean={arr.mean():5.1f}")
    print(f"-> {out}")


def mode_coarse(args):
    out_dir = DATA / "coarse"
    cache = DATA / "cache_coarse"
    out_dir.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    cz0, cy0, cx0 = args.cz, args.cy, args.cx
    nz, ny, nx = args.nz, args.ny, args.nx
    print(f"Reassembling coarse sub-region: chunk-grid origin "
          f"({cz0},{cy0},{cx0}) size {nz}x{ny}x{nx} ({nz*ny*nx} chunks, "
          f"{nz*ny*nx*CHUNK**3/1e6:.0f} MB)...")
    vol = np.zeros((nz*CHUNK, ny*CHUNK, nx*CHUNK), dtype=np.uint8)
    for dz in range(nz):
        for dy in range(ny):
            for dx in range(nx):
                arr = fetch_chunk(COARSE, cz0+dz, cy0+dy, cx0+dx, cache)
                vol[dz*CHUNK:(dz+1)*CHUNK, dy*CHUNK:(dy+1)*CHUNK,
                    dx*CHUNK:(dx+1)*CHUNK] = arr
        print(f"  z-plane {dz+1}/{nz} done")
    name = out_dir / f"coarse_z{cz0}_y{cy0}_x{cx0}_{nz}x{ny}x{nx}.u8"
    vol.tofile(name)
    print(f"-> {name}  shape={vol.shape}  nonzero={(vol!=0).mean():5.1%}  "
          f"mean={vol.mean():5.1f}")
    print(f"   bench {name} {vol.shape[0]} {vol.shape[1]} {vol.shape[2]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("chunks", help="individual high-res center 128^3 chunks")
    c = sub.add_parser("coarse", help="reassemble a coarse-volume sub-region")
    # Default: a small centered region of the coarse volume.
    c.add_argument("--cz", type=int, default=COARSE_SHAPE[0]//CHUNK//2)
    c.add_argument("--cy", type=int, default=COARSE_SHAPE[1]//CHUNK//2)
    c.add_argument("--cx", type=int, default=COARSE_SHAPE[2]//CHUNK//2)
    c.add_argument("--nz", type=int, default=2)
    c.add_argument("--ny", type=int, default=2)
    c.add_argument("--nx", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "chunks":
        mode_chunks(args)
    elif args.mode == "coarse":
        mode_coarse(args)


if __name__ == "__main__":
    main()
