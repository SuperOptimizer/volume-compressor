#!/usr/bin/env python3
"""volcomp export coordinator (Python 3 standard library only).

One process on the coordinator VM owns a sqlite queue of work units; each unit
is one 1024^3 zarr v3 shard of one multiscale level of one source volume.
Compute VMs run tools/export/worker.py against the HTTP API below.

  coordinator.py manifest  --db export.db [--volume PREFIX ...] [--levels 0,1,...]
      Enumerate the public S3 bucket, read every uncompressed volume's zarr v2
      metadata and insert all (volume, level, shard) units. Re-runnable: units
      already present are left alone (progress is never lost).
  coordinator.py manifest-surfaces --db export.db [--volume PREFIX ...]
                                   [--encoding mask|mask-lossless|ramp] [--no-resample]
      The same for the published surface predictions
      (<Scroll>/representations/predictions/surfaces/*.zarr): resolve each one's
      native voxel size, snap it to the ladder rung, and insert one unit per
      1024^3 output shard. A unit emits the four finest levels at once.
      --no-resample (the default for --encoding mask-lossless) keeps the source
      grid exactly instead and names the levels by their true voxel size.
  coordinator.py pool-levels --db export.db --volcomp PATH --src DIR|URL --out DIR
      Build the coarse levels (everything above the four a unit writes) offline
      by 2x mean pooling (2x2x2 majority, for a mask level), one 128^3 shard at a
      time. Streaming and resumable:
      shards that already exist under --out are skipped.
  coordinator.py metadata  --db export.db --out DIR [--q 8]
      Write the zarr v3 group/array metadata (OME-Zarr 0.5 multiscales,
      sharding_indexed 1024^3 -> 128^3, codec "volcomp" with q = 8 at level 0,
      4 / 2 / 1 at levels 1 / 2 / >=3) into DIR mirroring
      the bucket keys; upload DIR to the SFTP destination with worker.py's
      `upload-tree` (or any sftp client).
  coordinator.py serve     --db export.db [--bind 0.0.0.0] [--port 8765] [--lease 900]
      HTTP API:  POST /claim  {"worker": id}            -> unit JSON or 204
                 POST /done   {"id", "worker", "bytes", "present", "psnr_min", "max_err"}
                 POST /fail   {"id", "worker", "error"} -> unit goes back to the queue
                 GET  /status                            -> totals and per-volume progress
  coordinator.py report    --db export.db
      Print progress.
  coordinator.py requeue   --db export.db
      Return leased/failed units to the queue (e.g. after fixing a worker bug).
  coordinator.py occupancy --db export.db --volcomp /usr/local/bin/volcomp --tmp DIR [--volume PREFIX ...]
      Per volume: download the coarsest level (<= 5; a level-5 voxel spans 32^3 native
      voxels), run `volcomp occupancy` on it once per level and store, for every unit, a
      512-bit chunk mask (dilated by one coarse voxel) and `est`, the number of chunks
      that can hold data. Workers only fetch masked-in chunks; units with est = 0 are
      marked done without any transfer (a missing shard is the fill value). Safe to run
      while serving; rerun with --force to recompute.

The bucket is read anonymously over HTTPS; nothing here needs AWS credentials.
"""
import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# the open-data bucket over plain HTTPS; VOLCOMP_S3_ENDPOINT points the whole
# pipeline at another S3-style endpoint (the offline end-to-end test serves one)
BUCKET = os.environ.get("VOLCOMP_S3_ENDPOINT", "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com")
SHARD = 1024
CHUNK = 128
Q_NATIVE = 8.0

# ----------------------------------------------------------------- the resolution ladder
#
# THE quality schedule of the export, in one place: q is a function of the
# PHYSICAL voxel size of a level, not of its index. The ladder is the power of
# two grid rung k = 0.6 * 2^k um, and every exported array lives exactly on it:
#
#   rung k    0      1      2      3      4      5      6 ...
#   um        0.6    1.2    2.4    4.8    9.6    19.2   38.4 ...
#   q         32     16     8      4      2      1      0 (lossless)
#
# so a 2.4 um array is q 8 and each 2x downscale halves q down to 1 at 19.2 um; every
# coarser rung is stored lossless (on ramp data q 1 saves only ~20 % over lossless, and
# the dead zone would erase the small pooled values at the top of the ladder). The CT export
# still uses level_q() below; it is the same rule for a 2.4 um scan and can move
# over to rung_q() at its next re-export.
RUNG0_UM = 0.6
Q_RUNG0 = 32.0
LADDER_TOP = 11  # 1228.8 um: the whole scroll fits one 128^3 chunk at this rung


def rung_um(k):
    """Exact voxel size of rung k, in micrometres."""
    return round(RUNG0_UM * 2 ** k, 4)


def rung_name(k):
    """Directory name of rung k: "0.6", "1.2", ... "1228.8" (exact spellings)."""
    return f"{rung_um(k):.1f}"


def native_level_name(um):
    """Directory name of a level on a NATIVE (unresampled) grid: its TRUE voxel size
    in micrometres, up to three decimals, no trailing zeros — 2.4 -> "2.4",
    9.596 -> "9.596", 19.192 -> "19.192", 9.362 -> "9.362". A 2.400 um scan
    therefore keeps the same spellings the ladder gave it."""
    t = f"{um:.3f}".rstrip("0").rstrip(".")
    return t or "0"


def rung_of(um):
    """The rung a measured voxel size snaps to (nearest in log2: 9.362 and 8.640 -> 9.6)."""
    return min(range(LADDER_TOP + 1), key=lambda k: abs(math.log2(um / rung_um(k))))


LOSSLESS_FROM = 6  # 38.4 um and coarser: q 0


def rung_q(k):
    """Quantiser for an array whose voxels are rung k: 32 / 2^k down to 1 at 19.2 um, 0 (lossless) above."""
    return 0.0 if k >= LOSSLESS_FROM else max(1.0, Q_RUNG0 / 2 ** k)


def level_q(level, q0=Q_NATIVE):
    """Quantiser per multiscale level: q0 at native resolution, halved per 2x downscale,
    floor 1 (downscaled levels average away noise, so they carry more signal per voxel)."""
    return max(1.0, q0 / (2 ** level))
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# ----------------------------------------------------------------------------- S3 listing


def http_get(url, retries=6, timeout=60):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code < 500 and e.code != 429:
                raise
        except (urllib.error.URLError, socket.timeout, ConnectionError):
            pass
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError("GET failed after retries: " + url)


def s3_prefixes(prefix):
    """Immediate sub-prefixes ("directories") under prefix."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "delimiter": "/", "max-keys": "1000", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(http_get(BUCKET + "/?" + urllib.parse.urlencode(q)))
        for cp in root.iter(S3NS + "CommonPrefixes"):
            p = cp.find(S3NS + "Prefix").text
            if p != prefix:
                out.append(p)
        if root.find(S3NS + "IsTruncated").text != "true":
            return out
        token = root.find(S3NS + "NextContinuationToken").text


def read_json(key):
    b = http_get(BUCKET + "/" + urllib.parse.quote(key))
    return None if b is None else json.loads(b)


# ----------------------------------------------------------------------------- database

SCHEMA = """
CREATE TABLE IF NOT EXISTS volume (
  name TEXT PRIMARY KEY,              -- e.g. PHerc0009B/volumes/2025...-masked.zarr
  zattrs TEXT NOT NULL,               -- source .zattrs (multiscales), verbatim
  levels TEXT NOT NULL,               -- JSON: {level: {"shape": [z,y,x]}}
  kind TEXT NOT NULL DEFAULT 'ct',    -- 'ct' | 'surface'
  info TEXT                           -- surface: JSON (ladder, scale, csize, per-level paths)
);
CREATE TABLE IF NOT EXISTS unit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  volume TEXT NOT NULL, level INTEGER NOT NULL,
  sz INTEGER NOT NULL, sy INTEGER NOT NULL, sx INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'todo', -- todo | leased | done
  worker TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
  bytes INTEGER, present INTEGER, psnr_min REAL, max_err INTEGER, done_at REAL, error TEXT,
  est INTEGER, mask BLOB,             -- occupancy pass: chunks that may hold data, 512-bit mask (see occupancy)
  UNIQUE(volume, level, sz, sy, sx)
);
CREATE INDEX IF NOT EXISTS unit_state ON unit(state, lease_until);
CREATE INDEX IF NOT EXISTS unit_volume ON unit(volume, state);
"""


def open_db(path):
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA busy_timeout=30000")
    db.executescript(SCHEMA)
    have = {r[1] for r in db.execute("PRAGMA table_info(unit)")}
    for col, typ in (("est", "INTEGER"), ("mask", "BLOB")):  # databases created before the occupancy pass
        if col not in have:
            db.execute(f"ALTER TABLE unit ADD COLUMN {col} {typ}")
    have_v = {r[1] for r in db.execute("PRAGMA table_info(volume)")}
    for col, typ, dflt in (("occ_level", "INTEGER", ""), ("kind", "TEXT", " NOT NULL DEFAULT 'ct'"),
                           ("info", "TEXT", "")):
        if col not in have_v:
            db.execute(f"ALTER TABLE volume ADD COLUMN {col} {typ}{dflt}")
    return db


# ----------------------------------------------------------------------------- manifest


def shard_grid(shape):
    return [math.ceil(n / SHARD) for n in shape]


def cmd_manifest(a):
    db = open_db(a.db)
    if a.volume:
        vols = [v if v.endswith("/") else v + "/" for v in a.volume]
    else:
        vols = []
        for scroll in s3_prefixes(""):
            vols += s3_prefixes(scroll + "volumes/")
    want_levels = None if a.levels == "all" else {int(x) for x in a.levels.split(",")}
    n_new = 0
    for v in vols:
        za0 = read_json(v + "0/.zarray")
        if za0 is None:
            print(f"skip {v}: no zarr v2 level 0", file=sys.stderr)
            continue
        if za0.get("compressor") is not None and not a.include_compressed:
            print(f"skip {v}: already compressed ({za0['compressor'].get('id')})", file=sys.stderr)
            continue
        if za0["dtype"] != "|u1" or za0["chunks"] != [CHUNK] * 3 or za0.get("dimension_separator", ".") != "/":
            print(f"skip {v}: unsupported layout {za0}", file=sys.stderr)
            continue
        zattrs = read_json(v + ".zattrs") or {}
        levels = {}
        for lp in sorted(s3_prefixes(v)):
            lvl = lp[len(v):-1]
            if not lvl.isdigit():
                continue
            za = read_json(lp + ".zarray")
            if za is None:
                continue
            levels[int(lvl)] = {"shape": za["shape"]}
        db.execute("INSERT OR REPLACE INTO volume(name, zattrs, levels) VALUES (?,?,?)",
                   (v, json.dumps(zattrs), json.dumps(levels)))
        rows = []
        for lvl, info in levels.items():
            if want_levels is not None and lvl not in want_levels:
                continue
            gz, gy, gx = shard_grid(info["shape"])
            rows += [(v, lvl, z, y, x) for z in range(gz) for y in range(gy) for x in range(gx)]
        cur = db.executemany("INSERT OR IGNORE INTO unit(volume, level, sz, sy, sx) VALUES (?,?,?,?,?)", rows)
        n_new += cur.rowcount if cur.rowcount > 0 else 0
        print(f"{v}: levels {sorted(levels)} shape0 {levels.get(0, {}).get('shape')} units {len(rows)}")
    total = db.execute("SELECT COUNT(*) FROM unit").fetchone()[0]
    print(f"manifest: {n_new} new units, {total} total")


# ------------------------------------------------------------------- surface predictions

SURF_PREFIX = "representations/predictions/surfaces/"
SURF_LEVELS = 4  # levels one unit writes: the native rung and three above it
SURF_DMAX = 3    # ramp clip, in source voxels


def surface_source_volume(scroll, pred_name):
    """The CT volume a prediction was made from: its key starts with the same
    14-digit timestamp. Returns (volume prefix, voxel size in um) or (None, None)."""
    stamp = pred_name[:14]
    if not stamp.isdigit():
        return None, None
    for v in s3_prefixes(scroll + "volumes/"):
        base = v.rstrip("/").split("/")[-1]
        if base.startswith(stamp):
            m = re.search(r"-(\d+\.\d+)um", base)
            if m:
                return v, float(m.group(1))
    return None, None


def surface_native_level(vol, pred_shape, pred_name):
    """Which multiscale level of the source volume the prediction was computed on:
    the level whose shape matches, else the -L<k>- in the prediction name."""
    for k in range(6):
        za = read_json(f"{vol}{k}/.zarray")
        if za is not None and list(za["shape"]) == list(pred_shape):
            return k
    m = re.search(r"-L(\d)-", pred_name)
    return int(m.group(1)) if m else None


SURF_ENCODINGS = ("mask", "mask-lossless", "ramp")


def surface_encoding(name):
    """The per-level storage of a prediction export, for the group attributes.

    "mask" is the 2x form the Paris 4 tree was written with: the published binary
    mask, stored as volcomp MASK chunks (a 2x2x2 majority pool per chunk).
    "mask-lossless" stores the mask itself, voxel for voxel, with the same coder.
    "ramp" is the older signed-distance form, kept for re-exports that ask for it."""
    if name == "mask-lossless":
        return {
            "name": "surface-mask-lossless", "dmax": SURF_DMAX,
            "storage": "volcomp lossless mask chunk (format revision 3, mode 5): the binary mask "
                       "of this level at full resolution, coded exactly with the same 12-neighbour "
                       "context-model range coder the 2x mask mode uses on its pool",
            "decode": "the mask itself, as u8 0 or 255, voxel for voxel; nothing is interpolated",
            "pyramid": "each level is the 2x2x2 majority pool (count*2 >= 8) of the level below, "
                       "voxels outside the array counting as air",
            "resample": "none: level 0 is the published grid exactly (scale 1)",
        }
    if name == "mask":
        return {
            "name": "surface-mask", "dmax": SURF_DMAX,
            "storage": "volcomp mask chunk: the binary mask at this rung, 2x2x2 majority-pooled "
                       "(count*2 >= 8) to a 64^3 grid per 128^3 chunk and stored exactly with a "
                       "12-neighbour context-model range coder",
            "decode": "the stored grid trilinearly interpolated back to 128^3 as u8 0..255 = "
                      "round(255 * v): output voxel j samples the grid at j/2 (the upper neighbour "
                      "clamped at the top edge), so a block centre is 0 or 255 exactly and the edge "
                      "carries a two-voxel ramp",
            "pyramid": "each rung is the 2x2x2 majority pool of the rung below, which is exactly "
                       "the rung below's own stored grid",
            "resample": "trilinear on the 0/1 mask, threshold 0.5, onto the exact ladder grid; "
                        "output voxel i samples source coordinate i * scale",
        }
    return {
        "name": "surface-ramp", "dmax": SURF_DMAX,
        "formula": "v = round(127.5 + (127.5/dmax) * clip(s, -dmax, dmax)), s = signed Euclidean "
                   "distance to the published mask boundary in source voxels, positive inside; "
                   "inside 1/2/3 voxels -> 170/213/255, outside -> 85/43/0, the 128 crossing lies "
                   "on the published edge",
        "resample": "trilinear, on the ramp (never on the mask), output voxel i samples source "
                    "coordinate i * scale",
    }


def surface_info(name, za0, zattrs, encoding="mask", resample=True):
    """Everything the workers and the metadata step need for one prediction.
    Raises ValueError if the array is not a published binary surface mask.

    With resample=False the export keeps the SOURCE grid exactly: scale 1, output
    shape = source shape, and the levels are named by their TRUE voxel size,
    native_um * 2^L (an m7 prediction read at L2 of a 2.399 um scan is natively
    9.596 um, so "9.596", "19.192", "38.384", ...). With resample=True (the
    historical behaviour) the array is resampled onto the 0.6 * 2^k ladder and the
    levels are named by the rung."""
    scroll = name.split("/")[0] + "/"
    base = name.rstrip("/").split("/")[-1]
    stem = base[:-5] if base.endswith(".zarr") else base
    if za0["dtype"] != "|u1" or za0.get("dimension_separator", ".") != "/":
        raise ValueError(f"unsupported layout {za0.get('dtype')} / {za0.get('dimension_separator')}")
    comp = za0.get("compressor") or {}
    if comp.get("id") != "blosc" or comp.get("cname") != "zstd":
        raise ValueError(f"unsupported compressor {comp}")
    if len(set(za0["chunks"])) != 1:
        raise ValueError(f"non-cubic chunks {za0['chunks']}")
    src_shape = [int(n) for n in za0["shape"]]
    vol, um = surface_source_volume(scroll, stem)
    if vol is None:
        raise ValueError("no source volume with that timestamp")
    k = surface_native_level(vol, src_shape, stem)
    if k is None:
        raise ValueError("cannot tell which level of the source volume this is")
    native_um = um * 2 ** k
    native_um = round(native_um, 6)
    if resample:
        rk = rung_of(native_um)
        scale = rung_um(rk) / native_um  # source voxels per output voxel
        out0 = [int(round(n / scale)) for n in src_shape]
        n_levels = LADDER_TOP - rk + 1
        level_um = lambda j: rung_um(rk + j)          # noqa: E731
        level_path = lambda j: rung_name(rk + j)      # noqa: E731
        level_rung = lambda j: rk + j                 # noqa: E731
        rung_size = rung_um(rk)
    else:
        # the source grid, untouched: no scale, no snapping, and the levels are named
        # by the voxel size they really have
        rk, scale, out0 = None, 1.0, list(src_shape)
        # as many levels as the ladder would have given this voxel size, so the two
        # encodings produce the same depth of pyramid for the same data
        n_levels = LADDER_TOP - rung_of(native_um) + 1
        level_um = lambda j: round(native_um * 2 ** j, 6)          # noqa: E731
        level_path = lambda j: native_level_name(native_um * 2 ** j)  # noqa: E731
        level_rung = lambda j: rung_of(native_um * 2 ** j)         # noqa: E731
        rung_size = native_um
    levels = []
    shape = out0
    for j in range(n_levels):
        # The ladder's q applies to every level the fleet writes and to every coarse level
        # bigger than a single chunk. The last rungs
        # of a prediction fit in one 128^3 chunk and hold a handful of small values, which a
        # dead-zone quantiser at q = 1 can wipe out entirely; there lossless costs a few
        # hundred bytes, so the top of the ladder is stored exactly.
        # a mask level has no quantiser at all: the mask is stored exactly (mask-lossless)
        # or at half the level's resolution and interpolated on the way out (mask), so q
        # is recorded as 0
        q = 0.0 if encoding in ("mask", "mask-lossless") else (
            0.0 if j >= SURF_LEVELS and max(shape) <= CHUNK else rung_q(level_rung(j)))
        levels.append({"path": level_path(j), "um": level_um(j), "shape": list(shape),
                       "q": q, "encoding": encoding,
                       "shard": SHARD >> j if j < SURF_LEVELS else CHUNK})
        shape = [math.ceil(n / 2) for n in shape]
    th = re.search(r"-th([0-9]*\.?[0-9]+)", stem)
    return {
        "source": BUCKET + "/" + name, "source_volume": vol, "source_level": k,
        "volume_um": um, "native_um": native_um, "rung": rk, "rung_um": rung_size,
        "resampled": bool(resample),
        "scale": scale, "csize": int(za0["chunks"][0]), "src_shape": src_shape, "out_shape": out0,
        "threshold": float(th.group(1)) if th else None, "levels": levels,
        "encoding": surface_encoding(encoding),
        "zattrs": zattrs,
    }


def surface_coarse_level(pred, info):
    """The coarsest published level of a prediction and how many level-0 voxels one of its
    voxels spans. The published pyramids are exact max pools, so a zero there is a
    guaranteed block of zeros at level 0."""
    best = None
    for k in range(9, -1, -1):
        za = read_json(f"{pred}{k}/.zarray")
        if za is not None:
            best = (k, za)
            break
    if best is None:
        return None
    k, za = best
    if len(set(za["chunks"])) != 1:
        return None
    return {"level": k, "factor": 2 ** k, "shape": [int(n) for n in za["shape"]], "csize": int(za["chunks"][0])}


def surface_maxpool_verified(pred, coarse, a):
    """Is this prediction's pyramid an exact max pool, all the way down to level 0? The
    occupancy mask is only sound if a zero coarse voxel guarantees a zero block at level 0,
    and that is NOT true of every published pyramid: the m7 models' levels are downsampled
    probabilities re-thresholded (level 1 of the PHercMANBp m7 has 1.7 % of its voxels zero
    over a nonzero 2^3 block), while the PHercParis4 recto's pyramid is an exact max pool at
    every level. So sample chunks of every level pair from the coarse level down to 1,
    compare each with the eight chunks below it, and only use the mask when all match."""
    k0, C = coarse["level"], coarse["csize"]
    if k0 < 1:
        return False
    work = tempfile.mkdtemp(prefix="surfchk-", dir=a.tmp)
    try:
        checked = 0
        for k in range(k0, 0, -1):
            za = read_json(f"{pred}{k}/.zarray")
            if za is None or za["chunks"][0] != C:
                print(f"  occupancy: level {k} is missing or not chunked {C}^3 -> no mask", flush=True)
                return False
            grid = [max(1, math.ceil(n / C)) for n in za["shape"]]
            coords = [(cz, cy, cx) for cz in range(grid[0]) for cy in range(grid[1]) for cx in range(grid[2])]
            n = max(1, min(a.occupancy_samples, len(coords)))
            picks = [coords[(i * len(coords)) // n] for i in range(n)]
            for c in picks:
                def get(level, cc, name):
                    data = http_get(BUCKET + "/" + urllib.parse.quote(f"{pred}{level}/{cc[0]}/{cc[1]}/{cc[2]}"))
                    if data is None:
                        return "-"
                    path = os.path.join(work, name)
                    with open(path, "wb") as f:
                        f.write(data)
                    return path

                coarse_file = get(k, c, "c.blosc")
                fine = [get(k - 1, (2 * c[0] + dz, 2 * c[1] + dy, 2 * c[2] + dx), f"f{dz}{dy}{dx}.blosc")
                        for dz in range(2) for dy in range(2) for dx in range(2)]
                if coarse_file == "-" and all(f == "-" for f in fine):
                    continue
                r = subprocess.run([a.volcomp, "surface-maxpool-check",
                                    coarse_file if coarse_file != "-" else "/dev/null", *fine, f"--csize={C}"],
                                   capture_output=True, text=True)
                if r.returncode:
                    print(f"  occupancy: max-pool check failed to run: {r.stderr[-300:]}", file=sys.stderr)
                    return False
                kv = dict(re.findall(r"(\w+)=(\d+)", r.stdout))
                if int(kv.get("misses", 1)):
                    print(f"  occupancy: level {k} chunk {c} is NOT the max pool of level {k - 1} "
                          f"({kv['misses']} voxels zero over a nonzero block) -> no mask for this prediction",
                          flush=True)
                    return False
                if int(kv.get("pooled_nonzero", 0)):
                    checked += 1
        if not checked:
            print("  occupancy: no sampled chunk held data; not trusting the pyramid", flush=True)
            return False
        print(f"  occupancy: levels {k0}..0 verified as exact max pools on {checked} sampled chunks", flush=True)
        return True
    finally:
        shutil.rmtree(work, ignore_errors=True)


def surface_occupancy(db, pred, info, a):
    """Per-unit 512-bit chunk masks from the coarsest published level, exactly as the CT
    occupancy pass does it: units with est = 0 are marked done on the spot (a missing shard
    key is the fill value) and the workers only fetch source chunks that an occupied chunk
    reads."""
    coarse = surface_coarse_level(pred, info)
    if coarse is None:
        print(f"{pred}: no usable coarse level; every chunk stays occupied", file=sys.stderr)
        return
    os.makedirs(a.tmp, exist_ok=True)
    if not surface_maxpool_verified(pred, coarse, a):
        return  # every chunk stays occupied; the exact skips inside surface-pack still apply
    work = tempfile.mkdtemp(prefix="surfocc-", dir=a.tmp)
    t0 = time.time()
    try:
        C = coarse["csize"]
        grid = [math.ceil(n / C) for n in coarse["shape"]]
        keys = [(cz, cy, cx) for cz in range(grid[0]) for cy in range(grid[1]) for cx in range(grid[2])]

        def fetch(c):
            data = http_get(BUCKET + "/" + urllib.parse.quote(f"{pred}{coarse['level']}/{c[0]}/{c[1]}/{c[2]}"))
            if data is None:
                return 0
            with open(os.path.join(work, f"{c[0]}_{c[1]}_{c[2]}.blosc"), "wb") as f:
                f.write(data)
            return 1

        with concurrent.futures.ThreadPoolExecutor(a.threads) as ex:
            got = sum(ex.map(fetch, keys))
        t1 = time.time()
        out = os.path.join(work, "masks.bin")
        r = subprocess.run([a.volcomp, "surface-occupancy", work, out, f"--csize={C}",
                            "--coarse-shape={},{},{}".format(*coarse["shape"]),
                            f"--factor={coarse['factor']}",
                            "--out-shape={},{},{}".format(*info["out_shape"]),
                            "--scale=%.17g" % info["scale"], f"--dmax={info['encoding']['dmax']}",
                            "--dilate=1"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"surface-occupancy failed for {pred}: {r.stderr[-500:]}")
        with open(out, "rb") as f:
            masks = f.read()
        sg = shard_grid(info["out_shape"])
        assert len(masks) == sg[0] * sg[1] * sg[2] * 64, (len(masks), sg)
        rows, empties, est_sum = [], [], 0
        for uid, sz, sy, sx in db.execute("SELECT id, sz, sy, sx FROM unit WHERE volume=?", (pred,)):
            i = ((sz * sg[1] + sy) * sg[2] + sx) * 64
            m = masks[i:i + 64]
            est = int.from_bytes(m, "little").bit_count()
            est_sum += est
            rows.append((est, m if est else None, uid))
            if est == 0:
                empties.append((uid,))
        db.execute("BEGIN")
        db.executemany("UPDATE unit SET est=?, mask=? WHERE id=?", rows)
        n_skip = db.executemany("UPDATE unit SET state='done', bytes=0, present=0, done_at=strftime('%s','now'), "
                                "error='occupancy:empty', worker=NULL, lease_until=NULL WHERE id=? "
                                "AND state IN ('todo','failed')", empties).rowcount
        db.execute("COMMIT")
        print(f"  occupancy: level {coarse['level']} {coarse['shape']} ({got}/{len(keys)} chunks, "
              f"{t1 - t0:.0f}s dl, {time.time() - t1:.0f}s scan): {est_sum}/{len(rows) * 512} chunks occupied, "
              f"{len(empties)}/{len(rows)} units empty ({n_skip} marked done)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def cmd_manifest_surfaces(a):
    db = open_db(a.db)
    # --no-resample / --resample; mask-lossless keeps the native grid unless told otherwise
    no_resample = a.resample is False if a.resample is not None else a.encoding == "mask-lossless"
    if a.volume:
        preds = [v if v.endswith("/") else v + "/" for v in a.volume]
    else:
        preds = []
        for scroll in s3_prefixes(""):
            if scroll.startswith("_"):
                continue
            preds += [p for p in s3_prefixes(scroll + SURF_PREFIX) if p.rstrip("/").endswith(".zarr")]
    n_new = 0
    for p in preds:
        za0 = read_json(p + "0/.zarray")
        if za0 is None:
            print(f"skip {p}: no level 0", file=sys.stderr)
            continue
        try:
            info = surface_info(p, za0, read_json(p + ".zattrs") or {}, a.encoding,
                                resample=not no_resample)
        except ValueError as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        levels = {lv["path"]: {"shape": lv["shape"]} for lv in info["levels"]}
        db.execute("INSERT OR REPLACE INTO volume(name, zattrs, levels, kind, info) VALUES (?,?,?,?,?)",
                   (p, json.dumps(info["zattrs"]), json.dumps(levels), "surface", json.dumps(info)))
        gz, gy, gx = shard_grid(info["out_shape"])
        rows = [(p, 0, z, y, x) for z in range(gz) for y in range(gy) for x in range(gx)]
        cur = db.executemany("INSERT OR IGNORE INTO unit(volume, level, sz, sy, sx) VALUES (?,?,?,?,?)", rows)
        n_new += max(0, cur.rowcount)
        grid = (f"rung {info['rung_um']}um (scale {info['scale']:.5f})" if info["resampled"]
                else "native grid, not resampled")
        print(f"{p}: native {info['native_um']:.4f}um -> {grid}, "
              f"{info['src_shape']} -> {info['out_shape']}, "
              f"chunks {info['csize']}, levels {[lv['path'] for lv in info['levels']]}, {len(rows)} units",
              flush=True)
        if not a.no_occupancy:
            os.makedirs(a.tmp, exist_ok=True)
            surface_occupancy(db, p, info, a)
    total = db.execute("SELECT COUNT(*) FROM unit").fetchone()[0]
    print(f"manifest-surfaces: {n_new} new units, {total} total")


# ----------------------------------------------------------------------------- metadata


def array_metadata(shape, q):
    """zarr v3 array metadata for one level; q is that level's quantiser."""
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": "uint8",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [SHARD, SHARD, SHARD]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [CHUNK, CHUNK, CHUNK],
                "codecs": [{"name": "volcomp", "configuration": {"q": q}}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                "index_location": "end",
            },
        }],
        "attributes": {},
        "dimension_names": ["z", "y", "x"],
    }


def group_metadata(volume_name, zattrs, levels):
    """OME-Zarr 0.5 multiscales carried over from the source .zattrs (0.4)."""
    src = (zattrs.get("multiscales") or [{}])[0]
    datasets = []
    for lvl in sorted(levels):
        ds = next((d for d in src.get("datasets", []) if d.get("path") == str(lvl)), None)
        ct = ds["coordinateTransformations"] if ds else [{"type": "scale", "scale": [float(2 ** lvl)] * 3}]
        datasets.append({"path": str(lvl), "coordinateTransformations": ct})
    ms = {
        "version": "0.5",
        "name": volume_name.rstrip("/").split("/")[-1],
        "axes": src.get("axes", [{"name": n, "type": "space"} for n in "zyx"]),
        "datasets": datasets,
        "type": "downscale",
        "metadata": {"source": BUCKET + "/" + volume_name, "codec": "volcomp"},
    }
    return {"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": "0.5", "multiscales": [ms]}},
            "consolidated_metadata": None}


def surface_array_metadata(level, info):
    """zarr v3 array metadata for one rung of a prediction: shard 1024/512/256 for the
    three finest levels and 128 for everything from the fourth up, 128^3 volcomp chunks."""
    md = array_metadata(level["shape"], level["q"])
    enc = level.get("encoding")
    if enc in ("mask", "mask-lossless"):
        md["codecs"][0]["configuration"]["codecs"] = [
            {"name": "volcomp", "configuration": {"mode": enc, "q": 0}}]
    sh = int(level["shard"])
    md["chunk_grid"]["configuration"]["chunk_shape"] = [sh, sh, sh]
    md["attributes"] = {"volcomp": {"q": level["q"], "voxel_size_um": level["um"],
                                    "encoding": level.get("encoding", "ramp")}}
    return md


def surface_group_metadata(name, info):
    """OME-Zarr 0.5 multiscales: dataset paths are the voxel size in micrometres and
    the coordinate transforms state that size. Resampled exports sit on the exact
    0.6 * 2^k ladder; an unresampled one keeps the source grid and its levels carry
    their TRUE sizes (native_um * 2^L)."""
    datasets = [{"path": lv["path"],
                 "coordinateTransformations": [{"type": "scale", "scale": [lv["um"]] * 3}]}
                for lv in info["levels"]]
    # the 2x "mask" and "ramp" trees keep the wording they were written with;
    # mask-lossless says what it really does
    lossless = info["encoding"].get("name") == "surface-mask-lossless"
    pyramid_note = ("2x2x2 majority pooling of the level below"
                    if lossless else "2x mean pooling of the level below")
    if not info.get("resampled", True):
        pyramid_note += "; level 0 is the published grid, unresampled"
    ms = {
        "version": "0.5",
        "name": name.rstrip("/").split("/")[-1],
        "axes": [{"name": n, "type": "space", "unit": "micrometer"} for n in "zyx"],
        "datasets": datasets,
        "type": "majority" if lossless else "mean",
        "metadata": {"source": info["source"], "codec": "volcomp",
                     "description": pyramid_note},
    }
    export = {
        "source": info["source"], "source_volume": BUCKET + "/" + info["source_volume"],
        "source_level": info["source_level"], "source_chunk": info["csize"],
        "threshold": info["threshold"], "encoding": info["encoding"],
        "native_voxel_size_um": info["native_um"], "rung_voxel_size_um": info["rung_um"],
        "resampled": info.get("resampled", True),
        "resample_scale": info["scale"], "source_shape": info["src_shape"],
        "shape": info["out_shape"],
        "levels": [{"path": lv["path"], "voxel_size_um": lv["um"], "shape": lv["shape"],
                    "encoding": lv.get("encoding", "ramp"), "q": lv["q"], "shard": lv["shard"]}
                   for lv in info["levels"]],
    }
    return {"zarr_format": 3, "node_type": "group",
            "attributes": {"ome": {"version": "0.5", "multiscales": [ms]}, "volcomp": export},
            "consolidated_metadata": None}


def cmd_metadata(a):
    db = open_db(a.db)
    n = 0
    for name, info in db.execute("SELECT name, info FROM volume WHERE kind='surface'"):
        info = json.loads(info)
        root = os.path.join(a.out, name.rstrip("/"))
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "zarr.json"), "w") as f:
            json.dump(surface_group_metadata(name, info), f, indent=2)
        for lv in info["levels"]:
            d = os.path.join(root, lv["path"])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "zarr.json"), "w") as f:
                json.dump(surface_array_metadata(lv, info), f, indent=2)
        n += 1
    for name, zattrs, levels in db.execute("SELECT name, zattrs, levels FROM volume WHERE kind='ct'"):
        levels = {int(k): v for k, v in json.loads(levels).items()}
        root = os.path.join(a.out, name.rstrip("/"))
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "zarr.json"), "w") as f:
            json.dump(group_metadata(name, json.loads(zattrs), levels), f, indent=2)
        for lvl, info in levels.items():
            d = os.path.join(root, str(lvl))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "zarr.json"), "w") as f:
                json.dump(array_metadata(info["shape"], level_q(lvl, a.q)), f, indent=2)
        n += 1
    print(f"wrote metadata for {n} volumes under {a.out}")


# ----------------------------------------------------------------------------- server


class Coordinator:
    def __init__(self, db, lease):
        self.db, self.lease, self.lock = db, lease, threading.RLock()
        self.max_attempts = 8
        self._levels = {}  # volume -> (kind, levels|info); filled lazily so manifests can be added while serving

    def volume(self, vol):
        if vol not in self._levels:
            row = self.db.execute("SELECT levels, kind, info FROM volume WHERE name=?", (vol,)).fetchone()
            if row is None:
                raise KeyError(vol)
            if row[1] == "surface":
                self._levels[vol] = ("surface", json.loads(row[2]))
            else:
                self._levels[vol] = ("ct", {int(k): v for k, v in json.loads(row[0]).items()})
        return self._levels[vol]

    def levels(self, vol):
        kind, v = self.volume(vol)
        return v

    def claim(self, worker):
        with self.lock:
            now = time.time()
            while True:
                row = self.db.execute(
                    "SELECT id, volume, level, sz, sy, sx, attempts, mask FROM unit WHERE state='todo' "
                    "OR (state='leased' AND lease_until < ?) ORDER BY attempts, volume, level, id LIMIT 1", (now,)).fetchone()
                if row is None:
                    return None
                uid, vol, lvl, sz, sy, sx, attempts, mask = row
                if attempts < self.max_attempts:
                    break
                # give up on poison units so the queue can drain; they stay visible in /status as "failed"
                self.db.execute("UPDATE unit SET state='failed' WHERE id=?", (uid,))
            kind, meta = self.volume(vol)  # before the lease so a bad row cannot leak a lease
            unit = {"id": uid, "kind": kind, "volume": vol, "level": lvl, "shard": [sz, sy, sx],
                    "lease_seconds": self.lease}
            if kind == "surface":
                lv = meta["levels"][:SURF_LEVELS]
                unit.update(shape=meta["out_shape"], src_shape=meta["src_shape"], csize=meta["csize"],
                            scale=meta["scale"], dmax=meta["encoding"]["dmax"],
                            encoding=lv[0].get("encoding", "ramp"),
                            q=[x["q"] for x in lv], paths=[x["path"] for x in lv],
                            mask=mask.hex() if mask else None)
            else:
                unit.update(shape=meta[lvl]["shape"], q=level_q(lvl), mask=mask.hex() if mask else None)
            self.db.execute("UPDATE unit SET state='leased', worker=?, lease_until=?, attempts=attempts+1 WHERE id=?",
                            (worker, now + self.lease, uid))
        return unit

    def done(self, body):
        with self.lock:
            self.db.execute("UPDATE unit SET state='done', bytes=?, present=?, psnr_min=?, max_err=?, done_at=?, error=NULL "
                            "WHERE id=? AND state='leased'",
                            (body.get("bytes"), body.get("present"), body.get("psnr_min"), body.get("max_err"),
                             time.time(), body["id"]))

    def fail(self, body):
        with self.lock:
            self.db.execute("UPDATE unit SET state='todo', error=?, worker=NULL, lease_until=NULL WHERE id=? AND state='leased'",
                            (str(body.get("error", ""))[:2000], body["id"]))

    def status(self):
        with self.lock:
            tot = dict(self.db.execute("SELECT state, COUNT(*) FROM unit GROUP BY state").fetchall())
            out_bytes = self.db.execute("SELECT COALESCE(SUM(bytes),0), COALESCE(SUM(present),0) FROM unit WHERE state='done'").fetchone()
            per = [{"volume": v, "done": d, "total": t, "bytes": b or 0}
                   for v, d, t, b in self.db.execute(
                       "SELECT volume, SUM(state='done'), COUNT(*), SUM(bytes) FROM unit GROUP BY volume ORDER BY volume")]
            recent = self.db.execute("SELECT COUNT(*), COALESCE(SUM(present),0), COALESCE(SUM(bytes),0) FROM unit "
                                     "WHERE state='done' AND done_at > ?", (time.time() - 600,)).fetchone()
            est = self.db.execute("SELECT COALESCE(SUM(est),0), SUM(est IS NULL) FROM unit WHERE state IN ('todo','leased')").fetchone()
            skipped = self.db.execute("SELECT COUNT(*) FROM unit WHERE state='done' AND error='occupancy:empty'").fetchone()[0]
        return {"units": tot, "output_bytes": out_bytes[0], "chunks_present": out_bytes[1],
                "done_last_10min": recent[0], "chunks_last_10min": recent[1], "bytes_last_10min": recent[2],
                "est_chunks_remaining": est[0], "units_without_estimate": est[1] or 0, "skipped_empty": skipped,
                "volumes": per}


class Handler(BaseHTTPRequestHandler):
    coord: Coordinator = None

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj=None):
        body = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.startswith("/status"):
            return self._send(200, self.coord.status())
        self._send(404, {"error": "no such route"})

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/claim":
                u = self.coord.claim(body.get("worker", self.client_address[0]))
                return self._send(200, u) if u else self._send(204)
            if self.path == "/done":
                self.coord.done(body)
                return self._send(200, {"ok": True})
            if self.path == "/fail":
                self.coord.fail(body)
                return self._send(200, {"ok": True})
            self._send(404, {"error": "no such route"})
        except Exception as e:  # noqa: BLE001
            self._send(400, {"error": repr(e)})


def cmd_serve(a):
    Handler.coord = Coordinator(open_db(a.db), a.lease)
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"coordinator on http://{a.bind}:{a.port}  db={a.db}  lease={a.lease}s", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_requeue(a):
    db = open_db(a.db)
    n = db.execute("UPDATE unit SET state='todo', worker=NULL, lease_until=NULL, attempts=0, error=NULL "
                   "WHERE state IN ('leased','failed')").rowcount
    if a.done_levels:
        lv = [int(x) for x in a.done_levels.split(",")]
        n += db.execute("UPDATE unit SET state='todo', worker=NULL, lease_until=NULL, attempts=0, error=NULL, "
                        f"bytes=NULL, done_at=NULL WHERE state='done' AND level IN ({','.join('?' * len(lv))})", lv).rowcount
    print(f"requeued {n} units")


def cmd_report(a):
    s = Coordinator(open_db(a.db), 0).status()
    u = s["units"]
    total = sum(u.values())
    print(f"units: {u.get('done', 0)}/{total} done, {u.get('leased', 0)} leased, {u.get('todo', 0)} todo, "
          f"{u.get('failed', 0)} failed; output {s['output_bytes'] / 1e9:.2f} GB, "
          f"{s['chunks_present']} chunks, {s['done_last_10min']} units in the last 10 min")
    per_chunk = s["output_bytes"] / s["chunks_present"] if s["chunks_present"] else 0
    rate = s["chunks_last_10min"] / 600.0
    rem = s["est_chunks_remaining"]
    print(f"occupancy: {s['skipped_empty']} units skipped as empty, {s['units_without_estimate']} units without an estimate; "
          f"~{rem} chunks remaining (~{rem * per_chunk / 1e9:.0f} GB); last 10 min {s['chunks_last_10min']} chunks, "
          f"{s['bytes_last_10min'] / 1e6 / 600:.1f} MB/s out"
          + (f"; ETA {rem / rate / 3600:.1f} h at that rate" if rate > 0 and rem else ""))
    for v in s["volumes"]:
        print(f"  {v['done']:>7}/{v['total']:<7} {v['bytes'] / 1e9:9.2f} GB  {v['volume']}")


# ----------------------------------------------------------------------------- occupancy


OCC_LEVEL = 5  # a level-5 voxel spans a 32^3 block of native voxels; level-5 data is 1/32768 of native
OCC_DILATE = 1  # coarse voxels: absorbs averaging/rounding at the edge of masked regions


def cmd_occupancy(a):
    db = open_db(a.db)
    # CT volumes only: a surface unit lists the source rows it needs instead
    vols = [r[0] for r in db.execute("SELECT name FROM volume WHERE kind='ct' ORDER BY name")]
    if a.volume:
        want = [v if v.endswith("/") else v + "/" for v in a.volume]
        vols = [v for v in vols if v in want]
    os.makedirs(a.tmp, exist_ok=True)
    t_all = time.time()
    for vol in vols:
        levels_json, occ_done = db.execute("SELECT levels, occ_level FROM volume WHERE name=?", (vol,)).fetchone()
        if occ_done is not None and not a.force:
            continue
        levels = {int(k): v for k, v in json.loads(levels_json).items()}
        L = min(OCC_LEVEL, max(levels))
        shape = levels[L]["shape"]
        cg = [math.ceil(n / CHUNK) for n in shape]
        t0 = time.time()
        work = tempfile.mkdtemp(prefix="occ-", dir=a.tmp)
        try:
            keys = [(cz, cy, cx) for cz in range(cg[0]) for cy in range(cg[1]) for cx in range(cg[2])]

            def fetch(c):
                data = http_get(BUCKET + "/" + urllib.parse.quote(f"{vol}{L}/{c[0]}/{c[1]}/{c[2]}"))
                if data is None:
                    return 0
                if len(data) != CHUNK ** 3:
                    raise RuntimeError(f"{vol}{L}/{c}: {len(data)} bytes")
                with open(os.path.join(work, f"{c[0]}_{c[1]}_{c[2]}.u8"), "wb") as f:
                    f.write(data)
                return 1

            with concurrent.futures.ThreadPoolExecutor(a.threads) as ex:
                got = sum(ex.map(fetch, keys))
            t1 = time.time()
            summary = []
            unit_levels = [r[0] for r in db.execute("SELECT DISTINCT level FROM unit WHERE volume=? ORDER BY level", (vol,))]
            for k in unit_levels:
                factor = 2 ** (7 + k - L)  # native chunk (128 voxels) measured in level-L voxels, times 2^k
                grid = [math.ceil(n / CHUNK) for n in levels[k]["shape"]]
                sgrid = [math.ceil(g / 8) for g in grid]
                out = os.path.join(work, f"masks-{k}.bin")
                r = subprocess.run([a.volcomp, "occupancy", work, out, f"--shape={shape[0]},{shape[1]},{shape[2]}",
                                    f"--factor={factor}", f"--dilate={OCC_DILATE}", f"--grid={grid[0]},{grid[1]},{grid[2]}",
                                    "--shards", "--chunks"], capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"volcomp occupancy failed for {vol} level {k}: {r.stderr[-500:]}")
                with open(out, "rb") as f:
                    masks = f.read()
                assert len(masks) == sgrid[0] * sgrid[1] * sgrid[2] * 64, (len(masks), sgrid)
                rows, empties, est_sum = [], [], 0
                for uid, sz, sy, sx in db.execute("SELECT id, sz, sy, sx FROM unit WHERE volume=? AND level=?", (vol, k)):
                    i = ((sz * sgrid[1] + sy) * sgrid[2] + sx) * 64
                    m = masks[i:i + 64]
                    est = int.from_bytes(m, "little").bit_count()
                    est_sum += est
                    rows.append((est, m if est else None, uid))
                    if est == 0:
                        empties.append((uid,))
                db.execute("BEGIN")
                db.executemany("UPDATE unit SET est=?, mask=? WHERE id=?", rows)
                n_skip = db.executemany("UPDATE unit SET state='done', bytes=0, present=0, done_at=strftime('%s','now'), "
                                        "error='occupancy:empty', worker=NULL, lease_until=NULL WHERE id=? "
                                        "AND state IN ('todo','failed')", empties).rowcount
                db.execute("COMMIT")
                summary.append(f"L{k}: {est_sum}/{len(rows) * 512} chunks, {len(empties)} empty units ({n_skip} skipped now)")
            db.execute("UPDATE volume SET occ_level=? WHERE name=?", (L, vol))
            print(f"{vol}: level {L} {shape} {got}/{len(keys)} chunks in {t1 - t0:.0f}s; " + "; ".join(summary)
                  + f"; {time.time() - t1:.0f}s", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print(f"occupancy done in {time.time() - t_all:.0f}s")


# ----------------------------------------------------------------------------- pool-levels


def pool_fetch(src, key, dest):
    """Copy one shard from the level-below tree into dest. `src` is a local directory or
    an http(s) root (the published tree). Returns False if the shard does not exist."""
    if src.startswith("http://") or src.startswith("https://"):
        data = http_get(src.rstrip("/") + "/" + urllib.parse.quote(key))
        if data is None:
            return False
        with open(dest, "wb") as f:
            f.write(data)
        return True
    path = os.path.join(src, key)
    if not os.path.exists(path):
        return False
    if os.path.realpath(path) != os.path.realpath(dest):
        shutil.copyfile(path, dest)
    return True


def cmd_pool_levels(a):
    """Levels above the four a unit writes, by 2x mean pooling, one 128^3 shard at a time.
    Streaming (never more than 8 input shards in memory) and resumable: an output shard
    that already exists is left alone."""
    db = open_db(a.db)
    rows = db.execute("SELECT name, info FROM volume WHERE kind='surface' ORDER BY name").fetchall()
    if a.volume:
        want = {v if v.endswith("/") else v + "/" for v in a.volume}
        rows = [r for r in rows if r[0] in want]
    os.makedirs(a.tmp, exist_ok=True)
    t_all = time.time()
    for name, info in rows:
        info = json.loads(info)
        levels = info["levels"]
        made = skipped = 0
        t0 = time.time()
        for j in range(a.first, len(levels)):
            below, here = levels[j - 1], levels[j]
            grid = [math.ceil(n / CHUNK) for n in here["shape"]]
            for sz in range(grid[0]):
                for sy in range(grid[1]):
                    for sx in range(grid[2]):
                        key = f"{name}{here['path']}/c/{sz}/{sy}/{sx}"
                        out = os.path.join(a.out, key)
                        if os.path.exists(out):
                            skipped += 1
                            continue
                        work = tempfile.mkdtemp(prefix="pool-", dir=a.tmp)
                        try:
                            ins, got = [], 0
                            for dz in range(2):
                                for dy in range(2):
                                    for dx in range(2):
                                        k = f"{name}{below['path']}/c/{2 * sz + dz}/{2 * sy + dy}/{2 * sx + dx}"
                                        dest = os.path.join(work, f"{dz}{dy}{dx}.shard")
                                        if pool_fetch(a.src, k, dest):
                                            ins.append(dest)
                                            got += 1
                                        else:
                                            ins.append("-")
                            if not got:
                                continue  # nothing below: a missing shard is the fill value
                            tmp_out = os.path.join(work, "out.shard")
                            enc = here.get("encoding")
                            flag = ("--mask-lossless" if enc == "mask-lossless" else
                                    "--mask" if enc == "mask" else f"--q={here['q']:g}")
                            r = subprocess.run([a.volcomp, "shard-pool", tmp_out, flag,
                                                "--shape={},{},{}".format(*below["shape"]),
                                                f"--pos={sz},{sy},{sx}"] + ins, capture_output=True, text=True)
                            if r.returncode:
                                raise RuntimeError(f"shard-pool failed for {key}: {r.stderr[-400:]}")
                            if not os.path.exists(tmp_out):
                                continue  # pooled to all zero
                            os.makedirs(os.path.dirname(out), exist_ok=True)
                            shutil.move(tmp_out, out + ".part")
                            os.replace(out + ".part", out)
                            made += 1
                        finally:
                            shutil.rmtree(work, ignore_errors=True)
            a.src = a.out if a.chain else a.src  # levels above the first read what we just wrote
        print(f"{name}: {made} shards written, {skipped} already there, {time.time() - t0:.0f}s", flush=True)
    print(f"pool-levels done in {time.time() - t_all:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--db", required=True)
    p.add_argument("--volume", action="append", help="bucket prefix of one volume (repeatable); default: all")
    p.add_argument("--levels", default="all", help="comma list of multiscale levels, default all")
    p.add_argument("--include-compressed", action="store_true")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("manifest-surfaces", help="queue the published surface predictions")
    p.add_argument("--db", required=True)
    p.add_argument("--volume", action="append", help="bucket prefix of one prediction (repeatable); default: all")
    p.add_argument("--volcomp", default="volcomp")
    p.add_argument("--tmp", default="/var/tmp/volcomp-occ")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--no-occupancy", action="store_true",
                   help="skip the occupancy masks (every chunk is then assumed occupied)")
    p.add_argument("--occupancy-samples", type=int, default=6,
                   help="coarse chunks compared with the level below before trusting the pyramid")
    p.add_argument("--encoding", choices=SURF_ENCODINGS, default="mask",
                   help="mask (default): the binary mask as 2x volcomp mask chunks, resampled onto "
                        "the 0.6*2^k ladder; mask-lossless: the mask stored exactly, on its native "
                        "grid (implies --no-resample); ramp: the older signed-distance ramp at the "
                        "ladder's q")
    p.add_argument("--no-resample", dest="resample", action="store_false", default=None,
                   help="keep the source grid exactly (scale 1, out shape = source shape) and name "
                        "the levels by their true voxel size; the default for --encoding mask-lossless")
    p.add_argument("--resample", dest="resample", action="store_true", default=None,
                   help="resample onto the 0.6*2^k ladder (the default for --encoding mask and ramp)")
    p.set_defaults(fn=cmd_manifest_surfaces)
    p = sub.add_parser("pool-levels", help="build the coarse levels of the predictions offline")
    p.add_argument("--db", required=True)
    p.add_argument("--volcomp", default="volcomp")
    p.add_argument("--src", required=True, help="tree holding the levels a unit wrote: a directory or an https root")
    p.add_argument("--out", required=True, help="directory to write the coarse levels into")
    p.add_argument("--volume", action="append")
    p.add_argument("--first", type=int, default=SURF_LEVELS, help="first level index to build (default 4)")
    p.add_argument("--tmp", default="/var/tmp/volcomp-pool")
    p.add_argument("--no-chain", dest="chain", action="store_false", default=True,
                   help="read every level from --src instead of chaining through --out")
    p.set_defaults(fn=cmd_pool_levels)
    p = sub.add_parser("metadata")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--q", type=float, default=Q_NATIVE, help="native-resolution q; halved per level, floor 1")
    p.set_defaults(fn=cmd_metadata)
    p = sub.add_parser("serve")
    p.add_argument("--db", required=True)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--lease", type=int, default=900)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("report")
    p.add_argument("--db", required=True)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("requeue", help="put leased/failed units back in the queue")
    p.add_argument("--db", required=True)
    p.add_argument("--done-levels", help="also redo finished units of these levels (comma list)")
    p.set_defaults(fn=cmd_requeue)
    p = sub.add_parser("occupancy", help="per-unit chunk masks from the coarsest level; marks empty units done")
    p.add_argument("--db", required=True)
    p.add_argument("--volcomp", default="volcomp")
    p.add_argument("--tmp", default="/var/tmp/volcomp-occ")
    p.add_argument("--volume", action="append")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--force", action="store_true", help="recompute volumes that already have masks")
    p.set_defaults(fn=cmd_occupancy)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
