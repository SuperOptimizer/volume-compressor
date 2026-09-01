#!/usr/bin/env python3
"""volcomp export coordinator (Python 3 standard library only).

One process on the coordinator VM owns a sqlite queue of work units; each unit
is one 1024^3 zarr v3 shard of one multiscale level of one source volume.
Compute VMs run tools/export/worker.py against the HTTP API below.

  coordinator.py manifest  --db export.db [--volume PREFIX ...] [--levels 0,1,...]
      Enumerate the public S3 bucket, read every uncompressed volume's zarr v2
      metadata and insert all (volume, level, shard) units. Re-runnable: units
      already present are left alone (progress is never lost).
  coordinator.py metadata  --db export.db --out DIR [--q 8]
      Write the zarr v3 group/array metadata (OME-Zarr 0.5 multiscales,
      sharding_indexed 1024^3 -> 128^3, codec "volcomp") into DIR mirroring
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

The bucket is read anonymously over HTTPS; nothing here needs AWS credentials.
"""
import argparse
import json
import math
import os
import socket
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SHARD = 1024
CHUNK = 128
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
  levels TEXT NOT NULL                -- JSON: {level: {"shape": [z,y,x]}}
);
CREATE TABLE IF NOT EXISTS unit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  volume TEXT NOT NULL, level INTEGER NOT NULL,
  sz INTEGER NOT NULL, sy INTEGER NOT NULL, sx INTEGER NOT NULL,
  state TEXT NOT NULL DEFAULT 'todo', -- todo | leased | done
  worker TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
  bytes INTEGER, present INTEGER, psnr_min REAL, max_err INTEGER, done_at REAL, error TEXT,
  UNIQUE(volume, level, sz, sy, sx)
);
CREATE INDEX IF NOT EXISTS unit_state ON unit(state, lease_until);
CREATE INDEX IF NOT EXISTS unit_volume ON unit(volume, state);
"""


def open_db(path):
    db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(SCHEMA)
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


# ----------------------------------------------------------------------------- metadata


def array_metadata(shape, q):
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


def cmd_metadata(a):
    db = open_db(a.db)
    n = 0
    for name, zattrs, levels in db.execute("SELECT name, zattrs, levels FROM volume"):
        levels = {int(k): v for k, v in json.loads(levels).items()}
        root = os.path.join(a.out, name.rstrip("/"))
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "zarr.json"), "w") as f:
            json.dump(group_metadata(name, json.loads(zattrs), levels), f, indent=2)
        for lvl, info in levels.items():
            d = os.path.join(root, str(lvl))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "zarr.json"), "w") as f:
                json.dump(array_metadata(info["shape"], a.q), f, indent=2)
        n += 1
    print(f"wrote metadata for {n} volumes under {a.out}")


# ----------------------------------------------------------------------------- server


class Coordinator:
    def __init__(self, db, lease):
        self.db, self.lease, self.lock = db, lease, threading.Lock()
        self._levels = {}  # volume -> {level: {"shape": ...}}; filled lazily so manifests can be added while serving

    def levels(self, vol):
        if vol not in self._levels:
            row = self.db.execute("SELECT levels FROM volume WHERE name=?", (vol,)).fetchone()
            if row is None:
                raise KeyError(vol)
            self._levels[vol] = {int(k): v for k, v in json.loads(row[0]).items()}
        return self._levels[vol]

    def claim(self, worker):
        with self.lock:
            now = time.time()
            row = self.db.execute(
                "SELECT id, volume, level, sz, sy, sx, attempts FROM unit WHERE state='todo' "
                "OR (state='leased' AND lease_until < ?) ORDER BY attempts, volume, level, id LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            uid, vol, lvl, sz, sy, sx, attempts = row
            if attempts >= 8:
                # give up on poison units so the queue can drain; they stay visible in /status
                self.db.execute("UPDATE unit SET state='failed' WHERE id=?", (uid,))
                return self.claim(worker)
            shape = self.levels(vol)[lvl]["shape"]  # before the lease so a bad row cannot leak a lease
            self.db.execute("UPDATE unit SET state='leased', worker=?, lease_until=?, attempts=attempts+1 WHERE id=?",
                            (worker, now + self.lease, uid))
        return {"id": uid, "volume": vol, "level": lvl, "shard": [sz, sy, sx], "shape": shape,
                "lease_seconds": self.lease}

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
            recent = self.db.execute("SELECT COUNT(*) FROM unit WHERE state='done' AND done_at > ?", (time.time() - 600,)).fetchone()[0]
        return {"units": tot, "output_bytes": out_bytes[0], "chunks_present": out_bytes[1],
                "done_last_10min": recent, "volumes": per}


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
    n = db.execute("UPDATE unit SET state='todo', worker=NULL, lease_until=NULL WHERE state IN ('leased','failed')").rowcount
    print(f"requeued {n} units")


def cmd_report(a):
    s = Coordinator(open_db(a.db), 0).status()
    u = s["units"]
    total = sum(u.values())
    print(f"units: {u.get('done', 0)}/{total} done, {u.get('leased', 0)} leased, {u.get('todo', 0)} todo, "
          f"{u.get('failed', 0)} failed; output {s['output_bytes'] / 1e9:.2f} GB, "
          f"{s['chunks_present']} chunks, {s['done_last_10min']} units in the last 10 min")
    for v in s["volumes"]:
        print(f"  {v['done']:>7}/{v['total']:<7} {v['bytes'] / 1e9:9.2f} GB  {v['volume']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("--db", required=True)
    p.add_argument("--volume", action="append", help="bucket prefix of one volume (repeatable); default: all")
    p.add_argument("--levels", default="all", help="comma list of multiscale levels, default all")
    p.add_argument("--include-compressed", action="store_true")
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("metadata")
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--q", type=float, default=8.0)
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
    p.set_defaults(fn=cmd_requeue)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
