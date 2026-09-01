#!/usr/bin/env python3
"""volcomp export worker (Python 3 standard library + the `volcomp` CLI + curl).

Runs on every compute VM, one process per N cores:

  worker.py run --coordinator http://COORD:8765 --volcomp ./volcomp --tmp /dev/shm/volcomp \
                --sftp sftp://dl.ash2txt.org:9238/volcomp --netrc ~/.volcomp-netrc [--parallel 8] [--q 8]
      loop: claim a shard -> download its 512 source chunks from S3 (threads) ->
            volcomp shard-pack -> volcomp shard-verify (index CRC, every chunk
            decodes, sampled PSNR vs source) -> upload to SFTP as <key>.part and
            rename -> report done. Any failure reports /fail and the unit is
            re-queued; the coordinator's lease also expires if this process dies.
  worker.py run ... --local-out DIR        write shards under DIR instead of SFTP (testing)
  worker.py upload-tree DIR --sftp URL --netrc FILE
      upload a local tree (e.g. the coordinator's metadata directory) to SFTP.

--parallel is the number of shards in flight per process (each shard is one
download pool of 32 connections + one shard-pack); the encode is ~1 s of CPU per
shard, so a VM is bound by its S3 download rate. Shard keys mirror the bucket:
<sftp>/<scroll>/volumes/<volume>.zarr/<level>/c/<sz>/<sy>/<sx>.
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
CHUNK_BYTES = 128 ** 3
SHARD_CHUNKS = 8


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- coordinator API


def api(url, path, body=None, retries=1000):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 204:
                    return None
                return json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if 400 <= e.code < 500:
                raise RuntimeError(f"coordinator rejected {path}: HTTP {e.code} {body}")
            log(f"coordinator error on {path}: HTTP {e.code} {body}; retrying")
            time.sleep(min(60, 2 + attempt * 2))
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            log(f"coordinator unreachable ({e}); retrying")
            time.sleep(min(60, 2 + attempt * 2))
    raise RuntimeError("coordinator unreachable")


# ----------------------------------------------------------------------------- S3 download

BUCKET_HOST = "vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
_tls = threading.local()


class ChunkFetchError(Exception):
    pass


def s3_conn(reset=False):
    """One keep-alive HTTPS connection per thread (a fresh TLS handshake per 2 MiB
    object was the bottleneck: ~25 MB/s per VM and a saturated CPU)."""
    c = getattr(_tls, "conn", None)
    if c is None or reset:
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        c = http.client.HTTPSConnection(BUCKET_HOST, timeout=120)
        _tls.conn = c
    return c


def s3_request(path, retries=8):
    """GET an object path (already URL-encoded). Returns bytes, or None on 404."""
    err = "?"
    for attempt in range(retries):
        try:
            c = s3_conn(reset=attempt > 0)
            c.request("GET", path, headers={"Connection": "keep-alive"})
            r = c.getresponse()
            body = r.read()
            if r.status == 200:
                return body
            if r.status == 404:
                return None
            err = f"HTTP {r.status}"
            if 400 <= r.status < 500 and r.status not in (408, 429):
                raise ChunkFetchError(f"{path}: {err}")
        except ChunkFetchError:
            raise
        except (http.client.HTTPException, socket.timeout, ConnectionError, OSError) as e:
            err = repr(e)
        time.sleep(min(30, 1.5 ** attempt))
    raise ChunkFetchError(f"{path}: giving up after {retries} attempts ({err})")


def s3_list_row(prefix):
    """Keys under prefix (a single 128-chunk row: <vol><level>/<cz>/<cy>/), <= 1000 keys."""
    q = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "max-keys": "1000"})
    body = s3_request("/?" + q)
    if body is None:
        raise ChunkFetchError(f"list {prefix}: 404")
    return set(re.findall(rb"<Key>([^<]+)</Key>", body))


def fetch_chunk(key, dest):
    """GET one source chunk into dest. True if stored, False if the key does not exist
    (masked region). Transient failures retry and then raise: they are never
    mistaken for a missing chunk."""
    data = s3_request("/" + urllib.parse.quote(key))
    if data is None:
        return False
    if len(data) != CHUNK_BYTES:
        raise ChunkFetchError(f"{key}: {len(data)} bytes, expected {CHUNK_BYTES}")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return True


def download_shard(unit, workdir, pool):
    """Fetch the (up to) 512 chunks of one shard into workdir/z_y_x.u8.
    Chunks outside the level's chunk grid are skipped. Each (cz, cy) row is listed
    first (64 small requests per shard) so masked/absent chunks cost no GETs."""
    vol, lvl = unit["volume"], unit["level"]
    sz, sy, sx = unit["shard"]
    grid = [(n + 127) // 128 for n in unit["shape"]]
    rows = [(cz, cy) for cz in range(sz * 8, min(sz * 8 + 8, grid[0])) for cy in range(sy * 8, min(sy * 8 + 8, grid[1]))]
    listed = list(pool.map(lambda r: s3_list_row(f"{vol}{lvl}/{r[0]}/{r[1]}/"), rows))
    jobs, n_in_grid = [], 0
    for (cz, cy), keys in zip(rows, listed):
        for cx in range(sx * 8, min(sx * 8 + 8, grid[2])):
            n_in_grid += 1
            key = f"{vol}{lvl}/{cz}/{cy}/{cx}"
            if key.encode() not in keys:
                continue
            dest = os.path.join(workdir, f"{cz - sz * 8}_{cy - sy * 8}_{cx - sx * 8}.u8")
            jobs.append(pool.submit(fetch_chunk, key, dest))
    present = 0
    for j in jobs:
        present += bool(j.result())  # raises ChunkFetchError on a hard failure
    return n_in_grid, present


# ----------------------------------------------------------------------------- pack / verify / upload


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} -> {r.returncode}: {r.stderr.strip()[-500:]}")
    return r.stdout


def parse_kv(s):
    return {k: (float(v) if re.fullmatch(r"-?\d+\.\d+", v) else int(v) if v.isdigit() else v)
            for k, v in re.findall(r"(\w+)=(\S+)", s)}


def shard_key(unit):
    sz, sy, sx = unit["shard"]
    return f"{unit['volume']}{unit['level']}/c/{sz}/{sy}/{sx}"


def netrc_password(netrc, host):
    """password for host from a netrc file (machine H login U password P)."""
    with open(netrc) as f:
        toks = f.read().split()
    for i, t in enumerate(toks):
        if t == "machine" and i + 1 < len(toks) and toks[i + 1] == host and "password" in toks[i:]:
            return toks[toks.index("password", i) + 1]
    raise RuntimeError(f"no password for {host} in {netrc}")


def netrc_login(netrc, host):
    with open(netrc) as f:
        toks = f.read().split()
    for i, t in enumerate(toks):
        if t == "machine" and i + 1 < len(toks) and toks[i + 1] == host and "login" in toks[i:]:
            return toks[toks.index("login", i) + 1]
    raise RuntimeError(f"no login for {host} in {netrc}")


def sftp_batch(sftp_root, netrc, commands):
    """Run an OpenSSH sftp batch (one connection) against sftp://host:port/dir.
    Commands prefixed with '-' may fail without aborting the batch."""
    m = re.match(r"sftp://([^/:]+)(?::(\d+))?(/.*)?$", sftp_root.rstrip("/"))
    if not m:
        raise RuntimeError("--sftp must look like sftp://host:port/dir")
    host, port, prefix = m.group(1), m.group(2) or "22", (m.group(3) or "")
    env = dict(os.environ, SSHPASS=netrc_password(netrc, host))
    script = "".join(c.replace("{root}", prefix) + "\n" for c in commands)
    r = subprocess.run(["sshpass", "-e", "sftp", "-q", "-o", "BatchMode=no", "-o", "PubkeyAuthentication=no",
                        "-o", "PreferredAuthentications=password", "-o", "StrictHostKeyChecking=no",
                        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30",
                        "-b", "-", "-P", port, f"{netrc_login(netrc, host)}@{host}"],
                       input=script, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"sftp batch failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-500:]}")


def sftp_upload(local, sftp_root, key, netrc):
    """Upload to <key>.part, then rename into place (readers never see a partial shard).
    Parent directories are created on the way (mkdir is not recursive in sftp)."""
    parts = key.split("/")
    cmds = []
    for i in range(1, len(parts)):
        cmds.append("-mkdir {root}/" + "/".join(parts[:i]))
    cmds += [f"put {local} {{root}}/{key}.part", f"-rm {{root}}/{key}", f"rename {{root}}/{key}.part {{root}}/{key}"]
    sftp_batch(sftp_root, netrc, cmds)


def local_store(local, root, key):
    dest = os.path.join(root, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(local, dest + ".part")
    os.replace(dest + ".part", dest)


def process_unit(unit, a, pool):
    t0 = time.time()
    workdir = tempfile.mkdtemp(prefix="shard-", dir=a.tmp)
    try:
        n_jobs, present = download_shard(unit, workdir, pool)
        t1 = time.time()
        shard = os.path.join(workdir, "out.shard")
        q = float(unit.get("q", a.q))  # the coordinator sets q per multiscale level
        key = shard_key(unit)
        if not present:
            # all chunks masked/absent: no shard is written at all — in zarr v3 a missing
            # shard key reads as the fill value, and this saves an SFTP session per empty shard
            log(f"done #{unit['id']} {key}: q{q:g} 0/{n_jobs} chunks, empty (not written), dl {t1 - t0:.1f}s")
            return {"id": unit["id"], "bytes": 0, "present": 0, "psnr_min": None, "max_err": None}
        info = parse_kv(run([a.volcomp, "shard-pack", workdir, shard, f"--q={q}"]))
        info.update(parse_kv(run([a.volcomp, "shard-verify", shard, workdir, f"--samples={a.samples}"])))
        t2 = time.time()
        size = os.path.getsize(shard)
        if a.local_out:
            local_store(shard, a.local_out, key)
        else:
            sftp_upload(shard, a.sftp, key, a.netrc)
        t3 = time.time()
        log(f"done #{unit['id']} {key}: q{q:g} {present}/{n_jobs} chunks, {size / 1e6:.1f} MB, "
            f"psnr_min {info.get('psnr_min', 0):.1f}, dl {t1 - t0:.1f}s enc {t2 - t1:.1f}s up {t3 - t2:.1f}s")
        return {"id": unit["id"], "bytes": size, "present": info.get("present", present),
                "psnr_min": info.get("psnr_min"), "max_err": info.get("max_err")}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def cmd_run(a):
    if not a.local_out and not (a.sftp and a.netrc):
        sys.exit("need --sftp and --netrc, or --local-out")
    os.makedirs(a.tmp, exist_ok=True)
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    log(f"worker {worker_id} -> {a.coordinator}, {a.parallel} shards in flight")
    dl_pool = cf.ThreadPoolExecutor(max_workers=a.connections * a.parallel)
    idle_since = None
    with cf.ThreadPoolExecutor(max_workers=a.parallel) as units:
        inflight = {}
        while True:
            while len(inflight) < a.parallel:
                try:
                    unit = api(a.coordinator, "/claim", {"worker": worker_id})
                except RuntimeError as e:
                    log(str(e))
                    time.sleep(30)
                    break
                if unit is None:
                    break
                inflight[units.submit(process_unit, unit, a, dl_pool)] = unit
            if not inflight:
                if a.exit_when_idle:
                    log("queue empty; exiting")
                    return
                idle_since = idle_since or time.time()
                time.sleep(30)
                continue
            idle_since = None
            done, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                unit = inflight.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:  # noqa: BLE001
                    log(f"FAIL #{unit['id']} {shard_key(unit)}: {e}")
                    try:
                        api(a.coordinator, "/fail", {"id": unit["id"], "worker": worker_id, "error": repr(e)})
                    except RuntimeError as e2:
                        log(str(e2))
                    continue
                try:
                    api(a.coordinator, "/done", {"worker": worker_id, **result})
                except RuntimeError as e:
                    log(f"could not report #{unit['id']} done: {e} (lease will expire and it will be redone)")


def cmd_upload_tree(a):
    n = 0
    for root, _, files in os.walk(a.dir):
        for f in files:
            local = os.path.join(root, f)
            key = os.path.relpath(local, a.dir)
            sftp_upload(local, a.sftp, key, a.netrc)
            n += 1
    log(f"uploaded {n} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--coordinator", required=True)
    p.add_argument("--volcomp", default="volcomp", help="path to the volcomp CLI binary")
    p.add_argument("--tmp", default="/dev/shm/volcomp", help="scratch dir; tmpfs recommended (1 GiB per shard in flight)")
    p.add_argument("--sftp", help="sftp://host:port/dir destination root")
    p.add_argument("--netrc", help="netrc file with the SFTP credentials")
    p.add_argument("--local-out", help="store shards under this directory instead of uploading")
    p.add_argument("--parallel", type=int, default=4, help="shards in flight per process")
    p.add_argument("--connections", type=int, default=16, help="keep-alive S3 connections per shard in flight")
    p.add_argument("--q", type=float, default=8.0, help="fallback only; the coordinator assigns q per level")
    p.add_argument("--samples", type=int, default=8, help="chunks per shard compared against the source (0 = all)")
    p.add_argument("--exit-when-idle", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("upload-tree")
    p.add_argument("dir")
    p.add_argument("--sftp", required=True)
    p.add_argument("--netrc", required=True)
    p.set_defaults(fn=cmd_upload_tree)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
