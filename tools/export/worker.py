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
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
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
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
            log(f"coordinator unreachable ({e}); retrying")
            time.sleep(min(60, 2 + attempt * 2))
    raise RuntimeError("coordinator unreachable")


# ----------------------------------------------------------------------------- S3 download


class ChunkFetchError(Exception):
    pass


def fetch_chunk(key, dest, retries=8):
    """GET one source chunk. Returns True if stored, False if the key does not exist
    (masked region). Anything else retries, then raises: a transient failure must
    never be mistaken for a missing chunk."""
    url = BUCKET + "/" + key
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            if len(data) != CHUNK_BYTES:
                raise ChunkFetchError(f"{key}: {len(data)} bytes, expected {CHUNK_BYTES}")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError, ChunkFetchError) as e:
            err = repr(e)
        time.sleep(min(30, 1.5 ** attempt))
    raise ChunkFetchError(f"{key}: giving up after {retries} attempts ({err})")


def download_shard(unit, workdir, pool):
    """Fetch the (up to) 512 chunks of one shard into workdir/z_y_x.u8.
    Chunks outside the level's chunk grid are skipped; missing keys are masked air."""
    vol, lvl = unit["volume"], unit["level"]
    sz, sy, sx = unit["shard"]
    grid = [(n + 127) // 128 for n in unit["shape"]]
    jobs = []
    for z in range(SHARD_CHUNKS):
        for y in range(SHARD_CHUNKS):
            for x in range(SHARD_CHUNKS):
                cz, cy, cx = sz * 8 + z, sy * 8 + y, sx * 8 + x
                if cz >= grid[0] or cy >= grid[1] or cx >= grid[2]:
                    continue
                key = f"{vol}{lvl}/{cz}/{cy}/{cx}"
                jobs.append(pool.submit(fetch_chunk, key, os.path.join(workdir, f"{z}_{y}_{x}.u8")))
    present = 0
    for j in jobs:
        present += bool(j.result())  # raises ChunkFetchError on a hard failure
    return len(jobs), present


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


def sftp_upload(local, sftp_root, key, netrc):
    """Upload to <key>.part, then rename into place (readers never see a partial shard)."""
    base = sftp_root.rstrip("/")
    url = f"{base}/{key}.part"
    m = re.match(r"(sftp://[^/]+)(/.*)?$", base)
    if not m:
        raise RuntimeError("--sftp must look like sftp://host:port/dir")
    path_prefix = (m.group(2) or "")
    remote_part = f"{path_prefix}/{key}.part"
    remote_final = f"{path_prefix}/{key}"
    run(["curl", "-sS", "--netrc-file", netrc, "--insecure", "--ftp-create-dirs", "-T", local, url,
         "-Q", f"-rm {remote_final}", "-Q", f"rename {remote_part} {remote_final}"])


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
        info = {"present": 0, "payload": 0}
        if present:
            info = parse_kv(run([a.volcomp, "shard-pack", workdir, shard, f"--q={a.q}"]))
            v = parse_kv(run([a.volcomp, "shard-verify", shard, workdir, f"--samples={a.samples}"]))
            info.update(v)
        else:
            # every chunk is masked/absent: still write a shard so readers get an all-missing index
            info = parse_kv(run([a.volcomp, "shard-pack", workdir, shard, f"--q={a.q}"]))
        t2 = time.time()
        size = os.path.getsize(shard)
        key = shard_key(unit)
        if a.local_out:
            local_store(shard, a.local_out, key)
        else:
            sftp_upload(shard, a.sftp, key, a.netrc)
        t3 = time.time()
        log(f"done #{unit['id']} {key}: {present}/{n_jobs} chunks, {size / 1e6:.1f} MB, "
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
                unit = api(a.coordinator, "/claim", {"worker": worker_id})
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
                    api(a.coordinator, "/done", {"worker": worker_id, **fut.result()})
                except Exception as e:  # noqa: BLE001
                    log(f"FAIL #{unit['id']} {shard_key(unit)}: {e}")
                    api(a.coordinator, "/fail", {"id": unit["id"], "worker": worker_id, "error": repr(e)})


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
    p.add_argument("--connections", type=int, default=32, help="S3 connections per shard in flight")
    p.add_argument("--q", type=float, default=8.0)
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
