#!/usr/bin/env python3
"""End-to-end test of the surface-prediction export, offline.

A synthetic zarr v2 prediction (blosc/zstd, exactly like the bucket's) is served
from a local directory by a tiny S3-listing HTTP server; the real coordinator and
the real worker then run against it and write the zarr v3 + volcomp tree, which is
decoded back through python/volcomp_zarr and compared with a reference computed
with numpy/scipy. Both encodings are exercised: "mask" (the default: the binary
mask stored as volcomp mask chunks) and "ramp" (the older signed-distance form).

    pytest tools/export/test_export.py          # or: python3 tools/export/test_export.py

Needs numpy, scipy, numcodecs and a built `volcomp` + `libvolcomp.so`; the tests
skip themselves when something is missing (no network is ever used).
"""
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

VOLCOMP = os.environ.get("VOLCOMP_BIN", os.path.join(ROOT, "build", "release", "volcomp"))
LIB = os.environ.get("VOLCOMP_LIB", os.path.join(ROOT, "build", "release", "libvolcomp.so"))

np = pytest.importorskip("numpy")
ndimage = pytest.importorskip("scipy.ndimage")
numcodecs = pytest.importorskip("numcodecs")
if not os.path.exists(VOLCOMP):
    pytest.skip(f"no volcomp binary at {VOLCOMP}", allow_module_level=True)
os.environ.setdefault("VOLCOMP_LIB", LIB)
vz = pytest.importorskip("volcomp_zarr._lib")

SCROLL = "PHercTest"
STAMP = "20260101000000"
PRED = f"{STAMP}-surface-20260101000001-surface-m7-L0-th0.2.zarr"
PRED_KEY = f"{SCROLL}/representations/predictions/surfaces/{PRED}/"
VOL_KEY = f"{SCROLL}/volumes/{STAMP}-2.400um-0.2m-78keV-masked.zarr/"
SHAPE = (200, 180, 160)
CSIZE = 64

# --------------------------------------------------------------- the reference


def synthetic_mask(shape=SHAPE):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    m = np.zeros(shape, bool)
    m |= ((x + z // 3) % 29) < 2                       # a stack of slanted sheets
    m |= (z > 150) & (z < 156) & (y > 30)              # one thick sheet
    m |= (z - 60) ** 2 + (y - 90) ** 2 + (x - 80) ** 2 < 25 ** 2  # a blob
    m[:, :, :4] = False
    return np.where(m, 255, 0).astype(np.uint8)


def reference_ramp(mask, dmax=3):
    """The published transform, written with scipy: v = round(127.5 + 42.5 * s).
    Everything outside the published array counts as air, exactly as the exporter
    treats it, so the volume is padded with dmax zero voxels first."""
    m = np.pad(mask != 0, dmax)
    din = ndimage.distance_transform_edt(m)        # inside: distance to air
    dout = ndimage.distance_transform_edt(~m)      # outside: distance to the mask
    s = np.where(m, np.minimum(din, dmax), -np.minimum(dout, dmax))
    v = np.floor(127.5 + (127.5 / dmax) * s + 0.5).clip(0, 255).astype(np.uint8)
    return v[dmax:-dmax, dmax:-dmax, dmax:-dmax]


def pool2(a):
    """Exact 2x mean pooling with the edge averaged over what exists."""
    out = np.zeros([(n + 1) // 2 for n in a.shape], np.uint8)
    for z in range(out.shape[0]):
        for y in range(out.shape[1]):
            blk = a[2 * z:2 * z + 2, 2 * y:2 * y + 2, :]
            sub = blk.reshape(-1, blk.shape[2]).astype(np.uint32)
            n0 = sub.shape[0]
            even = sub[:, : (a.shape[2] // 2) * 2].reshape(n0, -1, 2).sum(axis=(0, 2))
            cnt = np.full(even.shape, n0 * 2, np.uint32)
            if a.shape[2] % 2:
                even = np.append(even, sub[:, -1].sum())
                cnt = np.append(cnt, n0)
            out[z, y] = ((even + cnt // 2) // cnt).astype(np.uint8)
    return out


def test_ramp_formula():
    """The spec's numbers, on a sphere: 170/213/255 inside and 85/43/0 outside."""
    m = np.zeros((40, 40, 40), np.uint8)
    z, y, x = np.ogrid[:40, :40, :40]
    m[(z - 20) ** 2 + (y - 20) ** 2 + (x - 20) ** 2 < 12 ** 2] = 255
    r = reference_ramp(m)
    # along +x from the centre the boundary is flat enough that distances are integers
    line = r[20, 20]
    inside = np.where(m[20, 20] != 0)[0]
    edge = inside.max()
    assert (line[edge], line[edge - 1], line[edge - 2]) == (170, 213, 255)
    assert (line[edge + 1], line[edge + 2], line[edge + 3]) == (85, 43, 0)
    assert (int(line[edge]) + int(line[edge + 1]) + 1) // 2 == 128


# ------------------------------------------------------ a local S3-ish server


class S3Handler(BaseHTTPRequestHandler):
    root = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        if q.get("list-type") == ["2"]:
            return self._list(q.get("prefix", [""])[0], q.get("delimiter", [""])[0])
        path = os.path.join(self.root, urllib.parse.unquote(u.path.lstrip("/")))
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _list(self, prefix, delimiter):
        keys = []
        base = self.root
        for dirpath, _, files in os.walk(base):
            for f in files:
                k = os.path.relpath(os.path.join(dirpath, f), base)
                if k.startswith(prefix):
                    keys.append(k)
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
               "<IsTruncated>false</IsTruncated>"]
        if delimiter:
            common = sorted({prefix + k[len(prefix):].split("/")[0] + "/" for k in keys if "/" in k[len(prefix):]})
            out += [f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in common]
            out += [f"<Contents><Key>{k}</Key></Contents>" for k in keys if "/" not in k[len(prefix):]]
        else:
            out += [f"<Contents><Key>{k}</Key></Contents>" for k in sorted(keys)]
        out.append("</ListBucketResult>")
        body = "".join(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(root):
    S3Handler.root = root
    srv = ThreadingHTTPServer(("127.0.0.1", 0), S3Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def write_source(root, mask):
    """The synthetic prediction, byte for byte in the bucket's layout."""
    d = os.path.join(root, PRED_KEY)
    os.makedirs(os.path.join(d, "0"), exist_ok=True)
    codec = numcodecs.Blosc(cname="zstd", clevel=1, shuffle=1)
    zarray = {"chunks": [CSIZE] * 3, "compressor": {"id": "blosc", "cname": "zstd", "clevel": 1,
                                                    "shuffle": 1, "blocksize": 0},
              "dtype": "|u1", "fill_value": 0, "filters": None, "order": "C",
              "shape": list(mask.shape), "zarr_format": 2, "dimension_separator": "/"}
    json.dump(zarray, open(os.path.join(d, "0", ".zarray"), "w"))
    json.dump({"zarr_format": 2}, open(os.path.join(d, ".zgroup"), "w"))
    json.dump({"multiscales": [{"version": "0.4", "axes": [{"name": n, "type": "space"} for n in "zyx"],
                                "datasets": [{"path": "0", "coordinateTransformations":
                                              [{"type": "scale", "scale": [1.0] * 3}]}]}]},
              open(os.path.join(d, ".zattrs"), "w"))
    g = [math.ceil(n / CSIZE) for n in mask.shape]
    for cz in range(g[0]):
        for cy in range(g[1]):
            for cx in range(g[2]):
                blk = np.zeros((CSIZE,) * 3, np.uint8)
                sub = mask[cz * CSIZE:(cz + 1) * CSIZE, cy * CSIZE:(cy + 1) * CSIZE, cx * CSIZE:(cx + 1) * CSIZE]
                blk[: sub.shape[0], : sub.shape[1], : sub.shape[2]] = sub
                if not blk.any():
                    continue  # absent chunk, as in the bucket
                p = os.path.join(d, "0", str(cz), str(cy))
                os.makedirs(p, exist_ok=True)
                open(os.path.join(p, str(cx)), "wb").write(codec.encode(blk.tobytes()))
    # the CT volume the prediction was made from: only its level-0 .zarray is read
    v = os.path.join(root, VOL_KEY, "0")
    os.makedirs(v, exist_ok=True)
    json.dump({"chunks": [128] * 3, "compressor": None, "dtype": "|u1", "fill_value": 0,
               "filters": None, "order": "C", "shape": list(mask.shape), "zarr_format": 2,
               "dimension_separator": "/"}, open(os.path.join(v, ".zarray"), "w"))


def decode_shard(path, count):
    """Decode one shard image into {(z,y,x): 128^3 array}."""
    img = open(path, "rb").read()
    idxn = count * 16 + 4
    idx = img[len(img) - idxn:]
    out = {}
    g = round(count ** (1 / 3))
    for i in range(count):
        off = int.from_bytes(idx[i * 16:i * 16 + 8], "little")
        nb = int.from_bytes(idx[i * 16 + 8:i * 16 + 16], "little")
        if off == 2 ** 64 - 1:
            continue
        a = np.frombuffer(bytes(vz.decode(img[off:off + nb])), np.uint8).reshape(128, 128, 128)
        out[(i // (g * g), (i // g) % g, i % g)] = a
    return out


def assemble(shards, shape, shard_dim):
    vol = np.zeros(shape, np.uint8)
    g = shard_dim // 128
    for (z, y, x), a in shards.items():
        o = (z * 128, y * 128, x * 128)
        sl = tuple(slice(o[d], min(o[d] + 128, shape[d])) for d in range(3))
        vol[sl] = a[: sl[0].stop - sl[0].start, : sl[1].stop - sl[1].start, : sl[2].stop - sl[2].start]
    assert g >= 1
    return vol


def run_export(tmp, encoding):
    """Run the whole pipeline once: manifest -> metadata -> serve -> worker -> pool-levels."""
    src_root, out, meta = str(tmp / "bucket"), str(tmp / "out"), str(tmp / "meta")
    mask = synthetic_mask()
    write_source(src_root, mask)
    srv, url = serve(src_root)
    env = dict(os.environ, VOLCOMP_S3_ENDPOINT=url)
    db = str(tmp / "e.db")
    coord = [sys.executable, os.path.join(HERE, "coordinator.py")]
    subprocess.run(coord + ["manifest-surfaces", "--db", db, "--encoding", encoding], env=env, check=True)
    subprocess.run(coord + ["metadata", "--db", db, "--out", meta], env=env, check=True)
    port = 8791
    for _ in range(20):  # a free port for the coordinator
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)):
                break
        port += 1
    serve_proc = subprocess.Popen(coord + ["serve", "--db", db, "--bind", "127.0.0.1", "--port", str(port)], env=env)
    try:
        for _ in range(100):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        r = subprocess.run([sys.executable, os.path.join(HERE, "worker.py"), "run",
                            "--coordinator", f"http://127.0.0.1:{port}", "--volcomp", VOLCOMP,
                            "--tmp", str(tmp / "scratch"), "--local-out", out, "--exit-when-idle",
                            "--parallel", "1", "--threads", "2"],
                           env=env, check=True, capture_output=True, text=True)
        print(r.stdout)
    finally:
        serve_proc.terminate()
        serve_proc.wait()
        srv.shutdown()
    subprocess.run(coord + ["pool-levels", "--db", db, "--volcomp", VOLCOMP, "--src", out,
                            "--out", out, "--tmp", str(tmp / "pooltmp")], env=env, check=True)
    for root, _, files in os.walk(meta):  # the metadata tree ships alongside the shards
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), meta)
            os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
            shutil.copyfile(os.path.join(root, f), os.path.join(out, rel))
    return {"out": out, "mask": mask, "meta": meta}


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return run_export(tmp_path_factory.mktemp("export_mask"), "mask")


@pytest.fixture(scope="module")
def export_ramp(tmp_path_factory):
    return run_export(tmp_path_factory.mktemp("export_ramp"), "ramp")


# ----------------------------------------------------------- the mask encoding


def majority_pool(a, pad_to):
    """The 2x2x2 majority pool a mask chunk stores: `a` is zero padded out to the
    128^3 chunk grid first, exactly as the exporter pads it."""
    p = np.zeros(pad_to, np.uint8)
    p[: a.shape[0], : a.shape[1], : a.shape[2]] = a != 0
    c = sum(p[dz::2, dy::2, dx::2].astype(np.uint16)
            for dz in range(2) for dy in range(2) for dx in range(2))
    return np.where(c * 2 >= 8, 255, 0).astype(np.uint8)


def chunk_grid_shape(shape):
    return tuple(math.ceil(n / 128) * 128 for n in shape)


def test_mask_tree_and_metadata(export):
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    group = json.load(open(os.path.join(root, "zarr.json")))
    ex = group["attributes"]["volcomp"]
    assert ex["encoding"]["name"] == "surface-mask"
    assert all(lv["encoding"] == "mask" for lv in ex["levels"])
    assert all(lv["q"] == 0 for lv in ex["levels"])
    for lv in ex["levels"]:
        md = json.load(open(os.path.join(root, lv["path"], "zarr.json")))
        assert md["codecs"][0]["configuration"]["codecs"][0] == {
            "name": "volcomp", "configuration": {"mode": "mask", "q": 0}}
        assert md["attributes"]["volcomp"]["encoding"] == "mask"


def test_mask_level0_is_the_published_mask(export):
    """Every stored block centre is the 2x2x2 majority of the published mask, and
    the decode is the interpolation of that grid (nine possible values)."""
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    got = assemble(decode_shard(os.path.join(root, "2.4", "c", "0", "0", "0"), 512), SHAPE, 1024)
    grid = majority_pool(export["mask"], chunk_grid_shape(SHAPE))
    h = [(n + 1) // 2 for n in SHAPE]
    assert np.array_equal(got[0::2, 0::2, 0::2], grid[: h[0], : h[1], : h[2]])
    assert set(np.unique(got)) <= {0, 32, 64, 96, 128, 159, 191, 223, 255}
    # the field crosses 128 where the mask edge is: the thresholded decode is the
    # mask to within the 2-voxel pooling
    assert np.mean((got >= 128) != (export["mask"] != 0)) < 0.12


def test_mask_pyramid_is_majority_pooling(export):
    """Each rung's stored grid is the 2x2x2 majority pool of the rung below's."""
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    shape = list(SHAPE)
    ref = export["mask"]
    for path, shard in (("2.4", 1024), ("4.8", 512), ("9.6", 256), ("19.2", 128)):
        count = (shard // 128) ** 3
        got = assemble(decode_shard(os.path.join(root, path, "c", "0", "0", "0"), count), shape, shard)
        grid = majority_pool(ref, chunk_grid_shape(shape))
        h = [(n + 1) // 2 for n in shape]
        assert np.array_equal(got[0::2, 0::2, 0::2], grid[: h[0], : h[1], : h[2]]), path
        # the next rung's mask IS this rung's stored grid
        shape = [(n + 1) // 2 for n in shape]
        ref = grid[: shape[0], : shape[1], : shape[2]]


def test_mask_pool_levels_and_size(export):
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    assert os.path.exists(os.path.join(root, "38.4", "c", "0", "0", "0"))
    assert os.path.exists(os.path.join(root, "307.2", "c", "0", "0", "0"))
    # a majority pool can pool a thin sheet away entirely: this synthetic volume is
    # 480 um across, so the top rungs are empty and their shards are simply absent
    # (a missing shard key is the fill value, exactly as for an air region)
    assert not os.path.exists(os.path.join(root, "1228.8", "c", "0", "0", "0"))
    # a mask level is far smaller than the ramp it replaces
    n = os.path.getsize(os.path.join(root, "2.4", "c", "0", "0", "0"))
    assert n < 0.05 * export["mask"].size, n


# ----------------------------------------------------------- the ramp encoding


def test_tree_and_metadata(export_ramp):
    export = export_ramp
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    group = json.load(open(os.path.join(root, "zarr.json")))
    ms = group["attributes"]["ome"]["multiscales"][0]
    paths = [d["path"] for d in ms["datasets"]]
    assert paths == ["2.4", "4.8", "9.6", "19.2", "38.4", "76.8", "153.6", "307.2", "614.4", "1228.8"]
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [2.4, 2.4, 2.4]
    assert ms["datasets"][3]["coordinateTransformations"][0]["scale"] == [19.2, 19.2, 19.2]
    ex = group["attributes"]["volcomp"]
    assert ex["encoding"]["name"] == "surface-ramp" and ex["encoding"]["dmax"] == 3
    assert ex["threshold"] == 0.2
    assert ex["native_voxel_size_um"] == 2.4 and ex["rung_voxel_size_um"] == 2.4
    assert ex["resample_scale"] == 1.0
    assert ex["shape"] == list(SHAPE)
    qs = {lv["path"]: lv["q"] for lv in ex["levels"]}
    assert qs["2.4"] == 8 and qs["4.8"] == 4 and qs["9.6"] == 2 and qs["19.2"] == 1 and qs["38.4"] == 0
    assert qs["1228.8"] == 0  # the top of the ladder fits one chunk and is stored losslessly
    # every level has an array metadata document with the right shard shape
    for lv in ex["levels"]:
        md = json.load(open(os.path.join(root, lv["path"], "zarr.json")))
        assert md["chunk_grid"]["configuration"]["chunk_shape"] == [lv["shard"]] * 3
        assert md["codecs"][0]["configuration"]["codecs"][0] == {"name": "volcomp",
                                                                 "configuration": {"q": lv["q"]}}
        assert md["shape"] == lv["shape"]
    assert [lv["shard"] for lv in ex["levels"]][:5] == [1024, 512, 256, 128, 128]


def test_level0_matches_the_ramp(export_ramp):
    export = export_ramp
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    want = reference_ramp(export["mask"])
    got = assemble(decode_shard(os.path.join(root, "2.4", "c", "0", "0", "0"), 512), SHAPE, 1024)
    err = np.abs(got.astype(int) - want.astype(int))
    # q = 8 at 2.4 um: P99 ~ 2.5 q as everywhere in volcomp, with DCT ringing on the
    # ramp's 85-wide steps in the tail
    assert np.percentile(err, 99) <= 2.5 * 8, np.percentile(err, 99)
    assert err.mean() < 2.5, err.mean()
    # the shape of the data is what matters: the 128 crossing must not move
    assert np.mean((got >= 128) != (want >= 128)) < 0.01


def test_pyramid_is_mean_pooling(export_ramp):
    export = export_ramp
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    ref = reference_ramp(export["mask"])
    shape = list(SHAPE)
    for path, shard, q in (("4.8", 512, 4), ("9.6", 256, 2), ("19.2", 128, 1)):
        ref = pool2(ref)
        shape = [(n + 1) // 2 for n in shape]
        count = (shard // 128) ** 3
        got = assemble(decode_shard(os.path.join(root, path, "c", "0", "0", "0"), count), shape, shard)
        # the exporter pools the exact ramp, not the decoded level below, so the only
        # error here is this level's own quantiser
        err = np.abs(got.astype(int) - ref.astype(int))
        assert np.percentile(err, 99) <= 2.5 * q + 6, (path, np.percentile(err, 99))
        assert err.mean() < 2.0, (path, err.mean())


def test_pool_levels_built_the_coarse_ladder(export_ramp):
    export = export_ramp
    root = os.path.join(export["out"], PRED_KEY.rstrip("/"))
    group = json.load(open(os.path.join(root, "zarr.json")))
    ref = reference_ramp(export["mask"])
    for _ in range(4):
        ref = pool2(ref)          # 4.8, 9.6, 19.2, 38.4
    levels = group["attributes"]["volcomp"]["levels"]
    p38 = os.path.join(root, "38.4", "c", "0", "0", "0")
    assert os.path.exists(p38), "pool-levels did not write the 38.4 um level"
    got = assemble(decode_shard(p38, 1), levels[4]["shape"], 128)
    # pool-levels pools the DECODED level below, so this carries the 19.2 um q = 1 error too
    assert np.abs(got.astype(int) - ref.astype(int)).mean() < 3.0
    # and it goes all the way to the top of the ladder
    assert os.path.exists(os.path.join(root, "1228.8", "c", "0", "0", "0"))


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v", "-x"]))
