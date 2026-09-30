"""Reference checksums for the viewer: fetch one inner chunk of a volcomp zarr v3
level over HTTP (shard index + Range read, like the viewer) and decode it with
python/volcomp_zarr (libvolcomp), or decode local .volc files.

  python3 web/test/ref_checksums.py url LEVEL_URL CZ CY CX     # sha256 of plain / CT-smooth / pred-smooth decodes
  python3 web/test/ref_checksums.py files a.volc b.volc ...    # same line format as wasm_check.mjs
"""
import hashlib, json, os, struct, subprocess, sys, urllib.error, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
from volcomp_zarr import _lib as L  # noqa: E402


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


_metas = {}


def level_meta(level_url):
    if level_url not in _metas:
        _metas[level_url] = json.loads(get(level_url.rstrip("/") + "/zarr.json"))
    return _metas[level_url]


def fetch_chunk(level_url, c, shard_url=None):
    """The volcomp stream of inner chunk c (global chunk coords) of a sharded level,
    or None when the shard (404) or the chunk (index all ones) is absent. A zstd
    bytes codec after volcomp is undone with the zstd CLI."""
    meta = level_meta(level_url)
    shard = meta["chunk_grid"]["configuration"]["chunk_shape"]
    sh = next(x for x in meta["codecs"] if x["name"] == "sharding_indexed")["configuration"]
    inner = sh["chunk_shape"]
    cps = [s // i for s, i in zip(shard, inner)]
    s = [ci // n for ci, n in zip(c, cps)]
    l = [ci - si * n for ci, si, n in zip(c, s, cps)]
    k = (l[0] * cps[1] + l[1]) * cps[2] + l[2]
    n = cps[0] * cps[1] * cps[2]
    url = shard_url or level_url.rstrip("/") + "/c/" + "/".join(map(str, s))
    try:
        idx = get(url, f"bytes=-{n * 16 + 4}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    off, nb = struct.unpack_from("<QQ", idx, 16 * k)
    if off == 2**64 - 1:
        return None
    b = get(url, f"bytes={off}-{off + nb - 1}")
    if any(x["name"] == "zstd" for x in sh.get("codecs", [])):
        b = subprocess.run(["zstd", "-dc"], input=b, capture_output=True, check=True).stdout
    return b


def sums(b):
    h = lambda x: hashlib.sha256(bytes(x)).hexdigest()
    return (h(L.decode(b)), h(L.decode_smooth(b, 2.0, zero_guard=True, gated=True)),
            h(L.decode_smooth(b, 0.6, zero_guard=True)))


if __name__ == "__main__":
    if sys.argv[1] == "url":
        b = fetch_chunk(sys.argv[2], [int(x) for x in sys.argv[3:6]])
        if b is None:
            print(json.dumps({"missing": "chunk"}))
        else:
            p, c, q = sums(b)
            print(json.dumps({"nbytes": len(b), "mode": b[5], "plain": p, "ct_smooth": c, "pred_smooth": q}))
    else:
        for f in sys.argv[2:]:
            print(os.path.basename(f), *(x[:16] for x in sums(open(f, "rb").read())))
