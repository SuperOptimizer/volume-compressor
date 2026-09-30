"""Reference compositor for the viewer's layer stack: fetches the chunks a tile
needs over HTTP (shard index + Range, like the viewer), decodes them with
python/volcomp_zarr (libvolcomp), resamples every layer into the tile in world
micrometres and blends them, all in integer arithmetic, and prints the sha256 of
the RGBA bytes. It computes each layer's pitch itself from the store metadata
(web/README.md, "Geometry"), so it checks the viewer's geometry as well as its
resampling and blending.

  python3 web/test/ref_composite.py spec.json
spec: {"axis": 0|1|2, "s": um, "u0": um, "v0": um, "du": um, "w": px, "h": px,
       "layers": [{"url": store, "level": i, "opts": "<the viewer's layer opts string>"}]}
"""
import hashlib, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_checksums import fetch_chunk, get, level_meta  # noqa: E402
from volcomp_zarr import _lib as VL  # noqa: E402  (ref_checksums put python/ on sys.path)

UV = [[2, 1], [2, 0], [1, 0]]
HUES = {"red": [255, 64, 64], "green": [80, 230, 90], "blue": [80, 140, 255], "cyan": [40, 220, 230],
        "magenta": [240, 80, 230], "yellow": [255, 225, 60], "orange": [255, 150, 40]}


def name_pitch(name):
    m = re.search(r"(\d+(?:\.\d+)?)\s*um(?![a-z])", name, re.I)
    return float(m.group(1)) if m else None


def geometry(url):
    """[(level_url, pitch[3], origin[3], shape[3])] per level."""
    root = json.loads(get(url.rstrip("/") + "/zarr.json"))
    a = root.get("attributes", {})
    vc = a.get("volcomp", {})
    ms = (a.get("ome") or {}).get("multiscales") or a.get("multiscales")
    m = ms[0]
    ds = []
    for d in m["datasets"]:
        ts = d.get("coordinateTransformations", [])
        assert not any(t["type"] == "translation" for t in ts), "translations: not needed by these tests"
        ds.append((d["path"], next(t["scale"] for t in ts if t["type"] == "scale")[-3:]))
    s0 = ds[0][1]
    name = url.rstrip("/").split("/")[-1]
    if vc.get("source_volume") and isinstance(vc.get("source_level"), int):
        p = name_pitch(vc["source_volume"].rstrip("/").split("/")[-1]) * 2 ** vc["source_level"]
        p0 = [p, p, p]
    elif all(str(x.get("unit", "")).lower() in ("micrometer", "micron", "um", "µm") for x in m["axes"][-3:]):
        p0 = list(s0)
    else:
        p = name_pitch(name)
        p0 = [p, p, p]
    out = []
    for path, sc in ds:
        lu = url.rstrip("/") + "/" + path
        meta = level_meta(lu)
        out.append((lu, [p0[i] * sc[i] / s0[i] for i in range(3)], [0.0, 0.0, 0.0], meta["shape"]))
    return out


def parse_opts(s):
    o = {"lut": "grey", "op": 1.0, "w": 255, "lv": 128, "t": None, "bl": "normal", "ip": "auto", "k": "ct", "hue": "red"}
    for kv in s.split(","):
        k, _, v = kv.partition(":")
        if k in ("op",):
            o[k] = float(v)
        elif k in ("w", "lv"):
            o[k] = int(v)
        elif k == "t":
            o[k] = None if v == "-" else int(v)
        elif k:
            o[k] = v
    return o


def hsl(h, s, l):
    def k(n): return (n + h / 30) % 12
    a = s * min(l, 1 - l)
    def f(n): return l - a * max(-1, min(k(n) - 3, 9 - k(n), 1))
    return [math.floor(x * 255 + 0.5) for x in (f(0), f(8), f(4))]


def lut(o):
    lo = o["lv"] - o["w"] / 2
    win = max(1, o["w"])
    O = math.floor(o["op"] * 255 + 0.5)
    out = []
    for v in range(256):
        w = min(255, max(0, math.floor(((v - lo) / win) * 255 + 0.5)))
        L = o["lut"]
        if L == "grey":
            c, a = [w, w, w], 255
        elif L == "categorical":
            c, a = (HUES[o["hue"]] if v == 255 else hsl((v * 137.50776) % 360, 0.75, 0.58)), (255 if v else 0)
        elif L == "fire":
            c, a = [min(255, 3 * w), min(255, max(0, 3 * w - 255)), min(255, max(0, 3 * w - 510))], w
        else:
            c, a = HUES[L], w
        if o["t"] is not None:
            a = ((255 if v else 0) if L == "categorical" else 255) if v >= o["t"] else 0
        out.append((a * O, c))
    return out


def axis_samples(w0, d, N, org, pit, n, lin):
    out = []
    for i in range(N):
        c = (w0 + (i + 0.5) * d - org) / pit
        if not (0 <= c < n):
            out.append(None)
            continue
        if lin:
            t = c - 0.5
            a = math.floor(t)
            f = math.floor((t - a) * 256 + 0.5)
            if f == 256:
                a, f = a + 1, 0
            b = a + 1
            a, b = min(max(a, 0), n - 1), min(max(b, 0), n - 1)
            if a == b:
                f = 0
        else:
            a = b = math.floor(c)
            f = 0
        out.append((a, b, f))
    return out


class Vol:
    def __init__(self, level_url, shape):
        self.url, self.shape, self.chunks = level_url, shape, {}

    def __call__(self, z, y, x):
        c = (z >> 7, y >> 7, x >> 7)
        if c not in self.chunks:
            b = fetch_chunk(self.url, list(c))
            self.chunks[c] = None if b is None else bytes(VL.decode(b))
        a = self.chunks[c]
        return 0 if a is None else a[((z & 127) * 128 + (y & 127)) * 128 + (x & 127)]


def main(spec):
    ax, W, H = spec["axis"], spec["w"], spec["h"]
    ua, va = UV[ax]
    R, G, B = [0] * (W * H), [0] * (W * H), [0] * (W * H)
    info = []
    for lay in spec["layers"]:
        o = parse_opts(lay["opts"])
        if o.get("h") == "1":
            continue
        lu, pit, org, shape = geometry(lay["url"])[lay["level"]]
        info.append({"level_url": lu, "pitch": pit})
        V = Vol(lu, shape)
        lin = (o["ip"] == "linear") or (o["ip"] == "auto" and not (o["k"] in ("mask", "label") or o["lut"] == "categorical"))
        cs = (spec["s"] - org[ax]) / pit[ax]
        if not (0 <= cs < shape[ax]):
            continue
        s0, s1, ws = slice_samples(cs, shape[ax], lin)
        cols = axis_samples(spec["u0"], spec["du"], W, org[ua], pit[ua], shape[ua], lin)
        rows = axis_samples(spec["v0"], spec["du"], H, org[va], pit[va], shape[va], lin)

        def vox(s, yv, xu):
            p = [0, 0, 0]
            p[ax], p[va], p[ua] = s, yv, xu
            return V(*p)

        def plane(yv, xu):
            if not lin:
                return vox(s0, yv, xu)
            if ws == 0:
                return vox(s0, yv, xu) * 256
            return vox(s0, yv, xu) * (256 - ws) + vox(s1, yv, xu) * ws
        L = lut(o)
        bl = o["bl"]
        for j in range(H):
            r = rows[j]
            if r is None:
                continue
            ya, yb, wy = r
            for i in range(W):
                c = cols[i]
                if c is None:
                    continue
                xa, xb, wx = c
                if lin:
                    p00 = plane(ya, xa)
                    p01 = plane(ya, xb) if wx else 0
                    p10 = plane(yb, xa) if wy else 0
                    p11 = plane(yb, xb) if wx and wy else 0
                    v = ((p00 * (256 - wx) + p01 * wx) * (256 - wy) + (p10 * (256 - wx) + p11 * wx) * wy + 8388608) // 16777216
                else:
                    v = plane(ya, xa)
                A, col = L[v]
                if A == 0:
                    continue
                k = j * W + i
                for D, ch in ((R, 0), (G, 1), (B, 2)):
                    if bl == "normal":
                        D[k] = (D[k] * (65025 - A) + col[ch] * A + 32512) // 65025
                    elif bl == "add":
                        D[k] = min(255, D[k] + (col[ch] * A + 32512) // 65025)
                    else:
                        D[k] = max(D[k], (col[ch] * A + 32512) // 65025)
    rgba = bytearray(W * H * 4)
    coloured = 0
    for k in range(W * H):
        rgba[4 * k:4 * k + 4] = bytes((R[k], G[k], B[k], 255))
        coloured += not (R[k] == G[k] == B[k])
    return {"sha256": hashlib.sha256(rgba).hexdigest(), "coloured": coloured, "layers": info,
            "rgbaHead": list(rgba[:16])}


def slice_samples(cs, n, lin):
    if not lin:
        a = math.floor(cs)
        return a, a, 0
    t = cs - 0.5
    a = math.floor(t)
    f = math.floor((t - a) * 256 + 0.5)
    if f == 256:
        a, f = a + 1, 0
    b = a + 1
    a, b = min(max(a, 0), n - 1), min(max(b, 0), n - 1)
    if a == b:
        f = 0
    return a, b, f


if __name__ == "__main__":
    print(json.dumps(main(json.load(open(sys.argv[1])))))
