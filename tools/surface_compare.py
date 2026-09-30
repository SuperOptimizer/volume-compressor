#!/usr/bin/env python3
"""Our volcomp m7 surface prediction against the PUBLISHED upstream m7 prediction of the same scan: slice
images, disagreement panels and dice/precision/recall, for docs/comparison/surfaces/<Sample>/.

Sources (all fetched over anonymous HTTPS and cached under --cache):
  ours       https://dl.ash2txt.org/community-uploads/forrest/volcomp/<Sample>/representations/predictions/surfaces/
             <scan>-surface-<run>-surface-m7-L<k>-prob.zarr   zarr v3 sharded, volcomp q8, uint8 round(p * 255)
  published  s3://vesuvius-challenge-open-data/<Sample>/representations/predictions/surfaces/
             <scan>-surface-<run>-surface-m7-L<k>-th0.2.zarr   OME-zarr v2 blosc, binary mask (p >= 0.2)
  CT         s3://vesuvius-challenge-open-data/<Sample>/volumes/<scan>-*.zarr level k (uncompressed original)

Our store is decoded three ways: plain (what any zarr reader gets), `gated2` (what
volcomp_zarr.set_read_smoothing(2) gives: decode_smooth 2.0, gated, zero guard) and `gauss0.6` (the setting
docs/deblocking.md recommends for probability maps: decode_smooth 0.6, zero guard). Images use plain and
gated2 (on the kept float box gated2 has the lowest error of the three: MAE 3.70 against 4.10 / 4.67 of 255).

Per sample it writes, into --out:
  <view>_ct.png            CT slice (grey)
  <view>_published.png     CT + published binary mask (amber, 60 % opacity)
  <view>_ours.png          CT + our probability, plain decode (amber, opacity 0.6 * p)
  <view>_ours_smooth.png   the same, gated2 decode smoothing (set_read_smoothing(2))
  <view>_diff.png          darkened CT; red = published only, blue = ours only (plain, p >= 0.5), grey = both,
                           dark red = published surface in masked air (CT == 0)
  <view>_all.png           the five panels side by side at half resolution, labelled
  <view>_zoom.png          a 256^2 crop of the five panels (the most surface-dense one), 2x nearest, labelled
  report.json, README.md

<view> is z/y/x plus the box-local index; slices are at 1/3 and 2/3 of the box on each axis (not picked).
The box is --box z0,y0,x0,dz,dy,dx at the prediction's level-0 grid, or `auto`: the --size^3 window with the
most published surface voxels, searched on the published pyramid.

    VOLCOMP_LIB=build/release/libvolcomp.so PYTHONPATH=python python3 tools/surface_compare.py \
        --sample PHerc0343P --scan 20260304131111 --box auto --out docs/comparison/surfaces/PHerc0343P \
        [--whole] [--float-keep /vesuvius/tmp_cmp0343P_keep]

--whole adds whole-volume dice/precision/recall (streams every level-0 shard; downloads both level-0 arrays).
--float-keep adds the compression-only check on the kept PHerc0343P 8.64 um box (a DIFFERENT scan: our q8
store against the uncompressed float m7 blend of the same run).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from volcomp_zarr import _lib as vc  # noqa: E402

OPEN = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
OURS = "https://dl.ash2txt.org/community-uploads/forrest/volcomp"
CH = 128
EMPTY = (1 << 64) - 1
MODES = ("plain", "gated2", "gauss0.6")
AMBER = np.array([255.0, 190.0, 0.0])
RED, BLUE, BOTH = np.array([255.0, 40, 40]), np.array([60.0, 140, 255]), np.array([205.0, 205, 205])
AIR = np.array([95.0, 25, 25])  # published surface where the CT is 0 (masked air)
T50, T20 = 128, 51  # uint8 thresholds for p >= 0.5 and p >= 0.2 (51 / 255 = 0.2)


# --------------------------------------------------------------------------- HTTP with a disk cache
def get(url, cache=None, tries=6):
    """GET url -> bytes, None on 404. Cached under `cache` by URL hash (404s too)."""
    p = None
    if cache:
        h = hashlib.sha1(url.encode()).hexdigest()
        p = os.path.join(cache, h[:2], h)
        if os.path.exists(p):
            return open(p, "rb").read()
        if os.path.exists(p + ".404"):
            return None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                b = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                if p:
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    open(p + ".404", "w").close()
                return None
            if i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(2 ** i)
    if p:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.{threading.get_ident()}.tmp"
        open(tmp, "wb").write(b)
        os.replace(tmp, p)
    return b


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    for i in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0
        except Exception:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"HEAD failed {url}")


def s3_list(prefix, delimiter=None):
    """(list of (key, size), list of common prefixes) under prefix in the open-data bucket."""
    keys, prefs, tok = [], [], None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if tok:
            q["continuation-token"] = tok
        url = OPEN + "/?" + urllib.parse.urlencode(q)
        root = ET.fromstring(get(url))
        keys += [(c.find(ns + "Key").text, int(c.find(ns + "Size").text)) for c in root.findall(ns + "Contents")]
        prefs += [c.find(ns + "Prefix").text for c in root.findall(ns + "CommonPrefixes")]
        t = root.find(ns + "NextContinuationToken")
        if t is None:
            return keys, prefs
        tok = t.text


def http_index(url):
    return re.findall(r'href="([^"?/][^"]*)"', get(url).decode())


# --------------------------------------------------------------------------- arrays
class V2Array:
    """OME-zarr v2 array (blosc or uncompressed) over HTTP; missing chunks are fill 0."""

    def __init__(self, base, cache):
        import numcodecs
        self.base, self.cache = base.rstrip("/"), cache
        m = json.loads(get(self.base + "/.zarray", cache))
        self.shape, self.chunks = np.array(m["shape"]), np.array(m["chunks"])
        assert m["dtype"] == "|u1" and m.get("dimension_separator", ".") == "/"
        self.codec = numcodecs.get_codec(m["compressor"]) if m["compressor"] else None

    def chunk(self, key):
        b = get(self.base + "/" + "/".join(map(str, key)), self.cache)
        if b is None:
            return None
        a = np.frombuffer(self.codec.decode(b) if self.codec else b, np.uint8)
        return a.reshape(tuple(self.chunks))

    def read(self, lo, hi, pool=None):
        lo, hi = np.asarray(lo), np.minimum(np.asarray(hi), self.shape)
        out = np.zeros(tuple(hi - lo), np.uint8)
        c0, c1 = lo // self.chunks, -(-hi // self.chunks)
        keys = [(a, b, c) for a in range(c0[0], c1[0]) for b in range(c0[1], c1[1]) for c in range(c0[2], c1[2])]

        def one(k):
            ch = self.chunk(k)
            if ch is None:
                return
            g0 = np.array(k) * self.chunks
            s0, s1 = np.maximum(lo, g0), np.minimum(hi, g0 + self.chunks)
            out[tuple(slice(a - l, b - l) for a, b, l in zip(s0, s1, lo))] = \
                ch[tuple(slice(a - g, b - g) for a, b, g in zip(s0, s1, g0))]
        list((pool or _POOL).map(one, keys))
        return out


class OursArray:
    """Our zarr v3 sharded volcomp array (URL or local dir); decode modes plain / gated2 / gauss0.6."""

    def __init__(self, base, cache, meta=None):
        self.base, self.cache = base.rstrip("/"), cache
        self.local = not self.base.startswith("http")
        m = meta or json.loads(self._get("zarr.json"))
        self.shape = np.array(m["shape"])
        self.side = int(m["chunk_grid"]["configuration"]["chunk_shape"][0])
        assert m["codecs"][0]["configuration"]["chunk_shape"] == [CH] * 3
        self.n = self.side // CH
        self._shards, self._lock, self._fetching = {}, threading.Lock(), {}

    def _get(self, rel):
        if self.local:
            p = os.path.join(self.base, rel)
            return open(p, "rb").read() if os.path.exists(p) else None
        return get(self.base + "/" + rel, self.cache)

    def shard(self, key):
        with self._lock:
            if key in self._shards:
                return self._shards[key]
            kl = self._fetching.setdefault(key, threading.Lock())
        with kl:  # one fetch per shard even when several threads want it at once
            with self._lock:
                if key in self._shards:
                    return self._shards[key]
            return self._fetch(key)

    def _fetch(self, key):
        b = self._get("c/" + "/".join(map(str, key)))
        ent = None
        if b is not None:
            buf = np.frombuffer(b, np.uint8)
            n3 = self.n ** 3
            idx = np.frombuffer(b[len(b) - 16 * n3 - 4:len(b) - 4], "<u8").reshape(self.n, self.n, self.n, 2)
            ent = (buf, idx)
        with self._lock:
            self._shards[key] = ent
        return ent

    def drop(self):
        with self._lock:
            self._shards.clear()
            self._fetching.clear()

    def read(self, lo, hi, mode="plain", pool=None):
        lo, hi = np.asarray(lo), np.minimum(np.asarray(hi), self.shape)
        out = np.zeros(tuple(hi - lo), np.uint8)
        c0, c1 = lo // CH, -(-hi // CH)
        keys = [(a, b, c) for a in range(c0[0], c1[0]) for b in range(c0[1], c1[1]) for c in range(c0[2], c1[2])]

        def one(k):
            ent = self.shard(tuple(x // self.n for x in k))
            if ent is None:
                return
            buf, idx = ent
            o, nb = (int(v) for v in idx[k[0] % self.n, k[1] % self.n, k[2] % self.n])
            if o == EMPTY:
                return
            tmp = np.empty(CH ** 3, np.uint8)
            src = buf[o:o + nb]
            if mode == "plain":
                vc.decode(src, tmp)
            elif mode == "gated2":
                vc.decode_smooth(src, 2.0, tmp, zero_guard=True, gated=True)
            elif mode == "gauss0.6":
                vc.decode_smooth(src, 0.6, tmp, zero_guard=True)
            else:
                raise ValueError(mode)
            tmp = tmp.reshape(CH, CH, CH)
            g0 = np.array(k) * CH
            s0, s1 = np.maximum(lo, g0), np.minimum(hi, g0 + CH)
            out[tuple(slice(a - l, b - l) for a, b, l in zip(s0, s1, lo))] = \
                tmp[tuple(slice(a - g, b - g) for a, b, g in zip(s0, s1, g0))]
        list((pool or _POOL).map(one, keys))
        return out


_POOL = ThreadPoolExecutor(16)


# --------------------------------------------------------------------------- discovery
def discover(sample, scan, cache):
    sd = f"{OURS}/{sample}/representations/predictions/surfaces/"
    ours = sorted(h.rstrip("/") for h in http_index(sd)
                  if h.startswith(f"{scan}-surface-") and re.search(r"-m7-L\d+-prob\.zarr/$", h))
    if not ours:
        raise SystemExit(f"no finished (non-.part) m7 prob store for {sample} {scan} under {sd}")
    ours = ours[-1]
    k = int(re.search(r"-m7-L(\d+)-", ours).group(1))
    _, prefs = s3_list(f"{sample}/representations/predictions/surfaces/", "/")
    pub = [p for p in prefs if p.split("/")[-2].startswith(f"{scan}-surface-")
           and re.search(rf"-m7-L{k}-th[0-9.]+\.zarr/$", p)]
    if not pub:
        raise SystemExit(f"no published m7 L{k} mask for {sample} {scan}: {prefs}")
    _, vprefs = s3_list(f"{sample}/volumes/", "/")
    ct = [p for p in vprefs if p.split("/")[-2].startswith(scan + "-") and p.endswith(".zarr/")]
    g = json.loads(get(sd + ours + "/zarr.json", cache))
    ds = g["attributes"]["ome"]["multiscales"][0]["datasets"]
    return dict(level=k, ours_name=ours, ours_url=sd + ours, ours_levels=[d["path"] for d in ds],
                pub_name=pub[0].split("/")[-2], pub_prefix=pub[0], ct_prefix=ct[0], ct_name=ct[0].split("/")[-2])


def store_sizes(info, cache):
    """Published level 0 / all levels (S3 listing); ours level 0 / all levels (HEAD per shard)."""
    keys, _ = s3_list(info["pub_prefix"])
    pub0 = sum(s for k, s in keys if k[len(info["pub_prefix"]):].startswith("0/"))
    puball = sum(s for _, s in keys)
    tot = level_sizes(info["ours_url"], info["ours_levels"], cache)
    return dict(pub_l0=pub0, pub_all=puball, pub_objects=len(keys), ours_l0=tot[0], ours_all=sum(tot))


def level_sizes(base, levels, cache):
    """Bytes per level of one of our sharded stores (HEAD per shard)."""
    tot = []
    for lv in levels:
        m = json.loads(get(f"{base}/{lv}/zarr.json", cache))
        sh = np.array(m["shape"])
        side = int(m["chunk_grid"]["configuration"]["chunk_shape"][0])
        g = -(-sh // side)
        urls = [f"{base}/{lv}/c/{a}/{b}/{c}" for a in range(g[0]) for b in range(g[1]) for c in range(g[2])]
        tot.append(sum(ThreadPoolExecutor(16).map(head_size, urls)))
    return tot


def mask_box(info, lo, hi, P, C, out, cache):
    """The published mask re-exported as volcomp mask stores: mode 5 (exact, read back from the mirror store)
    and mode 4 (2x2x2 majority pool, decoded as a trilinear ramp; encoded here chunk by chunk on the store's
    128^3 grid) against the published mask in the box, and one panel strip + zoom."""
    sd = f"{OURS}/{info['sample']}/representations/predictions/surfaces/{info['pub_name']}"
    g = json.loads(get(sd + "/zarr.json", cache))
    lv = [d["path"] for d in g["attributes"]["ome"]["multiscales"][0]["datasets"]]
    m5 = OursArray(f"{sd}/{lv[0]}", cache)
    assert (m5.shape == np.array(info["shape"])).all(), (m5.shape, info["shape"])
    M5 = m5.read(lo, hi)
    m5.drop()
    # mode 4 on the aligned 128 grid around the box
    alo, ahi = lo // CH * CH, -(-hi // CH) * CH
    pub = V2Array(f"{OPEN}/{info['pub_prefix'].rstrip('/')}/0", cache)
    A = pub.read(alo, np.minimum(ahi, pub.shape))
    A = np.pad(A, [(0, int(b - a - n)) for a, b, n in zip(alo, ahi, A.shape)])
    M4 = np.zeros_like(A)
    n4 = n5 = 0
    tmp = np.empty(CH ** 3, np.uint8)
    for z in range(0, A.shape[0], CH):
        for y in range(0, A.shape[1], CH):
            for x in range(0, A.shape[2], CH):
                c = A[z:z + CH, y:y + CH, x:x + CH]
                if not c.any():
                    continue
                b = np.ascontiguousarray(c).tobytes()
                e = vc.mask_encode(b)
                n4 += len(e)
                n5 += len(vc.mask_encode_lossless(b))
                vc.decode(e, tmp)
                M4[z:z + CH, y:y + CH, x:x + CH] = tmp.reshape(CH, CH, CH)
    sl = tuple(slice(a - b, c - b) for a, c, b in zip(lo, hi, alo))
    M4 = M4[sl]
    Pb, In = P > 0, C > 0
    r = {"store": sd, "levels": lv, "box_lo": lo.tolist(), "box_hi": hi.tolist(),
         "aligned_box_lo": alo.tolist(), "aligned_box_hi": ahi.tolist(),
         "mode5_mismatch_voxels": int(np.count_nonzero((M5 > 0) != Pb)),
         "mode5_values": np.unique(M5[::4, ::4, ::4]).tolist(),
         "mode4_box_bytes": n4, "mode5_box_bytes_encoded_here": n5}
    A4 = M4 >= 128
    for name, W in (("all_voxels", None), ("inside_ct_mask", In)):
        a4, pb = (A4, Pb) if W is None else (A4 & W, Pb & W)
        r["mode4_vs_published_" + name] = scores([int(a4.sum()), int(pb.sum()), int((a4 & pb).sum())])
    r["mode4_mismatch_voxels"] = int(np.count_nonzero(A4 != Pb))
    r["mode4_mae_u8"] = float(np.abs(M4.astype(np.float32) - Pb * 255.0).mean())
    # panel: the first z view at 1/3 of the box
    i = int((hi[0] - lo[0]) / 3)
    ct, pb, m4, m5v = C[i], Pb[i], M4[i], M5[i]
    grey = lambda v: np.repeat(v[..., None], 3, -1).astype(np.uint8)  # noqa: E731
    d = np.repeat(ct.astype(np.float32)[..., None] * 0.4, 3, -1)
    a4 = m4 >= 128
    d[pb & ~a4], d[a4 & ~pb], d[a4 & pb] = RED, BLUE, BOTH
    panels = [over(ct, 0.6 * pb), grey(m4), grey(m5v), rgb(d)]
    names = ["published m7 mask (blosc, CT underneath)", "volcomp mode 4 decoded (2x pooled, trilinear ramp)",
             "volcomp mode 5 decoded (exact)", "mode 4 >= 128 vs published: red=published only, blue=mode 4 only"]
    view = f"z{lo[0] + i}"
    save(strip([p_[::2, ::2] for p_ in panels], names, 1.0), os.path.join(out, f"mask_{view}_all.png"))
    U = pb | a4
    S = np.zeros((U.shape[0] + 1, U.shape[1] + 1), np.int64)
    S[1:, 1:] = U.cumsum(0).cumsum(1)
    c = min(256, *U.shape)
    best, org = -1, (0, 0)
    for y in range(0, U.shape[0] - c + 1, 32):
        for x in range(0, U.shape[1] - c + 1, 32):
            v = S[y + c, x + c] - S[y, x + c] - S[y + c, x] + S[y, x]
            if v > best:
                best, org = v, (y, x)
    y, x = org
    save(strip([p_[y:y + c, x:x + c].repeat(2, 0).repeat(2, 1) for p_ in panels], names, 1.0),
         os.path.join(out, f"mask_{view}_zoom.png"))
    r["view"], r["zoom_yx_size"] = view, [int(y), int(x), int(c)]
    r["view_mode4_mismatch"] = int(np.count_nonzero(a4 != pb))
    r["view_mode5_mismatch"] = int(np.count_nonzero((m5v > 0) != pb))
    r["mask_sizes"] = dict(zip(lv, level_sizes(sd, lv, cache)))
    return r


def auto_box(pubroot, ctroot, k, shape0, size, cache):
    """Origin of the size^3 window (level-0 grid) holding the most published surface voxels INSIDE the CT mask
    (the published mask also covers blocks of masked air, which must not win)."""
    lev = 0
    while lev < min(5, 5 - k) and max(shape0) >> (lev + 1) >= 256:
        lev += 1
    a = V2Array(f"{pubroot}/{lev}", cache)
    c = V2Array(f"{ctroot}{k + lev}", cache)
    sh = np.minimum(a.shape, c.shape)
    m = ((a.read((0, 0, 0), sh) > 0) & (c.read((0, 0, 0), sh) > 0)).astype(np.float64)
    b = min(size >> lev, *m.shape)
    S = np.zeros(tuple(np.array(m.shape) + 1))
    S[1:, 1:, 1:] = m.cumsum(0).cumsum(1).cumsum(2)
    step = max(1, b // 16)
    best, org = -1, (0, 0, 0)
    for z in range(0, m.shape[0] - b + 1, step):
        for y in range(0, m.shape[1] - b + 1, step):
            xs = np.arange(0, m.shape[2] - b + 1, step)
            z1, y1, x1 = z + b, y + b, xs + b
            v = (S[z1, y1, x1] - S[z, y1, x1] - S[z1, y, x1] - S[z1, y1, xs] + S[z, y, x1] + S[z, y1, xs]
                 + S[z1, y, xs] - S[z, y, xs])
            i = int(v.argmax())
            if v[i] > best:
                best, org = v[i], (z, y, int(xs[i]))
    o = np.minimum(np.array(org) << lev, np.maximum(shape0 - size, 0))
    return [int(v) for v in o], float(best / b ** 3), lev


# --------------------------------------------------------------------------- stats
def counts(ours, pub, inside=None):
    """{thr: [n_ours, n_pub, n_both]} for ours >= 0.5 and >= 0.2 against the published mask, over the voxels
    where `inside` (the CT mask, CT > 0) is true, or over all voxels when it is None."""
    P = pub > 0
    if inside is not None:
        P &= inside
    npub = int(np.count_nonzero(P))
    r = {}
    for name, t in (("p>=0.5", T50), ("p>=0.2", T20)):
        A = ours >= t
        if inside is not None:
            A &= inside
        r[name] = [int(np.count_nonzero(A)), npub, int(np.count_nonzero(A & P))]
    return r


def scores(c):
    na, npub, nb = c
    return dict(dice=2 * nb / max(1, na + npub), precision=nb / max(1, na), recall=nb / max(1, npub),
                n_ours=na, n_pub=npub, n_both=nb)


def add(acc, c):
    for k, v in c.items():
        acc.setdefault(k, [0, 0, 0])
        acc[k] = [a + b for a, b in zip(acc[k], v)]


# --------------------------------------------------------------------------- images
def rgb(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def over(ct, alpha, col=AMBER):
    g = ct.astype(np.float32)[..., None]
    a = alpha.astype(np.float32)[..., None]
    return rgb(g * (1 - a) + col * a)


def diff_panel(ct, ours, pub):
    A, P, In = ours >= T50, pub > 0, ct > 0
    img = np.repeat(ct.astype(np.float32)[..., None] * 0.4, 3, -1)
    img[P & ~A & In] = RED
    img[P & ~A & ~In] = AIR
    img[A & ~P] = BLUE
    img[A & P] = BOTH
    return rgb(img)


def label(img, text, scale):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.load_default(size=max(12, int(15 * scale)))
    except TypeError:
        f = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((6, 4), text, font=f)
    d.rectangle((x0 - 4, y0 - 3, x1 + 4, y1 + 3), fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255), font=f)
    return np.asarray(im)


def strip(panels, names, scale):
    gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    out = []
    for i, (p, n) in enumerate(zip(panels, names)):
        out += ([gap] if i else []) + [label(p, n, scale)]
    return np.concatenate(out, 1)


def save(img, path):
    from PIL import Image
    Image.fromarray(img).save(path, optimize=True)


PANEL_NAMES = ["CT", "published m7 (binary, p>=0.2)", "ours m7 q8 (prob, plain decode)",
               "ours m7 q8 (prob, decode smoothing)",
               "red=published only, blue=ours only (p>=0.5), grey=both"]


def render(view, ct, pub, ours, ours_s, out, lean=False):
    panels = [np.repeat(ct[..., None], 3, -1), over(ct, 0.6 * (pub > 0)), over(ct, 0.6 * ours / 255.0),
              over(ct, 0.6 * ours_s / 255.0), diff_panel(ct, ours, pub)]
    for suf, p in zip(["ct", "published", "ours", "ours_smooth", "diff"], panels):
        if lean and suf != "diff":  # lean: the four single panels are only in the strips
            continue
        save(p if suf != "ct" else ct, os.path.join(out, f"{view}_{suf}.png"))
    half = [p[::2, ::2] for p in panels]
    save(strip(half, PANEL_NAMES, 1.0), os.path.join(out, f"{view}_all.png"))
    # 256^2 crop with the most surface (union of both masks), stride 32
    U = (((pub > 0) | (ours >= T50)) & (ct > 0)).astype(np.int64)
    S = np.zeros((U.shape[0] + 1, U.shape[1] + 1), np.int64)
    S[1:, 1:] = U.cumsum(0).cumsum(1)
    c = min(256, *U.shape)
    best, org = -1, (0, 0)
    for y in range(0, U.shape[0] - c + 1, 32):
        for x in range(0, U.shape[1] - c + 1, 32):
            v = S[y + c, x + c] - S[y, x + c] - S[y + c, x] + S[y, x]
            if v > best:
                best, org = v, (y, x)
    y, x = org
    crop = [p[y:y + c, x:x + c].repeat(2, 0).repeat(2, 1) for p in panels]
    save(strip(crop, PANEL_NAMES, 1.0), os.path.join(out, f"{view}_zoom.png"))
    return [int(y), int(x), int(c)]


# --------------------------------------------------------------------------- main
def block_counts(o, p, In):
    """counts() for one block with the CT mask `In`, as full-array boolean counts (the same numbers as counts()):
    returns (counts inside the CT mask, counts over all voxels, published outside it, ours >= 0.2 outside it)."""
    P = p > 0
    PI = P & In
    npub, npub_in = int(np.count_nonzero(P)), int(np.count_nonzero(PI))
    r_in, r_all = {}, {}
    for name, t in (("p>=0.5", T50), ("p>=0.2", T20)):
        A = o >= t
        AI = A & In
        na, nai = int(np.count_nonzero(A)), int(np.count_nonzero(AI))
        r_in[name] = [nai, npub_in, int(np.count_nonzero(AI & PI))]
        r_all[name] = [na, npub, int(np.count_nonzero(A & P))]
        if t == T20:
            air1 = na - nai
    return r_in, r_all, npub - npub_in, air1


def whole_volume(ours, pub, ctmask, f, modes=("plain", "gated2"), mask_sizes=False, workers=12):
    """Stream the level-0 grid in 384^3 blocks (a multiple of both the published 192^3 chunks and our 128^3
    chunks, so every published chunk is fetched exactly once): counts for each decode mode over the whole volume,
    inside the CT mask `ctmask` (a boolean CT > 0 at f x coarser, looked up per voxel), plus the published voxels
    outside it. mask_sizes: also encode every non-empty 128^3 chunk of the published mask as a volcomp mode-4 mask
    chunk and sum the bytes (what a mode-4 re-export of level 0 would store)."""
    assert modes[0] == "plain", modes
    sub = 384
    acc = {m: {} for m in modes}
    acc["all_voxels_plain"] = {}
    air = [0, 0]
    m4 = {"bytes": 0, "chunks": 0, "shards": set()}
    lock = threading.Lock()
    g = -(-ours.shape // sub)
    blocks = [(a, b, c) for a in range(g[0]) for b in range(g[1]) for c in range(g[2])]
    inner = ThreadPoolExecutor(48)  # shared by all block workers: this bounds the requests in flight
    done = [0]
    t0 = time.time()
    assert sub % f == 0, (sub, f)

    def one(k):
        lo = np.array(k) * sub
        hi = np.minimum(lo + sub, ours.shape)
        p = pub.read(lo, hi, inner)
        r = {}
        o = ours.read(lo, hi, modes[0], inner)
        if not p.any() and not o.any():  # nothing on either side: every count is 0
            r = {m: {"p>=0.5": [0, 0, 0], "p>=0.2": [0, 0, 0]} for m in modes + ("all_voxels_plain",)}
            ai = (0, 0)
        else:
            cl, n = lo // f, sub // f
            ms = ctmask[cl[0]:cl[0] + n, cl[1]:cl[1] + n, cl[2]:cl[2] + n]
            ms = np.pad(ms, [(0, n - x) for x in ms.shape])
            In = np.broadcast_to(ms[:, None, :, None, :, None], (n, f, n, f, n, f)).reshape(sub, sub, sub)
            In = In[:hi[0] - lo[0], :hi[1] - lo[1], :hi[2] - lo[2]]
            for m in modes:
                if m != modes[0]:
                    o = ours.read(lo, hi, m, inner)
                r[m], r_all, a0, a1 = block_counts(o, p, In)
                if m == "plain":
                    r["all_voxels_plain"], ai = r_all, (a0, a1)
        nb, nc, shards = 0, 0, set()
        if mask_sizes and p.any():
            for cz in range(0, p.shape[0], CH):
                for cy in range(0, p.shape[1], CH):
                    for cx in range(0, p.shape[2], CH):
                        c = p[cz:cz + CH, cy:cy + CH, cx:cx + CH]
                        if not c.any():
                            continue
                        if c.shape != (CH, CH, CH):
                            c = np.pad(c, [(0, CH - n) for n in c.shape])
                        nb += len(vc.mask_encode(np.ascontiguousarray(c).tobytes()))
                        nc += 1
                        shards.add(tuple((int(v) + w) // 1024 for v, w in zip(lo, (cz, cy, cx))))
        with lock:
            for m in r:
                add(acc[m], r[m])
            air[0] += ai[0]
            air[1] += ai[1]
            m4["bytes"] += nb
            m4["chunks"] += nc
            m4["shards"] |= shards
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  whole: {done[0]}/{len(blocks)} blocks, {time.time() - t0:.0f} s", flush=True)
    # z-major order, dropping our shard cache (on disk) every 3072 slices (a multiple of 384 and 1024)
    per = 3072 // sub
    for zs in range(0, g[0], per):
        list(ThreadPoolExecutor(workers).map(one, [b for b in blocks if zs <= b[0] < zs + per]))
        ours.drop()
    out = {m: {t: scores(c) for t, c in v.items()} for m, v in acc.items()}
    out["published_in_air"], out["ours_in_air_p>=0.2"], out["ct_mask_level_factor"] = air[0], air[1], f
    out["modes"] = list(modes)
    if mask_sizes:
        ns = len(m4["shards"])
        out["mode4_level0"] = {"chunk_bytes": m4["bytes"], "chunks": m4["chunks"], "shards_1024": ns,
                               "index_bytes": ns * (16 * 512 + 4), "bytes": m4["bytes"] + ns * (16 * 512 + 4)}
    return out


def float_check(keep, cache):
    """Compression only, on the kept PHerc0343P 8.64 um box: our published q8 L0 store vs the float m7 blend."""
    ex = json.load(open(os.path.join(keep, "median_experiment.json")))
    lo, hi = np.array(ex["box_lo"]), np.array(ex["box_hi"])
    pf = np.load(os.path.join(keep, "pfloat_plainCT_box.npy")).astype(np.float32)
    meta = json.loads(get(f"{OURS}/PHerc0343P/representations/predictions/surfaces/"
                          "20250521134555-surface-20260925170000-surface-m7-L0-prob.zarr/8.64/zarr.json", cache))
    arr = OursArray(os.path.join(keep, "dl", "pr"), cache, meta)
    r = {"scan": "PHerc0343P 20250521134555-8.640um (level 0)", "box_lo": lo.tolist(), "box_hi": hi.tolist(),
         "stride": 96, "ct_input": "volcomp q8 CT, plain decode"}
    for m in MODES:
        o = arr.read(lo, hi, m)
        for name, t, ft in (("p>=0.5", T50, 0.5), ("p>=0.2", T20, 0.2)):
            F, A = pf >= ft, o >= t
            r.setdefault(m, {})[name] = scores([int(A.sum()), int(F.sum()), int((A & F).sum())])
        r[m]["MAE_u8"] = float(np.abs(o.astype(np.float32) - np.clip(np.round(pf * 255), 0, 255)).mean())
    return r


def fmt_size(b):
    return f"{b / 2**30:.2f} GiB" if b >= 2**30 else f"{b / 2**20:.0f} MiB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sample", required=True)
    ap.add_argument("--scan", required=True, help="scan timestamp, e.g. 20260304131111")
    ap.add_argument("--box", default="auto", help="z0,y0,x0,dz,dy,dx at the prediction's level 0, or auto")
    ap.add_argument("--size", type=int, default=1024, help="auto box edge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/surface_compare"))
    ap.add_argument("--whole", action="store_true")
    ap.add_argument("--float-keep", default=None)
    ap.add_argument("--lean", action="store_true", help="write only the diff, all and zoom images per view")
    ap.add_argument("--whole-modes", default="plain,gated2", help="decode modes for --whole")
    ap.add_argument("--mask-out", default=None, help="also compare the mirror's volcomp mask stores (mode 5) and a "
                    "mode-4 re-encode with the published mask; images + report.json into this directory")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    info = discover(a.sample, a.scan, a.cache)
    print(json.dumps(info, indent=1), flush=True)
    ours = OursArray(f"{info['ours_url']}/{info['ours_levels'][0]}", a.cache)
    pubroot = f"{OPEN}/{info['pub_prefix'].rstrip('/')}"
    pub = V2Array(f"{pubroot}/0", a.cache)
    ct = V2Array(f"{OPEN}/{info['ct_prefix']}{info['level']}", a.cache)
    assert (pub.shape == ours.shape).all() and (ct.shape == ours.shape).all(), (pub.shape, ours.shape, ct.shape)
    params = {}
    try:
        params = json.loads(get(f"{info['ours_url']}/zarr.json", a.cache))["attributes"]["volcomp"]["inference"]
    except Exception:
        pass
    if a.box == "auto":
        org, dens, lev = auto_box(pubroot, f"{OPEN}/{info['ct_prefix']}", info["level"], ours.shape, a.size, a.cache)
        size = np.minimum(a.size, ours.shape)
        box = np.array(org + size.tolist())
        print(f"auto box {box.tolist()} (published surface fraction {dens:.3f}, searched at level {lev})", flush=True)
    else:
        box = np.array([int(v) for v in a.box.split(",")])
        dens = None
    lo, hi = box[:3], box[:3] + box[3:]
    rep = dict(sample=a.sample, info=info, shape=ours.shape.tolist(), voxels=int(np.prod(ours.shape)),
               box_lo=lo.tolist(), box_hi=hi.tolist(), box_published_fraction=dens, producer_params=params)
    rep["sizes"] = store_sizes(info, a.cache)
    print("sizes", rep["sizes"], f"{time.time() - t0:.0f} s", flush=True)

    C = ct.read(lo, hi)
    P = pub.read(lo, hi)
    print(f"read CT + published box {time.time() - t0:.0f} s; published values {np.unique(P[::4, ::4, ::4]).tolist()}",
          flush=True)
    O = {m: ours.read(lo, hi, m) for m in MODES}
    ours.drop()
    print(f"decoded ours x3 {time.time() - t0:.0f} s", flush=True)
    In = C > 0
    rep["box"] = {m: {t: scores(c) for t, c in counts(O[m], P, In).items()} for m in MODES}
    rep["box_all_voxels"] = {m: {t: scores(c) for t, c in counts(O[m], P).items()} for m in MODES}
    rep["box"]["ct_nonzero_fraction"] = float(np.count_nonzero(In) / In.size)
    rep["box"]["published_in_air"] = int(np.count_nonzero((P > 0) & ~In))
    rep["box"]["ours_in_air_p>=0.2"] = int(np.count_nonzero((O["plain"] >= T20) & ~In))
    del In
    print(json.dumps(rep["box"], indent=1), flush=True)

    views = []
    for ax, name in enumerate("zyx"):
        for f in (1 / 3, 2 / 3):
            i = int(box[3 + ax] * f)
            sl = [slice(None)] * 3
            sl[ax] = i
            sl = tuple(sl)
            v = f"{name}{lo[ax] + i}"
            crop = render(v, C[sl], P[sl], O["plain"][sl], O["gated2"][sl], a.out, a.lean)
            c2 = counts(O["plain"][sl], P[sl], C[sl] > 0)["p>=0.5"]
            views.append(dict(view=v, axis=name, index=int(lo[ax] + i), zoom_yx_size=crop, slice_scores=scores(c2)))
            print(f"rendered {v} {time.time() - t0:.0f} s", flush=True)
    rep["views"] = views
    rep["lean"] = a.lean
    del O
    if a.mask_out:
        os.makedirs(a.mask_out, exist_ok=True)
        rep["mask"] = mask_box(dict(info, sample=a.sample, shape=ours.shape.tolist()), lo, hi, P, C, a.mask_out, a.cache)
        print("mask", json.dumps(rep["mask"], indent=1), flush=True)
    del C, P
    if a.float_keep:
        rep["float_check"] = float_check(a.float_keep, a.cache)
        print("float check", json.dumps(rep["float_check"], indent=1), flush=True)
    if a.whole:
        # CT mask: the finest level >= 2 steps coarser that stays under 6 Gvox (a boolean held in memory)
        m = 2
        while np.prod(ours.shape.astype(np.float64) / 2 ** m) > 6e9:
            m += 1
        f = 2 ** m
        cta = V2Array(f"{OPEN}/{info['ct_prefix']}{info['level'] + m}", a.cache)
        ctmask = np.pad(cta.read((0, 0, 0), cta.shape) > 0, [(0, 2)] * 3)
        print(f"CT mask level {info['level'] + m} {ctmask.shape} {time.time() - t0:.0f} s", flush=True)
        # the published level 0 is streamed uncached (tens of GB on the 8.64 um scans); ours is cached on disk
        pubw = V2Array(f"{pubroot}/0", None)
        rep["whole"] = whole_volume(ours, pubw, ctmask, f, tuple(a.whole_modes.split(",")), bool(a.mask_out))
        print("whole", json.dumps(rep["whole"], indent=1), flush=True)
        if a.mask_out and "mask" in rep:
            rep["mask"]["mode4_level0"] = rep["whole"]["mode4_level0"]
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    write_readme(rep, a.out)
    if a.mask_out and "mask" in rep:
        json.dump(dict(sample=a.sample, info=info, sizes=rep["sizes"], shape=rep["shape"], **rep["mask"]),
                  open(os.path.join(a.mask_out, "report.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s", flush=True)


def write_readme(r, out):
    i, s, pp = r["info"], r["sizes"], r.get("producer_params") or {}
    pitch = i["ours_levels"][0]
    L = [f"# {r['sample']}: our volcomp m7 surface prediction vs the published m7 prediction", "",
         f"Scan `{i['ct_name']}`, CT level {i['level']} ({pitch} µm), prediction grid {' x '.join(map(str, r['shape']))} "
         f"(z, y, x) = {r['voxels'] / 1e9:.1f} Gvox. Made by `tools/surface_compare.py`; every number is in "
         "[report.json](report.json).", "",
         f"- **Published**: `s3://vesuvius-challenge-open-data/{i['pub_prefix']}` level 0, OME-zarr v2, blosc-zstd "
         "192³ chunks. A **binary mask**: the m7 probability thresholded at **0.2**.",
         f"- **Ours**: `{i['ours_url']}` level `{pitch}`, zarr v3 sharded, volcomp q8. An **8-bit probability** "
         "(uint8 = round(p · 255)), decoded plain (what any zarr reader gets) or with volcomp's decode-only "
         "smoothing: `gated2` is what `volcomp_zarr.set_read_smoothing(2)` gives (`decode_smooth(…, 2, GATED | "
         "ZERO_GUARD)`), `gauss0.6` is `decode_smooth(…, 0.6, ZERO_GUARD)`, the setting docs/deblocking.md "
         "recommends for probability maps. The images use plain and gated2.",
         f"- **CT** under the images: the original uncompressed level {i['level']} "
         f"(`s3://vesuvius-challenge-open-data/{i['ct_prefix']}{i['level']}`).", "",
         "**What differs besides compression.** Both are the same m7 checkpoint (surface_m7_nnunet) on the same "
         f"scan at the same level, but they are separate inference runs. Ours: {pp.get('window', 192)}³ windows at "
         f"**stride {pp.get('stride', '?')}**, separable Gaussian blend, {pp.get('backend', 'TensorRT fp16')}, no TTA, "
         f"run on our volcomp q8 CT ({pp.get('ct_decode', 'plain decode')}). "
         "Published: the upstream pipeline (nnU-Net predictor, TTA disabled per its metadata.json) on the "
         "original CT. m7's mask moves with window placement (stride 128 against 96 on the same input gave dice "
         "0.946; no stride pair reaches 0.99), so a large part of the disagreement below is inference, not "
         "compression. The compression-only loss is measured separately at the end.", ""]
    L += ["## Table", "",
          "| | |", "|---|---|",
          f"| voxels (level 0) | {r['voxels']:,} ({r['voxels'] / 1e9:.1f} Gvox) |",
          f"| published store, level 0 | {fmt_size(s['pub_l0'])} ({s['pub_l0']:,} B) |",
          f"| our store, level 0 | {fmt_size(s['ours_l0'])} ({s['ours_l0']:,} B) |",
          f"| ratio published / ours, level 0 | {s['pub_l0'] / s['ours_l0']:.2f}x |",
          f"| published, all 6 levels / ours, all {len(i['ours_levels'])} levels | {fmt_size(s['pub_all'])} / "
          f"{fmt_size(s['ours_all'])} ({s['pub_all'] / s['ours_all']:.2f}x) |", ""]
    b = r["box"]
    L += [f"Agreement with the published mask in the comparison box z {r['box_lo'][0]}-{r['box_hi'][0]}, "
          f"y {r['box_lo'][1]}-{r['box_hi'][1]}, x {r['box_lo'][2]}-{r['box_hi'][2]} "
          f"({np.prod(np.array(r['box_hi']) - np.array(r['box_lo'])) / 1e9:.2f} Gvox, the window with the most "
          f"published surface inside the CT mask; {b['ct_nonzero_fraction'] * 100:.0f} % of it is inside the CT mask), "
          "counted **inside the CT mask** (CT > 0). Precision = fraction of our voxels that the published mask also "
          "has; recall = fraction of the published voxels we also have. `p>=0.2` compares at the published "
          "threshold, `p>=0.5` at the usual one.", "",
          f"**The published mask also marks masked air.** In this box {b['published_in_air']:,} published surface "
          f"voxels lie where the published CT is 0 (outside the scroll mask), mostly in solid blocks; our store has "
          f"{b['ours_in_air_p>=0.2']:,} voxels >= 0.2 there (it is 0 wherever our volcomp copy of the CT is 0; the "
          "few left are at mask edges where the two copies differ). They "
          "are drawn dark red in the diff panels and left out of the scores below; "
          f"counted in, the plain p>=0.5 dice would be {r['box_all_voxels']['plain']['p>=0.5']['dice']:.4f}.", "",
          "| ours decoded | threshold | dice | precision | recall | our voxels | published voxels |",
          "|---|---|---|---|---|---|---|"]
    for m in MODES:
        for t in ("p>=0.5", "p>=0.2"):
            x = b[m][t]
            L.append(f"| {m} | {t} | {x['dice']:.4f} | {x['precision']:.4f} | {x['recall']:.4f} | "
                     f"{x['n_ours']:,} | {x['n_pub']:,} |")
    if "whole" in r:
        w = r["whole"]
        L += ["", f"Whole volume (every level-0 voxel inside the CT mask; the mask is CT level "
              f"{i['level'] + int(w['ct_mask_level_factor']).bit_length() - 1} > 0, nearest-upsampled {w['ct_mask_level_factor']}x). Published surface voxels "
              f"outside the mask: {w['published_in_air']:,}; ours >= 0.2 outside it: {w['ours_in_air_p>=0.2']:,} (mask "
              f"edges: the {w['ct_mask_level_factor']}x-coarser mask is approximate there). The whole volume includes the sparse outer wraps "
              "and mask edges, so it can score below the dense box.", "",
              "| ours decoded | threshold | dice | precision | recall | our voxels | published voxels |",
              "|---|---|---|---|---|---|---|"]
        for m, v in w.items():
            if not isinstance(v, dict) or "p>=0.5" not in v:
                continue
            for t in ("p>=0.5", "p>=0.2"):
                x = v[t]
                L.append(f"| {m} | {t} | {x['dice']:.4f} | {x['precision']:.4f} | {x['recall']:.4f} | "
                         f"{x['n_ours']:,} | {x['n_pub']:,} |")
    if "float_check" in r:
        f = r["float_check"]
        L += ["", "### Compression only", "",
              "No uncompressed float m7 prediction exists for this scan. The one kept float run is on "
              f"**{f['scan']}**, a different scan of the same scroll, box z {f['box_lo'][0]}-{f['box_hi'][0]}, "
              f"y {f['box_lo'][1]}-{f['box_hi'][1]}, x {f['box_lo'][2]}-{f['box_hi'][2]}: the same engine, stride "
              "96 and CT input as our production store there, blended in fp32 and kept uncompressed. It cannot be "
              "compared to the published mask (different scan, no registration), but against our q8 store of the "
              "same run it isolates what volcomp q8 costs:", "",
              "| ours decoded | MAE (/255) | dice p>=0.5 | dice p>=0.2 | precision p>=0.5 | recall p>=0.5 |",
              "|---|---|---|---|---|---|"]
        for m in MODES:
            x = f[m]
            L.append(f"| {m} | {x['MAE_u8']:.2f} | {x['p>=0.5']['dice']:.4f} | {x['p>=0.2']['dice']:.4f} | "
                     f"{x['p>=0.5']['precision']:.4f} | {x['p>=0.5']['recall']:.4f} |")
    elif r["sample"] != "PHerc0343P":  # noqa: SIM114
        L += ["", "No uncompressed float m7 prediction exists for this scroll; see "
              "[../PHerc0343P/README.md](../PHerc0343P/README.md#compression-only) for the compression-only loss "
              "measured on the kept PHerc0343P box.", ""]
    L += ["", "## Images", "",
          "Slices at 1/3 and 2/3 of the box along each axis (fixed positions, not picked), named by axis and "
          "level-0 index. Full resolution: 1 image pixel = 1 voxel, 1024² unless the volume is smaller.", "",
          "- `<view>_ct.png`: the CT slice.",
          "- `<view>_published.png`: the published **binary threshold-0.2 mask** in amber at 60 % opacity over the CT.",
          "- `<view>_ours.png`: our **8-bit probability**, plain decode, in amber at opacity 0.6 · p over the CT "
          "(faint amber = low probability; the published mask has no such gradation).",
          "- `<view>_ours_smooth.png`: the same with decode smoothing (`gated2`, what "
          "`volcomp_zarr.set_read_smoothing(2)` gives; on the kept float box it has the lowest error of the three "
          "decodes).",
          "- `<view>_diff.png` (plain decode): darkened CT; **red** = published only, **blue** = ours only (plain decode, p ≥ 0.5), "
          "**grey** = both, **dark red** = published surface where the CT is 0 (masked air). Because the published threshold (0.2) is lower than ours (0.5), a thin red rim along "
          "surfaces is expected even from identical models.",
          "- `<view>_all.png`: the five panels side by side at half resolution, labelled.",
          "- `<view>_zoom.png`: the 256² crop of the slice with the most surface, five panels at 2x (nearest), labelled.",
          "", "| view | slice dice (p>=0.5, plain) | precision | recall | images |", "|---|---|---|---|---|"]
    if r.get("lean"):
        L.insert(len(L) - 2, "- Only `_all`, `_zoom` and `_diff` are kept for this scroll (the four single panels "
                 "are in the `_all` strip at half resolution and in the zoom at 2x).")
    for v in r["views"]:
        n, x = v["view"], v["slice_scores"]
        imgs = (f"[all]({n}_all.png) · [zoom]({n}_zoom.png) · [diff]({n}_diff.png)" if r.get("lean") else
                f"[all]({n}_all.png) · [zoom]({n}_zoom.png) · [CT]({n}_ct.png) · [published]({n}_published.png) · "
                f"[ours]({n}_ours.png) · [ours smooth]({n}_ours_smooth.png) · [diff]({n}_diff.png)")
        L.append(f"| {n} | {x['dice']:.4f} | {x['precision']:.4f} | {x['recall']:.4f} | {imgs} |")
    v0 = r["views"][0]["view"]
    L += ["", f"![{v0} all]({v0}_all.png)", "", f"![{v0} zoom]({v0}_zoom.png)", ""]
    open(os.path.join(out, "README.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
