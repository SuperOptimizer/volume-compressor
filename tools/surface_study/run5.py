"""Robust exact-threshold variants: DCT + independent lossless mask (mode-5 coder) with side forcing,
and mask-predicted residual coding (ramp or blurred-mask predictor from the exact mask)."""
import sys, json
import numpy as np
from scipy import ndimage as ndi
from common import *
TEACH, REG, NK = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
QS = [float(x) for x in sys.argv[4].split(",")]
mask5 = lib_codec(mode="mask-ll")


def force(d, m):
    return np.where(m, np.maximum(d, 128), np.minimum(d, 127)).astype(np.uint8)


def ramp(m, kind):
    if kind == "none":
        return None
    if kind.startswith("edt"):
        dmax = float(kind[3:])
        s = ndi.distance_transform_edt(m) - ndi.distance_transform_edt(~m)  # >0 inside (>=1), <=0 outside... 
        # boundary between voxel centres: shift by 0.5 so inside voxels start at +0.5, outside at -0.5
        s = np.where(m, s - 0.5, s + 0.5)
        return np.clip(np.rint(127.5 + 127.5 * np.clip(s / dmax, -1, 1)), 0, 255)
    if kind.startswith("blur"):
        sg = float(kind[4:])
        return np.clip(np.rint(ndi.gaussian_filter(m.astype(np.float32) * 255, sg)), 0, 255)


def variant(q, kind):
    def f(c):
        m = c >= 128
        mb = mask5(np.where(m, 255, 0).astype(np.uint8))[0]
        g = ramp(m, kind)
        if g is None:
            n, d = lib_codec(q=q)(c)
        else:
            r = np.clip(c.astype(np.int32) - g + 128, 0, 255).astype(np.uint8)
            n, dr = lib_codec(q=q)(r)
            d = np.clip(g + dr.astype(np.int32) - 128, 0, 255).astype(np.uint8)
        return n + mb, force(d, m)
    return f


for kind in ("none", "edt3", "edt6", "blur1.0", "blur2.0"):
    for q in QS:
        agg = []
        for reg in REG:
            for k in range(NK):
                p = load_cube(reg, TEACH, k); u8 = to_u8(p); cache = {}
                nb_, d = run_chunkwise(u8, variant(q, kind))
                m = metrics(p, d, cache=cache); m["bpv"] = nb_ * 8 / u8.size; agg.append(m)
        r = {kk: float(np.mean([x[kk] for x in agg])) for kk in agg[0]}
        r.update(tag=f"mask5+{kind}", q=q, teacher=TEACH)
        print(json.dumps(r), flush=True)
