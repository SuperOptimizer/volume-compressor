"""2x mean-pool -> DCT on the 64^3 grid (packed as one octant of a 128^3 chunk) -> trilinear 2x back."""
import sys, json
import numpy as np
from scipy import ndimage as ndi
from common import *
import nl
TEACH, REG, NK = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
QS = [float(x) for x in sys.argv[4].split(",")]


def up2(g):
    """64^3 -> 128^3 trilinear, cell centres at 2i+0.5, edge-clamped"""
    out = g.astype(np.float32)
    for ax in range(3):
        n = out.shape[ax]
        j = np.arange(2 * n)
        t = (j - 0.5) / 2.0
        i0 = np.clip(np.floor(t).astype(int), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        w = np.clip(t - np.floor(t), 0, 1)
        w = np.where(np.floor(t) < 0, 0.0, w)
        a = np.take(out, i0, axis=ax); b = np.take(out, i1, axis=ax)
        sh = [1, 1, 1]; sh[ax] = -1
        out = a * (1 - w.reshape(sh)) + b * w.reshape(sh)
    return out


def pooled(q, fix):
    base = lib_codec(q=q)
    def f(c):
        g = c.reshape(64, 2, 64, 2, 64, 2).astype(np.float32).mean(axis=(1, 3, 5))
        buf = np.zeros((128, 128, 128), np.uint8)
        buf[:64, :64, :64] = np.clip(np.rint(g), 0, 255)
        n, d = base(buf)
        o = np.clip(np.rint(up2(d[:64, :64, :64])), 0, 255).astype(np.uint8)
        if fix:
            b, o = nl.fix_cost(c, o, 5)
            n += int(np.ceil(b / 8))
        return n, o
    return f


for fix in (False, True):
    for q in QS:
        agg = []
        for reg in REG:
            for k in range(NK):
                p = load_cube(reg, TEACH, k); u8 = to_u8(p); cache = {}
                nb_, d = run_chunkwise(u8, pooled(q, fix))
                m = metrics(p, d, cache=cache); m["bpv"] = nb_ * 8 / u8.size; agg.append(m)
        r = {kk: float(np.mean([x[kk] for x in agg])) for kk in agg[0]}
        r.update(tag="pool2 dct" + ("+fix" if fix else ""), q=q, teacher=TEACH)
        print(json.dumps(r), flush=True)
