"""Round 2: transform vs in-loop predictive coding, and the threshold fix-up layer."""
import sys, json, time
import numpy as np
from common import *
import nl

TEACH = sys.argv[1]
REG = sys.argv[2].split(",")
NK = int(sys.argv[3]) if len(sys.argv) > 3 else 1
only = sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] != "-" else None


def dct_fix(q):
    base = lib_codec(q=q)
    def f(c):
        n, d = base(c)
        b, o = nl.fix_cost(c, d, 5)
        return n + int(np.ceil(b / 8)), o
    return f


def compand(q, k):
    """DCT in a companded domain: y = 127.5 + 127.5*tanh(k*(v-127.5)/127.5)/tanh(k) (steeper near 0.5)."""
    v = np.arange(256, dtype=np.float64)
    fw = 127.5 + 127.5 * np.tanh(k * (v - 127.5) / 127.5) / np.tanh(k)
    fwd = np.clip(np.rint(fw), 0, 255).astype(np.uint8)
    # inverse LUT over 0..255
    inv = np.clip(np.rint(127.5 + 127.5 / k * np.arctanh(np.clip((np.arange(256) - 127.5) / 127.5 * np.tanh(k), -0.999999, 0.999999))), 0, 255).astype(np.uint8)
    base = lib_codec(q=q)
    def f(c):
        n, d = base(fwd[c])
        return n, inv[d]
    return f


cands = {}
for q in (8, 16, 24, 32, 48):
    cands[f"dct q{q}"] = lib_codec(q=q)
    cands[f"dct q{q} +fix"] = dct_fix(q)
for q, k in ((16, 2.0), (32, 2.0), (32, 3.0)):
    cands[f"compand k{k} dct q{q}"] = compand(q, k)
for dl in (1, 2, 3, 4, 6, 8, 12):
    for pr in (1, 2):
        cands[f"nl d{dl} p{pr}"] = nl.nl_codec(dl, dl, zb=8, pred=pr)
if only:
    cands = {k: v for k, v in cands.items() if any(o in k for o in only)}

rows = {}
for reg in REG:
    for k in range(NK):
        p = load_cube(reg, TEACH, k)
        u8 = to_u8(p)
        cache = {}
        for name, fn in cands.items():
            t0 = time.time()
            nb_, d = run_chunkwise(u8, fn)
            m = metrics(p, d, cache=cache)
            m["bpv"] = nb_ * 8 / u8.size
            m["sec"] = time.time() - t0
            rows.setdefault(name, []).append(m)
        print(reg, k, "done", flush=True)
keys = ["bpv", "dice", "flip", "mae", "mae_band", "p99_band", "maxerr", "disp_mean", "disp_p99", "bdist_mean", "sdf_mae"]
print(f"\n{TEACH} {REG} cubes={NK}")
print(f"{'candidate':26s} " + " ".join(f"{k:>9s}" for k in keys))
out = {}
for name, rs in rows.items():
    mm = {k: float(np.mean([r[k] for r in rs])) for k in keys}
    out[name] = mm
    print(f"{name:26s} " + " ".join(f"{mm[k]:9.4f}" for k in keys))
json.dump(out, open(f"/vesuvius/usrm2/volcomp_surface/run2_{TEACH}_{'-'.join(REG)}.json", "w"), indent=1)
