"""Step-2 table: every candidate on every cube; bytes and quality."""
import sys, json, time
import numpy as np
from common import *
import ctxcoder as cc

TEACH = sys.argv[1]
REG = sys.argv[2].split(",")
NK = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SKEL = "--skel" in sys.argv

cands = {}
for q in (1, 2, 4, 8, 16):
    cands[f"dct q{q}"] = ("lib", lib_codec(q=q))
cands["lossless u8 (q0)"] = ("lib", lib_codec(q=0))
cands["mask4 (2x pool)"] = ("mask", lib_codec(mode="mask"))
cands["mask5 (exact mask)"] = ("mask", lib_codec(mode="mask-ll"))
cbs = {
    "U4 d2": cc.cb_uniform(4, dead=2),
    "U8 d4": cc.cb_uniform(8, dead=4),
    "U16 d8": cc.cb_uniform(16, dead=8),
    "U32 d16": cc.cb_uniform(32, dead=16),
    "logit8x2 d3": cc.cb_logit(8, 0.02, dead=3),
    "logit4x2 d5": cc.cb_logit(4, 0.03, dead=5),
    "two 4/16 w32 d6": cc.cb_twostep(4, 16, 32, 6),
    "two 8/32 w32 d8": cc.cb_twostep(8, 32, 32, 8),
}
for k, (enc, dec) in cbs.items():
    cands[f"{k} [L={len(dec)}] q0"] = ("cbll", (enc, dec))
    cands[f"{k} [L={len(dec)}] ctx"] = ("cbctx", (enc, dec))

rows = {}
for reg in REG:
    for k in range(NK):
        p = load_cube(reg, TEACH, k)
        u8 = to_u8(p)
        cache = {}
        for name, (kind, arg) in cands.items():
            t0 = time.time()
            if kind == "lib":
                nb_, d = run_chunkwise(u8, arg)
            elif kind == "mask":
                nb_, d = run_chunkwise(np.where(u8 >= 128, 255, 0).astype(np.uint8), arg)
            elif kind == "cbll":
                enc, dec = arg
                nb_, d = run_chunkwise(dec[enc[u8]], lib_codec(q=0))
            else:
                enc, dec = arg
                nb_, d = run_chunkwise(u8, lambda c: cc.encode_cost(c, enc, dec, pred=1, zb=8))
            m = metrics(p, d, skel=SKEL, cache=cache)
            m["bpv"] = nb_ * 8 / u8.size
            rows.setdefault(name, []).append(m)
        print(reg, k, "done", flush=True)

keys = ["bpv", "dice", "flip", "mae", "mae_band", "p99_band", "maxerr", "disp_mean", "disp_p99", "bdist_mean", "sdf_mae"] + (["skel_f1"] if SKEL else [])
ref = np.mean([r["bpv"] for r in rows["dct q8"]])
print(f"\n{TEACH} {REG} cubes={NK}  (bpv = bits/voxel; x q8 = size relative to dct q8)")
print(f"{'candidate':34s} {'x q8':>6s} " + " ".join(f"{k:>9s}" for k in keys))
out = {}
for name, rs in rows.items():
    mm = {k: float(np.mean([r[k] for r in rs])) for k in keys}
    out[name] = mm
    print(f"{name:34s} {mm['bpv'] / ref:6.3f} " + " ".join(f"{mm[k]:9.4f}" for k in keys))
json.dump(out, open(f"/vesuvius/usrm2/volcomp_surface/run1_{TEACH}.json", "w"), indent=1)
