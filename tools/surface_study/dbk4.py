"""Final deblocking evaluation with the C implementations. usage: dbk4.py ct|recto|m7"""
import os, sys, json, time
os.environ["VOLCOMP_LIB"] = "/vesuvius/usrm2/volcomp_surface/vc2/build/release/libvolcomp.so"
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc2/python")
import numpy as np
import volcomp_zarr._lib as L
from dbk import octet, ct_metrics, C
from dbk3 import region, ssim_slices, edge_mae, per_chunk_mse
from common import load_cube, to_u8, metrics

def blocks(v):
    for z in range(0, v.shape[0], C):
        for y in range(0, v.shape[1], C):
            for x in range(0, v.shape[2], C):
                yield (slice(z, z + C), slice(y, y + C), slice(x, x + C))

def run(src, enc, dec_opts, q_db):
    streams = {}; nb = 0
    outs = {k: np.zeros_like(src) for k in dec_opts}; ts = {k: 0.0 for k in dec_opts}
    for s in blocks(src):
        c = np.ascontiguousarray(src[s])
        if not c.any():
            continue
        e = enc(c); nb += len(e)
        for k, f in dec_opts.items():
            t0 = time.time(); outs[k][s] = np.frombuffer(bytes(f(e)), np.uint8).reshape(C, C, C); ts[k] += time.time() - t0
    return nb, outs, ts

def db(v, q, zg):
    w = np.ascontiguousarray(v.copy()); L.deblock(w.reshape(-1).view(np.uint8), *w.shape, float(q), zero_guard=zg); return w

which = sys.argv[1]
rows = []
if which == "ct":
    vols = [("interior 512^3", region("interior", 269, 118, 144)), ("mask-edge 512^3", region("edge", 269, 158, 226)),
            ("tune octet", octet("tune", 250)), ("heldout octet", octet("heldout", 350))]
    settings = [("dct", q) for q in (2, 4, 8, 16)]
else:
    vols = [(f"{which} evalbox", to_u8(load_cube("evalbox", which, 0))), (f"{which} r32k", to_u8(load_cube("r32k", which, 0)))]
    settings = [("dct", 8), ("dct", 16), ("surface", 48), ("surface", 64)]
for vname, src in vols:
    nch = sum(1 for s in blocks(src) if src[s].any())
    for kind, q in settings:
        enc = (lambda c, q=q: L.encode(c.tobytes(), float(q))) if kind == "dct" else (lambda c, q=q: L.surface_encode(c.tobytes(), float(q)))
        opts = {"none": L.decode}
        for st in (1.0, 2.0, 3.0):
            opts[f"smooth gated{st:g} zg"] = lambda e, st=st: L.decode_smooth(e, st, gated=True, zero_guard=True)
        opts["smooth gauss0.6 zg"] = lambda e: L.decode_smooth(e, 0.6, zero_guard=True)
        nb, outs, ts = run(src, enc, opts, q)
        d = outs["none"]
        qdb = q if kind == "dct" else q / 3.0  # surface q is a flat-law step: ~3x mode-0's for the same size
        cands = {"none": d, "deblock (existing)": db(d, qdb, False), "deblock zg": db(d, qdb, True)}
        for k in opts:
            if k != "none":
                cands[k] = outs[k]
        cands["smooth gated2 zg + deblock zg"] = db(outs["smooth gated2 zg"], qdb, True)
        pool = [d, outs["smooth gated1 zg"], outs["smooth gated2 zg"], outs["smooth gated3 zg"]]
        pick = np.argmin(np.stack([per_chunk_mse(src, o) for o in pool]), axis=0)
        sig = np.empty_like(src)
        for i, s in enumerate(blocks(src)):
            sig[s] = pool[pick[i]][s]
        cands["signalled per chunk (2 bits)"] = sig
        base = per_chunk_mse(src, d)
        for cn, out in cands.items():
            m = ct_metrics(src, out)
            m.update(vol=vname, kind=kind, q=q, cand=cn, bpv=(nb + (nch // 4 if cn.startswith("signalled") else 0)) * 8 / src.size,
                     chunks_worse=int((per_chunk_mse(src, out) > base * 1.0001).sum()), chunks=int(len(base)))
            if cn in ts:
                m["ms_per_chunk"] = 1000 * ts[cn] / max(nch, 1)
            if which == "ct":
                dd = out.astype(np.float64) - src
                m.update(psnr_tissue=float(10 * np.log10(255 ** 2 / max(np.mean(dd[src > 0] ** 2), 1e-12))),
                         ssim=ssim_slices(src, out, 32), edge_mae=edge_mae(src, out))
            else:
                pm = metrics(src.astype(np.float32) / 255, out)
                m.update(dice=pm["dice"], disp=pm["disp_mean"], mae_band=pm["mae_band"], p99_band=pm["p99_band"])
            rows.append(m); print(json.dumps(m), flush=True)
json.dump(rows, open(f"/vesuvius/usrm2/volcomp_surface/dbk4_{which}.json", "w"))
