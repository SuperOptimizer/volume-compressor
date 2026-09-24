"""Downstream check: the recto teacher on raw vs decoded (and deblocked) CT; predictions compared to the raw-CT one."""
import os, sys, json
sys.path.insert(0, "/home/forrest/usrm2")
import numpy as np, torch
from usrm2.predict import slide
from usrm2 import trt
os.environ["VOLCOMP_LIB"] = "/vesuvius/usrm2/volcomp_surface/vc2/build/release/libvolcomp.so"
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc2/python")
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface")
import volcomp_zarr._lib as L
from dbk3 import region
C = 128
dev = torch.device("cuda")
net = trt.Engine(trt.plan("recto", 256), dev)
fn = lambda t: torch.softmax(net(t), 1)[:, 1]

def predict(v):
    return slide(fn, v, 256, 32, dev)[128:384, 128:384, 128:384]

def codec(src, q, dec):
    out = np.zeros_like(src)
    for z in range(0, 512, C):
        for y in range(0, 512, C):
            for x in range(0, 512, C):
                s = (slice(z, z + C), slice(y, y + C), slice(x, x + C)); c = np.ascontiguousarray(src[s])
                if c.any():
                    out[s] = np.frombuffer(bytes(dec(L.encode(c.tobytes(), float(q)))), np.uint8).reshape(C, C, C)
    return out

def db(v, q, zg):
    w = np.ascontiguousarray(v.copy()); L.deblock(w.reshape(-1).view(np.uint8), *w.shape, float(q), zero_guard=zg); return w

rows = []
for rname, args in (("interior", (269, 118, 144)), ("mask-edge", (269, 158, 226))):
    src = region(rname, *args)
    ref = predict(src)
    m0 = ref >= 0.5
    for q in (4, 8, 16):
        dplain = codec(src, q, L.decode)
        variants = {"none": dplain, "deblock (existing)": db(dplain, q, False), "deblock zg": db(dplain, q, True),
                    "smooth gated2 zg": codec(src, q, lambda e: L.decode_smooth(e, 2.0, gated=True, zero_guard=True))}
        variants["smooth gated2 zg + deblock zg"] = db(variants["smooth gated2 zg"], q, True)
        for vn, v in variants.items():
            p = predict(v); m1 = p >= 0.5
            r = dict(region=rname, q=q, cand=vn, dice=float(2 * (m0 & m1).sum() / max(m0.sum() + m1.sum(), 1)),
                     mad=float(np.abs(p - ref).mean() * 255), corr=float(np.corrcoef(p.ravel(), ref.ravel())[0, 1]))
            rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open("/vesuvius/usrm2/volcomp_surface/downstream.json", "w"))
