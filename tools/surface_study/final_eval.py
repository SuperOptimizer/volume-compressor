"""Final table: the implemented surface mode (C) vs the existing modes, on every study cube.
usage: final_eval.py TEACHER REGION  -> final_<teacher>_<region>.jsonl (one line per cube x candidate)"""
import os, sys, json, time
os.environ["VOLCOMP_LIB"] = "/vesuvius/usrm2/volcomp_surface/vc/build/release/libvolcomp.so"
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc/python")
import numpy as np
import volcomp_zarr._lib as L
from common import load_cube, to_u8, run_chunkwise, metrics, CUBES

def dec(e):
    return np.frombuffer(bytes(L.decode(e)), np.uint8).reshape(128, 128, 128)

def cand(kind, q):
    def f(c):
        raw = c.tobytes()
        if kind == "dct":
            e = L.encode(raw, q)
        elif kind == "surface":
            e = L.surface_encode(raw, q)
        elif kind == "mask5":
            e = L.mask_encode_lossless(np.where(c >= 128, 255, 0).astype(np.uint8).tobytes())
        return len(e), dec(e)
    return f

CANDS = [("dct", 8), ("dct", 16), ("dct", 4), ("mask5", 0)] + [("surface", q) for q in (32, 48, 64, 96, 128)]
teach, reg = sys.argv[1], sys.argv[2]
with open(f"/vesuvius/usrm2/volcomp_surface/final_{teach}_{reg}.jsonl", "w") as out:
    for k in range(len(CUBES[reg])):
        p = load_cube(reg, teach, k); u8 = to_u8(p); cache = {}
        for kind, q in CANDS:
            t0 = time.time()
            nb, d = run_chunkwise(u8, cand(kind, q))
            m = metrics(p, d, skel=True, cache=cache)
            m.update(teacher=teach, region=reg, cube=k, cand=f"{kind} q{q}" if kind != "mask5" else "mask5",
                     bytes=nb, bpv=nb * 8 / u8.size, secs=time.time() - t0)
            out.write(json.dumps(m) + "\n"); out.flush()
