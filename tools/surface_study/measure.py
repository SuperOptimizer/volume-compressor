"""Per-store statistics of the current q=8 teacher stores (step 1 of the surface-mode study)."""
import os, sys, json, glob, random
import numpy as np
from numcodecs import Zstd
os.environ.setdefault("VOLCOMP_LIB", "/home/forrest/volume-compressor/build/release/libvolcomp.so")
import volcomp_zarr as vz
T = "/vesuvius/usrm2/teacher/"
STORES = ["eval.zarr", "a.zarr", "b.zarr", "p4val256_m7_tta0.zarr", "a_m7.zarr", "b_m7.zarr"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
dz = Zstd()
res = {}
for s in STORES:
    meta = json.load(open(T + s + "/zarr.json"))
    shape = meta["shape"]; nch = np.prod([-(-a // 128) for a in shape])
    files = [f for f in glob.glob(T + s + "/c/*/*/*") if os.path.isfile(f)]
    fb = sum(os.path.getsize(f) for f in files)
    random.seed(0); pick = random.sample(files, min(N, len(files)))
    hist = np.zeros(256, np.int64); vbytes = []; ent = []; f0 = []; f255 = []; fg = []; band = []
    for f in pick:
        raw = open(f, "rb").read()
        try: raw = bytes(dz.decode(raw))
        except Exception: pass
        vbytes.append(len(raw))
        a = np.frombuffer(bytes(vz.decode(raw)), np.uint8)
        h = np.bincount(a, minlength=256); hist += h
        p = h[h > 0] / a.size; ent.append(float(-(p * np.log2(p)).sum()))
        f0.append(h[0] / a.size); f255.append(h[255] / a.size); fg.append(h[128:].sum() / a.size)
        band.append(h[1:255].sum() / a.size)
    p = hist / hist.sum()
    r = dict(shape=shape, chunks_total=int(nch), chunks_stored=len(files), store_bytes=fb,
             bpv_store=fb / np.prod(shape), bpv_stored_chunks=fb / (len(files) * 128**3),
             sample_bpv_volcomp=float(np.mean(vbytes)) / 128**3, sample_bytes_per_chunk=float(np.mean(vbytes)),
             ent_order0_bits=float(np.mean(ent)), frac0=float(np.mean(f0)), frac255=float(np.mean(f255)),
             frac_fg=float(np.mean(fg)), frac_band_1_254=float(np.mean(band)),
             hist_coarse=[float(p[i:i + 16].sum()) for i in range(0, 256, 16)])
    res[s] = r
    print(s, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "hist_coarse"}))
    print("   hist/16:", " ".join(f"{x:.4f}" for x in r["hist_coarse"]), flush=True)
json.dump(res, open("/vesuvius/usrm2/volcomp_surface/measure_stores.json", "w"), indent=1)
