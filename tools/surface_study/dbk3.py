"""CT deblocking evaluation on raw (uncompressed) PHercParis4 CT: decode-only / encode+decode candidates in C."""
import os, sys, json, time, glob
import numpy as np
from dbk import octet, ct_metrics, C
import volcomp_zarr._lib as L

RAW = "/vesuvius/usrm2/volcomp_surface/rawct/"

def region(name, z0, y0, x0, n=4):
    v = np.zeros((n * C,) * 3, np.uint8)
    for z in range(n):
        for y in range(n):
            for x in range(n):
                f = f"{RAW}{name}/{z0 + z}_{y0 + y}_{x0 + x}.u8"
                if os.path.exists(f):
                    v[z * C:(z + 1) * C, y * C:(y + 1) * C, x * C:(x + 1) * C] = np.fromfile(f, np.uint8).reshape(C, C, C)
    return v

def ssim_slices(a, b, step=16):
    from skimage.metrics import structural_similarity
    return float(np.mean([structural_similarity(a[z], b[z], data_range=255) for z in range(4, a.shape[0], step)]))

def edge_mae(src, dec):
    m = src > 0
    from scipy import ndimage as ndi
    band = m & ~ndi.binary_erosion(m, iterations=2) | (~m & ndi.binary_dilation(m, iterations=2))
    return float(np.abs(dec.astype(np.int32) - src)[band].mean()) if band.any() else 0.0

def per_chunk_mse(src, dec):
    out = []
    for z in range(0, src.shape[0], C):
        for y in range(0, src.shape[1], C):
            for x in range(0, src.shape[2], C):
                s = (slice(z, z + C), slice(y, y + C), slice(x, x + C))
                out.append(float(np.mean((dec[s].astype(np.float64) - src[s]) ** 2)))
    return np.array(out)

if __name__ == "__main__":
    vols = [("interior 512^3", region("interior", 269, 118, 144)), ("mask-edge 512^3", region("edge", 269, 158, 226)),
            ("tune octet", octet("tune", 250)), ("heldout octet", octet("heldout", 350))]
    rows = []
    for vname, src in vols:
        nz = int((src > 0).sum())
        for q in (2, 4, 8, 16):
            streams, d = {}, np.empty_like(src)
            outs = {k: np.empty_like(src) for k in ("smooth0.6", "smooth0.6 zg", "smooth1.0 zg")}
            ts = {k: 0.0 for k in outs}
            nb = 0
            for z in range(0, src.shape[0], C):
                for y in range(0, src.shape[1], C):
                    for x in range(0, src.shape[2], C):
                        s = (slice(z, z + C), slice(y, y + C), slice(x, x + C))
                        c = np.ascontiguousarray(src[s])
                        if not c.any():
                            d[s] = 0
                            for k in outs: outs[k][s] = 0
                            continue
                        e = L.encode(c.tobytes(), float(q)); nb += len(e)
                        d[s] = np.frombuffer(bytes(L.decode(e)), np.uint8).reshape(C, C, C)
                        for k, (sg, zg) in (("smooth0.6", (0.6, False)), ("smooth0.6 zg", (0.6, True)), ("smooth1.0 zg", (1.0, True))):
                            t0 = time.time()
                            outs[k][s] = np.frombuffer(bytes(L.decode_smooth(e, sg, zero_guard=zg)), np.uint8).reshape(C, C, C)
                            ts[k] += time.time() - t0
            def db(v, zg):
                w = np.ascontiguousarray(v.copy()); L.deblock(w.reshape(-1).view(np.uint8), *w.shape, float(q), zero_guard=zg); return w
            cands = {"none": d, "deblock (existing)": db(d, False), "deblock zg": db(d, True),
                     "smooth0.6 (per chunk)": outs["smooth0.6"], "smooth0.6 zg": outs["smooth0.6 zg"],
                     "smooth1.0 zg": outs["smooth1.0 zg"], "smooth0.6 zg + deblock zg": db(outs["smooth0.6 zg"], True)}
            # both: the encoder signals per chunk which of {none, smooth0.6 zg, smooth1.0 zg} to apply (2 bits)
            opts = [d, outs["smooth0.6 zg"], outs["smooth1.0 zg"]]
            pick = np.argmin(np.stack([per_chunk_mse(src, o) for o in opts]), axis=0)
            sig = np.empty_like(src); i = 0
            for z in range(0, src.shape[0], C):
                for y in range(0, src.shape[1], C):
                    for x in range(0, src.shape[2], C):
                        s = (slice(z, z + C), slice(y, y + C), slice(x, x + C)); sig[s] = opts[pick[i]][s]; i += 1
            cands["signalled per chunk (none/0.6/1.0 zg)"] = sig
            base_mse = per_chunk_mse(src, d)
            for cn, out in cands.items():
                m = ct_metrics(src, out)
                dd = out.astype(np.float64) - src
                m.update(vol=vname, q=q, cand=cn, bpv=nb * 8 / src.size,
                         psnr_tissue=float(10 * np.log10(255 ** 2 / max(np.mean(dd[src > 0] ** 2), 1e-12))) if nz else 0,
                         ssim=ssim_slices(src, out), edge_mae=edge_mae(src, out),
                         chunks_worse=int((per_chunk_mse(src, out) > base_mse * 1.0001).sum()), chunks=len(base_mse))
                if cn.startswith("smooth"):
                    m["ms_per_chunk"] = 1000 * ts[cn.split(" (")[0].split(" +")[0]] / max(1, int((d.reshape(-1) >= 0).size // C ** 3))
                rows.append(m); print(json.dumps(m), flush=True)
    json.dump(rows, open("/vesuvius/usrm2/volcomp_surface/dbk3_ct.json", "w"))
