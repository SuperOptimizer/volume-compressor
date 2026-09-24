"""Deblocking round 2: per-chunk (intra-chunk) POCS, encode-only error feedback, signalled per-chunk strength,
and the surface mode with a side-preserving POCS."""
import sys, json, time
import numpy as np
from scipy import ndimage as ndi
from dbk import *

def seam_blur(sigma):
    def f(d):
        b = ndi.gaussian_filter(d.astype(np.float32), sigma, mode="nearest")
        idx = np.arange(d.shape[0]) % 16
        near = (idx <= 1) | (idx >= 14)
        m = near[:, None, None] | near[None, :, None] | near[None, None, :]
        return np.where(m, b, d.astype(np.float32))
    return f

def pocs_chunks(dec, q, sigma, flat=False):
    """intra-chunk only: every 128^3 chunk filtered and projected on its own (what a per-chunk decoder can do)"""
    if sigma == 0:
        return dec
    out = np.empty_like(dec)
    for z in range(0, dec.shape[0], C):
        for y in range(0, dec.shape[1], C):
            for x in range(0, dec.shape[2], C):
                s = (slice(z, z + C), slice(y, y + C), slice(x, x + C))
                out[s] = pocs(np.ascontiguousarray(dec[s]), seam_blur(sigma), q, flat)
    return out

def encode_feedback(src, q, alpha):
    """encode-only: re-encode with the face-adjacent error pushed back into the source"""
    nb, d = chunkwise(src, enc_dct(q))
    idx = np.arange(src.shape[0]) % 16
    near = (idx <= 1) | (idx >= 14)
    m = near[:, None, None] | near[None, :, None] | near[None, None, :]
    s2 = np.clip(np.rint(src.astype(np.float32) + alpha * m * (src.astype(np.float32) - d)), 0, 255).astype(np.uint8)
    return chunkwise(s2, enc_dct(q))

def signalled(src, dec, q, sigmas, flat=False):
    """both: the encoder picks each chunk's sigma (2 bits) by MSE against the source"""
    outs = [pocs_chunks(dec, q, s, flat) for s in sigmas]
    best = np.empty_like(dec)
    for z in range(0, dec.shape[0], C):
        for y in range(0, dec.shape[1], C):
            for x in range(0, dec.shape[2], C):
                s = (slice(z, z + C), slice(y, y + C), slice(x, x + C))
                errs = [np.mean((o[s].astype(np.float64) - src[s]) ** 2) for o in outs]
                best[s] = outs[int(np.argmin(errs))][s]
    return best

def side_clamp(out, surf, thr=128):
    return np.where(surf >= thr, np.maximum(out, thr), np.minimum(out, thr - 1)).astype(np.uint8)

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "ct":
        vols = [("tune-octet", octet("tune", 250)), ("heldout-octet", octet("heldout", 350))]
        qs = (4, 8, 16, 32)
    else:
        vols = [(f"{which}-evalbox", to_u8(load_cube("evalbox", which, 0))), (f"{which}-r32k", to_u8(load_cube("r32k", which, 0)))]
        qs = (8, 16, 32)
    rows = []
    def emit(name, q, cand, nbytes, src, out, t=None):
        m = ct_metrics(src, out); m.update(vol=name, q=q, cand=cand, bpv=nbytes * 8 / src.size)
        if t is not None:
            m["filter_s_per_chunk"] = t
        if which != "ct":
            pm = metrics(src.astype(np.float32) / 255, out); m.update(dice=pm["dice"], disp=pm["disp_mean"], mae_band=pm["mae_band"])
        rows.append(m); print(json.dumps(m), flush=True)
    for name, src in vols:
        nch = src.size // C ** 3
        for q in qs:
            nb, d = chunkwise(src, enc_dct(q))
            emit(name, q, "none", nb, src, d)
            for sg in (0.6, 0.8, 1.0):
                t0 = time.time(); o = pocs_chunks(d, q, sg); t = (time.time() - t0) / nch
                emit(name, q, f"D pocs-chunk seam{sg}", nb, src, o, t)
            for a in (0.5, 1.0):
                nb2, d2 = encode_feedback(src, q, a)
                emit(name, q, f"E feedback a{a}", nb2, src, d2)
            o = signalled(src, d, q, (0, 0.6, 0.8, 1.2))
            emit(name, q, "B signalled sigma (2 bits/chunk)", nb + nch // 4, src, o)
        if which != "ct":
            for q in (48, 64, 96):
                nb, d = chunkwise(src, enc_surface(q))
                emit(name, q, "surface", nb, src, d)
                # the base of a surface chunk is flat-law; project against the base's cells, then keep sides
                for sg in (0.6, 0.8):
                    o = side_clamp(pocs_chunks(d, q, sg, flat=True), d)
                    emit(name, q, f"surface + D pocs-chunk seam{sg} + side clamp", nb, src, o)
    json.dump(rows, open(f"/vesuvius/usrm2/volcomp_surface/dbk2_{which}.json", "w"))
