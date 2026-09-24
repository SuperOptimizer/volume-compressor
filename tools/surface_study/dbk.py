"""Deblocking study: decode-only / encode-only / both, on CT octets and teacher probability cubes."""
import os, sys, json, time, glob
os.environ["VOLCOMP_LIB"] = "/vesuvius/usrm2/volcomp_surface/vc/build/release/libvolcomp.so"
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc/python")
import numpy as np
from scipy.fft import dctn, idctn
from scipy import ndimage as ndi
import volcomp_zarr._lib as L
from common import load_cube, to_u8, metrics

C = 128
CORP = "/vesuvius/usrm2/volcomp_surface/corpus/"


def octet(setname, z):
    v = np.zeros((256, 256, 256), np.uint8)
    for f in glob.glob(f"{CORP}{setname}/octet_L0_z{z:04d}_*.u8"):
        b = os.path.basename(f)[:-3].split("_")
        cz, cy, cx = int(b[2][1:]), int(b[3][1:]), int(b[4][1:])
        zz, yy, xx = (cz % 2) * C, (cy % 2) * C, (cx % 2) * C
        v[zz:zz + C, yy:yy + C, xx:xx + C] = np.fromfile(f, np.uint8).reshape(C, C, C)
    return v


def chunkwise(v, fn):
    out = np.empty_like(v); nb = 0
    for z in range(0, v.shape[0], C):
        for y in range(0, v.shape[1], C):
            for x in range(0, v.shape[2], C):
                s = (slice(z, z + C), slice(y, y + C), slice(x, x + C))
                n, d = fn(np.ascontiguousarray(v[s]))
                nb += n; out[s] = d
    return nb, out


def enc_dct(q):
    def f(c):
        e = L.encode(c.tobytes(), q)
        return len(e), np.frombuffer(bytes(L.decode(e)), np.uint8).reshape(C, C, C)
    return f


def enc_surface(q):
    def f(c):
        e = L.surface_encode(c.tobytes(), q)
        return len(e), np.frombuffer(bytes(L.decode(e)), np.uint8).reshape(C, C, C)
    return f


# ---------------- metrics
def blocking(src, dec, B=16):
    """boundary / interior RMSE of the error's first difference (volcomp-bench's blocking amplification),
    plus the same restricted to chunk (128) faces"""
    e = dec.astype(np.int32) - src.astype(np.int32)
    bd, it, ch = [], [], []
    for ax in range(3):
        g = np.diff(e, axis=ax)  # g[i] = e[i+1]-e[i]; boundary between i=B-1 and B -> index B-1
        idx = np.arange(g.shape[ax])
        bmask = (idx % B) == B - 1
        imask = (idx % B) == B // 2 - 1
        cmask = (idx % C) == C - 1
        bd.append(np.take(g, idx[bmask], axis=ax).ravel()); it.append(np.take(g, idx[imask], axis=ax).ravel())
        ch.append(np.take(g, idx[cmask], axis=ax).ravel())
    r = lambda a: float(np.sqrt(np.mean(np.concatenate(a).astype(np.float64) ** 2)))
    return r(bd) / max(r(it), 1e-9), r(ch) / max(r(it), 1e-9)


def ct_metrics(src, dec):
    d = dec.astype(np.float64) - src
    mse = float(np.mean(d * d))
    amp, champ = blocking(src, dec)
    return dict(psnr=10 * np.log10(255 ** 2 / max(mse, 1e-12)), mae=float(np.abs(d).mean()), block_amp=amp, chunkface_amp=champ)


# ---------------- decode-only filters
def f_deblock(d, q, s=1.0):
    v = np.ascontiguousarray(d.copy())
    L.deblock(v.reshape(-1).view(np.uint8), v.shape[0], v.shape[1], v.shape[2], q * s)
    return v


def steps(q, flat):
    st = np.empty(46)
    st[0] = q * 0.125
    st[1:] = q if flat else q * (1 + np.arange(1, 46)) ** 0.65
    k = np.arange(16)
    r = k[:, None, None] + k[None, :, None] + k[None, None, :]
    return st[r]


def pocs(dec, smooth, q, flat=False, iters=1):
    """smooth -> per 16^3 block orthonormal DCT -> clamp every coefficient into the quantisation cell of the
    level the unfiltered decode implies -> inverse. Levels are read back from the unfiltered decode."""
    S = steps(q, flat)
    dz = np.full((16, 16, 16), 0.2); dz[0, 0, 0] = 0.5
    rec = np.full((16, 16, 16), 0.30)
    out = dec.astype(np.float64)
    Z, Y, X = dec.shape
    # levels from the unfiltered decode
    Bn = lambda a: a.reshape(Z // 16, 16, Y // 16, 16, X // 16, 16).transpose(0, 2, 4, 1, 3, 5)
    UB = lambda a: a.transpose(0, 3, 1, 4, 2, 5).reshape(Z, Y, X)
    c0 = dctn(Bn(dec.astype(np.float64) - 128), axes=(3, 4, 5), norm="ortho")
    a = np.abs(c0) / S
    Lv = np.where(a < 0.5, 0, np.rint(a - 0.15))  # |L|=1 reconstructs at 1.15, >=2 at +0.30
    Lv = np.where(Lv >= 2, np.rint(a - 0.30), Lv)
    dc = np.zeros((16, 16, 16), bool); dc[0, 0, 0] = True
    Lv = np.where(dc, np.rint(a), Lv)
    lo = np.where(Lv == 0, -0.8, 0) * S  # for L == 0: |X| < 0.8 step
    sgn = np.sign(c0)
    lo_abs = np.where(Lv == 0, 0.0, (Lv - dz) * S)
    hi_abs = np.where(Lv == 0, (1 - dz) * S, (Lv + 1 - dz) * S)
    for _ in range(iters):
        sm = smooth(np.clip(np.rint(out), 0, 255).astype(np.uint8)).astype(np.float64)
        c = dctn(Bn(sm - 128), axes=(3, 4, 5), norm="ortho")
        cz = np.clip(c, -hi_abs, hi_abs)  # zero cell: [-0.8, 0.8] step
        cn = sgn * np.clip(np.abs(c) * (np.sign(c) == sgn), lo_abs, hi_abs)
        c = np.where(Lv == 0, cz, cn)
        out = UB(idctn(c, axes=(3, 4, 5), norm="ortho")) + 128
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def gauss_faces(sigma):
    """blur only within 2 voxels of a 16-plane (the seams), elsewhere untouched"""
    def f(d):
        b = ndi.gaussian_filter(d.astype(np.float32), sigma)
        idx = np.arange(d.shape[0]) % 16
        near = (idx <= 1) | (idx >= 14)
        m = near[:, None, None] | near[None, :, None] | near[None, None, :]
        return np.where(m, b, d.astype(np.float32))
    return f


if __name__ == "__main__":
    which = sys.argv[1]
    rows = []
    if which == "ct":
        vols = [("tune-octet", octet("tune", 250)), ("heldout-octet", octet("heldout", 350))]
        qs = (4, 8, 16, 32)
    else:
        vols = [(f"{which}-evalbox", to_u8(load_cube("evalbox", which, 0))), (f"{which}-r32k", to_u8(load_cube("r32k", which, 0)))]
        qs = (8, 16, 32)
    for name, src in vols:
        pf = src.astype(np.float32) / 255.0
        for q in qs:
            nb, d = chunkwise(src, enc_dct(q))
            cands = {"none": d,
                     "deblock": f_deblock(d, q),
                     "deblock x2": f_deblock(d, q, 2.0),
                     "pocs(deblock)": pocs(d, lambda v: f_deblock(v, q), q),
                     "pocs(gauss0.8 seams)": pocs(d, gauss_faces(0.8), q),
                     "pocs(gauss0.8 seams) x3": pocs(d, gauss_faces(0.8), q, iters=3),
                     "pocs(gauss1.2 all)": pocs(d, lambda v: ndi.gaussian_filter(v.astype(np.float32), 1.2), q)}
            for cn, out in cands.items():
                m = ct_metrics(src, out); m.update(vol=name, q=q, cand=cn, bpv=nb * 8 / src.size)
                if which != "ct":
                    pm = metrics(pf, out); m.update(dice=pm["dice"], disp=pm["disp_mean"], mae_band=pm["mae_band"])
                rows.append(m); print(json.dumps(m), flush=True)
    json.dump(rows, open(f"/vesuvius/usrm2/volcomp_surface/dbk_{which}.json", "w"))
