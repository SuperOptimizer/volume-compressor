"""Surface-mode study: data, metrics, and the existing-codec candidates."""
import os
import numpy as np
from scipy import ndimage as ndi

os.environ.setdefault("VOLCOMP_LIB", "/vesuvius/usrm2/volcomp_surface/vc/build/release/libvolcomp.so")
import sys
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc/python")
import volcomp_zarr._lib as vl  # noqa: E402

FLOAT = "/vesuvius/usrm2/volcomp_surface/float/"
# (region, teacher) -> list of 256^3 cube origins inside the region file
CUBES = {
    "evalbox": [(0, 256, 256), (0, 512, 640)],
    "bbox": [(0, 256, 256), (0, 640, 512)],
    "r15k": [(256, 256, 256), (512, 512, 640)],
    "r32k": [(256, 256, 256), (512, 512, 640)],
    "r47k": [(256, 256, 256), (512, 512, 640)],
    "r60k": [(256, 256, 256), (512, 512, 640)],
}
C = 128


def load_cube(region, teacher, k, n=256):
    p = np.load(f"{FLOAT}{region}_{teacher}.npy", mmap_mode="r")
    z, y, x = CUBES[region][k]
    return np.asarray(p[z:z + n, y:y + n, x:x + n], dtype=np.float32)


def to_u8(p):
    return np.clip(np.rint(p * 255.0), 0, 255).astype(np.uint8)


def chunks(v):
    """yield (slices, contiguous 128^3 array) over a cube whose sides are multiples of 128"""
    for z in range(0, v.shape[0], C):
        for y in range(0, v.shape[1], C):
            for x in range(0, v.shape[2], C):
                s = (slice(z, z + C), slice(y, y + C), slice(x, x + C))
                yield s, np.ascontiguousarray(v[s])


def run_chunkwise(u8, enc_fn):
    """enc_fn(chunk u8) -> (nbytes, decoded chunk u8). Returns (total bytes, decoded cube)."""
    out = np.empty_like(u8)
    nb = 0
    for s, c in chunks(u8):
        if not c.any():
            out[s] = 0  # zarr stores all-zero chunks as missing: 0 bytes
            continue
        n, d = enc_fn(c)
        nb += n
        out[s] = d
    return nb, out


def lib_codec(q=None, mode=None):
    def f(c):
        raw = c.tobytes()
        if mode == "mask":
            e = vl.mask_encode(raw)
        elif mode == "mask-ll":
            e = vl.mask_encode_lossless(raw)
        else:
            e = vl.encode(raw, q)
        return len(e), np.frombuffer(bytes(vl.decode(e)), np.uint8).reshape(c.shape)
    return f


# ---------------------------------------------------------------- metrics
def _sdf(m):
    return ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)  # >0 outside the mask


def _boundary(m):
    return m & ~ndi.binary_erosion(m)


def metrics(p, d, skel=False, cache=None):
    """p: float prob cube, d: decoded u8 cube. Units: u8 levels (1/255) and voxels."""
    P = p * 255.0
    D = d.astype(np.float32)
    cache = cache if cache is not None else {}
    if "m0" not in cache:
        m0 = P >= 127.5
        cache["m0"] = m0
        cache["sd0"] = _sdf(m0)
        cache["b0"] = _boundary(m0)
        g = np.gradient(P)
        cache["g0"] = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
    m0, sd0, b0, g0 = cache["m0"], cache["sd0"], cache["b0"], cache["g0"]
    m1 = D >= 127.5
    inter = (m0 & m1).sum()
    dice = 2 * inter / max(m0.sum() + m1.sum(), 1)
    err = np.abs(D - P)
    band = np.abs(sd0) <= 3
    near = (np.abs(P - 127.5) < g0) & (g0 > 4)  # voxels whose 0.5 iso-surface lies within ~1 voxel
    disp = err[near] / g0[near]  # first-order shift of the iso-surface, voxels
    r = dict(dice=dice, flip=float((m0 != m1).mean()), mae=float(err.mean()), mae_band=float(err[band].mean()),
             p99_band=float(np.percentile(err[band], 99)), maxerr=float(err.max()),
             disp_mean=float(disp.mean()), disp_p99=float(np.percentile(disp, 99)))
    # voxel-level boundary displacement and distance-field error
    if (m0 != m1).any():
        b1 = _boundary(m1)
        d01 = ndi.distance_transform_edt(~b1)[b0]
        d10 = ndi.distance_transform_edt(~b0)[b1]
        r["bdist_mean"] = float((d01.sum() + d10.sum()) / max(len(d01) + len(d10), 1))
        r["bdist_max"] = float(max(d01.max(initial=0), d10.max(initial=0)))
        sd1 = _sdf(m1)
        w = np.abs(sd0) <= 16
        r["sdf_mae"] = float(np.abs(sd1 - sd0)[w].mean())
    else:
        r.update(bdist_mean=0.0, bdist_max=0.0, sdf_mae=0.0)
    if skel:
        from skimage.morphology import skeletonize
        sub = (slice(64, 192), slice(0, 256), slice(0, 256))
        if "s0" not in cache:
            cache["s0"] = skeletonize(m0[sub]) > 0
        s0 = cache["s0"]
        s1 = skeletonize(m1[sub]) > 0 if (m0 != m1).any() else s0
        if s0.any() and s1.any():
            prec = (ndi.distance_transform_edt(~s0)[s1] <= 1.75).mean()
            rec = (ndi.distance_transform_edt(~s1)[s0] <= 1.75).mean()
            r["skel_f1"] = float(2 * prec * rec / max(prec + rec, 1e-9))
            r["skel_exact"] = float((s0 == s1).all())
        else:
            r["skel_f1"] = float(s0.any() == s1.any())
    return r
