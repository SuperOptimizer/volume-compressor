"""Fix-layer contexts, round 2: absolute distance classes + final/base neighbour sides + mask-coder pattern."""
import sys, numpy as np, numba as nb
from common import *
from ctxcoder import _bit, ONE

@nb.njit(cache=True)
def dcls(dd, scheme):
    if scheme == 0:
        if dd <= 1: return 0
        if dd <= 3: return 1
        if dd <= 7: return 2
        if dd <= 15: return 3
        if dd <= 31: return 4
        return 5
    else:
        if dd <= 1: return 0
        if dd <= 3: return 1
        if dd <= 5: return 2
        if dd <= 9: return 3
        if dd <= 15: return 4
        if dd <= 23: return 5
        if dd <= 35: return 6
        return 7

@nb.njit(cache=True)
def fixc(src, dec, variant, scheme, maxsh):
    N = src.shape[0]
    M = 0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if (src[z, y, x] >= 128) != (dec[z, y, x] >= 128):
                    dd = abs(2 * int(dec[z, y, x]) - 255)
                    if dd > M:
                        M = dd
    if M == 0:
        return 8.0
    prob = np.full(1 << 16, ONE // 2, np.int32)
    age = np.zeros(1 << 16, np.int32)
    P = 2
    fin = np.zeros((N + 2 * P, N + 2 * P, N + 2 * P), np.int32)
    for z in range(N):
        for y in range(N):
            for x in range(N):
                fin[z + P, y + P, x + P] = 1 if dec[z, y, x] >= 128 else 0
    # outside the chunk: replicate nearest (edge) so the border is not a fake boundary
    bits = 8.0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                d = int(dec[z, y, x])
                dd = abs(2 * d - 255)
                if dd > M:
                    continue
                Z, Y, X = z + P, y + P, x + P
                s = 1 if d >= 128 else 0
                f = 1 if (src[z, y, x] >= 128) != (s == 1) else 0
                dc = dcls(dd, scheme)
                xm = X - 1 if x > 0 else X
                ym = Y - 1 if y > 0 else Y
                zm = Z - 1 if z > 0 else Z
                xp = X + 1 if x + 1 < N else X
                yp = Y + 1 if y + 1 < N else Y
                zp = Z + 1 if z + 1 < N else Z
                a = (fin[Z, Y, xm] != s) + (fin[Z, ym, X] != s) + (fin[zm, Y, X] != s)
                b = (fin[Z, Y, xp] != s) + (fin[Z, yp, X] != s) + (fin[zp, Y, X] != s)
                if variant == 0:
                    c = (dc * 4 + a) * 4 + b
                elif variant == 1:
                    # plus side s itself (asymmetric bands) and diagonal causal disagreement
                    e = (fin[Z, ym, xm] != s) + (fin[zm, Y, xm] != s) + (fin[zm, ym, X] != s)
                    c = (((dc * 4 + a) * 4 + b) * 2 + s) * 4 + e
                else:
                    # causal pattern bits (6 face+diag) and noncausal count
                    pat = (fin[Z, Y, xm] != s) | (fin[Z, ym, X] != s) << 1 | (fin[zm, Y, X] != s) << 2 | \
                          (fin[Z, ym, xm] != s) << 3 | (fin[zm, Y, xm] != s) << 4 | (fin[zm, ym, X] != s) << 5
                    c = ((dc * 64 + pat) * 4 + b) * 2 + s
                bits += _bit(prob, age, c, f, maxsh)
                if f:
                    fin[Z, Y, X] = 1 - s
    return bits

teach = sys.argv[1]
for reg in ("evalbox", "r32k"):
    p = load_cube(reg, teach, 0); u8 = to_u8(p)
    for q in (8, 16, 32):
        base = 0; res = {}
        m5 = 0
        for s, c in chunks(u8):
            if not c.any():
                continue
            n, d = lib_codec(q=q)(c)
            base += n
            if q == 8:
                m5 += lib_codec(mode="mask-ll")(np.where(c >= 128, 255, 0).astype(np.uint8))[0]
            for var in (0, 1, 2):
                for sch in (0, 1):
                    for msh in (4, 5):
                        res[(var, sch, msh)] = res.get((var, sch, msh), 0) + fixc(c, d, var, sch, msh) / 8
        best = sorted(res.items(), key=lambda kv: kv[1])
        print(teach, reg, f"q{q} dct {base} B;" + (f" mask5 {m5} B;" if q == 8 else ""), " best:", " ".join(f"{k}:{v:.0f}" for k, v in best[:6]), flush=True)
