"""Speed: far candidates (dd > NEAR) coded only inside 8^3 blocks flagged 'has a far flip'."""
import sys, os
os.environ["VOLCOMP_LIB"] = "/vesuvius/usrm2/volcomp_surface/vc/build/release/libvolcomp.so"
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc/python")
import numpy as np, numba as nb
from common import *
from ctxcoder import _bit, ONE
from fixexp3 import dcl
import os
os.environ.setdefault("VOLCOMP_LIB", "/vesuvius/usrm2/volcomp_surface/vc/build/release/libvolcomp.so")
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc/python")
import volcomp_zarr._lib as L

@nb.njit(cache=True)
def cost(src, v0, NEAR, B, maxsh):
    N = src.shape[0]
    v = v0.astype(np.int32).copy()
    M = 0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if (src[z, y, x] >= 128) != (v[z, y, x] >= 128):
                    dd = abs(2 * v[z, y, x] - 255)
                    if dd > M:
                        M = dd
    if M == 0:
        return 0.0, 0.0, 0
    prob = np.full(1 << 16, ONE // 2, np.int32)
    age = np.zeros(1 << 16, np.int32)
    nbk = N // B
    has = np.zeros((nbk + 1, nbk + 1, nbk + 1), np.int32)
    anyfar = np.zeros((nbk, nbk, nbk), np.int32)
    fl = 0.0
    if NEAR < M:
        for z in range(N):
            for y in range(N):
                for x in range(N):
                    dd = abs(2 * v[z, y, x] - 255)
                    if dd > NEAR and dd <= M:
                        anyfar[z // B, y // B, x // B] = 1
                        if (src[z, y, x] >= 128) != (v[z, y, x] >= 128):
                            has[z // B + 1, y // B + 1, x // B + 1] = 1
        for bz in range(nbk):
            for by in range(nbk):
                for bx in range(nbk):
                    if anyfar[bz, by, bx]:
                        c = 60000 + has[bz, by + 1, bx + 1] + 2 * has[bz + 1, by, bx + 1] + 4 * has[bz + 1, by + 1, bx]
                        fl += _bit(prob, age, c, has[bz + 1, by + 1, bx + 1], maxsh)
    bits = 0.0; nc = 0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                d = v[z, y, x]
                dd = abs(2 * d - 255)
                if dd > M:
                    continue
                if dd > NEAR and has[z // B + 1, y // B + 1, x // B + 1] == 0:
                    continue
                nc += 1
                s = 1 if d >= 128 else 0
                f = 1 if (src[z, y, x] >= 128) != (s == 1) else 0
                xm = x - 1 if x > 0 else x; ym = y - 1 if y > 0 else y; zm = z - 1 if z > 0 else z
                xp = x + 1 if x + 1 < N else x; yp = y + 1 if y + 1 < N else y; zp = z + 1 if z + 1 < N else z
                pat = ((v[z, y, xm] >= 128) != s) | ((v[z, ym, x] >= 128) != s) << 1 | ((v[zm, y, x] >= 128) != s) << 2 | \
                      ((v[z, ym, xm] >= 128) != s) << 3 | ((v[zm, y, xm] >= 128) != s) << 4 | ((v[zm, ym, x] >= 128) != s) << 5
                b = ((v[z, y, xp] >= 128) != s) + ((v[z, yp, x] >= 128) != s) + ((v[zp, y, x] >= 128) != s)
                c = ((dcl(dd) * 64 + pat) * 4 + b) * 2 + s
                bits += _bit(prob, age, c, f, maxsh)
                if f:
                    v[z, y, x] = 127 if s else 128
    return bits, fl, nc

for teach in ("recto", "m7"):
    for reg in ("evalbox", "r32k"):
        u8 = to_u8(load_cube(reg, teach, 0))
        for q in (48, 96):
            res = []
            for NEAR, B in ((509, 8), (31, 8), (15, 8), (15, 16), (7, 8), (7, 4)):
                tb = tf = tc = 0
                for s, c in chunks(u8):
                    if not c.any():
                        continue
                    e = L.surface_encode(c.tobytes(), q)
                    # the base alone: recover it by decoding the base part (margin 0 stream)
                    base = bytearray(e[:16 + int.from_bytes(e[12:16], "little")]); base[10] = base[11] = 0
                    d = np.frombuffer(bytes(L.decode(bytes(base))), np.uint8).reshape(128, 128, 128)
                    b, fl, nc = cost(c, d, NEAR, B, 4)
                    tb += b; tf += fl; tc += nc
                res.append(f"near{NEAR}/B{B}: {(tb + tf) / 8:.0f}B cand {tc}")
            print(teach, reg, f"q{q}", " | ".join(res), flush=True)
