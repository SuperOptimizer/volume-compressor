"""Prototype: near-lossless predictive coding (JPEG-LS style, quantiser in the loop) with the mask coder's
adaptive binary state machine, and the DCT + threshold fix-up layer. Cost = ideal adaptive code length."""
import numpy as np
import numba as nb
from ctxcoder import _bit, _gcls, ONE


@nb.njit(cache=True)
def nl_code(u, delta, dead, zb, maxsh, keep_side, pred):
    """u: (N,N,N) uint8 chunk. Error <= delta except: voxels <= dead reconstruct to 0 (when reachable), and
    keep_side forces the reconstruction onto the source's side of 127.5. Returns (bits, recon)."""
    N = u.shape[0]
    step = 2 * delta + 1
    prob = np.full(1 << 14, ONE // 2, np.int32)
    age = np.zeros(1 << 14, np.int32)
    v = np.zeros((N + 2, N + 2, N + 2), np.int32)
    rq = np.zeros((N + 2, N + 2, N + 2), np.int32)
    out = np.zeros((N, N, N), np.uint8)
    bits = 0.0
    B0 = 8000
    nbk = N // zb if zb > 0 else 1
    flag = np.ones((nbk + 1, nbk + 1, nbk + 1), np.int32)
    if zb > 0:
        flag[:] = 0
        for bz in range(nbk):
            for by in range(nbk):
                for bx in range(nbk):
                    nz = 0
                    for z in range(bz * zb, bz * zb + zb):
                        for y in range(by * zb, by * zb + zb):
                            for x in range(bx * zb, bx * zb + zb):
                                if u[z, y, x] > dead:
                                    nz = 1
                    flag[bz + 1, by + 1, bx + 1] = nz
                    c = flag[bz, by + 1, bx + 1] + flag[bz + 1, by, bx + 1] * 2 + flag[bz + 1, by + 1, bx] * 4
                    bits += _bit(prob, age, B0 + c, nz, maxsh)
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if zb > 0 and flag[z // zb + 1, y // zb + 1, x // zb + 1] == 0:
                    continue
                Z, Y, X = z + 2, y + 2, x + 2
                a = v[Z, Y, X - 1]; b = v[Z, Y - 1, X]; c = v[Z - 1, Y, X]
                d = v[Z, Y - 1, X - 1]; e = v[Z - 1, Y, X - 1]; f = v[Z - 1, Y - 1, X]; g = v[Z - 1, Y - 1, X - 1]
                lor = a + b + c - d - e - f + g
                lo_ = min(a, min(b, c)); hi_ = max(a, max(b, c))
                if pred == 0:
                    pv = lor
                elif pred == 1:
                    pv = min(max(lor, lo_), hi_)
                else:
                    ga = abs(a - v[Z, Y, X - 2]); gb = abs(b - v[Z, Y - 2, X]); gc = abs(c - v[Z - 2, Y, X])
                    if ga <= gb and ga <= gc:
                        pv = 2 * a - v[Z, Y, X - 2]
                    elif gb <= gc:
                        pv = 2 * b - v[Z, Y - 2, X]
                    else:
                        pv = 2 * c - v[Z - 2, Y, X]
                    pv = min(max(pv, lo_ - 2 * step), hi_ + 2 * step)
                pv = min(max(pv, 0), 255)
                mx = max(a, max(b, max(c, max(d, max(e, max(f, g))))))
                mn = min(a, min(b, min(c, min(d, min(e, min(f, g))))))
                gcl = _gcls(mx - mn)
                src = int(u[z, y, x])
                if src <= dead:
                    # aim for exactly 0
                    qe = -((pv + step - 1) // step) if pv > 0 else 0
                    # smallest |qe| that reaches 0 after clamping
                else:
                    err = src - pv
                    qe = (err + delta) // step if err >= 0 else -((-err + delta) // step)
                    rv = min(max(pv + qe * step, 0), 255)
                    if keep_side:
                        s_src = src >= 128
                        if (rv >= 128) != s_src:
                            qe += 1 if s_src else -1
                    # a nonzero source must not decode to 0 where that would change a zero test? (no)
                rv = min(max(pv + qe * step, 0), 255)
                # context: activity class, prediction class, neighbour residual activity
                pc = 0 if pv == 0 else (1 if pv < 64 else (2 if pv < 192 else 3))
                rn = rq[Z, Y, X - 1] + rq[Z, Y - 1, X] + rq[Z - 1, Y, X]
                rc = 0 if rn == 0 else (1 if rn <= 2 else 2)
                cz = ((gcl * 4 + pc) * 3 + rc)
                bits += _bit(prob, age, cz, 1 if qe != 0 else 0, maxsh)
                if qe != 0:
                    neg = 1 if qe < 0 else 0
                    if pv > 0 and pv < 255:
                        bits += _bit(prob, age, 200 + gcl * 4 + pc, neg, maxsh)
                    m = abs(qe) - 1
                    j = 0
                    while True:
                        stop = 1 if m == j else 0
                        bits += _bit(prob, age, 300 + (gcl * 16 + min(j, 15)) * 2 + neg, 1 - stop, maxsh)
                        if stop:
                            break
                        j += 1
                v[Z, Y, X] = rv
                rq[Z, Y, X] = min(abs(qe), 2)
                out[z, y, x] = rv
    return bits, out


def nl_codec(delta, dead, zb=8, maxsh=5, keep_side=True, pred=1):
    def f(c):
        bits, out = nl_code(c, delta, dead, zb, maxsh, keep_side, pred)
        return int(np.ceil(bits / 8)) + 12, out
    return f


@nb.njit(cache=True)
def fix_cost(src, dec, maxsh):
    """Cost of a threshold fix-up layer over a lossy decode: M = 1 + max |dec-127.5| over flipped voxels
    (sent in a byte); every voxel with |dec-127.5| < M codes one bit 'flipped?' (context: distance class and
    whether a causal neighbour flipped). Flipped voxels become 128 / 127. Returns (bits, fixed)."""
    N = src.shape[0]
    M = 0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if (src[z, y, x] >= 128) != (dec[z, y, x] >= 128):
                    dd = abs(2 * int(dec[z, y, x]) - 255)  # 2|d-127.5|
                    if dd > M:
                        M = dd
    out = dec.copy()
    if M == 0:
        return 8.0, out
    prob = np.full(4096, ONE // 2, np.int32)
    age = np.zeros(4096, np.int32)
    fl = np.zeros((N + 1, N + 1, N + 1), np.int32)
    bits = 8.0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                dd = abs(2 * int(dec[z, y, x]) - 255)
                if dd <= M:
                    f = 1 if (src[z, y, x] >= 128) != (dec[z, y, x] >= 128) else 0
                    nbf = fl[z + 1, y + 1, x] + fl[z + 1, y, x + 1] + fl[z, y + 1, x + 1]
                    c = min(dd // 2, 31) * 4 + min(nbf, 3)
                    bits += _bit(prob, age, c, f, maxsh)
                    if f:
                        fl[z + 1, y + 1, x + 1] = 1
                        out[z, y, x] = 128 if src[z, y, x] >= 128 else 127
    return bits, out
