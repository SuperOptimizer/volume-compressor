"""Prototype: level quantisation + context-adaptive binary coding (cost model = ideal adaptive code length,
same probability state machine as volcomp's mask coder, so a range coder lands within ~0.1%)."""
import numpy as np
import numba as nb

# ---------------------------------------------------------------- codebooks


def codebook(edges, recon=None):
    """edges: ascending u8 thresholds t_1..t_{L-1}; bin i = [t_i, t_{i+1}). Returns (enc LUT 256->idx, dec LUT idx->u8)."""
    edges = np.asarray(sorted(set(int(e) for e in edges if 0 < e <= 255)))
    enc = np.searchsorted(edges, np.arange(256), side="right").astype(np.uint8)
    L = len(edges) + 1
    if recon is None:
        lo = np.concatenate([[0], edges])
        hi = np.concatenate([edges, [256]]) - 1
        recon = np.rint((lo + hi) / 2.0)
        recon[0] = 0  # the bottom bin is the dead zone: it decodes to exactly 0
        if hi[-1] == 255 and lo[-1] >= 250:
            recon[-1] = 255
    dec = np.clip(np.asarray(recon), 0, 255).astype(np.uint8)
    assert len(dec) == L
    return enc, dec


def cb_uniform(step, dead=None, thr=128):
    """uniform bins of `step` with an edge at `thr` (so the 0.5 threshold is preserved exactly),
    and a dead zone [0, dead) -> 0 (default: the first bin below the lowest edge)."""
    e = list(range(thr, 256, step)) + list(range(thr - step, 0, -step))
    if dead is not None:
        e = [x for x in e if x > dead] + [dead]
    return codebook(e)


def cb_logit(n_half, lo_p=0.01, dead=None):
    """edges uniform in log-odds between logit(lo_p) and 0 (mirrored above): fine near 0.5."""
    l0 = np.log(lo_p / (1 - lo_p))
    ls = np.linspace(l0, 0, n_half + 1)
    e = np.rint(255 / (1 + np.exp(-ls))).astype(int)
    e = list(e) + list(256 - e[:-1])
    e = [x if x != 128 else 128 for x in e]
    if dead is not None:
        e = [x for x in e if x > dead] + [dead]
    return codebook(sorted(set(e)))


def cb_twostep(fine, coarse, w, dead):
    """step `fine` within +-w of 128, `coarse` outside, dead zone below `dead`."""
    e = set([128])
    x = 128
    while x < 256:
        x += fine if x - 128 < w else coarse
        e.add(min(x, 256))
    x = 128
    while x > dead:
        x -= fine if 128 - x < w else coarse
        e.add(max(x, dead))
    e.add(dead)
    return codebook(sorted(e))


# ---------------------------------------------------------------- coder
PBITS = 12
ONE = 1 << PBITS


@nb.njit(cache=True, inline="always")
def _bit(prob, age, c, bit, maxsh):
    p = prob[c]
    a = age[c]
    sh = 1 + (a >> 1)
    if sh > maxsh:
        sh = maxsh
    age[c] = a + 1 if a < 60 else a
    if bit:
        cost = -np.log2((ONE - p) / ONE)
        p -= p >> sh
    else:
        cost = -np.log2(p / ONE)
        p += (ONE - p) >> sh
    if p < 16:
        p = 16
    if p > ONE - 16:
        p = ONE - 16
    prob[c] = p
    return cost


@nb.njit(cache=True)
def _gcls(g):
    if g == 0:
        return 0
    if g <= 2:
        return 1
    if g <= 6:
        return 2
    if g <= 14:
        return 3
    if g <= 30:
        return 4
    if g <= 62:
        return 5
    return 6


@nb.njit(cache=True)
def code_cost(idx, dec, near, thr_idx, pred, zb, maxsh, maskfirst):
    """idx: (128,128,128) uint8 level indices; dec: idx->value LUT; near: value(0..255 clamp)->nearest idx.
    pred: 0 lorenzo, 1 median(a,b,c,lorenzo), 2 median(a,b,c), 3 = gradient-selected.
    zb: zero-block edge (0 = off). Returns (bits, bits of zero-block flags, bits of mask bits)."""
    N = idx.shape[0]
    L = dec.shape[0]
    NG = 7
    prob = np.full(1 << 16, ONE // 2, np.int32)
    age = np.zeros(1 << 16, np.int32)
    # context bases
    Z0 = 0            # zero flag: NG * 4 pclass * 3 (neighbour residual state) * 2 mask side
    S0 = 2048         # sign
    M0 = 4096         # magnitude unary: NG * 16 positions * 2 side
    B0 = 8192         # zero-block flags: 8 contexts
    K0 = 8448         # mask bits: 4096 (12-neighbour)
    v = np.zeros((N + 2, N + 2, N + 2), np.int32)   # values, padded by 2 at the low end (zeros outside)
    msk = np.zeros((N + 2, N + 2, N + 2), np.int32)
    res0 = np.zeros((N + 2, N + 2, N + 2), np.int32)  # |residual| clipped to 2, for context
    bits = 0.0
    zbits = 0.0
    kbits = 0.0
    # zero blocks
    if zb > 0:
        nbk = N // zb
        flag = np.zeros((nbk + 1, nbk + 1, nbk + 1), np.int32)
        for bz in range(nbk):
            for by in range(nbk):
                for bx in range(nbk):
                    nz = 0
                    for z in range(bz * zb, bz * zb + zb):
                        for y in range(by * zb, by * zb + zb):
                            for x in range(bx * zb, bx * zb + zb):
                                if idx[z, y, x] != 0:
                                    nz = 1
                    flag[bz + 1, by + 1, bx + 1] = nz
                    c = flag[bz, by + 1, bx + 1] + flag[bz + 1, by, bx + 1] * 2 + flag[bz + 1, by + 1, bx] * 4
                    zbits += _bit(prob, age, B0 + c, nz, maxsh)
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if zb > 0 and flag[z // zb + 1, y // zb + 1, x // zb + 1] == 0:
                    continue
                Z, Y, X = z + 2, y + 2, x + 2
                a = v[Z, Y, X - 1]
                b = v[Z, Y - 1, X]
                c = v[Z - 1, Y, X]
                d = v[Z, Y - 1, X - 1]
                e = v[Z - 1, Y, X - 1]
                f = v[Z - 1, Y - 1, X]
                g = v[Z - 1, Y - 1, X - 1]
                lor = a + b + c - d - e - f + g
                if pred == 0:
                    pv = lor
                elif pred == 2 or pred == 1:
                    lo_ = min(a, min(b, c))
                    hi_ = max(a, max(b, c))
                    md = a + b + c - lo_ - hi_
                    if pred == 2:
                        pv = md
                    else:
                        # median of (a,b,c,lor) ~ clamp lor into [lo, hi]
                        pv = min(max(lor, lo_), hi_)
                else:
                    # 2nd-order along the smoothest axis
                    ga = abs(a - v[Z, Y, X - 2])
                    gb = abs(b - v[Z, Y - 2, X])
                    gc = abs(c - v[Z - 2, Y, X])
                    if ga <= gb and ga <= gc:
                        pv = 2 * a - v[Z, Y, X - 2]
                    elif gb <= gc:
                        pv = 2 * b - v[Z, Y - 2, X]
                    else:
                        pv = 2 * c - v[Z - 2, Y, X]
                    lo_ = min(a, min(b, c))
                    hi_ = max(a, max(b, c))
                    pv = min(max(pv, lo_ - 16), hi_ + 16)
                if pv < 0:
                    pv = 0
                if pv > 255:
                    pv = 255
                kp = near[pv]
                mx = max(a, max(b, max(c, max(d, max(e, max(f, g))))))
                mn = min(a, min(b, min(c, min(d, min(e, min(f, g))))))
                gc_ = _gcls(mx - mn)
                k = idx[z, y, x]
                side = 0
                if maskfirst:
                    mb = 1 if k >= thr_idx else 0
                    cx = (msk[Z, Y, X - 1] | msk[Z, Y - 1, X] << 1 | msk[Z - 1, Y, X] << 2 | msk[Z, Y - 1, X - 1] << 3 |
                          msk[Z - 1, Y, X - 1] << 4 | msk[Z - 1, Y - 1, X] << 5 | msk[Z, Y, X - 2] << 6 |
                          msk[Z, Y - 2, X] << 7 | msk[Z - 2, Y, X] << 8 | msk[Z, Y - 1, X + 1 if x + 1 < N else X] << 9 |
                          msk[Z - 1, Y, X + 1 if x + 1 < N else X] << 10 | msk[Z - 1, Y + 1 if y + 1 < N else Y, X] << 11)
                    kbits += _bit(prob, age, K0 + cx, mb, maxsh)
                    msk[Z, Y, X] = mb
                    side = mb
                    if mb == 1 and kp < thr_idx:
                        kp = thr_idx
                    if mb == 0 and kp >= thr_idx:
                        kp = thr_idx - 1
                if kp == 0:
                    pc = 0
                elif kp == 1:
                    pc = 1
                elif kp == L - 1:
                    pc = 3
                else:
                    pc = 2
                rn = res0[Z, Y, X - 1] + res0[Z, Y - 1, X] + res0[Z - 1, Y, X]
                rc = 0 if rn == 0 else (1 if rn <= 2 else 2)
                r = int(k) - int(kp)
                cz = Z0 + ((gc_ * 4 + pc) * 3 + rc) * 2 + side
                bits += _bit(prob, age, cz, 1 if r != 0 else 0, maxsh)
                if r != 0:
                    # sign: skip when only one direction is possible
                    lo_k = 0
                    hi_k = L - 1
                    if maskfirst:
                        if side:
                            lo_k = thr_idx
                        else:
                            hi_k = thr_idx - 1
                    can_neg = kp > lo_k
                    can_pos = kp < hi_k
                    neg = 1 if r < 0 else 0
                    if can_neg and can_pos:
                        bits += _bit(prob, age, S0 + (gc_ * 4 + pc) * 2 + side, neg, maxsh)
                    m = abs(r) - 1
                    lim = (kp - lo_k) if neg else (hi_k - kp)
                    lim -= 1  # max value of m
                    j = 0
                    while j < lim:
                        cm = M0 + (gc_ * 16 + min(j, 15)) * 2 + side
                        stop = 1 if m == j else 0
                        bits += _bit(prob, age, cm, 1 - stop, maxsh)
                        if stop:
                            break
                        j += 1
                v[Z, Y, X] = dec[k]
                res0[Z, Y, X] = min(abs(r), 2)
    return bits + zbits + kbits, zbits, kbits


def near_lut(dec):
    return np.array([int(np.argmin(np.abs(dec.astype(int) - vv))) for vv in range(256)], np.int64)


def encode_cost(c_u8, enc, dec, pred=1, zb=8, maxsh=5, maskfirst=False):
    """returns (bytes, decoded chunk)"""
    idx = enc[c_u8]
    thr_idx = int(enc[128])
    bits, zb_bits, kbits = code_cost(idx, dec.astype(np.int64), near_lut(dec), thr_idx, pred, zb, maxsh, maskfirst)
    hdr = 8 + 1 + 1 + len(dec)  # header, flags, L, LUT
    return int(np.ceil(bits / 8)) + hdr + 4, dec[idx]
