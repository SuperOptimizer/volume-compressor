"""Where do refinement bits go (by distance class), and do richer contexts help?"""
import sys, numpy as np, numba as nb
from common import *
from ctxcoder import _bit, ONE

@nb.njit(cache=True)
def dcl(dd):
    if dd <= 1: return 0
    if dd <= 3: return 1
    if dd <= 7: return 2
    if dd <= 15: return 3
    if dd <= 31: return 4
    return 5

@nb.njit(cache=True)
def fixc(src, dec, variant, maxsh, stats):
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
        return 0.0
    prob = np.full(1 << 20, ONE // 2, np.int32)
    age = np.zeros(1 << 20, np.int32)
    v = dec.astype(np.int32).copy()
    bits = 0.0
    for z in range(N):
        for y in range(N):
            for x in range(N):
                d = v[z, y, x]
                dd = abs(2 * d - 255)
                if dd > M:
                    continue
                s = 1 if d >= 128 else 0
                f = 1 if (src[z, y, x] >= 128) != (s == 1) else 0
                xm = x - 1 if x > 0 else x; ym = y - 1 if y > 0 else y; zm = z - 1 if z > 0 else z
                xp = x + 1 if x + 1 < N else x; yp = y + 1 if y + 1 < N else y; zp = z + 1 if z + 1 < N else z
                pat = ((v[z, y, xm] >= 128) != s) | ((v[z, ym, x] >= 128) != s) << 1 | ((v[zm, y, x] >= 128) != s) << 2 | \
                      ((v[z, ym, xm] >= 128) != s) << 3 | ((v[zm, y, xm] >= 128) != s) << 4 | ((v[zm, ym, x] >= 128) != s) << 5
                b = ((v[z, y, xp] >= 128) != s) + ((v[z, yp, x] >= 128) != s) + ((v[zp, y, x] >= 128) != s)
                dc = dcl(dd)
                if variant == 0:
                    c = ((dc * 64 + pat) * 4 + b) * 2 + s
                elif variant == 1:
                    # + closeness of the forward neighbours' base values (any within 7 of thr?)
                    nclose = (abs(2 * v[z, y, xp] - 255) <= 15) + (abs(2 * v[z, yp, x] - 255) <= 15) + (abs(2 * v[zp, y, x] - 255) <= 15)
                    c = (((dc * 64 + pat) * 4 + b) * 2 + s) * 4 + nclose
                elif variant == 2:
                    # + the signed step of the base across the voxel along the steepest axis (does the base cross thr nearby)
                    gx = v[z, y, xp] - v[z, y, xm]; gy = v[z, yp, x] - v[z, ym, x]; gz = v[zp, y, x] - v[zm, y, x]
                    g = max(abs(gx), max(abs(gy), abs(gz)))
                    gc = 0 if g <= 2 else (1 if g <= 6 else (2 if g <= 14 else 3))
                    c = (((dc * 64 + pat) * 4 + b) * 2 + s) * 4 + gc
                else:
                    # dd relative to local gradient: expected distance to the surface in voxels
                    gx = v[z, y, xp] - v[z, y, xm]; gy = v[z, yp, x] - v[z, ym, x]; gz = v[zp, y, x] - v[zm, y, x]
                    g2 = gx * gx + gy * gy + gz * gz
                    # dist ~ (dd/2) / (|g|/2) = dd/|g|
                    r = dd * dd * 16 // (g2 + 1)  # 16*dist^2
                    rc = 0 if r < 4 else (1 if r < 16 else (2 if r < 64 else (3 if r < 256 else 4)))
                    c = (((rc * 6 + dc) * 64 + pat) * 4 + b) * 2 + s
                bb = _bit(prob, age, c, f, maxsh)
                bits += bb
                stats[dc, 0] += 1; stats[dc, 1] += f; stats[dc, 2] += bb
                if f:
                    v[z, y, x] = 127 if s else 128
    return bits

if __name__ == '__main__':
  teach = sys.argv[1]
  for reg in ("evalbox", "r32k"):
      p = load_cube(reg, teach, 0); u8 = to_u8(p)
      for q in (16,):
          res = {}
          st = np.zeros((6, 3))
          for s, c in chunks(u8):
              if not c.any():
                  continue
              n, d = lib_codec(q=q)(c)
              for var in (0, 1, 2, 3):
                  for msh in (4,):
                      stt = np.zeros((6, 3))
                      res[var] = res.get(var, 0) + fixc(c, d, var, msh, stt) / 8
                      if var == 0: st += stt
          print(teach, reg, f"q{q}", " ".join(f"v{k}:{v:.0f}" for k, v in res.items()))
          for k in range(6):
              print(f"   dcls {k}: candidates {st[k,0]:9.0f} flips {st[k,1]:8.0f} bytes {st[k,2]/8:8.0f}")
