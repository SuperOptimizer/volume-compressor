import sys, numpy as np
sys.path.insert(0, "/home/forrest/usrm2")
from usrm2 import data
base = data.CT.rstrip("/").rsplit("/", 1)[0]
L = 4; f = 1 << L
ct = data.open_zarr(base + f"/{L}")
print("level", L, ct.shape, "level0 shape", data.open_zarr(data.CT).shape)
v = ct[:]
n = 1024 // f
best = []
for z in range(0, v.shape[0] - n + 1, n // 2):
    for y in range(0, v.shape[1] - n + 1, n // 2):
        for x in range(0, v.shape[2] - n + 1, n // 2):
            c = v[z:z + n, y:y + n, x:x + n]
            best.append(((c > 0).mean(), float(c.mean()), z * f, y * f, x * f))
best.sort(reverse=True)
import collections
byz = collections.defaultdict(list)
[byz[b[2] // 8192].append(b) for b in best if b[0] >= 0.95]
for k in sorted(byz): print("zband", k * 8192, len(byz[k]), byz[k][len(byz[k])//2], byz[k][0])
sys.exit()
for b in best[:40]:
    print("fill %.3f mean %.1f  z %d y %d x %d" % b)
