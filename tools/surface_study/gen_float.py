"""Fresh float teacher predictions (no uint8 rounding, no volcomp) for the surface-mode study.
usage: gen_float.py recto|m7 name z0 y0 x0 Z Y X"""
import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/forrest/usrm2")
from usrm2 import data
from usrm2.predict import slide
from usrm2 import trt
model, name = sys.argv[1], sys.argv[2]
z0, y0, x0, Z, Y, X = map(int, sys.argv[3:9])
dev = torch.device("cuda")
out = f"/vesuvius/usrm2/volcomp_surface/float/{name}_{model}.npy"
t0 = time.time()
if model == "recto":
    net = trt.Engine(trt.plan("recto", 256), dev)
    fn = lambda t: torch.softmax(net(t), 1)[:, 1]
    m = 128
    ct = data.open_zarr(data.CT)
    roi = ct[z0:z0 + Z, y0 - m:y0 + Y + m, x0 - m:x0 + X + m]
    p = slide(fn, roi, 256, 32, dev)[:, m:m + Y, m:m + X]
else:
    from usrm2 import m7
    netc = m7.load(dev="cpu"); mean, std, lo, hi = netc.norm
    prep = lambda c, _: ((np.clip(c.astype(np.float32), lo, hi) - mean) / std)[None]
    net = trt.Engine(trt.plan("m7", 192), dev)
    fn = lambda t: torch.softmax(net(t).float(), 1)[:, 1]
    f, margin = 4, 64
    ct = data.open_zarr(data.CT.rstrip("/").rsplit("/", 1)[0] + "/2")
    o, s = np.array([z0, y0, x0]) // f, np.array([Z, Y, X]) // f
    a, b = np.maximum(o - margin, 0), np.minimum(o + s + margin, ct.shape)
    roi = ct[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    r = np.pad(roi, [(0, max(192 - n, 0)) for n in roi.shape])
    c = o - a
    prob = slide(fn, r, 192, 32, dev, prep)[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]]
    up = F.interpolate(torch.from_numpy(prob)[None, None], size=(Z, Y, X), mode="trilinear", align_corners=False)[0, 0].numpy()
    air = np.repeat(np.repeat(np.repeat(roi[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] == 0, f, 0), f, 1), f, 2)
    p = np.where(air, 0, up).astype(np.float32)
np.save(out, p.astype(np.float16 if Z >= 512 else np.float32))
print(out, p.shape, float(p.mean()), f"{time.time() - t0:.0f}s", flush=True)
