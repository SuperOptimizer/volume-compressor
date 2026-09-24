"""What the raw outputs cost: float32 / float16 / u8 through zstd, per voxel, on the study cubes."""
import sys, numpy as np
from numcodecs import Zstd, Blosc
from common import *
z19 = Zstd(level=19)
bl = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
for teach in ("recto", "m7"):
    acc = {}
    n = 0
    for reg in ("evalbox", "r32k"):
        p = load_cube(reg, teach, 0); n += p.size
        for name, arr in (("float32", p), ("float16", p.astype(np.float16)), ("u8", to_u8(p)), ("mask u8 0/255", np.where(p >= 0.5, 255, 0).astype(np.uint8))):
            acc.setdefault(name + " raw", 0); acc[name + " raw"] += arr.nbytes
            acc.setdefault(name + " zstd19", 0); acc[name + " zstd19"] += len(z19.encode(arr.tobytes()))
            acc.setdefault(name + " blosc-zstd5-bitshuf", 0); acc[name + " blosc-zstd5-bitshuf"] += len(bl.encode(np.ascontiguousarray(arr)))
    for k, v in acc.items():
        print(f"{teach:6s} {k:28s} {v * 8 / n:8.4f} bits/voxel")
