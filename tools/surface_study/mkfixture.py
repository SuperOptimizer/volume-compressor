import os, sys, struct
os.environ["VOLCOMP_LIB"] = "/vesuvius/usrm2/volcomp_surface/vc/build/release/libvolcomp.so"
sys.path.insert(0, "/vesuvius/usrm2/volcomp_surface/vc/python")
import numpy as np
import volcomp_zarr._lib as L
FL = "/vesuvius/usrm2/volcomp_surface/float/"
r = np.load(FL + "evalbox_recto.npy", mmap_mode="r"); m = np.load(FL + "evalbox_m7.npy", mmap_mode="r")
best = None
for z in (0, 128):
    for y in range(0, 1024, 128):
        for x in range(0, 1024, 128):
            a = np.asarray(r[z:z + 128, y:y + 128, x:x + 128]); b = np.asarray(m[z:z + 128, y:y + 128, x:x + 128])
            fr, fm = (a >= 0.5).mean(), (b >= 0.5).mean()
            sc = abs(fr - 0.22) + abs(fm - 0.16)
            if best is None or sc < best[0]:
                best = (sc, z, y, x, fr, fm)
print("chunk", best)
_, z, y, x, _, _ = best
streams = []
for arr in (r, m):
    c = np.clip(np.rint(np.asarray(arr[z:z + 128, y:y + 128, x:x + 128], np.float32) * 255), 0, 255).astype(np.uint8)
    streams.append(L.encode(c.tobytes(), 1.0))
out = struct.pack("<I", 2) + b"".join(struct.pack("<I", len(s)) for s in streams) + b"".join(streams)
open("/vesuvius/usrm2/volcomp_surface/vc/tests/data/surface_teachers.vlz", "wb").write(out)
print("fixture bytes", len(out), [len(s) for s in streams], "origin zyx in PHercParis4 level 0:", 34432 + z, 15104 + y, 18432 + x)
