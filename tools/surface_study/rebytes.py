"""Surface-mode byte counts with the current lib (decoded output is unchanged by refinement-coding changes)."""
import json, glob, sys
from common import load_cube, to_u8, chunks, CUBES
import volcomp_zarr._lib as L
t, reg = sys.argv[1], sys.argv[2]
out = {}
for k in range(len(CUBES[reg])):
    u8 = to_u8(load_cube(reg, t, k))
    for q in (32, 48, 64, 96, 128):
        out[f"{k} surface q{q}"] = sum(len(L.surface_encode(c.tobytes(), q)) for s, c in chunks(u8) if c.any())
json.dump(out, open(f"/vesuvius/usrm2/volcomp_surface/rebytes_{t}_{reg}.json", "w"))
