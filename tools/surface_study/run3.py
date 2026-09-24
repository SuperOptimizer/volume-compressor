"""DCT step-law sweep: VOLCOMP_LIB selects a build with a different VF_HF_EXP. Prints JSON lines."""
import sys, json, os
import numpy as np
from common import *
import nl
TEACH, REG, NK = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
QS = [float(x) for x in sys.argv[4].split(",")]
tag = sys.argv[5]
for q in QS:
    agg, aggf = [], []
    for reg in REG:
        for k in range(NK):
            p = load_cube(reg, TEACH, k); u8 = to_u8(p); cache = {}
            nb_, d = run_chunkwise(u8, lib_codec(q=q))
            m = metrics(p, d, cache=cache); m["bpv"] = nb_ * 8 / u8.size; agg.append(m)
            fb = 0; df = d.copy()
            for s, c in chunks(u8):
                if not c.any():
                    continue
                b, o = nl.fix_cost(c, np.ascontiguousarray(d[s]), 5); fb += int(np.ceil(b / 8)); df[s] = o
            m2 = metrics(p, df, cache=cache); m2["bpv"] = (nb_ + fb) * 8 / u8.size; aggf.append(m2)
    for lab, a in (("", agg), ("+fix", aggf)):
        r = {k: float(np.mean([x[k] for x in a])) for k in a[0]}
        r.update(tag=tag + lab, q=q, teacher=TEACH)
        print(json.dumps(r), flush=True)
