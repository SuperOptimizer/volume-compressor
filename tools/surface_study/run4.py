"""Dead-zone floor before the DCT (source values < t -> 0; decoded values < t2 -> 0), with and without fix."""
import sys, json
import numpy as np
from common import *
import nl
TEACH, REG, NK = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
QS = [float(x) for x in sys.argv[4].split(",")]
TS = [int(x) for x in sys.argv[5].split(",")]
tag = sys.argv[6] if len(sys.argv) > 6 else "e0.65"
for t in TS:
    for q in QS:
        agg, aggf = [], []
        for reg in REG:
            for k in range(NK):
                p = load_cube(reg, TEACH, k); u8 = to_u8(p); cache = {}
                src = np.where(u8 < t, 0, u8).astype(np.uint8)
                nb_, d = run_chunkwise(src, lib_codec(q=q))
                d = np.where(d < (t + 1) // 2, 0, d).astype(np.uint8)
                m = metrics(p, d, cache=cache); m["bpv"] = nb_ * 8 / u8.size; m["zero_exact"] = float(((d == 0) == (u8 == 0)).mean()); agg.append(m)
                fb = 0; df = d.copy()
                for s, c in chunks(u8):
                    if not c.any():
                        continue
                    b, o = nl.fix_cost(c, np.ascontiguousarray(d[s]), 5); fb += int(np.ceil(b / 8)); df[s] = o
                m2 = metrics(p, df, cache=cache); m2["bpv"] = (nb_ + fb) * 8 / u8.size; aggf.append(m2)
        for lab, a in (("", agg), ("+fix", aggf)):
            r = {kk: float(np.mean([x[kk] for x in a])) for kk in a[0]}
            r.update(tag=f"{tag} t{t}{lab}", q=q, teacher=TEACH)
            print(json.dumps(r), flush=True)
