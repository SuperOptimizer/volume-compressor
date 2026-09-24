import json, glob, collections
import numpy as np
rows = [json.loads(l) for f in sorted(glob.glob("/vesuvius/usrm2/volcomp_surface/final_*.jsonl")) for l in open(f)]
import os
for r in rows:
    fb = f"/vesuvius/usrm2/volcomp_surface/rebytes_{r['teacher']}_{r['region']}.json"
    if r["cand"].startswith("surface") and os.path.exists(fb):
        r["bytes"] = json.load(open(fb))[f"{r['cube']} {r['cand']}"]; r["bpv"] = r["bytes"] * 8 / 256 ** 3
order = ["dct q4", "dct q8", "dct q16", "mask5", "surface q32", "surface q48", "surface q64", "surface q96", "surface q128"]
keys = ["dice", "flip", "mae_band", "p99_band", "maxerr", "disp_mean", "disp_p99", "bdist_mean", "sdf_mae", "skel_f1"]
for t in ("recto", "m7"):
    R = [r for r in rows if r["teacher"] == t]
    ref = {(r["region"], r["cube"]): r["bytes"] for r in R if r["cand"] == "dct q8"}
    print(f"\n### {t}: {len(ref)} cubes of 256^3 ({len(ref) * 16.777216:.0f} M voxels)\n")
    print("| candidate | bits/voxel | size vs q8 | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * (len(keys) + 2))
    for c in order:
        rs = [r for r in R if r["cand"] == c]
        tot = sum(r["bytes"] for r in rs); totref = sum(ref[(r["region"], r["cube"])] for r in rs)
        bpv = tot * 8 / (len(rs) * 256 ** 3)
        vals = [np.mean([r.get(k, np.nan) for r in rs]) for k in keys]
        print(f"| {c} | {bpv:.4f} | {tot / totref:.3f} | " + " | ".join(f"{v:.4f}" if k in ('dice', 'flip', 'skel_f1') else f"{v:.3f}" for k, v in zip(keys, vals)) + " |")
    print("\nper region, size vs q8 (surface q48 / q64 / q96):")
    for reg in ("evalbox", "bbox", "r15k", "r32k", "r47k", "r60k"):
        s = []
        for c in ("surface q48", "surface q64", "surface q96"):
            rs = [r for r in R if r["cand"] == c and r["region"] == reg]
            s.append(sum(r["bytes"] for r in rs) / sum(ref[(r["region"], r["cube"])] for r in rs))
        q8 = [r for r in R if r["cand"] == "dct q8" and r["region"] == reg]
        print(f"  {reg:8s} q8 {np.mean([r['bpv'] for r in q8]):.4f} bpv   " + " / ".join(f"{x:.3f}" for x in s))
