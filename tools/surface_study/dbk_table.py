import json
ct = json.load(open("/vesuvius/usrm2/volcomp_surface/dbk4_ct.json"))
names = ["none", "deblock (existing)", "deblock zg", "smooth gated2 zg", "smooth gated3 zg", "smooth gauss0.6 zg",
         "smooth gated2 zg + deblock zg", "signalled per chunk (2 bits)"]
print("### CT: PSNR gain over no filter (dB) / chunks worse than no filter\n")
vols = ["interior 512^3", "mask-edge 512^3", "tune octet", "heldout octet"]
qs = [2, 4, 8, 16]
print("| candidate | " + " | ".join(f"{v.split()[0]} q{q}" for v in vols for q in qs) + " |")
print("|---|" + "---|" * (len(vols) * len(qs)))
base = {(r["vol"], r["q"]): r for r in ct if r["cand"] == "none"}
for n in names:
    cells = []
    for v in vols:
        for q in qs:
            r = [x for x in ct if x["vol"] == v and x["q"] == q and x["cand"] == n][0]
            cells.append(f"{r['psnr'] - base[(v, q)]['psnr']:+.2f} / {r['chunks_worse']}")
    print(f"| {n} | " + " | ".join(cells) + " |")
print("\n### CT at q8 and q16: all metrics\n")
print("| volume | q | candidate | PSNR | tissue PSNR | SSIM (z slices) | block amp | chunk-face amp | mask-edge MAE | ms/chunk |")
print("|---|---|---|---|---|---|---|---|---|---|")
for v in vols:
    for q in (8, 16):
        for n in ["none", "deblock (existing)", "deblock zg", "smooth gated2 zg", "smooth gauss0.6 zg", "smooth gated2 zg + deblock zg"]:
            r = [x for x in ct if x["vol"] == v and x["q"] == q and x["cand"] == n][0]
            ms = f"{r['ms_per_chunk']:.0f}" if "ms_per_chunk" in r else ""
            print(f"| {v} | {q} | {n} | {r['psnr']:.2f} | {r['psnr_tissue']:.2f} | {r['ssim']:.4f} | {r['block_amp']:.2f} | {r['chunkface_amp']:.2f} | {r['edge_mae']:.2f} | {ms} |")
ds = json.load(open("/vesuvius/usrm2/volcomp_surface/downstream.json"))
print("\n### Downstream: recto teacher on decoded CT vs on the raw CT (interior 512^3, central 256^3 scored)\n")
print("| q | candidate | dice at 0.5 | mean abs diff (u8) | corr |")
print("|---|---|---|---|---|")
for r in ds:
    if r["region"] == "interior":
        print(f"| {r['q']} | {r['cand']} | {r['dice']:.4f} | {r['mad']:.2f} | {r['corr']:.5f} |")
for t in ("recto", "m7"):
    pr = json.load(open(f"/vesuvius/usrm2/volcomp_surface/dbk4_{t}.json"))
    print(f"\n### {t} probability maps (evalbox + r32k cubes, mean)\n")
    print("| setting | candidate | PSNR | block amp | dice | disp mean | band MAE | chunks worse |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, q in (("dct", 8), ("dct", 16), ("surface", 48), ("surface", 64)):
        for n in ["none", "deblock (existing)", "smooth gated2 zg", "smooth gauss0.6 zg", "signalled per chunk (2 bits)"]:
            rs = [x for x in pr if x["kind"] == kind and x["q"] == q and x["cand"] == n]
            m = lambda k: sum(x[k] for x in rs) / len(rs)
            print(f"| {kind} q{q} | {n} | {m('psnr'):.2f} | {m('block_amp'):.2f} | {m('dice'):.4f} | {m('disp'):.3f} | {m('mae_band'):.2f} | {sum(x['chunks_worse'] for x in rs)}/{sum(x['chunks'] for x in rs)} |")
