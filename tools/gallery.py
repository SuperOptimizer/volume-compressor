#!/usr/bin/env python3
"""Index of the comparison images: 256-px-wide thumbnails under docs/comparison/thumbs/ and
docs/comparison/GALLERY.md (grouped CT cubes / surfaces / masks, one line per image saying what it shows), plus the
two summary READMEs made from the per-scroll report.json files: docs/comparison/surfaces/README.md (all scrolls,
sizes and agreement) and docs/comparison/masks/README.md (published masks as volcomp mask stores).

    python3 tools/gallery.py docs/comparison
Needs numpy and Pillow. Reads only report.json files and PNGs already in the tree.
"""
import json
import os
import sys

import numpy as np

TW = 256  # thumbnail width
CT_ORDER = ["PHerc0500P2_0.55um", "PHerc0139_1.13um", "PHerc1667_1.13um", "PHerc0343P_2.21um", "PHercParis4_2.40um",
            "PHerc0500P2_4.32um", "PHerc0800_8.64um", "PHerc0191_9.36um"]
SURF_ORDER = ["PHerc0343P", "PHerc0009B", "PHerc0800", "PHerc1218", "PHerc0175A", "PHerc0268", "PHerc1447",
              "PHerc0125"]
QS = [1, 2, 4, 8, 16, 32]


def panels(img, n, gap, top=0):
    """Split a strip of n equal panels separated by `gap` px columns; `top` px label bar dropped."""
    w = (img.shape[1] - gap * (n - 1)) // n
    return [img[top:, i * (w + gap):i * (w + gap) + w] for i in range(n)]


def thumb(src, dst, pick=None, n=1, gap=0, top=0):
    """256-px-wide thumbnail of src; for strips, only the panels `pick` (indices) side by side."""
    from PIL import Image
    im = Image.open(src)
    a = np.asarray(im.convert("RGB"))
    if pick is not None:
        ps = panels(a, n, gap, top)
        sep = np.full((ps[0].shape[0], max(2, ps[0].shape[1] // 64), 3), 255, np.uint8)
        parts = []
        for i, k in enumerate(pick):
            parts += ([sep] if i else []) + [ps[k]]
        a = np.concatenate(parts, 1)
    t = Image.fromarray(a)
    t = t.resize((TW, max(1, round(a.shape[0] * TW / a.shape[1]))), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t.save(dst, optimize=True)


def fmt_size(b):
    return f"{b / 2**30:.2f} GiB" if b >= 2**30 else f"{b / 2**20:.0f} MiB"


def ct_rows(root, rep):
    rows = []
    for name in [n for n in CT_ORDER if n in rep]:
        r = rep[name]
        q8, q32 = r["q"]["8"], r["q"]["32"]
        sam, vox = r["sample"], r["voxel"].replace("um", " µm")
        tag = f"{sam} {vox}"
        items = [
            (f"{name}/zmid_all.png", (0, 4, 6), 7, 0, 22,
             f"{tag}, centre z plane: raw and q 1-32 (plain decode); q 8 {q8['ratio']:.0f}x at "
             f"{q8['psnr']:.1f} dB, q 32 {q32['ratio']:.0f}x at {q32['psnr']:.1f} dB"),
            (f"{name}/zmid_all_db.png", (0, 4, 6), 7, 0, 22,
             f"{tag}, the same with the deblocking post-filter; q 32 {q32['deblocked']['psnr']:.1f} dB"),
            (f"{name}/ymid_all.png", (0, 4, 6), 7, 0, 22, f"{tag}, centre y plane (z vertical), raw and q 1-32"),
            (f"{name}/xmid_all.png", (0, 4, 6), 7, 0, 22, f"{tag}, centre x plane (z vertical), raw and q 1-32"),
            (f"{name}/diff_zmid_q8.png", None, 1, 0, 0,
             f"{tag}, q 8 error overlay on the centre z plane (red opacity = 4 x abs(error) / 255; max error {q8['max']:.0f})"),
            (f"{name}/diff_zmid_q32.png", None, 1, 0, 0,
             f"{tag}, q 32 error overlay on the centre z plane (max error {q32['max']:.0f})"),
        ]
        rows.append((f"### {tag} ([table](README.md#{name.lower().replace('.', '')}))", items))
    return rows


PENDING = "numbers pending: the whole-volume pass is still running (in progress)"


def views_on_disk(d, prefix=""):
    """View names of the `<prefix><view>_all.png` strips in d, z first, then y, x (the report's order)."""
    import glob
    vs = [os.path.basename(f)[len(prefix):-len("_all.png")] for f in glob.glob(os.path.join(d, prefix + "*_all.png"))]
    return sorted(vs, key=lambda v: ("zyx".index(v[0]), int(v[1:])))


def surf_rows(root):
    rows = []
    for s in SURF_ORDER:
        p = os.path.join(root, "surfaces", s, "report.json")
        if not os.path.exists(p):
            vs = views_on_disk(os.path.join(root, "surfaces", s))
            if not vs:
                continue
            items = []
            for n in vs[:2]:
                items += [(f"surfaces/{s}/{n}_all.png", (1, 2, 4), 5, 6, 0,
                           f"{s} level 0, slice {n}: CT, published m7 mask, ours plain / smoothed, diff (thumb: "
                           f"published, ours, diff); {PENDING}"),
                          (f"surfaces/{s}/{n}_zoom.png", (1, 2, 4), 5, 6, 0,
                           f"{s} slice {n}, the 256² crop with the most surface at 2x: same five panels")]
            items.append((f"surfaces/{s}/{vs[0]}_diff.png", None, 1, 0, 0,
                          f"{s} slice {vs[0]} full resolution disagreement: red published only, blue ours only "
                          "(p>=0.5), grey both"))
            rows.append((f"### {s} (in progress)", items))
            continue
        r = json.load(open(p))
        pitch = r["info"]["ours_levels"][0]
        items = []
        for v in r["views"][:2]:
            n, x = v["view"], v["slice_scores"]
            items += [
                (f"surfaces/{s}/{n}_all.png", (1, 2, 4), 5, 6, 0,
                 f"{s} {pitch} µm, slice {n}: CT, published m7 mask, ours plain / smoothed, diff (thumb: published, "
                 f"ours, diff); slice dice {x['dice']:.3f} at p>=0.5"),
                (f"surfaces/{s}/{n}_zoom.png", (1, 2, 4), 5, 6, 0,
                 f"{s} slice {n}, the 256² crop with the most surface at 2x: same five panels"),
            ]
        n = r["views"][0]["view"]
        items.append((f"surfaces/{s}/{n}_diff.png", None, 1, 0, 0,
                      f"{s} slice {n} full resolution disagreement: red published only, blue ours only (p>=0.5), grey both"))
        rows.append((f"### {s} ([README](surfaces/{s}/README.md))", items))
    return rows


def mask_rows(root):
    rows = []
    for s in SURF_ORDER:
        p = os.path.join(root, "masks", s, "report.json")
        if not os.path.exists(p):
            vs = views_on_disk(os.path.join(root, "masks", s), "mask_")
            if vs:
                v = vs[0]
                rows.append((f"### {s} mask (in progress)", [
                    (f"masks/{s}/mask_{v}_all.png", (0, 1, 3), 4, 6, 0,
                     f"{s} slice {v}: published mask, mode 4 decoded (ramp), mode 5 decoded (exact), mode 4 vs "
                     f"published (thumb: published, mode 4, diff); {PENDING}"),
                    (f"masks/{s}/mask_{v}_zoom.png", (0, 1, 3), 4, 6, 0,
                     f"{s} slice {v}, 256² crop at 2x: the same four panels")]))
            continue
        r = json.load(open(p))
        v = r["view"]
        m = r["mode4_vs_published_inside_ct_mask"]
        rows.append((f"### {s} mask ([README](masks/README.md))", [
            (f"masks/{s}/mask_{v}_all.png", (0, 1, 3), 4, 6, 0,
             f"{s} slice {v}: published mask, mode 4 decoded (ramp), mode 5 decoded (exact), mode 4 vs published "
             f"(thumb: published, mode 4, diff); mode 4 dice {m['dice']:.3f}, mode 5 mismatches {r['mode5_mismatch_voxels']}"),
            (f"masks/{s}/mask_{v}_zoom.png", (0, 1, 3), 4, 6, 0, f"{s} slice {v}, 256² crop at 2x: the same four panels"),
        ]))
    return rows


def write_gallery(root, rep):
    L = ["# Comparison gallery", "",
         "Every thumbnail links to the full-resolution lossless PNG. Numbers are in the README and report.json next "
         "to each image: CT cubes [README.md](README.md), surfaces [surfaces/README.md](surfaces/README.md), masks "
         "[masks/README.md](masks/README.md). Strip thumbnails show only some panels (named per row); the full strip "
         "has them all. Made by `tools/gallery.py`.", ""]
    for title, groups in (("CT cubes (512³, q 1-32)", ct_rows(root, rep)),
                          ("Surface predictions: ours (volcomp q8 probability) vs the published m7 mask", surf_rows(root)),
                          ("Published masks as volcomp mask stores", mask_rows(root))):
        L += [f"## {title}", ""]
        for head, items in groups:
            L += [head, "", "| thumbnail | shows |", "|---|---|"]
            for rel, pick, n, gap, top, text in items:
                src = os.path.join(root, rel)
                if not os.path.exists(src):
                    continue
                t = os.path.join("thumbs", rel.replace("/", "__"))
                thumb(src, os.path.join(root, t), pick, n, gap, top)
                L.append(f"| [![{os.path.basename(rel)}]({t})]({rel}) | [{rel}]({rel}): {text} |")
            L.append("")
    open(os.path.join(root, "GALLERY.md"), "w").write("\n".join(L))


def write_surfaces_readme(root):
    reps = {}
    for s in SURF_ORDER:
        p = os.path.join(root, "surfaces", s, "report.json")
        if os.path.exists(p):
            reps[s] = json.load(open(p))
    L = ["# m7 surface predictions: our volcomp store vs the published upstream mask", "",
         "For each scroll, our m7 surface prediction (volcomp q8, **8-bit probability**) is compared with the "
         "**published** upstream m7 prediction (**binary mask, threshold 0.2**, blosc) of the **same scan at the same "
         "level**. Made by `tools/surface_compare.py`; per-scroll images, tables and captions are in the scroll "
         "folders, every number in their report.json. A thumbnail index is in [../GALLERY.md](../GALLERY.md).", "",
         "PHerc0343P and PHerc0009B compare at level 2 of 2.2/2.4 µm scans (8.86/9.60 µm); the other six at level 0 "
         "of 8.64/9.36 µm scans, where upstream publishes m7 level-0 masks. Our stores are separate inference runs "
         "of the same checkpoint (see each README for stride and CT input), so the disagreement mixes inference "
         "differences (stride, blending, fp16 TensorRT vs the upstream predictor, q8 CT input) with compression. On "
         "the one box where an uncompressed float prediction was kept, compression alone costs about 0.02 dice "
         "([PHerc0343P](PHerc0343P/README.md#compression-only)).", "",
         "## Summary", "",
         "| scroll | scan | level (pitch) | voxels | published L0 | ours L0 | ratio | stride | dice p>=0.2 box | "
         "dice p>=0.2 whole | dice p>=0.5 box | dice p>=0.5 whole | published in masked air |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tp = to = 0
    for s, r in reps.items():
        i, z = r["info"], r["sizes"]
        tp, to = tp + z["pub_l0"], to + z["ours_l0"]
        w = r.get("whole", {}).get("plain")
        b = r["box"]["plain"]
        wh = r.get("whole", {})
        n_air = wh.get("published_in_air")
        n_pub_all = wh.get("all_voxels_plain", {}).get("p>=0.2", {}).get("n_pub")
        air = f"{n_air / n_pub_all * 100:.0f} %" if n_air is not None and n_pub_all else "-"
        L.append(f"| [{s}]({s}/README.md) | {i['ct_name'].split('-masked')[0]} | {i['level']} ({i['ours_levels'][0]} µm) | "
                 f"{r['voxels'] / 1e9:.0f} Gvox | {fmt_size(z['pub_l0'])} | {fmt_size(z['ours_l0'])} | "
                 f"{z['pub_l0'] / z['ours_l0']:.2f}x | {(r.get('producer_params') or {}).get('stride', '?')} | "
                 f"{b['p>=0.2']['dice']:.3f} | {w['p>=0.2']['dice']:.3f} | {b['p>=0.5']['dice']:.3f} | "
                 f"{w['p>=0.5']['dice']:.3f} | {air} |" if w else
                 f"| [{s}]({s}/README.md) | {i['ct_name'].split('-masked')[0]} | {i['level']} | {r['voxels'] / 1e9:.0f} Gvox | "
                 f"{fmt_size(z['pub_l0'])} | {fmt_size(z['ours_l0'])} | {z['pub_l0'] / z['ours_l0']:.2f}x | ? | "
                 f"{b['p>=0.2']['dice']:.3f} | - | {b['p>=0.5']['dice']:.3f} | - | - |")
    for x in SURF_ORDER:
        if x not in reps and views_on_disk(os.path.join(root, "surfaces", x)):
            L.append(f"| {x} | level 0 | in progress: images below are final, numbers pending | | | | | | | | | | |")
    L += [f"| **all {len(reps)} with numbers** | | | | **{fmt_size(tp)}** | **{fmt_size(to)}** | **{tp / to:.2f}x** | | | | | | |", "",
          "Sizes are level 0 only (published: S3 listing of the blosc chunks; ours: HEAD of every shard). Dice is "
          "our plain decode at p>=0.2 (the published threshold) or p>=0.5 against the published mask, counted inside "
          "the CT mask (CT > 0); `box` is the 1024³ window with the most published surface inside the CT mask, "
          "`whole` every level-0 voxel. The last column is the share of published surface voxels (whole volume) that "
          "lie where the CT is 0; they are left out of the scores. Decode smoothing moves the scores by at most "
          "0.0005 (the first two scrolls' READMEs list gated2 and gauss0.6 too).", "",
          "**Reading the images.** Amber = surface over the CT: solid for the published binary mask, opacity 0.6·p "
          "for our probability. In the diff panels red = published only, blue = ours only (p >= 0.5), grey = both, "
          "dark red = published surface in masked air. Since the published threshold (0.2) is below ours (0.5), a "
          "thin red rim along every sheet is expected even from identical predictions.", ""]
    open(os.path.join(root, "surfaces", "README.md"), "w").write("\n".join(L))


def write_masks_readme(root):
    reps = {}
    for s in SURF_ORDER:
        p = os.path.join(root, "masks", s, "report.json")
        if os.path.exists(p):
            reps[s] = json.load(open(p))
    pend = [x for x in SURF_ORDER if x not in reps and views_on_disk(os.path.join(root, "masks", x), "mask_")]
    if not reps and not pend:
        return
    L = ["# Published m7 masks as volcomp mask stores", "",
         "The mirror carries every published surface mask re-exported as a volcomp **mode 5** store (exact binary "
         "mask, 0/255, context-model range coder on the full 128³ chunk; same OME-zarr v3 layout, same grid). "
         "**Mode 4** stores the 2x2x2 majority pool of each chunk and decodes it as a trilinear 0..255 ramp; it is "
         "not on the mirror, so its level-0 size is measured here by encoding every non-empty 128³ chunk of the "
         "published level 0 (payload + a 16 B/slot index per non-empty 1024³ shard). Made by "
         "`tools/surface_compare.py --mask-out`; numbers in each scroll's report.json.", "",
         "## Sizes (level 0)", "",
         "| scroll | voxels | published (blosc-zstd, 192³ chunks) | mode 5 (exact) | published / mode 5 | mode 4 (2x pooled) | "
         "published / mode 4 | all levels: published / mode 5 |", "|---|---|---|---|---|---|---|---|"]
    for s, r in reps.items():
        z = r["sizes"]
        m5l0 = r["mask_sizes"][r["levels"][0]]
        m4 = r.get("mode4_level0", {}).get("bytes")
        r4 = f"{z['pub_l0'] / m4:.1f}x" if m4 else "-"
        L.append(f"| {s} | {np.prod(r['shape']) / 1e9:.0f} Gvox | {fmt_size(z['pub_l0'])} | {fmt_size(m5l0)} | "
                 f"{z['pub_l0'] / m5l0:.1f}x | {fmt_size(m4) if m4 else '-'} | {r4} | "
                 f"{fmt_size(z['pub_all'])} / {fmt_size(sum(r['mask_sizes'].values()))} |")
    L += [f"| {x} | in progress: numbers pending | | | | | | |" for x in pend]
    L += ["", "## Fidelity in the comparison box", "",
          "The same 1024³ box as the surface comparison (surfaces/<scroll>/README.md). Mode 5 is read back from the "
          "mirror store; mode 4 is encoded here on the store's 128³ grid and thresholded at 128 for the counts.", "",
          "| scroll | mode 5 mismatching voxels | mode 4 dice (inside CT mask) | precision | recall | mode 4 mismatching voxels | "
          "mode 4 MAE vs 0/255 | box bytes mode 4 / mode 5 |", "|---|---|---|---|---|---|---|---|"]
    for s, r in reps.items():
        m = r["mode4_vs_published_inside_ct_mask"]
        L.append(f"| {s} | {r['mode5_mismatch_voxels']:,} | {m['dice']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                 f"{r['mode4_mismatch_voxels']:,} | {r['mode4_mae_u8']:.2f} | "
                 f"{r['mode4_box_bytes'] / 2**20:.2f} / {r['mode5_box_bytes_encoded_here'] / 2**20:.2f} MiB |")
    L += ["", "## Panels", "",
          "Four panels per strip: the published mask (amber over the CT), mode 4 decoded (the 2x-pooled mask as a "
          "trilinear ramp, grey), mode 5 decoded (exact), and mode 4 >= 128 against the published mask (red = "
          "published only, blue = mode 4 only, grey = both). `_all` is the slice at half resolution, `_zoom` the "
          "256² crop with the most surface at 2x.", ""]
    for s in SURF_ORDER:
        if s in reps:
            v = reps[s]["view"]
        elif s in pend:
            v = views_on_disk(os.path.join(root, "masks", s), "mask_")[0]
        else:
            continue
        L += [f"### {s}", "", f"![{s} {v}]({s}/mask_{v}_all.png)", "", f"![{s} {v} zoom]({s}/mask_{v}_zoom.png)", ""]
    open(os.path.join(root, "masks", "README.md"), "w").write("\n".join(L))


def main(root):
    rep = json.load(open(os.path.join(root, "report.json")))
    write_surfaces_readme(root)
    write_masks_readme(root)
    write_gallery(root, rep)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/comparison")
