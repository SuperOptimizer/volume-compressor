#!/usr/bin/env python3
"""Visual comparison of volcomp quality levels on raw scroll CT.

For each volume a 512^3 cube of the ORIGINAL uint8 data (the open-data bucket stores the level-0 chunks
uncompressed) is fetched, encoded chunk by chunk (128^3) at q = 1, 2, 4, 8, 16, 32 and decoded again, once as
a plain decode (what a zarr reader gets) and once followed by the optional `volcomp_deblock()` post-filter
(`q<N>` / `q<N>_db`). Nine 512^2 views of every version are written as lossless greyscale PNGs -- the six faces of
the cube and the three planes through its exact centre -- plus, per view, two strips that concatenate raw and the
six q levels side by side: `<view>_all.png` (no deblocking) and `<view>_all_db.png` (deblocked). A README with size
ratio, PSNR, MAE and error percentiles per cube, q and filter is written next to the images.

The cube is chosen automatically: the densest 512^3 window (mean intensity) of the middle 512 slices, found on
the 16x downsampled level and snapped to the 128-voxel chunk grid, so every cube is papyrus rather than air.

    VOLCOMP_LIB=build/release/libvolcomp.so python3 tools/compare_images.py docs/comparison [--set first|more|all]
        [--no-per-q] [--stats-only]
Needs numpy, scipy (SSIM), zarr>=3, s3fs, Pillow and python/volcomp_zarr (the ctypes binding).
"""
import json
import os
import sys
import time

import numpy as np

VOLUMES = [  # sample, volume (ESRF 2025/2026 scans only), label
    ("PHerc0500P2", "20250821110041-0.550um-0.1m-65keV-masked.zarr", "0.55um"),
    ("PHerc1667", "20260323082859-1.129um-0.2m-59keV-masked.zarr", "1.13um"),
    ("PHercParis4", "20260411134726-2.400um-0.2m-78keV-masked.zarr", "2.40um"),
    ("PHerc0191", "20250821151635-9.362um-1.2m-113keV-masked.zarr", "9.36um"),
]
MORE_VOLUMES = [  # the second set (other scrolls, other pitches); `--set more`
    ("PHerc0139", "20260413113053-1.129um-0.2m-59keV-masked.zarr", "1.13um"),
    ("PHerc0343P", "20260304131111-2.215um-0.4m-111keV-masked.zarr", "2.21um"),
    ("PHerc0500P2", "20250528085330-4.317um-1.2m-111keV-masked.zarr", "4.32um"),
    ("PHerc0800", "20250521135224-8.640um-1.2m-116keV-masked.zarr", "8.64um"),
]
SETS = {"first": VOLUMES, "more": MORE_VOLUMES, "all": VOLUMES + MORE_VOLUMES}
QS = [1, 2, 4, 8, 16, 32]
N = 512
C = 128
BUCKET = "vesuvius-challenge-open-data"
VIEWS = ["z0", "z511", "y0", "y511", "x0", "x511", "zmid", "ymid", "xmid"]


def open_level(fs, sample, volume, level):
    import zarr
    return zarr.open(zarr.storage.FsspecStore(fs, path=f"{BUCKET}/{sample}/volumes/{volume}/{level}"), mode="r")


def pick_cube(fs, sample, volume, level=4):
    """Origin (z, y, x) of the densest 512^3 window of the middle slab, chunk-aligned."""
    a0 = open_level(fs, sample, volume, 0)
    Z, Y, X = a0.shape
    f = 2 ** level
    n = N // f  # window size at this level
    a = open_level(fs, sample, volume, level)
    zc = Z // 2 // f
    slab = np.asarray(a[max(zc - n // 2, 0):max(zc - n // 2, 0) + n]).astype(np.float32)
    m = slab.mean(0)  # (Y/f, X/f) mean over the slab's depth
    # box filter (n x n) via cumulative sums
    cs = np.pad(m, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    box = cs[n:, n:] - cs[:-n, n:] - cs[n:, :-n] + cs[:-n, :-n]
    iy, ix = np.unravel_index(box.argmax(), box.shape)
    org = [(zc - n // 2) * f, iy * f, ix * f]
    org = [int(min(max(round(o / C) * C, 0), s - N)) for o, s in zip(org, (Z, Y, X))]
    return org, (Z, Y, X)


def fetch_cube(fs, sample, volume, org):
    a = open_level(fs, sample, volume, 0)
    z, y, x = org
    return np.ascontiguousarray(np.asarray(a[z:z + N, y:y + N, x:x + N]), dtype=np.uint8)


def roundtrip(cube, q):
    """Encode every 128^3 chunk at q and decode it back; returns (decoded cube, encoded bytes)."""
    from volcomp_zarr import _lib
    out = np.empty_like(cube)
    nbytes = 0
    for z in range(0, N, C):
        for y in range(0, N, C):
            for x in range(0, N, C):
                blk = np.ascontiguousarray(cube[z:z + C, y:y + C, x:x + C])
                enc = _lib.encode(blk.tobytes(), float(q))
                nbytes += len(enc)
                out[z:z + C, y:y + C, x:x + C] = np.frombuffer(_lib.decode(enc), np.uint8).reshape(C, C, C)
    return out, nbytes


def deblocked(dec, q):
    from volcomp_zarr import _lib
    buf = bytearray(dec.tobytes())
    _lib.deblock(buf, N, N, N, float(q))
    return np.frombuffer(bytes(buf), np.uint8).reshape(N, N, N)


def views(v):
    h = N // 2
    return {"z0": v[0], "z511": v[-1], "y0": v[:, 0], "y511": v[:, -1], "x0": v[:, :, 0], "x511": v[:, :, -1],
            "zmid": v[h], "ymid": v[:, h], "xmid": v[:, :, h]}


def ssim(a, b, sigma=1.5):
    """Mean 3-D SSIM (Gaussian window sigma 1.5 voxels, K1 0.01, K2 0.03, data range 255) over the cube."""
    from scipy.ndimage import gaussian_filter
    x, y = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda v: gaussian_filter(v, sigma, mode="reflect", truncate=3.5)  # noqa: E731
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    return float((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean())


def stats(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = float((d * d).mean())
    ad = np.abs(d)
    return {"psnr": 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"), "mae": float(ad.mean()),
            "p99": float(np.percentile(ad, 99)), "max": float(ad.max()), "ssim": ssim(a, b)}


def save_png(path, img):
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8), "L").save(path, optimize=True)  # PNG is lossless


def strip(panels, labels, path, bar=22):
    """Panels side by side with a label bar above each (greyscale, lossless)."""
    from PIL import Image, ImageDraw
    W = N * len(panels)
    img = Image.new("L", (W, N + bar), 0)
    d = ImageDraw.Draw(img)
    for i, (p, l) in enumerate(zip(panels, labels)):
        img.paste(Image.fromarray(np.ascontiguousarray(p, dtype=np.uint8), "L"), (i * N, bar))
        d.text((i * N + 6, 5), l, fill=255)
    img.save(path, optimize=True)


def main(out_root, vols=VOLUMES, stats_only=False, per_q=True):
    """stats_only: recompute the numbers (report.json, README) without rewriting any image.
    per_q=False: write only the raw views and the strips (`<view>_all*.png`), not the 108 single q views."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    os.makedirs(out_root, exist_ok=True)
    rp = os.path.join(out_root, "report.json")
    report = json.load(open(rp)) if os.path.exists(rp) else {}
    for sample, volume, res in vols:
        t0 = time.time()
        name = f"{sample}_{res}"
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        org, shape = pick_cube(fs, sample, volume)
        cube = fetch_cube(fs, sample, volume, org)
        print(f"{name}: volume {shape}, cube origin zyx {org}, mean {cube.mean():.1f}, fetched in {time.time() - t0:.0f} s", flush=True)
        vers = {"raw": (cube, cube.nbytes, None)}
        for q in QS:
            dec, nb = roundtrip(cube, q)
            vers[f"q{q}"] = (dec, nb, stats(cube, dec))
            db = deblocked(dec, q)
            vers[f"q{q}_db"] = (db, nb, stats(cube, db))
            s, t = vers[f"q{q}"][2], vers[f"q{q}_db"][2]
            print(f"  q={q:<3d} {cube.nbytes / nb:7.1f}x  PSNR {s['psnr']:.1f} dB  SSIM {s['ssim']:.4f}  MAE {s['mae']:.2f}  P99 {s['p99']:.0f}  max {s['max']:.0f}"
                  f"   | deblocked: PSNR {t['psnr']:.1f} dB  SSIM {t['ssim']:.4f}  MAE {t['mae']:.2f}  P99 {t['p99']:.0f}  max {t['max']:.0f}", flush=True)
            if stats_only:  # keep memory flat: only the numbers are needed
                vers[f"q{q}"] = (None, nb, s)
                vers[f"q{q}_db"] = (None, nb, t)
        if not stats_only:
            allviews = {k: views(v[0]) for k, v in vers.items()}
            for k in vers:
                if per_q or k == "raw":
                    for vn, img in allviews[k].items():
                        save_png(os.path.join(d, f"{vn}_{k}.png"), img)
            for vn in VIEWS:
                for suf, tag in (("", ""), ("_db", " deblocked")):
                    keys = ["raw"] + [f"q{q}{suf}" for q in QS]
                    labels = [f"{name} {vn}  raw ({cube.nbytes / 2 ** 20:.0f} MiB)"]
                    labels += [f"q={q}{tag}  {cube.nbytes / vers[k][1]:.0f}x  PSNR {vers[k][2]['psnr']:.1f} dB  MAE {vers[k][2]['mae']:.2f}" for q, k in zip(QS, keys[1:])]
                    strip([allviews[k][vn] for k in keys], labels, os.path.join(d, f"{vn}_all{suf}.png"))
        prev = report.get(name, {})
        if prev and prev.get("origin_zyx") != org:
            raise SystemExit(f"{name}: cube origin {org} differs from the report's {prev.get('origin_zyx')}")
        report[name] = {"sample": sample, "volume": volume, "voxel": res, "shape": list(shape), "origin_zyx": org,
                        "raw_bytes": int(cube.nbytes), "per_q_views": prev.get("per_q_views", per_q) if stats_only else per_q,
                        "q": {str(q): {"bytes": int(vers[f"q{q}"][1]), "ratio": cube.nbytes / vers[f"q{q}"][1], **vers[f"q{q}"][2],
                                       "deblocked": vers[f"q{q}_db"][2]} for q in QS}}
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
        with open(rp, "w") as f:
            json.dump(report, f, indent=1)
    write_readme(out_root, report)


def write_readme(out_root, report):
    known = [f"{a}_{c}" for a, _, c in SETS["all"]]
    names = sorted(report, key=lambda n: known.index(n) if n in known else len(known))
    L = ["# volcomp quality comparison on raw scroll CT", "",
         f"512^3 cubes of ORIGINAL uint8 CT (the open-data bucket's uncompressed level-0 chunks) from {len(names)} scans "
         "(ESRF 2025/2026) of seven scrolls at seven voxel sizes, encoded at q = 1 .. 32 (128^3 chunks) and decoded twice: "
         "plain (`q<N>`, what a zarr reader gets) and with the optional `volcomp_deblock()` post-filter (`q<N>_db`). "
         "Nine 512^2 lossless PNG views per cube: the six cube faces (`z0 z511 y0 y511 x0 x511`) and the three "
         "centre planes (`zmid ymid xmid`); `<view>_raw.png` is the original, `<view>_all.png` puts raw and the six "
         "q levels side by side without deblocking, `<view>_all_db.png` with it. The first four cubes also have "
         "every single q view (`<view>_q<N>[_db].png`); the later four omit them (the strips hold the same pixels). "
         "The cube is the densest 512^3 window of the middle 512 slices, chunk-aligned (not picked by hand). "
         "Generated by `tools/compare_images.py` (`--set first`, `--set more`); numbers in `report.json`. "
         "SSIM is the mean 3-D SSIM over the cube (Gaussian window sigma 1.5 voxels, K1 0.01, K2 0.03, range 255). "
         "A thumbnail index of every image is in [GALLERY.md](GALLERY.md).", "",
         "## Summary at q = 8 (by voxel size)", "",
         "| cube | voxel | ratio | PSNR dB | SSIM | deblocked PSNR | SSIM |", "|---|---|---|---|---|---|---|"]
    for name in sorted(names, key=lambda n: float(report[n]["voxel"].rstrip("um"))):
        r = report[name]
        s = r["q"]["8"]
        t = s["deblocked"]
        L.append(f"| [{name}](#{name.lower().replace('.', '')}) | {r['voxel'].replace('um', ' µm')} | {s['ratio']:.1f}x | "
                 f"{s['psnr']:.1f} | {s.get('ssim', float('nan')):.4f} | {t['psnr']:.1f} | {t.get('ssim', float('nan')):.4f} |")
    L.append("")
    for name in names:
        r = report[name]
        L += [f"## {name}", "", f"`{r['sample']}/volumes/{r['volume']}`, volume shape {r['shape']}, cube origin zyx {r['origin_zyx']}, raw {r['raw_bytes'] / 2 ** 20:.0f} MiB.", "",
              "| q | size | ratio | PSNR dB | SSIM | MAE | P99 | max | deblocked PSNR | SSIM | MAE | P99 | max |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for q, s in r["q"].items():
            t = s["deblocked"]
            L.append(f"| {q} | {s['bytes'] / 2 ** 20:.2f} MiB | {s['ratio']:.1f}x | {s['psnr']:.1f} | {s.get('ssim', float('nan')):.4f} | {s['mae']:.2f} | {s['p99']:.0f} | {s['max']:.0f}"
                     f" | {t['psnr']:.1f} | {t.get('ssim', float('nan')):.4f} | {t['mae']:.2f} | {t['p99']:.0f} | {t['max']:.0f} |")
        L += [""] + [f"- {v}: [{v}_all.png]({name}/{v}_all.png)  ·  deblocked [{v}_all_db.png]({name}/{v}_all_db.png)  ·  raw [{name}/{v}_raw.png]({name}/{v}_raw.png)" for v in VIEWS] + [""]
        L += [f"![{name} zmid]({name}/zmid_all.png)", "", f"![{name} zmid deblocked]({name}/zmid_all_db.png)", ""]
    L += ["## Error overlays", "",
          "For the three centre planes of every cube: the raw slice in grey with a red overlay whose opacity is "
          "the absolute error of the plain decode at that pixel with a fixed 4x gain (min(1, 4 |raw - decoded| / 255): "
          "an error of 32 shows as 50 % red, 64 or more as solid red), magnified 2x by pixel replication to 1024^2. "
          "`tools/diff_images.py`.", ""]
    for name in names:
        L.append(f"- {name}: " + "  ·  ".join(
            f"{p} " + " ".join(f"[q{q}]({name}/diff_{p}_q{q}.png)" for q in QS) for p in ("zmid", "ymid", "xmid")))
    L += ["", "![PHercParis4_2.40um zmid q8 error](PHercParis4_2.40um/diff_zmid_q8.png)", "",
          "## Surface predictions and masks", "",
          "Our volcomp m7 surface-probability stores against the published upstream m7 masks of the same scans "
          "(eight scrolls): CT slices, overlays, disagreement panels, dice/precision/recall and store sizes. "
          "See [surfaces/README.md](surfaces/README.md); `tools/surface_compare.py`. The published masks "
          "re-exported as volcomp mask stores (mode 5 exact, mode 4 2x pooled): [masks/README.md](masks/README.md).", ""]
    with open(os.path.join(out_root, "README.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="docs/comparison")
    ap.add_argument("--set", choices=sorted(SETS), default="first")
    ap.add_argument("--stats-only", action="store_true", help="numbers only, no images rewritten")
    ap.add_argument("--no-per-q", action="store_true", help="skip the single q views (raw + strips only)")
    a = ap.parse_args()
    main(a.out, SETS[a.set], a.stats_only, not a.no_per_q)
