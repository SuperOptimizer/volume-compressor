#!/usr/bin/env python3
"""Visual comparison of volcomp quality levels on raw scroll CT.

For each volume a 512^3 cube of the ORIGINAL uint8 data (the open-data bucket stores the level-0 chunks
uncompressed) is fetched, encoded chunk by chunk (128^3) at q = 1, 2, 4, 8, 16, 32 and decoded again (plain
decode, no deblocking: what a zarr reader gets). Nine 512^2 views of every version are written as lossless
greyscale PNGs -- the six faces of the cube and the three planes through its exact centre -- plus, per view, one
strip that concatenates raw and the six q levels side by side. A README with size ratio, PSNR, MAE and error
percentiles per cube and q is written next to the images.

The cube is chosen automatically: the densest 512^3 window (mean intensity) of the middle 512 slices, found on
the 16x downsampled level and snapped to the 128-voxel chunk grid, so every cube is papyrus rather than air.

    VOLCOMP_LIB=build/release/libvolcomp.so python3 tools/compare_images.py docs/comparison
Needs numpy, zarr>=3, s3fs, Pillow and python/volcomp_zarr (the ctypes binding).
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


def views(v):
    h = N // 2
    return {"z0": v[0], "z511": v[-1], "y0": v[:, 0], "y511": v[:, -1], "x0": v[:, :, 0], "x511": v[:, :, -1],
            "zmid": v[h], "ymid": v[:, h], "xmid": v[:, :, h]}


def stats(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = float((d * d).mean())
    ad = np.abs(d)
    return {"psnr": 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf"), "mae": float(ad.mean()),
            "p99": float(np.percentile(ad, 99)), "max": float(ad.max())}


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


def main(out_root):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    os.makedirs(out_root, exist_ok=True)
    report = {}
    for sample, volume, res in VOLUMES:
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
            s = vers[f"q{q}"][2]
            print(f"  q={q:<3d} {cube.nbytes / nb:7.1f}x  PSNR {s['psnr']:.1f} dB  MAE {s['mae']:.2f}  P99 {s['p99']:.0f}  max {s['max']:.0f}", flush=True)
        allviews = {k: views(v[0]) for k, v in vers.items()}
        for k in vers:
            for vn, img in allviews[k].items():
                save_png(os.path.join(d, f"{vn}_{k}.png"), img)
        for vn in VIEWS:
            labels = [f"{name} {vn}  raw ({cube.nbytes / 2 ** 20:.0f} MiB)"]
            labels += [f"q={q}  {cube.nbytes / vers[f'q{q}'][1]:.0f}x  PSNR {vers[f'q{q}'][2]['psnr']:.1f} dB  MAE {vers[f'q{q}'][2]['mae']:.2f}" for q in QS]
            strip([allviews[k][vn] for k in vers], labels, os.path.join(d, f"{vn}_all.png"))
        report[name] = {"sample": sample, "volume": volume, "voxel": res, "shape": list(shape), "origin_zyx": org,
                        "raw_bytes": int(cube.nbytes),
                        "q": {str(q): {"bytes": int(vers[f"q{q}"][1]), "ratio": cube.nbytes / vers[f"q{q}"][1], **vers[f"q{q}"][2]} for q in QS}}
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(out_root, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    write_readme(out_root, report)


def write_readme(out_root, report):
    L = ["# volcomp quality comparison on raw scroll CT", "",
         "512^3 cubes of ORIGINAL uint8 CT (the open-data bucket's uncompressed level-0 chunks) from four ESRF scans "
         "at four voxel sizes, encoded at q = 1 .. 32 (128^3 chunks) and decoded without deblocking. Every version "
         "has nine 512^2 lossless PNG views: the six cube faces (`z0 z511 y0 y511 x0 x511`) and the three centre "
         "planes (`zmid ymid xmid`); `<view>_all.png` puts raw and the six q levels side by side. Generated by "
         "`tools/compare_images.py`; numbers in `report.json`.", ""]
    for name, r in report.items():
        L += [f"## {name}", "", f"`{r['sample']}/volumes/{r['volume']}`, volume shape {r['shape']}, cube origin zyx {r['origin_zyx']}, raw {r['raw_bytes'] / 2 ** 20:.0f} MiB.", "",
              "| q | size | ratio | PSNR dB | MAE | P99 | max |", "|---|---|---|---|---|---|---|"]
        for q, s in r["q"].items():
            L.append(f"| {q} | {s['bytes'] / 2 ** 20:.2f} MiB | {s['ratio']:.1f}x | {s['psnr']:.1f} | {s['mae']:.2f} | {s['p99']:.0f} | {s['max']:.0f} |")
        L += [""] + [f"- {v}: [{v}_all.png]({name}/{v}_all.png)  ·  raw [{name}/{v}_raw.png]({name}/{v}_raw.png)" for v in VIEWS] + [""]
        L += [f"![{name} zmid]({name}/zmid_all.png)", ""]
    with open(os.path.join(out_root, "README.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/comparison")
