#!/usr/bin/env python3
"""Magnified error overlays for the comparison set: for every volume and its three centre planes (zmid, ymid,
xmid), the raw slice in grey with a red overlay whose opacity is the absolute decode error of that pixel at each
q with a fixed 4x gain (min(1, 4 * |raw - decoded| / 255): an error of 32 is a 50 % red overlay, 64 or more is
solid red), upscaled 2x by pixel replication to 1024^2 and written as lossless RGB PNG: `diff_<view>_q<N>.png`
(plain decode).

    VOLCOMP_LIB=build/release/libvolcomp.so python3 tools/diff_images.py docs/comparison [--set first|more|all]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_images import N, QS, SETS, VOLUMES, fetch_cube, pick_cube, roundtrip, views, write_readme  # noqa: E402

PLANES = ["zmid", "ymid", "xmid"]
GAIN = 4.0  # overlay opacity = min(1, GAIN * error / 255)


def overlay(raw, dec):
    """Grey base + red at opacity min(1, GAIN * |raw - dec| / 255), 2x nearest upscale, uint8 RGB."""
    a = np.minimum(GAIN * np.abs(raw.astype(np.float32) - dec.astype(np.float32)) / 255.0, 1.0)
    g = raw.astype(np.float32)
    rgb = np.stack([g * (1 - a) + 255.0 * a, g * (1 - a), g * (1 - a)], -1)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return rgb.repeat(2, axis=0).repeat(2, axis=1)


def main(out_root, vols=VOLUMES):
    import s3fs
    from PIL import Image
    fs = s3fs.S3FileSystem(anon=True)
    for sample, volume, res in vols:
        name = f"{sample}_{res}"
        d = os.path.join(out_root, name)
        org, _ = pick_cube(fs, sample, volume)
        cube = fetch_cube(fs, sample, volume, org)
        rv = views(cube)
        for q in QS:
            dec, _ = roundtrip(cube, q)
            dv = views(dec)
            for p in PLANES:
                img = overlay(rv[p], dv[p])
                Image.fromarray(img, "RGB").save(os.path.join(d, f"diff_{p}_q{q}.png"), optimize=True)
            print(f"{name} q={q}: max error on the centre planes "
                  f"{max(int(np.abs(rv[p].astype(int) - dv[p].astype(int)).max()) for p in PLANES)}", flush=True)
    # the README (tables and the error-overlay index) is written by compare_images.write_readme from report.json
    import json
    rp = os.path.join(out_root, "report.json")
    if os.path.exists(rp):
        write_readme(out_root, json.load(open(rp)))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="docs/comparison")
    ap.add_argument("--set", choices=sorted(SETS), default="first")
    a = ap.parse_args()
    main(a.out, SETS[a.set])
