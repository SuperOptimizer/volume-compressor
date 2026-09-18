#!/usr/bin/env python3
"""Magnified error overlays for the comparison set: for every volume and its three centre planes (zmid, ymid,
xmid), the raw slice in grey with a red overlay whose opacity is the absolute decode error of that pixel at each
q, unstretched (|raw - decoded| / 255, so an error of 128 is a 50 % red overlay), upscaled 2x by pixel
replication to 1024^2 and written as lossless RGB PNG: `diff_<view>_q<N>.png` (plain decode).

    VOLCOMP_LIB=build/release/libvolcomp.so python3 tools/diff_images.py docs/comparison
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_images import N, QS, VOLUMES, fetch_cube, pick_cube, roundtrip, views  # noqa: E402

PLANES = ["zmid", "ymid", "xmid"]


def overlay(raw, dec):
    """Grey base + red at opacity |raw - dec| / 255, 2x nearest upscale, uint8 RGB."""
    a = np.abs(raw.astype(np.float32) - dec.astype(np.float32)) / 255.0
    g = raw.astype(np.float32)
    rgb = np.stack([g * (1 - a) + 255.0 * a, g * (1 - a), g * (1 - a)], -1)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return rgb.repeat(2, axis=0).repeat(2, axis=1)


def main(out_root):
    import s3fs
    from PIL import Image
    fs = s3fs.S3FileSystem(anon=True)
    for sample, volume, res in VOLUMES:
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
    # index in the README
    rp = os.path.join(out_root, "README.md")
    s = open(rp).read()
    if "## Error overlays" not in s:
        L = ["", "## Error overlays", "",
             "For the three centre planes of every cube: the raw slice in grey with a red overlay whose opacity is "
             "the absolute error of the plain decode at that pixel, unstretched (|raw - decoded| / 255), magnified "
             "2x by pixel replication to 1024^2. `tools/diff_images.py`.", ""]
        for name in [f"{s_}_{r}" for s_, _, r in VOLUMES]:
            L.append(f"- {name}: " + "  ·  ".join(
                f"{p} " + " ".join(f"[q{q}]({name}/diff_{p}_q{q}.png)" for q in QS) for p in PLANES))
        L += ["", f"![{VOLUMES[2][0]}_{VOLUMES[2][2]} zmid q8 error]({VOLUMES[2][0]}_{VOLUMES[2][2]}/diff_zmid_q8.png)", ""]
        open(rp, "w").write(s.rstrip("\n") + "\n" + "\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/comparison")
