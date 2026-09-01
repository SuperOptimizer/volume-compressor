#!/usr/bin/env python3
"""Standard-library check of the ctypes binding: encode/decode/decode_block/deblock
round trips on a synthetic 128^3 chunk. Run: VOLCOMP_LIB=… python3 test_ctypes.py"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from volcomp_zarr import _lib as vc  # noqa: E402


def synth():
    rnd = random.Random(7)
    v = bytearray(vc.CHUNK_VOXELS)
    i = 0
    for z in range(128):
        for y in range(128):
            for x in range(128):
                s = 120 + 60 * math.sin(x / 9.0) * math.cos(y / 11.0) + 30 * math.sin((z + x) / 7.0)
                v[i] = max(0, min(255, int(s + rnd.randint(-3, 3))))
                i += 1
    return v


def psnr(a, b):
    se = sum((p - q) ** 2 for p, q in zip(a, b))
    return 999.0 if se == 0 else 10 * math.log10(255 ** 2 * len(a) / se)


def main():
    print("libvolcomp", vc.VERSION, "bound", vc.ENCODE_BOUND)
    src = synth()
    for q in (2.0, 8.0, 32.0):
        enc = vc.encode(src, q)
        dec = vc.decode(enc)
        p = psnr(src, dec)
        # block (2,3,4) of the full decode must equal decode_block
        blk = vc.decode_block(enc, 2, 3, 4)
        ref = bytearray()
        for z in range(16):
            for y in range(16):
                o = ((2 * 16 + z) * 128 + (3 * 16 + y)) * 128 + 4 * 16
                ref += dec[o:o + 16]
        assert blk == ref, "decode_block != full decode"
        print(f"q={q:4.0f}: {len(enc):7d} bytes ({vc.CHUNK_VOXELS / len(enc):6.1f}x)  psnr {p:6.2f} dB")
        assert p > 25, "implausible quality"
    # error paths
    try:
        vc.decode(b"VOLC\x02\x00\x00\x08")
        raise SystemExit("expected an error")
    except vc.VolcompError as e:
        print("truncated stream ->", e)
    try:
        vc.encode(b"\x00" * 10, 8.0)
        raise SystemExit("expected a size error")
    except vc.VolcompError as e:
        print("short input ->", e)
    vol = bytearray(vc.decode(vc.encode(src, 8.0)))
    vc.deblock(vol, 128, 128, 128, 8.0)
    print("deblock ok, psnr", round(psnr(src, vol), 2))
    print("PASS")


if __name__ == "__main__":
    main()
