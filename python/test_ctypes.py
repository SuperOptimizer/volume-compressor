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



def test_mask():
    """Mask chunks: the stored grid is the 2x2x2 majority pool, exactly, and the
    decode is its trilinear interpolation (0/255 at every block centre)."""
    src = bytearray(vc.CHUNK_VOXELS)
    i = 0
    for z in range(128):
        for y in range(128):
            for x in range(128):
                s = math.sin(z * 0.11) * math.cos(y * 0.07) + math.sin((x + y) * 0.05)
                src[i] = 255 if s > 0 else 0
                i += 1
    enc = vc.mask_encode(src)
    assert vc.mask_info(enc) == vc.MASK_DIM, vc.mask_info(enc)
    assert not vc.is_lossless(enc)
    assert vc.stream_q(enc) == 0.0
    grid = vc.mask_decode_stored(enc)
    assert len(grid) == vc.MASK_VOXELS
    bad = 0
    for z in range(64):
        for y in range(64):
            for x in range(64):
                c = sum(src[((2 * z + dz) * 128 + 2 * y + dy) * 128 + 2 * x + dx] != 0
                        for dz in range(2) for dy in range(2) for dx in range(2))
                bad += grid[(z * 64 + y) * 64 + x] != (255 if c * 2 >= 8 else 0)
    assert bad == 0, bad
    dec = vc.decode(enc)
    centres = sum(dec[((2 * z) * 128 + 2 * y) * 128 + 2 * x] != grid[(z * 64 + y) * 64 + x]
                  for z in range(0, 64, 7) for y in range(0, 64, 5) for x in range(0, 64, 3))
    assert centres == 0
    assert set(dec) <= {0, 32, 64, 96, 128, 159, 191, 223, 255}
    # a plain stream is not a mask chunk
    assert vc.mask_info(vc.encode(src, 0.0)) is None
    print(f"mask: {len(enc)} bytes ({8 * len(enc) / vc.CHUNK_VOXELS:.5f} bits/voxel), "
          f"values {sorted(set(dec))}")


def main():
    print("libvolcomp", vc.VERSION, "bound", vc.ENCODE_BOUND)
    test_mask()
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
    # lossless: q = 0 reconstructs exactly, and the stream says so
    for name, buf in (("synthetic", src), ("all-zero", bytearray(vc.CHUNK_VOXELS)),
                      ("8-class", bytearray(b >> 5 for b in src))):
        enc = vc.encode(buf, vc.Q_LOSSLESS)
        assert vc.decode(enc) == buf, f"lossless round trip failed on {name}"
        assert vc.is_lossless(enc) and vc.stream_q(enc) == vc.Q_LOSSLESS
        assert len(enc) <= vc.CHUNK_VOXELS + 8
        assert vc.decode_block(enc, 1, 2, 3) == vc.decode_block(enc, 1, 2, 3)
        print(f"q=   0 [{name:>9}]: {len(enc):7d} bytes "
              f"({vc.CHUNK_VOXELS / len(enc):8.1f}x)  exact")
    assert vc.stream_q(vc.encode(src, 8.0)) == 8.0

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
