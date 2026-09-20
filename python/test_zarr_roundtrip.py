#!/usr/bin/env python3
"""Lossless (q=0) round trips through the zarr v3 codec and the bytes-level binding.

Needs zarr>=3 and numpy:
    uv venv /tmp/vcvenv && uv pip install --python /tmp/vcvenv/bin/python ./python[zarr]
    VOLCOMP_LIB=build/release/libvolcomp.so /tmp/vcvenv/bin/python python/test_zarr_roundtrip.py
"""
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volcomp_zarr as vz  # noqa: E402
from volcomp_zarr import VolcompCodec  # noqa: E402

D = vz.CHUNK_DIM


def all_zero():
    return np.zeros((D, D, D), np.uint8)


def three_class():
    a = np.zeros((D, D, D), np.uint8)
    z, y, x = np.ogrid[:D, :D, :D]
    a[(z - 40) ** 2 + (y - 60) ** 2 + (x - 70) ** 2 < 30 ** 2] = 1
    a[(z - 90) ** 2 + (y - 50) ** 2 + (x - 30) ** 2 < 22 ** 2] = 2
    return a


def smooth_sdf():
    z, y, x = np.ogrid[:D, :D, :D]
    s = 26.0 * np.sin(x * 0.04 + z * 0.021) + 0.35 * y - 20.0
    d = 128.0 + 9.0 * s / (1.0 + np.abs(s) * 0.06)
    return np.clip(d, 0, 255).astype(np.uint8)


def noise():
    return np.random.default_rng(7).integers(0, 256, (D, D, D), dtype=np.uint8)


CASES = [("all-zero", all_zero), ("3-class mask", three_class), ("smooth SDF", smooth_sdf),
         ("random noise", noise)]


def main():
    print("libvolcomp", vz.VERSION)
    raw = D ** 3
    fails = 0

    print("\n-- bytes-level binding, q = Q_LOSSLESS --")
    for name, gen in CASES:
        a = gen()
        enc = vz.encode(a.tobytes(), vz.Q_LOSSLESS)
        got = np.frombuffer(bytes(vz.decode(enc)), np.uint8).reshape(a.shape)
        exact = np.array_equal(a, got)
        assert vz.is_lossless(enc) and vz.stream_q(enc) == 0.0
        # never larger than raw plus a small header
        assert len(enc) <= raw + 8, (name, len(enc))
        # decode_block agrees with the full decode
        blk = np.frombuffer(bytes(vz.decode_block(enc, 2, 3, 4)), np.uint8).reshape(16, 16, 16)
        assert np.array_equal(blk, got[32:48, 48:64, 64:80]), name
        print(f"{name:>14}: {len(enc):9d} bytes ({raw / len(enc):9.1f}x)  exact={exact}")
        fails += not exact
    # a lossy stream is reported as lossy
    assert vz.stream_q(vz.encode(smooth_sdf().tobytes(), 8.0)) == 8.0
    assert not vz.is_lossless(vz.encode(smooth_sdf().tobytes(), 8.0))

    print("\n-- zarr v3 array, codec volcomp(q=0) --")
    for name, gen in CASES:
        a = gen()
        store = zarr.storage.MemoryStore()
        z = zarr.create_array(store, shape=a.shape, chunks=(D, D, D), dtype="uint8",
                              serializer=VolcompCodec(q=0.0), fill_value=0)
        z[:] = a
        got = z[:]
        exact = np.array_equal(a, got)
        print(f"{name:>14}: exact={exact}")
        fails += not exact
        # reopening from the stored metadata must pick the codec back up
        assert np.array_equal(zarr.open_array(store, mode="r")[:], a), name

    print("\n-- zarr v3 array, codec volcomp(mode=mask) --")
    mask = np.where(three_class() != 0, 255, 0).astype(np.uint8)
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(store, shape=mask.shape, chunks=(D, D, D), dtype="uint8",
                          serializer=VolcompCodec(mode="mask"), fill_value=0)
    z[:] = mask
    got = z[:]
    # the stored grid is the 2x2x2 majority pool, and the decode interpolates it
    c = sum((mask != 0)[dz::2, dy::2, dx::2].astype(np.uint16)
            for dz in range(2) for dy in range(2) for dx in range(2))
    grid = np.where(c * 2 >= 8, 255, 0).astype(np.uint8)
    assert np.array_equal(got[0::2, 0::2, 0::2], grid), "block centres are not the majority pool"
    assert set(np.unique(got)) <= {0, 32, 64, 96, 128, 159, 191, 223, 255}
    enc = vz.mask_encode(mask.tobytes())
    assert vz.mask_info(enc) == vz.MASK_DIM and not vz.is_lossless(enc)
    assert np.array_equal(np.frombuffer(bytes(vz.mask_decode_stored(enc)), np.uint8).reshape(64, 64, 64),
                          grid)
    assert vz.mask_info(vz.encode(mask.tobytes(), 0.0)) is None
    assert np.array_equal(zarr.open_array(store, mode="r")[:], got)
    print(f"{'mask':>14}: {len(enc):9d} bytes ({raw / len(enc):9.1f}x)  "
          f"{8 * len(enc) / raw:.5f} bits/voxel")

    print("\n-- zarr v3 array, codec volcomp(mode=mask-lossless) --")
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(store, shape=mask.shape, chunks=(D, D, D), dtype="uint8",
                          serializer=VolcompCodec(mode="mask-lossless"), fill_value=0)
    z[:] = mask
    got = z[:]
    assert np.array_equal(got, mask), "mask-lossless is not exact"
    enc = vz.mask_encode_lossless(mask.tobytes())
    assert vz.mask_info(enc) == vz.MASK_LL_DIM and not vz.is_lossless(enc)
    assert len(enc) <= vz.MASK_LL_BOUND
    assert np.array_equal(np.frombuffer(bytes(vz.mask_decode_stored(enc)), np.uint8).reshape(D, D, D),
                          mask)
    assert np.array_equal(zarr.open_array(store, mode="r")[:], mask)
    print(f"{'mask-lossless':>14}: {len(enc):9d} bytes ({raw / len(enc):9.1f}x)  "
          f"{8 * len(enc) / raw:.5f} bits/voxel")

    # the codec's own metadata round trip and validation
    assert VolcompCodec.from_dict(VolcompCodec(q=0.0).to_dict()) == VolcompCodec(q=0.0)
    assert VolcompCodec.from_dict(VolcompCodec(mode="mask").to_dict()) == VolcompCodec(mode="mask")
    assert (VolcompCodec.from_dict(VolcompCodec(mode="mask-lossless").to_dict())
            == VolcompCodec(mode="mask-lossless"))
    try:
        VolcompCodec(mode="nope")
        raise SystemExit("expected an unknown mode to be rejected")
    except ValueError:
        pass
    for bad in (0.5, -1.0, 256.0):
        try:
            VolcompCodec(q=bad)
            raise SystemExit(f"expected q={bad} to be rejected")
        except ValueError:
            pass
    try:
        VolcompCodec(mode="nope")
        raise SystemExit("expected mode=nope to be rejected")
    except ValueError:
        pass

    print("\nFAIL" if fails else "\nok")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
