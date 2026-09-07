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

    # the codec's own metadata round trip and validation
    assert VolcompCodec.from_dict(VolcompCodec(q=0.0).to_dict()) == VolcompCodec(q=0.0)
    for bad in (0.5, -1.0, 256.0):
        try:
            VolcompCodec(q=bad)
            raise SystemExit(f"expected q={bad} to be rejected")
        except ValueError:
            pass

    print("\nFAIL" if fails else "\nok")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
