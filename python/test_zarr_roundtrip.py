#!/usr/bin/env python3
"""Round trips through the zarr v3 codec and the bytes-level binding: uint8 (lossless, mask,
surface, reader-side deblocking) and the typed modes (uint16 / int16 / uint32 / int32 / float32,
exact and error-bounded).

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

    print("\n-- sharded zarr v3 array (1024^3 shards of 128^3 chunks), codec chain [volcomp(mode=surface)] --")
    # a smooth probability field with sheets: the kind of volume the surface mode is for
    zz, yy, xx = np.ogrid[:256, :256, :384]
    sd = 9.0 * np.sin(0.045 * xx + 0.02 * zz) + 0.33 * yy - 20.0
    sd = np.mod(sd + 400.0, 22.0) - 11.0
    prob = np.clip(np.rint(255.0 / (1.0 + np.exp(-(3.5 - np.abs(sd)) / 1.2))), 0, 255).astype(np.uint8)
    prob[:, :, 300:] = 0  # an empty strip: its chunks are stored as missing
    for q, thr in ((16.0, 128), (32.0, 100)):
        store = zarr.storage.MemoryStore()
        z = zarr.create_array(store, shape=prob.shape, chunks=(D, D, D), shards=(1024, 1024, 1024),
                              dtype="uint8", serializer=VolcompCodec(mode="surface", q=q, threshold=thr),
                              compressors=None, fill_value=0)
        z[:] = prob
        meta = zarr.open_array(store, mode="r").metadata.to_dict()
        inner = meta["codecs"][0]["configuration"]["codecs"]
        assert meta["codecs"][0]["name"] == "sharding_indexed" and len(inner) == 1, meta["codecs"]
        assert inner[0]["name"] == "volcomp" and inner[0]["configuration"]["mode"] == "surface", inner
        got = zarr.open_array(store, mode="r")[:]
        wrong = int(((got >= thr) != (prob >= thr)).sum())
        band = (prob > 32) & (prob < 224)
        mae = float(np.abs(got.astype(int) - prob)[band].mean())
        nbytes = sum(len(v) for k, v in store._store_dict.items() if not k.endswith("zarr.json"))
        enc = vz.surface_encode(np.ascontiguousarray(prob[:D, :D, :D]).tobytes(), q, thr)
        assert vz.surface_info(enc)[0] == thr and vz.stream_q(enc) == q and not vz.is_lossless(enc)
        assert vz.surface_info(vz.encode(prob[:D, :D, :D].tobytes(), q)) is None
        print(f"{'surface q%g t%d' % (q, thr):>14}: {nbytes:9d} bytes for {prob.size} voxels "
              f"({8 * nbytes / prob.size:.4f} bits/voxel), wrong-side voxels {wrong}, band MAE {mae:.2f}")
        fails += wrong != 0
        assert mae < 0.3 * q + 2
    assert (VolcompCodec.from_dict(VolcompCodec(mode="surface", q=16).to_dict())
            == VolcompCodec(mode="surface", q=16))
    assert (VolcompCodec.from_dict(VolcompCodec(mode="surface", q=16, threshold=90).to_dict())
            == VolcompCodec(mode="surface", q=16, threshold=90))
    for kw in ({"q": 0.0}, {"q": 16, "threshold": 0}, {"q": 16, "threshold": 256}):
        try:
            VolcompCodec(mode="surface", **kw)
            raise SystemExit(f"expected surface {kw} to be rejected")
        except ValueError:
            pass

    print("\n-- reader-side deblocking (set_read_smoothing): same stored bytes, smoother reconstruction --")
    ct = smooth_sdf().astype(np.int16) + np.random.default_rng(3).integers(-3, 4, (D, D, D))
    ct = np.clip(ct, 1, 255).astype(np.uint8)
    ct[:, :, 80:] = 0  # masked air on a block face
    store = zarr.storage.MemoryStore()
    z = zarr.create_array(store, shape=ct.shape, chunks=(D, D, D), dtype="uint8",
                          serializer=VolcompCodec(q=16.0), compressors=None, fill_value=0)
    z[:] = ct
    plain = zarr.open_array(store, mode="r")[:]
    prev = vz.set_read_smoothing(2.0)
    try:
        smooth = zarr.open_array(store, mode="r")[:]
    finally:
        vz.set_read_smoothing(prev)
    e0 = float(np.mean((plain.astype(float) - ct) ** 2)); e1 = float(np.mean((smooth.astype(float) - ct) ** 2))
    print(f"{'q16 smooth=2':>14}: MSE {e0:.3f} -> {e1:.3f}, masked air kept at 0: {bool((smooth[ct == 0] == 0).all())}")
    assert (smooth[plain == 0] == 0).all() and e1 <= e0 * 1.02
    assert "smooth" not in str(zarr.open_array(store, mode="r").metadata.to_dict())

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
    for bad in (-1.0, float("nan"), float("inf")):
        try:
            VolcompCodec(q=bad)
            raise SystemExit(f"expected q={bad} to be rejected")
        except ValueError:
            pass
    # q's range depends on the dtype (typed q is in value units): uint8 arrays refuse q
    # outside {0} u [1, 255] when the array is created
    for bad in (0.5, 256.0):
        try:
            zarr.create_array(zarr.storage.MemoryStore(), shape=(D, D, D), chunks=(D, D, D), dtype="uint8",
                              serializer=VolcompCodec(q=bad), fill_value=0)
            raise SystemExit(f"expected q={bad} to be rejected for uint8")
        except ValueError:
            pass
    try:
        VolcompCodec(mode="nope")
        raise SystemExit("expected mode=nope to be rejected")
    except ValueError:
        pass

    fails += typed()
    print("\nFAIL" if fails else "\nok")
    return 1 if fails else 0


def winding(dtype, shape=(D, D, D), tilt=0.0):
    """a winding-like field W (radial ramp around an axis outside the chunk + waves), stored
    as W * 256 in uint16 (the ufsm layout) and as scaled copies in the other dtypes"""
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    cy, cx = -60.0 + tilt * z, 40.0
    r, th = np.hypot(y - cy, x - cx), np.arctan2(y - cy, x - cx)
    w = 4.0 + r / 30.0 + 0.15 * np.sin(0.09 * z + 3.0 * th) + 0.05 * np.sin(0.21 * x + 0.13 * y)
    scale = {"uint16": (256.0, 0.0), "int16": (256.0, -20000.0), "uint32": (1 << 20, 0.0),
             "int32": (-(1 << 20), 0.0), "float32": (1.0, 0.0)}[dtype]
    v = w * scale[0] + scale[1]
    return (v if dtype == "float32" else np.rint(v)).astype(dtype)


def typed():
    """The typed modes (spec/format.md §14) through the bytes-level binding and zarr v3 arrays."""
    fails = 0
    names = ("uint16", "int16", "uint32", "int32", "float32")
    rng = np.random.default_rng(11)

    print("\n-- typed chunks, bytes-level binding --")
    for name in names:
        a = winding(name, tilt=0.3)
        raw = a.nbytes
        enc = vz.encode_typed(a, name, vz.Q_LOSSLESS)
        got = np.frombuffer(bytes(vz.decode_typed(enc, name)), name).reshape(a.shape)
        exact = np.array_equal(a.view(np.uint8), got.view(np.uint8))
        assert vz.chunk_dtype(enc) == name and vz.is_lossless(enc) and vz.stream_q(enc) == 0.0
        assert len(enc) <= vz.typed_encode_bound(name)
        blk = np.frombuffer(bytes(vz.decode_block_typed(enc, 2, 3, 4, name)), name).reshape(16, 16, 16)
        assert np.array_equal(blk, got[32:48, 48:64, 64:80]), name
        other = "int16" if name != "int16" else "uint16"
        try:
            vz.decode_typed(enc, other)
            raise SystemExit("expected a dtype mismatch to be refused")
        except vz.VolcompError:
            pass
        try:
            vz.decode(enc)
            raise SystemExit("expected the u8 decode to refuse a typed chunk")
        except vz.VolcompError:
            pass
        line = f"{name:>8} exact: {len(enc):8d} bytes ({raw / len(enc):5.2f}x) exact={exact}"
        fails += not exact
        q = 16.0 if name in ("uint16", "int16") else (4096.0 if name.endswith("32") and name != "float32" else 0.01)
        enc = vz.encode_typed(a, name, q)
        got = np.frombuffer(bytes(vz.decode_typed(enc, name)), name).reshape(a.shape)
        err = float(np.abs(got.astype(np.float64) - a.astype(np.float64)).max())
        assert not vz.is_lossless(enc) and 0.0 < vz.stream_q(enc) <= q
        line += f" | q {q:g}: {len(enc):8d} bytes ({raw / len(enc):5.2f}x) max err {err:g} (<= {q / 2:g})"
        print(line)
        fails += not err <= q / 2
        # incompressible noise is the RAW form: raw + 10 bytes, exact
        nz = np.frombuffer(rng.bytes(raw), name).reshape(a.shape)
        enc = vz.encode_typed(nz, name, 0.0)
        assert len(enc) == raw + 10 and vz.decode_typed(enc, name) == nz.tobytes()
    # float32 specials (NaN payloads, -0, inf, subnormals) are exact in mode 7, and a chunk
    # holding any NaN or inf is kept exact at q > 0 too
    f = winding("float32")
    bits = f.view(np.uint32).copy()
    bits[0, 0, :8] = [0x7FC01234, 0xFFC00000, 0x7F800001, 0x80000000, 0x7F800000, 0xFF800000, 1, 0x807FFFFF]
    f = bits.view(np.float32)
    for q in (0.0, 0.5):
        enc = vz.encode_typed(f, "float32", q)
        back = np.frombuffer(bytes(vz.decode_typed(enc, "float32")), np.uint32).reshape(f.shape)
        assert np.array_equal(back, bits) and vz.stream_q(enc) == 0.0, q
    print(f"{'float32':>8} specials: exact at q = 0 and kept exact at q > 0")

    print("\n-- typed zarr v3 arrays (sharded 1024^3 / 128^3 chunks, codec chain [volcomp]) --")
    shape = (256, 256, 384)
    for name in names:
        a = winding(name, shape=shape, tilt=0.2)
        a[:, :, 300:] = 0  # an empty strip: all-zero chunks are stored as missing
        for q in (0.0, 8.0 if name != "float32" else 0.001):
            store = zarr.storage.MemoryStore()
            z = zarr.create_array(store, shape=shape, chunks=(D, D, D), shards=(1024, 1024, 1024), dtype=name,
                                  serializer=VolcompCodec(q=q), compressors=None, fill_value=0)
            z[:] = a
            meta = zarr.open_array(store, mode="r").metadata.to_dict()
            inner = meta["codecs"][0]["configuration"]["codecs"]
            assert inner[0]["name"] == "volcomp" and inner[0]["configuration"]["q"] == q, inner
            got = zarr.open_array(store, mode="r")[:]
            assert got.dtype == np.dtype(name)
            err = float(np.abs(got.astype(np.float64) - a.astype(np.float64)).max())
            ok = np.array_equal(got, a) if q == 0 else err <= q / 2
            nbytes = sum(len(v) for k, v in store._store_dict.items() if not k.endswith("zarr.json"))
            print(f"{name:>8} q {q:<6g}: {nbytes:9d} bytes for {a.nbytes} ({a.nbytes / nbytes:6.2f}x), "
                  f"max err {err:g}, ok={ok}")
            fails += not ok
    # mask / surface modes are uint8 only
    for mode in ("mask", "mask-lossless", "surface"):
        try:
            zarr.create_array(zarr.storage.MemoryStore(), shape=(D, D, D), chunks=(D, D, D), dtype="uint16",
                              serializer=VolcompCodec(mode=mode, q=16), fill_value=0)
            raise SystemExit(f"expected mode={mode} to be refused for uint16")
        except ValueError:
            pass
    # typed q is in value units: any positive q, and it round-trips through the metadata
    assert VolcompCodec.from_dict(VolcompCodec(q=0.001).to_dict()) == VolcompCodec(q=0.001)
    zarr.create_array(zarr.storage.MemoryStore(), shape=(D, D, D), chunks=(D, D, D), dtype="float32",
                      serializer=VolcompCodec(q=1e-4), fill_value=0)
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
