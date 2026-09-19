# volcomp_zarr — Python binding and zarr v3 codec

- `volcomp_zarr._lib`: ctypes binding over `libvolcomp.so` (built from
  `volcomp_shim.c`; standard library only): `encode(bytes, q) -> bytes`,
  `decode(bytes) -> bytearray`, `decode_block(bytes, bz, by, bx)`,
  `stream_q(bytes) -> float`, `is_lossless(bytes) -> bool`,
  `deblock(buf, nz, ny, nx, q)`.
- `volcomp_zarr.VolcompCodec`: zarr ≥ 3 array→bytes codec registered as `"volcomp"`
  (`{"name": "volcomp", "configuration": {"q": 8}}`), uint8, 128³ inner chunks;
  all-zero chunks are stored as missing when the array's fill value is 0.
  Needs `zarr>=3` and `numpy`.

**`q = 0` (`volcomp_zarr.Q_LOSSLESS`) is the lossless mode**: `VolcompCodec(q=0)`
reconstructs every chunk byte for byte and never stores more than the raw chunk
plus 8 bytes. Use it for class maps, masks and valid/occupancy flags — anything
where a value *is* the datum — and `q` in 1..255 for greyscale CT and
probability maps. The same `decode`/`decode_block` calls serve both; ask
`stream_q()` which one a chunk used. `q` outside `{0} ∪ [1, 255]` is rejected.

```python
import numpy as np, zarr, volcomp_zarr
from volcomp_zarr import VolcompCodec
z = zarr.create_array(store, shape=(256, 512, 512), chunks=(128, 128, 128),
                      dtype="uint8", serializer=VolcompCodec(q=0), fill_value=0)
z[:] = labels
assert np.array_equal(z[:], labels)   # exact
```

```sh
cmake --preset release && cmake --build --preset release --target volcomp_shim   # build/release/libvolcomp.so
VOLCOMP_LIB=build/release/libvolcomp.so python3 python/test_ctypes.py            # stdlib self-test
cp build/release/libvolcomp.so python/volcomp_zarr/ && pip install ./python[zarr]  # codec for zarr users
```

```python
import zarr, volcomp_zarr
a = zarr.open("volcomp/PHerc0343P/volumes/<volume>.zarr/0", mode="r")
block = a[1024:1152, 2048:2176, 512:640]
```

`python/test_zarr_roundtrip.py` exercises the codec class itself (zarr ≥ 3 and
numpy required): lossless round trips through both the bytes API and a zarr v3
array on an all-zero chunk, a 3-class mask, a smooth SDF-like chunk and random
noise.

```sh
uv venv /tmp/vcvenv && uv pip install --python /tmp/vcvenv/bin/python './python[zarr]'
cp build/release/libvolcomp.so python/volcomp_zarr/
/tmp/vcvenv/bin/python python/test_zarr_roundtrip.py
```
