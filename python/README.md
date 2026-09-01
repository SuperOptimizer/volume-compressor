# volcomp_zarr — Python binding and zarr v3 codec

- `volcomp_zarr._lib`: ctypes binding over `libvolcomp.so` (built from
  `volcomp_shim.c`; standard library only): `encode(bytes, q) -> bytes`,
  `decode(bytes) -> bytearray`, `decode_block(bytes, bz, by, bx)`, `deblock(buf, nz, ny, nx, q)`.
- `volcomp_zarr.VolcompCodec`: zarr ≥ 3 array→bytes codec registered as `"volcomp"`
  (`{"name": "volcomp", "configuration": {"q": 8}}`), uint8, 128³ inner chunks;
  all-zero chunks are stored as missing. Needs `zarr>=3` and `numpy`.

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

The codec class has not been exercised in the development environment (no zarr
install there); the bytes-level binding is covered by `test_ctypes.py`.
