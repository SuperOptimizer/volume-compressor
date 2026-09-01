"""zarr v3 codec "volcomp" (array -> bytes) for 128^3 uint8 chunks.

    import volcomp_zarr            # registers the codec with zarr >= 3
    z = zarr.open("…/volume.zarr/0")   # sharding_indexed(1024^3) -> volcomp(q) inner 128^3 chunks

The low-level binding (encode/decode/decode_block/deblock over bytes) lives in
volcomp_zarr._lib and needs only the standard library; the codec class below
needs zarr >= 3 and numpy and is skipped if they are not installed.
"""
from ._lib import (BLOCK_VOXELS, CHUNK_DIM, CHUNK_VOXELS, ENCODE_BOUND, VERSION, VolcompError, deblock,
                   decode, decode_block, encode)

__all__ = ["encode", "decode", "decode_block", "deblock", "VolcompError", "VERSION", "ENCODE_BOUND",
           "CHUNK_DIM", "CHUNK_VOXELS", "BLOCK_VOXELS", "VolcompCodec"]

try:  # zarr v3 codec registration (optional dependency)
    from dataclasses import dataclass, replace

    import numpy as np
    from zarr.abc.codec import ArrayBytesCodec
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, NDBuffer
    from zarr.core.common import JSON, parse_named_configuration
    from zarr.registry import register_codec

    @dataclass(frozen=True)
    class VolcompCodec(ArrayBytesCodec):
        """{"name": "volcomp", "configuration": {"q": 8}} — lossy, uint8, 128^3 chunks only."""

        q: float = 8.0
        is_fixed_size = False

        @classmethod
        def from_dict(cls, data: dict[str, JSON]) -> "VolcompCodec":
            _, cfg = parse_named_configuration(data, "volcomp", require_configuration=False)
            cfg = cfg or {}
            return cls(q=float(cfg.get("q", 8.0)))

        def to_dict(self) -> dict[str, JSON]:
            return {"name": "volcomp", "configuration": {"q": self.q}}

        def validate(self, *, shape, dtype, chunk_grid) -> None:  # noqa: D401
            if str(np.dtype(dtype.to_native_dtype() if hasattr(dtype, "to_native_dtype") else dtype)) != "uint8":
                raise ValueError("volcomp codec requires uint8 data")
            cs = tuple(chunk_grid.chunk_shape)
            if cs != (CHUNK_DIM,) * 3:
                raise ValueError(f"volcomp codec requires 128^3 (inner) chunks, got {cs}")

        def evolve_from_array_spec(self, array_spec: ArraySpec) -> "VolcompCodec":
            return self

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            return ENCODE_BOUND

        async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
            arr = np.ascontiguousarray(chunk_array.as_numpy_array(), dtype=np.uint8)
            if arr.shape != (CHUNK_DIM,) * 3:
                raise ValueError(f"volcomp: chunk must be 128^3, got {arr.shape}")
            if not arr.any():
                return None  # all-zero chunk -> stored as missing (fill value 0)
            return chunk_spec.prototype.buffer.from_bytes(encode(arr.tobytes(), self.q))

        async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
            out = np.empty((CHUNK_DIM,) * 3, dtype=np.uint8)
            decode(chunk_bytes.as_numpy_array().tobytes(), out.reshape(-1).view(np.uint8))
            return chunk_spec.prototype.nd_buffer.from_numpy_array(out)

    register_codec("volcomp", VolcompCodec)
except ImportError:  # zarr / numpy not installed: the bytes-level API still works
    VolcompCodec = None
