"""zarr v3 codec "volcomp" (array -> bytes) for 128^3 uint8 chunks.

    import volcomp_zarr            # registers the codec with zarr >= 3
    z = zarr.open("…/volume.zarr/0")   # sharding_indexed(1024^3) -> volcomp(q) inner 128^3 chunks

The low-level binding (encode/decode/decode_block/deblock over bytes) lives in
volcomp_zarr._lib and needs only the standard library; the codec class below
needs zarr >= 3 and numpy and is skipped if they are not installed.
"""
from ._lib import (BLOCK_VOXELS, CHUNK_DIM, CHUNK_VOXELS, ENCODE_BOUND, MASK_DIM, MASK_LL_BOUND,
                   MASK_LL_DIM, MASK_LL_VOXELS, MASK_VOXELS, Q_LOSSLESS, SURFACE_BOUND, SURFACE_THRESHOLD,
                   VERSION, VolcompError, deblock, decode, decode_block, encode, is_lossless,
                   mask_decode_stored, mask_encode, mask_encode_lossless, mask_info, stream_q,
                   surface_encode, surface_info)

__all__ = ["encode", "decode", "decode_block", "deblock", "stream_q", "is_lossless", "mask_encode",
           "mask_encode_lossless", "mask_info", "mask_decode_stored", "VolcompError", "VERSION",
           "ENCODE_BOUND", "Q_LOSSLESS", "CHUNK_DIM", "CHUNK_VOXELS", "BLOCK_VOXELS", "MASK_DIM",
           "MASK_VOXELS", "MASK_LL_DIM", "MASK_LL_VOXELS", "MASK_LL_BOUND", "VolcompCodec",
           "surface_encode", "surface_info", "SURFACE_BOUND", "SURFACE_THRESHOLD"]

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
        """{"name": "volcomp", "configuration": {"q": 8}} — uint8, 128^3 chunks only.

        q = 0 (Q_LOSSLESS) is the exact codec; q in 1..255 is the lossy DCT codec.
        {"mode": "mask"} instead selects the binary MASK mode: the chunk is taken
        as a mask (nonzero = 1), its 2x2x2 majority pool is stored exactly, and
        decoding returns that grid trilinearly interpolated back to 128^3 as a
        continuous 0..255 field. {"mode": "mask-lossless"} stores the whole mask
        exactly instead, with the same coder, and decodes to 0/255 voxel for
        voxel. `q` is ignored in both mask modes.
        {"mode": "surface", "q": 16} selects the SURFACE mode for probability
        maps that consumers threshold: the DCT codec at q plus a refinement layer
        that keeps `value >= threshold` (default 128, i.e. p >= 0.5) exact for
        every voxel; optional "threshold" in 1..255.
        """

        q: float = 8.0
        mode: str = "q"
        threshold: int = SURFACE_THRESHOLD
        is_fixed_size = False

        @classmethod
        def from_dict(cls, data: dict[str, JSON]) -> "VolcompCodec":
            _, cfg = parse_named_configuration(data, "volcomp", require_configuration=False)
            cfg = cfg or {}
            return cls(q=float(cfg.get("q", 8.0)), mode=str(cfg.get("mode", "q")),
                       threshold=int(cfg.get("threshold", SURFACE_THRESHOLD)))

        def to_dict(self) -> dict[str, JSON]:
            if self.mode in ("mask", "mask-lossless"):
                return {"name": "volcomp", "configuration": {"mode": self.mode, "q": 0}}
            if self.mode == "surface":
                cfg = {"mode": "surface", "q": self.q}
                if self.threshold != SURFACE_THRESHOLD:
                    cfg["threshold"] = self.threshold
                return {"name": "volcomp", "configuration": cfg}
            return {"name": "volcomp", "configuration": {"q": self.q}}

        def __post_init__(self) -> None:
            if self.mode not in ("q", "mask", "mask-lossless", "surface"):
                raise ValueError(f'volcomp: mode must be "q", "mask", "mask-lossless" or "surface", '
                                 f"got {self.mode!r}")
            if self.mode in ("mask", "mask-lossless"):
                object.__setattr__(self, "q", 0.0)  # a mask chunk has no quantiser
            elif self.mode == "surface":
                if not 2.0 <= self.q <= 255.0:
                    raise ValueError(f"volcomp: surface mode needs q in 2..255, got {self.q}")
                if not 1 <= self.threshold <= 255:
                    raise ValueError(f"volcomp: threshold must be 1..255, got {self.threshold}")
            elif not (self.q == 0.0 or 1.0 <= self.q <= 255.0):
                raise ValueError(f"volcomp: q must be 0 (lossless) or 1..255, got {self.q}")

        def validate(self, *, shape, dtype, chunk_grid) -> None:  # noqa: D401
            if str(np.dtype(dtype.to_native_dtype() if hasattr(dtype, "to_native_dtype") else dtype)) != "uint8":
                raise ValueError("volcomp codec requires uint8 data")
            cs = tuple(chunk_grid.chunk_shape)
            if cs != (CHUNK_DIM,) * 3:
                raise ValueError(f"volcomp codec requires 128^3 (inner) chunks, got {cs}")

        def evolve_from_array_spec(self, array_spec: ArraySpec) -> "VolcompCodec":
            return self

        def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
            return SURFACE_BOUND if self.mode == "surface" else ENCODE_BOUND

        async def _encode_single(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer | None:
            arr = np.ascontiguousarray(chunk_array.as_numpy_array(), dtype=np.uint8)
            if arr.shape != (CHUNK_DIM,) * 3:
                raise ValueError(f"volcomp: chunk must be 128^3, got {arr.shape}")
            if not arr.any() and not chunk_spec.fill_value:
                return None  # all-zero chunk -> stored as missing (fill value is 0 too)
            raw = arr.tobytes()
            if self.mode == "mask":
                enc = mask_encode(raw)
            elif self.mode == "mask-lossless":
                enc = mask_encode_lossless(raw)
            elif self.mode == "surface":
                enc = surface_encode(raw, self.q, self.threshold)
            else:
                enc = encode(raw, self.q)
            return chunk_spec.prototype.buffer.from_bytes(enc)

        async def _decode_single(self, chunk_bytes: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
            out = np.empty((CHUNK_DIM,) * 3, dtype=np.uint8)
            decode(chunk_bytes.as_numpy_array().tobytes(), out.reshape(-1).view(np.uint8))
            return chunk_spec.prototype.nd_buffer.from_numpy_array(out)

    register_codec("volcomp", VolcompCodec)
except ImportError:  # zarr / numpy not installed: the bytes-level API still works
    VolcompCodec = None
