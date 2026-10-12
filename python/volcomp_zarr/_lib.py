"""ctypes binding to libvolcomp (built from python/volcomp_shim.c). Standard library only."""
import ctypes
import os
import sys

Q_LOSSLESS = 0.0  # volcomp_encode(..., q=0) is exact
CHUNK_DIM = 128
CHUNK_VOXELS = CHUNK_DIM ** 3
BLOCK_VOXELS = 16 ** 3
MASK_DIM = 64            # the grid a mode-4 mask chunk stores: CHUNK_DIM // 2
MASK_VOXELS = MASK_DIM ** 3
MASK_LL_DIM = CHUNK_DIM  # a mode-5 (lossless) mask chunk stores the chunk itself
MASK_LL_VOXELS = CHUNK_VOXELS
STATUS = ["OK", "ERR_ARG", "ERR_CORRUPT", "ERR_VERSION", "ERR_NOMEM", "ERR_SHORT_BUF"]
# element types (volcomp_dtype), by numpy name; uint8 is the u8 modes 0..6, the rest the typed modes 7 and 8
DTYPES = {"uint8": 0, "uint16": 1, "int16": 2, "uint32": 3, "int32": 4, "float32": 5}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}


class VolcompError(RuntimeError):
    pass


def _find_lib():
    env = os.environ.get("VOLCOMP_LIB")
    cands = [env] if env else []
    here = os.path.dirname(os.path.abspath(__file__))
    name = "libvolcomp.dylib" if sys.platform == "darwin" else "libvolcomp.so"
    cands += [os.path.join(here, name), os.path.join(here, "..", name),
              os.path.join(here, "..", "..", "build", "release", name), name]
    for c in cands:
        if c and (os.path.exists(c) or c == name):
            try:
                return ctypes.CDLL(c)
            except OSError:
                continue
    raise VolcompError("libvolcomp not found; build python/volcomp_shim.c (see python/README.md) "
                       "or set VOLCOMP_LIB=/path/to/libvolcomp.so")


_L = _find_lib()
_L.volcomp_shim_version.restype = ctypes.c_char_p
_L.volcomp_shim_encode_bound.restype = ctypes.c_size_t
_L.volcomp_shim_status_string.restype = ctypes.c_char_p
_L.volcomp_shim_status_string.argtypes = [ctypes.c_int]
_L.volcomp_shim_encode.restype = ctypes.c_int
_L.volcomp_shim_encode.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_size_t)]
_L.volcomp_shim_decode.restype = ctypes.c_int
_L.volcomp_shim_decode.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
_L.volcomp_shim_decode_smooth.restype = ctypes.c_int
_L.volcomp_shim_decode_smooth.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.c_float, ctypes.c_uint]
_L.volcomp_shim_deblock_ex.restype = None
_L.volcomp_shim_deblock_ex.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float,
                                       ctypes.c_uint]
DEBLOCK_ZERO_GUARD = 1  # leave exact zeros (masked air) and the faces touching them alone
_L.volcomp_shim_decode_block.restype = ctypes.c_int
_L.volcomp_shim_decode_block.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint,
                                         ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t]
_L.volcomp_shim_stream_q.restype = ctypes.c_int
_L.volcomp_shim_stream_q.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float)]
_L.volcomp_shim_mask_encode.restype = ctypes.c_int
_L.volcomp_shim_mask_encode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.POINTER(ctypes.c_size_t)]
_L.volcomp_shim_mask_info.restype = ctypes.c_int
_L.volcomp_shim_mask_info.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32)]
_L.volcomp_shim_mask_decode_stored.restype = ctypes.c_int
_L.volcomp_shim_mask_decode_stored.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                               ctypes.c_size_t]
_L.volcomp_shim_mask_voxels.restype = ctypes.c_size_t
_L.volcomp_shim_mask_encode_lossless.restype = ctypes.c_int
_L.volcomp_shim_mask_encode_lossless.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                                 ctypes.POINTER(ctypes.c_size_t)]
_L.volcomp_shim_mask_ll_bound.restype = ctypes.c_size_t
_L.volcomp_shim_surface_bound.restype = ctypes.c_size_t
_L.volcomp_shim_surface_encode.restype = ctypes.c_int
_L.volcomp_shim_surface_encode.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_uint, ctypes.c_void_p,
                                           ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_L.volcomp_shim_surface_info.restype = ctypes.c_int
_L.volcomp_shim_surface_info.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint32),
                                         ctypes.POINTER(ctypes.c_uint32)]
_L.volcomp_shim_is_lossless.restype = ctypes.c_int
_L.volcomp_shim_is_lossless.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
_L.volcomp_shim_dtype_size.restype = ctypes.c_size_t
_L.volcomp_shim_dtype_size.argtypes = [ctypes.c_int]
_L.volcomp_shim_typed_encode_bound.restype = ctypes.c_size_t
_L.volcomp_shim_typed_encode_bound.argtypes = [ctypes.c_int]
_L.volcomp_shim_encode_typed.restype = ctypes.c_int
_L.volcomp_shim_encode_typed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
                                         ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_L.volcomp_shim_decode_typed.restype = ctypes.c_int
_L.volcomp_shim_decode_typed.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                                         ctypes.c_int]
_L.volcomp_shim_decode_block_typed.restype = ctypes.c_int
_L.volcomp_shim_decode_block_typed.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint,
                                               ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_L.volcomp_shim_chunk_dtype.restype = ctypes.c_int
_L.volcomp_shim_chunk_dtype.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_int)]
_L.volcomp_shim_deblock.restype = None
_L.volcomp_shim_deblock.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]

VERSION = _L.volcomp_shim_version().decode()
ENCODE_BOUND = _L.volcomp_shim_encode_bound()
MASK_LL_BOUND = _L.volcomp_shim_mask_ll_bound()  # 9 + 128^3/8 = 262 153
SURFACE_BOUND = _L.volcomp_shim_surface_bound()
SURFACE_THRESHOLD = 128  # the default exact threshold of a surface chunk: u8 >= 128 is p >= 0.5
TYPED_ENCODE_BOUND = _L.volcomp_shim_typed_encode_bound(5)  # 10 + 128^3 * 4: the largest typed chunk


def _check(st):
    if st:
        raise VolcompError(_L.volcomp_shim_status_string(st).decode())


def _buf(obj):
    """(address, length) of any writable or read-only contiguous bytes-like object."""
    mv = memoryview(obj)
    if not mv.c_contiguous:
        raise VolcompError("buffer must be C-contiguous")
    n = mv.nbytes
    if mv.readonly:
        arr = (ctypes.c_char * n).from_buffer_copy(mv) if n < 1 << 20 else None
        if arr is not None:
            return ctypes.addressof(arr), n, arr
        # large read-only input: avoid a copy via the buffer protocol
        c = (ctypes.c_char * n).from_buffer(bytearray(mv))
        return ctypes.addressof(c), n, c
    c = (ctypes.c_char * n).from_buffer(mv)
    return ctypes.addressof(c), n, c


def encode(src, q):
    """src: 128^3 z-major uint8 bytes-like (2,097,152 bytes) -> bytes.

    q is Q_LOSSLESS (0) for the exact codec, or 1..255 for the lossy DCT codec.
    """
    p, n, keep = _buf(src)
    if n != CHUNK_VOXELS:
        raise VolcompError(f"source must be {CHUNK_VOXELS} bytes, got {n}")
    out = ctypes.create_string_buffer(ENCODE_BOUND)
    got = ctypes.c_size_t()
    _check(_L.volcomp_shim_encode(p, q, out, ENCODE_BOUND, ctypes.byref(got)))
    return out.raw[: got.value]


def decode(enc, out=None):
    """enc: bytes-like stream -> bytearray of 128^3 voxels (or fills `out`, a writable 2,097,152-byte buffer)."""
    p, n, keep = _buf(enc)
    dst = out if out is not None else bytearray(CHUNK_VOXELS)
    dp, dn, dkeep = _buf(dst)
    _check(_L.volcomp_shim_decode(p, n, dp, dn))
    return dst


SMOOTH_GATED = 2  # smooth with the gated face filter (strength x q) rather than a Gaussian


def decode_smooth(enc, sigma=0.6, out=None, zero_guard=False, gated=False):
    """Like `decode`, plus the optional decode-side deblocking: the voxels within two of
    every block face are blurred (Gaussian sigma) and each block is projected back onto
    the quantisation cells of its coefficients (lossy and surface chunks; any other chunk
    decodes as `decode`). A surface chunk keeps its exact threshold. Not part of the
    format: the stream is the same, only the reader's reconstruction differs."""
    p, n, keep = _buf(enc)
    dst = out if out is not None else bytearray(CHUNK_VOXELS)
    dp, dn, dkeep = _buf(dst)
    flags = (DEBLOCK_ZERO_GUARD if zero_guard else 0) | (SMOOTH_GATED if gated else 0)
    _check(_L.volcomp_shim_decode_smooth(p, n, dp, dn, sigma, flags))
    return dst


def mask_encode(src):
    """src: 128^3 z-major uint8 bytes-like -> a volcomp MASK chunk.

    Any nonzero source voxel is 1; the encoder stores the 2x2x2 MAJORITY pool
    (a 64^3 grid) exactly, and `decode` returns it trilinearly interpolated back
    to 128^3 as a continuous 0..255 field. Nothing is configurable.
    """
    p, n, keep = _buf(src)
    if n != CHUNK_VOXELS:
        raise VolcompError(f"source must be {CHUNK_VOXELS} bytes, got {n}")
    out = ctypes.create_string_buffer(ENCODE_BOUND)
    got = ctypes.c_size_t()
    _check(_L.volcomp_shim_mask_encode(p, out, ENCODE_BOUND, ctypes.byref(got)))
    return out.raw[: got.value]


def mask_encode_lossless(src):
    """src: 128^3 z-major uint8 bytes-like -> a volcomp LOSSLESS MASK chunk (mode 5).

    Any nonzero source voxel is 1; the whole 128^3 mask is stored exactly with the
    same context-model range coder mask_encode uses on the 2x pool, and `decode`
    returns it as 0/255, voxel for voxel. Nothing is configurable.
    """
    p, n, keep = _buf(src)
    if n != CHUNK_VOXELS:
        raise VolcompError(f"source must be {CHUNK_VOXELS} bytes, got {n}")
    out = ctypes.create_string_buffer(ENCODE_BOUND)
    got = ctypes.c_size_t()
    _check(_L.volcomp_shim_mask_encode_lossless(p, out, ENCODE_BOUND, ctypes.byref(got)))
    return out.raw[: got.value]


def surface_encode(src, q, threshold=SURFACE_THRESHOLD):
    """src: 128^3 z-major uint8 bytes-like -> a volcomp SURFACE chunk (mode 6).

    For smooth probability fields that consumers threshold: the DCT codec at step
    q plus a refinement layer that makes `decoded >= threshold` equal
    `src >= threshold` for every voxel (the mask is exact; the soft values carry
    the DCT error at q). `decode` reads it like any other chunk.
    """
    p, n, keep = _buf(src)
    if n != CHUNK_VOXELS:
        raise VolcompError(f"source must be {CHUNK_VOXELS} bytes, got {n}")
    out = ctypes.create_string_buffer(SURFACE_BOUND)
    got = ctypes.c_size_t()
    _check(_L.volcomp_shim_surface_encode(p, q, threshold, out, SURFACE_BOUND, ctypes.byref(got)))
    return out.raw[: got.value]


def surface_info(enc):
    """(threshold, margin) if `enc` is a surface chunk, else None (margin 0: no refinement was needed)."""
    p, n, keep = _buf(enc)
    thr, margin = ctypes.c_uint32(), ctypes.c_uint32()
    st = _L.volcomp_shim_surface_info(p, n, ctypes.byref(thr), ctypes.byref(margin))
    if st == 1:  # ERR_ARG: a well-formed stream that is not a surface chunk
        return None
    _check(st)
    return thr.value, margin.value


def mask_info(enc):
    """The stored grid edge if `enc` is a mask chunk, else None.

    64 for the 2x mode (mask_encode), 128 for the lossless one
    (mask_encode_lossless); that is how the two are told apart.
    """
    p, n, keep = _buf(enc)
    dim = ctypes.c_uint32()
    st = _L.volcomp_shim_mask_info(p, n, ctypes.byref(dim))
    if st == 1:  # ERR_ARG: a well-formed stream that is not a mask chunk
        return None
    _check(st)
    return dim.value


def mask_decode_stored(enc, out=None):
    """The STORED grid of a mask chunk, as 0/255: 64^3 for mode 4, 128^3 for mode 5
    (a bytearray, or `out` filled)."""
    p, n, keep = _buf(enc)
    dim = mask_info(enc)
    if dim is None:
        raise VolcompError("not a mask chunk")
    dst = out if out is not None else bytearray(dim ** 3)
    dp, dn, dkeep = _buf(dst)
    _check(_L.volcomp_shim_mask_decode_stored(p, n, dp, dn))
    return dst


def stream_q(enc):
    """The q a stream was encoded with; 0.0 (Q_LOSSLESS) for a lossless stream."""
    p, n, keep = _buf(enc)
    q = ctypes.c_float()
    _check(_L.volcomp_shim_stream_q(p, n, ctypes.byref(q)))
    return q.value


def is_lossless(enc):
    """True if the stream decodes to its source exactly (false for a mask chunk)."""
    p, n, keep = _buf(enc)
    out = ctypes.c_int()
    _check(_L.volcomp_shim_is_lossless(p, n, ctypes.byref(out)))
    return bool(out.value)


def decode_block(enc, bz, by, bx):
    p, n, keep = _buf(enc)
    dst = bytearray(BLOCK_VOXELS)
    dp, dn, dkeep = _buf(dst)
    _check(_L.volcomp_shim_decode_block(p, n, bz, by, bx, dp, dn))
    return dst


def deblock(vol, nz, ny, nx, q, zero_guard=True):
    """In-place post-filter of a z-major uint8 volume (writable bytes-like of nz*ny*nx bytes)."""
    p, n, keep = _buf(vol)
    if n != nz * ny * nx:
        raise VolcompError("volume size does not match nz*ny*nx")
    _L.volcomp_shim_deblock_ex(p, nz, ny, nx, q, DEBLOCK_ZERO_GUARD if zero_guard else 0)


def _dtype_code(dtype):
    """volcomp_dtype code of a numpy dtype / type / name ("uint16", np.uint16, np.dtype("<u2")) or an int code."""
    if isinstance(dtype, int):
        if dtype in DTYPE_NAMES:
            return dtype
        raise VolcompError(f"unknown dtype code {dtype}")
    name = getattr(dtype, "name", None)
    if not isinstance(name, str):
        name = getattr(dtype, "__name__", None) or str(dtype)
    if name not in DTYPES:
        raise VolcompError(f"volcomp has no element type {dtype!r}; one of {sorted(DTYPES)}")
    return DTYPES[name]


def dtype_size(dtype):
    """Bytes per element of a volcomp dtype (1, 2 or 4)."""
    return _L.volcomp_shim_dtype_size(_dtype_code(dtype))


def typed_encode_bound(dtype):
    """The largest chunk encode_typed writes for this dtype: 10 + 128^3 * size (8 + 128^3 for uint8)."""
    return _L.volcomp_shim_typed_encode_bound(_dtype_code(dtype))


def encode_typed(src, dtype, q=0.0):
    """src: 128^3 z-major elements of `dtype` in host byte order (bytes-like, e.g. a C-contiguous
    numpy array) -> a volcomp TYPED chunk.

    dtype: uint16, int16, uint32, int32 or float32 (or uint8: the u8 lossless mode, q = 0 only).
    q = 0 (Q_LOSSLESS) is exact (mode 7); q > 0 is error-bounded (mode 8): every element
    decodes within q / 2 of its source. An integer dtype uses the step floor(q) (< 2: exact);
    a chunk mode 8 cannot represent (NaN or infinity, out-of-range steps) is stored exactly.
    """
    code = _dtype_code(dtype)
    p, n, keep = _buf(src)
    want = CHUNK_VOXELS * _L.volcomp_shim_dtype_size(code)
    if n != want:
        raise VolcompError(f"source must be {want} bytes for {DTYPE_NAMES[code]}, got {n}")
    cap = _L.volcomp_shim_typed_encode_bound(code)
    out = ctypes.create_string_buffer(cap)
    got = ctypes.c_size_t()
    _check(_L.volcomp_shim_encode_typed(p, code, q, out, cap, ctypes.byref(got)))
    return out.raw[: got.value]


def decode_typed(enc, dtype, out=None):
    """enc: a chunk of element type `dtype` -> bytearray of 128^3 elements in host byte order (or
    fills `out`, a writable buffer of 128^3 * dtype_size(dtype) bytes, e.g. a numpy array).
    Raises VolcompError if the chunk holds another dtype (chunk_dtype tells which)."""
    code = _dtype_code(dtype)
    p, n, keep = _buf(enc)
    dst = out if out is not None else bytearray(CHUNK_VOXELS * _L.volcomp_shim_dtype_size(code))
    dp, dn, dkeep = _buf(dst)
    _check(_L.volcomp_shim_decode_typed(p, n, dp, dn, code))
    return dst


def decode_block_typed(enc, bz, by, bx, dtype):
    """Block (bz, by, bx), each 0..7, of a chunk as 16^3 elements of `dtype` (a bytearray)."""
    code = _dtype_code(dtype)
    p, n, keep = _buf(enc)
    dst = bytearray(BLOCK_VOXELS * _L.volcomp_shim_dtype_size(code))
    dp, dn, dkeep = _buf(dst)
    _check(_L.volcomp_shim_decode_block_typed(p, n, bz, by, bx, dp, dn, code))
    return dst


def chunk_dtype(enc):
    """The element type a chunk decodes to, as a numpy name: "uint8" for the u8 modes, else the typed chunk's."""
    p, n, keep = _buf(enc)
    out = ctypes.c_int()
    _check(_L.volcomp_shim_chunk_dtype(p, n, ctypes.byref(out)))
    return DTYPE_NAMES[out.value]
