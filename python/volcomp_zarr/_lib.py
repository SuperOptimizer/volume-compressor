"""ctypes binding to libvolcomp (built from python/volcomp_shim.c). Standard library only."""
import ctypes
import os
import sys

CHUNK_DIM = 128
CHUNK_VOXELS = CHUNK_DIM ** 3
BLOCK_VOXELS = 16 ** 3
STATUS = ["OK", "ERR_ARG", "ERR_CORRUPT", "ERR_VERSION", "ERR_NOMEM", "ERR_SHORT_BUF"]


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
_L.volcomp_shim_decode_block.restype = ctypes.c_int
_L.volcomp_shim_decode_block.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint,
                                         ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t]
_L.volcomp_shim_deblock.restype = None
_L.volcomp_shim_deblock.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float]

VERSION = _L.volcomp_shim_version().decode()
ENCODE_BOUND = _L.volcomp_shim_encode_bound()


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
    """src: 128^3 z-major uint8 bytes-like (2,097,152 bytes) -> bytes."""
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


def decode_block(enc, bz, by, bx):
    p, n, keep = _buf(enc)
    dst = bytearray(BLOCK_VOXELS)
    dp, dn, dkeep = _buf(dst)
    _check(_L.volcomp_shim_decode_block(p, n, bz, by, bx, dp, dn))
    return dst


def deblock(vol, nz, ny, nx, q):
    """In-place post-filter of a z-major uint8 volume (writable bytes-like of nz*ny*nx bytes)."""
    p, n, keep = _buf(vol)
    if n != nz * ny * nx:
        raise VolcompError("volume size does not match nz*ny*nx")
    _L.volcomp_shim_deblock(p, nz, ny, nx, q)
