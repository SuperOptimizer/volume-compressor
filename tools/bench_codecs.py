#!/usr/bin/env python3
"""Benchmark volcomp against general-purpose, image, video and 3-D compressors on raw scroll CT.

Every codec x setting x cube is encoded, decoded and scored on the same uint8 cube:

    bits per voxel, PSNR, SSIM (3-D Gaussian, sigma 1.5), MAE, max |error|,
    encode / decode MB/s (one thread wherever the codec lets us say so),
    random-access cost: bytes read + wall time to reconstruct the 128^3 chunk at the cube centre.

Storage model per codec family (this is what "bytes" and "random access" mean):

    chunked 3-D  (volcomp, zfp, sz3, jp3d, zstd, blosc, lz4)   one blob per 128^3 chunk, as a zarr store keeps them;
                                                                 random access = that chunk's blob
    per-slice 2-D (jpeg, jpeg2000, jpegxl, webp, avif)          one image per full z-slice;
                                                                 random access = the 128 slices through the chunk, cropped
    video (x264, x265, svtav1, aom; z = time)                   one closed GOP of 128 frames per file (keyint 128) or
                                                                 all-intra; random access = the GOP holding the chunk

    python3 tools/bench_codecs.py run --data /vesuvius/tmp_codecbench/cubes --out docs/bench --workers 10
    python3 tools/bench_codecs.py run --cubes PHercParis4_2.40um --codecs volcomp,jpegxl --out /tmp/b
    python3 tools/bench_codecs.py report --out docs/bench          # CSV/JSON, plots, tables

`run` is resumable: finished (cube, codec, setting) rows in <out>/results.jsonl are skipped.
Missing cubes are fetched from the open-data bucket (s3fs, anonymous) into --data.
Needs numpy, scipy, imagecodecs, zfpy, pillow (with AVIF), av (PyAV), matplotlib, ffmpeg on PATH (libx264,
libx265, libsvtav1, libaom-av1), python/volcomp_zarr (VOLCOMP_LIB=.../libvolcomp.so) and, for JP3D,
opj_jp3d_compress / opj_jp3d_decompress from OpenJPEG 2.1.x built with -DBUILD_JP3D=ON (--jp3d-bin).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLOSC_NTHREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

C = 128  # chunk edge, = volcomp chunk = zarr chunk
BUCKET = "vesuvius-challenge-open-data"
CUBES = {  # name: (sample, volume, origin zyx, edge). The 512^3 ones are docs/comparison's cubes.
    "PHerc0500P2_0.55um": ("PHerc0500P2", "20250821110041-0.550um-0.1m-65keV-masked.zarr", (5888, 7936, 1536), 512),
    "PHerc1667_1.13um": ("PHerc1667", "20260323082859-1.129um-0.2m-59keV-masked.zarr", (20864, 7168, 2944), 512),
    "PHercParis4_2.40um": ("PHercParis4", "20260411134726-2.400um-0.2m-78keV-masked.zarr", (37632, 22272, 22016), 512),
    "PHerc0191_9.36um": ("PHerc0191", "20250821151635-9.362um-1.2m-113keV-masked.zarr", (9216, 3584, 6016), 512),
    "PHercParis4_2.40um_1024": ("PHercParis4", "20260411134726-2.400um-0.2m-78keV-masked.zarr", (37376, 22016, 21760), 1024),
}

# codec name -> (family, parameter name, settings)
CODECS = {
    "zstd19": ("lossless", "level", [19]),
    "blosc-zstd5": ("lossless", "clevel", [5]),
    "lz4": ("lossless", "level", [0]),
    "jpeg": ("2d", "quality", [2, 5, 10, 25, 50, 70, 85, 95]),
    "jpeg2000": ("2d", "target PSNR", [28, 31, 34, 37, 40, 43, 46]),
    "jpegxl": ("2d", "distance", [0.5, 1, 2, 3, 5, 8, 12, 18, 25]),
    "webp": ("2d", "quality", [0, 5, 20, 40, 60, 80, 95]),
    "avif": ("2d", "quality", [0, 5, 10, 25, 40, 55, 70, 85, 95]),
    "x264": ("video", "crf", [8, 14, 20, 26, 32, 38, 44, 51]),
    "x265": ("video", "crf", [8, 14, 20, 26, 32, 38, 44, 51]),
    "svtav1": ("video", "crf", [8, 18, 28, 38, 48, 58, 63]),
    "aom": ("video", "crf", [8, 18, 28, 38, 48, 58, 63]),
    "x264-intra": ("video-intra", "crf", [8, 16, 24, 32, 40]),
    "x265-intra": ("video-intra", "crf", [8, 16, 24, 32, 40]),
    "svtav1-intra": ("video-intra", "crf", [8, 23, 38, 53, 63]),
    "aom-intra": ("video-intra", "crf", [8, 23, 38, 53]),
    "zfp-acc": ("3d", "tolerance", [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
    "zfp-rate": ("3d", "bits/value", [0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4]),
    "sz3": ("3d", "abs error", [0.5, 1, 2, 4, 8, 16, 32]),
    "jp3d": ("3d", "ratio", [10, 20, 40, 80, 160, 320]),
    "volcomp": ("volcomp", "q", [1, 2, 4, 8, 16, 32, 64, 128, 255]),
    "volcomp-smooth": ("volcomp", "q", [1, 2, 4, 8, 16, 32, 64, 128, 255]),
}
GOP = 128
# jp3d: OpenJPEG 2.1.2's JP3D tools build (gcc 15, -std=gnu89) but do not round-trip: even lossless (-r 1)
# decodes to noise (9 dB). Kept for anyone with a working build; not run by default.
DEFAULT_SKIP = {"jp3d"}


# ----------------------------------------------------------------------------------------------- data

def fetch_cube(name, data_dir):
    path = os.path.join(data_dir, f"{name}.u8")
    if os.path.exists(path):
        return path
    import s3fs
    import zarr
    sample, volume, o, n = CUBES[name]
    fs = s3fs.S3FileSystem(anon=True)
    a = zarr.open(zarr.storage.FsspecStore(fs, path=f"{BUCKET}/{sample}/volumes/{volume}/0", read_only=True), mode="r")
    x = np.ascontiguousarray(a[o[0]:o[0] + n, o[1]:o[1] + n, o[2]:o[2] + n], dtype=np.uint8)
    os.makedirs(data_dir, exist_ok=True)
    x.tofile(path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def load_cube(name, data_dir):
    n = CUBES[name][3]
    return np.fromfile(os.path.join(data_dir, f"{name}.u8"), dtype=np.uint8).reshape(n, n, n)


def centre_chunk(n):
    """Origin of the 128^3 chunk at the cube centre (chunk-grid aligned, lower-corner-of-centre)."""
    return (n // 2 // C) * C


# -------------------------------------------------------------------------------------------- metrics

def metrics(a, b, sigma=1.5, slab=64):
    """PSNR / MAE / max over the whole cube; SSIM = mean of the 3-D Gaussian-window SSIM map (sigma 1.5,
    truncate 3.5, K1 .01, K2 .03, L 255), excluding a 5-voxel border, computed in z-slabs with a halo."""
    n = a.shape[0]
    sse = 0
    sae = 0
    mx = 0
    for z in range(0, n, slab):
        d = a[z:z + slab].astype(np.int16) - b[z:z + slab].astype(np.int16)
        sse += int(np.einsum("ijk,ijk->", d, d, dtype=np.int64))
        ad = np.abs(d)
        sae += int(ad.sum(dtype=np.int64))
        mx = max(mx, int(ad.max()))
    nv = a.size
    mse = sse / nv
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    r = int(3.5 * sigma + 0.5)  # scipy.ndimage.gaussian_filter(sigma, truncate=3.5) support
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)[:, None]
    import cv2
    cv2.setNumThreads(1)

    def g(v):  # separable 3-D Gaussian, reflect borders (= scipy's mode="reflect")
        o = np.empty_like(v)
        for i in range(v.shape[0]):
            o[i] = cv2.sepFilter2D(v[i], -1, k, k, borderType=cv2.BORDER_REFLECT)
        sh = o.shape
        return cv2.sepFilter2D(o.reshape(sh[0], -1), -1, np.ones((1, 1), np.float32), k,
                               borderType=cv2.BORDER_REFLECT).reshape(sh)

    tot = 0.0
    cnt = 0
    for z in range(r, n - r, slab):
        z1 = min(z + slab, n - r)
        lo, hi = z - r, z1 + r
        x = a[lo:hi].astype(np.float32)  # float32 is within 1e-5 of float64 SSIM here (c2 = 58.5 >> eps)
        y = b[lo:hi].astype(np.float32)
        mx_, my_ = g(x), g(y)
        sxx = g(x * x) - mx_ * mx_
        syy = g(y * y) - my_ * my_
        sxy = g(x * y) - mx_ * my_
        s = ((2 * mx_ * my_ + c1) * (2 * sxy + c2)) / ((mx_ * mx_ + my_ * my_ + c1) * (sxx + syy + c2))
        s = s[r:r + z1 - z, r:s.shape[1] - r, r:s.shape[2] - r]
        tot += float(s.sum(dtype=np.float64))
        cnt += s.size
    return dict(psnr=psnr, ssim=tot / cnt, mae=sae / nv, maxerr=mx, mse=mse)


# ----------------------------------------------------------------------------------------- codecs
# Each codec: encode(vol, p) -> units (dict key -> bytes); decode(units, shape) -> vol;
# ra_keys(n, c0) -> the unit keys needed for the centre chunk; ra_decode(units, keys, n, c0) -> 128^3 chunk.

def chunk_keys(n):
    return [(z, y, x) for z in range(0, n, C) for y in range(0, n, C) for x in range(0, n, C)]


class Chunked:
    """Chunk-wise 3-D codec: one blob per 128^3 chunk."""
    threads = 1

    def enc1(self, c, p):
        raise NotImplementedError

    def dec1(self, b):
        raise NotImplementedError

    def encode(self, vol, p):
        n = vol.shape[0]
        return {k: self.enc1(np.ascontiguousarray(vol[k[0]:k[0] + C, k[1]:k[1] + C, k[2]:k[2] + C]), p)
                for k in chunk_keys(n)}

    def decode(self, units, n):
        out = np.empty((n, n, n), np.uint8)
        for k, b in units.items():
            out[k[0]:k[0] + C, k[1]:k[1] + C, k[2]:k[2] + C] = self.dec1(b)
        return out

    def ra_keys(self, n, c0):
        return [(c0, c0, c0)]

    def ra_decode(self, units, keys, n, c0):
        return self.dec1(units[keys[0]])


class Volcomp(Chunked):
    def __init__(self, smooth=False):
        import volcomp_zarr._lib as L
        self.L = L
        self.smooth = smooth

    def enc1(self, c, p):
        self.q = p
        return bytes(self.L.encode(c.tobytes(), float(p)))

    def dec1(self, b):
        if self.smooth:  # the README's CT recipe: gated face filter, strength 2, zero guard
            d = self.L.decode_smooth(b, sigma=2.0, zero_guard=True, gated=True)
        else:
            d = self.L.decode(b)
        return np.frombuffer(d, np.uint8).reshape(C, C, C)


class Zstd(Chunked):
    def enc1(self, c, p):
        import imagecodecs
        return imagecodecs.zstd_encode(c.tobytes(), level=p)

    def dec1(self, b):
        import imagecodecs
        return np.frombuffer(imagecodecs.zstd_decode(b), np.uint8).reshape(C, C, C)


class BloscZstd(Chunked):
    def __init__(self):
        import numcodecs
        from numcodecs import blosc
        blosc.set_nthreads(1)
        blosc.use_threads = False
        self.numcodecs = numcodecs

    def enc1(self, c, p):
        return self.numcodecs.Blosc(cname="zstd", clevel=p, shuffle=self.numcodecs.Blosc.SHUFFLE).encode(c)

    def dec1(self, b):
        return np.frombuffer(self.numcodecs.Blosc().decode(b), np.uint8).reshape(C, C, C)


class Lz4(Chunked):
    def enc1(self, c, p):
        import imagecodecs
        return imagecodecs.lz4_encode(c.tobytes())

    def dec1(self, b):
        import imagecodecs
        return np.frombuffer(imagecodecs.lz4_decode(b), np.uint8).reshape(C, C, C)


def _to_u8(f):
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


class Zfp(Chunked):
    def __init__(self, mode):
        import zfpy
        self.zfpy = zfpy
        self.mode = mode

    def enc1(self, c, p):
        f = c.astype(np.float32)
        if self.mode == "acc":
            return self.zfpy.compress_numpy(f, tolerance=float(p))
        return self.zfpy.compress_numpy(f, rate=float(p))

    def dec1(self, b):
        return _to_u8(self.zfpy.decompress_numpy(b)).reshape(C, C, C)


class Sz3(Chunked):
    def enc1(self, c, p):
        import imagecodecs
        return imagecodecs.sz3_encode(c.astype(np.float32), mode="abs", abs=float(p))

    def dec1(self, b):
        import imagecodecs
        return _to_u8(np.asarray(imagecodecs.sz3_decode(b, (C, C, C), np.float32))).reshape(C, C, C)


class Jp3d(Chunked):
    """OpenJPEG 2.1's JPEG 2000 Part 10 (JP3D) verification-model tools: 3-D 9/7 DWT (4 levels on each
    axis), 2-D EBCOT, one quality layer at the given compression ratio. One process per chunk and file I/O
    through /dev/shm, so its speeds include ~ms of process start per chunk."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp(dir="/dev/shm" if os.path.isdir("/dev/shm") else None, prefix="jp3d")
        with open(os.path.join(self.tmp, "c.img"), "w") as f:
            f.write(f"Bpp\t8\nColor Map\t2\nDimensions\t{C}\t{C}\t{C}\nResolution(mm)\t1\t1\t1\t\n")

    def _run(self, *args):
        r = subprocess.run(args, cwd=self.tmp, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{args[0]}: {r.stdout[-500:]} {r.stderr[-500:]}")

    def enc1(self, c, p):
        c.tofile(os.path.join(self.tmp, "c.bin"))
        o = os.path.join(self.tmp, "o.jp3d")
        self._run(os.path.join(os.environ["JP3D_BIN"], "opj_jp3d_compress"), "-i", "c.bin", "-m", "c.img", "-o", "o.jp3d",
                  "-I", "-n", "4,4,4", "-r", str(p))
        with open(o, "rb") as f:
            return f.read()

    def dec1(self, b):
        with open(os.path.join(self.tmp, "i.jp3d"), "wb") as f:
            f.write(b)
        self._run(os.path.join(os.environ["JP3D_BIN"], "opj_jp3d_decompress"), "-i", "i.jp3d", "-o", "d.bin")
        return np.fromfile(os.path.join(self.tmp, "d.bin"), np.uint8).reshape(C, C, C)

    def __del__(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class Slices:
    """Per-slice 2-D codec: one image per full z-slice (xy)."""
    threads = 1

    def encode(self, vol, p):
        return {z: self.enc2(np.ascontiguousarray(vol[z]), p) for z in range(vol.shape[0])}

    def decode(self, units, n):
        out = np.empty((n, n, n), np.uint8)
        for z, b in units.items():
            out[z] = self.dec2(b)
        return out

    def ra_keys(self, n, c0):
        return list(range(c0, c0 + C))

    def ra_decode(self, units, keys, n, c0):
        out = np.empty((C, C, C), np.uint8)
        for i, z in enumerate(keys):
            out[i] = self.dec2(units[z])[c0:c0 + C, c0:c0 + C]
        return out


class Jpeg(Slices):
    def enc2(self, s, p):
        import imagecodecs
        return imagecodecs.jpeg8_encode(s, level=int(p), optimize=True)

    def dec2(self, b):
        import imagecodecs
        return imagecodecs.jpeg8_decode(b)


class Jpeg2000(Slices):
    def enc2(self, s, p):
        import imagecodecs
        return imagecodecs.jpeg2k_encode(s, level=p, codecformat="J2K", reversible=False, numthreads=1)

    def dec2(self, b):
        import imagecodecs
        return imagecodecs.jpeg2k_decode(b, numthreads=1)


class JpegXL(Slices):
    def enc2(self, s, p):
        import imagecodecs
        return imagecodecs.jpegxl_encode(s, distance=float(p), effort=7, numthreads=1)

    def dec2(self, b):
        import imagecodecs
        return imagecodecs.jpegxl_decode(b, numthreads=1)


class WebP(Slices):
    """libwebp has no grey mode: the slice is replicated to RGB and coded as YUV 4:2:0 (neutral chroma)."""

    def enc2(self, s, p):
        import imagecodecs
        return imagecodecs.webp_encode(np.repeat(s[..., None], 3, 2), level=p, lossless=False, method=4)

    def dec2(self, b):
        import imagecodecs
        return np.ascontiguousarray(imagecodecs.webp_decode(b)[..., 1])


class Avif(Slices):
    """Pillow's libavif (aom encoder, dav1d/aom decoder), 8-bit monochrome 4:0:0, speed 6."""

    def __init__(self):
        from PIL import AvifImagePlugin
        AvifImagePlugin.DEFAULT_MAX_THREADS = 1

    def enc2(self, s, p):
        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(s).save(b, format="AVIF", quality=int(p), speed=6, max_threads=1)
        return b.getvalue()

    def dec2(self, b):
        from PIL import Image
        im = Image.open(io.BytesIO(b))
        a = np.asarray(im)
        return a if a.ndim == 2 else np.ascontiguousarray(a[..., 0])


class Video:
    """ffmpeg encoders with z as time: one closed GOP of 128 frames per bitstream (or all-intra), 8-bit.
    Encoded by the ffmpeg CLI (process start included in the time), decoded in-process with PyAV
    (h264 / hevc native decoders, dav1d for AV1), one thread each."""
    threads = 1

    def __init__(self, enc, intra):
        self.enc, self.intra = enc, intra
        self.fmt = {"x264": "h264", "x265": "hevc", "svtav1": "ivf", "aom": "ivf"}[enc]
        self.decname = {"x264": "h264", "x265": "hevc", "svtav1": "libdav1d", "aom": "libdav1d"}[enc]
        self.yuv = enc == "svtav1"  # libsvtav1 takes yuv420p only: neutral (128) chroma planes are added

    def cmd(self, n, p):
        k = 1 if self.intra else GOP
        pix = "yuv420p" if self.yuv else "gray"
        c = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{n}x{n}",
             "-r", "25", "-i", "-"]
        if self.enc == "x264":
            c += ["-c:v", "libx264", "-preset", "medium", "-tune", "psnr", "-crf", str(p), "-threads", "1",
                  "-x264-params", f"keyint={k}:min-keyint={k}:scenecut=0:lookahead-threads=1"]
        elif self.enc == "x265":
            c += ["-c:v", "libx265", "-preset", "medium", "-tune", "psnr", "-crf", str(p),
                  "-x265-params", f"pools=none:frame-threads=1:keyint={k}:min-keyint={k}:scenecut=0:log-level=error"]
        elif self.enc == "svtav1":
            c += ["-c:v", "libsvtav1", "-preset", "8", "-crf", str(p), "-g", str(k),
                  "-svtav1-params", "lp=1:tune=1"]
        elif self.enc == "aom":
            # all-intra via libaom's allintra usage (-g 1 in good-quality usage segfaults on 128 frames)
            c += ["-c:v", "libaom-av1", "-cpu-used", "6", "-crf", str(p), "-b:v", "0"]
            c += ["-usage", "allintra"] if self.intra else ["-g", str(k)]
            c += ["-threads", "1", "-row-mt", "0", "-tune", "psnr"]
        return c + ["-pix_fmt", pix, "-f", self.fmt, "-"]

    def frames(self, vol):
        if not self.yuv:
            return vol.tobytes()
        n = vol.shape[1]
        ch = np.full((vol.shape[0], n * n // 2), 128, np.uint8)
        return np.concatenate([vol.reshape(vol.shape[0], -1), ch], 1).tobytes()

    def encode(self, vol, p):
        n = vol.shape[0]
        units = {}
        for g in range(0, n, GOP):
            r = subprocess.run(self.cmd(n, p), input=self.frames(vol[g:g + GOP]), capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg rc {r.returncode}: {r.stderr.decode()[-800:]}")
            units[g] = r.stdout
        return units

    def dec_gop(self, b, n):
        import av
        ctx = av.CodecContext.create(self.decname, "r")
        ctx.thread_count = 1
        cont = av.open(io.BytesIO(b), format=self.fmt)
        out = []

        def take(fr):
            pl = fr.planes[0]
            a = np.frombuffer(pl, np.uint8).reshape(fr.height, pl.line_size)[:, :fr.width]
            out.append(a.copy())

        for pkt in cont.demux(cont.streams.video[0]):
            if pkt.size == 0:  # PyAV's end-of-stream marker; flushed once below
                continue
            for fr in ctx.decode(pkt):
                take(fr)
        try:
            for fr in ctx.decode(None):
                take(fr)
        except EOFError:
            pass
        cont.close()
        v = np.stack(out)
        assert v.shape[1:] == (n, n), v.shape
        return v

    def decode(self, units, n):
        out = np.empty((n, n, n), np.uint8)
        for g, b in units.items():
            v = self.dec_gop(b, n)
            assert v.shape[0] == min(GOP, n - g), (v.shape, g)
            out[g:g + GOP] = v
        return out

    def ra_keys(self, n, c0):
        return [(c0 // GOP) * GOP]

    def ra_decode(self, units, keys, n, c0):
        v = self.dec_gop(units[keys[0]], n)
        o = c0 - keys[0]
        return v[o:o + C, c0:c0 + C, c0:c0 + C]


def make_codec(name):
    if name == "volcomp":
        return Volcomp()
    if name == "volcomp-smooth":
        return Volcomp(smooth=True)
    if name == "zstd19":
        return Zstd()
    if name == "blosc-zstd5":
        return BloscZstd()
    if name == "lz4":
        return Lz4()
    if name == "zfp-acc":
        return Zfp("acc")
    if name == "zfp-rate":
        return Zfp("rate")
    if name == "sz3":
        return Sz3()
    if name == "jp3d":
        return Jp3d()
    if name in ("jpeg", "jpeg2000", "jpegxl", "webp", "avif"):
        return {"jpeg": Jpeg, "jpeg2000": Jpeg2000, "jpegxl": JpegXL, "webp": WebP, "avif": Avif}[name]()
    base = name.replace("-intra", "")
    return Video(base, name.endswith("-intra"))


# ------------------------------------------------------------------------------------------ one job

def run_job(job, timing_only=False):
    """timing_only: the quiet single-worker speed pass (sizes and times; quality metrics skipped)."""
    cube, name, p, data_dir = job
    vol = load_cube(cube, data_dir)
    n = vol.shape[0]
    codec = make_codec(name)
    reps = 3 if timing_only else 1  # speed pass: best of up to 3 while a repetition takes < 10 s
    t_enc = t_dec = float("inf")
    for i in range(reps):
        t0 = time.perf_counter()
        units = codec.encode(vol, p)
        t_enc = min(t_enc, time.perf_counter() - t0)
        if t_enc > 10:
            break
    for i in range(reps):
        t0 = time.perf_counter()
        dec = codec.decode(units, n)
        t_dec = min(t_dec, time.perf_counter() - t0)
        if t_dec > 10:
            break
    nbytes = sum(len(b) for b in units.values())
    m = dict(psnr=float("nan"), ssim=float("nan"), mae=float("nan"), maxerr=-1) if timing_only else metrics(vol, dec)
    c0 = centre_chunk(n)
    keys = codec.ra_keys(n, c0)
    ra_bytes = sum(len(units[k]) for k in keys)
    ra_t = []
    for _ in range(3):
        t0 = time.perf_counter()
        ch = codec.ra_decode(units, keys, n, c0)
        ra_t.append(time.perf_counter() - t0)
    ref = dec[c0:c0 + C, c0:c0 + C, c0:c0 + C]
    ra_ok = bool(np.array_equal(ch, ref))
    mb = vol.nbytes / 1e6
    fam = CODECS[name][0]
    return dict(cube=cube, codec=name, family=fam, param=CODECS[name][1], setting=p, n=n, bytes=nbytes,
                bpv=8.0 * nbytes / vol.size, ratio=vol.nbytes / nbytes, psnr=float(m["psnr"]), ssim=m["ssim"],
                mae=m["mae"], maxerr=m["maxerr"], enc_s=t_enc, dec_s=t_dec, enc_MBps=mb / t_enc,
                dec_MBps=mb / t_dec, ra_bytes=ra_bytes, ra_ms=1e3 * min(ra_t), ra_units=len(keys),
                ra_matches_full_decode=ra_ok, units=len(units), threads=codec.threads, host=os.uname().nodename,
                cpu=os.sched_getaffinity(0).__repr__())


def _init_worker(cpuq):
    try:  # a replacement worker (after a crash) finds the queue empty and runs unpinned
        os.sched_setaffinity(0, {cpuq.get(timeout=1)})
    except Exception:
        pass


def _safe_job(job):
    try:
        return run_job(job[:4], timing_only=len(job) > 4 and job[4])
    except Exception as e:  # recorded, not fatal: one failing setting must not kill a multi-hour run
        return dict(cube=job[0], codec=job[1], setting=job[2], error=f"{type(e).__name__}: {e}"[:1000])


def cmd_run(a):
    import multiprocessing as mp
    cubes = a.cubes.split(",") if a.cubes else list(CUBES)
    names = a.codecs.split(",") if a.codecs else [c for c in CODECS if c not in DEFAULT_SKIP]
    for c in cubes:
        fetch_cube(c, a.data)
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "timing.jsonl" if a.timing else "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        for line in open(res_path):
            r = json.loads(line)
            if "error" not in r:
                done.add((r["cube"], r["codec"], r["setting"]))
    pick = (lambda s: [s[len(s) // 2]]) if a.mid_only else (lambda s: s)
    jobs = [(c, nm, p, a.data, a.timing) for c in cubes for nm in names for p in pick(CODECS[nm][2])
            if (c, nm, p) not in done]
    # quick results first: every 512^3 job before the 1024^3 cube, and within a cube the fast codecs first
    slow = {"aom": 9, "aom-intra": 8, "x265": 7, "x265-intra": 6, "svtav1": 5, "svtav1-intra": 5, "jpegxl": 3,
            "avif": 3, "zstd19": 2, "jp3d": 2}
    # the reference cube (Paris4 2.4 um, the corpus volcomp was tuned on) goes first so a full first plot exists early
    jobs.sort(key=lambda j: (CUBES[j[0]][3], j[0] != "PHercParis4_2.40um", slow.get(j[1], 0), CUBE_ORDER.index(j[0])))
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    cpus = [int(x) for x in a.cpus.split(",")] if a.cpus else sorted(os.sched_getaffinity(0))[:a.workers]
    ctx = mp.get_context("spawn")
    cpuq = ctx.Queue()
    for c in cpus[:a.workers]:
        cpuq.put(c)
    with ctx.Pool(a.workers, initializer=_init_worker, initargs=(cpuq,)) as pool, \
            open(res_path, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_safe_job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if "error" in r:
                print(f"[{i + 1}/{len(jobs)}] {r['cube']} {r['codec']} {r['setting']}: ERROR {r['error']}", flush=True)
            else:
                print(f"[{i + 1}/{len(jobs)}] {r['cube']} {r['codec']} {r['setting']}: {r['bpv']:.4f} bpv "
                      f"{r['psnr']:.2f} dB ssim {r['ssim']:.4f} enc {r['enc_MBps']:.1f} dec {r['dec_MBps']:.1f} MB/s "
                      f"ra {r['ra_bytes']} B {r['ra_ms']:.1f} ms", flush=True)


# ------------------------------------------------------------------------------------------- report

FAMILY_STYLE = {  # colour per family (validated default categorical slots), volcomp highlighted
    "2d": ("#2a78d6", "per-slice 2-D image"),
    "video": ("#eb6834", "video, GOP 128"),
    "video-intra": ("#eda100", "video, all-intra"),
    "3d": ("#1baf7a", "3-D native"),
    "volcomp": ("#4a3aa7", "volcomp"),
}
MARKERS = {"jpeg": "o", "jpeg2000": "s", "jpegxl": "D", "webp": "v", "avif": "^",
           "x264": "o", "x265": "s", "svtav1": "D", "aom": "^",
           "x264-intra": "o", "x265-intra": "s", "svtav1-intra": "D", "aom-intra": "^",
           "zfp-acc": "o", "zfp-rate": "s", "sz3": "D", "jp3d": "^", "volcomp": "o", "volcomp-smooth": "s"}
CUBE_ORDER = list(CUBES)
OPS = (30.0, 35.0, 40.0)
RA_PSNR = 35.0


def load_results(out):
    rows = []
    for line in open(os.path.join(out, "results.jsonl")):
        r = json.loads(line)
        if "error" not in r:
            rows.append(r)
    # last write wins for a repeated key
    d = {(r["cube"], r["codec"], r["setting"]): r for r in rows}
    return sorted(d.values(), key=lambda r: (CUBE_ORDER.index(r["cube"]), list(CODECS).index(r["codec"]), r["bpv"]))


def bpv_at(rows, target, metric="psnr"):
    """log-linear interpolation of bpv at a quality target along one codec's sweep. Outside the sweep: None when
    the target is above the best quality reached, ("<=", bpv) when even the lowest rate is above the target."""
    pts = sorted((r[metric], r["bpv"]) for r in rows if np.isfinite(r[metric]))
    if len(pts) < 2 or target > pts[-1][0]:
        return None
    if target < pts[0][0]:
        return ("<=", min(p[1] for p in pts))
    q = np.array([p[0] for p in pts])
    b = np.log(np.array([p[1] for p in pts]))
    # enforce monotone (a sweep point that is both bigger and worse is dominated): running max of log bpv
    b = np.maximum.accumulate(b)
    return float(np.exp(np.interp(target, q, b)))


def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load_results(a.out)
    quiet = {}
    tp = os.path.join(a.out, "timing.jsonl")
    if os.path.exists(tp):
        for line in open(tp):
            t = json.loads(line)
            if "error" not in t:
                quiet[(t["cube"], t["codec"], t["setting"])] = t
    for r in rows:  # speeds from the quiet single-worker pass where it exists (the parallel pass is contended)
        t = quiet.get((r["cube"], r["codec"], r["setting"]))
        r["q_enc_MBps"] = t["enc_MBps"] if t else float("nan")
        r["q_dec_MBps"] = t["dec_MBps"] if t else float("nan")
        r["q_ra_ms"] = t["ra_ms"] if t else float("nan")
        if t:
            assert t["bytes"] == r["bytes"], (r["cube"], r["codec"], r["setting"], t["bytes"], r["bytes"])
    keys = ["cube", "codec", "family", "param", "setting", "n", "bytes", "bpv", "ratio", "psnr", "ssim", "mae",
            "maxerr", "q_enc_MBps", "q_dec_MBps", "q_ra_ms", "enc_MBps", "dec_MBps", "ra_bytes", "ra_ms", "ra_units",
            "ra_matches_full_decode", "threads", "host"]
    with open(os.path.join(a.out, "results.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(("%.6g" % r[k]) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump([{k: r[k] for k in keys} for r in rows], f, indent=0)
    cubes = [c for c in CUBE_ORDER if any(r["cube"] == c for r in rows)]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.linewidth": 0.6,
                         "xtick.color": "#444444", "ytick.color": "#444444", "axes.labelcolor": "#222222"})
    for metric, ylab, fn in (("psnr", "PSNR (dB)", "rd_psnr.png"), ("ssim", "SSIM (3-D, sigma 1.5)", "rd_ssim.png")):
        ncol = 3 if len(cubes) > 4 else 2
        nrow = (len(cubes) + ncol - 1) // ncol
        fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow), squeeze=False)
        handles = {}
        for ax, cube in zip(axs.flat, cubes):
            for name in CODECS:
                rr = [r for r in rows if r["cube"] == cube and r["codec"] == name]
                fam = CODECS[name][0]
                if not rr or fam == "lossless":
                    continue
                rr.sort(key=lambda r: r["bpv"])
                col, _ = FAMILY_STYLE[fam]
                is_vc = fam == "volcomp"
                ls = "--" if name.endswith("smooth") else "-"
                h, = ax.plot([r["bpv"] for r in rr], [r[metric] for r in rr], ls, color=col, marker=MARKERS[name],
                             ms=6.5 if is_vc else 4.5, lw=2.6 if is_vc else 1.2, alpha=1.0 if is_vc else 0.85,
                             zorder=5 if is_vc else 2, mec="white", mew=0.6, label=name)
                handles[name] = h
            ax.set_xscale("log")
            ax.set_title(cube.replace("_", "  ").replace("  1024", "  1024^3"), fontsize=10)
            ax.set_xlabel("bits per voxel")
            ax.set_ylabel(ylab)
            ax.grid(True, which="major", color="#dddddd", lw=0.5)
            ax.grid(True, which="minor", color="#f0f0f0", lw=0.4)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if metric == "psnr":
                ax.set_ylim(22, 58)
                for t in OPS:
                    ax.axhline(t, color="#bbbbbb", lw=0.6, ls=":", zorder=0)
            else:
                ax.set_ylim(0.5, 1.0)
        for ax in list(axs.flat)[len(cubes):]:
            ax.axis("off")
        order = [n for n in CODECS if n in handles]
        fig.legend([handles[n] for n in order], order, loc="lower center", ncol=min(10, len(order)), frameon=False,
                   fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
        fam_txt = "   ".join(f"{v[1]} = {k}" for k, v in FAMILY_STYLE.items())
        fig.suptitle(f"{ylab} vs bits per voxel, 8-bit scroll CT, one thread   (colour: family; volcomp thick)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        fig.savefig(os.path.join(a.out, fn), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # random access: per codec and cube, the cheapest setting with PSNR >= RA_PSNR
    ra = {}
    for name in CODECS:
        for cube in cubes:
            rr = [r for r in rows if r["cube"] == cube and r["codec"] == name]
            if not rr:
                continue
            ok = [r for r in rr if r["psnr"] >= RA_PSNR]  # the cheapest setting that reaches the target
            best = min(ok, key=lambda r: r["bpv"]) if ok else max(rr, key=lambda r: r["psnr"])
            ra.setdefault(name, {})[cube] = best
    with open(os.path.join(a.out, "random_access.csv"), "w") as f:
        f.write("cube,codec,setting,psnr,bpv,ra_bytes,ra_ms_quiet,ra_ms_parallel,ra_units\n")
        for name, d in ra.items():
            for cube, r in d.items():
                f.write(f"{cube},{name},{r['setting']},{r['psnr']:.2f},{r['bpv']:.4f},{r['ra_bytes']},"
                        f"{r['q_ra_ms']:.2f},{r['ra_ms']:.2f},{r['ra_units']}\n")
    sm = ["# Speed and random access at the cheapest setting reaching 35 dB PSNR", "",
          "One thread. `quiet` = the single-worker timing pass (`run --timing --workers 1`); `par` = the 12-worker "
          "quality pass (contended; shown only where no quiet number exists). Random access = bytes read and wall "
          "time to reconstruct the 128^3 chunk at the cube centre (best of 3).", ""]
    for cube in cubes:
        sm += [f"## {cube}", "", "| codec | setting | PSNR dB | bpv | enc MB/s | dec MB/s | RA bytes | RA ms | units read |",
               "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for name in CODECS:
            r = ra.get(name, {}).get(cube)
            if not r:
                continue
            qt = np.isfinite(r["q_ra_ms"])
            f = (lambda qk, pk: f"{r[qk]:.1f}" if qt else f"{r[pk]:.1f} (par)")
            sm.append(f"| {name} | {CODECS[name][1]} {r['setting']} | {r['psnr']:.2f} | {r['bpv']:.4f} | "
                      f"{f('q_enc_MBps', 'enc_MBps')} | {f('q_dec_MBps', 'dec_MBps')} | {r['ra_bytes']:,} | "
                      f"{f('q_ra_ms', 'ra_ms')} | {r['ra_units']} |")
        sm.append("")
    with open(os.path.join(a.out, "speed_random_access.md"), "w") as f:
        f.write("\n".join(sm) + "\n")
    for cube in [c for c in ("PHercParis4_2.40um", "PHercParis4_2.40um_1024") if c in cubes]:
        names = [n for n in CODECS if cube in ra.get(n, {})]
        fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(names) + 1.6), sharey=True)
        y = np.arange(len(names))
        for ax, key, lab in ((axs[0], "ra_bytes", "bytes read (KiB)"), (axs[1], "ra_ms", "decode time (ms)")):
            def val(r):
                if key == "ra_bytes":
                    return r["ra_bytes"] / 1024
                return r["q_ra_ms"] if np.isfinite(r["q_ra_ms"]) else r["ra_ms"]
            vals = [val(ra[n][cube]) for n in names]
            cols = [FAMILY_STYLE.get(CODECS[n][0], ("#8a8a85", ""))[0] for n in names]
            ax.barh(y, vals, color=cols, height=0.7, edgecolor="white", linewidth=1)
            ax.set_xscale("log")
            ax.set_xlabel(lab)
            ax.grid(True, axis="x", which="major", color="#dddddd", lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for yi, v in zip(y, vals):
                ax.text(v * 1.08, yi, f"{v:,.0f}" if v >= 10 else f"{v:.2f}", va="center", fontsize=7.5,
                        color="#444444")
        axs[0].set_yticks(y)
        axs[0].set_yticklabels([f"{n}  ({ra[n][cube]['psnr']:.1f} dB)" if np.isfinite(ra[n][cube]['psnr'])
                                else f"{n}  (lossless)" for n in names])
        axs[0].invert_yaxis()
        fig.suptitle(f"Random access, {cube}: one 128^3 chunk at the cube centre (cheapest setting >= 35 dB, 1 thread)",
                     fontsize=10.5)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"random_access_{cube}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # operating-point table
    md = ["# Operating points: bits per voxel at 30 / 35 / 40 dB PSNR", "",
          "Log-linear interpolation along each codec's sweep; `-` = the codec's best swept setting stays below "
          "the target; `<=x` = even its lowest swept rate (x bpv) is above the target. "
          "Lower is better; **bold** = best in the column.", ""]
    for cube in cubes:
        md.append(f"## {cube}")
        md.append("")
        md.append("| codec | family | bpv @ 30 dB | bpv @ 35 dB | bpv @ 40 dB | x volcomp @ 35 dB |")
        md.append("|---|---|--:|--:|--:|--:|")
        vals = {}
        for name in CODECS:
            rr = [r for r in rows if r["cube"] == cube and r["codec"] == name]
            if not rr or CODECS[name][0] == "lossless":
                continue
            vals[name] = [bpv_at(rr, t) for t in OPS]
        num = lambda x: x if isinstance(x, float) else None
        best = [min((num(v[i]) for v in vals.values() if num(v[i]) is not None), default=None) for i in range(3)]
        vc35 = num(vals.get("volcomp", [None] * 3)[1])
        for name, v in vals.items():
            cells = []
            for i, x in enumerate(v):
                if x is None:
                    cells.append("-")
                elif isinstance(x, tuple):
                    cells.append(f"<={x[1]:.3f}")
                else:
                    s = f"{x:.3f}"
                    cells.append(f"**{s}**" if best[i] is not None and abs(x - best[i]) < 1e-12 else s)
            rel = f"{v[1] / vc35:.2f}" if (num(v[1]) is not None and vc35) else "-"
            md.append(f"| {name} | {CODECS[name][0]} | " + " | ".join(cells) + f" | {rel} |")
        lossless = [r for r in rows if r["cube"] == cube and CODECS[r["codec"]][0] == "lossless"]
        if lossless:
            md.append("")
            md.append("Lossless: " + ", ".join(f"{r['codec']} {r['bpv']:.3f} bpv" for r in lossless) + ".")
        md.append("")
    with open(os.path.join(a.out, "operating_points.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cubes", default="", help="comma list of " + ",".join(CUBES) + " (default all)")
    r.add_argument("--codecs", default="", help="comma list of " + ",".join(CODECS) + " (default all)")
    r.add_argument("--out", required=True)
    r.add_argument("--data", default="cubes", help="directory of <cube>.u8 raw files (fetched if missing)")
    r.add_argument("--workers", type=int, default=1, help="parallel jobs, one pinned core each")
    r.add_argument("--cpus", default="", help="comma list of cores to pin workers to")
    r.add_argument("--timing", action="store_true",
                   help="speed pass: sizes + encode/decode/random-access times only, into timing.jsonl "
                        "(run with --workers 1 on a quiet core)")
    r.add_argument("--mid-only", action="store_true", help="smoke test: only the middle setting of each codec")
    r.add_argument("--jp3d-bin", default=os.environ.get("JP3D_BIN", ""), help="dir with opj_jp3d_compress")
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        if a.jp3d_bin:
            os.environ["JP3D_BIN"] = a.jp3d_bin
        cmd_run(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
