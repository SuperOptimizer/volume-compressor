/* Minimal blosc1 container reader (enough for zarr v2 chunks written by
 * numcodecs' Blosc codec with cname="zstd").
 *
 * Why not vendor c-blosc: the whole format we must read is a 16-byte header, a
 * table of block start offsets, and one zstd frame per block (per stream, when
 * the block is split). That is ~80 lines here against ~10k lines of vendored
 * code, and the only external dependency is libzstd, which the bench target
 * already optionally links.
 *
 * Header (little endian):
 *   0 version, 1 versionlz, 2 flags, 3 typesize, 4..8 nbytes, 8..12 blocksize,
 *   12..16 cbytes.  flags: 0x1 byte shuffle, 0x2 memcpyed, 0x4 bitshuffle,
 *   0x10 "do not split", 0xe0 compressor FORMAT code. Note that the format code
 *   is not blosc's compressor code: blosclz 0, lz4 and lz4hc 1, snappy 2,
 *   zlib 3, zstd 4 (BLOSC_*_FORMAT in blosc.h).
 * A non-memcpyed buffer is followed by nblocks int32 offsets (from the start of
 * the buffer) of the blocks; each block holds `nstreams` streams, each a 4-byte
 * little-endian compressed size and that many bytes (a stream whose compressed
 * size equals its uncompressed size is stored verbatim). */
#ifndef VOLCOMP_BLOSC1_H
#define VOLCOMP_BLOSC1_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef VOLCOMP_HAVE_ZSTD
#include <zstd.h>
#endif

#define BLOSC1_HEADER 16
#define BLOSC1_SHUFFLE 0x1u
#define BLOSC1_MEMCPYED 0x2u
#define BLOSC1_BITSHUFFLE 0x4u
#define BLOSC1_NOSPLIT 0x10u
#define BLOSC1_ZSTD 4u  /* BLOSC_ZSTD_FORMAT, not BLOSC_ZSTD (= 5) */

static inline uint32_t blosc1_rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* inverse of blosc's byte shuffle: dst[i * typesize + j] = src[j * n + i] */
static void blosc1_unshuffle(uint8_t *dst, const uint8_t *src, size_t bsize, size_t typesize) {
  if (typesize <= 1) {
    memcpy(dst, src, bsize);
    return;
  }
  size_t n = bsize / typesize, rem = bsize % typesize;
  for (size_t j = 0; j < typesize; j++)
    for (size_t i = 0; i < n; i++) dst[i * typesize + j] = src[j * n + i];
  if (rem) memcpy(dst + n * typesize, src + n * typesize, rem);
}

/* Decompress one blosc1 buffer into dst (capacity dstn). Returns the number of
 * bytes written, or -1 with *err set. `tmp` must have room for one block
 * (blocksize bytes); pass NULL to let the function use dst directly when the
 * buffer is not shuffled. */
static int64_t blosc1_decompress(const uint8_t *src, size_t srcn, uint8_t *dst, size_t dstn,
                                 uint8_t *tmp, size_t tmpn, const char **err) {
  const char *dummy;
  if (!err) err = &dummy;
  *err = NULL;
  if (srcn < BLOSC1_HEADER) return *err = "truncated blosc header", -1;
  unsigned flags = src[2], typesize = src[3];
  size_t nbytes = blosc1_rd32(src + 4), blocksize = blosc1_rd32(src + 8), cbytes = blosc1_rd32(src + 12);
  if (src[0] > 2) return *err = "unsupported blosc format version", -1;
  if (cbytes > srcn) return *err = "blosc cbytes past end of buffer", -1;
  if (nbytes > dstn) return *err = "blosc buffer larger than destination", -1;
  if (flags & BLOSC1_BITSHUFFLE) return *err = "bitshuffle is not supported", -1;
  if (flags & BLOSC1_MEMCPYED) {
    if (srcn < BLOSC1_HEADER + nbytes) return *err = "truncated memcpyed blosc buffer", -1;
    memcpy(dst, src + BLOSC1_HEADER, nbytes);
    return (int64_t)nbytes;
  }
  unsigned compcode = (flags & 0xe0u) >> 5;
  if (compcode != BLOSC1_ZSTD) return *err = "only the zstd blosc codec is supported", -1;
#ifndef VOLCOMP_HAVE_ZSTD
  return *err = "built without libzstd", -1;
#else
  if (!blocksize || nbytes == 0) return *err = "blosc blocksize is zero", -1;
  size_t nblocks = (nbytes + blocksize - 1) / blocksize;
  if (srcn < BLOSC1_HEADER + nblocks * 4) return *err = "truncated blosc block table", -1;
  bool shuffle = (flags & BLOSC1_SHUFFLE) && typesize > 1;
  if (shuffle && (!tmp || tmpn < blocksize)) return *err = "no scratch for unshuffle", -1;
  for (size_t b = 0; b < nblocks; b++) {
    size_t start = blosc1_rd32(src + BLOSC1_HEADER + b * 4);
    size_t bsize = nbytes - b * blocksize < blocksize ? nbytes - b * blocksize : blocksize;
    bool leftover = bsize != blocksize;
    size_t nstreams = (!(flags & BLOSC1_NOSPLIT) && !leftover && typesize > 1) ? typesize : 1;
    if (bsize % nstreams) return *err = "blosc block is not divisible into streams", -1;
    size_t neblock = bsize / nstreams;
    uint8_t *out = shuffle ? tmp : dst + b * blocksize;
    if (start >= srcn) return *err = "blosc block start out of range", -1;
    const uint8_t *p = src + start;
    for (size_t s = 0; s < nstreams; s++) {
      if ((size_t)(p - src) + 4 > srcn) return *err = "truncated blosc stream header", -1;
      size_t csize = blosc1_rd32(p);
      p += 4;
      if ((size_t)(p - src) + csize > srcn) return *err = "truncated blosc stream", -1;
      if (csize == neblock) {
        memcpy(out + s * neblock, p, neblock);
      } else {
        size_t got = ZSTD_decompress(out + s * neblock, neblock, p, csize);
        if (ZSTD_isError(got) || got != neblock) return *err = "zstd block decode failed", -1;
      }
      p += csize;
    }
    if (shuffle) blosc1_unshuffle(dst + b * blocksize, tmp, bsize, typesize);
  }
  return (int64_t)nbytes;
#endif
}

#endif
