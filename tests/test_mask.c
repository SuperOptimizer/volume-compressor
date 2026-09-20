/* Binary mask chunks: mode 4 (spec/format.md §11) — the 2x2x2 majority pool
 * stored exactly and interpolated back — and mode 5 (§12) — the full 128^3 mask
 * stored exactly with the same coder. Round trips, the const and raw forms,
 * stream validation, and the size of a real PHercParis4 recto mask at both. */
#include "../volcomp.h"
#include "check.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define N VOLCOMP_MASK_DIM
#define C VOLCOMP_CHUNK_DIM

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ---------------------------------------------------------------- reference */

/* the stored grid: 2x2x2 majority (count * 2 >= 8) of the 0/nonzero source */
static void ref_pool(const uint8_t *src, uint8_t *grid) {
  for (uint32_t z = 0; z < N; z++)
    for (uint32_t y = 0; y < N; y++)
      for (uint32_t x = 0; x < N; x++) {
        uint32_t c = 0;
        for (uint32_t dz = 0; dz < 2; dz++)
          for (uint32_t dy = 0; dy < 2; dy++)
            for (uint32_t dx = 0; dx < 2; dx++)
              c += src[(((size_t)(2 * z + dz) * C) + 2 * y + dy) * C + 2 * x + dx] != 0;
        grid[((size_t)z * N + y) * N + x] = (uint8_t)(c * 2 >= 8);
      }
}

/* straightforward trilinear interpolation of the 0/1 grid: output voxel j
 * samples the grid at j/2, the upper neighbour clamped at the top edge */
static uint8_t ref_interp(const uint8_t *g, uint32_t jz, uint32_t jy, uint32_t jx) {
  uint32_t j[3] = {jz, jy, jx}, i0[3], i1[3];
  double w[3];
  for (int d = 0; d < 3; d++) {
    double t = (double)j[d] * 0.5;
    uint32_t a = (uint32_t)floor(t);
    w[d] = t - (double)a;
    i0[d] = a < N ? a : N - 1;
    i1[d] = i0[d] + 1 < N ? i0[d] + 1 : i0[d];
  }
  double v = 0;
  for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
      for (int dx = 0; dx < 2; dx++) {
        double ww = (dz ? w[0] : 1 - w[0]) * (dy ? w[1] : 1 - w[1]) * (dx ? w[2] : 1 - w[2]);
        uint32_t zz = dz ? i1[0] : i0[0], yy = dy ? i1[1] : i0[1], xx = dx ? i1[2] : i0[2];
        v += ww * (double)g[((size_t)zz * N + yy) * N + xx];
      }
  return (uint8_t)floor(255.0 * v + 0.5);
}

/* a filled sphere: a large smooth surface, the easy end for the context model */
static void synth_sphere(uint8_t *v, double r) {
  for (uint32_t z = 0; z < C; z++)
    for (uint32_t y = 0; y < C; y++)
      for (uint32_t x = 0; x < C; x++) {
        double dz = (double)z - 63.5, dy = (double)y - 63.5, dx = (double)x - 63.5;
        v[((size_t)z * C + y) * C + x] = dz * dz + dy * dy + dx * dx <= r * r ? 255u : 0u;
      }
}

/* uniformly random bits: the hard end — the coder must fall back to RAW */
static void synth_random(uint8_t *v, uint32_t seed) {
  uint32_t r = seed | 1u;
  for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) v[i] = (uint8_t)(vt_rng(&r) & 1u ? 255u : 0u);
}

/* deterministic blobby binary chunk */
static void synth_mask(uint8_t *v, uint32_t seed) {
  uint32_t r = seed | 1u;
  for (uint32_t z = 0; z < C; z++)
    for (uint32_t y = 0; y < C; y++)
      for (uint32_t x = 0; x < C; x++) {
        double s = sin(z * 0.11 + seed) * cos(y * 0.07) + sin((x + y) * 0.05) + 0.4 * sin(x * 0.3 + z * 0.2);
        s += 0.02 * (double)(vt_rng(&r) % 100) - 1.0;
        v[((size_t)z * C + y) * C + x] = s > 0 ? 255u : 0u;
      }
}

/* ------------------------------------------------------------ the fixture ---
 * tests/data/recto256.vlz: u32 count, count u32 lengths, then the streams —
 * eight 128^3 chunks of the PHercParis4 recto surface prediction (2.4 um,
 * 19.6 % foreground) stored as volcomp LOSSLESS chunks. */
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint8_t *fixture_chunks(const char *path, uint32_t *count) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *buf = malloc((size_t)sz);
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    fclose(f);
    free(buf);
    return NULL;
  }
  fclose(f);
  uint32_t n = rd32(buf);
  uint8_t *out = malloc((size_t)n * VOLCOMP_CHUNK_VOXELS);
  size_t off = 4u + 4u * n;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t len = rd32(buf + 4 + 4 * i);
    if (volcomp_decode(buf + off, len, out + (size_t)i * VOLCOMP_CHUNK_VOXELS, VOLCOMP_CHUNK_VOXELS) !=
        VOLCOMP_OK) {
      free(buf);
      free(out);
      return NULL;
    }
    off += len;
  }
  free(buf);
  *count = n;
  return out;
}

/* ------------------------------------------------------------------ tests */

static void test_roundtrip(uint8_t *src, uint8_t *enc, uint8_t *dec, uint8_t *grid, uint8_t *ref) {
  for (uint32_t seed = 1; seed <= 3; seed++) {
    synth_mask(src, seed);
    size_t n = 0;
    CHECK_EQ(volcomp_mask_encode(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK(n <= VMK_HDR_BYTES + VOLCOMP_MASK_VOXELS / 8u);
    uint32_t dim = 0;
    CHECK_EQ(volcomp_mask_info(enc, n, &dim), VOLCOMP_OK);
    CHECK_EQ(dim, VOLCOMP_MASK_DIM);
    bool ll = true;
    CHECK_EQ(volcomp_is_lossless(enc, n, &ll), VOLCOMP_OK);
    CHECK_EQ(ll, 0); /* a mask chunk is not an exact copy of its source */
    float q = -1;
    CHECK_EQ(volcomp_stream_q(enc, n, &q), VOLCOMP_OK);
    CHECK(q == 0.0f);

    /* the stored grid is exact */
    ref_pool(src, ref);
    CHECK_EQ(volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
    unsigned bad = 0;
    for (size_t i = 0; i < VOLCOMP_MASK_VOXELS; i++) bad += grid[i] != (ref[i] ? 255u : 0u);
    CHECK_EQ(bad, 0);

    /* the decode is the reference trilinear interpolation of that grid */
    CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
    bad = 0;
    unsigned centres = 0, centre_bad = 0;
    for (uint32_t z = 0; z < C; z++)
      for (uint32_t y = 0; y < C; y++)
        for (uint32_t x = 0; x < C; x++) {
          uint8_t got = dec[((size_t)z * C + y) * C + x], want = ref_interp(ref, z, y, x);
          bad += got != want;
          if (!(z & 1) && !(y & 1) && !(x & 1)) { /* a stored block centre: 0 or 255 exactly */
            centres++;
            uint8_t g = ref[((size_t)(z / 2) * N + y / 2) * N + x / 2];
            centre_bad += got != (g ? 255u : 0u);
          }
        }
    CHECK_EQ(bad, 0);
    CHECK_EQ(centre_bad, 0);
    CHECK_EQ(centres, N * N * N);

    /* every block decode matches the full decode */
    uint8_t blk[VOLCOMP_BLOCK_VOXELS];
    for (uint32_t b = 0; b < 512; b += 37) {
      uint32_t bz = b >> 6, by = (b >> 3) & 7, bx = b & 7;
      CHECK_EQ(volcomp_decode_block(enc, n, bz, by, bx, blk, sizeof blk), VOLCOMP_OK);
      unsigned d = 0;
      for (uint32_t z = 0; z < 16; z++)
        for (uint32_t y = 0; y < 16; y++)
          d += memcmp(blk + z * 256 + y * 16,
                      dec + ((size_t)(bz * 16 + z) * C + by * 16 + y) * C + bx * 16, 16) != 0;
      CHECK_EQ(d, 0);
    }
  }
}

static void test_const(uint8_t *src, uint8_t *enc, uint8_t *dec, uint8_t *grid) {
  for (int one = 0; one <= 1; one++) {
    memset(src, one ? 255 : 0, VOLCOMP_CHUNK_VOXELS);
    size_t n = 0;
    CHECK_EQ(volcomp_mask_encode(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK_EQ(n, VMK_HDR_BYTES + 1u); /* the one-byte const form */
    CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
    unsigned bad = 0;
    for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) bad += dec[i] != (one ? 255u : 0u);
    CHECK_EQ(bad, 0);
    CHECK_EQ(volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
    CHECK_EQ(grid[0], one ? 255 : 0);
    CHECK_EQ(grid[VOLCOMP_MASK_VOXELS - 1], one ? 255 : 0);
  }
  /* a source that is nonzero but pools to all zero is still const: single voxels */
  memset(src, 0, VOLCOMP_CHUNK_VOXELS);
  for (uint32_t z = 0; z < C; z += 2)
    for (uint32_t y = 0; y < C; y += 2)
      for (uint32_t x = 0; x < C; x += 2) src[((size_t)z * C + y) * C + x] = 255;
  size_t n = 0;
  CHECK_EQ(volcomp_mask_encode(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  CHECK_EQ(n, VMK_HDR_BYTES + 1u);
  CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
  CHECK_EQ(dec[0], 0);
}

static void test_reject(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  synth_mask(src, 7);
  size_t n = 0;
  CHECK_EQ(volcomp_mask_encode(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  uint8_t *copy = malloc(n);
  /* truncation at every length */
  for (size_t k = 0; k < n; k++) {
    memcpy(copy, enc, k);
    volcomp_status st = volcomp_decode(copy, k, dec, VOLCOMP_CHUNK_VOXELS);
    CHECK(st == VOLCOMP_ERR_CORRUPT || st == VOLCOMP_ERR_VERSION);
  }
  /* trailing garbage */
  memcpy(copy, enc, n);
  uint8_t *longer = malloc(n + 1);
  memcpy(longer, enc, n);
  longer[n] = 0x5A;
  CHECK_EQ(volcomp_decode(longer, n + 1, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  /* reserved flag bits */
  memcpy(copy, enc, n);
  copy[VF_HDR_BYTES] |= 0x80u;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  /* an unknown form */
  memcpy(copy, enc, n);
  copy[VF_HDR_BYTES] = 3u;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  /* a nonzero q_raw in a mask header */
  memcpy(copy, enc, n);
  copy[6] = 1;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  /* the range coder's priming byte must be zero */
  memcpy(copy, enc, n);
  copy[VMK_HDR_BYTES] = 1;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  /* flipped payload bytes never crash and never silently pass the length check */
  uint32_t r = 12345;
  for (int t = 0; t < 200; t++) {
    memcpy(copy, enc, n);
    size_t pos = VMK_HDR_BYTES + 1u + vt_rng(&r) % (n - VMK_HDR_BYTES - 1u);
    copy[pos] ^= (uint8_t)(1u << (vt_rng(&r) % 8));
    volcomp_status st = volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS);
    CHECK(st == VOLCOMP_OK || st == VOLCOMP_ERR_CORRUPT);
  }
  /* a short destination */
  CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS - 1), VOLCOMP_ERR_SHORT_BUF);
  /* mask_info on a stream that is not a mask chunk */
  size_t ln = 0;
  CHECK_EQ(volcomp_encode(src, VOLCOMP_Q_LOSSLESS, copy = realloc(copy, VOLCOMP_ENCODE_BOUND),
                          VOLCOMP_ENCODE_BOUND, &ln),
           VOLCOMP_OK);
  CHECK_EQ(volcomp_mask_info(copy, ln, NULL), VOLCOMP_ERR_ARG);
  free(copy);
  free(longer);
}

/* a short destination buffer falls back to the raw form, and that decodes */
static void test_raw(uint8_t *src, uint8_t *dec, uint8_t *grid, uint8_t *ref) {
  synth_mask(src, 11);
  uint8_t *small = malloc(VMK_HDR_BYTES + VOLCOMP_MASK_VOXELS / 8u);
  size_t n = 0;
  CHECK_EQ(volcomp_mask_encode(src, small, 32u, &n), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_mask_encode(src, small, VMK_HDR_BYTES + VOLCOMP_MASK_VOXELS / 8u, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(small, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
  ref_pool(src, ref);
  CHECK_EQ(volcomp_mask_decode_stored(small, n, grid, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
  unsigned bad = 0;
  for (size_t i = 0; i < VOLCOMP_MASK_VOXELS; i++) bad += grid[i] != (ref[i] ? 255u : 0u);
  CHECK_EQ(bad, 0);
  free(small);
}

/* ------------------------------------------------- mode 5: spatially lossless */

/* `src` round trips through mode 5 byte for byte (as 0/255), every block decode
 * agrees with the full decode, and the stream is never over the bound. */
static size_t ll_roundtrip(const uint8_t *src, uint8_t *enc, uint8_t *dec) {
  size_t n = 0;
  CHECK_EQ(volcomp_mask_encode_lossless(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  CHECK(n <= VOLCOMP_MASK_LL_BOUND);
  uint32_t dim = 0;
  CHECK_EQ(volcomp_mask_info(enc, n, &dim), VOLCOMP_OK);
  CHECK_EQ(dim, VOLCOMP_CHUNK_DIM); /* how a reader tells mode 5 from mode 4 */
  bool ll = true;
  CHECK_EQ(volcomp_is_lossless(enc, n, &ll), VOLCOMP_OK);
  CHECK_EQ(ll, 0); /* spatially exact, but 0/255 is not the u8 source */
  float q = -1;
  CHECK_EQ(volcomp_stream_q(enc, n, &q), VOLCOMP_OK);
  CHECK(q == 0.0f);

  CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
  unsigned bad = 0;
  for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) bad += dec[i] != (src[i] ? 255u : 0u);
  CHECK_EQ(bad, 0);

  /* the stored grid is the chunk itself */
  uint8_t *grid = malloc(VOLCOMP_CHUNK_VOXELS);
  CHECK_EQ(volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
  CHECK_EQ(memcmp(grid, dec, VOLCOMP_CHUNK_VOXELS), 0);
  CHECK_EQ(volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_CHUNK_VOXELS - 1), VOLCOMP_ERR_SHORT_BUF);
  free(grid);

  uint8_t blk[VOLCOMP_BLOCK_VOXELS];
  for (uint32_t b = 0; b < 512; b += 37) {
    uint32_t bz = b >> 6, by = (b >> 3) & 7, bx = b & 7;
    CHECK_EQ(volcomp_decode_block(enc, n, bz, by, bx, blk, sizeof blk), VOLCOMP_OK);
    unsigned d = 0;
    for (uint32_t z = 0; z < 16; z++)
      for (uint32_t y = 0; y < 16; y++)
        d += memcmp(blk + z * 256 + y * 16,
                    dec + ((size_t)(bz * 16 + z) * C + by * 16 + y) * C + bx * 16, 16) != 0;
    CHECK_EQ(d, 0);
  }
  return n;
}

static void test_ll_roundtrip(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  for (uint32_t seed = 1; seed <= 3; seed++) {
    synth_mask(src, seed);
    ll_roundtrip(src, enc, dec);
  }
  synth_sphere(src, 40.0);
  ll_roundtrip(src, enc, dec);
  /* a source that is nonzero but pools to nothing under mode 4 is kept exactly here */
  memset(src, 0, VOLCOMP_CHUNK_VOXELS);
  for (uint32_t z = 0; z < C; z += 2)
    for (uint32_t y = 0; y < C; y += 2)
      for (uint32_t x = 0; x < C; x += 2) src[((size_t)z * C + y) * C + x] = 1; /* any nonzero is 1 */
  ll_roundtrip(src, enc, dec);
  /* random bits: incompressible, so the encoder must choose RAW and stay at the bound */
  synth_random(src, 99);
  size_t n = ll_roundtrip(src, enc, dec);
  CHECK_EQ(n, VOLCOMP_MASK_LL_BOUND);
  uint32_t flags = 0;
  memcpy(&flags, enc + 8, 1);
  CHECK_EQ(flags, 2u); /* VMK_FORM_RAW */
}

static void test_ll_const(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  for (int one = 0; one <= 1; one++) {
    memset(src, one ? 255 : 0, VOLCOMP_CHUNK_VOXELS);
    size_t n = 0;
    CHECK_EQ(volcomp_mask_encode_lossless(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK_EQ(n, VMK_HDR_BYTES + 1u);
    CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
    unsigned bad = 0;
    for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) bad += dec[i] != (one ? 255u : 0u);
    CHECK_EQ(bad, 0);
  }
}

/* a destination that only fits the raw form, and one that fits nothing */
static void test_ll_raw(uint8_t *src, uint8_t *dec) {
  synth_random(src, 7);
  uint8_t *small = malloc(VOLCOMP_MASK_LL_BOUND);
  size_t n = 0;
  CHECK_EQ(volcomp_mask_encode_lossless(src, small, 32u, &n), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_mask_encode_lossless(src, small, VOLCOMP_MASK_LL_BOUND - 1u, &n), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_mask_encode_lossless(src, small, VOLCOMP_MASK_LL_BOUND, &n), VOLCOMP_OK);
  CHECK_EQ(n, VOLCOMP_MASK_LL_BOUND);
  CHECK_EQ(volcomp_decode(small, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
  unsigned bad = 0;
  for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) bad += dec[i] != (src[i] ? 255u : 0u);
  CHECK_EQ(bad, 0);
  free(small);
}

static void test_ll_reject(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  /* a sphere, not a noisy blob: a truncated stream decodes garbage, which
   * renormalises on nearly every bit and is several times slower than a real
   * one, so the sweep below wants the smallest representative stream */
  synth_sphere(src, 50.0);
  size_t n = 0;
  CHECK_EQ(volcomp_mask_encode_lossless(src, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  uint8_t *copy = malloc(n + 1);
  /* truncation: every length up to the header, then a sample (a 128^3 decode is
   * ~8 ms, so every one of the thousands of lengths would take minutes) */
  for (size_t k = 0; k < n; k += k < VMK_HDR_BYTES + 8u ? 1u : 61u) {
    memcpy(copy, enc, k);
    volcomp_status st = volcomp_decode(copy, k, dec, VOLCOMP_CHUNK_VOXELS);
    CHECK(st == VOLCOMP_ERR_CORRUPT || st == VOLCOMP_ERR_VERSION);
  }
  CHECK_EQ(volcomp_decode(enc, n - 1, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  memcpy(copy, enc, n); /* trailing garbage */
  copy[n] = 0x5A;
  CHECK_EQ(volcomp_decode(copy, n + 1, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  memcpy(copy, enc, n); /* reserved flag bits */
  copy[VF_HDR_BYTES] |= 0x40u;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  memcpy(copy, enc, n); /* an unknown form */
  copy[VF_HDR_BYTES] = 3u;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  memcpy(copy, enc, n); /* a nonzero q_raw */
  copy[6] = 1;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  memcpy(copy, enc, n); /* an unknown mode is still refused */
  copy[5] = 6;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  memcpy(copy, enc, n); /* the range coder's priming byte must be zero */
  copy[VMK_HDR_BYTES] = 1;
  CHECK_EQ(volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_ERR_CORRUPT);
  uint32_t r = 777;
  for (int t = 0; t < 40; t++) { /* flipped payload bits never crash */
    memcpy(copy, enc, n);
    size_t pos = VMK_HDR_BYTES + 1u + vt_rng(&r) % (n - VMK_HDR_BYTES - 1u);
    copy[pos] ^= (uint8_t)(1u << (vt_rng(&r) % 8));
    volcomp_status st = volcomp_decode(copy, n, dec, VOLCOMP_CHUNK_VOXELS);
    CHECK(st == VOLCOMP_OK || st == VOLCOMP_ERR_CORRUPT);
  }
  CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS - 1), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_mask_encode_lossless(NULL, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
  free(copy);
}

int main(void) {
  uint8_t *src = malloc(VOLCOMP_CHUNK_VOXELS), *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
  uint8_t *grid = malloc(VOLCOMP_MASK_VOXELS), *ref = malloc(VOLCOMP_MASK_VOXELS);
  test_roundtrip(src, enc, dec, grid, ref);
  test_const(src, enc, dec, grid);
  test_reject(src, enc, dec);
  test_raw(src, dec, grid, ref);
  test_ll_roundtrip(src, enc, dec);
  test_ll_const(src, enc, dec);
  test_ll_raw(src, dec);
  test_ll_reject(src, enc, dec);

  /* ---- the real recto fixture: size and speed ---- */
  uint32_t nch = 0;
  uint8_t *chunks = fixture_chunks("tests/data/recto256.vlz", &nch);
  CHECK(chunks != NULL);
  if (chunks) {
    uint64_t bytes = 0, fg = 0;
    double t_enc = 0, t_dec = 0;
    size_t sizes[8];
    for (uint32_t i = 0; i < nch; i++) {
      const uint8_t *c = chunks + (size_t)i * VOLCOMP_CHUNK_VOXELS;
      for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS; k++) fg += c[k] != 0;
      size_t n = 0;
      double t0 = now();
      CHECK_EQ(volcomp_mask_encode(c, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
      t_enc += now() - t0;
      sizes[i & 7] = n;
      bytes += n;
      t0 = now();
      CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
      t_dec += now() - t0;
      /* the decode's block centres are the stored grid, exactly */
      CHECK_EQ(volcomp_mask_decode_stored(enc, n, grid, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
      unsigned bad = 0;
      for (uint32_t z = 0; z < N; z++)
        for (uint32_t y = 0; y < N; y++)
          for (uint32_t x = 0; x < N; x++)
            bad += dec[((size_t)(2 * z) * C + 2 * y) * C + 2 * x] != grid[((size_t)z * N + y) * N + x];
      CHECK_EQ(bad, 0);
    }
    double vox = (double)nch * VOLCOMP_CHUNK_VOXELS;
    double bpv = 8.0 * (double)bytes / vox;
    printf("recto fixture: %u chunks, %.1f %% foreground, %llu bytes, %.5f bits/voxel "
           "(encode %.1f Mvox/s, decode %.1f Mvox/s)\n",
           nch, 100.0 * (double)fg / vox, (unsigned long long)bytes, bpv, vox / t_enc / 1e6,
           vox / t_dec / 1e6);
    printf("  per chunk:");
    for (uint32_t i = 0; i < nch && i < 8; i++) printf(" %zu", sizes[i]);
    printf("\n");
    CHECK(bpv <= 0.007);

    /* the same fixture at mode 5: no downscale, every voxel exact */
    uint64_t llbytes = 0;
    double t_lle = 0, t_lld = 0;
    for (uint32_t i = 0; i < nch; i++) {
      const uint8_t *c = chunks + (size_t)i * VOLCOMP_CHUNK_VOXELS;
      size_t n = 0;
      double t0 = now();
      CHECK_EQ(volcomp_mask_encode_lossless(c, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
      t_lle += now() - t0;
      sizes[i & 7] = n;
      llbytes += n;
      t0 = now();
      CHECK_EQ(volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
      t_lld += now() - t0;
      unsigned bad = 0;
      for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS; k++) bad += dec[k] != (c[k] ? 255u : 0u);
      CHECK_EQ(bad, 0);
    }
    double llbpv = 8.0 * (double)llbytes / vox;
    printf("recto fixture, mask-lossless: %llu bytes, %.5f bits/voxel "
           "(encode %.1f Mvox/s, decode %.1f Mvox/s)\n",
           (unsigned long long)llbytes, llbpv, vox / t_lle / 1e6, vox / t_lld / 1e6);
    printf("  per chunk:");
    for (uint32_t i = 0; i < nch && i < 8; i++) printf(" %zu", sizes[i]);
    printf("\n");
    CHECK(llbpv <= 0.03);
    free(chunks);
  }
  free(src);
  free(dec);
  free(enc);
  free(grid);
  free(ref);
  TEST_END();
}
