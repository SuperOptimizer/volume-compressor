/* Surface chunks (mode 6, spec/format.md §13): a DCT base at step q plus a
 * refinement layer that makes the threshold exact. Round trips and the exact
 * threshold on synthetic probability fields at every q and several thresholds,
 * the edge cases (empty, constant, binary, noise), stream validation, a golden
 * stream whose decode must be bit-identical on every build (this test also runs
 * as test_surfchunk_c with the C kernels), and size / quality / speed on real
 * PHercParis4 recto and m7 teacher chunks (tests/data/surface_teachers.vlz). */
#include "../volcomp.h"
#include "check.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define C VOLCOMP_CHUNK_DIM
#define NV VOLCOMP_CHUNK_VOXELS
#define BOUND VOLCOMP_SURFACE_ENCODE_BOUND

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static uint64_t fnv1a(const uint8_t *p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* A surface-prediction-like field: sheets (a wavy signed distance), a logistic
 * profile of width w across them, and a little noise. */
static void synth_prob(uint8_t *v, uint32_t seed, double w, double noise) {
  uint32_t r = seed | 1u;
  for (uint32_t z = 0; z < C; z++)
    for (uint32_t y = 0; y < C; y++)
      for (uint32_t x = 0; x < C; x++) {
        double s = 9.0 * sin(0.045 * x + 0.02 * z + seed) + 0.33 * y - 20.0;
        s = fmod(s + 400.0, 22.0) - 11.0; /* repeating sheets every 22 voxels */
        double p = 255.0 / (1.0 + exp(-(3.5 - fabs(s)) / w));
        p += noise * ((double)(vt_rng(&r) % 1001) / 500.0 - 1.0);
        int iv = (int)floor(p + 0.5);
        v[((size_t)z * C + y) * C + x] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
      }
}

static unsigned wrong_side(const uint8_t *a, const uint8_t *b, uint32_t thr) {
  unsigned n = 0;
  for (size_t i = 0; i < NV; i++) n += (a[i] >= thr) != (b[i] >= thr);
  return n;
}

/* encode, check the threshold is exact, decode_block agrees with decode, and the
 * info calls report the right things; returns the stream length */
static size_t roundtrip(const uint8_t *src, float q, uint32_t thr, uint8_t *enc, uint8_t *dec) {
  size_t n = 0;
  CHECK_EQ(volcomp_surface_encode(src, q, thr, enc, BOUND, &n), VOLCOMP_OK);
  CHECK(n <= BOUND);
  CHECK_EQ(volcomp_decode(enc, n, dec, NV), VOLCOMP_OK);
  CHECK_EQ(wrong_side(src, dec, thr), 0);
  float sq = -1.0f;
  bool ll = true;
  uint32_t t = 0, m = 999;
  CHECK_EQ(volcomp_stream_q(enc, n, &sq), VOLCOMP_OK);
  CHECK(fabsf(sq - q) <= 1.0f / 512.0f);
  CHECK_EQ(volcomp_is_lossless(enc, n, &ll), VOLCOMP_OK);
  CHECK(!ll);
  CHECK_EQ(volcomp_surface_info(enc, n, &t, &m), VOLCOMP_OK);
  CHECK_EQ(t, thr);
  CHECK(m == 0 || (m & 1u));
  uint32_t dim;
  CHECK_EQ(volcomp_mask_info(enc, n, &dim), VOLCOMP_ERR_ARG);
  uint8_t blk[VOLCOMP_BLOCK_VOXELS];
  const uint32_t probe[3][3] = {{0, 0, 0}, {3, 5, 2}, {7, 7, 7}};
  for (int k = 0; k < 3; k++) {
    CHECK_EQ(volcomp_decode_block(enc, n, probe[k][0], probe[k][1], probe[k][2], blk, sizeof blk), VOLCOMP_OK);
    unsigned bad = 0;
    for (uint32_t z = 0; z < 16; z++)
      for (uint32_t y = 0; y < 16; y++)
        bad += memcmp(blk + (z * 16 + y) * 16,
                      dec + ((size_t)(probe[k][0] * 16 + z) * C + probe[k][1] * 16 + y) * C + probe[k][2] * 16,
                      16) != 0;
    CHECK_EQ(bad, 0);
  }
  return n;
}

static void test_synthetic(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  const float qs[] = {2.0f, 4.0f, 16.0f, 32.0f, 64.0f, 128.0f, 255.0f};
  for (uint32_t seed = 1; seed <= 2; seed++) {
    synth_prob(src, seed, seed == 1 ? 0.8 : 2.0, seed == 1 ? 0.0 : 3.0);
    for (size_t k = 0; k < sizeof qs / sizeof qs[0]; k++) {
      size_t n = roundtrip(src, qs[k], 128, enc, dec);
      /* the soft values carry the base's error: bounded like the DCT's at that q */
      double mae = 0;
      for (size_t i = 0; i < NV; i++) mae += fabs((double)dec[i] - (double)src[i]);
      mae /= NV;
      printf("synthetic seed %u q%-3.0f %6zu bytes, MAE %.2f\n", seed, (double)qs[k], n, mae);
      CHECK(mae < 0.03 * (double)qs[k] + 2.0);
    }
    const uint32_t thrs[] = {1, 64, 200, 255};
    for (size_t k = 0; k < 4; k++) roundtrip(src, 48.0f, thrs[k], enc, dec);
  }
}

static void test_edges(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  size_t n;
  uint32_t t, m;
  /* empty: the base is exact, no refinement */
  memset(src, 0, NV);
  n = roundtrip(src, 16.0f, 128, enc, dec);
  CHECK_EQ(volcomp_surface_info(enc, n, &t, &m), VOLCOMP_OK);
  CHECK_EQ(m, 0);
  size_t nonzero = 0;
  for (size_t i = 0; i < NV; i++) nonzero += dec[i] != 0;
  CHECK_EQ(nonzero, 0);
  /* constants on either side of, and at, the threshold */
  const uint8_t vals[] = {127, 128, 255, 3};
  for (int k = 0; k < 4; k++) {
    memset(src, vals[k], NV);
    roundtrip(src, 32.0f, 128, enc, dec);
  }
  /* a binary 0/255 mask: the special case the mask modes store; here exact too */
  uint32_t r = 99;
  for (size_t i = 0; i < NV; i++) src[i] = 0;
  for (int b = 0; b < 40; b++) {
    uint32_t cz = vt_rng(&r) % C, cy = vt_rng(&r) % C, cx = vt_rng(&r) % C, rad = 3 + vt_rng(&r) % 12;
    for (uint32_t z = 0; z < C; z++)
      for (uint32_t y = 0; y < C; y++)
        for (uint32_t x = 0; x < C; x++) {
          int dz = (int)z - (int)cz, dy = (int)y - (int)cy, dx = (int)x - (int)cx;
          if ((uint32_t)(dz * dz + dy * dy + dx * dx) < rad * rad) src[((size_t)z * C + y) * C + x] = 255;
        }
  }
  roundtrip(src, 16.0f, 128, enc, dec);
  /* noise: the worst case for the refinement (a margin near its maximum) */
  for (size_t i = 0; i < NV; i++) src[i] = (uint8_t)vt_rng(&r);
  n = roundtrip(src, 64.0f, 128, enc, dec);
  CHECK_EQ(volcomp_surface_info(enc, n, &t, &m), VOLCOMP_OK);
  CHECK(m > 100);
  printf("noise chunk at q64: %zu bytes (margin %u)\n", n, m);
  /* arguments */
  CHECK_EQ(volcomp_surface_encode(src, 0.0f, 128, enc, BOUND, &n), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_surface_encode(src, 1.5f, 128, enc, BOUND, &n), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_surface_encode(src, 300.0f, 128, enc, BOUND, &n), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_surface_encode(src, 8.0f, 0, enc, BOUND, &n), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_surface_encode(src, 8.0f, 256, enc, BOUND, &n), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_surface_encode(NULL, 8.0f, 128, enc, BOUND, &n), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_surface_encode(src, 8.0f, 128, enc, 100, &n), VOLCOMP_ERR_SHORT_BUF);
  /* other modes are not surface chunks */
  CHECK_EQ(volcomp_encode(src, 8.0f, enc, BOUND, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_surface_info(enc, n, &t, &m), VOLCOMP_ERR_ARG);
}

static void test_reject(uint8_t *src, uint8_t *enc, uint8_t *dec) {
  synth_prob(src, 3, 1.0, 2.0);
  size_t n = 0;
  CHECK_EQ(volcomp_surface_encode(src, 24.0f, 128, enc, BOUND, &n), VOLCOMP_OK);
  uint32_t t, m;
  CHECK_EQ(volcomp_surface_info(enc, n, &t, &m), VOLCOMP_OK);
  CHECK(m > 0); /* the cases below need a refinement payload */
  uint8_t *bad = malloc(n + 16);
  /* every truncation is rejected */
  unsigned accepted = 0;
  for (size_t k = 0; k < n; k += 1 + k / 64) accepted += volcomp_decode(enc, k, dec, NV) == VOLCOMP_OK;
  CHECK_EQ(accepted, 0);
  /* trailing bytes are rejected (the refinement must be consumed exactly) */
  memcpy(bad, enc, n);
  bad[n] = 0;
  CHECK_EQ(volcomp_decode(bad, n + 1, dec, NV), VOLCOMP_ERR_CORRUPT);
  /* header fields */
  memcpy(bad, enc, n);
  bad[8] = 0; /* thr 0 */
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  memcpy(bad, enc, n);
  bad[9] = 0; /* a step law other than flat */
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  memcpy(bad, enc, n);
  bad[10] = (uint8_t)(bad[10] + 1u); /* an even margin */
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  memcpy(bad, enc, n);
  bad[10] = 0xFF, bad[11] = 0x01; /* margin 511 > 509 */
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  memcpy(bad, enc, n);
  bad[15] = 0x7F; /* base longer than the stream */
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  memcpy(bad, enc, n);
  bad[6] = 0, bad[7] = 1; /* q_raw 256: q 1 is below the flat law's minimum */
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  /* a margin of 0 with a refinement payload present */
  memcpy(bad, enc, n);
  bad[10] = 0, bad[11] = 0;
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  /* random byte flips never crash (the result may or may not decode) */
  uint32_t r = 5;
  for (int it = 0; it < 300; it++) {
    memcpy(bad, enc, n);
    for (int f = 0; f < 3; f++) bad[vt_rng(&r) % n] ^= (uint8_t)(1u << (vt_rng(&r) & 7u));
    (void)volcomp_decode(bad, n, dec, NV);
  }
  /* a revision-3 reader would stop at the mode byte: mode 7 is still unknown */
  memcpy(bad, enc, n);
  bad[5] = 7;
  CHECK_EQ(volcomp_decode(bad, n, dec, NV), VOLCOMP_ERR_CORRUPT);
  free(bad);
}

/* tests/golden/surf_q16.volc was written by volcomp_surface_encode(synth_prob(seed 1,
 * w 0.8, noise 0), q 16, thr 128). Its decode must hash the same on every build:
 * this is what makes the refinement's contexts safe to read from the base. */
#define GOLDEN_SURF_HASH 0x7ebfbc08cd801e8eull
static void test_golden(uint8_t *dec) {
  FILE *f = fopen("tests/golden/surf_q16.volc", "rb");
  CHECK(f != NULL);
  if (!f) return;
  uint8_t *buf = malloc(1u << 20);
  size_t n = fread(buf, 1, 1u << 20, f);
  fclose(f);
  CHECK_EQ(volcomp_decode(buf, n, dec, NV), VOLCOMP_OK);
  uint64_t h = fnv1a(dec, NV);
  printf("golden surf_q16: %zu bytes, decode hash 0x%016llx (kernels %s)\n", n, (unsigned long long)h,
         volcomp_kernels());
  CHECK(h == GOLDEN_SURF_HASH);
  free(buf);
}

/* ---- the teacher fixture: u32 count, count u32 lengths, then the streams:
 * 128^3 chunks of the PHercParis4 surface teachers' FLOAT predictions rounded to
 * u8 (recto 3D U-Net, then m7 nnU-Net at 9.6 um upsampled 4x), each stored as a
 * volcomp q-1 DCT chunk; the decoded q-1 chunks are the sources here. */
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
  uint8_t *out = malloc((size_t)n * NV);
  size_t off = 4u + 4u * n;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t len = rd32(buf + 4 + 4 * i);
    if (volcomp_decode(buf + off, len, out + (size_t)i * NV, NV) != VOLCOMP_OK) {
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

static void test_teachers(uint8_t *enc, uint8_t *dec) {
  uint32_t nch = 0;
  uint8_t *chunks = fixture_chunks("tests/data/surface_teachers.vlz", &nch);
  CHECK(chunks != NULL);
  if (!chunks) return;
  static const char *name[] = {"recto", "m7"};
  for (uint32_t i = 0; i < nch && i < 2; i++) {
    const uint8_t *src = chunks + (size_t)i * NV;
    size_t n8 = 0;
    double t0 = now();
    CHECK_EQ(volcomp_encode(src, 8.0f, enc, BOUND, &n8), VOLCOMP_OK);
    double te8 = now() - t0;
    t0 = now();
    CHECK_EQ(volcomp_decode(enc, n8, dec, NV), VOLCOMP_OK);
    double td8 = now() - t0;
    unsigned flips8 = wrong_side(src, dec, 128);
    const float qs[] = {32.0f, 64.0f, 96.0f};
    for (int k = 0; k < 3; k++) {
      size_t n = 0;
      t0 = now();
      CHECK_EQ(volcomp_surface_encode(src, qs[k], 128, enc, BOUND, &n), VOLCOMP_OK);
      double te = now() - t0;
      t0 = now();
      CHECK_EQ(volcomp_decode(enc, n, dec, NV), VOLCOMP_OK);
      double td = now() - t0;
      CHECK_EQ(wrong_side(src, dec, 128), 0);
      /* error in the band: voxels whose source is within 64 of the threshold */
      double e = 0;
      size_t nb = 0;
      for (size_t j = 0; j < NV; j++)
        if (src[j] >= 64 && src[j] < 192) e += fabs((double)dec[j] - (double)src[j]), nb++;
      double mae_band = nb ? e / (double)nb : 0.0;
      printf("%-5s q%-2.0f surface %7zu B (%.3fx q8 = %zu B, whose threshold flips %u voxels), band MAE %.2f, "
             "enc %.0f / dec %.0f MB/s (q8: %.0f / %.0f)\n",
             name[i], (double)qs[k], n, (double)n / (double)n8, n8, flips8, mae_band, 2.097152 / te,
             2.097152 / td, 2.097152 / te8, 2.097152 / td8);
      /* quality and size gates (measured values in docs/surface_mode.md) */
      if (qs[k] == 32.0f) CHECK(mae_band < 3.6);
      if (qs[k] == 64.0f) {
        CHECK(mae_band < 6.0);
        CHECK((double)n < 0.85 * (double)n8);
      }
      if (qs[k] == 96.0f) {
        CHECK(mae_band < 8.0);
        CHECK((double)n < 0.75 * (double)n8);
      }
    }
  }
  free(chunks);
}

int main(void) {
  uint8_t *src = malloc(NV), *dec = malloc(NV), *enc = malloc(BOUND);
  printf("kernels: %s\n", volcomp_kernels());
  test_synthetic(src, enc, dec);
  test_edges(src, enc, dec);
  test_reject(src, enc, dec);
  test_golden(dec);
  test_teachers(enc, dec);
  free(src);
  free(dec);
  free(enc);
  TEST_END();
}
