/* Optional decode-side deblocking (not part of the format): volcomp_deblock_ex
 * with the zero guard, and volcomp_decode_smooth (smooth + project onto the
 * quantisation cells). The masked-edge regression: masked CT stores air as exact
 * 0, and a mask edge that falls on a block or chunk face is a real edge — no filter
 * may blur it, move air off 0, or make the volume worse than no filter. */
#include "../volcomp.h"
#include "check.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define C VOLCOMP_CHUNK_DIM
#define NV VOLCOMP_CHUNK_VOXELS

static double mse(const uint8_t *a, const uint8_t *b, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; i++) s += ((double)a[i] - b[i]) * ((double)a[i] - b[i]);
  return s / (double)n;
}

/* two chunks along z: CT-like texture where the mask is on, exact 0 elsewhere.
 * The mask edge lies on the chunk face (z = 128), on a block face inside the
 * second chunk (x = 64) and on an oblique plane. */
static void masked_pair(uint8_t *v) {
  uint8_t *tmp = malloc(NV);
  for (int k = 0; k < 2; k++) {
    vt_synth_chunk(tmp, 11u + (uint32_t)k);
    memcpy(v + (size_t)k * NV, tmp, NV);
  }
  free(tmp);
  for (uint32_t z = 0; z < 2 * C; z++)
    for (uint32_t y = 0; y < C; y++)
      for (uint32_t x = 0; x < C; x++) {
        size_t i = ((size_t)z * C + y) * C + x;
        bool air = (z >= C && x >= 64) || (z < C && x + y > 200);
        if (air) v[i] = 0;
        else if (v[i] == 0) v[i] = 1; /* tissue is never exactly 0 */
      }
}

static void test_masked_edges(void) {
  uint8_t *src = malloc(2 * NV), *dec = malloc(2 * NV), *flt = malloc(2 * NV), *sm = malloc(2 * NV);
  uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
  masked_pair(src);
  const float qs[] = {2.0f, 4.0f, 8.0f, 16.0f, 32.0f};
  for (size_t k = 0; k < sizeof qs / sizeof qs[0]; k++) {
    for (int c = 0; c < 2; c++) {
      size_t n;
      CHECK_EQ(volcomp_encode(src + (size_t)c * NV, qs[k], enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
      CHECK_EQ(volcomp_decode(enc, n, dec + (size_t)c * NV, NV), VOLCOMP_OK);
      CHECK_EQ(volcomp_decode_smooth(enc, n, sm + (size_t)c * NV, NV, 2.0f,
                                     VOLCOMP_SMOOTH_GATED | VOLCOMP_DEBLOCK_ZERO_GUARD),
               VOLCOMP_OK);
    }
    /* the zero guard: every exact zero of the decode stays 0, in both filters */
    memcpy(flt, dec, 2 * NV);
    volcomp_deblock_ex(flt, 2 * C, C, C, qs[k], VOLCOMP_DEBLOCK_ZERO_GUARD);
    size_t moved = 0, moved_sm = 0;
    for (size_t i = 0; i < 2 * NV; i++) {
      moved += dec[i] == 0 && flt[i] != 0;
      moved_sm += dec[i] == 0 && sm[i] != 0;
    }
    CHECK_EQ(moved, 0);
    CHECK_EQ(moved_sm, 0);
    /* never worse than no filter, and the edge voxels in particular */
    double e0 = mse(src, dec, 2 * NV), e1 = mse(src, flt, 2 * NV), e2 = mse(src, sm, 2 * NV);
    printf("masked pair q%-2.0f: MSE none %.3f, deblock zg %.3f, smooth gated zg %.3f\n", (double)qs[k], e0, e1, e2);
    /* this texture is white noise on a smooth field, the worst case for any smoothing:
     * allow 2 %; the real-CT numbers are in docs/deblocking.md */
    CHECK(e1 <= e0 * 1.02);
    CHECK(e2 <= e0 * 1.02);
    /* the tissue next to the chunk-face mask edge (z = 124..127, x >= 64): not made worse */
    double b0 = 0, b1 = 0, b2 = 0;
    for (uint32_t z = C - 4; z < C; z++)
      for (uint32_t y = 0; y < C; y++)
        for (uint32_t x = 64; x < C; x++) {
          size_t i = ((size_t)z * C + y) * C + x;
          double s0 = src[i];
          b0 += (dec[i] - s0) * (dec[i] - s0), b1 += (flt[i] - s0) * (flt[i] - s0), b2 += (sm[i] - s0) * (sm[i] - s0);
        }
    CHECK(b1 <= b0 * 1.02);
    CHECK(b2 <= b0 * 1.02);
  }
  free(src), free(dec), free(flt), free(sm), free(enc);
}

/* decode_smooth: lossless / mask chunks decode as volcomp_decode; a surface chunk
 * keeps its exact threshold; strength 0 is the plain decode */
static void test_smooth_modes(void) {
  uint8_t *src = malloc(NV), *a = malloc(NV), *b = malloc(NV), *enc = malloc(VOLCOMP_SURFACE_ENCODE_BOUND);
  vt_synth_chunk(src, 5);
  size_t n;
  CHECK_EQ(volcomp_encode(src, 0.0f, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode_smooth(enc, n, a, NV, 2.0f, VOLCOMP_SMOOTH_GATED), VOLCOMP_OK);
  CHECK(memcmp(a, src, NV) == 0);
  CHECK_EQ(volcomp_encode(src, 8.0f, enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode_smooth(enc, n, a, NV, 0.0f, 0), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(enc, n, b, NV), VOLCOMP_OK);
  CHECK(memcmp(a, b, NV) == 0);
  CHECK_EQ(volcomp_decode_smooth(enc, n, a, NV, 2.0f, VOLCOMP_SMOOTH_GATED), VOLCOMP_OK);
  CHECK(mse(src, a, NV) <= mse(src, b, NV) * 1.001);
  /* a smooth probability-like field as a surface chunk */
  for (uint32_t z = 0; z < C; z++)
    for (uint32_t y = 0; y < C; y++)
      for (uint32_t x = 0; x < C; x++) {
        double s = 9.0 * sin(0.045 * x + 0.02 * z) + 0.33 * y - 20.0;
        s = fmod(s + 400.0, 22.0) - 11.0;
        src[((size_t)z * C + y) * C + x] = (uint8_t)floor(255.0 / (1.0 + exp(-(3.5 - fabs(s)) / 1.2)) + 0.5);
      }
  CHECK_EQ(volcomp_surface_encode(src, 64.0f, 128, enc, VOLCOMP_SURFACE_ENCODE_BOUND, &n), VOLCOMP_OK);
  const unsigned fl[] = {0u, VOLCOMP_SMOOTH_GATED | VOLCOMP_DEBLOCK_ZERO_GUARD};
  for (int k = 0; k < 2; k++) {
    CHECK_EQ(volcomp_decode_smooth(enc, n, a, NV, k ? 2.0f : 0.8f, fl[k]), VOLCOMP_OK);
    size_t wrong = 0;
    for (size_t i = 0; i < NV; i++) wrong += (src[i] >= 128) != (a[i] >= 128);
    CHECK_EQ(wrong, 0);
  }
  CHECK_EQ(volcomp_decode_smooth(enc, 5, a, NV, 1.0f, 0), VOLCOMP_ERR_CORRUPT);
  CHECK_EQ(volcomp_decode_smooth(enc, n, a, NV - 1, 1.0f, 0), VOLCOMP_ERR_SHORT_BUF);
  free(src), free(a), free(b), free(enc);
}

int main(void) {
  printf("kernels: %s\n", volcomp_kernels());
  test_masked_edges();
  test_smooth_modes();
  TEST_END();
}
