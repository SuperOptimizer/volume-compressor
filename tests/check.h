/* Always-active test assertions (never assert(): NDEBUG must not disable them). */
#ifndef VOLCOMP_TEST_CHECK_H
#define VOLCOMP_TEST_CHECK_H
#include <stdio.h>
#include <stdlib.h>

static int vt_failures = 0;
#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond);                      \
      vt_failures++;                                                                               \
    }                                                                                              \
  } while (0)
#define CHECK_EQ(a, b)                                                                             \
  do {                                                                                             \
    long long va_ = (long long)(a), vb_ = (long long)(b);                                          \
    if (va_ != vb_) {                                                                              \
      fprintf(stderr, "CHECK_EQ failed %s:%d: %s=%lld %s=%lld\n", __FILE__, __LINE__, #a, va_, #b,   \
              vb_);                                                                                \
      vt_failures++;                                                                               \
    }                                                                                              \
  } while (0)
#define TEST_END()                                                                                 \
  do {                                                                                             \
    if (vt_failures) fprintf(stderr, "%d failure(s)\n", vt_failures);                              \
    return vt_failures ? 1 : 0;                                                                    \
  } while (0)

static inline uint32_t vt_rng(uint32_t *s) {
  *s ^= *s << 13;
  *s ^= *s >> 17;
  *s ^= *s << 5;
  return *s;
}
/* deterministic "CT-like" synthetic chunk: smooth field + texture + noise */
static inline void vt_synth_chunk(uint8_t *v, uint32_t seed) {
  uint32_t r = seed | 1u;
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        double s = 120 + 50 * sin(z * 0.05 + seed) * cos(y * 0.07) + 30 * sin((x + y) * 0.11);
        s += 12 * sin(x * 0.9 + z * 0.3) + (double)(vt_rng(&r) % 9) - 4.0;
        int iv = (int)(s + 0.5);
        v[((size_t)z * 128 + y) * 128 + x] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
      }
}
#endif
