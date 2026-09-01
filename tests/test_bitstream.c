#include "../volcomp.h"
#include "check.h"

/* bit writer/reader: every width 1..32 x every starting alignment x random payloads */
static void bits_roundtrip(void) {
  uint8_t buf[4096];
  uint32_t seed = 7;
  for (uint32_t pre = 0; pre < 8; pre++)
    for (uint32_t n = 1; n <= 32; n++) {
      vf_bitw w;
      vf_bw_init(&w, buf, sizeof buf);
      uint32_t vals[64];
      if (pre) CHECK(vf_bw_put(&w, 0x55u & ((1u << pre) - 1u), pre));
      for (int i = 0; i < 64; i++) {
        vals[i] = vt_rng(&seed) & (n == 32u ? 0xffffffffu : ((1u << n) - 1u));
        CHECK(vf_bw_put(&w, vals[i], n));
      }
      size_t bn;
      CHECK(vf_bw_flush(&w, &bn));
      vf_bitr r;
      vf_br_init(&r, buf, bn);
      uint32_t v;
      if (pre) {
        CHECK(vf_br_get(&r, pre, &v));
        CHECK_EQ(v, 0x55u & ((1u << pre) - 1u));
      }
      for (int i = 0; i < 64; i++) {
        CHECK(vf_br_get(&r, n, &v));
        CHECK_EQ(v, vals[i]);
      }
      CHECK(vf_br_finished(&r));
    }
  /* non-zero padding is detected */
  vf_bitw w;
  vf_bw_init(&w, buf, sizeof buf);
  CHECK(vf_bw_put(&w, 1, 1));
  size_t bn;
  CHECK(vf_bw_flush(&w, &bn));
  buf[0] |= 0x80u;
  vf_bitr r;
  vf_br_init(&r, buf, bn);
  uint32_t v;
  CHECK(vf_br_get(&r, 1, &v));
  CHECK(!vf_br_finished(&r));
  /* writer overflow is reported */
  vf_bw_init(&w, buf, 1);
  CHECK(vf_bw_put(&w, 0xff, 8));
  CHECK(!vf_bw_put(&w, 0xff, 8));
}

static void hyb_roundtrip(void) {
  uint8_t buf[1 << 16];
  vf_bitw w;
  vf_bw_init(&w, buf, sizeof buf);
  uint32_t toks[2048], vals[2048];
  size_t k = 0;
  for (uint32_t u = 0; u < 300; u++) vals[k++] = u; /* all small classes densely */
  for (uint32_t b = 8; b < 18; b++) { /* class boundaries up to k=17 */
    vals[k++] = (1u << b) - 1u;
    vals[k++] = 1u << b;
    vals[k++] = (1u << b) + 1u;
  }
  vals[k++] = 262144u; /* max DC zigzag (2*131072) */
  for (size_t i = 0; i < k; i++) CHECK(vf_hyb_emit(&w, vals[i], &toks[i]));
  size_t bn;
  CHECK(vf_bw_flush(&w, &bn));
  vf_bitr r;
  vf_br_init(&r, buf, bn);
  for (size_t i = 0; i < k; i++) {
    uint32_t u;
    CHECK(vf_hyb_read(&r, toks[i], &u));
    CHECK_EQ(u, vals[i]);
    CHECK(toks[i] <= 20u);
  }
  CHECK(vf_br_finished(&r));
  /* token classes match the spec caps */
  uint32_t t;
  vf_bw_init(&w, buf, sizeof buf);
  CHECK(vf_hyb_emit(&w, 5219u, &t));
  CHECK_EQ(t, VF_TOKMAX_LVL);
  CHECK(vf_hyb_emit(&w, 4094u, &t));
  CHECK_EQ(t, VF_TOKMAX_RUN);
  CHECK(vf_hyb_emit(&w, vf_zigzag(131072), &t)); /* largest legal DC delta */
  CHECK_EQ(t, VF_TOKMAX_DC);
  CHECK(vf_hyb_emit(&w, vf_zigzag(-131072), &t));
  CHECK_EQ(t, VF_TOKMAX_DC - 1u);
}

static void leb_zigzag(void) {
  uint8_t buf[8];
  for (uint32_t v = 0; v < 70000; v += 7) {
    size_t n = vf_leb_put(buf, v);
    vf_cur c = {buf, buf + n};
    uint32_t o;
    CHECK(vf_leb_get(&c, &o));
    CHECK_EQ(o, v);
    CHECK(c.p == c.end);
  }
  /* non-canonical: 0x80 0x00 encodes 0 with a trailing zero byte */
  uint8_t bad1[] = {0x80, 0x00};
  vf_cur c = {bad1, bad1 + 2};
  uint32_t o;
  CHECK(!vf_leb_get(&c, &o));
  uint8_t bad2[] = {0xff, 0xff, 0xff, 0xff, 0x7f}; /* > 32 bits */
  c = (vf_cur){bad2, bad2 + 5};
  CHECK(!vf_leb_get(&c, &o));
  uint8_t bad3[] = {0x80}; /* truncated */
  c = (vf_cur){bad3, bad3 + 1};
  CHECK(!vf_leb_get(&c, &o));
  for (int32_t v = -200000; v <= 200000; v += 997) CHECK_EQ(vf_unzigzag(vf_zigzag(v)), v);
  CHECK_EQ(vf_zigzag(0), 0);
  CHECK_EQ(vf_zigzag(-1), 1);
  CHECK_EQ(vf_zigzag(1), 2);
}

int main(void) {
  bits_roundtrip();
  hyb_roundtrip();
  leb_zigzag();
  TEST_END();
}
