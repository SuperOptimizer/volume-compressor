#include "../volcomp.h"
#include "check.h"

static void roundtrip_models(uint32_t seed, uint32_t density) {
  vf_model m[VF_NMODELS], back[VF_NMODELS];
  for (uint32_t i = 0; i < VF_NMODELS; i++) {
    uint32_t counts[VF_NTOK] = {0};
    for (uint32_t s = 0; s < VF_NTOK; s++)
      if (vt_rng(&seed) % 100 < density) counts[s] = 1 + vt_rng(&seed) % 5000;
    counts[vt_rng(&seed) % VF_NTOK] += 1; /* at least one */
    CHECK(vf_model_build(&m[i], counts));
    uint32_t sum = 0;
    for (uint32_t s = 0; s < VF_NTOK; s++) sum += m[i].freq[s];
    CHECK_EQ(sum, VF_PROB_SCALE);
  }
  uint8_t buf[VF_TABLES_MAX_BYTES];
  size_t n = vf_tables_write(m, buf);
  CHECK(n <= VF_TABLES_MAX_BYTES);
  vf_cur c = {buf, buf + n};
  CHECK(vf_tables_read(&c, back));
  CHECK(c.p == c.end);
  for (uint32_t i = 0; i < VF_NMODELS; i++) {
    CHECK(memcmp(m[i].freq, back[i].freq, sizeof m[i].freq) == 0);
    CHECK(memcmp(m[i].cum, back[i].cum, sizeof m[i].cum) == 0);
    CHECK(memcmp(m[i].dt, back[i].dt, sizeof m[i].dt) == 0);
  }
  /* truncation at every byte is rejected */
  for (size_t cut = 0; cut < n; cut++) {
    vf_cur t = {buf, buf + cut};
    CHECK(!vf_tables_read(&t, back));
  }
}

static void rejections(void) {
  uint32_t f[VF_NTOK] = {0};
  vf_model m;
  f[0] = VF_PROB_SCALE - 1;
  CHECK(!vf_model_from_freqs(&m, f)); /* one short */
  f[0] = VF_PROB_SCALE + 1;
  CHECK(!vf_model_from_freqs(&m, f)); /* one over */
  f[0] = VF_PROB_SCALE;
  CHECK(vf_model_from_freqs(&m, f));
  CHECK_EQ(m.dt[0].sym, 0);
  CHECK_EQ(m.dt[VF_PROB_SCALE - 1u].sym, 0);
  /* a stream whose table does not sum to 4096 must be rejected by tables_read */
  uint8_t buf[64];
  size_t n = 0;
  vf_wr_u32(buf + n, 3u);
  n += 4;
  n += vf_leb_put(buf + n, 500); /* 500 + 500 = 1000 != VF_PROB_SCALE */
  n += vf_leb_put(buf + n, 500);
  vf_cur c = {buf, buf + n};
  vf_model mm[VF_NMODELS];
  CHECK(!vf_tables_read(&c, mm));
  /* empty bitmap rejected */
  vf_wr_u32(buf, 0);
  c = (vf_cur){buf, buf + 4};
  CHECK(!vf_tables_read(&c, mm));
  /* zero frequency with bit set rejected */
  n = 0;
  vf_wr_u32(buf + n, 3u);
  n += 4;
  n += vf_leb_put(buf + n, 0);
  n += vf_leb_put(buf + n, VF_PROB_SCALE);
  c = (vf_cur){buf, buf + n};
  CHECK(!vf_tables_read(&c, mm));
}

int main(void) {
  for (uint32_t d = 5; d <= 100; d += 19) roundtrip_models(1000 + d, d);
  rejections();
  TEST_END();
}
