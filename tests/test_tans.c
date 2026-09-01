#include "../volcomp.h"
#include "check.h"

static void roundtrip(uint32_t seed, size_t n, uint32_t skew) {
  vf_model m[VF_NMODELS];
  for (uint32_t i = 0; i < VF_NMODELS; i++) {
    uint32_t counts[VF_NTOK];
    for (uint32_t s = 0; s < VF_NTOK; s++) counts[s] = 1 + (vt_rng(&seed) % skew) * (s < 6 ? 50 : 1);
    CHECK(vf_model_build(&m[i], counts));
  }
  uint16_t *syms = malloc(n * sizeof *syms);
  for (size_t i = 0; i < n; i++) {
    uint32_t ctx = vt_rng(&seed) % VF_NMODELS;
    uint32_t s = vt_rng(&seed) % VF_NTOK;
    if (vt_rng(&seed) % 4) s = vt_rng(&seed) % 6;
    syms[i] = (uint16_t)(s | ctx << 8);
  }
  size_t cap = 2 * n + 8;
  uint8_t *out = malloc(cap);
  static vf_etab es[VF_NMODELS];
  static uint32_t fields[3 * 131072];
  vf_etabs_init(m, es);
  size_t rn = vf_tans_encode2(es, syms, n, fields, out, cap);
  CHECK(rn > 0 && rn <= cap);
  vf_rdec d;
  CHECK(vf_rdec_init(&d, out, rn));
  for (size_t i = 0; i < n; i++) {
    int t = vf_rdec_get(&d, &m[syms[i] >> 8]);
    CHECK_EQ(t, syms[i] & 0xff);
  }
  CHECK(vf_rdec_finished(&d));
  /* a truncated stream fails closed; an appended byte fails the finish check */
  if (rn > 12) {
    vf_rdec d2;
    CHECK(vf_rdec_init(&d2, out, rn - 4 > 3 ? rn - 4 : 3));
    bool failed = false;
    for (size_t i = 0; i < n && !failed; i++) failed = vf_rdec_get(&d2, &m[syms[i] >> 8]) < 0;
    CHECK(failed || !vf_rdec_finished(&d2));
  }
  uint8_t *out2 = malloc(rn + 1);
  memcpy(out2, out, rn);
  out2[rn] = 0x5a;
  vf_rdec d3;
  CHECK(vf_rdec_init(&d3, out2, rn + 1));
  for (size_t i = 0; i < n; i++) (void)vf_rdec_get(&d3, &m[syms[i] >> 8]);
  CHECK(!vf_rdec_finished(&d3));
  free(out2);
  free(out);
  free(syms);
}

int main(void) {
  roundtrip(1, 1, 3);
  roundtrip(2, 2, 3);
  roundtrip(3, 1000, 100);
  roundtrip(4, 131072, 4000); /* one full substream worst case token count */
  roundtrip(5, 77777, 2);
  /* too short for two initial states */
  uint8_t bad[8] = {0};
  vf_rdec d;
  CHECK(!vf_rdec_init(&d, bad, 2));
  TEST_END();
}
