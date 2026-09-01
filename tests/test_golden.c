/* Golden vectors freeze the v1 bitstream and decoder output. src128.u8 is a
 * deterministic synthetic chunk; for each q the checked-in .volc must be
 * byte-identical to a fresh encode and its checked-in .out.u8 must match the
 * decode within VOLCOMP_GOLDEN_TOLERANCE (default 0; set 1 for builds with a
 * different vector width / compiler). Run with --regen to (re)write them. */
#include "../volcomp.h"
#include "check.h"

static uint8_t *readf(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = malloc((size_t)sz);
  *n = fread(b, 1, (size_t)sz, f);
  fclose(f);
  return b;
}
static void writef(const char *p, const void *b, size_t n) {
  FILE *f = fopen(p, "wb");
  CHECK(f && fwrite(b, 1, n, f) == n);
  if (f) fclose(f);
}

int main(int argc, char **argv) {
  bool regen = argc > 1 && !strcmp(argv[1], "--regen");
  const char *tol_env = getenv("VOLCOMP_GOLDEN_TOLERANCE");
  int tol = tol_env ? atoi(tol_env) : 0;
  static uint8_t src[VOLCOMP_CHUNK_VOXELS], dec[VOLCOMP_CHUNK_VOXELS];
  static uint8_t enc[VOLCOMP_ENCODE_BOUND];
  vt_synth_chunk(src, 2026);
  memset(src, 0, 40000); /* include flat blocks */
  if (regen) writef("tests/golden/src128.u8", src, sizeof src);
  size_t sn;
  uint8_t *gs = readf("tests/golden/src128.u8", &sn);
  CHECK(gs && sn == sizeof src && memcmp(gs, src, sn) == 0);
  const float qs[] = {2.0f, 32.0f};
  const char *names[] = {"q2", "q32"};
  for (int k = 0; k < 2; k++) {
    size_t n;
    CHECK_EQ(volcomp_encode(src, qs[k], enc, sizeof enc, &n), VOLCOMP_OK);
    CHECK_EQ(volcomp_decode(enc, n, dec, sizeof dec), VOLCOMP_OK);
    char pe[64], po[64];
    snprintf(pe, sizeof pe, "tests/golden/%s.volc", names[k]);
    snprintf(po, sizeof po, "tests/golden/%s.out.u8", names[k]);
    if (regen) {
      writef(pe, enc, n);
      writef(po, dec, sizeof dec);
      printf("wrote %s (%zu bytes)\n", pe, n);
      continue;
    }
    size_t gn, on;
    uint8_t *ge = readf(pe, &gn), *go = readf(po, &on);
    CHECK(ge && go);
    if (!ge || !go) continue;
    CHECK_EQ(gn, n);
    CHECK(gn == n && memcmp(ge, enc, n) == 0); /* bitstream frozen */
    CHECK_EQ(on, sizeof dec);
    int mx = 0;
    for (size_t i = 0; i < sizeof dec; i++) {
      int d = abs((int)go[i] - (int)dec[i]);
      mx = d > mx ? d : mx;
    }
    printf("%s: %zu bytes, decode max diff vs golden %d (tolerance %d)\n", names[k], n, mx, tol);
    CHECK(mx <= tol);
    /* the checked-in stream decodes too (decoder frozen against the file) */
    CHECK_EQ(volcomp_decode(ge, gn, dec, sizeof dec), VOLCOMP_OK);
    free(ge);
    free(go);
  }
  free(gs);
  TEST_END();
}
