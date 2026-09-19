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
  static uint8_t src[VOLCOMP_CHUNK_VOXELS], srcl[VOLCOMP_CHUNK_VOXELS], dec[VOLCOMP_CHUNK_VOXELS];
  static uint8_t enc[VOLCOMP_ENCODE_BOUND];
  vt_synth_chunk(src, 2026);
  memset(src, 0, 40000); /* include flat blocks */
  if (regen) writef("tests/golden/src128.u8", src, sizeof src);
  size_t sn;
  uint8_t *gs = readf("tests/golden/src128.u8", &sn);
  CHECK(gs && sn == sizeof src && memcmp(gs, src, sn) == 0);
  /* the lossless golden runs on a label-like reduction of the same chunk (8
   * classes): it exercises the CONST / SPARSE / DENSE block modes and keeps the
   * checked-in stream small. RAW blocks are covered by test_lossless. */
  for (size_t i = 0; i < sizeof src; i++) srcl[i] = (uint8_t)(src[i] >> 5);
  const uint8_t *srcs[] = {src, src, srcl};
  const float qs[] = {2.0f, 32.0f, VOLCOMP_Q_LOSSLESS};
  const char *names[] = {"q2", "q32", "q0"};
  for (int k = 0; k < 3; k++) {
    size_t n;
    CHECK_EQ(volcomp_encode(srcs[k], qs[k], enc, sizeof enc, &n), VOLCOMP_OK);
    CHECK_EQ(volcomp_decode(enc, n, dec, sizeof dec), VOLCOMP_OK);
    char pe[64], po[64];
    snprintf(pe, sizeof pe, "tests/golden/%s.volc", names[k]);
    snprintf(po, sizeof po, "tests/golden/%s.out.u8", names[k]);
    /* The lossless stream reconstructs src128.u8 exactly, so it has no separate
     * decoded golden: the source file is its expected output, and the bitstream
     * is frozen on every kernel set (the lossless path has no float math). */
    const bool lossless = qs[k] == VOLCOMP_Q_LOSSLESS;
    if (lossless) CHECK(memcmp(dec, srcl, sizeof dec) == 0);
    if (regen) {
      writef(pe, enc, n);
      if (!lossless) writef(po, dec, sizeof dec);
      printf("wrote %s (%zu bytes)\n", pe, n);
      continue;
    }
    size_t gn, on = sizeof dec;
    uint8_t *ge = readf(pe, &gn), *go = lossless ? srcl : readf(po, &on);
    CHECK(ge && go);
    if (!ge || !go) continue;
    const bool avx2 = lossless || !strcmp(volcomp_kernels(), "avx2");
    const bool same_bytes = gn == n && memcmp(ge, enc, n) == 0;
    /* The goldens were produced by the AVX2 kernels: that path is frozen byte
     * for byte. The C kernels are held to the spec's cross-build contract
     * (+-1 LSB on decode) and are reported, not required, to match the bytes. */
    if (avx2) {
      CHECK_EQ(gn, n);
      CHECK(same_bytes); /* bitstream frozen */
    }
    if (!avx2 && tol < 1) tol = 1;
    if (lossless) tol = 0;
    CHECK_EQ(on, sizeof dec);
    int mx = 0;
    for (size_t i = 0; i < sizeof dec; i++) {
      int d = abs((int)go[i] - (int)dec[i]);
      mx = d > mx ? d : mx;
    }
    printf("%s [%s kernels]: %zu bytes (%s golden), decode max diff vs golden %d (tolerance %d)\n", names[k],
           volcomp_kernels(), n, same_bytes ? "identical to" : "differs from", mx, tol);
    CHECK(mx <= tol);
    /* the checked-in stream decodes too (decoder frozen against the file),
     * to within the cross-build tolerance */
    CHECK_EQ(volcomp_decode(ge, gn, dec, sizeof dec), VOLCOMP_OK);
    mx = 0;
    for (size_t i = 0; i < sizeof dec; i++) {
      int d = abs((int)go[i] - (int)dec[i]);
      mx = d > mx ? d : mx;
    }
    CHECK(mx <= tol);
    free(ge);
    if (!lossless) free(go);
  }
  free(gs);
  TEST_END();
}
