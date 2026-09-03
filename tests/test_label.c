/* Label codec: layout, exact round trip at t=0, tolerance guarantee at t>0,
 * absent classes, mode selection, bound, hostile streams. */
#include "../volcomp_label.h"
#include "check.h"

#include <math.h>

#define N VOLCOMP_CHUNK_VOXELS
static uint8_t g_mask1[N], g_mask2[N], g_ids[N], g_prob[N], g_const[N], g_dec[N];
static uint8_t g_enc[VOLCOMP_LABEL_ENCODE_BOUND(6)];

static inline size_t at(uint32_t z, uint32_t y, uint32_t x) { return ((size_t)z * 128 + y) * 128 + x; }

/* two blobs that touch, an id map with 6 values, a smooth probability map */
static void synth(void) {
  uint32_t r = 7;
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        size_t i = at(z, y, x);
        double dz = (double)z - 64, dy = (double)y - 64, dx = (double)x - 50;
        double d1 = sqrt(dz * dz + dy * dy + dx * dx);
        double dx2 = (double)x - 80;
        double d2 = sqrt(dz * dz * 0.5 + dy * dy + dx2 * dx2);
        /* jagged edge: noise on the radius */
        double n = (double)(vt_rng(&r) % 5) - 2.0;
        g_mask1[i] = d1 + n < 30 ? 255 : 0;
        g_mask2[i] = d2 + n < 28 && !g_mask1[i] ? 255 : 0;
        /* single-voxel specks */
        if ((vt_rng(&r) % 4000) == 0) g_mask1[i] = 255;
        g_ids[i] = (uint8_t)(d1 < 20 ? 1 : d1 < 30 ? 2 : d2 < 25 ? 3 : (z < 10 ? 4 : (x > 120 ? 5 : 0)));
        double p = 128 + 100 * sin(x * 0.05) * cos(y * 0.04 + z * 0.03);
        g_prob[i] = (uint8_t)(p < 0 ? 0 : p > 255 ? 255 : p);
        g_const[i] = 9;
      }
}

static uint32_t count_changed(const uint8_t *a, const uint8_t *b) {
  uint32_t c = 0;
  for (size_t i = 0; i < N; i++) c += a[i] != b[i];
  return c;
}
/* every changed voxel must lie within t of a source boundary: its (2t+1)^3
 * source window is non-uniform, and its new value occurs in that window */
static bool tolerance_ok(const uint8_t *src, const uint8_t *dec, uint32_t t) {
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        size_t i = at(z, y, x);
        if (src[i] == dec[i]) continue;
        bool seen_new = false;
        for (int dz = -(int)t; dz <= (int)t; dz++)
          for (int dy = -(int)t; dy <= (int)t; dy++)
            for (int dx = -(int)t; dx <= (int)t; dx++) {
              int zz = (int)z + dz, yy = (int)y + dy, xx = (int)x + dx;
              if (zz < 0 || yy < 0 || xx < 0 || zz > 127 || yy > 127 || xx > 127) continue;
              if (src[at((uint32_t)zz, (uint32_t)yy, (uint32_t)xx)] == dec[i]) seen_new = true;
            }
        if (!seen_new) return false;
      }
  return true;
}
static bool values_subset(const uint8_t *src, const uint8_t *dec) {
  bool has[256] = {0};
  for (size_t i = 0; i < N; i++) has[src[i]] = true;
  for (size_t i = 0; i < N; i++)
    if (!has[dec[i]]) return false;
  return true;
}

static void roundtrip(void) {
  volcomp_label_plane pl[6] = {{1, g_mask1}, {2, g_mask2}, {3, NULL},   {7, g_ids},
                               {9, g_prob},  {200, g_const}};
  size_t n;
  CHECK_EQ(volcomp_label_encode(pl, 6, 0, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK(memcmp(g_enc, "VOLL", 4) == 0);
  CHECK_EQ(g_enc[4], 1);
  CHECK_EQ(g_enc[5], 5); /* class 3 absent */
  CHECK_EQ(g_enc[6], 0);
  uint8_t cls[255];
  uint32_t nc;
  CHECK_EQ(volcomp_label_classes(g_enc, n, cls, &nc), VOLCOMP_OK);
  CHECK_EQ(nc, 5);
  CHECK_EQ(cls[0], 1);
  CHECK_EQ(cls[1], 2);
  CHECK_EQ(cls[2], 7);
  CHECK_EQ(cls[3], 9);
  CHECK_EQ(cls[4], 200);
  /* modes */
  const uint8_t *dir = g_enc + VOLCOMP_LABEL_HDR_BYTES;
  CHECK_EQ(dir[1], VL_MODE_PALETTE);
  CHECK_EQ(dir[8 + 1], VL_MODE_PALETTE);
  CHECK_EQ(dir[16 + 1], VL_MODE_PALETTE);
  CHECK_EQ(dir[24 + 1], VL_MODE_IMAGE);
  CHECK_EQ(dir[32 + 1], VL_MODE_CONST);
  uint32_t tol;
  float q;
  CHECK_EQ(volcomp_label_params(g_enc, n, &tol, &q), VOLCOMP_OK);
  CHECK_EQ(tol, 0);
  CHECK(q == 4.0f);
  /* exact at t=0 for palette planes */
  CHECK_EQ(volcomp_label_decode(g_enc, n, 1, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK_EQ(count_changed(g_mask1, g_dec), 0);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 2, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK_EQ(count_changed(g_mask2, g_dec), 0);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 7, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK_EQ(count_changed(g_ids, g_dec), 0);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 200, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK_EQ(count_changed(g_const, g_dec), 0);
  /* image plane: lossy but close */
  CHECK_EQ(volcomp_label_decode(g_enc, n, 9, g_dec, sizeof g_dec), VOLCOMP_OK);
  uint32_t mx = 0;
  for (size_t i = 0; i < N; i++) {
    uint32_t e = (uint32_t)abs((int)g_prob[i] - (int)g_dec[i]);
    if (e > mx) mx = e;
  }
  CHECK(mx < 40);
  /* absent classes decode as zeros */
  CHECK_EQ(volcomp_label_decode(g_enc, n, 3, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 0, g_dec, sizeof g_dec), VOLCOMP_OK);
  uint32_t nz = 0;
  for (size_t i = 0; i < N; i++) nz += g_dec[i] != 0;
  CHECK_EQ(nz, 0);
  size_t dir_bytes = VOLCOMP_LABEL_HDR_BYTES + 5 * VOLCOMP_LABEL_DIR_ENTRY;
  printf("t=0: %zu bytes total; mask1 %u B, mask2 %u B, ids %u B, prob %u B, const %u B\n", n,
         vf_rd_u32(dir + 4) + vf_rd_u16(dir + 2), vf_rd_u32(dir + 12) + vf_rd_u16(dir + 10),
         vf_rd_u32(dir + 20) + vf_rd_u16(dir + 18), vf_rd_u32(dir + 28), vf_rd_u16(dir + 34));
  (void)dir_bytes;
  /* short buffer and args */
  size_t n2;
  CHECK_EQ(volcomp_label_encode(pl, 6, 0, 4.0f, g_enc, n - 1, &n2), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_label_encode(pl, 6, 0, 4.0f, g_enc, 10, &n2), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_label_encode(pl, 6, 8, 4.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_label_encode(pl, 6, 0, 0.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_ERR_ARG);
  volcomp_label_plane bad[2] = {{5, g_mask1}, {5, g_mask2}};
  CHECK_EQ(volcomp_label_encode(bad, 2, 0, 4.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 1, g_dec, N - 1), VOLCOMP_ERR_SHORT_BUF);
  /* empty chunk */
  CHECK_EQ(volcomp_label_encode(NULL, 0, 0, 4.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_OK);
  CHECK_EQ(n2, VOLCOMP_LABEL_HDR_BYTES);
  CHECK_EQ(volcomp_label_decode(g_enc, n2, 1, g_dec, sizeof g_dec), VOLCOMP_OK);
}

static void tolerance(void) {
  volcomp_label_plane pl[3] = {{1, g_mask1}, {2, g_mask2}, {7, g_ids}};
  size_t n0, n;
  CHECK_EQ(volcomp_label_encode(pl, 3, 0, 4.0f, g_enc, sizeof g_enc, &n0), VOLCOMP_OK);
  for (uint32_t t = 1; t <= 3; t++) {
    CHECK_EQ(volcomp_label_encode(pl, 3, t, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
    CHECK_EQ(g_enc[6], t);
    CHECK(n < n0);
    const uint8_t *srcs[3] = {g_mask1, g_mask2, g_ids};
    uint8_t ids[3] = {1, 2, 7};
    for (int k = 0; k < 3; k++) {
      CHECK_EQ(volcomp_label_decode(g_enc, n, ids[k], g_dec, sizeof g_dec), VOLCOMP_OK);
      uint32_t ch = count_changed(srcs[k], g_dec);
      CHECK(values_subset(srcs[k], g_dec));
      CHECK(tolerance_ok(srcs[k], g_dec, t));
      /* a specific interior voxel and a specific exterior voxel are untouched */
      CHECK_EQ(g_dec[at(64, 64, 50)], srcs[k][at(64, 64, 50)]);
      CHECK_EQ(g_dec[at(5, 120, 5)], srcs[k][at(5, 120, 5)]);
      printf("t=%u class %u: %u voxels changed (%.3f%%)\n", t, ids[k], ch, 100.0 * ch / N);
    }
    printf("t=%u: %zu bytes (t=0: %zu)\n", t, n, n0);
  }
}

static void bound_and_raw(void) {
  /* incompressible two-valued noise must still round-trip (raw fallback) within the bound */
  uint32_t r = 99;
  for (size_t i = 0; i < N; i++) g_dec[i] = (vt_rng(&r) & 1) ? 255 : 0;
  memcpy(g_prob, g_dec, N);
  volcomp_label_plane pl[1] = {{4, g_prob}};
  size_t n;
  CHECK_EQ(volcomp_label_encode(pl, 1, 0, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK(n <= VOLCOMP_LABEL_ENCODE_BOUND(1));
  CHECK_EQ(volcomp_label_decode(g_enc, n, 4, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK_EQ(count_changed(g_prob, g_dec), 0);
  printf("noise plane: %zu bytes, mode %u\n", n, g_enc[VOLCOMP_LABEL_HDR_BYTES + 1]);
  /* full-entropy 256-valued noise -> image or raw, still within bound */
  for (size_t i = 0; i < N; i++) g_prob[i] = (uint8_t)vt_rng(&r);
  CHECK_EQ(volcomp_label_encode(pl, 1, 0, 1.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK(n <= VOLCOMP_LABEL_ENCODE_BOUND(1));
  CHECK_EQ(volcomp_label_decode(g_enc, n, 4, g_dec, sizeof g_dec), VOLCOMP_OK);
}

static void hostile(void) {
  synth();
  volcomp_label_plane pl[3] = {{1, g_mask1}, {7, g_ids}, {9, g_prob}};
  size_t n;
  CHECK_EQ(volcomp_label_encode(pl, 3, 1, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  /* truncations */
  for (size_t cut = 0; cut < n; cut += (cut < 64 ? 1 : n / 97))
    CHECK(volcomp_label_decode(g_enc, cut, 1, g_dec, sizeof g_dec) != VOLCOMP_OK);
  /* magic / version / reserved */
  uint8_t *m = malloc(n);
  memcpy(m, g_enc, n);
  m[0] = 'X';
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[4] = 2;
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_VERSION);
  memcpy(m, g_enc, n);
  m[7] = 1;
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  /* directory lies: lengths, order, mode */
  memcpy(m, g_enc, n);
  vf_wr_u32(m + VOLCOMP_LABEL_HDR_BYTES + 4, vf_rd_u32(m + VOLCOMP_LABEL_HDR_BYTES + 4) + 1);
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[VOLCOMP_LABEL_HDR_BYTES + 8] = 1; /* second class id == first */
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[VOLCOMP_LABEL_HDR_BYTES + 1] = 4;
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  /* random byte flips must never crash and must not be accepted silently as a
   * different valid stream more often than the range coder allows: we only
   * require no crash and a status */
  uint32_t r = 5;
  for (int k = 0; k < 200; k++) {
    memcpy(m, g_enc, n);
    size_t pos = vt_rng(&r) % n;
    m[pos] ^= (uint8_t)(1u << (vt_rng(&r) % 8));
    volcomp_status st = volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec);
    (void)st;
    st = volcomp_label_decode(m, n, 9, g_dec, sizeof g_dec);
    (void)st;
  }
  free(m);
}

int main(void) {
  synth();
  roundtrip();
  tolerance();
  bound_and_raw();
  hostile();
  TEST_END();
}
