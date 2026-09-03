/* Label codec: layout, round trip, absent classes, block decode, bound, hostile streams. */
#include "../volcomp_label.h"
#include "check.h"
#include "metrics.h"

#define N VOLCOMP_CHUNK_VOXELS
static uint8_t g_p1[N], g_p2[N], g_dec[N], g_ref[N], g_blk[VOLCOMP_BLOCK_VOXELS];
static uint8_t g_enc[VOLCOMP_LABEL_ENCODE_BOUND(4)];

static void roundtrip(void) {
  vt_synth_chunk(g_p1, 3);
  vt_synth_chunk(g_p2, 11);
  volcomp_label_plane pl[4] = {{1, g_p1}, {2, NULL}, {7, g_p2}, {200, NULL}};
  size_t n;
  CHECK_EQ(volcomp_label_encode(pl, 4, 8.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK(memcmp(g_enc, "VOLL", 4) == 0);
  CHECK_EQ(g_enc[4], 1);
  CHECK_EQ(g_enc[5], 2); /* classes 2 and 200 absent */
  CHECK_EQ(vf_rd_u16(g_enc + 6), 8 * 256);
  uint8_t cls[255];
  uint32_t nc;
  CHECK_EQ(volcomp_label_classes(g_enc, n, cls, &nc), VOLCOMP_OK);
  CHECK_EQ(nc, 2);
  CHECK_EQ(cls[0], 1);
  CHECK_EQ(cls[1], 7);
  float q;
  CHECK_EQ(volcomp_label_q(g_enc, n, &q), VOLCOMP_OK);
  CHECK(q == 8.0f);
  /* each stored plane is byte-identical to a standalone volcomp encode at q */
  uint8_t *ref = malloc(VOLCOMP_ENCODE_BOUND);
  size_t rn;
  CHECK_EQ(volcomp_encode(g_p1, 8.0f, ref, VOLCOMP_ENCODE_BOUND, &rn), VOLCOMP_OK);
  const uint8_t *dir = g_enc + VOLCOMP_LABEL_HDR_BYTES;
  CHECK_EQ(dir[1], VL_MODE_IMAGE);
  CHECK_EQ(vf_rd_u32(dir + 4), rn);
  CHECK(memcmp(g_enc + VOLCOMP_LABEL_HDR_BYTES + 2 * VOLCOMP_LABEL_DIR_ENTRY, ref, rn) == 0);
  CHECK_EQ(volcomp_decode(ref, rn, g_ref, sizeof g_ref), VOLCOMP_OK);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 1, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK(memcmp(g_dec, g_ref, N) == 0);
  printf("plane 1: %zu bytes psnr %.2f\n", rn, metric_psnr_u8(g_p1, g_dec, N));
  CHECK(metric_psnr_u8(g_p1, g_dec, N) > 30.0);
  /* second plane, and block decode == region of full decode */
  CHECK_EQ(volcomp_label_decode(g_enc, n, 7, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK(metric_psnr_u8(g_p2, g_dec, N) > 30.0);
  CHECK_EQ(volcomp_label_decode_block(g_enc, n, 7, 3, 5, 6, g_blk, sizeof g_blk), VOLCOMP_OK);
  for (uint32_t z = 0; z < 16; z++)
    for (uint32_t y = 0; y < 16; y++)
      CHECK(memcmp(g_blk + (z * 16 + y) * 16, g_dec + (((size_t)3 * 16 + z) * 128 + 5 * 16 + y) * 128 + 6 * 16, 16) == 0);
  /* absent classes decode as zeros (full and block) */
  CHECK_EQ(volcomp_label_decode(g_enc, n, 2, g_dec, sizeof g_dec), VOLCOMP_OK);
  uint32_t nz = 0;
  for (size_t i = 0; i < N; i++) nz += g_dec[i] != 0;
  CHECK_EQ(nz, 0);
  memset(g_blk, 1, sizeof g_blk);
  CHECK_EQ(volcomp_label_decode_block(g_enc, n, 200, 0, 0, 0, g_blk, sizeof g_blk), VOLCOMP_OK);
  for (size_t i = 0; i < VOLCOMP_BLOCK_VOXELS; i++) nz += g_blk[i];
  CHECK_EQ(nz, 0);
  /* short buffer and args */
  size_t n2;
  CHECK_EQ(volcomp_label_encode(pl, 4, 8.0f, g_enc, n - 1, &n2), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_label_encode(pl, 4, 8.0f, g_enc, 10, &n2), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_label_encode(pl, 4, 0.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_ERR_ARG);
  volcomp_label_plane bad[2] = {{5, g_p1}, {5, g_p2}};
  CHECK_EQ(volcomp_label_encode(bad, 2, 8.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_label_decode(g_enc, n, 1, g_dec, N - 1), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_label_decode_block(g_enc, n, 1, 8, 0, 0, g_blk, sizeof g_blk), VOLCOMP_ERR_ARG);
  /* empty chunk */
  CHECK_EQ(volcomp_label_encode(NULL, 0, 8.0f, g_enc, sizeof g_enc, &n2), VOLCOMP_OK);
  CHECK_EQ(n2, VOLCOMP_LABEL_HDR_BYTES);
  CHECK_EQ(volcomp_label_decode(g_enc, n2, 1, g_dec, sizeof g_dec), VOLCOMP_OK);
  free(ref);
}

static void bound_and_raw(void) {
  /* full-entropy noise at q=1 exceeds 2 MiB encoded -> raw fallback, exact, within bound */
  uint32_t r = 99;
  for (size_t i = 0; i < N; i++) g_p1[i] = (uint8_t)vt_rng(&r);
  volcomp_label_plane pl[1] = {{4, g_p1}};
  size_t n;
  CHECK_EQ(volcomp_label_encode(pl, 1, 1.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK(n <= VOLCOMP_LABEL_ENCODE_BOUND(1));
  CHECK_EQ(volcomp_label_decode(g_enc, n, 4, g_dec, sizeof g_dec), VOLCOMP_OK);
  printf("noise plane: %zu bytes, mode %u\n", n, g_enc[VOLCOMP_LABEL_HDR_BYTES + 1]);
  if (g_enc[VOLCOMP_LABEL_HDR_BYTES + 1] == VL_MODE_RAW) {
    CHECK(memcmp(g_dec, g_p1, N) == 0);
    CHECK_EQ(volcomp_label_decode_block(g_enc, n, 4, 7, 7, 7, g_blk, sizeof g_blk), VOLCOMP_OK);
    CHECK(memcmp(g_blk + 15 * 256, g_p1 + ((size_t)127 * 128 + 112) * 128 + 112, 16) == 0);
  }
}

static void hostile(void) {
  vt_synth_chunk(g_p1, 5);
  volcomp_label_plane pl[2] = {{1, g_p1}, {9, g_p1}};
  size_t n;
  CHECK_EQ(volcomp_label_encode(pl, 2, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  for (size_t cut = 0; cut < n; cut += (cut < 64 ? 1 : n / 97))
    CHECK(volcomp_label_decode(g_enc, cut, 1, g_dec, sizeof g_dec) != VOLCOMP_OK);
  uint8_t *m = malloc(n);
  memcpy(m, g_enc, n);
  m[0] = 'X';
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[4] = 2;
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_VERSION);
  memcpy(m, g_enc, n);
  m[8] = 1;
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  vf_wr_u16(m + 6, 0);
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  vf_wr_u32(m + VOLCOMP_LABEL_HDR_BYTES + 4, vf_rd_u32(m + VOLCOMP_LABEL_HDR_BYTES + 4) + 1);
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[VOLCOMP_LABEL_HDR_BYTES + 8] = 1; /* duplicate class id */
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[VOLCOMP_LABEL_HDR_BYTES + 1] = 2; /* unknown mode */
  CHECK_EQ(volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  uint32_t r = 5;
  for (int k = 0; k < 300; k++) {
    memcpy(m, g_enc, n);
    size_t pos = vt_rng(&r) % n;
    m[pos] ^= (uint8_t)(1u << (vt_rng(&r) % 8));
    (void)volcomp_label_decode(m, n, 1, g_dec, sizeof g_dec);
    (void)volcomp_label_decode_block(m, n, 9, 1, 2, 3, g_blk, sizeof g_blk);
  }
  free(m);
}

int main(void) {
  roundtrip();
  bound_and_raw();
  hostile();
  TEST_END();
}
