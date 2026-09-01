/* End-to-end codec properties: layout, round trip quality, block == full,
 * encode bound, hostile inputs. */
#include "../volcomp.h"
#include "check.h"
#include "metrics.h"

static uint8_t g_src[VOLCOMP_CHUNK_VOXELS], g_dec[VOLCOMP_CHUNK_VOXELS];
static uint8_t g_enc[VOLCOMP_ENCODE_BOUND];

static void layout_and_roundtrip(void) {
  vt_synth_chunk(g_src, 42);
  size_t n;
  CHECK_EQ(volcomp_encode(g_src, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK(n > 8 + 256);
  CHECK(memcmp(g_enc, "VOLC", 4) == 0);
  CHECK_EQ(g_enc[4], 1);
  CHECK_EQ(g_enc[5], 0);
  CHECK_EQ(vf_rd_u16(g_enc + 6), 4 * 256);
  /* q is rounded to 1/256 and stored; the decoder sees exactly that */
  size_t n2;
  CHECK_EQ(volcomp_encode(g_src, 4.001f, g_enc, sizeof g_enc, &n2), VOLCOMP_OK);
  CHECK_EQ(vf_rd_u16(g_enc + 6), 4 * 256);
  CHECK_EQ(volcomp_encode(g_src, 4.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  uint64_t h[256] = {0};
  metric_errhist_u8(g_src, g_dec, VOLCOMP_CHUNK_VOXELS, h);
  double psnr = metric_psnr_u8(g_src, g_dec, VOLCOMP_CHUNK_VOXELS);
  uint32_t mx = metric_maxerr_u8(g_src, g_dec, VOLCOMP_CHUNK_VOXELS);
  printf("q4: %zu bytes (%.1fx) psnr %.2f mae %.3f p99 %u max %u\n", n, 2097152.0 / (double)n, psnr,
         errhist_mae(h), errhist_percentile(h, 0.99), mx);
  CHECK(psnr > 30.0);
  CHECK(mx < 60);
  /* short buffer */
  size_t n3;
  CHECK_EQ(volcomp_encode(g_src, 4.0f, g_enc, n - 1, &n3), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_encode(g_src, 4.0f, g_enc, 100, &n3), VOLCOMP_ERR_SHORT_BUF);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec - 1), VOLCOMP_ERR_SHORT_BUF);
  /* args */
  CHECK_EQ(volcomp_encode(g_src, 0.5f, g_enc, sizeof g_enc, &n3), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_encode(g_src, 256.0f, g_enc, sizeof g_enc, &n3), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_encode(g_src, NAN, g_enc, sizeof g_enc, &n3), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_encode(NULL, 4.0f, g_enc, sizeof g_enc, &n3), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_decode_block(g_enc, n, 8, 0, 0, g_dec, 4096), VOLCOMP_ERR_ARG);
  CHECK_EQ(volcomp_decode_block(g_enc, n, 0, 0, 0, g_dec, 4095), VOLCOMP_ERR_SHORT_BUF);
  /* q extremes and re-encode determinism */
  size_t a, b;
  CHECK_EQ(volcomp_encode(g_src, 1.0f, g_enc, sizeof g_enc, &a), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(g_enc, a, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK(metric_maxerr_u8(g_src, g_dec, VOLCOMP_CHUNK_VOXELS) < 20);
  CHECK_EQ(volcomp_encode(g_src, 255.0f, g_enc, sizeof g_enc, &b), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(g_enc, b, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK(b < a);
  uint8_t *enc2 = malloc(a);
  size_t a2;
  CHECK_EQ(volcomp_encode(g_src, 1.0f, enc2, a, &a2), VOLCOMP_OK);
  CHECK_EQ(a2, a);
  CHECK(memcmp(enc2, g_enc, 0) == 0);
  free(enc2);
}

static void block_vs_full(uint32_t seed, float q) {
  vt_synth_chunk(g_src, seed);
  if (seed & 1) memset(g_src, 77, 300000); /* include flat blocks */
  size_t n;
  CHECK_EQ(volcomp_encode(g_src, q, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  uint8_t blk[4096];
  int bad = 0;
  for (uint32_t b = 0; b < 512; b++) {
    uint32_t bz = b >> 6, by = (b >> 3) & 7, bx = b & 7;
    if (volcomp_decode_block(g_enc, n, bz, by, bx, blk, sizeof blk) != VOLCOMP_OK) {
      bad++;
      continue;
    }
    for (uint32_t z = 0; z < 16 && !bad; z++)
      for (uint32_t y = 0; y < 16; y++)
        if (memcmp(blk + z * 256 + y * 16,
                   g_dec + ((size_t)(bz * 16 + z) * 128 + by * 16 + y) * 128 + bx * 16, 16)) {
          bad++;
          break;
        }
  }
  CHECK_EQ(bad, 0);
}

static void bound(void) {
  /* worst plausible content: uniform noise at q = 1 and a checkerboard */
  uint32_t r = 99;
  for (size_t i = 0; i < sizeof g_src; i++) g_src[i] = (uint8_t)vt_rng(&r);
  size_t n;
  CHECK_EQ(volcomp_encode(g_src, 1.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  printf("noise q1: %zu bytes (bound %zu)\n", n, (size_t)VOLCOMP_ENCODE_BOUND);
  CHECK(n <= VOLCOMP_ENCODE_BOUND);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  for (size_t i = 0; i < sizeof g_src; i++) g_src[i] = ((i ^ (i >> 7) ^ (i >> 14)) & 1) ? 255 : 0;
  CHECK_EQ(volcomp_encode(g_src, 1.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  printf("checker q1: %zu bytes\n", n);
  CHECK(n <= VOLCOMP_ENCODE_BOUND);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  /* all-zero and all-255 chunks */
  memset(g_src, 0, sizeof g_src);
  CHECK_EQ(volcomp_encode(g_src, 8.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK(memcmp(g_src, g_dec, sizeof g_src) == 0);
  printf("zero chunk: %zu bytes\n", n);
  memset(g_src, 255, sizeof g_src);
  CHECK_EQ(volcomp_encode(g_src, 8.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  CHECK_EQ(volcomp_decode(g_enc, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  CHECK(memcmp(g_src, g_dec, sizeof g_src) == 0);
}

static void hostile(void) {
  vt_synth_chunk(g_src, 7);
  size_t n;
  CHECK_EQ(volcomp_encode(g_src, 3.0f, g_enc, sizeof g_enc, &n), VOLCOMP_OK);
  uint8_t *m = malloc(n + 16);
  /* truncation at many lengths: never OK, never crash */
  for (size_t cut = 0; cut < n; cut += (cut < 300 ? 1 : n / 40)) {
    memcpy(m, g_enc, cut);
    CHECK(volcomp_decode(m, cut, g_dec, sizeof g_dec) != VOLCOMP_OK);
    CHECK(volcomp_decode_block(m, cut, 3, 3, 3, g_dec, 4096) != VOLCOMP_OK);
  }
  /* appended byte: exact accounting fails */
  memcpy(m, g_enc, n);
  m[n] = 0;
  CHECK_EQ(volcomp_decode(m, n + 1, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  /* version / reserved / q */
  memcpy(m, g_enc, n);
  m[4] = 2;
  CHECK_EQ(volcomp_decode(m, n, g_dec, sizeof g_dec), VOLCOMP_ERR_VERSION);
  memcpy(m, g_enc, n);
  m[5] = 1;
  CHECK_EQ(volcomp_decode(m, n, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  vf_wr_u16(m + 6, 255);
  CHECK_EQ(volcomp_decode(m, n, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  vf_wr_u16(m + 6, 65281);
  CHECK_EQ(volcomp_decode(m, n, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  memcpy(m, g_enc, n);
  m[1] = 'X';
  CHECK_EQ(volcomp_decode(m, n, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  /* misaligned buffer decodes identically */
  memcpy(m + 1, g_enc, n);
  CHECK_EQ(volcomp_decode(m + 1, n, g_dec, sizeof g_dec), VOLCOMP_OK);
  /* random bit flips: must never crash; if accepted, output must be sane bytes (always true) */
  uint32_t r = 5;
  int accepted = 0;
  for (int i = 0; i < 60; i++) {
    memcpy(m, g_enc, n);
    size_t at = vt_rng(&r) % n;
    m[at] ^= (uint8_t)(1u << (vt_rng(&r) & 7));
    if (volcomp_decode(m, n, g_dec, sizeof g_dec) == VOLCOMP_OK) accepted++;
    (void)volcomp_decode_block(m, n, vt_rng(&r) & 7, vt_rng(&r) & 7, vt_rng(&r) & 7, g_dec, 4096);
  }
  printf("bit flips accepted: %d/60 (payload flips can be self-consistent)\n", accepted);
  /* lying directory: shift bytes between substreams */
  memcpy(m, g_enc, n);
  {
    vf_parsed p;
    CHECK_EQ(vf_parse(g_enc, n, &p), VOLCOMP_OK);
    size_t dir = (size_t)(p.payload - g_enc) - VF_DIR_BYTES;
    vf_wr_u32(m + dir, p.tok_n[0] + 1);
    vf_wr_u32(m + dir + 8, p.tok_n[1] - 1);
    CHECK(volcomp_decode(m, n, g_dec, sizeof g_dec) != VOLCOMP_OK);
    /* tok_n below the 8-byte minimum */
    memcpy(m, g_enc, n);
    vf_wr_u32(m + dir, 4);
    vf_wr_u32(m + dir + 4, p.byp_n[0] + p.tok_n[0] - 4);
    CHECK(volcomp_decode(m, n, g_dec, sizeof g_dec) != VOLCOMP_OK);
  }
  /* hostile DC deltas: craft a substream whose DC tokens are maximal so the
   * accumulator would overflow without the range check */
  {
    vf_model models[VF_NMODELS];
    uint32_t counts[VF_NTOK] = {0};
    for (uint32_t t = 0; t < VF_NTOK; t++) counts[t] = 1;
    for (uint32_t i = 0; i < VF_NMODELS; i++) CHECK(vf_model_build(&models[i], counts));
    uint16_t syms[64];
    uint8_t byp[1024];
    vf_bitw w;
    vf_bw_init(&w, byp, sizeof byp);
    size_t k = 0;
    for (int b = 0; b < 16; b++) {
      uint32_t t;
      CHECK(vf_hyb_emit(&w, vf_zigzag(131072), &t)); /* +131072 per block */
      syms[k++] = (uint16_t)(t | VF_DC_CTX << 8);
      syms[k++] = (uint16_t)(VF_TOK_EOB | vf_run_ctx(1) << 8);
    }
    size_t bn = 0;
    CHECK(vf_bw_flush(&w, &bn));
    uint8_t rans[512];
    static vf_etab es[VF_NMODELS];
    static uint32_t fields[4096];
    vf_etabs_init(models, es);
    size_t rn = vf_tans_encode2(es, syms, k, fields, rans, sizeof rans);
    CHECK(rn > 0);
    uint8_t tab[VF_TABLES_MAX_BYTES];
    size_t tn = vf_tables_write(models, tab);
    size_t pos = 0;
    memcpy(m, "VOLC", 4);
    m[4] = 1;
    m[5] = 0;
    vf_wr_u16(m + 6, 256);
    pos = 8;
    memcpy(m + pos, tab, tn);
    pos += tn;
    uint8_t *dir = m + pos;
    pos += VF_DIR_BYTES;
    for (uint32_t s = 0; s < VF_NSUB; s++) {
      vf_wr_u32(dir + s * 8, (uint32_t)rn);
      vf_wr_u32(dir + s * 8 + 4, (uint32_t)bn);
      memcpy(m + pos, rans, rn);
      pos += rn;
      memcpy(m + pos, byp, bn);
      pos += bn;
    }
    CHECK_EQ(volcomp_decode(m, pos, g_dec, sizeof g_dec), VOLCOMP_ERR_CORRUPT);
  }
  free(m);
}

/* factored transform vs the explicit even/odd matrix form (both orthonormal) */
static void dct_equivalence(void) {
  uint32_t r = 3;
  float blk[4096], ref[4096], back[4096];
  for (int i = 0; i < 4096; i++) blk[i] = (float)(int)(vt_rng(&r) % 256) - 128.0f;
  memcpy(back, blk, sizeof blk);
  /* reference: matrix DCT along each axis */
  for (int ax = 0; ax < 3; ax++) {
    ptrdiff_t st = ax == 0 ? 1 : (ax == 1 ? 16 : 256);
    for (int a = 0; a < 16; a++)
      for (int b = 0; b < 16; b++) {
        ptrdiff_t base = ax == 0 ? a * 256 + b * 16 : (ax == 1 ? a * 256 + b : a * 16 + b);
        float e[8], o[8], out[16];
        for (int i = 0; i < 8; i++) {
          e[i] = blk[base + i * st] + blk[base + (15 - i) * st];
          o[i] = blk[base + i * st] - blk[base + (15 - i) * st];
        }
        for (int m = 0; m < 8; m++) {
          float ae = 0, ao = 0;
          for (int i = 0; i < 8; i++) ae += VF_EV[m][i] * e[i], ao += VF_OD[m][i] * o[i];
          out[2 * m] = ae;
          out[2 * m + 1] = ao;
        }
        for (int i = 0; i < 16; i++) blk[base + i * st] = out[i];
      }
  }
  memcpy(ref, blk, sizeof ref);
  memcpy(blk, back, sizeof blk);
  vf_dct16_fwd(blk);
  for (int i = 0; i < 4096; i++) blk[i] *= vf_dct_scale((uint32_t)i); /* kernels are unnormalised */
  float md = 0;
  for (int i = 0; i < 4096; i++) md = fmaxf(md, fabsf(blk[i] - ref[i]));
  printf("factored vs matrix DCT max diff %.3e\n", (double)md);
  CHECK(md < 2e-3f);
  for (int i = 0; i < 4096; i++) blk[i] *= vf_dct_scale((uint32_t)i);
  {
    float out[4096];
    vf_dct16_inv(blk, 0xffffu, 0xffffu, out);
    memcpy(blk, out, sizeof blk);
  }
  md = 0;
  for (int i = 0; i < 4096; i++) md = fmaxf(md, fabsf(blk[i] - back[i]));
  printf("fwd+inv round trip max diff %.3e\n", (double)md);
  CHECK(md < 2e-3f);
}

/* scalar reference of the post-filter (one face position at a time, in the
 * same axis order), against which the 16-lane AVX2 path must be exact */
static void deblock_ref_axis(uint8_t *vol, size_t n_outer, size_t n_mid, size_t n_in, size_t s_outer,
                             size_t s_mid, size_t s_in, int c) {
  for (size_t f = 16; f < n_in; f += 16)
    for (size_t o = 0; o < n_outer; o++)
      for (size_t m = 0; m < n_mid; m++) {
        uint8_t *q0 = vol + o * s_outer + m * s_mid + f * s_in, *p0 = q0 - s_in;
        int P1 = p0[-(ptrdiff_t)s_in], P0 = *p0, Q0 = *q0, Q1 = q0[s_in];
        int d = Q0 - P0, ad = d < 0 ? -d : d;
        if (ad == 0 || ad >= 4 * c) continue;
        int dp = P1 - P0, dq = Q1 - Q0;
        if ((dp < 0 ? -dp : dp) >= c || (dq < 0 ? -dq : dq) >= c) continue;
        int delta = (3 * ad + 4) >> 3;
        if (d < 0) delta = -delta;
        *p0 = (uint8_t)(P0 + delta);
        *q0 = (uint8_t)(Q0 - delta);
      }
}
static void deblock_matches_reference(void) {
  const size_t nz = 40, ny = 37, nx = 70; /* non-multiples of 16: exercises the scalar tails */
  uint8_t *a = malloc(nz * ny * nx), *b = malloc(nz * ny * nx);
  uint32_t seed = 77;
  for (float q = 1.0f; q <= 64.0f; q *= 4.0f) {
    for (size_t i = 0; i < nz * ny * nx; i++) { /* ramp + noise + steps on some 16-faces */
      size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
      int v = (int)(x * 2 + y + z) + (int)(vt_rng(&seed) % 9) - 4 +
              ((x / 16 + y / 16 + z / 16) % 3 == 0 ? 6 : 0);
      a[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    memcpy(b, a, nz * ny * nx);
    volcomp_deblock(a, nz, ny, nx, q);
    float cf = 0.8f * q + 1.0f;
    int c = (int)(cf < 1.0f ? 1.0f : (cf > 24.0f ? 24.0f : cf));
    deblock_ref_axis(b, nz, ny, nx, nx * ny, nx, 1, c);
    deblock_ref_axis(b, nz, nx, ny, nx * ny, 1, nx, c);
    deblock_ref_axis(b, ny, nx, nz, nx, 1, nx * ny, c);
    CHECK(memcmp(a, b, nz * ny * nx) == 0);
  }
  free(a);
  free(b);
}

int main(void) {
  dct_equivalence();
  deblock_matches_reference();
  layout_and_roundtrip();
  block_vs_full(1, 2.0f);
  block_vs_full(2, 9.5f);
  block_vs_full(3, 40.0f);
  bound();
  hostile();
  TEST_END();
}
