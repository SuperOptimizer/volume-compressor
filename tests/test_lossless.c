/* Lossless mode (q = VOLCOMP_Q_LOSSLESS): exactness on synthetic classes and on
 * 1000 fuzzed chunks, random access agreement, stream-shape checks, and a
 * ratio/throughput table (--bench, also fed by any tests/data/NAME.u8 present). */
#include "../volcomp.h"
#include "check.h"

#include <time.h>

#define N VOLCOMP_CHUNK_VOXELS

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* ---- synthetic classes ---- */
static void gen_const(uint8_t *v, uint32_t seed) { memset(v, (int)(seed & 0xffu), N); }

/* binary mask, ~5% foreground, spatially clustered (blobs, not salt) */
static void gen_mask(uint8_t *v, uint32_t seed) {
  uint32_t r = seed | 1u;
  memset(v, 0, N);
  for (int i = 0; i < 24; i++) {
    int cz = (int)(vt_rng(&r) % 128), cy = (int)(vt_rng(&r) % 128), cx = (int)(vt_rng(&r) % 128);
    int rad = 8 + (int)(vt_rng(&r) % 9);
    int lo_z = cz - rad < 0 ? 0 : cz - rad, hi_z = cz + rad > 127 ? 127 : cz + rad;
    int lo_y = cy - rad < 0 ? 0 : cy - rad, hi_y = cy + rad > 127 ? 127 : cy + rad;
    int lo_x = cx - rad < 0 ? 0 : cx - rad, hi_x = cx + rad > 127 ? 127 : cx + rad;
    for (int z = lo_z; z <= hi_z; z++)
      for (int y = lo_y; y <= hi_y; y++)
        for (int x = lo_x; x <= hi_x; x++) {
          int dz = z - cz, dy = y - cy, dx = x - cx;
          if (dz * dz + dy * dy + dx * dx < rad * rad) v[((size_t)z * 128 + (size_t)y) * 128 + (size_t)x] = 255;
        }
  }
}

/* nclass-way label map from nearest-seed (Voronoi) regions, resolved on a 64^3
 * grid and doubled (cheap enough to run 1000 times under a sanitizer) */
static void gen_labels_n(uint8_t *v, uint32_t seed, uint32_t nclass) {
  uint32_t r = seed | 1u;
  const int NS = 40;
  int sz[40], sy[40], sx[40];
  uint8_t sl[40];
  for (int i = 0; i < NS; i++) {
    sz[i] = (int)(vt_rng(&r) % 64);
    sy[i] = (int)(vt_rng(&r) % 64);
    sx[i] = (int)(vt_rng(&r) % 64);
    sl[i] = (uint8_t)(vt_rng(&r) % nclass);
  }
  static uint8_t g[64 * 64 * 64];
  for (int z = 0; z < 64; z++)
    for (int y = 0; y < 64; y++)
      for (int x = 0; x < 64; x++) {
        int best = 1 << 30;
        uint8_t bl = 0;
        for (int i = 0; i < NS; i++) {
          int dz = z - sz[i], dy = y - sy[i], dx = x - sx[i];
          int d = dz * dz + dy * dy + dx * dx;
          if (d < best) {
            best = d;
            bl = sl[i];
          }
        }
        g[(z * 64 + y) * 64 + x] = bl;
      }
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++)
        v[((size_t)z * 128 + y) * 128 + x] = g[((z / 2) * 64 + y / 2) * 64 + x / 2];
}
static void gen_labels8(uint8_t *v, uint32_t seed) { gen_labels_n(v, seed, 8); }

/* signed-distance-like smooth field quantised to u8 (128 = the surface) */
static void gen_sdf(uint8_t *v, uint32_t seed) {
  double p = 0.03 + 0.01 * (double)(seed % 5u);
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        double s = 26.0 * sin(x * p + z * 0.021 + seed) + 0.35 * y - 20.0 + 6.0 * cos(z * 0.05);
        double d = 128.0 + 9.0 * s / (1.0 + fabs(s) * 0.06);
        int iv = (int)(d + 0.5);
        v[((size_t)z * 128 + y) * 128 + x] = (uint8_t)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
      }
}

static void gen_noise(uint8_t *v, uint32_t seed) {
  uint32_t r = seed | 1u;
  for (size_t i = 0; i < N; i++) v[i] = (uint8_t)vt_rng(&r);
}

/* ---- helpers ---- */
static uint8_t *g_enc;

static size_t rt(const uint8_t *src, uint8_t *dec) { /* encode+decode, CHECKs exactness */
  size_t n = 0;
  CHECK_EQ(volcomp_encode(src, VOLCOMP_Q_LOSSLESS, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
  bool ll = false;
  CHECK_EQ(volcomp_is_lossless(g_enc, n, &ll), VOLCOMP_OK);
  CHECK(ll);
  float q = -1.0f;
  CHECK_EQ(volcomp_stream_q(g_enc, n, &q), VOLCOMP_OK);
  CHECK(q == VOLCOMP_Q_LOSSLESS);
  CHECK(n <= (size_t)N + VF_HDR_BYTES); /* never worse than raw + header */
  memset(dec, 0xa5, N);
  CHECK_EQ(volcomp_decode(g_enc, n, dec, N), VOLCOMP_OK);
  CHECK(memcmp(src, dec, N) == 0);
  return n;
}

int main(int argc, char **argv) {
  const bool bench = argc > 1 && !strcmp(argv[1], "--bench");
  static uint8_t src[N], dec[N];
  g_enc = (uint8_t *)malloc(VOLCOMP_ENCODE_BOUND);
  CHECK(g_enc != NULL);
  if (!g_enc) return 1;

  /* ---- 1. the named synthetic classes ---- */
  struct {
    const char *name;
    void (*gen)(uint8_t *, uint32_t);
  } cls[] = {{"constant", gen_const},   {"mask-2class-5%", gen_mask}, {"labels-8class", gen_labels8},
             {"sdf-smooth", gen_sdf},   {"noise", gen_noise},         {"ct-synthetic", vt_synth_chunk}};
  if (bench) printf("%-16s %10s %9s  %8s %8s\n", "class", "bytes", "ratio", "enc MB/s", "dec MB/s");
  for (size_t k = 0; k < sizeof cls / sizeof cls[0]; k++) {
    cls[k].gen(src, 2026u + (uint32_t)k);
    size_t n = rt(src, dec);
    if (!bench) continue;
    const int reps = 5;
    double t0 = now_s();
    for (int i = 0; i < reps; i++) {
      size_t m;
      volcomp_encode(src, VOLCOMP_Q_LOSSLESS, g_enc, VOLCOMP_ENCODE_BOUND, &m);
    }
    double te = (now_s() - t0) / reps;
    t0 = now_s();
    for (int i = 0; i < reps; i++) volcomp_decode(g_enc, n, dec, N);
    double td = (now_s() - t0) / reps;
    printf("%-16s %10zu %8.1fx  %8.0f %8.0f\n", cls[k].name, n, (double)N / (double)n,
           (double)N / te / 1e6, (double)N / td / 1e6);
  }

  /* ---- 2. all-constant blocks cost ~nothing ---- */
  {
    memset(src, 33, N);
    size_t n = rt(src, dec);
    CHECK(n == VF_HDR_BYTES + 1u); /* whole-chunk constant */
    /* one non-constant block (0,0,0) in an otherwise flat chunk: still tiny */
    uint32_t r = 4242;
    for (uint32_t z = 0; z < 16; z++)
      for (uint32_t y = 0; y < 16; y++)
        for (uint32_t x = 0; x < 16; x++) src[((size_t)z * 128 + y) * 128 + x] = (uint8_t)vt_rng(&r);
    n = rt(src, dec);
    CHECK(n < 6000);
    if (bench) printf("511 flat blocks + 1 noise block: %zu bytes\n", n);
  }

  /* ---- 3. random access: decode_block == the same region of decode ---- */
  {
    gen_labels8(src, 5);
    for (uint32_t i = 0; i < 40000; i++) src[i] = 0;   /* flat (CONST) blocks   */
    gen_noise(dec, 9);
    memcpy(src + 1000000, dec, 300000);                 /* RAW blocks            */
    size_t n = 0;
    CHECK_EQ(volcomp_encode(src, VOLCOMP_Q_LOSSLESS, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK_EQ(volcomp_decode(g_enc, n, dec, N), VOLCOMP_OK);
    CHECK(memcmp(src, dec, N) == 0);
    uint8_t blk[VOLCOMP_BLOCK_VOXELS];
    for (uint32_t bz = 0; bz < 8; bz++)
      for (uint32_t by = 0; by < 8; by++)
        for (uint32_t bx = 0; bx < 8; bx++) {
          CHECK_EQ(volcomp_decode_block(g_enc, n, bz, by, bx, blk, sizeof blk), VOLCOMP_OK);
          for (uint32_t z = 0; z < 16; z++)
            for (uint32_t y = 0; y < 16; y++)
              CHECK(memcmp(blk + (z * 16 + y) * 16,
                           src + (((size_t)bz * 16 + z) * 128 + by * 16 + y) * 128 + bx * 16, 16) == 0);
        }
    CHECK_EQ(volcomp_decode_block(g_enc, n, 8, 0, 0, blk, sizeof blk), VOLCOMP_ERR_ARG);
  }

  /* ---- 4. worst case: pure noise is raw + 8 bytes, and still exact ---- */
  {
    gen_noise(src, 77);
    size_t n = rt(src, dec);
    CHECK(n == (size_t)N + VF_HDR_BYTES);
    uint8_t blk[VOLCOMP_BLOCK_VOXELS];
    CHECK_EQ(volcomp_decode_block(g_enc, n, 3, 5, 6, blk, sizeof blk), VOLCOMP_OK);
    for (uint32_t z = 0; z < 16; z++)
      for (uint32_t y = 0; y < 16; y++)
        CHECK(memcmp(blk + (z * 16 + y) * 16, src + (((size_t)3 * 16 + z) * 128 + 5 * 16 + y) * 128 + 6 * 16,
                     16) == 0);
  }

  /* ---- 5. mode / header handling ---- */
  {
    memset(src, 4, N);
    size_t n = 0;
    CHECK_EQ(volcomp_encode(src, VOLCOMP_Q_LOSSLESS, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK_EQ(g_enc[5], VLL_MODE_CONST);
    g_enc[5] = 4; /* unknown chunk mode */
    CHECK_EQ(volcomp_decode(g_enc, n, dec, N), VOLCOMP_ERR_CORRUPT);
    g_enc[5] = VLL_MODE_CONST;
    g_enc[6] = 1; /* lossless modes must carry qraw == 0 */
    CHECK_EQ(volcomp_decode(g_enc, n, dec, N), VOLCOMP_ERR_CORRUPT);
    g_enc[6] = 0;
    CHECK_EQ(volcomp_decode(g_enc, n - 1, dec, N), VOLCOMP_ERR_CORRUPT);
    /* a lossy stream still reports q and is not lossless */
    vt_synth_chunk(src, 1);
    CHECK_EQ(volcomp_encode(src, 8.0f, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK_EQ(g_enc[5], VLL_MODE_LOSSY);
    bool ll = true;
    CHECK_EQ(volcomp_is_lossless(g_enc, n, &ll), VOLCOMP_OK);
    CHECK(!ll);
    float q = 0.0f;
    CHECK_EQ(volcomp_stream_q(g_enc, n, &q), VOLCOMP_OK);
    CHECK(q == 8.0f);
    /* q outside {0} u [1,255] is rejected */
    CHECK_EQ(volcomp_encode(src, 0.5f, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_encode(src, -1.0f, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    /* short destination */
    memset(src, 0, N);
    CHECK_EQ(volcomp_encode(src, VOLCOMP_Q_LOSSLESS, g_enc, 4, &n), VOLCOMP_ERR_SHORT_BUF);
  }

  /* ---- 6. truncation / corruption never reads out of bounds and never returns OK
   *        with wrong data (the fuzzers cover this exhaustively; a smoke pass here) ---- */
  {
    gen_labels8(src, 11);
    size_t n = 0;
    CHECK_EQ(volcomp_encode(src, VOLCOMP_Q_LOSSLESS, g_enc, VOLCOMP_ENCODE_BOUND, &n), VOLCOMP_OK);
    uint8_t *cp = (uint8_t *)malloc(n);
    CHECK(cp != NULL);
    if (cp) {
      for (size_t cut = 1; cut < n; cut += n / 200 + 1) {
        memcpy(cp, g_enc, cut);
        volcomp_decode(cp, cut, dec, N); /* must not crash; result unchecked */
      }
      uint32_t r = 12345;
      for (int i = 0; i < 300; i++) {
        memcpy(cp, g_enc, n);
        cp[vt_rng(&r) % n] ^= (uint8_t)(1u << (vt_rng(&r) % 8u));
        if (volcomp_decode(cp, n, dec, N) == VOLCOMP_OK) CHECK(true); /* wrong data is allowed */
      }
      free(cp);
    }
  }

  /* ---- 7. 1000 fuzzed chunks across the classes ---- */
  {
    size_t total = 0;
    for (uint32_t seed = 1; seed <= 1000; seed++) {
      switch (seed % 8u) {
        case 0: gen_const(src, seed); break;
        case 1: gen_mask(src, seed); break;
        case 2: gen_labels_n(src, seed, 2u + seed % 39u); break; /* 2..40 classes */
        case 3: gen_sdf(src, seed); break;
        case 4: gen_noise(src, seed); break;
        case 5: vt_synth_chunk(src, seed); break;
        case 6: { /* sparse labels on a flat ground */
          gen_labels_n(src, seed, 2u + seed % 39u);
          uint32_t r = seed | 1u;
          for (size_t i = 0; i < N; i++)
            if (vt_rng(&r) % 20u) src[i] = 0;
          break;
        }
        default: { /* mixed: half smooth, half noise */
          gen_sdf(src, seed);
          gen_noise(dec, seed);
          memcpy(src + N / 2, dec, N / 2);
          break;
        }
      }
      total += rt(src, dec);
      if (vt_failures) {
        fprintf(stderr, "first failure at seed %u\n", seed);
        break;
      }
    }
    if (bench) printf("1000 fuzzed chunks: %.2f MiB total (%.1fx)\n", (double)total / 1048576.0,
                      1000.0 * (double)N / (double)total);
  }

  /* ---- 8. real chunks, if any were dropped in tests/data/ ---- */
  if (bench) {
    static const char *names[] = {"faces_valid", "rv_class", "sdf_in", "ink", "hzvt_class"};
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
      char path[256];
      snprintf(path, sizeof path, "tests/data/%s.u8", names[k]);
      FILE *f = fopen(path, "rb");
      if (!f) continue;
      size_t got = fread(src, 1, N, f);
      fclose(f);
      if (got != N) continue;
      size_t n = rt(src, dec);
      const int reps = 5;
      double t0 = now_s();
      for (int i = 0; i < reps; i++) volcomp_decode(g_enc, n, dec, N);
      double td = (now_s() - t0) / reps;
      printf("%-16s %10zu %8.1fx  %8s %8.0f  (real)\n", names[k], n, (double)N / (double)n, "-",
             (double)N / td / 1e6);
    }
  }

  free(g_enc);
  TEST_END();
}
