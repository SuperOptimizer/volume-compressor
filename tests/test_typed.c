/* Typed chunks (modes 7 and 8, spec/format.md §14): exact round trips for every
 * dtype on smooth, noisy, constant and extreme data and on float32 specials; the
 * RAW and CONST forms; the mode-8 error bound across q, including the integer
 * rounding and clamping and the float32 step; block decode == full decode; the
 * frozen golden streams; corrupt-stream and argument rejection; and a compression
 * table for a smooth winding-like u16 field (--bench adds throughput, --regen
 * rewrites the goldens). */
#include "../volcomp.h"
#include "check.h"

#include <time.h>

#define N VOLCOMP_CHUNK_VOXELS

static const volcomp_dtype DTS[] = {VOLCOMP_DTYPE_U16, VOLCOMP_DTYPE_I16, VOLCOMP_DTYPE_U32, VOLCOMP_DTYPE_I32,
                                    VOLCOMP_DTYPE_F32};
#define NDT (sizeof DTS / sizeof DTS[0])
static const char *dt_name(volcomp_dtype dt) {
  static const char *names[] = {"u8", "u16", "i16", "u32", "i32", "f32"};
  return (unsigned)dt < 6u ? names[dt] : "?";
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static uint64_t fnv1a(const void *p, size_t n) {
  const uint8_t *b = (const uint8_t *)p;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}

/* ---- element access (host order) ---- */
static double getd(const void *buf, volcomp_dtype dt, size_t i) {
  const uint8_t *p = (const uint8_t *)buf;
  switch (dt) {
  case VOLCOMP_DTYPE_U16: { uint16_t v; memcpy(&v, p + 2 * i, 2); return v; }
  case VOLCOMP_DTYPE_I16: { int16_t v; memcpy(&v, p + 2 * i, 2); return v; }
  case VOLCOMP_DTYPE_U32: { uint32_t v; memcpy(&v, p + 4 * i, 4); return v; }
  case VOLCOMP_DTYPE_I32: { int32_t v; memcpy(&v, p + 4 * i, 4); return v; }
  case VOLCOMP_DTYPE_F32: { float v; memcpy(&v, p + 4 * i, 4); return (double)v; }
  default: return p[i];
  }
}
static void type_range(volcomp_dtype dt, double *lo, double *hi) {
  switch (dt) {
  case VOLCOMP_DTYPE_U16: *lo = 0, *hi = 65535; break;
  case VOLCOMP_DTYPE_I16: *lo = -32768, *hi = 32767; break;
  case VOLCOMP_DTYPE_U32: *lo = 0, *hi = 4294967295.0; break;
  case VOLCOMP_DTYPE_I32: *lo = -2147483648.0, *hi = 2147483647.0; break;
  default: *lo = -3.4e38, *hi = 3.4e38; break;
  }
}
/* integers: round and clamp to the type; f32: the nearest float */
static void setd(void *buf, volcomp_dtype dt, size_t i, double v) {
  uint8_t *p = (uint8_t *)buf;
  if (dt == VOLCOMP_DTYPE_F32) {
    float f = (float)v;
    memcpy(p + 4 * i, &f, 4);
    return;
  }
  double lo, hi;
  type_range(dt, &lo, &hi);
  v = round(v);
  v = v < lo ? lo : (v > hi ? hi : v);
  const int64_t k = (int64_t)v;
  if (volcomp_dtype_size(dt) == 2) {
    uint16_t t = (uint16_t)k;
    memcpy(p + 2 * i, &t, 2);
  } else {
    uint32_t t = (uint32_t)k;
    memcpy(p + 4 * i, &t, 4);
  }
}

/* ---- generators (128^3 elements of dt) ---- */
/* a winding-like field: W = a radial ramp around an axis outside the chunk (one
 * turn per 30 voxels) plus waves along z and around the axis; each dtype stores a
 * scaled copy (u16: W * 256, the ufsm layout) */
static double wind_w(uint32_t z, uint32_t y, uint32_t x, double tilt) {
  const double cy = -60.0 + tilt * z, cx = 40.0, dy = y - cy, dx = x - cx;
  const double r = sqrt(dy * dy + dx * dx), th = atan2(dy, dx);
  return 4.0 + r / 30.0 + 0.15 * sin(0.09 * z + 3.0 * th) + 0.05 * sin(0.21 * x + 0.13 * y);
}
static void gen_wind(void *v, volcomp_dtype dt, double tilt) {
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        const double w = wind_w(z, y, x, tilt);
        double s;
        switch (dt) {
        case VOLCOMP_DTYPE_U16: s = w * 256.0; break;
        case VOLCOMP_DTYPE_I16: s = w * 256.0 - 20000.0; break;
        case VOLCOMP_DTYPE_U32: s = w * 1048576.0; break;
        case VOLCOMP_DTYPE_I32: s = -w * 1048576.0; break;
        default: s = w; break; /* f32: the float itself, every mantissa bit live */
        }
        setd(v, dt, ((size_t)z * 128 + y) * 128 + x, s);
      }
}
static void gen_noise(void *v, volcomp_dtype dt, uint32_t seed) {
  uint32_t r = seed | 1u;
  uint8_t *p = (uint8_t *)v;
  for (size_t i = 0; i < N * volcomp_dtype_size(dt); i++) p[i] = (uint8_t)vt_rng(&r);
}
/* the type's extremes: min / max / 0 in blocks, in alternating elements and as a
 * sprinkle over a ramp, so every residual magnitude up to 2^W - 1 occurs */
static void gen_extreme(void *v, volcomp_dtype dt, uint32_t seed) {
  double lo, hi;
  type_range(dt, &lo, &hi);
  if (dt == VOLCOMP_DTYPE_F32) lo = -3.4028234663852886e38, hi = 3.4028234663852886e38;
  uint32_t r = seed | 1u;
  for (size_t i = 0; i < N; i++) {
    const size_t z = i >> 14;
    double s;
    if (z < 32) s = (z & 4) ? hi : lo;                       /* whole blocks of min, max */
    else if (z < 64) s = (i & 1) ? hi : lo;                  /* alternating min / max */
    else if (z < 96) s = (vt_rng(&r) % 3u == 0) ? ((i & 2) ? hi : lo) : (double)(i & 1023); /* sprinkle */
    else s = (vt_rng(&r) & 1u) ? hi : 0.0;
    setd(v, dt, i, s);
  }
}
/* float32 specials: NaN payloads (quiet, signalling, negative), +-0, +-inf,
 * subnormals, FLT_MAX / FLT_MIN, between ordinary floats */
static void gen_f32_special(void *v, uint32_t seed) {
  static const uint32_t sp[] = {0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x7fbfffffu, 0xff800123u, 0x7fffffffu,
                                0xffffffffu, 0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x00000001u,
                                0x80000001u, 0x007fffffu, 0x807fffffu, 0x7f7fffffu, 0xff7fffffu, 0x00800000u,
                                0x80800000u, 0x3f800000u};
  uint32_t r = seed | 1u;
  uint8_t *p = (uint8_t *)v;
  for (size_t i = 0; i < N; i++) {
    uint32_t b;
    const uint32_t k = vt_rng(&r);
    if ((i >> 12) % 3u == 0) b = sp[k % (sizeof sp / sizeof sp[0])];         /* dense specials */
    else if ((i >> 12) % 3u == 1) b = (k & 15u) ? 0x3f800000u + (uint32_t)(i & 255u) : sp[k % 20u]; /* sparse */
    else b = k;                                                               /* random bit patterns */
    memcpy(p + 4 * i, &b, 4);
  }
}

/* ---- helpers ---- */
static uint8_t *g_enc;
static void *g_dec, *g_blk;

/* the blocks of a decode equal the region of the full decode, for nb blocks (all if 512) */
static void check_blocks(const uint8_t *enc, size_t n, volcomp_dtype dt, const void *full, uint32_t nb, uint32_t seed) {
  const size_t es = volcomp_dtype_size(dt);
  uint32_t r = seed | 1u;
  for (uint32_t k = 0; k < nb; k++) {
    const uint32_t bi = nb >= 512 ? k : vt_rng(&r) % 512u, bz = bi >> 6, by = (bi >> 3) & 7u, bx = bi & 7u;
    CHECK_EQ(volcomp_decode_block_typed(enc, n, bz, by, bx, g_blk, VOLCOMP_BLOCK_VOXELS * es, dt), VOLCOMP_OK);
    for (uint32_t z = 0; z < 16; z++)
      for (uint32_t y = 0; y < 16; y++)
        CHECK(memcmp((const uint8_t *)g_blk + (z * 16 + y) * 16 * es,
                     (const uint8_t *)full + ((((size_t)bz * 16 + z) * 128 + by * 16 + y) * 128 + bx * 16) * es,
                     16 * es) == 0);
  }
}
/* exact round trip at q = 0: mode 7, decodes bit for bit, within the bound */
static size_t rt_exact(const void *src, volcomp_dtype dt, uint32_t nblocks) {
  const size_t es = volcomp_dtype_size(dt);
  size_t n = 0;
  CHECK_EQ(volcomp_encode_typed(src, dt, VOLCOMP_Q_LOSSLESS, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
  CHECK(n <= volcomp_typed_encode_bound(dt));
  CHECK_EQ(g_enc[5], VLL_MODE_TYPED);
  bool ll = false;
  CHECK_EQ(volcomp_is_lossless(g_enc, n, &ll), VOLCOMP_OK);
  CHECK(ll);
  float q = -1.0f;
  CHECK_EQ(volcomp_stream_q(g_enc, n, &q), VOLCOMP_OK);
  CHECK(q == 0.0f);
  volcomp_dtype got = VOLCOMP_DTYPE_U8;
  CHECK_EQ(volcomp_chunk_dtype(g_enc, n, &got), VOLCOMP_OK);
  CHECK_EQ(got, dt);
  memset(g_dec, 0xa5, N * es);
  CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * es, dt), VOLCOMP_OK);
  CHECK(memcmp(src, g_dec, N * es) == 0);
  if (nblocks) check_blocks(g_enc, n, dt, g_dec, nblocks, (uint32_t)n);
  return n;
}
/* mode-8 round trip at q: |x - x_hat| <= q/2 everywhere (bit-exact if it fell back
 * to mode 7); returns the stream length, *mode the header's mode */
static size_t rt_bounded(const void *src, volcomp_dtype dt, float q, uint32_t *mode, double *maxerr) {
  const size_t es = volcomp_dtype_size(dt);
  size_t n = 0;
  CHECK_EQ(volcomp_encode_typed(src, dt, q, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
  CHECK(n <= volcomp_typed_encode_bound(dt));
  *mode = g_enc[5];
  CHECK(*mode == VLL_MODE_TYPED || *mode == VLL_MODE_TYPED_Q);
  float sq = -1.0f;
  CHECK_EQ(volcomp_stream_q(g_enc, n, &sq), VOLCOMP_OK);
  bool ll = true;
  CHECK_EQ(volcomp_is_lossless(g_enc, n, &ll), VOLCOMP_OK);
  CHECK(ll == (*mode == VLL_MODE_TYPED));
  if (*mode == VLL_MODE_TYPED_Q) {
    CHECK(sq > 0.0f && sq <= q); /* the stored step never exceeds the requested one */
    if (dt != VOLCOMP_DTYPE_F32) CHECK(sq == floorf(q));
  } else {
    CHECK(sq == 0.0f);
  }
  CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * es, dt), VOLCOMP_OK);
  double mx = 0.0;
  const double half = 0.5 * (double)q;
  size_t bad = 0;
  for (size_t i = 0; i < N; i++) {
    const double x = getd(src, dt, i), xh = getd(g_dec, dt, i);
    if (x != x) { /* NaN: only mode 7 can carry one, bit for bit */
      CHECK(*mode == VLL_MODE_TYPED);
      continue;
    }
    const double e = x == xh ? 0.0 : fabs(x - xh); /* inf == inf */
    mx = e > mx ? e : mx;
    bad += !(e <= half);
  }
  if (bad) fprintf(stderr, "  %s q=%g: %zu elements beyond q/2 (max err %g)\n", dt_name(dt), (double)q, bad, mx);
  CHECK_EQ(bad, 0);
  if (*mode == VLL_MODE_TYPED) CHECK(memcmp(src, g_dec, N * es) == 0);
  *maxerr = mx;
  return n;
}

/* ---- golden streams ---- */
static uint8_t *readf(const char *p, size_t *n) {
  FILE *f = fopen(p, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  uint8_t *b = (uint8_t *)malloc((size_t)sz);
  *n = b ? fread(b, 1, (size_t)sz, f) : 0;
  fclose(f);
  return b;
}
static void writef(const char *p, const void *b, size_t n) {
  FILE *f = fopen(p, "wb");
  CHECK(f && fwrite(b, 1, n, f) == n);
  if (f) fclose(f);
}
/* golden i16 source: every block mode -- CONST (flat), SPARSE (steps), DENSE
 * (the winding field, fine and coarse), RAW (one noise block) -- and negative values */
static void gen_golden_i16(int16_t *v) {
  uint32_t r = 77;
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        const size_t i = ((size_t)z * 128 + y) * 128 + x;
        double s;
        if (z < 32) s = -1200.0;                                                 /* flat */
        else if (z < 64) s = 400.0 * floor(wind_w(z, y, x, 0.0) * 2.0) - 3000.0; /* steps */
        else if (z < 80) s = wind_w(z, y, x, 0.2) * 256.0 - 1500.0;             /* smooth */
        else s = wind_w(z, y, x, 0.2) * 24.0 - 100.0;                            /* smooth, coarse */
        if (z >= 112 && y < 16 && x < 16) s = (double)(int16_t)vt_rng(&r);       /* one noise block */
        v[i] = (int16_t)lround(s);
      }
}
/* golden f32 source: a constant three quarters (CONST blocks of one k) over a
 * quarter of the winding field as floats */
static void gen_golden_f32(float *v) {
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++)
        v[((size_t)z * 128 + y) * 128 + x] = z < 96 ? 1.5f : (float)wind_w(z, y, x, 0.3);
}

int main(int argc, char **argv) {
  const bool bench = argc > 1 && !strcmp(argv[1], "--bench");
  const bool regen = argc > 1 && !strcmp(argv[1], "--regen");
  g_enc = (uint8_t *)malloc(VOLCOMP_TYPED_ENCODE_BOUND);
  g_dec = malloc(N * 4);
  g_blk = malloc(VOLCOMP_BLOCK_VOXELS * 4);
  uint8_t *src = (uint8_t *)malloc(N * 4);
  CHECK(g_enc && g_dec && g_blk && src);
  if (!g_enc || !g_dec || !g_blk || !src) return 1;

  /* ---- 1. the API's small pieces ---- */
  CHECK_EQ(volcomp_dtype_size(VOLCOMP_DTYPE_U8), 1);
  CHECK_EQ(volcomp_dtype_size(VOLCOMP_DTYPE_U16), 2);
  CHECK_EQ(volcomp_dtype_size(VOLCOMP_DTYPE_I16), 2);
  CHECK_EQ(volcomp_dtype_size(VOLCOMP_DTYPE_U32), 4);
  CHECK_EQ(volcomp_dtype_size(VOLCOMP_DTYPE_I32), 4);
  CHECK_EQ(volcomp_dtype_size(VOLCOMP_DTYPE_F32), 4);
  CHECK_EQ(volcomp_dtype_size((volcomp_dtype)6), 0);
  CHECK_EQ(volcomp_typed_encode_bound(VOLCOMP_DTYPE_U16), 10 + 2 * (size_t)N);
  CHECK_EQ(volcomp_typed_encode_bound(VOLCOMP_DTYPE_F32), VOLCOMP_TYPED_ENCODE_BOUND);
  CHECK_EQ(volcomp_typed_encode_bound(VOLCOMP_DTYPE_U8), 8 + (size_t)N);
  CHECK_EQ(volcomp_typed_encode_bound((volcomp_dtype)9), 0);
  CHECK_EQ(VOLCOMP_FORMAT_REVISION, 5);
  CHECK(!strcmp(VOLCOMP_VERSION_STRING, "1.4.0"));

  /* ---- 2. exact round trips, every dtype: smooth, noise (RAW), constant, extremes ---- */
  for (size_t d = 0; d < NDT; d++) {
    const volcomp_dtype dt = DTS[d];
    const size_t es = volcomp_dtype_size(dt);
    gen_wind(src, dt, 0.3);
    size_t n = rt_exact(src, dt, 512);
    CHECK_EQ(g_enc[9], VT_FORM_CODED);
    /* a smooth field compresses: W * 256 in 16 bits by ~11x; the 32-bit copies keep 20
     * fractional bits of W (float32: every mantissa bit), so about 2x is what is there */
    CHECK(es == 2 ? n * 8 < N * es : n * 3 < N * es * 2);
    printf("%s winding field, exact: %8zu bytes (%.2fx)\n", dt_name(dt), n, (double)(N * es) / (double)n);

    gen_noise(src, dt, 11u + (uint32_t)d); /* incompressible: the RAW form, raw + 10 bytes */
    n = rt_exact(src, dt, 16);
    CHECK_EQ(g_enc[9], VT_FORM_RAW);
    CHECK_EQ(n, volcomp_typed_encode_bound(dt));

    for (int k = 0; k < 3; k++) { /* constants, including the type's extremes: CONST, 10 + size bytes */
      double lo, hi;
      type_range(dt, &lo, &hi);
      const double c = k == 0 ? 0.0 : (k == 1 ? lo : hi);
      for (size_t i = 0; i < N; i++) setd(src, dt, i, c);
      n = rt_exact(src, dt, 4);
      CHECK_EQ(g_enc[9], VT_FORM_CONST);
      CHECK_EQ(n, 10 + es);
    }

    gen_extreme(src, dt, 5u + (uint32_t)d);
    n = rt_exact(src, dt, 512);
    CHECK(n <= volcomp_typed_encode_bound(dt));

    /* a smooth chunk with noise in one half: RAW blocks inside a CODED chunk */
    gen_wind(src, dt, 0.0);
    gen_noise(g_dec, dt, 99);
    memcpy(src + N / 2 * es, (uint8_t *)g_dec + N / 2 * es, N / 2 * es);
    n = rt_exact(src, dt, 512);
    CHECK_EQ(g_enc[9], VT_FORM_CODED);
  }

  /* ---- 3. float32 specials survive mode 7 bit for bit; q > 0 falls back to it ---- */
  {
    gen_f32_special(src, 3);
    size_t n = rt_exact(src, VOLCOMP_DTYPE_F32, 512);
    uint32_t mode;
    double mx;
    rt_bounded(src, VOLCOMP_DTYPE_F32, 0.25f, &mode, &mx); /* NaN present: mode 7, exact */
    CHECK_EQ(mode, VLL_MODE_TYPED);
    /* one NaN in an otherwise smooth chunk is enough */
    gen_wind(src, VOLCOMP_DTYPE_F32, 0.0);
    const uint32_t nan = 0x7fc01234u, inf = 0x7f800000u;
    memcpy(src + 4 * 777777, &nan, 4);
    rt_bounded(src, VOLCOMP_DTYPE_F32, 0.01f, &mode, &mx);
    CHECK_EQ(mode, VLL_MODE_TYPED);
    gen_wind(src, VOLCOMP_DTYPE_F32, 0.0);
    memcpy(src + 4 * 12345, &inf, 4);
    rt_bounded(src, VOLCOMP_DTYPE_F32, 0.01f, &mode, &mx);
    CHECK_EQ(mode, VLL_MODE_TYPED);
    /* -0 vs +0 and NaN payloads are distinct bit patterns: a chunk of only those */
    for (size_t i = 0; i < N; i++) {
      const uint32_t b = (i % 3u == 0) ? 0x80000000u : ((i % 3u == 1) ? 0u : 0x7fc00000u + (uint32_t)(i & 7u));
      memcpy(src + 4 * i, &b, 4);
    }
    n = rt_exact(src, VOLCOMP_DTYPE_F32, 8);
    printf("f32 specials: exact; -0/+0/NaN-payload chunk %zu bytes\n", n);
  }

  /* ---- 4. the mode-8 error bound across q ---- */
  {
    static const float qs_int[] = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 7.0f, 10.0f, 64.0f, 255.9f, 1000.0f, 65536.0f, 1e6f,
                                   3e9f, 1e12f};
    size_t cases = 0;
    for (size_t d = 0; d < NDT - 1; d++) { /* the integer dtypes */
      const volcomp_dtype dt = DTS[d];
      for (int f = 0; f < 3; f++) {
        if (f == 0) gen_wind(src, dt, 0.4);
        else if (f == 1) gen_extreme(src, dt, 21);
        else gen_noise(src, dt, 31);
        for (size_t k = 0; k < sizeof qs_int / sizeof qs_int[0]; k++) {
          uint32_t mode;
          double mx;
          rt_bounded(src, dt, qs_int[k], &mode, &mx);
          cases++;
          if (qs_int[k] < 2.0f) CHECK_EQ(mode, VLL_MODE_TYPED); /* floor(q) < 2: exact */
          /* u32 at the top of its range with a step of 2: k = 2^31 leaves i32, so mode 7 */
          if (dt == VOLCOMP_DTYPE_U32 && f == 1 && floorf(qs_int[k]) == 2.0f) CHECK_EQ(mode, VLL_MODE_TYPED);
          if (f == 0 && qs_int[k] >= 2.0f && qs_int[k] <= 1000.0f) CHECK_EQ(mode, VLL_MODE_TYPED_Q);
          if (f == 0 && qs_int[k] >= 2.0f && qs_int[k] <= 64.0f) /* the bound is reached, not just kept */
            CHECK(mx >= 0.5 * floor((double)qs_int[k]) - 1.0);
        }
      }
    }
    /* clamping at the ends of the range keeps the bound: u16 near 65535 and 0, i16
     * near -32768, steps that do not divide the range */
    for (size_t i = 0; i < N; i++) setd(src, VOLCOMP_DTYPE_U16, i, (i & 1) ? 65535.0 - (double)(i % 7) : (double)(i % 5));
    for (float q = 2.0f; q <= 40.0f; q += 3.0f) {
      uint32_t mode;
      double mx;
      rt_bounded(src, VOLCOMP_DTYPE_U16, q, &mode, &mx);
      CHECK_EQ(mode, VLL_MODE_TYPED_Q);
      cases++;
    }
    for (size_t i = 0; i < N; i++) setd(src, VOLCOMP_DTYPE_I16, i, (i & 1) ? -32768.0 + (double)(i % 9) : 32767.0 - (double)(i % 11));
    for (float q = 2.0f; q <= 40.0f; q += 3.0f) {
      uint32_t mode;
      double mx;
      rt_bounded(src, VOLCOMP_DTYPE_I16, q, &mode, &mx);
      cases++;
    }
    /* f32: random values at many scales against q, including steps of a few hundred
     * ulps (where the binary32 rounding of k q alone would miss q/2) and ranges near
     * and beyond 2^22 q (mode 7) */
    uint32_t r = 4321;
    static const double scales[] = {0.3, 1.0, 100.0, 1e4, 1e5, 1e6, 3e6, 1e7, 1e8};
    for (size_t s = 0; s < sizeof scales / sizeof scales[0]; s++) {
      for (int rep = 0; rep < 2; rep++) {
        const float q = (float)ldexp(1.0 + (double)(vt_rng(&r) % 100000u) / 100000.0, (int)(vt_rng(&r) % 30u) - 15);
        const double a = scales[s] * (double)q;
        for (size_t i = 0; i < N; i++) {
          const double u = (double)vt_rng(&r) / 4294967296.0 * 2.0 - 1.0;
          setd(src, VOLCOMP_DTYPE_F32, i, (i >> 14) & 1 ? a * u : a * sin((double)i * 1e-4) + 0.01 * a * u);
        }
        uint32_t mode;
        double mx;
        rt_bounded(src, VOLCOMP_DTYPE_F32, q, &mode, &mx);
        cases++;
        if (scales[s] <= 1e5) CHECK_EQ(mode, VLL_MODE_TYPED_Q);
        if (scales[s] >= 1e7) CHECK_EQ(mode, VLL_MODE_TYPED); /* beyond ~2^22 q: no step keeps the bound */
      }
    }
    /* f32 smooth field (W in 4..10) over q; below ~2 ulp(10) = 1.9e-6 no step keeps the bound */
    gen_wind(src, VOLCOMP_DTYPE_F32, 0.2);
    {
      uint32_t mode;
      double mx;
      rt_bounded(src, VOLCOMP_DTYPE_F32, 1e-6f, &mode, &mx);
      CHECK_EQ(mode, VLL_MODE_TYPED);
    }
    for (float q = 1e-5f; q < 10.0f; q *= 7.0f) {
      uint32_t mode;
      double mx;
      size_t n = rt_bounded(src, VOLCOMP_DTYPE_F32, q, &mode, &mx);
      CHECK_EQ(mode, VLL_MODE_TYPED_Q);
      check_blocks(g_enc, n, VOLCOMP_DTYPE_F32, g_dec, 8, (uint32_t)n);
      cases++;
    }
    /* a field mode 8 quantises to one k: the CONST form of mode 8, 18 bytes */
    gen_wind(src, VOLCOMP_DTYPE_U16, 0.0);
    {
      uint32_t mode;
      double mx;
      size_t n = rt_bounded(src, VOLCOMP_DTYPE_U16, 1e6f, &mode, &mx);
      CHECK_EQ(mode, VLL_MODE_TYPED_Q);
      CHECK_EQ(g_enc[9], VT_FORM_CONST);
      CHECK_EQ(n, 18);
      check_blocks(g_enc, n, VOLCOMP_DTYPE_U16, g_dec, 4, 1);
    }
    printf("mode-8 bound |x - x_hat| <= q/2 held in %zu chunk cases\n", cases);
  }

  /* ---- 5. the golden streams (frozen bitstreams; the typed coder has no float
   *         math except the mode-8 float32 step, which is IEEE-exact) ---- */
  {
    int16_t *g16 = (int16_t *)(void *)src;
    gen_golden_i16(g16);
    size_t n = 0;
    CHECK_EQ(volcomp_encode_typed(g16, VOLCOMP_DTYPE_I16, 0.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
    if (regen) {
      writef("tests/golden/typed_i16.volc", g_enc, n);
      printf("wrote tests/golden/typed_i16.volc (%zu bytes)\n", n);
    } else {
      size_t gn = 0;
      uint8_t *ge = readf("tests/golden/typed_i16.volc", &gn);
      CHECK(ge != NULL);
      if (ge) {
        CHECK_EQ(gn, n);
        CHECK(gn == n && memcmp(ge, g_enc, n) == 0);
        CHECK_EQ(volcomp_decode_typed(ge, gn, g_dec, N * 2, VOLCOMP_DTYPE_I16), VOLCOMP_OK);
        CHECK(memcmp(g_dec, g16, N * 2) == 0);
        printf("golden typed_i16 (mode 7): %zu bytes, identical to the checked-in stream: %s\n", n,
               gn == n && memcmp(ge, g_enc, n) == 0 ? "yes" : "NO");
        free(ge);
      }
    }
    gen_golden_f32((float *)(void *)src);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_F32, 0.004f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK_EQ(g_enc[5], VLL_MODE_TYPED_Q);
    if (regen) {
      writef("tests/golden/typed_f32_q.volc", g_enc, n);
      CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * 4, VOLCOMP_DTYPE_F32), VOLCOMP_OK);
      printf("wrote tests/golden/typed_f32_q.volc (%zu bytes), decode hash 0x%016llx\n", n,
             (unsigned long long)fnv1a(g_dec, N * 4));
    } else {
      size_t gn = 0;
      uint8_t *ge = readf("tests/golden/typed_f32_q.volc", &gn);
      CHECK(ge != NULL);
      if (ge) {
        CHECK_EQ(gn, n);
        CHECK(gn == n && memcmp(ge, g_enc, n) == 0);
        CHECK_EQ(volcomp_decode_typed(ge, gn, g_dec, N * 4, VOLCOMP_DTYPE_F32), VOLCOMP_OK);
        const uint64_t h = fnv1a(g_dec, N * 4);
        CHECK_EQ(h, 0x148f4c4a61a75858ull); /* frozen decode of the frozen stream */
        size_t bad = 0;
        for (size_t i = 0; i < N; i++)
          bad += !(fabs(getd(src, VOLCOMP_DTYPE_F32, i) - getd(g_dec, VOLCOMP_DTYPE_F32, i)) <= 0.5 * (double)0.004f);
        CHECK_EQ(bad, 0);
        printf("golden typed_f32_q (mode 8, q 0.004): %zu bytes, decode hash 0x%016llx\n", n, (unsigned long long)h);
        free(ge);
      }
    }
  }

  /* ---- 6. which call reads what: u8 calls refuse typed chunks, typed calls take u8 ones ---- */
  {
    gen_wind(src, VOLCOMP_DTYPE_U16, 0.0);
    size_t n = 0;
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, 0.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
    uint8_t *u8 = (uint8_t *)g_blk;
    CHECK_EQ(volcomp_decode(g_enc, n, (uint8_t *)g_dec, N), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_decode_block(g_enc, n, 1, 2, 3, u8, 4096), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_decode_smooth(g_enc, n, (uint8_t *)g_dec, N, 2.0f, 0), VOLCOMP_ERR_ARG);
    uint32_t dim;
    CHECK_EQ(volcomp_mask_info(g_enc, n, &dim), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_surface_info(g_enc, n, &dim, &dim), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * 4, VOLCOMP_DTYPE_I16), VOLCOMP_ERR_ARG); /* dtype mismatch */
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * 4, VOLCOMP_DTYPE_U8), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * 2 - 1, VOLCOMP_DTYPE_U16), VOLCOMP_ERR_SHORT_BUF);
    CHECK_EQ(volcomp_decode_block_typed(g_enc, n, 0, 0, 0, g_blk, 4096 * 2 - 1, VOLCOMP_DTYPE_U16),
             VOLCOMP_ERR_SHORT_BUF);
    CHECK_EQ(volcomp_decode_block_typed(g_enc, n, 8, 0, 0, g_blk, 4096 * 2, VOLCOMP_DTYPE_U16), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_decode_block_typed(g_enc, n, 0, 0, 0, g_blk, 4096 * 2, VOLCOMP_DTYPE_F32), VOLCOMP_ERR_ARG);
    /* a u8 chunk through the typed calls: U8 only, and then exactly volcomp_decode */
    uint8_t *s8 = src;
    vt_synth_chunk(s8, 7);
    CHECK_EQ(volcomp_encode(s8, 8.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
    volcomp_dtype got = VOLCOMP_DTYPE_F32;
    CHECK_EQ(volcomp_chunk_dtype(g_enc, n, &got), VOLCOMP_OK);
    CHECK_EQ(got, VOLCOMP_DTYPE_U8);
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N, VOLCOMP_DTYPE_U16), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N, VOLCOMP_DTYPE_U8), VOLCOMP_OK);
    uint8_t *ref = (uint8_t *)malloc(N);
    CHECK(ref != NULL);
    if (ref) {
      CHECK_EQ(volcomp_decode(g_enc, n, ref, N), VOLCOMP_OK);
      CHECK(memcmp(ref, g_dec, N) == 0);
      CHECK_EQ(volcomp_decode_block_typed(g_enc, n, 3, 4, 5, g_blk, 4096, VOLCOMP_DTYPE_U8), VOLCOMP_OK);
      CHECK(memcmp(g_blk, ref + ((size_t)3 * 16 * 128 + 4 * 16) * 128 + 5 * 16, 16) == 0);
      free(ref);
    }
    /* volcomp_encode_typed with U8 is the u8 lossless mode, q = 0 only */
    CHECK_EQ(volcomp_encode_typed(s8, VOLCOMP_DTYPE_U8, 0.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
    CHECK(g_enc[5] >= VLL_MODE_CODED && g_enc[5] <= VLL_MODE_CONST);
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N, VOLCOMP_DTYPE_U8), VOLCOMP_OK);
    CHECK(memcmp(g_dec, s8, N) == 0);
    CHECK_EQ(volcomp_encode_typed(s8, VOLCOMP_DTYPE_U8, 4.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    /* arguments */
    gen_wind(src, VOLCOMP_DTYPE_U16, 0.0);
    CHECK_EQ(volcomp_encode_typed(src, (volcomp_dtype)6, 0.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, -1.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, NAN, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, INFINITY, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n),
             VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_encode_typed(NULL, VOLCOMP_DTYPE_U16, 0.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_ERR_ARG);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, 0.0f, g_enc, 1000, &n), VOLCOMP_ERR_SHORT_BUF);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, 16.0f, g_enc, 1000, &n), VOLCOMP_ERR_SHORT_BUF);
    for (size_t i = 0; i < N; i++) setd(src, VOLCOMP_DTYPE_U16, i, 7.0);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, 0.0f, g_enc, 11, &n), VOLCOMP_ERR_SHORT_BUF);
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, 0.0f, g_enc, 12, &n), VOLCOMP_OK);
    CHECK_EQ(n, 12);
    CHECK_EQ(volcomp_chunk_dtype(g_enc, n, NULL), VOLCOMP_ERR_ARG);
  }

  /* ---- 7. corrupt streams: truncation, bad header bytes, bit flips ---- */
  {
    /* one chunk of each form, mode 7 and mode 8 */
    struct { volcomp_dtype dt; float q; int gen; } cs[] = {
        {VOLCOMP_DTYPE_U16, 0.0f, 0}, {VOLCOMP_DTYPE_F32, 0.0f, 0}, {VOLCOMP_DTYPE_I32, 16.0f, 0},
        {VOLCOMP_DTYPE_F32, 0.01f, 0}, {VOLCOMP_DTYPE_U16, 0.0f, 1}, {VOLCOMP_DTYPE_I16, 0.0f, 2},
        {VOLCOMP_DTYPE_U32, 1e9f, 0}};
    uint8_t *cp = (uint8_t *)malloc(VOLCOMP_TYPED_ENCODE_BOUND + 1);
    CHECK(cp != NULL);
    for (size_t c = 0; cp && c < sizeof cs / sizeof cs[0]; c++) {
      const volcomp_dtype dt = cs[c].dt;
      const size_t es = volcomp_dtype_size(dt);
      if (cs[c].gen == 0) gen_wind(src, dt, 0.1);
      else if (cs[c].gen == 1) gen_noise(src, dt, 5);
      else for (size_t i = 0; i < N; i++) setd(src, dt, i, -5.0);
      size_t n = 0;
      CHECK_EQ(volcomp_encode_typed(src, dt, cs[c].q, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
      const uint32_t hn = g_enc[5] == VLL_MODE_TYPED_Q ? 14u : 10u;
      /* every truncation is refused (exact accounting), and so is a trailing byte */
      size_t refused = 0, cuts = 0;
      for (size_t cut = 0; cut < n; cut += cut < 64 ? 1 : n / 97 + 1) {
        memcpy(cp, g_enc, cut);
        refused += volcomp_decode_typed(cp, cut, g_dec, N * es, dt) != VOLCOMP_OK;
        cuts++;
      }
      CHECK_EQ(refused, cuts);
      memcpy(cp, g_enc, n);
      cp[n] = 0;
      CHECK_EQ(volcomp_decode_typed(cp, n + 1, g_dec, N * es, dt), VOLCOMP_ERR_CORRUPT);
      volcomp_dtype got;
      /* the dtype byte: 0 and anything above 5 are corrupt */
      static const uint8_t bad_dt[] = {0, 6, 7, 128, 255};
      for (size_t k = 0; k < sizeof bad_dt; k++) {
        memcpy(cp, g_enc, n);
        cp[8] = bad_dt[k];
        CHECK_EQ(volcomp_decode_typed(cp, n, g_dec, N * 4, dt), VOLCOMP_ERR_CORRUPT);
        CHECK_EQ(volcomp_chunk_dtype(cp, n, &got), VOLCOMP_ERR_CORRUPT);
      }
      /* a valid dtype that is not this one: the payload no longer fits (or a mismatch) */
      memcpy(cp, g_enc, n);
      cp[8] = (uint8_t)(dt == VOLCOMP_DTYPE_U16 ? VOLCOMP_DTYPE_U32 : VOLCOMP_DTYPE_U16);
      CHECK(volcomp_decode_typed(cp, n, g_dec, N * 4, dt) != VOLCOMP_OK);
      /* the form byte */
      memcpy(cp, g_enc, n);
      cp[9] = hn == 14 ? 2 : 3; /* mode 8 has no RAW form */
      CHECK_EQ(volcomp_decode_typed(cp, n, g_dec, N * es, dt), VOLCOMP_ERR_CORRUPT);
      cp[9] = 0xff;
      CHECK_EQ(volcomp_decode_typed(cp, n, g_dec, N * es, dt), VOLCOMP_ERR_CORRUPT);
      /* q_raw must be 0 */
      memcpy(cp, g_enc, n);
      cp[7] = 1;
      CHECK_EQ(volcomp_decode_typed(cp, n, g_dec, N * es, dt), VOLCOMP_ERR_CORRUPT);
      /* mode 8's q: 0, negative, NaN, infinity; fractional or < 2 for an integer dtype */
      if (hn == 14) {
        static const float bad_q[] = {0.0f, -1.0f, NAN, INFINITY, -INFINITY, 2.5f, 1.0f, -0.0f};
        for (size_t k = 0; k < sizeof bad_q / sizeof bad_q[0]; k++) {
          if (dt == VOLCOMP_DTYPE_F32 && (k == 5 || k == 6)) continue; /* legal float steps */
          memcpy(cp, g_enc, n);
          uint32_t qb;
          memcpy(&qb, &bad_q[k], 4);
          vf_wr_u32(cp + 10, qb);
          CHECK_EQ(volcomp_decode_typed(cp, n, g_dec, N * es, dt), VOLCOMP_ERR_CORRUPT);
          float sq;
          CHECK_EQ(volcomp_stream_q(cp, n, &sq), VOLCOMP_ERR_CORRUPT);
        }
      }
      /* bit flips anywhere past the header: never a crash or an out-of-bounds read
       * (ASan/UBSan builds); the result is unchecked, wrong data is allowed */
      uint32_t r = 2024u + (uint32_t)c;
      for (int i = 0; i < 60; i++) {
        memcpy(cp, g_enc, n);
        cp[hn + vt_rng(&r) % (uint32_t)(n - hn)] ^= (uint8_t)(1u << (vt_rng(&r) % 8u));
        (void)volcomp_decode_typed(cp, n, g_dec, N * es, dt);
        (void)volcomp_decode_block_typed(cp, n, vt_rng(&r) & 7u, vt_rng(&r) & 7u, vt_rng(&r) & 7u, g_blk, 4096 * es, dt);
      }
    }
    free(cp);
    /* typed modes 7 and 8 are unknown to a revision-4 reader; here a mode above 8 is */
    gen_wind(src, VOLCOMP_DTYPE_U16, 0.0);
    size_t n = 0;
    CHECK_EQ(volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, 0.0f, g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n), VOLCOMP_OK);
    g_enc[5] = 9;
    CHECK_EQ(volcomp_decode_typed(g_enc, n, g_dec, N * 2, VOLCOMP_DTYPE_U16), VOLCOMP_ERR_CORRUPT);
    CHECK_EQ(volcomp_decode(g_enc, n, (uint8_t *)g_dec, N), VOLCOMP_ERR_CORRUPT);
  }

  /* ---- 8. the compression table: a smooth winding-like u16 field (W * 256) ---- */
  {
    gen_wind(src, VOLCOMP_DTYPE_U16, 0.0);
    const size_t raw = N * 2;
    printf("\nwinding-like u16 field (128^3, W * 256: radial ramp + waves), raw %zu bytes\n", raw);
    printf("%-10s %5s %9s %8s %6s %8s%s\n", "codec", "mode", "bytes", "ratio", "bpv", "max err",
           bench ? "  enc MB/s  dec MB/s" : "");
    static const float qs[] = {0.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f};
    for (size_t k = 0; k < sizeof qs / sizeof qs[0]; k++) {
      uint32_t mode;
      double mx;
      size_t n = rt_bounded(src, VOLCOMP_DTYPE_U16, qs[k], &mode, &mx);
      char name[32] = "lossless";
      if (qs[k] > 0.0f) snprintf(name, sizeof name, "q %g", (double)qs[k]);
      printf("%-10s %5u %9zu %7.2fx %6.3f %8.0f", name, mode, n, (double)raw / (double)n, 8.0 * (double)n / N, mx);
      if (bench) {
        const int reps = 5;
        double t0 = now_s();
        for (int i = 0; i < reps; i++)
          volcomp_encode_typed(src, VOLCOMP_DTYPE_U16, qs[k], g_enc, VOLCOMP_TYPED_ENCODE_BOUND, &n);
        const double te = (now_s() - t0) / reps;
        t0 = now_s();
        for (int i = 0; i < reps; i++) volcomp_decode_typed(g_enc, n, g_dec, raw, VOLCOMP_DTYPE_U16);
        const double td = (now_s() - t0) / reps;
        printf("  %8.0f  %8.0f", (double)raw / te / 1e6, (double)raw / td / 1e6);
      }
      printf("\n");
    }
  }

  free(src);
  free(g_blk);
  free(g_dec);
  free(g_enc);
  TEST_END();
}
