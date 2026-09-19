/* surface-pack: the signed-distance ramp, seamless tiling, 2x mean pooling,
 * the exact-ladder resample, and the blosc1/zstd source reader. */
#include "../volcomp.h"
#include "check.h"
#include "surface_pack.h"

#include <sys/stat.h>
#include <unistd.h>

/* ------------------------------------------------------------- reference ---
 * Brute force: the exact clipped Euclidean distance by scanning the (2d+1)^3
 * neighbourhood, and the ramp formula written out as the spec states it. */
static double ref_signed_distance(const uint8_t *m, int64_t mz, int64_t my, int64_t mx, int64_t z,
                                  int64_t y, int64_t x, int dmax) {
  int inside = m[(z * my + y) * mx + x] != 0;
  int best = dmax * dmax + 1;
  for (int dz = -dmax; dz <= dmax; dz++)
    for (int dy = -dmax; dy <= dmax; dy++)
      for (int dx = -dmax; dx <= dmax; dx++) {
        int64_t az = z + dz, ay = y + dy, ax = x + dx;
        if (az < 0 || ay < 0 || ax < 0 || az >= mz || ay >= my || ax >= mx) continue;
        int v = m[(az * my + ay) * mx + ax] != 0;
        if (v == inside) continue;
        int d2 = dz * dz + dy * dy + dx * dx;
        if (d2 < best) best = d2;
      }
  double s = sqrt((double)best);
  if (s > dmax) s = dmax;
  return inside ? s : -s;
}
static uint8_t ref_ramp(const uint8_t *m, int64_t mz, int64_t my, int64_t mx, int64_t z, int64_t y,
                        int64_t x, int dmax) {
  double s = ref_signed_distance(m, mz, my, mx, z, y, x, dmax);
  double v = floor(127.5 + (127.5 / dmax) * s + 0.5);
  return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
}
static void ramp_box(uint8_t *ramp, const uint8_t *mask, int64_t mz, int64_t my, int64_t mx, int dmax) {
  size_t n = (size_t)(mz * my * mx);
  uint8_t *loc = malloc(n), *a = malloc(n), *b = malloc(n), *s = malloc(n);
  uint8_t in_tab[SURF_INF + 1], out_tab[SURF_INF + 1];
  surf_ramp_tables(dmax, in_tab, out_tab);
  memcpy(loc, mask, n);
  surf_ramp_region(ramp, loc, mz, my, mx, dmax, in_tab, out_tab, a, b, s);
  free(loc), free(a), free(b), free(s);
}

/* ------------------------------------------------------ 1. the ramp values */
static void test_ramp_values(void) {
  const int64_t N = 48;
  size_t n = (size_t)(N * N * N);
  uint8_t *mask = calloc(1, n), *ramp = malloc(n);
  /* half space x >= 20: integer distances, the spec's numbers verbatim */
  for (int64_t z = 0; z < N; z++)
    for (int64_t y = 0; y < N; y++)
      for (int64_t x = 20; x < N; x++) mask[(z * N + y) * N + x] = 255;
  ramp_box(ramp, mask, N, N, N, SURF_DMAX);
  const int64_t z = 24, y = 24;
#define AT(X) ramp[(z * N + y) * N + (X)]
  CHECK_EQ(AT(23), 255); /* 3 voxels inside */
  CHECK_EQ(AT(22), 255);
  CHECK_EQ(AT(21), 213); /* 2 inside */
  CHECK_EQ(AT(20), 170); /* 1 inside: the first mask voxel */
  CHECK_EQ(AT(19), 85);  /* 1 outside: the last air voxel */
  CHECK_EQ(AT(18), 43);
  CHECK_EQ(AT(17), 0);
  CHECK_EQ(AT(16), 0);
  /* the 128 crossing lies on the published edge: the mean of the two voxels that straddle it */
  CHECK_EQ((AT(20) + AT(19) + 1) / 2, 128);
#undef AT
  /* a sphere, against the brute-force reference everywhere away from the box edge */
  memset(mask, 0, n);
  for (int64_t zz = 0; zz < N; zz++)
    for (int64_t yy = 0; yy < N; yy++)
      for (int64_t xx = 0; xx < N; xx++) {
        double r = (double)((zz - 24) * (zz - 24) + (yy - 23) * (yy - 23) + (xx - 25) * (xx - 25));
        mask[(zz * N + yy) * N + xx] = r < 13.0 * 13.0 ? 255 : 0;
      }
  ramp_box(ramp, mask, N, N, N, SURF_DMAX);
  unsigned bad = 0, seen_in[4] = {0}, seen_out[4] = {0};
  for (int64_t zz = SURF_DMAX; zz < N - SURF_DMAX; zz++)
    for (int64_t yy = SURF_DMAX; yy < N - SURF_DMAX; yy++)
      for (int64_t xx = SURF_DMAX; xx < N - SURF_DMAX; xx++) {
        uint8_t want = ref_ramp(mask, N, N, N, zz, yy, xx, SURF_DMAX);
        uint8_t got = ramp[(zz * N + yy) * N + xx];
        if (want != got) bad++;
        for (int k = 1; k <= 3; k++) {
          if (got == (k == 1 ? 170 : k == 2 ? 213 : 255)) seen_in[k]++;
          if (got == (k == 1 ? 85 : k == 2 ? 43 : 0)) seen_out[k]++;
        }
      }
  CHECK_EQ(bad, 0);
  for (int k = 1; k <= 3; k++) CHECK(seen_in[k] > 0 && seen_out[k] > 0);
  free(mask), free(ramp);
}

/* ------------------------------------------- 2. tiling with a halo is seamless */
static void test_tiling(void) {
  const int64_t N = 64, H = SURF_DMAX, T = 16; /* 4^3 tiles of 16 with a 3-voxel halo */
  size_t n = (size_t)(N * N * N);
  uint8_t *mask = malloc(n), *whole = malloc(n), *tiled = calloc(1, n);
  uint32_t rs = 12345;
  for (size_t i = 0; i < n; i++) mask[i] = (vt_rng(&rs) % 5) == 0 ? 255 : 0;
  /* a few solid slabs so there are large inside regions too */
  for (int64_t z = 20; z < 30; z++) memset(mask + (size_t)(z * N * N), 255, (size_t)(N * N));
  ramp_box(whole, mask, N, N, N, SURF_DMAX);
  size_t tn = (size_t)((T + 2 * H) * (T + 2 * H) * (T + 2 * H));
  uint8_t *sub = malloc(tn), *subramp = malloc(tn);
  for (int64_t tz = H; tz + T + H <= N; tz += T)
    for (int64_t ty = H; ty + T + H <= N; ty += T)
      for (int64_t tx = H; tx + T + H <= N; tx += T) {
        for (int64_t z = 0; z < T + 2 * H; z++)
          for (int64_t y = 0; y < T + 2 * H; y++)
            memcpy(sub + (size_t)((z * (T + 2 * H) + y) * (T + 2 * H)),
                   mask + (size_t)(((tz - H + z) * N + (ty - H + y)) * N + tx - H), (size_t)(T + 2 * H));
        ramp_box(subramp, sub, T + 2 * H, T + 2 * H, T + 2 * H, SURF_DMAX);
        for (int64_t z = 0; z < T; z++)
          for (int64_t y = 0; y < T; y++)
            for (int64_t x = 0; x < T; x++)
              tiled[(size_t)(((tz + z) * N + ty + y) * N + tx + x)] =
                  subramp[(size_t)(((z + H) * (T + 2 * H) + y + H) * (T + 2 * H) + x + H)];
      }
  int64_t ntile = (N - 2 * H) / T;  /* whole tiles that fit with their halo */
  unsigned bad = 0, checked = 0;
  for (int64_t z = H; z < H + ntile * T; z++)
    for (int64_t y = H; y < H + ntile * T; y++)
      for (int64_t x = H; x < H + ntile * T; x++) {
        int64_t i = (z * N + y) * N + x;
        checked++;
        if (whole[i] != tiled[i]) bad++;
      }
  CHECK_EQ(bad, 0);
  CHECK_EQ(checked, (unsigned)(ntile * T * ntile * T * ntile * T));
  free(mask), free(whole), free(tiled), free(sub), free(subramp);
}

/* --------------------------------------------------------- blosc1 chunks --- */

/* a "memcpyed" blosc1 container: what the reader must accept without libzstd */
static size_t blosc_memcpyed(uint8_t *dst, const uint8_t *src, size_t n) {
  memset(dst, 0, BLOSC1_HEADER);
  dst[0] = 2;
  dst[1] = 1;
  dst[2] = BLOSC1_MEMCPYED;
  dst[3] = 1;
  for (int i = 0; i < 4; i++) {
    dst[4 + i] = (uint8_t)(n >> (8 * i));
    dst[8 + i] = (uint8_t)(n >> (8 * i));
    dst[12 + i] = (uint8_t)((n + BLOSC1_HEADER) >> (8 * i));
  }
  memcpy(dst + BLOSC1_HEADER, src, n);
  return n + BLOSC1_HEADER;
}
static void test_blosc_fixture(void) {
  /* tests/data/blosc_zstd_64.bin: a 64^3 uint8 chunk written by numcodecs
   * Blosc(cname="zstd", clevel=1, shuffle=1) — exactly what the bucket holds. */
  size_t n = 0;
  uint8_t *raw = shard_read_file("tests/data/blosc_zstd_64.bin", &n);
  CHECK(raw != NULL);
  if (!raw) return;
  size_t want = 64 * 64 * 64;
  uint8_t *out = malloc(want), *tmp = malloc(want);
  const char *err = NULL;
  int64_t got = blosc1_decompress(raw, n, out, want, tmp, want, &err);
#ifdef VOLCOMP_HAVE_ZSTD
  CHECK_EQ(got, (int64_t)want);
  if (err) fprintf(stderr, "blosc: %s\n", err);
  uint64_t h = 0xcbf29ce484222325ull, nz = 0;
  for (size_t i = 0; i < want; i++) {
    h = (h ^ out[i]) * 0x100000001b3ull;
    nz += out[i] != 0;
  }
  CHECK_EQ(h, (int64_t)0x88fbbdca52007452ull); /* FNV-1a of the array numcodecs encoded */
  CHECK_EQ(nz, 16237);
#else
  CHECK_EQ(got, -1); /* built without libzstd: a clean error, never silent zeros */
#endif
  free(raw), free(out), free(tmp);
}

/* ------------------------------------- 3./4. end to end through surface_pack */

static int shard_entries(const uint8_t *img, size_t n, unsigned count, uint64_t *off, uint64_t *nb) {
  size_t idxn = (size_t)count * 16u + 4u;
  if (n < idxn) return 0;
  const uint8_t *idx = img + n - idxn;
  uint32_t crc = shard_crc32c(idx, (size_t)count * 16u), stored = 0;
  for (int i = 3; i >= 0; i--) stored = stored << 8 | idx[(size_t)count * 16u + (unsigned)i];
  if (crc != stored) return 0;
  for (unsigned i = 0; i < count; i++) {
    off[i] = shard_rd_u64(idx + i * 16);
    nb[i] = shard_rd_u64(idx + i * 16 + 8);
  }
  return 1;
}

static void test_end_to_end(double scale, int64_t csize) {
  char dir[] = "/tmp/volcomp_surface_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char src[4096], out[4096];
  snprintf(src, sizeof src, "%s/src", dir);
  snprintf(out, sizeof out, "%s/out", dir);
  mkdir(src, 0700);
  mkdir(out, 0700);
  const int64_t S0 = 261, S1 = 200, S2 = 140; /* deliberately not multiples of anything */
  size_t sn = (size_t)(S0 * S1 * S2);
  uint8_t *mask = calloc(1, sn);
  for (int64_t z = 0; z < S0; z++)
    for (int64_t y = 0; y < S1; y++)
      for (int64_t x = 0; x < S2; x++) {
        /* two sheets and a blob: like a surface prediction */
        int on = (x + z / 4) % 37 < 2 || (y > 40 && y < 44 && x > 20) ||
                 ((z - 130) * (z - 130) + (y - 100) * (y - 100) + (x - 70) * (x - 70) < 400);
        mask[(z * S1 + y) * S2 + x] = on ? 255 : 0;
      }
  /* write the source as zarr v2 blosc chunks */
  int64_t cg[3] = {(S0 + csize - 1) / csize, (S1 + csize - 1) / csize, (S2 + csize - 1) / csize};
  uint8_t *chunk = malloc((size_t)(csize * csize * csize));
  uint8_t *cbuf = malloc((size_t)(csize * csize * csize) + BLOSC1_HEADER);
  for (int64_t cz = 0; cz < cg[0]; cz++)
    for (int64_t cy = 0; cy < cg[1]; cy++)
      for (int64_t cx = 0; cx < cg[2]; cx++) {
        memset(chunk, 0, (size_t)(csize * csize * csize));
        int any = 0;
        for (int64_t z = 0; z < csize && cz * csize + z < S0; z++)
          for (int64_t y = 0; y < csize && cy * csize + y < S1; y++)
            for (int64_t x = 0; x < csize && cx * csize + x < S2; x++) {
              uint8_t v = mask[((cz * csize + z) * S1 + cy * csize + y) * S2 + cx * csize + x];
              chunk[(z * csize + y) * csize + x] = v;
              any |= v != 0;
            }
        if (!any) continue; /* absent chunk: the tool must treat it as zeros */
        size_t bn = blosc_memcpyed(cbuf, chunk, (size_t)(csize * csize * csize));
        char p[4200];
        snprintf(p, sizeof p, "%s/%lld_%lld_%lld.blosc", src, (long long)cz, (long long)cy, (long long)cx);
        FILE *f = fopen(p, "wb");
        CHECK(f && fwrite(cbuf, 1, bn, f) == bn);
        if (f) fclose(f);
      }
  surf_cfg cfg = {.srcdir = src,
                  .outdir = out,
                  .csize = csize,
                  .src_shape = {S0, S1, S2},
                  .shard = {0, 0, 0},
                  .scale = scale,
                  .q = {VOLCOMP_Q_LOSSLESS, VOLCOMP_Q_LOSSLESS, VOLCOMP_Q_LOSSLESS, VOLCOMP_Q_LOSSLESS},
                  .dmax = SURF_DMAX,
                  .threads = 3,
                  .samples = 8};
  for (int d = 0; d < 3; d++)
    cfg.out_shape[d] = (int64_t)floor((double)cfg.src_shape[d] / scale + 0.5);
  surf_result res;
  CHECK_EQ(surface_pack(&cfg, &res), 0);
  CHECK(res.present[0] > 0 && res.present[1] > 0 && res.present[2] > 0 && res.present[3] > 0);
  CHECK(res.src_nonzero);

  /* reference: the ramp over the whole source volume, then the resample */
  /* the tool treats everything outside the published array as air, so the reference
   * ramp is computed on the volume padded with SURF_DMAX zero voxels */
  const int64_t P = SURF_DMAX;
  int64_t P0 = S0 + 2 * P, P1 = S1 + 2 * P, P2 = S2 + 2 * P;
  uint8_t *pmask = calloc(1, (size_t)(P0 * P1 * P2)), *pramp = malloc((size_t)(P0 * P1 * P2));
  for (int64_t z = 0; z < S0; z++)
    for (int64_t y = 0; y < S1; y++)
      memcpy(pmask + (size_t)(((z + P) * P1 + y + P) * P2 + P), mask + (size_t)((z * S1 + y) * S2), (size_t)S2);
  ramp_box(pramp, pmask, P0, P1, P2, SURF_DMAX);
  uint8_t *ramp = malloc(sn);
  for (int64_t z = 0; z < S0; z++)
    for (int64_t y = 0; y < S1; y++)
      memcpy(ramp + (size_t)((z * S1 + y) * S2), pramp + (size_t)(((z + P) * P1 + y + P) * P2 + P), (size_t)S2);
  free(pmask), free(pramp);
  int64_t O0 = cfg.out_shape[0], O1 = cfg.out_shape[1], O2 = cfg.out_shape[2];
  uint8_t *want = calloc(1, (size_t)(O0 * O1 * O2));
  for (int64_t z = 0; z < O0; z++)
    for (int64_t y = 0; y < O1; y++)
      for (int64_t x = 0; x < O2; x++) {
        double v;
        if (scale == 1.0) {
          v = ramp[(z * S1 + y) * S2 + x];
        } else {
          double tz = (double)z * scale, ty = (double)y * scale, tx = (double)x * scale;
          int64_t az = (int64_t)floor(tz), ay = (int64_t)floor(ty), ax = (int64_t)floor(tx);
          double fz = tz - (double)az, fy = ty - (double)ay, fx = tx - (double)ax;
          v = 0;
          for (int dz = 0; dz < 2; dz++)
            for (int dy = 0; dy < 2; dy++)
              for (int dx = 0; dx < 2; dx++) {
                int64_t bz = az + dz < S0 ? az + dz : S0 - 1, by = ay + dy < S1 ? ay + dy : S1 - 1,
                        bx = ax + dx < S2 ? ax + dx : S2 - 1;
                double w = (dz ? fz : 1 - fz) * (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx);
                v += w * ramp[(bz * S1 + by) * S2 + bx];
              }
          v = floor(v + 0.5);
        }
        want[(z * O1 + y) * O2 + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
      }
  /* level 0: decode the shard and compare */
  char p[4200];
  snprintf(p, sizeof p, "%s/0.shard", out);
  size_t shn = 0;
  uint8_t *img = shard_read_file(p, &shn);
  CHECK(img != NULL);
  uint64_t off[512], nb[512];
  CHECK(img && shard_entries(img, shn, 512, off, nb));
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  unsigned bad = 0, badmiss = 0;
  for (unsigned i = 0; i < 512 && img; i++) {
    int64_t cz = i >> 6, cy = (i >> 3) & 7, cx = i & 7;
    int64_t o[3] = {cz * 128, cy * 128, cx * 128};
    if (off[i] == ~0ull) {
      /* missing: every in-bounds voxel of the reference must be zero */
      for (int64_t z = o[0]; z < o[0] + 128 && z < O0; z++)
        for (int64_t y = o[1]; y < o[1] + 128 && y < O1; y++)
          for (int64_t x = o[2]; x < o[2] + 128 && x < O2; x++)
            if (want[(z * O1 + y) * O2 + x]) badmiss++;
      continue;
    }
    CHECK_EQ(volcomp_decode(img + off[i], (size_t)nb[i], dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
    for (int64_t z = 0; z < 128; z++)
      for (int64_t y = 0; y < 128; y++)
        for (int64_t x = 0; x < 128; x++) {
          int64_t gz = o[0] + z, gy = o[1] + y, gx = o[2] + x;
          int w = (gz < O0 && gy < O1 && gx < O2) ? want[(gz * O1 + gy) * O2 + gx] : 0;
          int g = dec[(z * 128 + y) * 128 + x];
          if (scale == 1.0 ? (g != w) : (abs(g - w) > 1)) bad++;
        }
  }
  CHECK_EQ(badmiss, 0);
  CHECK_EQ(bad, 0);
  /* levels 1..3: exactly the 2x mean pooling of the level above */
  uint8_t *prev = want;
  int64_t pd[3] = {O0, O1, O2};
  for (int L = 1; L < SURF_LEVELS; L++) {
    int64_t nd[3] = {(pd[0] + 1) / 2, (pd[1] + 1) / 2, (pd[2] + 1) / 2};
    uint8_t *cur = calloc(1, (size_t)(nd[0] * nd[1] * nd[2]));
    for (int64_t z = 0; z < nd[0]; z++)
      for (int64_t y = 0; y < nd[1]; y++)
        for (int64_t x = 0; x < nd[2]; x++) {
          unsigned sum = 0, cnt = 0;
          for (int dz = 0; dz < 2 && 2 * z + dz < pd[0]; dz++)
            for (int dy = 0; dy < 2 && 2 * y + dy < pd[1]; dy++)
              for (int dx = 0; dx < 2 && 2 * x + dx < pd[2]; dx++) {
                sum += prev[((2 * z + dz) * pd[1] + 2 * y + dy) * pd[2] + 2 * x + dx];
                cnt++;
              }
          cur[(z * nd[1] + y) * nd[2] + x] = (uint8_t)((sum + cnt / 2) / cnt);
        }
    unsigned count = (unsigned)((1024 >> L) / 128) * (unsigned)((1024 >> L) / 128) * (unsigned)((1024 >> L) / 128);
    int64_t g = (1024 >> L) / 128;
    snprintf(p, sizeof p, "%s/%d.shard", out, L);
    size_t ln = 0;
    uint8_t *limg = shard_read_file(p, &ln);
    CHECK(limg != NULL);
    uint64_t *loff = malloc(count * sizeof *loff), *lnb = malloc(count * sizeof *lnb);
    CHECK(limg && shard_entries(limg, ln, count, loff, lnb));
    unsigned lbad = 0;
    for (unsigned i = 0; i < count && limg; i++) {
      int64_t o[3] = {(int64_t)(i / (unsigned)(g * g)) * 128, (int64_t)((i / (unsigned)g) % (unsigned)g) * 128,
                      (int64_t)(i % (unsigned)g) * 128};
      if (loff[i] == ~0ull) continue;
      CHECK_EQ(volcomp_decode(limg + loff[i], (size_t)lnb[i], dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
      for (int64_t z = 0; z < 128 && o[0] + z < nd[0]; z++)
        for (int64_t y = 0; y < 128 && o[1] + y < nd[1]; y++)
          for (int64_t x = 0; x < 128 && o[2] + x < nd[2]; x++) {
            int w = cur[((o[0] + z) * nd[1] + o[1] + y) * nd[2] + o[2] + x];
            int gv = dec[(z * 128 + y) * 128 + x];
            if (scale == 1.0 ? (gv != w) : (abs(gv - w) > 2)) lbad++;
          }
    }
    CHECK_EQ(lbad, 0);
    free(limg), free(loff), free(lnb);
    if (prev != want) free(prev);
    prev = cur;
    pd[0] = nd[0], pd[1] = nd[1], pd[2] = nd[2];
  }
  if (prev != want) free(prev);
  free(img), free(dec), free(ramp), free(want), free(mask), free(chunk), free(cbuf);
  char cmd[4200];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}

/* ---------------------------- 5. offline pooling of the coarse levels ------ */
static void test_pool_shard(void) {
  char dir[] = "/tmp/volcomp_pool_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  const int64_t C = SURF_CHUNK;
  /* input level: 200^3 voxels = a 2x2x2 block of 128^3 shards, the last partial */
  int64_t shape[3] = {200, 200, 200}, pos[3] = {0, 0, 0};
  uint8_t *big = calloc(1, (size_t)(8 * C * C * C));
  uint32_t rs = 999;
  for (int64_t z = 0; z < shape[0]; z++)
    for (int64_t y = 0; y < shape[1]; y++)
      for (int64_t x = 0; x < shape[2]; x++)
        big[((z * 2 * C + y) * 2 * C) + x] = (uint8_t)(vt_rng(&rs) & 0xff);
  char paths[8][4200];
  const char *in[8];
  uint8_t *chunk = malloc(VOLCOMP_CHUNK_VOXELS), *enc = malloc(VOLCOMP_ENCODE_BOUND);
  for (int i = 0; i < 8; i++) {
    int64_t o[3] = {(i >> 2) * C, ((i >> 1) & 1) * C, (i & 1) * C};
    memset(chunk, 0, VOLCOMP_CHUNK_VOXELS);
    for (int64_t z = 0; z < C; z++)
      for (int64_t y = 0; y < C; y++)
        memcpy(chunk + (z * C + y) * C, big + (((o[0] + z) * 2 * C + o[1] + y) * 2 * C + o[2]), (size_t)C);
    size_t en = 0;
    CHECK_EQ(volcomp_encode(chunk, VOLCOMP_Q_LOSSLESS, enc, VOLCOMP_ENCODE_BOUND, &en), VOLCOMP_OK);
    snprintf(paths[i], sizeof paths[i], "%s/in%d.shard", dir, i);
    uint8_t *one[1] = {enc};
    size_t onen[1] = {en};
    uint64_t payload = 0;
    CHECK_EQ(surf_write_shard(paths[i], one, onen, 1, &payload), 0);
    in[i] = paths[i];
  }
  char outp[4200];
  snprintf(outp, sizeof outp, "%s/out.shard", dir);
  unsigned present = 0;
  uint64_t bytes = 0;
  CHECK_EQ(surface_pool_shard(in, outp, VOLCOMP_Q_LOSSLESS, shape, pos, &present, &bytes), 0);
  CHECK_EQ(present, 1);
  size_t n = 0;
  uint8_t *img = shard_read_file(outp, &n);
  uint64_t off[1], nb[1];
  CHECK(img && shard_entries(img, n, 1, off, nb));
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  CHECK_EQ(volcomp_decode(img + off[0], (size_t)nb[0], dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
  unsigned bad = 0;
  for (int64_t z = 0; z < (shape[0] + 1) / 2; z++)
    for (int64_t y = 0; y < (shape[1] + 1) / 2; y++)
      for (int64_t x = 0; x < (shape[2] + 1) / 2; x++) {
        unsigned sum = 0, cnt = 0;
        for (int64_t dz = 0; dz < 2 && 2 * z + dz < shape[0]; dz++)
          for (int64_t dy = 0; dy < 2 && 2 * y + dy < shape[1]; dy++)
            for (int64_t dx = 0; dx < 2 && 2 * x + dx < shape[2]; dx++) {
              sum += big[(((2 * z + dz) * 2 * C + 2 * y + dy) * 2 * C) + 2 * x + dx];
              cnt++;
            }
        if (dec[(z * C + y) * C + x] != (uint8_t)((sum + cnt / 2) / cnt)) bad++;
      }
  CHECK_EQ(bad, 0);
  free(big), free(chunk), free(enc), free(img), free(dec);
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}

int main(void) {
  test_ramp_values();
  test_tiling();
  test_blosc_fixture();
  test_end_to_end(1.0, 192);   /* identity ladder, source chunks larger than the output tiling */
  test_end_to_end(1.25, 128);
  test_pool_shard();  /* resampled onto the exact rung */
  TEST_END();
}
