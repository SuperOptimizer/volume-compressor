/* surface-pack: the signed-distance ramp, seamless tiling, 2x mean pooling,
 * the exact-ladder resample, and the blosc1/zstd source reader. */
#include "../volcomp.h"
#include "check.h"
#include "surface_pack.h"

#include <stdbool.h>
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
  CHECK_EQ(surface_pool_shard(in, outp, VOLCOMP_Q_LOSSLESS, shape, pos, false, false, &present, &bytes), 0);
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

static int main_tests(void);

/* ---------------- 6. golden bytes: a dense and a sparse unit, end to end -----
 * Hashes of the four shard files of two synthetic units, at the real quantisers.
 * The constants were produced by the build BEFORE surface-pack was optimised
 * (uniform-box and occupancy skips, AVX2 min-plus, fused ramp pipeline, AVX2
 * resample and pooling): every one of those is required to be byte-identical, and
 * this is what says so. Run this file with --print to regenerate them. */
static uint64_t fnv1a(const uint8_t *p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}
/* a synthetic prediction: `dense` fills the unit with sheets, otherwise it holds
 * two small blobs and is empty everywhere else (the common case in the bucket) */
static void golden_mask(uint8_t *m, int64_t S0, int64_t S1, int64_t S2, bool dense) {
  for (int64_t z = 0; z < S0; z++)
    for (int64_t y = 0; y < S1; y++)
      for (int64_t x = 0; x < S2; x++) {
        int on;
        if (dense)
          on = (x + z / 3) % 23 < 3 || (y + x / 5) % 31 < 2 ||
               ((z - 200) * (z - 200) + (y - 150) * (y - 150) + (x - 150) * (x - 150) < 3600);
        else
          on = ((z - 70) * (z - 70) + (y - 60) * (y - 60) + (x - 50) * (x - 50) < 400) ||
               ((z - 300) * (z - 300) + (y - 260) * (y - 260) + (x - 300) * (x - 300) < 144);
        m[(z * S1 + y) * S2 + x] = on ? 255 : 0;
      }
}
static void golden_unit(bool dense, bool lossless, int print, const uint64_t want[4]) {
  char dir[] = "/tmp/volcomp_golden_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char src[4096], out[4096];
  snprintf(src, sizeof src, "%s/src", dir);
  snprintf(out, sizeof out, "%s/out", dir);
  mkdir(src, 0700);
  mkdir(out, 0700);
  const int64_t S0 = 400, S1 = 330, S2 = 350, csize = 192;
  uint8_t *mask = malloc((size_t)(S0 * S1 * S2));
  golden_mask(mask, S0, S1, S2, dense);
  uint8_t *chunk = malloc((size_t)(csize * csize * csize));
  uint8_t *cbuf = malloc((size_t)(csize * csize * csize) + BLOSC1_HEADER);
  for (int64_t cz = 0; cz * csize < S0; cz++)
    for (int64_t cy = 0; cy * csize < S1; cy++)
      for (int64_t cx = 0; cx * csize < S2; cx++) {
        memset(chunk, 0, (size_t)(csize * csize * csize));
        int any = 0;
        for (int64_t z = 0; z < csize && cz * csize + z < S0; z++)
          for (int64_t y = 0; y < csize && cy * csize + y < S1; y++)
            for (int64_t x = 0; x < csize && cx * csize + x < S2; x++) {
              uint8_t v = mask[((cz * csize + z) * S1 + cy * csize + y) * S2 + cx * csize + x];
              chunk[(z * csize + y) * csize + x] = v;
              any |= v != 0;
            }
        if (!any) continue;
        size_t bn = blosc_memcpyed(cbuf, chunk, (size_t)(csize * csize * csize));
        char p[4200];
        snprintf(p, sizeof p, "%s/%lld_%lld_%lld.blosc", src, (long long)cz, (long long)cy, (long long)cx);
        FILE *f = fopen(p, "wb");
        CHECK(f && fwrite(cbuf, 1, bn, f) == bn);
        if (f) fclose(f);
      }
  const double scale = 9.6 / 9.362;  /* a real rung snap: 9.362 um onto the 9.6 um rung */
  surf_cfg cfg = {.srcdir = src,
                  .outdir = out,
                  .csize = csize,
                  .src_shape = {S0, S1, S2},
                  .shard = {0, 0, 0},
                  .scale = scale,
                  .dmax = SURF_DMAX,
                  .threads = 3,
                  .samples = 8};
  for (int L = 0; L < SURF_LEVELS; L++)
    cfg.q[L] = lossless ? VOLCOMP_Q_LOSSLESS : (L == 0 ? 2.0f : 1.0f);
  for (int d = 0; d < 3; d++) cfg.out_shape[d] = (int64_t)floor((double)cfg.src_shape[d] / scale + 0.5);
  surf_result res;
  CHECK_EQ(surface_pack(&cfg, &res), 0);
  for (int L = 0; L < SURF_LEVELS; L++) {
    char p[4200];
    snprintf(p, sizeof p, "%s/%d.shard", out, L);
    size_t n = 0;
    uint8_t *img = shard_read_file(p, &n);
    uint64_t h = img ? fnv1a(img, n) : 0;
    if (print)
      printf("  0x%016llxull, /* %s %s level %d: %zu bytes */\n", (unsigned long long)h,
             dense ? "dense" : "sparse", lossless ? "lossless" : "q", L, n);
    else
      CHECK_EQ(h, (int64_t)want[L]);
    free(img);
  }
  free(mask), free(chunk), free(cbuf);
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}
/* q = 0: the lossless mode has no float math, so these hold on every build and
 * pin the transform itself (the ramp, the resample and the pooling). */
static const uint64_t golden_dense_l[4] = {
    0xaf598eed7eb514b2ull, /* level 0: 14879274 bytes */
    0x303907b8be7f0ae8ull, /* level 1: 2418235 bytes */
    0x4884ba471fd2bb6full, /* level 2: 399457 bytes */
    0x5df650d90bde7807ull, /* level 3: 73616 bytes */
};
static const uint64_t golden_sparse_l[4] = {
    0x187be9a7d082e0a2ull, /* level 0: 49175 bytes */
    0x645fc722436c31d7ull, /* level 1: 10497 bytes */
    0xc6dabd613b401a4cull, /* level 2: 2318 bytes */
    0xedd5c2622c021a5cull, /* level 3: 1079 bytes */
};
/* the real quantisers (q 2 / 1 / 1 / 1). volcomp's own DCT kernels are allowed to
 * differ by a bit between the AVX2 and the plain C kernel sets (spec "Cross-build
 * agreement"), so these are checked on the AVX2 kernels the fleet runs on. */
static const uint64_t golden_dense_q[4] = {
    0x83ba9af17a9ef3d2ull, /* level 0: 3681737 bytes */
    0xd8fbccd6b6fb1927ull, /* level 1: 1144917 bytes */
    0xf082226ed9cffb74ull, /* level 2: 208541 bytes */
    0x5b56069797689574ull, /* level 3: 29777 bytes */
};
static const uint64_t golden_sparse_q[4] = {
    0x661c1774bffc3f85ull, /* level 0: 19849 bytes */
    0x41495d0bbfe5f12cull, /* level 1: 10718 bytes */
    0x035174f8e93db178ull, /* level 2: 6313 bytes */
    0xb7c1d17ef34ef5eeull, /* level 3: 2665 bytes */
};

/* ------------- 7. the occupancy mask and the max-pool it depends on --------- */
static void test_occupancy(void) {
  char dir[] = "/tmp/volcomp_occ_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char l0[4096], l5[4096], out[4096];
  snprintf(l0, sizeof l0, "%s/l0", dir);
  snprintf(l5, sizeof l5, "%s/l5", dir);
  snprintf(out, sizeof out, "%s/out", dir);
  mkdir(l0, 0700);
  mkdir(l5, 0700);
  mkdir(out, 0700);
  const int64_t S0 = 400, S1 = 330, S2 = 350, csize = 192, factor = 32;
  uint8_t *mask = malloc((size_t)(S0 * S1 * S2));
  golden_mask(mask, S0, S1, S2, false); /* the sparse unit: two small blobs */
  /* level 0 chunks */
  uint8_t *chunk = malloc((size_t)(csize * csize * csize));
  uint8_t *cbuf = malloc((size_t)(csize * csize * csize) + BLOSC1_HEADER);
  for (int64_t cz = 0; cz * csize < S0; cz++)
    for (int64_t cy = 0; cy * csize < S1; cy++)
      for (int64_t cx = 0; cx * csize < S2; cx++) {
        memset(chunk, 0, (size_t)(csize * csize * csize));
        int any = 0;
        for (int64_t z = 0; z < csize && cz * csize + z < S0; z++)
          for (int64_t y = 0; y < csize && cy * csize + y < S1; y++)
            for (int64_t x = 0; x < csize && cx * csize + x < S2; x++) {
              uint8_t v = mask[((cz * csize + z) * S1 + cy * csize + y) * S2 + cx * csize + x];
              chunk[(z * csize + y) * csize + x] = v;
              any |= v != 0;
            }
        if (!any) continue;
        size_t bn = blosc_memcpyed(cbuf, chunk, (size_t)(csize * csize * csize));
        char p[4200];
        snprintf(p, sizeof p, "%s/%lld_%lld_%lld.blosc", l0, (long long)cz, (long long)cy, (long long)cx);
        FILE *f = fopen(p, "wb");
        CHECK(f && fwrite(cbuf, 1, bn, f) == bn);
        if (f) fclose(f);
      }
  /* the "published" coarse level: an exact 32x max pool, one chunk */
  int64_t c0 = (S0 + factor - 1) / factor, c1 = (S1 + factor - 1) / factor, c2 = (S2 + factor - 1) / factor;
  memset(chunk, 0, (size_t)(csize * csize * csize));
  for (int64_t z = 0; z < S0; z++)
    for (int64_t y = 0; y < S1; y++)
      for (int64_t x = 0; x < S2; x++)
        if (mask[(z * S1 + y) * S2 + x])
          chunk[((z / factor) * csize + y / factor) * csize + x / factor] = 255;
  {
    size_t bn = blosc_memcpyed(cbuf, chunk, (size_t)(csize * csize * csize));
    char p[4200];
    snprintf(p, sizeof p, "%s/0_0_0.blosc", l5);
    FILE *f = fopen(p, "wb");
    CHECK(f && fwrite(cbuf, 1, bn, f) == bn);
    if (f) fclose(f);
  }
  const double scale = 9.6 / 9.362;
  int64_t coarse_shape[3] = {c0, c1, c2}, src_shape[3] = {S0, S1, S2}, out_shape[3];
  for (int d = 0; d < 3; d++) out_shape[d] = (int64_t)floor((double)src_shape[d] / scale + 0.5);
  char maskfile[4200];
  snprintf(maskfile, sizeof maskfile, "%s/masks.bin", dir);
  uint64_t occ = 0, tot = 0;
  CHECK_EQ(surface_occupancy(l5, maskfile, csize, coarse_shape, factor, out_shape, scale, SURF_DMAX, 1, &occ,
                             &tot),
           0);
  size_t mn = 0;
  uint8_t *masks = shard_read_file(maskfile, &mn);
  CHECK(masks != NULL && mn == 64);       /* one shard */
  CHECK(occ > 0 && occ < tot);            /* the sparse unit is mostly air */
  /* the mask must not change a single byte: it may only drop chunks that are all air */
  surf_cfg cfg = {.srcdir = l0,
                  .outdir = out,
                  .csize = csize,
                  .src_shape = {S0, S1, S2},
                  .shard = {0, 0, 0},
                  .scale = scale,
                  .q = {VOLCOMP_Q_LOSSLESS, VOLCOMP_Q_LOSSLESS, VOLCOMP_Q_LOSSLESS, VOLCOMP_Q_LOSSLESS},
                  .occ = masks,
                  .dmax = SURF_DMAX,
                  .threads = 2,
                  .samples = 8};
  for (int d = 0; d < 3; d++) cfg.out_shape[d] = out_shape[d];
  surf_result res;
  CHECK_EQ(surface_pack(&cfg, &res), 0);
  CHECK(res.skip_mask > 0);
  for (int L = 0; L < SURF_LEVELS; L++) {
    char p[4200];
    snprintf(p, sizeof p, "%s/%d.shard", out, L);
    size_t n = 0;
    uint8_t *img = shard_read_file(p, &n);
    CHECK_EQ(img ? fnv1a(img, n) : 0, (int64_t)golden_sparse_l[L]);
    free(img);
  }
  /* and the check that guards it: this pyramid is a max pool, a thinned one is not */
  free(masks), free(mask), free(chunk), free(cbuf);
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}

/* the guard itself: an exact max pool passes, a pyramid that lost a thin sheet does not */
static void test_maxpool_check(void) {
  char dir[] = "/tmp/volcomp_mp_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  const int64_t C = 64;
  uint8_t *fine = calloc(1, (size_t)(C * C * C)), *coarse = calloc(1, (size_t)(C * C * C));
  uint8_t *buf = malloc((size_t)(C * C * C) + BLOSC1_HEADER);
  uint32_t rs = 4242;
  for (int64_t i = 0; i < C * C * C; i++) fine[i] = (vt_rng(&rs) % 9) == 0 ? 255 : 0;
  for (int64_t z = 0; z < C / 2; z++)  /* the true max pool of the (0,0,0) sub-block */
    for (int64_t y = 0; y < C / 2; y++)
      for (int64_t x = 0; x < C / 2; x++) {
        unsigned any = 0;
        for (int dz = 0; dz < 2; dz++)
          for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++) any |= fine[((2 * z + dz) * C + 2 * y + dy) * C + 2 * x + dx];
        coarse[(z * C + y) * C + x] = (uint8_t)(any ? 255 : 0);
      }
  char cp[4200], fp[4200];
  snprintf(cp, sizeof cp, "%s/c.blosc", dir);
  snprintf(fp, sizeof fp, "%s/f.blosc", dir);
  for (int pass = 0; pass < 2; pass++) {
    if (pass) {  /* lose one coarse voxel, as a re-thresholded probability pyramid does */
      for (int64_t i = 0; i < C * C * C; i++)
        if (coarse[i]) {
          coarse[i] = 0;
          break;
        }
    }
    FILE *f = fopen(cp, "wb");
    size_t bn = blosc_memcpyed(buf, coarse, (size_t)(C * C * C));
    CHECK(f && fwrite(buf, 1, bn, f) == bn);
    if (f) fclose(f);
    f = fopen(fp, "wb");
    bn = blosc_memcpyed(buf, fine, (size_t)(C * C * C));
    CHECK(f && fwrite(buf, 1, bn, f) == bn);
    if (f) fclose(f);
    const char *fines[8] = {fp, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint64_t misses = 0, cnz = 0, pnz = 0;
    CHECK_EQ(surf_maxpool_check(cp, fines, C, &misses, &cnz, &pnz), 0);
    CHECK(pnz > 0);
    CHECK_EQ(misses, pass ? 1 : 0);
  }
  free(fine), free(coarse), free(buf);
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}


int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "--print")) {
    printf("static const uint64_t golden_dense_l[4] = {\n");
    golden_unit(true, true, 1, NULL);
    printf("};\nstatic const uint64_t golden_sparse_l[4] = {\n");
    golden_unit(false, true, 1, NULL);
    printf("};\nstatic const uint64_t golden_dense_q[4] = {\n");
    golden_unit(true, false, 1, NULL);
    printf("};\nstatic const uint64_t golden_sparse_q[4] = {\n");
    golden_unit(false, false, 1, NULL);
    printf("};\n");
    return 0;
  }
  golden_unit(true, true, 0, golden_dense_l);
  golden_unit(false, true, 0, golden_sparse_l);
  if (vf_use_avx2()) {
    golden_unit(true, false, 0, golden_dense_q);
    golden_unit(false, false, 0, golden_sparse_q);
  }
  return main_tests();
}

/* ---------------- 8. mask mode: the pyramid the fleet writes ----------------
 * Level 0 stores the published mask itself as volcomp MASK chunks, levels 1..3
 * its successive 2x majority pools. Nothing here depends on the ramp, so the
 * checks are structural: every chunk is a mask chunk, its block centres come
 * back 0/255, and the level above is exactly the majority pool of the level
 * below's own stored grid. */
static void mask_write_source(const char *src, const uint8_t *mask, int64_t S0, int64_t S1, int64_t S2,
                              int64_t csize) {
  uint8_t *chunk = malloc((size_t)(csize * csize * csize));
  uint8_t *cbuf = malloc((size_t)(csize * csize * csize) + BLOSC1_HEADER);
  for (int64_t cz = 0; cz * csize < S0; cz++)
    for (int64_t cy = 0; cy * csize < S1; cy++)
      for (int64_t cx = 0; cx * csize < S2; cx++) {
        memset(chunk, 0, (size_t)(csize * csize * csize));
        int any = 0;
        for (int64_t z = 0; z < csize && cz * csize + z < S0; z++)
          for (int64_t y = 0; y < csize && cy * csize + y < S1; y++)
            for (int64_t x = 0; x < csize && cx * csize + x < S2; x++) {
              uint8_t v = mask[((cz * csize + z) * S1 + cy * csize + y) * S2 + cx * csize + x];
              chunk[(z * csize + y) * csize + x] = v;
              any |= v != 0;
            }
        if (!any) continue;
        size_t bn = blosc_memcpyed(cbuf, chunk, (size_t)(csize * csize * csize));
        char p[4200];
        snprintf(p, sizeof p, "%s/%lld_%lld_%lld.blosc", src, (long long)cz, (long long)cy, (long long)cx);
        FILE *f = fopen(p, "wb");
        CHECK(f && fwrite(cbuf, 1, bn, f) == bn);
        if (f) fclose(f);
      }
  free(chunk), free(cbuf);
}
/* the stored 64^3 grids of one shard level, as one (1024 >> level)/2 cube */
static uint8_t *mask_level_grids(const char *out, int level, int64_t *edge) {
  char p[4200];
  snprintf(p, sizeof p, "%s/%d.shard", out, level);
  size_t n = 0;
  uint8_t *img = shard_read_file(p, &n);
  if (!img) return NULL;
  unsigned g = (unsigned)((SURF_SHARD >> level) / SURF_CHUNK), nch = g * g * g;
  uint64_t *off = malloc(nch * sizeof *off), *nb = malloc(nch * sizeof *nb);
  CHECK(shard_entries(img, n, nch, off, nb));
  int64_t e = (SURF_SHARD >> level) / 2;
  uint8_t *grids = calloc(1, (size_t)(e * e * e));
  uint8_t *one = malloc(VOLCOMP_MASK_VOXELS);
  const int64_t G = VOLCOMP_MASK_DIM;
  for (unsigned i = 0; i < nch; i++) {
    if (off[i] == ~0ull) continue;
    uint32_t dim = 0;
    CHECK_EQ(volcomp_mask_info(img + off[i], (size_t)nb[i], &dim), VOLCOMP_OK);
    CHECK_EQ(dim, VOLCOMP_MASK_DIM);
    CHECK_EQ(volcomp_mask_decode_stored(img + off[i], (size_t)nb[i], one, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
    int64_t o[3] = {(i / (g * g)) * G, ((i / g) % g) * G, (i % g) * G};
    for (int64_t z = 0; z < G; z++)
      for (int64_t y = 0; y < G; y++)
        memcpy(grids + (((o[0] + z) * e + o[1] + y) * e + o[2]), one + (z * G + y) * G, (size_t)G);
  }
  free(off), free(nb), free(img), free(one);
  *edge = e;
  return grids;
}
static void test_mask_mode(void) {
  char dir[] = "/tmp/volcomp_maskmode_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char src[4096], out[4096];
  snprintf(src, sizeof src, "%s/src", dir);
  snprintf(out, sizeof out, "%s/out", dir);
  mkdir(src, 0700);
  mkdir(out, 0700);
  const int64_t S0 = 400, S1 = 330, S2 = 350, csize = 192;
  uint8_t *mask = malloc((size_t)(S0 * S1 * S2));
  golden_mask(mask, S0, S1, S2, true);
  mask_write_source(src, mask, S0, S1, S2, csize);
  const double scale = 9.6 / 9.362;
  surf_cfg cfg = {.srcdir = src,
                  .outdir = out,
                  .csize = csize,
                  .src_shape = {S0, S1, S2},
                  .shard = {0, 0, 0},
                  .scale = scale,
                  .mask_mode = true,
                  .dmax = SURF_DMAX,
                  .threads = 3,
                  .samples = 0};
  for (int d = 0; d < 3; d++) cfg.out_shape[d] = (int64_t)floor((double)cfg.src_shape[d] / scale + 0.5);
  surf_result res;
  CHECK_EQ(surface_pack(&cfg, &res), 0);
  CHECK(res.present[0] > 0);

  /* every stored chunk is a mask chunk whose block centres decode 0/255, and the
   * level above is the majority pool of this level's stored grids */
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS), *grid = malloc(VOLCOMP_MASK_VOXELS);
  for (int L = 0; L < SURF_LEVELS; L++) {
    char p[4200];
    snprintf(p, sizeof p, "%s/%d.shard", out, L);
    size_t n = 0;
    uint8_t *img = shard_read_file(p, &n);
    if (!img) continue;
    unsigned g = (unsigned)((SURF_SHARD >> L) / SURF_CHUNK), nch = g * g * g;
    uint64_t *off = malloc(nch * sizeof *off), *nb = malloc(nch * sizeof *nb);
    CHECK(shard_entries(img, n, nch, off, nb));
    unsigned checked = 0;
    for (unsigned i = 0; i < nch && checked < 4; i++) {
      if (off[i] == ~0ull) continue;
      checked++;
      CHECK_EQ(volcomp_decode(img + off[i], (size_t)nb[i], dec, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
      CHECK_EQ(volcomp_mask_decode_stored(img + off[i], (size_t)nb[i], grid, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
      unsigned bad = 0;
      const int64_t G = VOLCOMP_MASK_DIM, C = SURF_CHUNK;
      for (int64_t z = 0; z < G; z++)
        for (int64_t y = 0; y < G; y++)
          for (int64_t x = 0; x < G; x++)
            bad += dec[((2 * z) * C + 2 * y) * C + 2 * x] != grid[(z * G + y) * G + x];
      CHECK_EQ(bad, 0);
    }
    free(off), free(nb), free(img);
  }
  for (int L = 0; L + 1 < SURF_LEVELS; L++) {
    int64_t e0 = 0, e1 = 0;
    uint8_t *a = mask_level_grids(out, L, &e0), *b = mask_level_grids(out, L + 1, &e1);
    if (a && b) {
      /* grids of level L ARE level L+1's mask; level L+1's grids are its majority pool */
      unsigned bad = 0;
      for (int64_t z = 0; z < e1; z++)
        for (int64_t y = 0; y < e1; y++)
          for (int64_t x = 0; x < e1; x++) {
            unsigned c = 0;
            for (int dz = 0; dz < 2; dz++)
              for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                  c += a[(((2 * z + dz) * e0 + 2 * y + dy) * e0) + 2 * x + dx] != 0;
            bad += b[(z * e1 + y) * e1 + x] != (c * 2 >= 8 ? 255u : 0u);
          }
      CHECK_EQ(bad, 0);
    }
    free(a), free(b);
  }
  /* shard-pool carries the same pyramid up: the level-4 chunk is the assembly of
   * the eight level-3 stored grids below it (only the first is present here), so
   * its own stored grid is the majority pool of that */
  {
    char p3[4200], outp[4200];
    snprintf(p3, sizeof p3, "%s/3.shard", out);
    snprintf(outp, sizeof outp, "%s/pooled.shard", dir);
    const char *in[8] = {p3, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    int64_t shape[3] = {128, 128, 128}, pos[3] = {0, 0, 0};
    unsigned present = 0;
    uint64_t bytes = 0;
    CHECK_EQ(surface_pool_shard(in, outp, 0.0f, shape, pos, true, false, &present, &bytes), 0);
    int64_t e3 = 0;
    uint8_t *a = mask_level_grids(out, 3, &e3);  /* the level-4 mask, 64^3 */
    CHECK(a && e3 == VOLCOMP_MASK_DIM);
    size_t n = 0;
    uint8_t *img = shard_read_file(outp, &n);
    CHECK_EQ(present, 1);
    if (a && img && present) {
      uint64_t off[1], nb[1];
      CHECK(shard_entries(img, n, 1, off, nb));
      uint8_t *g4 = malloc(VOLCOMP_MASK_VOXELS);
      CHECK_EQ(volcomp_mask_decode_stored(img + off[0], (size_t)nb[0], g4, VOLCOMP_MASK_VOXELS), VOLCOMP_OK);
      const int64_t G = VOLCOMP_MASK_DIM;
      unsigned bad = 0;
      for (int64_t z = 0; z < G; z++)
        for (int64_t y = 0; y < G; y++)
          for (int64_t x = 0; x < G; x++) {
            unsigned c = 0;
            if (z < G / 2 && y < G / 2 && x < G / 2)
              for (int dz = 0; dz < 2; dz++)
                for (int dy = 0; dy < 2; dy++)
                  for (int dx = 0; dx < 2; dx++)
                    c += a[(((2 * z + dz) * G + 2 * y + dy) * G) + 2 * x + dx] != 0;
            bad += g4[(z * G + y) * G + x] != (c * 2 >= 8 ? 255u : 0u);
          }
      CHECK_EQ(bad, 0);
      free(g4);
    }
    free(a), free(img);
  }
  free(mask), free(dec), free(grid);
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}

/* The decoded voxels of one level of a mask-lossless pack, over the level's real
 * extent (ext), assembled from its shard's 128^3 chunks. */
static uint8_t *ll_level_volume(const char *out, int level, const int64_t ext[3]) {
  char p[4200];
  snprintf(p, sizeof p, "%s/%d.shard", out, level);
  size_t n = 0;
  uint8_t *img = shard_read_file(p, &n);
  if (!img) return NULL;
  unsigned g = (unsigned)((SURF_SHARD >> level) / SURF_CHUNK), nch = g * g * g;
  uint64_t *off = malloc(nch * sizeof *off), *nb = malloc(nch * sizeof *nb);
  CHECK(shard_entries(img, n, nch, off, nb));
  uint8_t *vol = calloc(1, (size_t)(ext[0] * ext[1] * ext[2]));
  uint8_t *one = malloc(VOLCOMP_CHUNK_VOXELS);
  const int64_t C = SURF_CHUNK;
  for (unsigned i = 0; i < nch; i++) {
    if (off[i] == ~0ull) continue;
    uint32_t dim = 0;
    CHECK_EQ(volcomp_mask_info(img + off[i], (size_t)nb[i], &dim), VOLCOMP_OK);
    CHECK_EQ(dim, VOLCOMP_CHUNK_DIM); /* mode 5: the grid IS the chunk */
    CHECK_EQ(volcomp_decode(img + off[i], (size_t)nb[i], one, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
    int64_t o[3] = {(int64_t)(i / (g * g)) * C, (int64_t)((i / g) % g) * C, (int64_t)(i % g) * C};
    for (int64_t z = 0; z < C && o[0] + z < ext[0]; z++)
      for (int64_t y = 0; y < C && o[1] + y < ext[1]; y++) {
        int64_t w = ext[2] - o[2] < C ? ext[2] - o[2] : C;
        if (w > 0)
          memcpy(vol + (((o[0] + z) * ext[1] + o[1] + y) * ext[2] + o[2]), one + (z * C + y) * C, (size_t)w);
      }
  }
  free(off), free(nb), free(img), free(one);
  return vol;
}

/* mask-lossless + no resample: level 0 must be the published mask voxel for
 * voxel, and every level above it the 2x2x2 majority pool of the one below. */
static void test_mask_lossless_mode(void) {
  char dir[] = "/tmp/volcomp_maskll_XXXXXX";
  CHECK(mkdtemp(dir) != NULL);
  char src[4096], out[4096];
  snprintf(src, sizeof src, "%s/src", dir);
  snprintf(out, sizeof out, "%s/out", dir);
  mkdir(src, 0700);
  mkdir(out, 0700);
  const int64_t S0 = 400, S1 = 330, S2 = 350, csize = 192;
  uint8_t *mask = malloc((size_t)(S0 * S1 * S2));
  golden_mask(mask, S0, S1, S2, true);
  mask_write_source(src, mask, S0, S1, S2, csize);
  surf_cfg cfg = {.srcdir = src,
                  .outdir = out,
                  .csize = csize,
                  .src_shape = {S0, S1, S2},
                  .out_shape = {S0, S1, S2}, /* --no-resample: the source grid itself */
                  .shard = {0, 0, 0},
                  .scale = 1.0,
                  .mask_mode = true,
                  .mask_lossless = true,
                  .dmax = SURF_DMAX,
                  .threads = 3,
                  .samples = 0};
  surf_result res;
  CHECK_EQ(surface_pack(&cfg, &res), 0);
  CHECK(res.present[0] > 0);

  int64_t ext[SURF_LEVELS][3];
  for (int d = 0; d < 3; d++) {
    ext[0][d] = cfg.src_shape[d];
    for (int L = 1; L < SURF_LEVELS; L++) ext[L][d] = (ext[L - 1][d] + 1) / 2;
  }
  uint8_t *lvl[SURF_LEVELS] = {0};
  for (int L = 0; L < SURF_LEVELS; L++) lvl[L] = ll_level_volume(out, L, ext[L]);
  CHECK(lvl[0] != NULL);
  if (lvl[0]) { /* level 0 is the published mask, exactly */
    unsigned bad = 0;
    for (size_t i = 0; i < (size_t)(S0 * S1 * S2); i++) bad += lvl[0][i] != (mask[i] ? 255u : 0u);
    CHECK_EQ(bad, 0);
  }
  for (int L = 0; L + 1 < SURF_LEVELS; L++) {
    if (!lvl[L] || !lvl[L + 1]) continue;
    const int64_t *a = ext[L], *b = ext[L + 1];
    unsigned bad = 0;
    for (int64_t z = 0; z < b[0]; z++)
      for (int64_t y = 0; y < b[1]; y++)
        for (int64_t x = 0; x < b[2]; x++) {
          unsigned c = 0; /* voxels outside the array count as air */
          for (int dz = 0; dz < 2; dz++)
            for (int dy = 0; dy < 2; dy++)
              for (int dx = 0; dx < 2; dx++) {
                int64_t zz = 2 * z + dz, yy = 2 * y + dy, xx = 2 * x + dx;
                if (zz < a[0] && yy < a[1] && xx < a[2]) c += lvl[L][(zz * a[1] + yy) * a[2] + xx] != 0;
              }
          bad += lvl[L + 1][(z * b[1] + y) * b[2] + x] != (c * 2 >= 8 ? 255u : 0u);
        }
    CHECK_EQ(bad, 0);
  }
  /* shard-pool carries the same majority pool up, decoding whole chunks */
  {
    char p3[4200], outp[4200];
    snprintf(p3, sizeof p3, "%s/3.shard", out);
    snprintf(outp, sizeof outp, "%s/pooled.shard", dir);
    const char *in[8] = {p3, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    int64_t shape[3] = {ext[3][0], ext[3][1], ext[3][2]}, pos[3] = {0, 0, 0};
    unsigned present = 0;
    uint64_t bytes = 0;
    CHECK_EQ(surface_pool_shard(in, outp, 0.0f, shape, pos, true, true, &present, &bytes), 0);
    size_t n = 0;
    uint8_t *img = shard_read_file(outp, &n);
    CHECK_EQ(present, 1);
    if (img && present && lvl[3]) {
      uint64_t off[1], nb[1];
      CHECK(shard_entries(img, n, 1, off, nb));
      uint32_t dim = 0;
      CHECK_EQ(volcomp_mask_info(img + off[0], (size_t)nb[0], &dim), VOLCOMP_OK);
      CHECK_EQ(dim, VOLCOMP_CHUNK_DIM);
      uint8_t *g4 = malloc(VOLCOMP_CHUNK_VOXELS);
      CHECK_EQ(volcomp_decode(img + off[0], (size_t)nb[0], g4, VOLCOMP_CHUNK_VOXELS), VOLCOMP_OK);
      const int64_t *a = ext[3], C = SURF_CHUNK;
      unsigned bad = 0;
      for (int64_t z = 0; z < C; z++)
        for (int64_t y = 0; y < C; y++)
          for (int64_t x = 0; x < C; x++) {
            unsigned c = 0;
            for (int dz = 0; dz < 2; dz++)
              for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                  int64_t zz = 2 * z + dz, yy = 2 * y + dy, xx = 2 * x + dx;
                  if (zz < a[0] && yy < a[1] && xx < a[2]) c += lvl[3][(zz * a[1] + yy) * a[2] + xx] != 0;
                }
            bad += g4[(z * C + y) * C + x] != (c * 2 >= 8 ? 255u : 0u);
          }
      CHECK_EQ(bad, 0);
      free(g4);
    }
    free(img);
  }
  for (int L = 0; L < SURF_LEVELS; L++) free(lvl[L]);
  free(mask);
  char cmd[4300];
  snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
  if (system(cmd)) { /* best effort */
  }
}

static int main_tests(void) {
  test_ramp_values();
  test_tiling();
  test_blosc_fixture();
  test_end_to_end(1.0, 192);   /* identity ladder, source chunks larger than the output tiling */
  test_end_to_end(1.25, 128);
  test_pool_shard();
  test_occupancy();
  test_maxpool_check();
  test_mask_mode();
  test_mask_lossless_mode();
  TEST_END();
}
