/* surface-pack: one export unit of a published surface-prediction volume.
 *
 * Input : the raw zarr v2 (blosc/zstd) chunks of ONE level-0 source array that
 *         cover one output shard's footprint, as files SRCDIR/<cz>_<cy>_<cx>.blosc
 *         (absolute source chunk indices; a missing file is all-zero).
 * Output: OUTDIR/<L>.shard for L = 0..3 — zarr v3 sharding_indexed images of
 *         128^3 volcomp chunks, shard edge 1024 / 512 / 256 / 128, all covering
 *         the same physical footprint, so a unit owns whole shard files at every
 *         level it writes and nothing crosses units.
 *
 * Per output voxel:
 *   1. the published mask (0 / 255) is turned into a signed-distance ramp at the
 *      SOURCE resolution: s = signed Euclidean distance to the mask boundary in
 *      source voxels, positive inside, clipped to [-dmax, +dmax] (dmax = 3), and
 *      value = round(127.5 + 42.5 * s) -> inside 1,2,3 voxels deep: 170, 213, 255;
 *      outside: 85, 43, 0; the 128 crossing lies exactly on the published edge.
 *      The distance is exact: with the clip at 3 the nearest opposite-class voxel
 *      has |dz|,|dy|,|dx| <= 3, so three min-plus passes with a +-3 window over the
 *      squared distance are exactly Euclidean (and each output chunk is computed
 *      with a 3-voxel halo, so the result does not depend on the tiling).
 *   2. the ramp is trilinearly resampled onto the exact ladder grid: output voxel
 *      i samples source coordinate i * scale, scale = rung_um / native_um (the
 *      identity, and skipped, when the source is already exactly on the rung).
 *      The mask itself is never resampled - only the continuous ramp.
 *   3. levels 1..3 are exact 2x mean pooling of the level-0 ramp (float mean,
 *      rounded), computed inside the unit.
 *
 * Memory: the source mask region of one unit (~(1024 * scale + 8)^3 bytes, 1.1-1.3
 * GB) plus ~15 MB of scratch per thread plus the level-1 block (134 MB); no float
 * volume is ever materialised.
 */
#ifndef VOLCOMP_SURFACE_PACK_H
#define VOLCOMP_SURFACE_PACK_H

#include "../../volcomp.h"
#include "blosc1.h"
#include "shard_pack.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SURF_LEVELS 4        /* levels produced by one unit: 0, 1, 2, 3 */
#define SURF_SHARD 1024      /* level-0 shard edge in output voxels */
#define SURF_CHUNK VOLCOMP_CHUNK_DIM
#define SURF_DMAX 3          /* ramp clip, in source voxels */
#define SURF_INF 10u         /* squared distance "> dmax^2" */

typedef struct {
  const char *srcdir, *outdir;
  int64_t csize;              /* source chunk edge (128 / 192 / 256) */
  int64_t src_shape[3];       /* level-0 source array shape */
  int64_t out_shape[3];       /* level-0 output shape = round(src_shape / scale) */
  int64_t shard[3];           /* shard index (same for every level) */
  double scale;               /* source voxels per output voxel = rung_um / native_um */
  float q[SURF_LEVELS];
  int dmax, threads, samples;
} surf_cfg;

typedef struct {
  unsigned present[SURF_LEVELS], chunks[SURF_LEVELS];
  uint64_t bytes[SURF_LEVELS];
  double psnr_min;
  unsigned max_err, compared;
  double t_decode, t_ramp, t_encode;
  uint64_t out_voxels, src_chunks;
  bool src_nonzero;
} surf_result;

static double surf_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static inline int64_t surf_min64(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t surf_max64(int64_t a, int64_t b) { return a > b ? a : b; }

/* ---------------------------------------------------------------- ramp tables */

/* value for a voxel whose squared distance to the opposite class is d2 (capped) */
static void surf_ramp_tables(int dmax, uint8_t in[SURF_INF + 1], uint8_t out[SURF_INF + 1]) {
  for (unsigned d2 = 0; d2 <= SURF_INF; d2++) {
    double s = sqrt((double)d2);
    if (s > (double)dmax || d2 >= SURF_INF) s = (double)dmax;
    double step = 127.5 / (double)dmax;   /* 42.5 for dmax = 3 */
    double vi = floor(127.5 + step * s + 0.5), vo = floor(127.5 - step * s + 0.5);
    in[d2] = (uint8_t)(vi < 0 ? 0 : vi > 255 ? 255 : vi);
    out[d2] = (uint8_t)(vo < 0 ? 0 : vo > 255 ? 255 : vo);
  }
}

/* ------------------------------------------------------- exact clipped EDT ---
 * Three min-plus passes with a +-dmax window over squared distances. Starting
 * from seed[i] = 0 at seeds and SURF_INF elsewhere, the result is the exact
 * squared Euclidean distance to the nearest seed whenever that distance is
 * <= dmax^2, and >= SURF_INF otherwise. Separable and exact because every
 * candidate seed within the clip has all three offsets within +-dmax. */

static void surf_pass_x(uint8_t *dst, const uint8_t *src, int64_t mz, int64_t my, int64_t mx, int R) {
  for (int64_t line = 0; line < mz * my; line++) {
    const uint8_t *s = src + line * mx;
    uint8_t *d = dst + line * mx;
    memcpy(d, s, (size_t)mx);
    for (int k = 1; k <= R; k++) {
      unsigned kk = (unsigned)(k * k);
      if (kk >= SURF_INF) break;
      for (int64_t x = k; x < mx; x++) {
        unsigned v = s[x - k] + kk;
        if (v < d[x]) d[x] = (uint8_t)v;
      }
      for (int64_t x = 0; x + k < mx; x++) {
        unsigned v = s[x + k] + kk;
        if (v < d[x]) d[x] = (uint8_t)v;
      }
    }
    for (int64_t x = 0; x < mx; x++)
      if (d[x] > SURF_INF) d[x] = (uint8_t)SURF_INF;
  }
}
/* one pass along an axis whose lines are `span` contiguous elements apart */
static void surf_pass_strided(uint8_t *dst, const uint8_t *src, int64_t nlines, int64_t span,
                              int64_t group, int R) {
  /* the volume is `group` blocks of `nlines` lines of `span` contiguous elements */
  for (int64_t g = 0; g < group; g++)
    for (int64_t l = 0; l < nlines; l++) {
      int64_t base = (g * nlines + l) * span;
      uint8_t *d = dst + base;
      memcpy(d, src + base, (size_t)span);
      for (int k = 1; k <= R; k++) {
        unsigned kk = (unsigned)(k * k);
        if (kk >= SURF_INF) break;
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          int64_t ll = l + sgn * k;
          if (ll < 0 || ll >= nlines) continue;
          const uint8_t *s = src + (g * nlines + ll) * span;
          for (int64_t i = 0; i < span; i++) {
            unsigned v = s[i] + kk;
            if (v < d[i]) d[i] = (uint8_t)v;
          }
        }
      }
      for (int64_t i = 0; i < span; i++)
        if (d[i] > SURF_INF) d[i] = (uint8_t)SURF_INF;
    }
}
/* squared distance field of `seed` (0 at seeds, SURF_INF elsewhere) into A; B is scratch */
static void surf_edt(uint8_t *A, uint8_t *B, const uint8_t *seed, int64_t mz, int64_t my, int64_t mx, int R) {
  surf_pass_x(A, seed, mz, my, mx, R);              /* x */
  surf_pass_strided(B, A, my, mx, mz, R);           /* y: my lines of mx, mz groups */
  surf_pass_strided(A, B, mz, my * mx, 1, R);       /* z: mz planes of my*mx */
}

/* Ramp over a sub-box of the 0/1 mask `loc` (dims m, which must include a dmax
 * halo on every side): writes `ramp` for the whole box, but only the voxels at
 * least dmax from the box edge are meaningful - those are exact and independent
 * of how the volume was tiled. `a`, `b`, `s` are scratch of the box size; `loc`
 * is destroyed. */
static void surf_ramp_region(uint8_t *ramp, uint8_t *loc, int64_t mz, int64_t my, int64_t mx, int dmax,
                             const uint8_t *in_tab, const uint8_t *out_tab, uint8_t *a, uint8_t *b, uint8_t *s) {
  size_t mn = (size_t)(mz * my * mx);
  for (size_t k = 0; k < mn; k++) s[k] = loc[k] ? (uint8_t)SURF_INF : 0;   /* seeds: background */
  surf_edt(a, b, s, mz, my, mx, dmax);                                     /* a = d^2 to background */
  for (size_t k = 0; k < mn; k++) loc[k] = loc[k] ? 0 : (uint8_t)SURF_INF; /* seeds: foreground */
  surf_edt(b, s, loc, mz, my, mx, dmax);                                   /* b = d^2 to foreground */
  for (size_t k = 0; k < mn; k++) ramp[k] = a[k] ? in_tab[a[k]] : out_tab[b[k]];
}

/* ------------------------------------------------------------- parallel for */

typedef void (*surf_job)(void *ctx, int64_t i, int tid);
typedef struct {
  surf_job fn;
  void *ctx;
  int64_t n;
  atomic_int_least64_t next;
  atomic_int tid_next;
  volatile int failed;
} surf_par;

static void *surf_par_thread(void *arg) {
  surf_par *p = arg;
  int tid = atomic_fetch_add(&p->tid_next, 1);
  for (;;) {
    int64_t i = atomic_fetch_add(&p->next, 1);
    if (i >= p->n || p->failed) return NULL;
    p->fn(p->ctx, i, tid);
  }
}
static void surf_parallel_for(int64_t n, int threads, surf_job fn, void *ctx) {
  surf_par p = {.fn = fn, .ctx = ctx, .n = n};
  atomic_init(&p.next, 0);
  atomic_init(&p.tid_next, 0);
  if (threads < 1) threads = 1;
  if ((int64_t)threads > n) threads = (int)(n > 0 ? n : 1);
  pthread_t *th = calloc((size_t)threads, sizeof *th);
  int started = 0;
  for (int i = 0; i < threads; i++)
    if (pthread_create(&th[i], NULL, surf_par_thread, &p) == 0) started++;
  if (started == 0) surf_par_thread(&p);  /* no threads available: run inline */
  for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
  free(th);
}

/* ------------------------------------------------------------------- state */

#define SURF_MAX_CHUNKS 512u  /* 8^3 inner chunks in the level-0 shard */

typedef struct {
  uint8_t *loc, *s, *a, *b, *ramp; /* per-thread scratch over the local sub-box */
  uint8_t *chunk;             /* 128^3 output chunk */
  uint8_t *enc, *dec;         /* encode/decode scratch */
  uint8_t *src;               /* one decompressed source chunk */
  uint8_t *tmp;               /* blosc unshuffle scratch */
} surf_scratch;

typedef struct {
  const surf_cfg *cfg;
  int64_t rlo[3], rn[3];      /* native mask region (may start below 0 / end past the array) */
  uint8_t *mask;              /* rn[0]*rn[1]*rn[2], 0 or 1 */
  int64_t co[3], ce[3];       /* output level-0 core extent of this shard */
  int64_t clo[3], chi[3];     /* source chunk index range covering the region */
  int64_t ldim[SURF_LEVELS];  /* shard edge per level */
  int64_t lext[SURF_LEVELS][3];
  uint8_t *lvl[SURF_LEVELS];  /* pooled shard volumes for levels 1..3 */
  uint8_t *enc[SURF_LEVELS][SURF_MAX_CHUNKS];
  size_t encn[SURF_LEVELS][SURF_MAX_CHUNKS];
  uint8_t in_tab[SURF_INF + 1], out_tab[SURF_INF + 1];
  surf_scratch *sc;
  int nthreads;
  int64_t mmax[3];            /* scratch sub-box capacity */
  int level;                  /* level currently being encoded */
  long stride;                /* PSNR sampling stride */
  pthread_mutex_t lock;
  surf_result *res;
  volatile int error;
  bool identity;
} surf_state;

/* ------------------------------------------------------- source assembly --- */

static void surf_decode_job(void *ctx, int64_t i, int tid) {
  surf_state *S = ctx;
  const surf_cfg *cfg = S->cfg;
  int64_t nx = S->chi[2] - S->clo[2] + 1, ny = S->chi[1] - S->clo[1] + 1;
  int64_t cx = S->clo[2] + i % nx, cy = S->clo[1] + (i / nx) % ny, cz = S->clo[0] + i / (nx * ny);
  char path[4096];
  snprintf(path, sizeof path, "%s/%lld_%lld_%lld.blosc", cfg->srcdir, (long long)cz, (long long)cy, (long long)cx);
  size_t n = 0;
  uint8_t *raw = shard_read_file(path, &n);
  if (!raw) return;  /* absent = all zero */
  const int64_t C = cfg->csize;
  const char *err = NULL;
  surf_scratch *sc = &S->sc[tid];
  int64_t got = blosc1_decompress(raw, n, sc->src, (size_t)(C * C * C), sc->tmp, (size_t)(C * C * C), &err);
  free(raw);
  if (got != C * C * C) {
    fprintf(stderr, "surface-pack: %s: %s\n", path, err ? err : "short blosc chunk");
    S->error = 1;
    return;
  }
  /* copy the part of this chunk that lands in the region */
  int64_t g0[3] = {cz * C, cy * C, cx * C};
  int64_t lo[3], hi[3];
  for (int d = 0; d < 3; d++) {
    lo[d] = surf_max64(g0[d], surf_max64(S->rlo[d], 0));
    hi[d] = surf_min64(g0[d] + C, surf_min64(S->rlo[d] + S->rn[d], cfg->src_shape[d]));
    if (hi[d] <= lo[d]) return;
  }
  bool nonzero = false;
  for (int64_t z = lo[0]; z < hi[0]; z++)
    for (int64_t y = lo[1]; y < hi[1]; y++) {
      const uint8_t *s = sc->src + ((z - g0[0]) * C + (y - g0[1])) * C + (lo[2] - g0[2]);
      uint8_t *d = S->mask + ((z - S->rlo[0]) * S->rn[1] + (y - S->rlo[1])) * S->rn[2] + (lo[2] - S->rlo[2]);
      for (int64_t x = 0; x < hi[2] - lo[2]; x++) {
        d[x] = s[x] != 0;
        nonzero |= d[x] != 0;
      }
    }
  if (nonzero) {
    pthread_mutex_lock(&S->lock);
    S->res->src_nonzero = true;
    pthread_mutex_unlock(&S->lock);
  }
}

/* ---------------------------------------------------------------- encoding */

static void surf_encode_chunk(surf_state *S, int level, unsigned idx, const uint8_t *chunk, int tid) {
  bool zero = true;
  for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS && zero; k++) zero = chunk[k] == 0;
  if (zero) return;
  surf_scratch *sc = &S->sc[tid];
  size_t en = 0;
  if (volcomp_encode(chunk, S->cfg->q[level], sc->enc, VOLCOMP_ENCODE_BOUND, &en) != VOLCOMP_OK) {
    fprintf(stderr, "surface-pack: encode failed at level %d chunk %u\n", level, idx);
    S->error = 1;
    return;
  }
  uint8_t *keep = malloc(en);
  memcpy(keep, sc->enc, en);
  /* every stored chunk must decode; a sampled subset is also compared with the ramp */
  if (volcomp_decode(keep, en, sc->dec, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK) {
    fprintf(stderr, "surface-pack: level %d chunk %u does not decode\n", level, idx);
    S->error = 1;
    free(keep);
    return;
  }
  double psnr = -1;
  unsigned mx = 0;
  if ((long)idx % S->stride == 0) {
    double se = 0;
    for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS; k++) {
      int d = (int)chunk[k] - (int)sc->dec[k];
      unsigned ad = (unsigned)(d < 0 ? -d : d);
      se += (double)d * d;
      if (ad > mx) mx = ad;
    }
    psnr = se == 0 ? 999.0 : 10.0 * log10(65025.0 * (double)VOLCOMP_CHUNK_VOXELS / se);
  }
  pthread_mutex_lock(&S->lock);
  S->enc[level][idx] = keep;
  S->encn[level][idx] = en;
  S->res->present[level]++;
  S->res->bytes[level] += en;
  if (psnr >= 0) {
    if (psnr < S->res->psnr_min) S->res->psnr_min = psnr;
    if (mx > S->res->max_err) S->res->max_err = mx;
    S->res->compared++;
  }
  pthread_mutex_unlock(&S->lock);
}

/* ------------------------------------------------------------ level 0 work */

static void surf_level0_job(void *ctx, int64_t i, int tid) {
  surf_state *S = ctx;
  const surf_cfg *cfg = S->cfg;
  surf_scratch *sc = &S->sc[tid];
  const int R = cfg->dmax;
  int64_t ci[3] = {i >> 6, (i >> 3) & 7, i & 7};
  int64_t o0[3], o1[3], ext[3];
  for (int d = 0; d < 3; d++) {
    o0[d] = S->co[d] + ci[d] * SURF_CHUNK;
    o1[d] = surf_min64(o0[d] + SURF_CHUNK, S->ce[d]);
    ext[d] = o1[d] - o0[d];
    if (ext[d] <= 0) return;  /* wholly outside the array */
  }
  /* native sub-box: interpolation support of this chunk, dilated by dmax */
  int64_t j0[3], m[3];
  for (int d = 0; d < 3; d++) {
    int64_t s0 = (int64_t)floor((double)o0[d] * cfg->scale);
    int64_t s1 = (int64_t)floor((double)(o1[d] - 1) * cfg->scale) + (S->identity ? 0 : 1);
    j0[d] = s0 - R;
    m[d] = s1 + R - j0[d] + 1;
    if (m[d] > S->mmax[d]) {  /* cannot happen; guard against a bad --scale */
      fprintf(stderr, "surface-pack: sub-box %lld exceeds scratch %lld\n", (long long)m[d], (long long)S->mmax[d]);
      S->error = 1;
      return;
    }
  }
  /* gather the mask sub-box */
  for (int64_t z = 0; z < m[0]; z++)
    for (int64_t y = 0; y < m[1]; y++) {
      const uint8_t *s = S->mask + ((j0[0] + z - S->rlo[0]) * S->rn[1] + (j0[1] + y - S->rlo[1])) * S->rn[2] +
                         (j0[2] - S->rlo[2]);
      memcpy(sc->loc + (z * m[1] + y) * m[2], s, (size_t)m[2]);
    }
  uint8_t *ramp = sc->s;
  surf_ramp_region(ramp, sc->loc, m[0], m[1], m[2], R, S->in_tab, S->out_tab, sc->a, sc->b, sc->ramp);
  /* resample onto the exact ladder grid */
  memset(sc->chunk, 0, VOLCOMP_CHUNK_VOXELS);
  if (S->identity) {
    for (int64_t z = 0; z < ext[0]; z++)
      for (int64_t y = 0; y < ext[1]; y++) {
        const uint8_t *s = ramp + ((z + R) * m[1] + (y + R)) * m[2] + R;
        memcpy(sc->chunk + (z * SURF_CHUNK + y) * SURF_CHUNK, s, (size_t)ext[2]);
      }
  } else {
    for (int64_t z = 0; z < ext[0]; z++) {
      double tz = (double)(o0[0] + z) * cfg->scale;
      int64_t az = (int64_t)floor(tz) - j0[0];
      double fz = tz - floor(tz), gz = 1.0 - fz;
      for (int64_t y = 0; y < ext[1]; y++) {
        double ty = (double)(o0[1] + y) * cfg->scale;
        int64_t ay = (int64_t)floor(ty) - j0[1];
        double fy = ty - floor(ty), gy = 1.0 - fy;
        const uint8_t *p000 = ramp + (az * m[1] + ay) * m[2], *p001 = p000 + m[2];
        const uint8_t *p100 = p000 + m[1] * m[2], *p101 = p100 + m[2];
        uint8_t *d = sc->chunk + (z * SURF_CHUNK + y) * SURF_CHUNK;
        for (int64_t x = 0; x < ext[2]; x++) {
          double tx = (double)(o0[2] + x) * cfg->scale;
          int64_t ax = (int64_t)floor(tx) - j0[2];
          double fx = tx - floor(tx), gx = 1.0 - fx;
          double v = gz * (gy * (gx * p000[ax] + fx * p000[ax + 1]) + fy * (gx * p001[ax] + fx * p001[ax + 1])) +
                     fz * (gy * (gx * p100[ax] + fx * p100[ax + 1]) + fy * (gx * p101[ax] + fx * p101[ax + 1]));
          v = floor(v + 0.5);
          d[x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
      }
    }
  }
  /* 2x mean pooling into the level-1 block of this chunk (disjoint per chunk) */
  int64_t h[3] = {(ext[0] + 1) / 2, (ext[1] + 1) / 2, (ext[2] + 1) / 2};
  uint8_t *l1 = S->lvl[1];
  int64_t d1 = S->ldim[1];
  for (int64_t z = 0; z < h[0]; z++)
    for (int64_t y = 0; y < h[1]; y++)
      for (int64_t x = 0; x < h[2]; x++) {
        unsigned sum = 0, cnt = 0;
        for (int64_t dz = 0; dz < 2 && 2 * z + dz < ext[0]; dz++)
          for (int64_t dy = 0; dy < 2 && 2 * y + dy < ext[1]; dy++)
            for (int64_t dx = 0; dx < 2 && 2 * x + dx < ext[2]; dx++) {
              sum += sc->chunk[((2 * z + dz) * SURF_CHUNK + (2 * y + dy)) * SURF_CHUNK + 2 * x + dx];
              cnt++;
            }
        int64_t oz = ci[0] * (SURF_CHUNK / 2) + z, oy = ci[1] * (SURF_CHUNK / 2) + y, ox = ci[2] * (SURF_CHUNK / 2) + x;
        l1[(oz * d1 + oy) * d1 + ox] = (uint8_t)((sum + cnt / 2) / cnt);
      }
  pthread_mutex_lock(&S->lock);
  S->res->out_voxels += (uint64_t)(ext[0] * ext[1] * ext[2]);
  pthread_mutex_unlock(&S->lock);
  surf_encode_chunk(S, 0, (unsigned)i, sc->chunk, tid);
}

/* pool one level into the next inside the shard */
static void surf_pool(surf_state *S, int level) {
  const uint8_t *src = S->lvl[level - 1];
  uint8_t *dst = S->lvl[level];
  int64_t ds = S->ldim[level - 1], dd = S->ldim[level];
  const int64_t *e = S->lext[level - 1];
  for (int64_t z = 0; z < S->lext[level][0]; z++)
    for (int64_t y = 0; y < S->lext[level][1]; y++)
      for (int64_t x = 0; x < S->lext[level][2]; x++) {
        unsigned sum = 0, cnt = 0;
        for (int64_t dz = 0; dz < 2 && 2 * z + dz < e[0]; dz++)
          for (int64_t dy = 0; dy < 2 && 2 * y + dy < e[1]; dy++)
            for (int64_t dx = 0; dx < 2 && 2 * x + dx < e[2]; dx++) {
              sum += src[((2 * z + dz) * ds + (2 * y + dy)) * ds + 2 * x + dx];
              cnt++;
            }
        dst[(z * dd + y) * dd + x] = (uint8_t)((sum + cnt / 2) / cnt);
      }
}

static void surf_levelN_job(void *ctx, int64_t i, int tid) {
  surf_state *S = ctx;
  int level = S->level;
  int64_t g = S->ldim[level] / SURF_CHUNK;
  int64_t ci[3] = {i / (g * g), (i / g) % g, i % g};
  int64_t ext[3];
  for (int d = 0; d < 3; d++) {
    ext[d] = surf_min64(SURF_CHUNK, S->lext[level][d] - ci[d] * SURF_CHUNK);
    if (ext[d] <= 0) return;
  }
  surf_scratch *sc = &S->sc[tid];
  memset(sc->chunk, 0, VOLCOMP_CHUNK_VOXELS);
  int64_t dim = S->ldim[level];
  for (int64_t z = 0; z < ext[0]; z++)
    for (int64_t y = 0; y < ext[1]; y++)
      memcpy(sc->chunk + (z * SURF_CHUNK + y) * SURF_CHUNK,
             S->lvl[level] + ((ci[0] * SURF_CHUNK + z) * dim + ci[1] * SURF_CHUNK + y) * dim + ci[2] * SURF_CHUNK,
             (size_t)ext[2]);
  surf_encode_chunk(S, level, (unsigned)i, sc->chunk, tid);
}

/* --------------------------------------------------------------- shard file */

static int surf_write_shard(const char *path, uint8_t *const *enc, const size_t *n, unsigned count,
                            uint64_t *payload) {
  FILE *out = fopen(path, "wb");
  if (!out) return 2;
  size_t idxn = (size_t)count * 16u + 4u;
  uint8_t *idx = calloc(1, idxn);
  uint64_t off = 0;
  int rc = 0;
  for (unsigned i = 0; i < count; i++) {
    if (!enc[i]) {
      shard_wr_u64(idx + i * 16, ~0ull);
      shard_wr_u64(idx + i * 16 + 8, ~0ull);
      continue;
    }
    if (fwrite(enc[i], 1, n[i], out) != n[i]) {
      rc = 2;
      break;
    }
    shard_wr_u64(idx + i * 16, off);
    shard_wr_u64(idx + i * 16 + 8, n[i]);
    off += n[i];
  }
  if (rc == 0) {
    uint32_t crc = shard_crc32c(idx, (size_t)count * 16u);
    for (unsigned i = 0; i < 4; i++) idx[(size_t)count * 16u + i] = (uint8_t)(crc >> (8 * i));
    if (fwrite(idx, 1, idxn, out) != idxn) rc = 2;
  }
  if (fclose(out) && rc == 0) rc = 2;
  free(idx);
  if (payload) *payload = off;
  return rc;
}

/* ------------------------------------------------------------------ driver */

static int surface_pack(const surf_cfg *cfg, surf_result *res) {
  surf_state S = {.cfg = cfg, .res = res};
  memset(res, 0, sizeof *res);
  res->psnr_min = 999.0;
  S.identity = cfg->scale == 1.0;
  pthread_mutex_init(&S.lock, NULL);
  surf_ramp_tables(cfg->dmax, S.in_tab, S.out_tab);
  for (int L = 0; L < SURF_LEVELS; L++) S.ldim[L] = SURF_SHARD >> L;
  for (int d = 0; d < 3; d++) {
    S.co[d] = cfg->shard[d] * SURF_SHARD;
    S.ce[d] = surf_min64(S.co[d] + SURF_SHARD, cfg->out_shape[d]);
    if (S.ce[d] <= S.co[d]) {
      fprintf(stderr, "surface-pack: shard is outside the array\n");
      return 1;
    }
    S.lext[0][d] = S.ce[d] - S.co[d];
    for (int L = 1; L < SURF_LEVELS; L++) S.lext[L][d] = (S.lext[L - 1][d] + 1) / 2;
  }
  /* native region: interpolation support of the whole shard, dilated by dmax */
  for (int d = 0; d < 3; d++) {
    int64_t s0 = (int64_t)floor((double)S.co[d] * cfg->scale);
    int64_t s1 = (int64_t)floor((double)(S.ce[d] - 1) * cfg->scale) + (S.identity ? 0 : 1);
    S.rlo[d] = s0 - cfg->dmax;
    S.rn[d] = s1 + cfg->dmax - S.rlo[d] + 1;
    S.clo[d] = surf_max64(S.rlo[d], 0) / cfg->csize;
    S.chi[d] = surf_min64(S.rlo[d] + S.rn[d] - 1, cfg->src_shape[d] - 1) / cfg->csize;
  }
  S.mask = calloc(1, (size_t)(S.rn[0] * S.rn[1] * S.rn[2]));
  for (int L = 1; L < SURF_LEVELS; L++) S.lvl[L] = calloc(1, (size_t)(S.ldim[L] * S.ldim[L] * S.ldim[L]));
  if (!S.mask || !S.lvl[1] || !S.lvl[2] || !S.lvl[3]) {
    fprintf(stderr, "surface-pack: out of memory\n");
    return 2;
  }
  S.nthreads = cfg->threads > 0 ? cfg->threads : 1;
  S.sc = calloc((size_t)S.nthreads, sizeof *S.sc);
  for (int d = 0; d < 3; d++)
    S.mmax[d] = (int64_t)floor((double)(SURF_CHUNK - 1) * cfg->scale) + (S.identity ? 1 : 3) + 2 * cfg->dmax;
  size_t mcap = (size_t)(S.mmax[0] * S.mmax[1] * S.mmax[2]);
  for (int t = 0; t < S.nthreads; t++) {
    S.sc[t].loc = malloc(mcap);
    S.sc[t].s = malloc(mcap);
    S.sc[t].a = malloc(mcap);
    S.sc[t].b = malloc(mcap);
    S.sc[t].ramp = malloc(mcap);
    S.sc[t].chunk = malloc(VOLCOMP_CHUNK_VOXELS);
    S.sc[t].enc = malloc(VOLCOMP_ENCODE_BOUND);
    S.sc[t].dec = malloc(VOLCOMP_CHUNK_VOXELS);
    S.sc[t].src = malloc((size_t)(cfg->csize * cfg->csize * cfg->csize));
    S.sc[t].tmp = malloc((size_t)(cfg->csize * cfg->csize * cfg->csize));
    if (!S.sc[t].loc || !S.sc[t].s || !S.sc[t].a || !S.sc[t].b || !S.sc[t].ramp || !S.sc[t].chunk || !S.sc[t].enc ||
        !S.sc[t].dec || !S.sc[t].src || !S.sc[t].tmp) {
      fprintf(stderr, "surface-pack: out of memory\n");
      return 2;
    }
  }
  S.stride = cfg->samples > 0 ? (long)(SURF_MAX_CHUNKS / (unsigned)cfg->samples) : 1;
  if (S.stride < 1) S.stride = 1;

  double t0 = surf_now();
  int64_t nsrc = (S.chi[0] - S.clo[0] + 1) * (S.chi[1] - S.clo[1] + 1) * (S.chi[2] - S.clo[2] + 1);
  res->src_chunks = (uint64_t)nsrc;
  surf_parallel_for(nsrc, S.nthreads, surf_decode_job, &S);
  double t1 = surf_now();
  if (S.error) return 3;
  S.level = 0;
  res->chunks[0] = SURF_MAX_CHUNKS;
  surf_parallel_for(SURF_MAX_CHUNKS, S.nthreads, surf_level0_job, &S);
  double t2 = surf_now();
  if (S.error) return 3;
  for (int L = 2; L < SURF_LEVELS; L++) surf_pool(&S, L);
  for (int L = 1; L < SURF_LEVELS; L++) {
    int64_t g = S.ldim[L] / SURF_CHUNK;
    S.level = L;
    res->chunks[L] = (unsigned)(g * g * g);
    surf_parallel_for(g * g * g, S.nthreads, surf_levelN_job, &S);
  }
  double t3 = surf_now();
  if (S.error) return 3;
  if (res->src_nonzero && res->present[0] == 0) {
    fprintf(stderr, "surface-pack: source is nonzero but the level-0 shard is empty\n");
    return 6;
  }
  for (int L = 0; L < SURF_LEVELS; L++) {
    if (!res->present[L]) continue;
    char path[4096];
    snprintf(path, sizeof path, "%s/%d.shard", cfg->outdir, L);
    uint64_t payload = 0;
    if (surf_write_shard(path, S.enc[L], S.encn[L], res->chunks[L], &payload)) {
      fprintf(stderr, "surface-pack: cannot write %s\n", path);
      return 2;
    }
    res->bytes[L] = payload + (uint64_t)res->chunks[L] * 16u + 4u;
  }
  res->t_decode = t1 - t0;
  res->t_ramp = t2 - t1;
  res->t_encode = t3 - t2;
  for (int L = 0; L < SURF_LEVELS; L++)
    for (unsigned i = 0; i < res->chunks[L]; i++) free(S.enc[L][i]);
  for (int t = 0; t < S.nthreads; t++) {
    free(S.sc[t].loc), free(S.sc[t].s), free(S.sc[t].a), free(S.sc[t].b), free(S.sc[t].ramp);
    free(S.sc[t].chunk), free(S.sc[t].enc), free(S.sc[t].dec), free(S.sc[t].src), free(S.sc[t].tmp);
  }
  free(S.sc);
  free(S.mask);
  for (int L = 1; L < SURF_LEVELS; L++) free(S.lvl[L]);
  pthread_mutex_destroy(&S.lock);
  return 0;
}

/* ------------------------------------------------------- offline pooling ---
 * One output shard of the next coarser level, from the (up to) 8 shards of this
 * level that cover it. Levels at and above the 4th rung of a prediction have a
 * 128^3 shard holding exactly one 128^3 chunk, so this is: decode 8 chunks,
 * assemble 256^3, 2x mean pool, encode one chunk. `in[i]` is the shard file for
 * sub-position (i>>2, (i>>1)&1, i&1) or NULL/absent.
 * `shape` is the input level's array shape and `pos` the output shard index, so
 * that a partial edge block averages only over the voxels that exist. */
static int surface_pool_shard(const char *const in[8], const char *out_path, float q,
                              const int64_t shape[3], const int64_t pos[3], unsigned *present,
                              uint64_t *bytes) {
  const int64_t C = SURF_CHUNK;
  uint8_t *big = calloc(1, (size_t)(8 * C * C * C));  /* 256^3 */
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  int rc = 0;
  for (int i = 0; i < 8 && rc == 0; i++) {
    if (!in[i]) continue;
    size_t n = 0;
    uint8_t *img = shard_read_file(in[i], &n);
    if (!img) continue;  /* a missing shard is the fill value */
    uint64_t off = 0, nb = 0;
    size_t idxn = 16u + 4u;
    if (n < idxn) {
      rc = 3;
      free(img);
      break;
    }
    const uint8_t *idx = img + n - idxn;
    uint32_t crc = shard_crc32c(idx, 16u), stored = 0;
    for (int k = 3; k >= 0; k--) stored = stored << 8 | idx[16u + (unsigned)k];
    off = shard_rd_u64(idx);
    nb = shard_rd_u64(idx + 8);
    if (crc != stored) {
      fprintf(stderr, "shard-pool: %s: bad index crc\n", in[i]);
      rc = 3;
    } else if (off != ~0ull) {
      if (off + nb > n - idxn || volcomp_decode(img + off, (size_t)nb, dec, VOLCOMP_CHUNK_VOXELS) != VOLCOMP_OK) {
        fprintf(stderr, "shard-pool: %s: chunk does not decode\n", in[i]);
        rc = 3;
      } else {
        int64_t o[3] = {(i >> 2) * C, ((i >> 1) & 1) * C, (i & 1) * C};
        for (int64_t z = 0; z < C; z++)
          for (int64_t y = 0; y < C; y++)
            memcpy(big + (((o[0] + z) * 2 * C + o[1] + y) * 2 * C + o[2]), dec + (z * C + y) * C, (size_t)C);
      }
    }
    free(img);
  }
  int64_t ext[3];
  for (int d = 0; d < 3; d++) {
    ext[d] = shape[d] - pos[d] * 2 * C;
    if (ext[d] > 2 * C) ext[d] = 2 * C;
    if (ext[d] < 0) ext[d] = 0;
  }
  uint8_t *chunk = calloc(1, VOLCOMP_CHUNK_VOXELS);
  bool any = false;
  for (int64_t z = 0; z < (ext[0] + 1) / 2; z++)
    for (int64_t y = 0; y < (ext[1] + 1) / 2; y++)
      for (int64_t x = 0; x < (ext[2] + 1) / 2; x++) {
        unsigned sum = 0, cnt = 0;
        for (int64_t dz = 0; dz < 2 && 2 * z + dz < ext[0]; dz++)
          for (int64_t dy = 0; dy < 2 && 2 * y + dy < ext[1]; dy++)
            for (int64_t dx = 0; dx < 2 && 2 * x + dx < ext[2]; dx++) {
              sum += big[(((2 * z + dz) * 2 * C + 2 * y + dy) * 2 * C) + 2 * x + dx];
              cnt++;
            }
        uint8_t v = (uint8_t)((sum + cnt / 2) / cnt);
        chunk[(z * C + y) * C + x] = v;
        any |= v != 0;
      }
  uint8_t *enc = NULL;
  size_t en = 0;
  if (rc == 0 && any) {
    enc = malloc(VOLCOMP_ENCODE_BOUND);
    if (volcomp_encode(chunk, q, enc, VOLCOMP_ENCODE_BOUND, &en) != VOLCOMP_OK) rc = 3;
  }
  if (rc == 0 && any) {
    uint64_t payload = 0;
    uint8_t *one[1] = {enc};
    size_t onen[1] = {en};
    rc = surf_write_shard(out_path, one, onen, 1, &payload);
    if (bytes) *bytes = payload + 20u;
  } else if (bytes) {
    *bytes = 0;
  }
  if (present) *present = any && rc == 0 ? 1u : 0u;
  free(enc), free(chunk), free(big), free(dec);
  return rc;
}

#endif
