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
 * In MASK mode (surf_cfg.mask_mode, `--mask`) there is no ramp and no quantiser:
 * level 0 stores the published binary mask itself as a volcomp MASK chunk (the
 * 2x2x2 majority pool, coded exactly, decoded trilinearly interpolated), the
 * resample is trilinear on the 0/255 mask thresholded at 128, and levels 1..3
 * are its successive 2x majority pools, each stored the same way. That is what
 * the fleet writes; the ramp above stays available for the old flags.
 *
 * With surf_cfg.mask_lossless (`--mask-lossless`) the pyramid is the same — the
 * same resample, the same 2x majority pools — but every level is stored with
 * volcomp's LOSSLESS mask chunk (spec §12): level 0 is the published mask voxel
 * for voxel, not its 2x pool, and decodes back to 0/255 exactly. Paired with the
 * coordinator's --no-resample (scale 1, out shape = source shape) that makes the
 * export a bit-exact copy of the published mask at its own grid.
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
  const uint8_t *occ;   /* optional 64-byte occupancy mask of the 8^3 level-0 chunks (see
                         * coordinator.py surface-occupancy); NULL = every chunk is occupied */
  bool mask_mode;       /* store binary MASK chunks (volcomp.h §11) instead of the ramp: level 0
                         * is the published mask itself, levels 1..3 its 2x majority pools, and
                         * every level is encoded with volcomp_mask_encode (q is unused) */
  bool mask_lossless;   /* with mask_mode: encode every level with volcomp_mask_encode_lossless
                         * (§12) instead, so a level decodes to its own 0/255 mask exactly */
  int dmax, threads, samples;
} surf_cfg;

typedef struct {
  unsigned present[SURF_LEVELS], chunks[SURF_LEVELS];
  uint64_t bytes[SURF_LEVELS];
  double psnr_min;
  unsigned max_err, compared;
  double t_decode, t_ramp, t_encode;
  double cpu_transform, cpu_encode0;  /* CPU seconds inside the level-0 loop: the ramp +
                                       * resample + pooling, and the encode + verify */
  uint64_t out_voxels, src_chunks;
  unsigned skip_mask, skip_zero, skip_full;  /* level-0 chunks skipped: masked out, all air, all mask */
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
 * candidate seed within the clip has all three offsets within +-dmax.
 *
 * Every pass is a sequence of two row kernels over CONTIGUOUS uint8 rows -
 * "d = min(d, s + kk)" and "d = min(d, SURF_INF)" - so the y and z passes walk
 * shifted rows (and whole shifted planes) instead of strided columns, and both
 * kernels are 32 lanes per instruction under AVX2. Inputs are always <= SURF_INF
 * and kk <= dmax^2, so the saturating add never saturates and the AVX2 and scalar
 * kernels produce identical bytes. */

#if VF_HAVE_AVX2
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void surf_ma_avx2(uint8_t *d, const uint8_t *s,
                                                                              int64_t n, unsigned kk) {
  const __m256i v = _mm256_set1_epi8((char)(unsigned char)kk);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_adds_epu8(_mm256_loadu_si256((const __m256i *)(s + i)), v);
    __m256i b = _mm256_loadu_si256((const __m256i *)(d + i));
    _mm256_storeu_si256((__m256i *)(d + i), _mm256_min_epu8(a, b));
  }
  for (; i < n; i++) {
    unsigned t = s[i] + kk;
    if (t < d[i]) d[i] = (uint8_t)t;
  }
}
VF_TARGET_AVX2 static void surf_min_add_avx2(uint8_t *d, const uint8_t *s, int64_t n, unsigned kk) {
  const __m256i v = _mm256_set1_epi8((char)(unsigned char)kk);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i a = _mm256_adds_epu8(_mm256_loadu_si256((const __m256i *)(s + i)), v);
    __m256i b = _mm256_loadu_si256((const __m256i *)(d + i));
    _mm256_storeu_si256((__m256i *)(d + i), _mm256_min_epu8(a, b));
  }
  for (; i < n; i++) {
    unsigned t = s[i] + kk;
    if (t < d[i]) d[i] = (uint8_t)t;
  }
}
VF_TARGET_AVX2 static void surf_clamp_avx2(uint8_t *d, int64_t n, unsigned cap) {
  const __m256i v = _mm256_set1_epi8((char)(unsigned char)cap);
  int64_t i = 0;
  for (; i + 32 <= n; i += 32)
    _mm256_storeu_si256((__m256i *)(d + i), _mm256_min_epu8(_mm256_loadu_si256((const __m256i *)(d + i)), v));
  for (; i < n; i++)
    if (d[i] > cap) d[i] = (uint8_t)cap;
}
#endif

static inline void surf_min_add(uint8_t *d, const uint8_t *s, int64_t n, unsigned kk) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) {
    surf_min_add_avx2(d, s, n, kk);
    return;
  }
#endif
  for (int64_t i = 0; i < n; i++) {
    unsigned t = s[i] + kk;
    if (t < d[i]) d[i] = (uint8_t)t;
  }
}
static inline void surf_clamp(uint8_t *d, int64_t n, unsigned cap) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) {
    surf_clamp_avx2(d, n, cap);
    return;
  }
#endif
  for (int64_t i = 0; i < n; i++)
    if (d[i] > cap) d[i] = (uint8_t)cap;
}

/* The two in-plane passes of the fused pipeline, one kernel dispatch per PLANE
 * (a box row is ~140 bytes: dispatching per row cost more than the work).
 *
 * The x pass treats the whole plane as one row. A shift of k then reaches across
 * a row boundary, but only into the first and last dmax columns - which are the
 * box's x halo, are never sampled by the resample, and never leak back: the x
 * pass reads the seed plane (not its own output), and the y and z passes stay
 * within one column. */
#define SURF_PASS_X_BODY(MA)                                                                       \
  memcpy(dst, src, (size_t)n);                                                                     \
  for (int k = 1; k <= R && k < n; k++) {                                                          \
    unsigned kk = (unsigned)(k * k);                                                               \
    MA(dst + k, src, n - k, kk);                                                                   \
    MA(dst, src + k, n - k, kk);                                                                   \
  }
#define SURF_PASS_Y_BODY(MA)                                                                       \
  for (int64_t y = 0; y < my; y++) {                                                               \
    uint8_t *d = dst + y * mx;                                                                     \
    memcpy(d, src + y * mx, (size_t)mx);                                                           \
    for (int k = 1; k <= R; k++) {                                                                 \
      unsigned kk = (unsigned)(k * k);                                                             \
      if (y >= k) MA(d, src + (y - k) * mx, mx, kk);                                               \
      if (y + k < my) MA(d, src + (y + k) * mx, mx, kk);                                           \
    }                                                                                              \
  }
/* seed planes of both fields out of one strided read of the assembled region */
#define SURF_SEED2_BODY(SEED)                                                                      \
  for (int64_t y = 0; y < my; y++) SEED(a + y * mx, b + y * mx, src + y * rstride_y, mx);

__attribute__((always_inline)) static inline void surf_ma_c(uint8_t *d, const uint8_t *s, int64_t n,
                                                            unsigned kk) {
  for (int64_t i = 0; i < n; i++) {
    unsigned t = s[i] + kk;
    if (t < d[i]) d[i] = (uint8_t)t;
  }
}
__attribute__((always_inline)) static inline void surf_seed_row_c(uint8_t *a, uint8_t *b, const uint8_t *s,
                                                                  int64_t n) {
  for (int64_t x = 0; x < n; x++) {
    a[x] = (uint8_t)(s[x] ? SURF_INF : 0u);
    b[x] = (uint8_t)(s[x] ? 0u : SURF_INF);
  }
}
static void surf_pass_x_c(uint8_t *dst, const uint8_t *src, int64_t n, int R) { SURF_PASS_X_BODY(surf_ma_c) }
static void surf_pass_y_c(uint8_t *dst, const uint8_t *src, int64_t my, int64_t mx, int R) {
  SURF_PASS_Y_BODY(surf_ma_c)
}
static void surf_seed2_c(uint8_t *a, uint8_t *b, const uint8_t *src, int64_t my, int64_t mx,
                         int64_t rstride_y) {
  SURF_SEED2_BODY(surf_seed_row_c)
}
#if VF_HAVE_AVX2
__attribute__((always_inline)) VF_TARGET_AVX2 static inline void surf_seed_row_avx2(uint8_t *a, uint8_t *b,
                                                                                    const uint8_t *s, int64_t n) {
  const __m256i zero = _mm256_setzero_si256(), inf = _mm256_set1_epi8((char)(unsigned char)SURF_INF);
  int64_t x = 0;
  for (; x + 32 <= n; x += 32) {
    __m256i is0 = _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)(s + x)), zero);
    _mm256_storeu_si256((__m256i *)(a + x), _mm256_andnot_si256(is0, inf));
    _mm256_storeu_si256((__m256i *)(b + x), _mm256_and_si256(is0, inf));
  }
  for (; x < n; x++) {
    a[x] = (uint8_t)(s[x] ? SURF_INF : 0u);
    b[x] = (uint8_t)(s[x] ? 0u : SURF_INF);
  }
}
VF_TARGET_AVX2 static void surf_pass_x_kavx2(uint8_t *dst, const uint8_t *src, int64_t n, int R) {
  SURF_PASS_X_BODY(surf_ma_avx2)
}
VF_TARGET_AVX2 static void surf_pass_y_kavx2(uint8_t *dst, const uint8_t *src, int64_t my, int64_t mx, int R) {
  SURF_PASS_Y_BODY(surf_ma_avx2)
}
VF_TARGET_AVX2 static void surf_seed2_avx2(uint8_t *a, uint8_t *b, const uint8_t *src, int64_t my, int64_t mx,
                                           int64_t rstride_y) {
  SURF_SEED2_BODY(surf_seed_row_avx2)
}
#endif

/* pass along x: the window shifts within each contiguous row */
static inline void surf_pass_x(uint8_t *dst, const uint8_t *src, int64_t mz, int64_t my, int64_t mx, int R) {
  for (int64_t line = 0; line < mz * my; line++) {
    const uint8_t *s = src + line * mx;
    uint8_t *d = dst + line * mx;
    memcpy(d, s, (size_t)mx);
    for (int k = 1; k <= R; k++) {
      unsigned kk = (unsigned)(k * k);
      if (kk >= SURF_INF) break;
      if (k >= mx) break;
      surf_min_add(d + k, s, mx - k, kk);
      surf_min_add(d, s + k, mx - k, kk);
    }
    surf_clamp(d, mx, SURF_INF);
  }
}
/* pass along an axis whose neighbours are whole contiguous blocks of `span`
 * elements `span` apart: y (rows within a plane) and z (planes) */
static inline void surf_pass_strided(uint8_t *dst, const uint8_t *src, int64_t nlines, int64_t span,
                              int64_t group, int R) {
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
          surf_min_add(d, src + (g * nlines + ll) * span, span, kk);
        }
      }
      surf_clamp(d, span, SURF_INF);
    }
}
/* squared distance field of `seed` (0 at seeds, SURF_INF elsewhere) into A; B is scratch */
static inline void surf_edt(uint8_t *A, uint8_t *B, const uint8_t *seed, int64_t mz, int64_t my, int64_t mx, int R) {
  surf_pass_x(A, seed, mz, my, mx, R);              /* x */
  surf_pass_strided(B, A, my, mx, mz, R);           /* y: my lines of mx, mz groups */
  surf_pass_strided(A, B, mz, my * mx, 1, R);       /* z: mz planes of my*mx */
}

/* ramp[i] = a[i] ? in_tab[a[i]] : out_tab[b[i]], with a, b <= SURF_INF < 16 (one
 * byte shuffle per 32 voxels under AVX2). */
#if VF_HAVE_AVX2
VF_TARGET_AVX2 static void surf_apply_tables_avx2(uint8_t *ramp, const uint8_t *a, const uint8_t *b,
                                                  size_t n, const uint8_t *in_tab, const uint8_t *out_tab) {
  uint8_t ti[32], to[32];
  for (int i = 0; i < 16; i++) {
    ti[i] = ti[i + 16] = i <= (int)SURF_INF ? in_tab[i] : 0;
    to[i] = to[i + 16] = i <= (int)SURF_INF ? out_tab[i] : 0;
  }
  const __m256i vi = _mm256_loadu_si256((const __m256i *)ti), vo = _mm256_loadu_si256((const __m256i *)to);
  const __m256i zero = _mm256_setzero_si256(), cap = _mm256_set1_epi8((char)(unsigned char)SURF_INF);
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_min_epu8(_mm256_loadu_si256((const __m256i *)(a + i)), cap);
    __m256i vb = _mm256_min_epu8(_mm256_loadu_si256((const __m256i *)(b + i)), cap);
    __m256i in = _mm256_shuffle_epi8(vi, va), out = _mm256_shuffle_epi8(vo, vb);
    __m256i is0 = _mm256_cmpeq_epi8(va, zero);
    _mm256_storeu_si256((__m256i *)(ramp + i), _mm256_blendv_epi8(in, out, is0));
  }
  for (; i < n; i++)
    ramp[i] = a[i] ? in_tab[a[i] < SURF_INF ? a[i] : SURF_INF] : out_tab[b[i] < SURF_INF ? b[i] : SURF_INF];
}
#endif
static inline void surf_apply_tables(uint8_t *ramp, const uint8_t *a, const uint8_t *b, size_t n,
                                     const uint8_t *in_tab, const uint8_t *out_tab) {
#if VF_HAVE_AVX2
  if (vf_use_avx2()) {
    surf_apply_tables_avx2(ramp, a, b, n, in_tab, out_tab);
    return;
  }
#endif
  for (size_t i = 0; i < n; i++)
    ramp[i] = a[i] ? in_tab[a[i] < SURF_INF ? a[i] : SURF_INF] : out_tab[b[i] < SURF_INF ? b[i] : SURF_INF];
}

/* Reference form of the ramp over a sub-box of the 0/1 mask `loc` (the fused
 * pipeline below is what runs; tests check the two agree).
 * Ramp over a sub-box of the 0/1 mask `loc` (dims m, which must include a dmax
 * halo on every side): writes `ramp` for the whole box, but only the voxels at
 * least dmax from the box edge are meaningful - those are exact and independent
 * of how the volume was tiled. `a`, `b`, `s` are scratch of the box size; `loc`
 * is destroyed. */
static inline void surf_ramp_region(uint8_t *ramp, uint8_t *loc, int64_t mz, int64_t my, int64_t mx, int dmax,
                             const uint8_t *in_tab, const uint8_t *out_tab, uint8_t *a, uint8_t *b, uint8_t *s) {
  size_t mn = (size_t)(mz * my * mx);
  for (size_t k = 0; k < mn; k++) s[k] = (uint8_t)(loc[k] ? SURF_INF : 0u);  /* seeds: background */
  surf_edt(a, b, s, mz, my, mx, dmax);                                     /* a = d^2 to background */
  for (size_t k = 0; k < mn; k++) loc[k] = loc[k] ? 0 : (uint8_t)SURF_INF; /* seeds: foreground */
  surf_edt(b, s, loc, mz, my, mx, dmax);                                   /* b = d^2 to foreground */
  surf_apply_tables(ramp, a, b, mn, in_tab, out_tab);
}

/* ------------------------------------------------------------ 2x mean pooling ---
 * (sum of the 8 voxels + 4) / 8, which is what the scalar form computes for a full
 * block. maddubs gives the pairwise sums along x for free. */
#if VF_HAVE_AVX2
VF_TARGET_AVX2 static unsigned surf_pool_rows_avx2(uint8_t *d, const uint8_t *r0, const uint8_t *r1,
                                                   const uint8_t *r2, const uint8_t *r3, int64_t nout) {
  const __m256i one = _mm256_set1_epi8(1), four = _mm256_set1_epi16(4);
  __m256i any = _mm256_setzero_si256();
  int64_t i = 0;
  for (; i + 16 <= nout; i += 16) {
    __m256i s = _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i *)(r0 + 2 * i)), one);
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i *)(r1 + 2 * i)), one));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i *)(r2 + 2 * i)), one));
    s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i *)(r3 + 2 * i)), one));
    any = _mm256_or_si256(any, s);
    s = _mm256_srli_epi16(_mm256_add_epi16(s, four), 3);
    __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(s, s), 0xD8);
    _mm_storeu_si128((__m128i *)(d + i), _mm256_castsi256_si128(p));
  }
  unsigned nz = !_mm256_testz_si256(any, any);
  for (; i < nout; i++) {
    unsigned sum = (unsigned)r0[2 * i] + r0[2 * i + 1] + r1[2 * i] + r1[2 * i + 1] + r2[2 * i] +
                   r2[2 * i + 1] + r3[2 * i] + r3[2 * i + 1];
    nz |= sum != 0;
    d[i] = (uint8_t)((sum + 4) / 8);
  }
  return nz;
}
#endif

/* ------------------------------------------------------------- the resample ---
 * One output row of the trilinear resample. The arithmetic is written out in the
 * order the scalar form uses and FMA contraction is switched off, so the AVX2 and
 * scalar kernels return the same doubles bit for bit; `(int)(v + 0.5)` is floor
 * for v >= 0, which every ramp value is.
 *
 * The AVX2 kernel does four output voxels at a time and needs the eight corner
 * bytes of each; when the four source indices are consecutive (they are, except
 * at the 1-in-1/(scale-1) voxels where the sample point steps over one) both
 * corner vectors come out of a single 8-byte load. */
#if defined(__clang__)
#define SURF_NO_FMA _Pragma("clang fp contract(off)")
#else
#define SURF_NO_FMA
#endif

#define SURF_LERP_ROW_BODY()                                                                       \
  for (; x < n; x++) {                                                                             \
    int64_t ax = idx[x];                                                                           \
    double fx = fr[x], gx = cf[x];                                                                 \
    double v = gz * (gy * (gx * p000[ax] + fx * p000[ax + 1]) + fy * (gx * p001[ax] + fx * p001[ax + 1])) + \
               fz * (gy * (gx * p100[ax] + fx * p100[ax + 1]) + fy * (gx * p101[ax] + fx * p101[ax + 1]));  \
    int iv = (int)(v + 0.5);                                                                       \
    d[x] = (uint8_t)(iv < 0 ? 0 : iv > 255 ? 255 : iv);                                            \
  }

static void surf_lerp_row_c(uint8_t *d, int64_t n, const int64_t *idx, const double *fr, const double *cf,
                            const uint8_t *p000, const uint8_t *p001, const uint8_t *p100,
                            const uint8_t *p101, double fz, double gz, double fy, double gy) {
  SURF_NO_FMA
  int64_t x = 0;
  SURF_LERP_ROW_BODY()
}

#if VF_HAVE_AVX2 && defined(__clang__)
VF_TARGET_AVX2 static inline __m256d surf_u8x4_pd(const uint8_t *p) {
  return _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_cvtsi32_si128((int)(uint32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 |
                                                                                (uint32_t)p[2] << 16 |
                                                                                (uint32_t)p[3] << 24))));
}
VF_TARGET_AVX2 static void surf_lerp_row_avx2(uint8_t *d, int64_t n, const int64_t *idx, const double *fr,
                                              const double *cf, const uint8_t *p000, const uint8_t *p001,
                                              const uint8_t *p100, const uint8_t *p101, double fz, double gz,
                                              double fy, double gy) {
  SURF_NO_FMA
  const __m256d vfz = _mm256_set1_pd(fz), vgz = _mm256_set1_pd(gz), vfy = _mm256_set1_pd(fy),
                vgy = _mm256_set1_pd(gy), half = _mm256_set1_pd(0.5);
  int64_t x = 0;
  for (; x + 4 <= n; x += 4) {
    int64_t a = idx[x];
    if (idx[x + 3] - a != 3) break;  /* the sample point stepped over a voxel: finish scalar */
    __m256d gx = _mm256_loadu_pd(cf + x), fx = _mm256_loadu_pd(fr + x);
    __m256d v = _mm256_setzero_pd();
    const uint8_t *p[4] = {p000, p001, p100, p101};
    __m256d row[4];
    for (int r = 0; r < 4; r++)
      row[r] = _mm256_add_pd(_mm256_mul_pd(gx, surf_u8x4_pd(p[r] + a)),
                             _mm256_mul_pd(fx, surf_u8x4_pd(p[r] + a + 1)));
    v = _mm256_add_pd(_mm256_mul_pd(vgz, _mm256_add_pd(_mm256_mul_pd(vgy, row[0]), _mm256_mul_pd(vfy, row[1]))),
                      _mm256_mul_pd(vfz, _mm256_add_pd(_mm256_mul_pd(vgy, row[2]), _mm256_mul_pd(vfy, row[3]))));
    __m128i iv = _mm256_cvttpd_epi32(_mm256_add_pd(v, half));
    iv = _mm_min_epi32(_mm_max_epi32(iv, _mm_setzero_si128()), _mm_set1_epi32(255));
    iv = _mm_shuffle_epi8(iv, _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
    memcpy(d + x, &iv, 4);
  }
  SURF_LERP_ROW_BODY()
}
#endif

static inline void surf_lerp_row(uint8_t *d, int64_t n, const int64_t *idx, const double *fr, const double *cf,
                                 const uint8_t *p000, const uint8_t *p001, const uint8_t *p100,
                                 const uint8_t *p101, double fz, double gz, double fy, double gy) {
#if VF_HAVE_AVX2 && defined(__clang__)
  if (vf_use_avx2()) {
    surf_lerp_row_avx2(d, n, idx, fr, cf, p000, p001, p100, p101, fz, gz, fy, gy);
    return;
  }
#endif
  surf_lerp_row_c(d, n, idx, fr, cf, p000, p001, p100, p101, fz, gz, fy, gy);
}

/* --------------------------------------------------- the fused ramp pipeline ---
 * The straightforward form above walks the whole sub-box six times (two fields x
 * three axes) plus the gather, the seeds and the table lookup: ~45 MB of traffic
 * per 128^3 output chunk, which is memory bound long before it is ALU bound.
 *
 * This form walks it ONCE. For each z plane of the box it builds both seed planes
 * straight out of the assembled region (no gather, no mask copy), runs the x and y
 * passes into two rings of 2*dmax+1 planes (~300 KB per thread: L2 resident), and
 * as soon as a plane has its full +-dmax z neighbourhood it runs the z pass for
 * both fields and writes that plane of the ramp. Only the planes the resample can
 * actually sample are produced - the box's dmax-deep border planes never were.
 *
 * Byte for byte the same result as surf_ramp_region: the same per-voxel min-plus
 * arithmetic in the same order. The per-pass clamp to SURF_INF is dropped (values
 * stay <= SURF_INF + 3 * dmax^2 = 37, so nothing saturates) and applied once when
 * the tables are looked up, which cannot change any value <= dmax^2. */

#define SURF_RING (2 * SURF_DMAX + 1)

/* ramp[(p - dmax) plane] for every box plane p in [dmax, m0 - dmax); planes are
 * m1 * m2 and include the y/x halo, exactly as surf_ramp_region's output does. */
static void surf_ramp_fused(const uint8_t *base, int64_t rstride_z, int64_t rstride_y, int64_t m0,
                            int64_t m1, int64_t m2, int R, const uint8_t *in_tab, const uint8_t *out_tab,
                            uint8_t *ramp, uint8_t *pool) {
  const int64_t plane = m1 * m2;
  uint8_t *seed = pool, *seed2 = pool + plane, *xt = pool + 2 * plane, *za = pool + 3 * plane,
          *zb = pool + 4 * plane;
  uint8_t *ring[2][SURF_RING];
  for (int f = 0; f < 2; f++)
    for (int r = 0; r < SURF_RING; r++) ring[f][r] = pool + (5 + f * SURF_RING + r) * plane;
  bool avx2 = false;
#if VF_HAVE_AVX2
  avx2 = vf_use_avx2();
#endif
  for (int64_t p = 0; p < m0; p++) {
    const uint8_t *src = base + p * rstride_z;
#if VF_HAVE_AVX2
    if (avx2)
      surf_seed2_avx2(seed, seed2, src, m1, m2, rstride_y);
    else
#endif
      surf_seed2_c(seed, seed2, src, m1, m2, rstride_y);
    for (int f = 0; f < 2; f++) {
      const uint8_t *sd = f ? seed2 : seed;
#if VF_HAVE_AVX2
      if (avx2) {
        surf_pass_x_kavx2(xt, sd, plane, R);
        surf_pass_y_kavx2(ring[f][p % SURF_RING], xt, m1, m2, R);
      } else
#endif
      {
        surf_pass_x_c(xt, sd, plane, R);
        surf_pass_y_c(ring[f][p % SURF_RING], xt, m1, m2, R);
      }
    }
    if (p < 2 * R) continue;
    int64_t p0 = p - R;  /* this plane now has its full +-R neighbourhood */
    for (int f = 0; f < 2; f++) {
      uint8_t *dst = f ? zb : za;
      memcpy(dst, ring[f][p0 % SURF_RING], (size_t)plane);
      for (int k = 1; k <= R; k++) {
        unsigned kk = (unsigned)(k * k);
        surf_min_add(dst, ring[f][(p0 - k + SURF_RING) % SURF_RING], plane, kk);
        surf_min_add(dst, ring[f][(p0 + k) % SURF_RING], plane, kk);
      }
    }
    surf_apply_tables(ramp + (p0 - R) * plane, za, zb, (size_t)plane, in_tab, out_tab);
  }
}

/* The mask-mode stand-in for surf_ramp_fused: the same `ramp` buffer, holding
 * the box planes [R, m0 - R) of the 0/255 mask, so the resample, the pooling and
 * the encoder below do not care which mode they are in. */
static void surf_mask_fill(const uint8_t *base, int64_t rstride_z, int64_t rstride_y, int64_t m0,
                           int64_t m1, int64_t m2, int R, uint8_t *ramp) {
  for (int64_t p = R; p < m0 - R; p++)
    for (int64_t y = 0; y < m1; y++) {
      const uint8_t *s = base + p * rstride_z + y * rstride_y;
      uint8_t *d = ramp + ((p - R) * m1 + y) * m2;
      for (int64_t x = 0; x < m2; x++) d[x] = s[x] ? 255u : 0u;
    }
}
/* Back to 0/255. A 0/255 field crosses 128 exactly where the 0/1 field crosses
 * 1/2, so this is the threshold of the trilinear resample, and — over a full
 * 2x2x2 block — the MAJORITY of the mean pool. */
static inline void surf_binarise(uint8_t *p, int64_t n) {
  for (int64_t i = 0; i < n; i++) p[i] = p[i] >= 128u ? 255u : 0u;
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
  double cpu_transform, cpu_encode0;
  uint8_t *pool;              /* per-thread plane pool for the fused pipeline (4 + 2 * SURF_RING planes) */
  uint8_t *ramp;              /* the ramp planes of one sub-box */
  uint8_t *chunk;             /* 128^3 output chunk */
  uint8_t *enc, *dec;         /* encode/decode scratch */
  uint8_t *src;               /* one decompressed source chunk */
  uint8_t *tmp;               /* blosc unshuffle scratch */
  int64_t ridx[3][SURF_CHUNK];        /* resample: source index per output voxel, per axis */
  double rfrac[3][SURF_CHUNK], rcofr[3][SURF_CHUNK];  /* and its weights (f, 1 - f) */
} surf_scratch;

typedef struct {
  const surf_cfg *cfg;
  int64_t rlo[3], rn[3];      /* native mask region (may start below 0 / end past the array) */
  uint8_t *mask;              /* rn[0]*rn[1]*rn[2], 0 or 1 */
  int64_t co[3], ce[3];       /* output level-0 core extent of this shard */
  int64_t clo[3], chi[3];     /* source chunk index range covering the region */
  int64_t cdim[3];            /* chi - clo + 1 */
  uint8_t *cstat;             /* per source chunk: 0 all air, 1 all mask, 2 mixed (absent = 0) */
  int64_t ldim[SURF_LEVELS];  /* shard edge per level */
  int64_t lext[SURF_LEVELS][3];
  uint8_t *lvl[SURF_LEVELS];  /* pooled shard volumes for levels 1..3 */
  uint8_t blk[SURF_MAX_CHUNKS]; /* level-0 chunks that produced data: the coarse levels only
                                 * have to pool and encode the blocks these fed */
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
  unsigned any = 0, all = 1;
  for (int64_t z = lo[0]; z < hi[0]; z++)
    for (int64_t y = lo[1]; y < hi[1]; y++) {
      const uint8_t *s = sc->src + ((z - g0[0]) * C + (y - g0[1])) * C + (lo[2] - g0[2]);
      uint8_t *d = S->mask + ((z - S->rlo[0]) * S->rn[1] + (y - S->rlo[1])) * S->rn[2] + (lo[2] - S->rlo[2]);
      for (int64_t x = 0; x < hi[2] - lo[2]; x++) {
        d[x] = s[x] != 0;
        any |= d[x];
        all &= d[x];
      }
    }
  /* what this chunk contributes to the region: all air, all mask, or mixed. An output
   * chunk whose haloed box meets only uniform sources needs no distance transform. */
  S->cstat[((cz - S->clo[0]) * S->cdim[1] + (cy - S->clo[1])) * S->cdim[2] + (cx - S->clo[2])] =
      (uint8_t)(!any ? 0 : all ? 1 : 2);
  if (any) {
    pthread_mutex_lock(&S->lock);
    S->res->src_nonzero = true;
    pthread_mutex_unlock(&S->lock);
  }
}

/* Is the haloed box [j0, j0 + m) uniform? 0 = all air, 1 = all mask, 2 = mixed.
 * Everything outside the published array is air. A source chunk that was not
 * downloaded is absent, hence air: with an occupancy mask the worker only fetches
 * chunks that meet an occupied output chunk, and unoccupied output chunks are
 * skipped before this is ever asked. */
static int surf_box_uniform(const surf_state *S, const int64_t j0[3], const int64_t m[3]) {
  const int64_t C = S->cfg->csize;
  int seen0 = 0, seen1 = 0;
  for (int d = 0; d < 3; d++)
    if (j0[d] < 0 || j0[d] + m[d] > S->cfg->src_shape[d]) seen0 = 1;  /* padding is air */
  int64_t lo[3], hi[3];
  for (int d = 0; d < 3; d++) {
    lo[d] = surf_max64(j0[d], 0) / C;
    hi[d] = surf_min64(j0[d] + m[d] - 1, S->cfg->src_shape[d] - 1) / C;
    if (hi[d] < lo[d]) return 0;  /* the box lies entirely outside the array */
  }
  for (int64_t cz = lo[0]; cz <= hi[0]; cz++)
    for (int64_t cy = lo[1]; cy <= hi[1]; cy++)
      for (int64_t cx = lo[2]; cx <= hi[2]; cx++) {
        int st = S->cstat[((cz - S->clo[0]) * S->cdim[1] + (cy - S->clo[1])) * S->cdim[2] + (cx - S->clo[2])];
        if (st == 2) return 2;
        if (st) seen1 = 1; else seen0 = 1;
        if (seen0 && seen1) return 2;
      }
  return seen1 ? 1 : 0;
}

/* ---------------------------------------------------------------- encoding */

/* `known` is 1 when the caller already knows the chunk holds data (the pooling sweep
 * saw it), -1 when it does not know. */
static void surf_encode_chunk(surf_state *S, int level, unsigned idx, const uint8_t *chunk, int tid, int known) {
  double t0 = surf_now();
  bool zero = known >= 0 ? known == 0 : true;
  if (known < 0)
    for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS && zero; k++) zero = chunk[k] == 0;
  if (zero) {
    S->sc[tid].cpu_encode0 += surf_now() - t0;
    return;
  }
  surf_scratch *sc = &S->sc[tid];
  size_t en = 0;
  volcomp_status est =
      S->cfg->mask_mode
          ? (S->cfg->mask_lossless ? volcomp_mask_encode_lossless(chunk, sc->enc, VOLCOMP_ENCODE_BOUND, &en)
                                   : volcomp_mask_encode(chunk, sc->enc, VOLCOMP_ENCODE_BOUND, &en))
          : volcomp_encode(chunk, S->cfg->q[level], sc->enc, VOLCOMP_ENCODE_BOUND, &en);
  if (est != VOLCOMP_OK) {
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
    if (S->cfg->mask_lossless) {
      /* A lossless mask chunk must come back voxel for voxel. */
      unsigned bad = 0;
      for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS; k++) bad += sc->dec[k] != (chunk[k] ? 255u : 0u);
      if (bad) {
        fprintf(stderr, "surface-pack: level %d chunk %u: %u voxels do not round trip\n", level, idx, bad);
        S->error = 1;
        return;
      }
      psnr = 999.0;
    } else if (S->cfg->mask_mode) {
      /* A mask chunk decodes to the interpolated field, so PSNR against the mask
       * says nothing. What must hold is the invariant: every stored block centre
       * (even coordinates) comes back as the 2x2x2 MAJORITY of the source, 0 or
       * 255 exactly. */
      const unsigned C = SURF_CHUNK;
      unsigned bad = 0;
      for (unsigned z = 0; z < C; z += 2)
        for (unsigned y = 0; y < C; y += 2)
          for (unsigned x = 0; x < C; x += 2) {
            unsigned c = 0;
            for (unsigned dz = 0; dz < 2; dz++)
              for (unsigned dy = 0; dy < 2; dy++)
                for (unsigned dx = 0; dx < 2; dx++)
                  c += chunk[((size_t)(z + dz) * C + y + dy) * C + x + dx] != 0;
            bad += sc->dec[((size_t)z * C + y) * C + x] != (c * 2 >= 8 ? 255u : 0u);
          }
      if (bad) {
        fprintf(stderr, "surface-pack: level %d chunk %u: %u block centres do not round trip\n", level,
                idx, bad);
        S->error = 1;
        return;
      }
      psnr = 999.0;
    } else {
      double se = 0;
      for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS; k++) {
        int d = (int)chunk[k] - (int)sc->dec[k];
        unsigned ad = (unsigned)(d < 0 ? -d : d);
        se += (double)d * d;
        if (ad > mx) mx = ad;
      }
      psnr = se == 0 ? 999.0 : 10.0 * log10(65025.0 * (double)VOLCOMP_CHUNK_VOXELS / se);
    }
  }
  S->sc[tid].cpu_encode0 += surf_now() - t0;
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
  double t_job = surf_now();
  if (cfg->occ && !(cfg->occ[i >> 3] >> (i & 7) & 1)) {  /* the occupancy mask says: no data here */
    pthread_mutex_lock(&S->lock);
    S->res->skip_mask++;
    pthread_mutex_unlock(&S->lock);
    return;
  }
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
  /* A uniform box needs no distance transform at all: all air -> the chunk is absent,
   * all mask -> every voxel is dmax deep inside, so the ramp (and its resample, whose
   * weights sum to one, and its pooling) is the constant 255. Both are common. */
  int uni = surf_box_uniform(S, j0, m);
  uint64_t vox = (uint64_t)(ext[0] * ext[1] * ext[2]);
  if (uni == 0) {
    pthread_mutex_lock(&S->lock);
    S->res->skip_zero++;
    S->res->out_voxels += vox;
    pthread_mutex_unlock(&S->lock);
    return;  /* all zero: no chunk, and the level-1 block is already zero */
  }
  memset(sc->chunk, 0, VOLCOMP_CHUNK_VOXELS);
  if (uni == 1) {
    for (int64_t z = 0; z < ext[0]; z++)
      for (int64_t y = 0; y < ext[1]; y++) memset(sc->chunk + (z * SURF_CHUNK + y) * SURF_CHUNK, 255, (size_t)ext[2]);
    pthread_mutex_lock(&S->lock);
    S->res->skip_full++;
    pthread_mutex_unlock(&S->lock);
  } else {
    /* the ramp, straight out of the assembled region, one z plane at a time; `ramp`
     * holds the m[0] - 2R planes the resample can sample, plane 0 being box plane R */
    const uint8_t *base = S->mask + ((j0[0] - S->rlo[0]) * S->rn[1] + (j0[1] - S->rlo[1])) * S->rn[2] +
                          (j0[2] - S->rlo[2]);
    uint8_t *ramp = sc->ramp;
    if (cfg->mask_mode)
      surf_mask_fill(base, S->rn[1] * S->rn[2], S->rn[2], m[0], m[1], m[2], R, ramp);
    else
      surf_ramp_fused(base, S->rn[1] * S->rn[2], S->rn[2], m[0], m[1], m[2], R, S->in_tab, S->out_tab, ramp,
                      sc->pool);
    /* resample onto the exact ladder grid */
    if (S->identity) {
      for (int64_t z = 0; z < ext[0]; z++)
        for (int64_t y = 0; y < ext[1]; y++) {
          const uint8_t *s = ramp + (z * m[1] + (y + R)) * m[2] + R;
          memcpy(sc->chunk + (z * SURF_CHUNK + y) * SURF_CHUNK, s, (size_t)ext[2]);
        }
    } else {
      /* the sample position of every output voxel, once per axis instead of once per
       * voxel (t >= 0, so the truncating cast is floor and the values are the same
       * doubles the per-voxel form produced) */
      for (int d = 0; d < 3; d++)
        for (int64_t k = 0; k < ext[d]; k++) {
          double t = (double)(o0[d] + k) * cfg->scale;
          double fl = (double)(int64_t)t;
          sc->ridx[d][k] = (int64_t)fl - j0[d];
          sc->rfrac[d][k] = t - fl;
          sc->rcofr[d][k] = 1.0 - (t - fl);
        }
      for (int64_t z = 0; z < ext[0]; z++) {
        int64_t az = sc->ridx[0][z];
        double fz = sc->rfrac[0][z], gz = sc->rcofr[0][z];
        for (int64_t y = 0; y < ext[1]; y++) {
          int64_t ay = sc->ridx[1][y];
          double fy = sc->rfrac[1][y], gy = sc->rcofr[1][y];
          const uint8_t *p000 = ramp + ((az - R) * m[1] + ay) * m[2], *p001 = p000 + m[2];
          const uint8_t *p100 = p000 + m[1] * m[2], *p101 = p100 + m[2];
          uint8_t *d = sc->chunk + (z * SURF_CHUNK + y) * SURF_CHUNK;
          surf_lerp_row(d, ext[2], sc->ridx[2], sc->rfrac[2], sc->rcofr[2], p000, p001, p100, p101, fz, gz, fy, gy);
          if (cfg->mask_mode) surf_binarise(d, ext[2]);  /* trilinear on 0/1, threshold 0.5 */
        }
      }
    }
  }
  /* 2x mean pooling into the level-1 block of this chunk (disjoint per chunk); the
   * same sweep answers "is this chunk all zero", so the encoder need not scan it */
  int64_t h[3] = {(ext[0] + 1) / 2, (ext[1] + 1) / 2, (ext[2] + 1) / 2};
  uint8_t *l1 = S->lvl[1];
  int64_t d1 = S->ldim[1];
  unsigned nonzero = 0;
  bool full = ext[0] == SURF_CHUNK && ext[1] == SURF_CHUNK && ext[2] == SURF_CHUNK;
  for (int64_t z = 0; z < h[0]; z++)
    for (int64_t y = 0; y < h[1]; y++) {
      int64_t oz = ci[0] * (SURF_CHUNK / 2) + z, oy = ci[1] * (SURF_CHUNK / 2) + y;
      uint8_t *d = l1 + (oz * d1 + oy) * d1 + ci[2] * (SURF_CHUNK / 2);
#if VF_HAVE_AVX2
      if (full && vf_use_avx2()) {
        const uint8_t *c = sc->chunk + ((2 * z) * SURF_CHUNK + 2 * y) * SURF_CHUNK;
        nonzero |= surf_pool_rows_avx2(d, c, c + SURF_CHUNK, c + (int64_t)SURF_CHUNK * SURF_CHUNK,
                                       c + (int64_t)SURF_CHUNK * SURF_CHUNK + SURF_CHUNK, h[2]);
        if (cfg->mask_mode) surf_binarise(d, h[2]);  /* a full block: the majority */
        continue;
      }
#endif
      for (int64_t x = 0; x < h[2]; x++) {
        unsigned sum = 0, cnt = 0;
        for (int64_t dz = 0; dz < 2 && 2 * z + dz < ext[0]; dz++)
          for (int64_t dy = 0; dy < 2 && 2 * y + dy < ext[1]; dy++)
            for (int64_t dx = 0; dx < 2 && 2 * x + dx < ext[2]; dx++) {
              sum += sc->chunk[((2 * z + dz) * SURF_CHUNK + (2 * y + dy)) * SURF_CHUNK + 2 * x + dx];
              cnt++;
            }
        nonzero |= sum != 0;
        /* mask mode: the majority over the whole 2x2x2 block, voxels outside the
         * array counting as air — the same rule volcomp_mask_encode applies, so
         * the pyramid stays exactly consistent at the array edges too */
        d[x] = cfg->mask_mode ? (uint8_t)(sum >= 4u * 255u ? 255u : 0u) : (uint8_t)((sum + cnt / 2) / cnt);
      }
    }
  if (cfg->mask_mode) { /* a majority pool can be empty where the chunk is not */
    nonzero = 0;
    for (int64_t z = 0; z < h[0]; z++)
      for (int64_t y = 0; y < h[1]; y++) {
        const uint8_t *d = l1 + ((ci[0] * (SURF_CHUNK / 2) + z) * d1 + ci[1] * (SURF_CHUNK / 2) + y) * d1 +
                           ci[2] * (SURF_CHUNK / 2);
        for (int64_t x = 0; x < h[2]; x++) nonzero |= d[x];
      }
  }
  pthread_mutex_lock(&S->lock);
  S->res->out_voxels += vox;
  pthread_mutex_unlock(&S->lock);
  S->blk[i] = nonzero != 0;
  sc->cpu_transform += surf_now() - t_job;
  surf_encode_chunk(S, 0, (unsigned)i, sc->chunk, tid, cfg->mask_mode ? (uni == 1 ? 1 : -1) : (nonzero != 0));
}

/* Pool one level into the next, block by block: level-0 chunk `i` owns a
 * (128 >> level)^3 block at every level, and 128 >> level is even down to level 3,
 * so no 2x2x2 group ever straddles two blocks. Blocks whose chunk produced no data
 * are already zero and are skipped, which is most of a sparse unit. */
static void surf_pool_job(void *ctx, int64_t i, int tid) {
  surf_state *S = ctx;
  (void)tid;
  if (!S->blk[i]) return;
  int level = S->level;
  const uint8_t *src = S->lvl[level - 1];
  uint8_t *dst = S->lvl[level];
  int64_t ds = S->ldim[level - 1], dd = S->ldim[level];
  int64_t bs = SURF_CHUNK >> (level - 1), bd = SURF_CHUNK >> level;  /* block edge, src and dst */
  int64_t ci[3] = {i >> 6, (i >> 3) & 7, i & 7};
  const int64_t *e = S->lext[level - 1];
  int64_t o0[3], n[3];
  for (int d = 0; d < 3; d++) {
    o0[d] = ci[d] * bs;
    n[d] = surf_min64(bs, e[d] - o0[d]);
    if (n[d] <= 0) return;
  }
  for (int64_t z = 0; z < (n[0] + 1) / 2; z++)
    for (int64_t y = 0; y < (n[1] + 1) / 2; y++)
      for (int64_t x = 0; x < (n[2] + 1) / 2; x++) {
        unsigned sum = 0, cnt = 0;
        for (int64_t dz = 0; dz < 2 && 2 * z + dz < n[0]; dz++)
          for (int64_t dy = 0; dy < 2 && 2 * y + dy < n[1]; dy++)
            for (int64_t dx = 0; dx < 2 && 2 * x + dx < n[2]; dx++) {
              sum += src[((o0[0] + 2 * z + dz) * ds + (o0[1] + 2 * y + dy)) * ds + o0[2] + 2 * x + dx];
              cnt++;
            }
        uint8_t v = S->cfg->mask_mode ? (uint8_t)(sum >= 4u * 255u ? 255u : 0u)
                                      : (uint8_t)((sum + cnt / 2) / cnt);
        dst[((ci[0] * bd + z) * dd + ci[1] * bd + y) * dd + ci[2] * bd + x] = v;
      }
}

static void surf_levelN_job(void *ctx, int64_t i, int tid) {
  surf_state *S = ctx;
  int level = S->level;
  int64_t g = S->ldim[level] / SURF_CHUNK;
  int64_t ci[3] = {i / (g * g), (i / g) % g, i % g};
  /* a level-L chunk covers the 2^L cube of level-0 chunks at 2^L * ci */
  int any = 0;
  for (int64_t dz = 0; dz < (1 << level) && !any; dz++)
    for (int64_t dy = 0; dy < (1 << level) && !any; dy++)
      for (int64_t dx = 0; dx < (1 << level); dx++)
        if (S->blk[(((ci[0] << level) + dz) << 6) | (((ci[1] << level) + dy) << 3) | ((ci[2] << level) + dx)]) {
          any = 1;
          break;
        }
  if (!any) return;
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
  surf_encode_chunk(S, level, (unsigned)i, sc->chunk, tid, -1);
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
    S.cdim[d] = S.chi[d] - S.clo[d] + 1;
  }
  S.cstat = calloc(1, (size_t)(S.cdim[0] * S.cdim[1] * S.cdim[2]));
  S.mask = calloc(1, (size_t)(S.rn[0] * S.rn[1] * S.rn[2]));
  if (!S.cstat) return 2;
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
    S.sc[t].pool = malloc((size_t)((5 + 2 * SURF_RING) * S.mmax[1] * S.mmax[2]));
    S.sc[t].ramp = malloc(mcap);
    S.sc[t].chunk = malloc(VOLCOMP_CHUNK_VOXELS);
    S.sc[t].enc = malloc(VOLCOMP_ENCODE_BOUND);
    S.sc[t].dec = malloc(VOLCOMP_CHUNK_VOXELS);
    S.sc[t].src = malloc((size_t)(cfg->csize * cfg->csize * cfg->csize));
    S.sc[t].tmp = malloc((size_t)(cfg->csize * cfg->csize * cfg->csize));
    if (!S.sc[t].pool || !S.sc[t].ramp || !S.sc[t].chunk || !S.sc[t].enc ||
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
  for (int t = 0; t < S.nthreads; t++) {
    res->cpu_transform += S.sc[t].cpu_transform;
    res->cpu_encode0 += S.sc[t].cpu_encode0;
  }
  if (S.error) return 3;
  for (int L = 2; L < SURF_LEVELS; L++) {
    S.level = L;
    surf_parallel_for(SURF_MAX_CHUNKS, S.nthreads, surf_pool_job, &S);
  }
  for (int L = 1; L < SURF_LEVELS; L++) {
    int64_t g = S.ldim[L] / SURF_CHUNK;
    S.level = L;
    res->chunks[L] = (unsigned)(g * g * g);
    surf_parallel_for(g * g * g, S.nthreads, surf_levelN_job, &S);
  }
  double t3 = surf_now();
  if (S.error) return 3;
  /* In mask mode a source with only specks (no 2x2x2 majority anywhere in the footprint, or nonzero only in
   * the halo) legitimately yields an empty level-0 shard: the per-chunk verify already re-decoded everything. */
  if (!cfg->mask_mode && res->src_nonzero && res->present[0] == 0) {
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
    free(S.sc[t].pool), free(S.sc[t].ramp);
    free(S.sc[t].chunk), free(S.sc[t].enc), free(S.sc[t].dec), free(S.sc[t].src), free(S.sc[t].tmp);
  }
  free(S.sc);
  free(S.mask);
  free(S.cstat);
  for (int L = 1; L < SURF_LEVELS; L++) free(S.lvl[L]);
  pthread_mutex_destroy(&S.lock);
  return 0;
}

/* --------------------------------------------------------- occupancy masks ---
 * The published prediction pyramids are exact MAX pools (checked on the Paris 4
 * recto's level 4 and the m7 models' level 3: the published coarse voxel equals
 * the max of its level-0 block for 100 % of the voxels), so a zero voxel at a
 * coarse level guarantees a whole block of zeros at level 0. That makes the
 * coarsest published level a perfect occupancy oracle: one output chunk is
 * "occupied" when any coarse voxel over the source footprint it reads - its
 * resampled extent, the dmax ramp halo and the interpolation voxel, dilated by
 * one coarse voxel - is nonzero. Everything else is air and its chunk is absent.
 *
 * This streams one chunk layer of the coarse level at a time and reduces it
 * separably (x, then y, then z), with an early exit per output chunk, so the cost
 * is a fraction of the coarse level and the memory is one layer plus the output
 * chunk grid (one byte per 128^3 output chunk). */

typedef struct {
  int64_t lo, hi;  /* inclusive coarse-voxel range an output chunk index reads */
} surf_range;

/* Is one published coarse chunk the exact 2x MAX pool of the eight finer chunks under
 * it? Returns the number of voxels that are zero in the coarse chunk although their
 * 2^3 block is not - any of those would make the occupancy mask drop published data, so
 * the coordinator only builds masks for a prediction whose sampled chunks all return 0.
 * `fine[i]` is the file for sub-position (i>>2, (i>>1)&1, i&1), or NULL for absent. */
static inline int surf_maxpool_check(const char *coarse_path, const char *const fine[8], int64_t C,
                                     uint64_t *misses, uint64_t *coarse_nz, uint64_t *pool_nz) {
  size_t n = 0;
  uint8_t *dec = malloc((size_t)(C * C * C)), *tmp = malloc((size_t)(C * C * C));
  uint8_t *coarse = calloc(1, (size_t)(C * C * C)), *finev = calloc(1, (size_t)(8 * C * C * C));
  if (!dec || !tmp || !coarse || !finev) return 2;
  const char *err = NULL;
  uint8_t *raw = shard_read_file(coarse_path, &n);
  if (raw) {
    if (blosc1_decompress(raw, n, coarse, (size_t)(C * C * C), tmp, (size_t)(C * C * C), &err) != C * C * C) {
      fprintf(stderr, "surface-maxpool-check: %s: %s\n", coarse_path, err ? err : "short chunk");
      return 3;
    }
    free(raw);
  }
  for (int i = 0; i < 8; i++) {
    if (!fine[i]) continue;
    raw = shard_read_file(fine[i], &n);
    if (!raw) continue;
    if (blosc1_decompress(raw, n, dec, (size_t)(C * C * C), tmp, (size_t)(C * C * C), &err) != C * C * C) {
      fprintf(stderr, "surface-maxpool-check: %s: %s\n", fine[i], err ? err : "short chunk");
      return 3;
    }
    free(raw);
    int64_t o[3] = {(i >> 2) * C, ((i >> 1) & 1) * C, (i & 1) * C};
    for (int64_t z = 0; z < C; z++)
      for (int64_t y = 0; y < C; y++)
        memcpy(finev + (((o[0] + z) * 2 * C + o[1] + y) * 2 * C + o[2]), dec + (z * C + y) * C, (size_t)C);
  }
  uint64_t miss = 0, cnz = 0, pnz = 0;
  for (int64_t z = 0; z < C; z++)
    for (int64_t y = 0; y < C; y++)
      for (int64_t x = 0; x < C; x++) {
        unsigned any = 0;
        for (int dz = 0; dz < 2; dz++)
          for (int dy = 0; dy < 2; dy++)
            for (int dx = 0; dx < 2; dx++)
              any |= finev[(((2 * z + dz) * 2 * C + 2 * y + dy) * 2 * C) + 2 * x + dx];
        unsigned c = coarse[(z * C + y) * C + x];
        cnz += c != 0;
        pnz += any != 0;
        if (any && !c) miss++;
      }
  free(dec), free(tmp), free(coarse), free(finev);
  if (misses) *misses = miss;
  if (coarse_nz) *coarse_nz = cnz;
  if (pool_nz) *pool_nz = pnz;
  return 0;
}

/* the coarse range one output chunk index reads along one axis */
static inline surf_range surf_chunk_range(int64_t c, int64_t out_extent, double scale, int dmax, int64_t factor,
                                   int dilate, int64_t coarse_n, bool identity) {
  int64_t o0 = c * SURF_CHUNK, o1 = surf_min64(o0 + SURF_CHUNK, out_extent);
  int64_t j0 = (int64_t)floor((double)o0 * scale) - dmax;
  int64_t j1 = (int64_t)floor((double)(o1 - 1) * scale) + (identity ? 0 : 1) + dmax;
  surf_range r = {j0 / factor - dilate, j1 / factor + dilate};
  if (r.lo < 0) r.lo = 0;
  if (r.hi > coarse_n - 1) r.hi = coarse_n - 1;
  return r;
}

/* out.bin: 64 bytes per output shard (bit (cz*8+cy)*8+cx), shards in z, y, x order -
 * the same layout the CT occupancy pass writes, so the coordinator stores it the same way. */
static inline int surface_occupancy(const char *dir, const char *out_path, int64_t csize, const int64_t coarse[3],
                             int64_t factor, const int64_t out_shape[3], double scale, int dmax, int dilate,
                             uint64_t *n_occ, uint64_t *n_tot) {
  const bool identity = scale == 1.0;
  int64_t cg[3];        /* output chunk grid */
  int64_t sg[3];        /* output shard grid */
  for (int d = 0; d < 3; d++) {
    cg[d] = (out_shape[d] + SURF_CHUNK - 1) / SURF_CHUNK;
    sg[d] = (out_shape[d] + SURF_SHARD - 1) / SURF_SHARD;
  }
  surf_range *rz = malloc((size_t)cg[0] * sizeof *rz), *ry = malloc((size_t)cg[1] * sizeof *ry),
             *rx = malloc((size_t)cg[2] * sizeof *rx);
  for (int64_t c = 0; c < cg[0]; c++) rz[c] = surf_chunk_range(c, out_shape[0], scale, dmax, factor, dilate, coarse[0], identity);
  for (int64_t c = 0; c < cg[1]; c++) ry[c] = surf_chunk_range(c, out_shape[1], scale, dmax, factor, dilate, coarse[1], identity);
  for (int64_t c = 0; c < cg[2]; c++) rx[c] = surf_chunk_range(c, out_shape[2], scale, dmax, factor, dilate, coarse[2], identity);
  uint8_t *grid = calloc(1, (size_t)(cg[0] * cg[1] * cg[2]));
  int64_t lchunks[3] = {(coarse[0] + csize - 1) / csize, (coarse[1] + csize - 1) / csize,
                        (coarse[2] + csize - 1) / csize};
  int64_t ly = lchunks[1] * csize, lx = lchunks[2] * csize;  /* padded layer extent */
  uint8_t *layer = malloc((size_t)(csize * ly * lx));
  uint8_t *dec = malloc((size_t)(csize * csize * csize)), *tmp = malloc((size_t)(csize * csize * csize));
  uint8_t *ax = malloc((size_t)(ly * cg[2]));  /* per coarse y: which output chunks x are hit */
  uint8_t *by = malloc((size_t)(cg[1] * cg[2]));
  if (!grid || !layer || !dec || !tmp || !ax || !by || !rz || !ry || !rx) {
    fprintf(stderr, "surface-occupancy: out of memory\n");
    return 2;
  }
  for (int64_t lz = 0; lz < lchunks[0]; lz++) {
    memset(layer, 0, (size_t)(csize * ly * lx));
    for (int64_t cy = 0; cy < lchunks[1]; cy++)
      for (int64_t cx = 0; cx < lchunks[2]; cx++) {
        char path[4096];
        snprintf(path, sizeof path, "%s/%lld_%lld_%lld.blosc", dir, (long long)lz, (long long)cy, (long long)cx);
        size_t n = 0;
        uint8_t *raw = shard_read_file(path, &n);
        if (!raw) continue;
        const char *err = NULL;
        int64_t got = blosc1_decompress(raw, n, dec, (size_t)(csize * csize * csize), tmp,
                                        (size_t)(csize * csize * csize), &err);
        free(raw);
        if (got != csize * csize * csize) {
          fprintf(stderr, "surface-occupancy: %s: %s\n", path, err ? err : "short chunk");
          return 3;
        }
        for (int64_t z = 0; z < csize; z++)
          for (int64_t y = 0; y < csize; y++)
            memcpy(layer + (z * ly + cy * csize + y) * lx + cx * csize, dec + (z * csize + y) * csize,
                   (size_t)csize);
      }
    for (int64_t z = 0; z < csize; z++) {
      int64_t gz = lz * csize + z;
      if (gz >= coarse[0]) break;
      const uint8_t *plane = layer + z * ly * lx;
      for (int64_t y = 0; y < coarse[1]; y++) {
        const uint8_t *row = plane + y * lx;
        uint8_t *o = ax + y * cg[2];
        for (int64_t c = 0; c < cg[2]; c++) {
          uint8_t hit = 0;
          for (int64_t x = rx[c].lo; x <= rx[c].hi && !hit; x++) hit = row[x] != 0;
          o[c] = hit;
        }
      }
      memset(by, 0, (size_t)(cg[1] * cg[2]));
      for (int64_t c = 0; c < cg[1]; c++)
        for (int64_t y = ry[c].lo; y <= ry[c].hi; y++)
          for (int64_t x = 0; x < cg[2]; x++) by[c * cg[2] + x] |= ax[y * cg[2] + x];
      for (int64_t c = 0; c < cg[0]; c++) {
        if (gz < rz[c].lo || gz > rz[c].hi) continue;
        uint8_t *g = grid + (c * cg[1]) * cg[2];
        for (int64_t i = 0; i < cg[1] * cg[2]; i++) g[i] |= by[i];
      }
    }
  }
  /* pack into one 64-byte mask per shard */
  size_t nsh = (size_t)(sg[0] * sg[1] * sg[2]);
  uint8_t *masks = calloc(nsh, 64);
  uint64_t occ = 0;
  for (int64_t cz = 0; cz < cg[0]; cz++)
    for (int64_t cy = 0; cy < cg[1]; cy++)
      for (int64_t cx = 0; cx < cg[2]; cx++) {
        if (!grid[(cz * cg[1] + cy) * cg[2] + cx]) continue;
        size_t sidx = (size_t)(((cz / 8) * sg[1] + cy / 8) * sg[2] + cx / 8);
        unsigned bit = (unsigned)(((cz % 8) * 8 + cy % 8) * 8 + cx % 8);
        masks[sidx * 64 + bit / 8] |= (uint8_t)(1u << (bit % 8));
        occ++;
      }
  FILE *f = fopen(out_path, "wb");
  if (!f || fwrite(masks, 64, nsh, f) != nsh) {
    fprintf(stderr, "surface-occupancy: cannot write %s\n", out_path);
    return 2;
  }
  fclose(f);
  if (n_occ) *n_occ = occ;
  if (n_tot) *n_tot = (uint64_t)(cg[0] * cg[1] * cg[2]);
  free(grid), free(layer), free(dec), free(tmp), free(ax), free(by), free(masks), free(rz), free(ry), free(rx);
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
                              const int64_t shape[3], const int64_t pos[3], bool mask_mode,
                              bool mask_lossless, unsigned *present, uint64_t *bytes) {
  const int64_t C = SURF_CHUNK, G = VOLCOMP_MASK_DIM;
  /* In the 2x MASK mode the coarser level IS the level below's stored grid: a
   * mask chunk stores the 2x2x2 majority pool of its own mask, which is exactly
   * the next rung's mask. So the eight 64^3 grids assemble the 128^3 chunk
   * directly — no pooling arithmetic, and nothing is lost on the way up. A
   * mask-lossless chunk stores the mask itself, so there is no shortcut: the
   * eight chunks are decoded into a 256^3 block and majority pooled, like the
   * ramp path but with the majority rule. */
  const bool grid_trick = mask_mode && !mask_lossless;
  uint8_t *big = grid_trick ? NULL : calloc(1, (size_t)(8 * C * C * C)); /* 256^3 */
  uint8_t *chunk = calloc(1, VOLCOMP_CHUNK_VOXELS);
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
      volcomp_status st =
          off + nb > n - idxn
              ? VOLCOMP_ERR_CORRUPT
              : grid_trick ? volcomp_mask_decode_stored(img + off, (size_t)nb, dec, VOLCOMP_MASK_VOXELS)
                           : volcomp_decode(img + off, (size_t)nb, dec, VOLCOMP_CHUNK_VOXELS);
      if (st != VOLCOMP_OK) {
        fprintf(stderr, "shard-pool: %s: chunk does not decode\n", in[i]);
        rc = 3;
      } else if (grid_trick) {
        int64_t o[3] = {(i >> 2) * G, ((i >> 1) & 1) * G, (i & 1) * G};
        for (int64_t z = 0; z < G; z++)
          for (int64_t y = 0; y < G; y++)
            memcpy(chunk + (((o[0] + z) * C + o[1] + y) * C + o[2]), dec + (z * G + y) * G, (size_t)G);
      } else {
        int64_t o[3] = {(i >> 2) * C, ((i >> 1) & 1) * C, (i & 1) * C};
        for (int64_t z = 0; z < C; z++)
          for (int64_t y = 0; y < C; y++)
            memcpy(big + (((o[0] + z) * 2 * C + o[1] + y) * 2 * C + o[2]), dec + (z * C + y) * C, (size_t)C);
      }
    }
    free(img);
  }
  bool any = false;
  if (grid_trick) {
    for (size_t k = 0; k < VOLCOMP_CHUNK_VOXELS && !any; k++) any = chunk[k] != 0;
  } else {
    int64_t ext[3];
    for (int d = 0; d < 3; d++) {
      ext[d] = shape[d] - pos[d] * 2 * C;
      if (ext[d] > 2 * C) ext[d] = 2 * C;
      if (ext[d] < 0) ext[d] = 0;
    }
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
          /* mask-lossless: the 2x2x2 MAJORITY, voxels outside the array counting
           * as air, exactly the rule surf_pool_job applies inside a unit */
          uint8_t v = mask_mode ? (uint8_t)(sum >= 4u * 255u ? 255u : 0u)
                                : (uint8_t)((sum + cnt / 2) / cnt);
          chunk[(z * C + y) * C + x] = v;
          any |= v != 0;
        }
  }
  uint8_t *enc = NULL;
  size_t en = 0;
  if (rc == 0 && any) {
    enc = malloc(VOLCOMP_ENCODE_BOUND);
    volcomp_status st =
        mask_lossless ? volcomp_mask_encode_lossless(chunk, enc, VOLCOMP_ENCODE_BOUND, &en)
                      : mask_mode ? volcomp_mask_encode(chunk, enc, VOLCOMP_ENCODE_BOUND, &en)
                                  : volcomp_encode(chunk, q, enc, VOLCOMP_ENCODE_BOUND, &en);
    if (st != VOLCOMP_OK) rc = 3;
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
