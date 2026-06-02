// Lagrangian (equal-slope, single global lambda) rate allocator.
// See ratectrl.h for the public contract and design rationale.
//
// Algorithm (PLAN §2 "Rate control"):
//   1. The 16^3 block is the universal quality atom. For every block and every
//      candidate dead-zone step on a small grid, build an (R,D) operating point:
//        D  = distortion, computed two ways for comparison (PLAN distortion row):
//             PARSEVAL  = mean over coefficients of (dequant(q)-coef)^2 (cheap,
//                         no inverse transform); equals spatial MSE exactly only
//                         for an orthonormal transform.
//             TRUEMSE   = real spatial MSE after dequant + inverse DCT (exact);
//                         one extra inverse pass => ~2x the per-point cost.
//        R  = bytes, via the actual entropy coder (PROBE) OR an analytical
//             Laplacian model from the block's coefficient variance (LAPLACIAN).
//   2. Convex-hull each unit's (R,D) points.
//   3. Bisect a SINGLE global lambda; each unit independently picks the hull
//      point minimizing D + lambda*R. Equal-slope => every unit sits at the same
//      R-D slope = lambda => globally rate-optimal. Sum bytes; raise lambda to
//      compress harder, lower to spend more bits, until the achieved overall
//      ratio hits the target.
//   4. Warm-start the lambda bracket via step ~= 2.94*sqrt(lambda).
//
// Granularity:
//   PER_BLOCK : each 16^3 block keeps its own hull-chosen step (the q-field).
//   PER_CHUNK : the chunk's blocks are pooled into one R-D curve (sum bytes + sum
//               distortion at each COMMON candidate step) so one step serves the
//               whole chunk.
//
// Self-contained + reentrant: it carries its own standalone 16^3 integer DCT
// (forward + inverse) so it needs neither the chunk-tiled transform's trip
// counts nor the 8^3 subband scan; it calls the real entropy coder for rate.
// All scratch is heap (PLAN §7). Encode-side tool; never on the decode path.
#include "ratectrl.h"
#include "../../include/vc/vc.h"
#include "../config.h"
#include "../blocks.h"
#include "../core/chunk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RB        16u
#define RBVOX     (RB * RB * RB)            // 4096
#define NSTEP     16
#define BPA       (VC_CHUNK_SIDE / RB)      // blocks per chunk axis
#define BPC       (BPA * BPA * BPA)         // blocks per chunk

// Dense geometric step grid (ratio ~1.41 between neighbours) so the per-unit R-D
// convex hull has closely-spaced vertices and the global-lambda bisection can
// land near ANY target ratio even with few coarse-granularity units. Spans
// near-lossless (1.0) to very aggressive (128).
static const f32 STEP_GRID[NSTEP] = {
    1.0f, 1.5f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f,
    16.f, 24.f, 32.f, 48.f, 64.f, 90.f, 128.f, 180.f
};

// Per-entropy-call FIXED overhead the codec amortizes over a WHOLE chunk (not a
// 16^3 block): static-rANS serializes a 512-byte frequency table + a 4-byte
// length header on every call. When we PROBE rate on an isolated 16^3 block that
// fixed cost would dominate (~1 bit/coef of pure table on 4096 coefs) and floor
// the achievable ratio at ~7x — but in the real codec one table is shared by the
// 64 blocks of a 64^3 chunk, so the marginal per-block rate is the token+bypass
// CONTENT only. We therefore subtract this fixed floor from each probe to get the
// marginal content rate the allocator should equalize, and add ONE such overhead
// back per chunk when reporting the achieved ratio. RLGR/Rice carry no table, so
// their overhead is ~0 (a few header bytes). Conservative value for rANS.
#ifndef VC_RC_ENTROPY_OVERHEAD
#  if VC_CFG_ENTROPY_TAG == 3
#    define VC_RC_ENTROPY_OVERHEAD 516.0   /* rANS: 4-byte len + 512-byte table */
#  else
#    define VC_RC_ENTROPY_OVERHEAD 4.0     /* rlgr/rice: tiny header */
#  endif
#endif
// Per-CHUNK fixed payload the codec writes once (codec.c header: tag+dc+step+len)
// plus one shared entropy table. Added to total bytes when reporting ratio.
#define VC_RC_CHUNK_OVERHEAD (VC_RC_ENTROPY_OVERHEAD + 11.0)

// --- standalone 16-point orthonormal integer DCT (Q14), same matrix as
//     transform/dct_int16.c so the allocator's transform matches the codec's. --
#define RQ 14u
static const i32 RC16[16][16] = {
  {  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096},
  {  5765,  5543,  5109,  4478,  3675,  2731,  1682,   568,  -568, -1682, -2731, -3675, -4478, -5109, -5543, -5765},
  {  5681,  4816,  3218,  1130, -1130, -3218, -4816, -5681, -5681, -4816, -3218, -1130,  1130,  3218,  4816,  5681},
  {  5543,  3675,   568, -2731, -5109, -5765, -4478, -1682,  1682,  4478,  5765,  5109,  2731,  -568, -3675, -5543},
  {  5352,  2217, -2217, -5352, -5352, -2217,  2217,  5352,  5352,  2217, -2217, -5352, -5352, -2217,  2217,  5352},
  {  5109,   568, -4478, -5543, -1682,  3675,  5765,  2731, -2731, -5765, -3675,  1682,  5543,  4478,  -568, -5109},
  {  4816, -1130, -5681, -3218,  3218,  5681,  1130, -4816, -4816,  1130,  5681,  3218, -3218, -5681, -1130,  4816},
  {  4478, -2731, -5543,   568,  5765,  1682, -5109, -3675,  3675,  5109, -1682, -5765,  -568,  5543,  2731, -4478},
  {  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096},
  {  3675, -5109, -1682,  5765,  -568, -5543,  2731,  4478, -4478, -2731,  5543,   568, -5765,  1682,  5109, -3675},
  {  3218, -5681,  1130,  4816, -4816, -1130,  5681, -3218, -3218,  5681, -1130, -4816,  4816,  1130, -5681,  3218},
  {  2731, -5765,  3675,  1682, -5543,  4478,   568, -5109,  5109,  -568, -4478,  5543, -1682, -3675,  5765, -2731},
  {  2217, -5352,  5352, -2217, -2217,  5352, -5352,  2217,  2217, -5352,  5352, -2217, -2217,  5352, -5352,  2217},
  {  1682, -4478,  5765, -5109,  2731,   568, -3675,  5543, -5543,  3675,  -568, -2731,  5109, -5765,  4478, -1682},
  {  1130, -3218,  4816, -5681,  5681, -4816,  3218, -1130, -1130,  3218, -4816,  5681, -5681,  4816, -3218,  1130},
  {   568, -1682,  2731, -3675,  4478, -5109,  5543, -5765,  5765, -5543,  5109, -4478,  3675, -2731,  1682,  -568},
};
static inline void rdct1(const i32 in[16], i32 out[16]) {
    const i32 rnd = (i32)1 << (RQ - 1);
    for (u32 k = 0; k < 16; ++k) { i32 a = 0; for (u32 n = 0; n < 16; ++n) a += RC16[k][n] * in[n]; out[k] = (a + rnd) >> RQ; }
}
static inline void ridct1(const i32 in[16], i32 out[16]) {
    const i32 rnd = (i32)1 << (RQ - 1);
    for (u32 n = 0; n < 16; ++n) { i32 a = 0; for (u32 k = 0; k < 16; ++k) a += RC16[k][n] * in[k]; out[n] = (a + rnd) >> RQ; }
}

// Forward DCT of one isolated 16^3 block (voxels minus dc) -> i16 coefficients.
static void block_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc) {
    static _Thread_local i32 a[16][16][16], b[16][16][16];
    for (u32 z = 0; z < 16; ++z) for (u32 y = 0; y < 16; ++y)
        for (u32 x = 0; x < 16; ++x) a[z][y][x] = (i32)vox[z*256u+y*16u+x] - dc;
    for (u32 z = 0; z < 16; ++z) for (u32 y = 0; y < 16; ++y) rdct1(a[z][y], b[z][y]);
    for (u32 z = 0; z < 16; ++z) for (u32 x = 0; x < 16; ++x) {
        i32 col[16], oc[16]; for (u32 y = 0; y < 16; ++y) col[y] = b[z][y][x];
        rdct1(col, oc); for (u32 y = 0; y < 16; ++y) a[z][y][x] = oc[y];
    }
    for (u32 y = 0; y < 16; ++y) for (u32 x = 0; x < 16; ++x) {
        i32 col[16], oc[16]; for (u32 z = 0; z < 16; ++z) col[z] = a[z][y][x];
        rdct1(col, oc);
        for (u32 z = 0; z < 16; ++z) { i32 v = oc[z]; v = v>32767?32767:(v<-32768?-32768:v); coef[z*256u+y*16u+x] = (i16)v; }
    }
}

// Inverse DCT of one 16^3 coefficient block -> u8 voxels (+ dc), for true-MSE.
static void block_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc) {
    static _Thread_local i32 a[16][16][16], b[16][16][16];
    for (u32 z = 0; z < 16; ++z) for (u32 y = 0; y < 16; ++y)
        for (u32 x = 0; x < 16; ++x) a[z][y][x] = (i32)coef[z*256u+y*16u+x];
    for (u32 y = 0; y < 16; ++y) for (u32 x = 0; x < 16; ++x) {
        i32 col[16], oc[16]; for (u32 z = 0; z < 16; ++z) col[z] = a[z][y][x];
        ridct1(col, oc); for (u32 z = 0; z < 16; ++z) b[z][y][x] = oc[z];
    }
    for (u32 z = 0; z < 16; ++z) for (u32 x = 0; x < 16; ++x) {
        i32 col[16], oc[16]; for (u32 y = 0; y < 16; ++y) col[y] = b[z][y][x];
        ridct1(col, oc); for (u32 y = 0; y < 16; ++y) a[z][y][x] = oc[y];
    }
    for (u32 z = 0; z < 16; ++z) for (u32 y = 0; y < 16; ++y) {
        i32 row[16], oc[16]; for (u32 x = 0; x < 16; ++x) row[x] = a[z][y][x];
        ridct1(row, oc);
        for (u32 x = 0; x < 16; ++x) { i32 v = oc[x] + dc; v = v<0?0:(v>255?255:v); vox[z*256u+y*16u+x] = (u8)v; }
    }
}

// --- dead-zone quant on a contiguous block (matches quant/subband.c) -------
static inline void quant_block(i16 *restrict qb, const i16 *restrict cb, f32 step) {
    f32 inv = 1.0f / step, half = 0.5f * step;
    for (u32 i = 0; i < RBVOX; ++i) {
        f32 c = (f32)cb[i], aa = c < 0.f ? -c : c;
        i32 m = (aa >= half);
        i32 lvl = m * ((i32)(aa * inv - 0.5f) + 1);
        qb[i] = (i16)(c < 0.f ? -lvl : lvl);
    }
}
static inline void dequant_block(i16 *restrict cb, const i16 *restrict qb, f32 step) {
    for (u32 i = 0; i < RBVOX; ++i) {
        i32 l = qb[i]; i32 v = (i32)lrintf((f32)l * step);
        cb[i] = (i16)(v>32767?32767:(v<-32768?-32768:v));
    }
}

// PARSEVAL coefficient-domain MSE estimate (no inverse transform).
static f64 parseval_mse(const i16 *restrict cb, f32 step) {
    f64 s = 0.0;
    for (u32 i = 0; i < RBVOX; ++i) {
        f32 c = (f32)cb[i], a = c<0.f?-c:c;
        i32 lvl = (a >= 0.5f*step) ? ((i32)(a*(1.0f/step)-0.5f)+1) : 0;
        f32 e = a - (f32)lvl*step;
        s += (f64)e*(f64)e;
    }
    // Orthonormal DCT => coefficient energy == spatial energy; divide by RBVOX
    // for per-voxel MSE.
    return s / RBVOX;
}

// TRUE spatial MSE added by quantization: inverse-transform unquantized vs
// quantized coefficients and difference (isolates quant error from the unquant
// inverse rounding).
static f64 true_mse(const i16 *restrict cb, f32 step, i16 *restrict qb,
                    i16 *restrict cbq, u8 *restrict v0, u8 *restrict v1) {
    block_inv(v0, cb, 128);
    quant_block(qb, cb, step);
    dequant_block(cbq, qb, step);
    block_inv(v1, cbq, 128);
    f64 s = 0.0;
    for (u32 i = 0; i < RBVOX; ++i) { f64 e = (f64)v1[i] - (f64)v0[i]; s += e*e; }
    return s / RBVOX;
}

// Analytical Laplacian rate model: bits/coef ~ 0.5*log2(c*(sigma/step)^2),
// floored at 0; a near-1-pass alternative to the entropy probe.
static f64 laplacian_bytes(const i16 *restrict cb, f32 step) {
    f64 sumsq = 0.0;
    for (u32 i = 0; i < RBVOX; ++i) { f64 c = cb[i]; sumsq += c*c; }
    f64 sigma = sqrt(sumsq / RBVOX + 1e-9);
    if (sigma < 1e-3) return 8.0;
    f64 ratio = sigma / step;
    f64 bits = 0.5 * log2((2.0*2.718281828*2.718281828/12.0) * ratio * ratio);
    if (bits < 0.0) bits = 0.0;
    f64 bytes = RBVOX * bits / 8.0;
    return bytes < 8.0 ? 8.0 : bytes;
}

// --- R-D point + hull ------------------------------------------------------
typedef struct { f64 r, d; f32 step; } rd_pt;

// Compute one raw (R,D) point for a block at `step`. Accumulates the
// Parseval-vs-true divergence stats when both are available.
static void raw_point(rd_pt *pt, const i16 *restrict cb, const vc_rc_config *cfg,
                      f32 step, i16 *qb, i16 *cbq, u8 *v0, u8 *v1, u8 *ebuf,
                      f64 *par_acc, f64 *tru_acc, u32 *acc_n) {
    pt->step = step;
    // Rate
    if (cfg->model == VC_RC_MODEL_LAPLACIAN) {
        pt->r = laplacian_bytes(cb, step);
    } else {
        quant_block(qb, cb, step);
        f64 bytes = (f64)VC_ENTROPY_ENC(ebuf, RBVOX*4+64, qb, RBVOX);
        // Subtract the per-call fixed entropy overhead (table/header): in the
        // real codec it is shared across a whole chunk, so the MARGINAL per-block
        // rate is the content-only part. Floored at a tiny positive value so a
        // near-empty block still costs a hair (keeps the hull well-ordered).
        bytes -= VC_RC_ENTROPY_OVERHEAD;
        pt->r = bytes < 1.0 ? 1.0 : bytes;
    }
    // Distortion (and divergence accounting)
    f64 par = parseval_mse(cb, step);
    if (cfg->dist == VC_RC_DIST_TRUEMSE) {
        f64 tru = true_mse(cb, step, qb, cbq, v0, v1);
        pt->d = tru;
        if (par_acc) { *par_acc += par; *tru_acc += tru; (*acc_n)++; }
    } else {
        pt->d = par;
        // even when using Parseval for allocation, sample true-MSE occasionally
        // for the divergence report (cheap: only at the mid step).
        if (par_acc && (step == STEP_GRID[NSTEP/2])) {
            f64 tru = true_mse(cb, step, qb, cbq, v0, v1);
            *par_acc += par; *tru_acc += tru; (*acc_n)++;
        }
    }
}

static int build_hull(rd_pt *p, int n) {
    for (int i = 1; i < n; ++i) {
        rd_pt k = p[i]; int j = i - 1;
        while (j >= 0 && (p[j].r > k.r || (p[j].r == k.r && p[j].d > k.d))) { p[j+1] = p[j]; --j; }
        p[j+1] = k;
    }
    int m = 0;
    for (int i = 0; i < n; ++i) if (m == 0 || p[i].d < p[m-1].d - 1e-9) p[m++] = p[i];
    // Lower convex hull (minimize D for given R). Points are R-ascending; D is
    // decreasing, so segment slopes dD/dR are negative and, for a convex frontier,
    // INCREASE (become less steep) with R. Drop the middle vertex p[h-1] whenever
    // it lies ABOVE the chord p[h-2]->p[i], i.e. when the earlier slope is not
    // less than the later slope (s1 >= s2) — that vertex is interior/dominated.
    int h = 0;
    for (int i = 0; i < m; ++i) {
        while (h >= 2) {
            f64 s1 = (p[h-1].d - p[h-2].d) / (p[h-1].r - p[h-2].r + 1e-12);
            f64 s2 = (p[i].d   - p[h-1].d) / (p[i].r   - p[h-1].r + 1e-12);
            if (s1 >= s2) --h; else break;
        }
        p[h++] = p[i];
    }
    return h;
}

// Lagrangian pick restricted to hull points with step in [smin, smax]. If none
// qualify (a unit's hull has no vertex in the window), snap to the hull vertex
// whose step is nearest the window so the unit still respects the bound.
static const rd_pt *pick_window(const rd_pt *hull, int h, f64 lambda, f32 smin, f32 smax) {
    const rd_pt *best = NULL; f64 bc = 0;
    for (int i = 0; i < h; ++i) {
        if (hull[i].step < smin || hull[i].step > smax) continue;
        f64 c = hull[i].d + lambda * hull[i].r;
        if (!best || c < bc) { bc = c; best = &hull[i]; }
    }
    if (best) return best;
    best = &hull[0]; f64 bd = 1e300;
    for (int i = 0; i < h; ++i) {
        f32 s = hull[i].step, cl = s < smin ? smin : (s > smax ? smax : s);
        f64 d = fabs((f64)s - (f64)cl);
        if (d < bd) { bd = d; best = &hull[i]; }
    }
    return best;
}

typedef struct { int n; rd_pt hull[NSTEP]; } unit_curve;

// Bytes of a unit if it were quantized at exactly `step`, linearly interpolated
// between the unit's bracketing hull vertices in step (R monotone in step). Used
// only to find the uniform base step; the final allocation snaps to real hull
// vertices. The hull is sorted ascending by R (descending by step).
static f64 unit_bytes_at_step(const rd_pt *hull, int h, f32 step) {
    // find vertices bracketing `step` by step value
    const rd_pt *lo_s = NULL, *hi_s = NULL;   // lo_s.step <= step <= hi_s.step
    for (int i = 0; i < h; ++i) {
        if (hull[i].step <= step && (!lo_s || hull[i].step > lo_s->step)) lo_s = &hull[i];
        if (hull[i].step >= step && (!hi_s || hull[i].step < hi_s->step)) hi_s = &hull[i];
    }
    if (lo_s && hi_s) {
        if (lo_s == hi_s) return lo_s->r;
        f64 t = ((f64)step - lo_s->step) / ((f64)hi_s->step - lo_s->step);
        return lo_s->r + t * (hi_s->r - lo_s->r);
    }
    return lo_s ? lo_s->r : (hi_s ? hi_s->r : 0.0);
}

// Bisect the continuous uniform step whose summed unit bytes == target_bytes.
static f32 uniform_base_step(const unit_curve *cur, u32 nunits, f64 target_bytes) {
    f64 lo = STEP_GRID[0], hi = STEP_GRID[NSTEP-1];
    f32 base = (f32)sqrt(lo * hi);
    for (int it = 0; it < 50; ++it) {
        base = (f32)sqrt(lo * hi);
        f64 bytes = 0;
        for (u32 u = 0; u < nunits; ++u) bytes += unit_bytes_at_step(cur[u].hull, cur[u].n, base);
        if (bytes > target_bytes) lo = base; else hi = base;  // coarser => fewer bytes
        if (fabs(bytes - target_bytes) < target_bytes * 0.005) break;
    }
    return base;
}

// FAITHFUL per-chunk R-D: encode the chunk sub-volume with the REAL codec at
// each candidate step and measure exact bytes + exact decode-MSE. This is the
// only model guaranteed consistent with what the codec actually emits, so the
// resulting per-chunk allocation provably matches-or-beats uniform-q (uniform is
// a feasible point of the same hull). `cbuf` is a reusable VC_CS^3 voxel buffer.
static void chunk_curve_real(unit_curve *uc, const u8 *restrict cbuf,
                             f64 *par_acc, f64 *tru_acc, u32 *acc_n) {
    const u32 cs = VC_CS;
    for (int g = 0; g < NSTEP; ++g) {
        u8 *arc = NULL; size_t alen = 0;
        vc_encode_volume(cbuf, cs, cs, cs, STEP_GRID[g], &arc, &alen);
        u8 *rec = NULL; u32 rd, rh, rw;
        vc_decode_volume(arc, alen, &rec, &rd, &rh, &rw);
        f64 s = 0.0;
        for (size_t i = 0; i < VC_CVOX; ++i) { f64 e = (f64)rec[i] - (f64)cbuf[i]; s += e*e; }
        f64 m = s / VC_CVOX;
        uc->hull[g].r = (f64)alen;
        uc->hull[g].d = m;
        uc->hull[g].step = STEP_GRID[g];
        if (acc_n && g == NSTEP/2) { *par_acc += m; *tru_acc += m; (*acc_n)++; }
        free(arc); free(rec);
    }
    uc->n = build_hull(uc->hull, NSTEP);
}

// ===========================================================================
// FAST-ENCODE tier: no per-step grid, no convex hull, no per-point entropy
// probing. Each block is transformed ONCE; its coefficient statistics (variance
// or AC sum-of-squares) are cached, then a single global parameter is solved.
// ===========================================================================

// Per-block cached coefficient statistics for the fast models.
typedef struct {
    f64 acsumsq;  // sum of squared AC coefficients (DC excluded)
    f64 sigma;    // RMS of AC coefficients (sqrt(acsumsq/RBVOX))
} blk_stat;

// Closed-form rate of a block at a step, from its cached sigma. Same Laplacian
// model as laplacian_bytes() but reads the precomputed sigma (no second pass).
static inline f64 cf_bytes(const blk_stat *bs, f32 step) {
    if (bs->sigma < 1e-3) return 1.0;
    f64 ratio = (f64)bs->sigma / (f64)step;
    f64 bits = 0.5 * log2((2.0*2.718281828*2.718281828/12.0) * ratio * ratio);
    if (bits < 0.0) bits = 0.0;
    f64 bytes = RBVOX * bits / 8.0;
    return bytes < 1.0 ? 1.0 : bytes;
}

// Closed-form distortion estimate: dead-zone quant MSE ~= step^2/12 per nonzero
// coef; for a block all coefs below ~step/2 vanish (contribute their own energy
// as error). Cheap Parseval-domain estimate without re-quantizing.
static inline f64 cf_dist(const blk_stat *bs, f32 step) {
    // High-rate quantizer distortion floor + the energy lost in the dead zone is
    // bounded by the uniform-quant term; step^2/12 per voxel is the standard
    // estimate and is monotone in step, which is all the reporting needs.
    (void)bs;
    return (f64)step * (f64)step / 12.0;
}

// Gather AC coefficient statistics for every 16^3 block of the volume in ONE
// forward-DCT pass per block (no grid, no probing). Fills `stats[ui]` and
// returns the unit count. The block layout matches vc_rc_count_units(PER_BLOCK).
static u32 gather_block_stats(const u8 *vol, u32 dz, u32 dy, u32 dx,
                              blk_stat *stats) {
    const u32 ncz = vc_nchunks(dz), ncy = vc_nchunks(dy), ncx = vc_nchunks(dx);
    u8  *vox = (u8 *)malloc(VC_CVOX);
    u8  *bvx = (u8 *)malloc(RBVOX);
    i16 *cb  = (i16*)malloc(RBVOX * sizeof(i16));
    if (!vox || !bvx || !cb) { free(vox); free(bvx); free(cb); return 0; }
    u32 ui = 0;
    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx) {
        vc_chunk_gather(vox, vol, dz, dy, dx, cz, cy, cx);
        for (u32 bz = 0; bz < BPA; ++bz)
        for (u32 by = 0; by < BPA; ++by)
        for (u32 bx = 0; bx < BPA; ++bx) {
            const u32 oz = bz*RB, oy = by*RB, ox = bx*RB;
            for (u32 z = 0; z < RB; ++z) for (u32 y = 0; y < RB; ++y)
                memcpy(bvx + (z*RB+y)*RB,
                       vox + (size_t)(oz+z)*VC_CHUNK_SIDE*VC_CHUNK_SIDE + (size_t)(oy+y)*VC_CHUNK_SIDE + ox, RB);
            u64 sum = 0; for (u32 i = 0; i < RBVOX; ++i) sum += bvx[i];
            i32 dc = (i32)((sum + RBVOX/2) / RBVOX);
            block_fwd(cb, bvx, dc);
            f64 ss = 0.0;
            for (u32 i = 1; i < RBVOX; ++i) { f64 c = cb[i]; ss += c*c; }
            stats[ui].acsumsq = ss;
            stats[ui].sigma   = sqrt(ss / RBVOX + 1e-9);
            ++ui;
        }
    }
    free(vox); free(bvx); free(cb);
    return ui;
}

// Per-atom variance-modulated step from a global base step. The equal-slope
// Lagrangian for a Gaussian/Laplacian source at fixed total rate is "reverse
// water-filling": blocks with energy above the water level are coded at the base
// step; blocks below it are DROPPED (pushed to a coarse step ~ their own sigma so
// they cost ~0). A constant step wastes bits on air and starves structure on a
// MIXED volume — so we let low-sigma atoms float coarser. This is the per-atom
// variance term the task asks for, in closed form (no grid, no probe):
//   step_b = base                       if sigma_b >= base     (codeable energy)
//          = clamp(base * base/sigma_b)  if sigma_b <  base     (push coarse/drop)
// so an air block (sigma << base) gets a very coarse step (≈ dropped) while a
// structured block (sigma >= base) keeps the fine base step.
static inline f32 cf_step_for(const blk_stat *bs, f32 base, f32 smax) {
    f64 sig = bs->sigma;
    if (sig >= (f64)base || sig < 1e-6) {
        return base;
    }
    f64 s = (f64)base * ((f64)base / sig);   // reverse water-fill: coarser when quiet
    if (s > (f64)smax) s = (f64)smax;
    return (f32)s;
}

// (a) CLOSED-FORM q-from-lambda + per-atom variance. step = 2.94*sqrt(lambda) is
// the base; each atom's actual step is the variance-modulated cf_step_for() above
// (reverse water-filling). We bisect the global BASE step (equiv. to bisecting
// lambda) over the closed-form rate sum only — NO grid, NO hull, NO per-point
// entropy probe: O(nunits) per bisection iteration over cached sigmas. The base
// maps back to lambda via lambda = (base/2.94)^2 for reporting.
static int alloc_closedform(blk_stat *stats, u32 nunits, u32 nchunks,
                            f64 target_bytes, vc_rc_unit *units, vc_rc_result *res,
                            f64 raw, f64 fixed_overhead) {
    const f32 smax = STEP_GRID[NSTEP-1];
    f64 lo = STEP_GRID[0], hi = STEP_GRID[NSTEP-1];
    f32 base = (f32)sqrt(lo*hi);
    for (int it = 0; it < 60; ++it) {
        base = (f32)sqrt(lo*hi);
        f64 bytes = 0;
        for (u32 u = 0; u < nunits; ++u)
            bytes += cf_bytes(&stats[u], cf_step_for(&stats[u], base, smax));
        if (bytes > target_bytes) lo = base; else hi = base;  // coarser => fewer bytes
        if (fabs(bytes - target_bytes) < target_bytes * 0.003) break;
    }
    f64 content = 0;
    for (u32 u = 0; u < nunits; ++u) {
        f32 step = cf_step_for(&stats[u], base, smax);
        f64 b = cf_bytes(&stats[u], step);
        units[u].step = step;        // per-atom variance-modulated step
        units[u].bytes = b;
        units[u].dist = cf_dist(&stats[u], step);
        content += b;
    }
    f32 step = base;
    f64 total = content + fixed_overhead;
    if (res) {
        // lambda back from step ~= 2.94*sqrt(lambda) => lambda = (step/2.94)^2
        f64 l = (f64)step / 2.94; res->lambda = l*l;
        res->total_bytes = total; res->raw_bytes = raw;
        res->achieved_ratio = raw / (total > 1 ? total : 1);
        res->n_units = nunits;
        res->parseval_mse = 0; res->true_mse = 0;
    }
    (void)nchunks;
    return 0;
}

// (b) SINGLE-PASS FEEDBACK controller. Walk blocks in raster order; keep a
// running bytes budget (target_bytes/nunits per block). After each block adjust a
// running step by a proportional law toward the cumulative budget: if we are over
// budget so far, coarsen the step; if under, refine it. step is clamped to the
// grid span. ONE forward DCT per block already done (cached sigma); the predicted
// per-block cost is cf_bytes at the current step, so the whole pass is O(nunits)
// with no bisection — strictly one sweep. This is a classic VBR leaky-bucket / CBR
// feedback regulator (cf. MPEG TM5 rate control), the fastest possible mode.
static int alloc_feedback(blk_stat *stats, u32 nunits, u32 nchunks,
                          f64 target_bytes, vc_rc_unit *units, vc_rc_result *res,
                          f64 raw, f64 fixed_overhead) {
    const f64 per_blk = target_bytes / (nunits ? nunits : 1);
    const f32 smin = STEP_GRID[0], smax = STEP_GRID[NSTEP-1];
    // Initialise the running step at the constant-step solution's ballpark so the
    // controller starts near the answer (one cheap mean-sigma estimate) — this is
    // still single-pass over the data; it just seeds the loop sensibly.
    f64 msig = 0; for (u32 u = 0; u < nunits; ++u) msig += stats[u].sigma;
    msig = nunits ? msig / nunits : 1.0;
    // crude seed: step s.t. mean block costs ~per_blk bytes (invert cf_bytes)
    f32 step = (f32)(msig / sqrt((per_blk*8.0/RBVOX)>0 ? exp2(2.0*(per_blk*8.0/RBVOX)) / (2.0*2.718281828*2.718281828/12.0) : 1.0));
    if (!(step > smin)) step = smin;
    if (step > smax) step = smax;
    f64 cum_bytes = 0, cum_budget = 0, content = 0;
    const f64 kp = 0.5;  // proportional gain on relative budget error
    for (u32 u = 0; u < nunits; ++u) {
        f64 b = cf_bytes(&stats[u], step);
        units[u].step = step;
        units[u].bytes = b;
        units[u].dist = cf_dist(&stats[u], step);
        content += b;
        cum_bytes += b; cum_budget += per_blk;
        // proportional update: relative cumulative error -> multiplicative step
        // correction. over budget (err>0) => coarsen (step up).
        f64 err = (cum_bytes - cum_budget) / (cum_budget + 1.0);
        f64 f = 1.0 + kp * err;
        if (f < 0.5) f = 0.5;
        if (f > 2.0) f = 2.0;
        step = (f32)((f64)step * f);
        if (step < smin) step = smin;
        if (step > smax) step = smax;
    }
    f64 total = content + fixed_overhead;
    if (res) {
        res->lambda = 0; res->total_bytes = total; res->raw_bytes = raw;
        res->achieved_ratio = raw / (total > 1 ? total : 1);
        res->n_units = nunits;
        res->parseval_mse = 0; res->true_mse = 0;
    }
    (void)nchunks;
    return 0;
}

// Honest whole-volume true-MSE of a per-block q-field (see header). Uses the
// codec's exact 16^3 DCT + dead-zone quant + inverse; no entropy, no per-block
// archive overhead — pure allocation quality.
f64 vc_rc_qfield_truemse(const u8 *vol, u32 dz, u32 dy, u32 dx,
                         const vc_rc_unit *units) {
    const u32 ncz = vc_nchunks(dz), ncy = vc_nchunks(dy), ncx = vc_nchunks(dx);
    u8  *vox = (u8 *)malloc(VC_CVOX);
    u8  *bvx = (u8 *)malloc(RBVOX);
    i16 *cb  = (i16*)malloc(RBVOX*sizeof(i16));
    i16 *qb  = (i16*)malloc(RBVOX*sizeof(i16));
    i16 *cbq = (i16*)malloc(RBVOX*sizeof(i16));
    u8  *vr  = (u8 *)malloc(RBVOX);
    if (!vox||!bvx||!cb||!qb||!cbq||!vr){free(vox);free(bvx);free(cb);free(qb);free(cbq);free(vr);return -1.0;}
    f64 sse = 0.0; u64 nv = 0; u32 ui = 0;
    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx) {
        vc_chunk_gather(vox, vol, dz, dy, dx, cz, cy, cx);
        for (u32 bz = 0; bz < BPA; ++bz)
        for (u32 by = 0; by < BPA; ++by)
        for (u32 bx = 0; bx < BPA; ++bx, ++ui) {
            const u32 oz=bz*RB, oy=by*RB, ox=bx*RB;
            for (u32 z=0;z<RB;++z) for (u32 y=0;y<RB;++y)
                memcpy(bvx+(z*RB+y)*RB,
                       vox+(size_t)(oz+z)*VC_CHUNK_SIDE*VC_CHUNK_SIDE+(size_t)(oy+y)*VC_CHUNK_SIDE+ox, RB);
            u64 sum=0; for (u32 i=0;i<RBVOX;++i) sum+=bvx[i];
            i32 dc=(i32)((sum+RBVOX/2)/RBVOX);
            block_fwd(cb, bvx, dc);
            f32 step = units[ui].step;
            quant_block(qb, cb, step);
            dequant_block(cbq, qb, step);
            block_inv(vr, cbq, dc);
            // only count voxels inside the real volume (chunk may be edge-padded)
            for (u32 z=0;z<RB;++z) for (u32 y=0;y<RB;++y) for (u32 x=0;x<RB;++x) {
                u32 az=cz*VC_CHUNK_SIDE+oz+z, ay=cy*VC_CHUNK_SIDE+oy+y, axx=cx*VC_CHUNK_SIDE+ox+x;
                if (az<dz && ay<dy && axx<dx) {
                    f64 e=(f64)vr[(z*RB+y)*RB+x]-(f64)bvx[(z*RB+y)*RB+x];
                    sse += e*e; ++nv;
                }
            }
        }
    }
    free(vox);free(bvx);free(cb);free(qb);free(cbq);free(vr);
    return nv ? sse/(f64)nv : 0.0;
}

// --- public ----------------------------------------------------------------
u32 vc_rc_count_units(u32 dz, u32 dy, u32 dx, vc_rc_gran gran) {
    u32 nc = vc_nchunks(dz) * vc_nchunks(dy) * vc_nchunks(dx);
    return gran == VC_RC_PER_CHUNK ? nc : nc * (u32)BPC;
}

int vc_rc_allocate(const u8 *vol, u32 dz, u32 dy, u32 dx,
                   const vc_rc_config *cfg, vc_rc_unit *units, vc_rc_result *res) {
    const u32 ncz = vc_nchunks(dz), ncy = vc_nchunks(dy), ncx = vc_nchunks(dx);
    const u32 nunits = vc_rc_count_units(dz, dy, dx, cfg->gran);
    const u32 nchunks0 = ncz * ncy * ncx;

    // ---- FAST-ENCODE tier (CLOSED-FORM / FEEDBACK): one fwd DCT per block, no
    // grid, no hull, no per-point entropy probe. Per-block granularity only (the
    // q-field is the fast tier's product); per-chunk falls through to the faithful
    // path. ----
    if ((cfg->model == VC_RC_MODEL_CLOSEDFORM || cfg->model == VC_RC_MODEL_FEEDBACK)
        && cfg->gran == VC_RC_PER_BLOCK) {
        blk_stat *stats = (blk_stat *)malloc((size_t)nunits * sizeof(blk_stat));
        if (!stats) return 1;
        u32 got = gather_block_stats(vol, dz, dy, dx, stats);
        if (got != nunits) { free(stats); return 1; }
        f64 raw = (f64)((size_t)dz * dy * dx);
        f64 fixed_overhead = (f64)nchunks0 * VC_RC_CHUNK_OVERHEAD;
        f64 target_bytes = raw / cfg->target_ratio - fixed_overhead;
        if (target_bytes < (f64)nchunks0) target_bytes = (f64)nchunks0;
        int rc = (cfg->model == VC_RC_MODEL_CLOSEDFORM)
            ? alloc_closedform(stats, nunits, nchunks0, target_bytes, units, res, raw, fixed_overhead)
            : alloc_feedback  (stats, nunits, nchunks0, target_bytes, units, res, raw, fixed_overhead);
        free(stats);
        return rc;
    }

    u8  *vox = (u8 *)malloc(VC_CVOX);
    u8  *bvx = (u8 *)malloc(RBVOX);
    i16 *coef= (i16*)malloc(VC_CVOX * sizeof(i16));
    i16 *cb  = (i16*)malloc(RBVOX * sizeof(i16));
    i16 *qb  = (i16*)malloc(RBVOX * sizeof(i16));
    i16 *cbq = (i16*)malloc(RBVOX * sizeof(i16));
    u8  *v0  = (u8 *)malloc(RBVOX);
    u8  *v1  = (u8 *)malloc(RBVOX);
    u8  *ebuf= (u8 *)malloc(RBVOX * 4 + 64);
    unit_curve *cur = (unit_curve*)malloc((size_t)nunits * sizeof(unit_curve));
    if (!vox||!bvx||!coef||!cb||!qb||!cbq||!v0||!v1||!ebuf||!cur) {
        free(vox);free(bvx);free(coef);free(cb);free(qb);free(cbq);free(v0);free(v1);free(ebuf);free(cur);
        return 1;
    }

    f64 par_acc = 0, tru_acc = 0; u32 acc_n = 0;
    u32 ui = 0;

    const u32 nchunks = ncz * ncy * ncx;
    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx) {
        vc_chunk_gather(vox, vol, dz, dy, dx, cz, cy, cx);

        if (cfg->gran == VC_RC_PER_CHUNK) {
            // FAITHFUL: build the chunk's R-D curve with the REAL codec (exact
            // bytes incl. overhead, exact decode-MSE). One unit per chunk.
            chunk_curve_real(&cur[ui++], vox, &par_acc, &tru_acc, &acc_n);
            continue;
        }

        // PER_BLOCK: the codec cannot (yet) store a per-16^3 q-field in the chunk
        // header, so per-block uses a self-contained 16^3-block model (standalone
        // DCT + dead-zone + entropy probe) to RANK block difficulty and PROJECT
        // the achievable ratio/quality of a q-field. Rate proxy subtracts the
        // per-call entropy table overhead (shared per-chunk in reality).
        for (u32 bz = 0; bz < BPA; ++bz)
        for (u32 by = 0; by < BPA; ++by)
        for (u32 bx = 0; bx < BPA; ++bx) {
            const u32 oz = bz*RB, oy = by*RB, ox = bx*RB;
            for (u32 z = 0; z < RB; ++z) for (u32 y = 0; y < RB; ++y)
                memcpy(bvx + (z*RB+y)*RB,
                       vox + (size_t)(oz+z)*VC_CHUNK_SIDE*VC_CHUNK_SIDE + (size_t)(oy+y)*VC_CHUNK_SIDE + ox, RB);
            u64 sum = 0; for (u32 i = 0; i < RBVOX; ++i) sum += bvx[i];
            i32 dc = (i32)((sum + RBVOX/2) / RBVOX);
            block_fwd(cb, bvx, dc);

            unit_curve bcur; bcur.n = NSTEP;
            for (int g = 0; g < NSTEP; ++g)
                raw_point(&bcur.hull[g], cb, cfg, STEP_GRID[g], qb, cbq, v0, v1, ebuf,
                          &par_acc, &tru_acc, &acc_n);
            bcur.n = build_hull(bcur.hull, NSTEP);
            cur[ui++] = bcur;
        }
    }

    f64 raw = (f64)((size_t)dz * dy * dx);
    // Per-CHUNK curves already include all codec overhead (real encode), so no
    // extra fixed_overhead. Per-BLOCK proxies exclude the shared per-chunk table,
    // so add ONE table+header back per chunk when reporting the ratio.
    f64 fixed_overhead = (cfg->gran == VC_RC_PER_CHUNK)
                         ? 0.0 : (f64)nchunks * VC_RC_CHUNK_OVERHEAD;
    f64 target_bytes = raw / cfg->target_ratio - fixed_overhead;
    if (target_bytes < (f64)nchunks) target_bytes = (f64)nchunks;  // sane floor

    // Single global-lambda bisection (equal-slope condition): each unit picks the
    // hull point minimizing D + lambda*R; raising lambda compresses harder. This
    // hits ANY feasible target ratio precisely (the allocator's core job).
    //
    // step_window (>=1) optionally bounds each unit to [base/W, base*W] around the
    // uniform base step that hits the target. NOTE (empirical, this scroll data):
    // a bounded window does NOT recover a PSNR win, because the MSE-optimal
    // allocation here is genuinely "bang-bang" (structure units want near-lossless
    // step ~1-4, air units want the coarsest) — exactly PLAN §2's "MSE-optimal
    // allocation provably starves HF / blur is the optimizer working as designed".
    // Any window wide enough to let structure stay fine is wide enough to be
    // bang-bang; any narrower window over-compresses. So the field is honored for
    // experimentation but the recommended use is unbounded (the target-hitting is
    // the deliverable; the quality objective is the separate distortion-metric
    // axis, PLAN §2 distortion row). `uniform_base_step` is still computed for the
    // diagnostic base step. The HONEST result: variable MSE-allocation MATCHES
    // uniform PSNR at a given ratio; the win is RATIO HEADROOM at fixed quality
    // (per-16^3 reaches 50x where per-chunk/uniform plateau ~35x), not PSNR.
    f64 W = cfg->step_window >= 1.0 ? cfg->step_window : 1e9;  // 1e9 ~= unbounded
    f32 base = uniform_base_step(cur, nunits, target_bytes);
    f32 smin = (f32)((f64)base / W), smax = (f32)((f64)base * W);
    (void)base;

    f64 lo = 1e-9, hi = 1e12, lam = sqrt(lo*hi);
    #define PICK_WIN(u, L) pick_window(cur[(u)].hull, cur[(u)].n, (L), smin, smax)
    for (int it = 0; it < 100; ++it) {
        lam = sqrt(lo * hi);
        f64 bytes = 0;
        for (u32 u = 0; u < nunits; ++u) bytes += PICK_WIN(u, lam)->r;
        if (bytes > target_bytes) lo = lam; else hi = lam;
        if (fabs(bytes - target_bytes) < target_bytes * 0.003) break;
    }

    f64 content = 0;
    for (u32 u = 0; u < nunits; ++u) {
        const rd_pt *pt = PICK_WIN(u, lam);
        units[u].step = pt->step; units[u].bytes = pt->r; units[u].dist = pt->d;
        content += pt->r;
    }
    #undef PICK_WIN
    f64 total = content + fixed_overhead;
    if (res) {
        res->lambda = lam; res->total_bytes = total; res->raw_bytes = raw;
        res->achieved_ratio = raw / (total > 1 ? total : 1);
        res->n_units = nunits;
        res->parseval_mse = acc_n ? par_acc/acc_n : 0;
        res->true_mse = acc_n ? tru_acc/acc_n : 0;
    }

    free(vox);free(bvx);free(coef);free(cb);free(qb);free(cbq);free(v0);free(v1);free(ebuf);free(cur);
    return 0;
}
