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
    int h = 0;
    for (int i = 0; i < m; ++i) {
        while (h >= 2) {
            f64 s1 = (p[h-1].d - p[h-2].d) / (p[h-1].r - p[h-2].r + 1e-12);
            f64 s2 = (p[i].d   - p[h-1].d) / (p[i].r   - p[h-1].r + 1e-12);
            if (s2 >= s1) --h; else break;
        }
        p[h++] = p[i];
    }
    return h;
}

static const rd_pt *pick(const rd_pt *hull, int h, f64 lambda) {
    const rd_pt *best = &hull[0];
    f64 bc = hull[0].d + lambda * hull[0].r;
    for (int i = 1; i < h; ++i) { f64 c = hull[i].d + lambda * hull[i].r; if (c < bc) { bc = c; best = &hull[i]; } }
    return best;
}

typedef struct { int n; rd_pt hull[NSTEP]; } unit_curve;

// --- public ----------------------------------------------------------------
u32 vc_rc_count_units(u32 dz, u32 dy, u32 dx, vc_rc_gran gran) {
    u32 nc = vc_nchunks(dz) * vc_nchunks(dy) * vc_nchunks(dx);
    return gran == VC_RC_PER_CHUNK ? nc : nc * (u32)BPC;
}

int vc_rc_allocate(const u8 *vol, u32 dz, u32 dy, u32 dx,
                   const vc_rc_config *cfg, vc_rc_unit *units, vc_rc_result *res) {
    const u32 ncz = vc_nchunks(dz), ncy = vc_nchunks(dy), ncx = vc_nchunks(dx);
    const u32 nunits = vc_rc_count_units(dz, dy, dx, cfg->gran);

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

    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx) {
        vc_chunk_gather(vox, vol, dz, dy, dx, cz, cy, cx);

        // For each 16^3 block in the chunk, transform it standalone (DC=its mean)
        // and build its raw (R,D) points across the step grid.
        unit_curve chunk_curve; chunk_curve.n = NSTEP;
        for (int g = 0; g < NSTEP; ++g) { chunk_curve.hull[g].r = 0; chunk_curve.hull[g].d = 0; chunk_curve.hull[g].step = STEP_GRID[g]; }

        for (u32 bz = 0; bz < BPA; ++bz)
        for (u32 by = 0; by < BPA; ++by)
        for (u32 bx = 0; bx < BPA; ++bx) {
            // slice the 16^3 block voxels
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

            if (cfg->gran == VC_RC_PER_BLOCK) {
                bcur.n = build_hull(bcur.hull, NSTEP);
                cur[ui++] = bcur;
            } else {
                for (int g = 0; g < NSTEP; ++g) { chunk_curve.hull[g].r += bcur.hull[g].r; chunk_curve.hull[g].d += bcur.hull[g].d; }
            }
        }
        if (cfg->gran == VC_RC_PER_CHUNK) {
            chunk_curve.n = build_hull(chunk_curve.hull, NSTEP);
            cur[ui++] = chunk_curve;
        }
    }

    f64 raw = (f64)((size_t)dz * dy * dx);
    // Per-chunk fixed overhead (entropy table + chunk header) is NOT part of the
    // content-rate the allocator equalizes; subtract it from the target so the
    // content budget is correct, then add it back when reporting the ratio.
    const u32 nchunks = ncz * ncy * ncx;
    f64 fixed_overhead = (f64)nchunks * VC_RC_CHUNK_OVERHEAD;
    f64 target_bytes = raw / cfg->target_ratio - fixed_overhead;
    if (target_bytes < (f64)nchunks) target_bytes = (f64)nchunks;  // sane floor

    f64 lo = 1e-6, hi = 1e9, lam = sqrt(lo*hi);
    for (int it = 0; it < 80; ++it) {
        lam = sqrt(lo * hi);
        f64 bytes = 0;
        for (u32 u = 0; u < nunits; ++u) bytes += pick(cur[u].hull, cur[u].n, lam)->r;
        if (bytes > target_bytes) lo = lam; else hi = lam;
        if (fabs(bytes - target_bytes) < target_bytes * 0.003) break;
    }

    f64 content = 0;
    for (u32 u = 0; u < nunits; ++u) {
        const rd_pt *pt = pick(cur[u].hull, cur[u].n, lam);
        units[u].step = pt->step; units[u].bytes = pt->r; units[u].dist = pt->d;
        content += pt->r;
    }
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
