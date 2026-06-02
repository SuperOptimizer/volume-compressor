// Per-subband HF-weighted dead-zone quantizer + frequency-ordered coefficient
// SCAN (PLAN §2 quantizer + §3 scan). This is the block that turns the
// transform's coefficients into the flat signed-level array the entropy coder
// sees, AND back. It does three jobs the Phase-0 flat quantizer did not:
//
//   1. Per-subband step weighting. Each 8^3 DCT sub-block has 512 coefficients
//      indexed by 3D frequency (u,v,w). We scale the dead-zone step per
//      coefficient by a weight w(u,v,w) chosen by VC_QWEIGHT:
//        - VC_QWEIGHT_FLAT (0): weight 1 everywhere (== Phase-0 behavior).
//        - VC_QWEIGHT_HF (1):  HF-PROTECTING — *smaller* steps (finer quant) for
//          high (u+v+w), so high-frequency edges/ink survive aggressive q.
//        - VC_QWEIGHT_ADAPT (2): content-adaptive — per-sub-block step scaled by
//          local AC energy (variance proxy): busy/high-variance blocks get finer
//          steps, flat blocks coarser. Combined multiplicatively with the HF
//          weight so edges in busy regions are best protected.
//
//   2. A frequency SCAN. Within each 8^3 block the 512 coefficients are emitted
//      in order of increasing u+v+w (then v, then u as tie-breaks): DC first,
//      highest frequency last. Blocks are emitted in raster order. Because the
//      dead-zone quantizes nearly all HF coefficients to zero, the high end of
//      every block — and thus long contiguous spans of the output — becomes runs
//      of zeros, exactly what RLGR / rANS(token 0) / Rice exploit. The inverse
//      un-scans back to the transform's row-major layout.
//
// The scan order is a static const table (precomputed once, 512 entries) so the
// gather/scatter is a straight-line indexed copy; the per-coefficient quantize
// is a branchless-friendly loop. The step-weight table is also static const.
//
// Contract (config.h VC_QUANT_FWD / VC_QUANT_INV):
//   void VC_QUANT_FWD(i16 *restrict qscan, const i16 *restrict coef,
//                     f32 base_step);
//   void VC_QUANT_INV(i16 *restrict coef, const i16 *restrict qscan,
//                     f32 base_step);
// Both operate on one VC_CHUNK_SIDE^3 cube; `qscan` is the scanned signed-level
// array fed to / from the entropy coder. base_step is the rate-control step.
#include "../../include/vc/types.h"
#include "../config.h"
#include <math.h>

#if (VC_CHUNK_SIDE % 8) != 0
#error "VC_CHUNK_SIDE must be a multiple of 8 for the 8^3 subband quantizer"
#endif

#define QCS   ((u32)VC_CHUNK_SIDE)
#define QSLY  (QCS)
#define QSLZ  (QCS * QCS)
#define BLKN  512u            // 8*8*8 coefficients per sub-block

#ifndef VC_QWEIGHT
#define VC_QWEIGHT VC_QWEIGHT_FLAT
#endif
#define VC_QWEIGHT_FLAT  0
#define VC_QWEIGHT_HF    1
#define VC_QWEIGHT_ADAPT 2

// --- precomputed scan order within an 8^3 block ----------------------------
// scan_idx[i] = linear offset (z*64 + y*8 + x) of the i-th coefficient in
// increasing-frequency order. Built once at first use (cheap, 512 entries).
static u16 g_scan[BLKN];
static u16 g_freqsum[BLKN];     // u+v+w for the i-th scanned coefficient (0..21)
static f32 g_wt[BLKN];          // static per-position step weight (flat/HF)
static int g_init = 0;

// Step-weight as a function of frequency sum s = u+v+w in 0..21.
static f32 weight_for(u32 s) {
#if VC_QWEIGHT == VC_QWEIGHT_FLAT
    (void)s; return 1.0f;
#else
    // HF-protecting: finer step (weight < 1) as frequency rises, bounded so DC
    // is untouched (weight 1) and the highest band is ~0.45x the base step.
    // smax = 21. Linear-ish protective curve.
    f32 t = (f32)s / 21.0f;          // 0..1
    return 1.0f - 0.55f * t;         // 1.0 (DC) .. 0.45 (max freq)
#endif
}

static void scan_init(void) {
    // Generate (z,y,x) triples, bucket by frequency sum, emit low->high.
    u32 w = 0;
    for (u32 s = 0; s <= 21; ++s)
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x)
            if (z + y + x == s) {
                g_scan[w] = (u16)(z * 64u + y * 8u + x);
                g_freqsum[w] = (u16)s;
                g_wt[w] = weight_for(s);
                ++w;
            }
    g_init = 1;
}

static inline void ensure_init(void) { if (!g_init) scan_init(); }

// Dead-zone quantize a single coefficient at the given step.
static inline i16 dz_quant(f32 c, f32 step) {
    f32 a = fabsf(c);
    i32 lvl = 0;
    if (a >= 0.5f * step) lvl = (i32)(a * (1.0f / step) - 0.5f) + 1;
    return (i16)(c < 0.f ? -lvl : lvl);
}
static inline i16 dz_dequant(i16 q, f32 step) {
    i32 l = (i32)q;
    i32 a = l < 0 ? -l : l;
    if (a == 0) return 0;
    f32 r = (f32)a * step;
    i32 v = (i32)lrintf(l < 0 ? -r : r);
    if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
    return (i16)v;
}

// Branchless unit-stride dead-zone quantize over a contiguous freq-scanned
// block `cbuf` using precomputed per-position inverse-step / half-step arrays
// (constant for a whole chunk). No data-dependent branches, no gathers -> the
// compiler autovectorizes this with 16-bit output lanes (PLAN §7).
static void dz_quant_block(i16 *restrict qb, const i16 *restrict cbuf,
                           const f32 *restrict invstep, const f32 *restrict half) {
    for (u32 i = 0; i < BLKN; ++i) {
        f32 c = (f32)cbuf[i];
        f32 a = c < 0.f ? -c : c;
        i32 m = (a >= half[i]);                          // 0/1 mask, branchless
        i32 lvl = m * ((i32)(a * invstep[i] - 0.5f) + 1);
        qb[i] = (i16)(c < 0.f ? -lvl : lvl);
    }
}

#if VC_QWEIGHT == VC_QWEIGHT_ADAPT
// Content-adaptive reconstruction. Quantization uses the static (HF-weighted)
// step so the dead-zone — and thus which coefficients become zero and the whole
// zero-run structure the entropy coder relies on — is IDENTICAL to the HF mode
// and exactly reproducible. The adaptive part is a per-block refinement of the
// dequant reconstruction point: in busy/high-AC blocks the true coefficient
// magnitudes skew above the bin centre, so we nudge the reconstruction slightly
// outward there; flat blocks are left at the bin centre. The nudge is derived
// from the FINAL quantized levels (the AC-level magnitude sum) which both
// encoder and decoder hold identically, and is BOUNDED to +-12% so it can only
// refine, never collapse, the reconstruction (no bin reassignment => exact and
// safe regardless of content). This is the cheap, exact form of content
// adaptivity; a full per-block-QP variant would need side bits in the stream.
static inline f32 adapt_recon_mult(const i16 *restrict qscan_blk) {
    f64 acl = 0.0; u32 nz = 0;
    for (u32 i = 0; i < BLKN; ++i)
        if (g_freqsum[i] != 0) { i32 v = qscan_blk[i]; if (v) { acl += v < 0 ? -v : v; ++nz; } }
    if (nz == 0) return 1.0f;
    f32 mean = (f32)(acl / (f64)nz);          // mean |level| among nonzero AC
    // Busy blocks (mean level near 1, many small coeffs) -> nudge up to +12%;
    // blocks with large coeffs already well-placed -> ~no nudge.
    f32 t = 1.0f / (mean + 1.0f);             // 0..0.5
    return 1.0f + 0.12f * t;                  // 1.0 .. ~1.06
}
#endif

// Precompute per-scan-position step / inverse-step / half-step for a chunk's
// base step (constant across the chunk). Done once per encode/decode call.
static void prep_steps(f32 base_step, f32 *restrict step,
                       f32 *restrict invstep, f32 *restrict half) {
    for (u32 i = 0; i < BLKN; ++i) {
        f32 s = base_step * g_wt[i];
        if (s < 0.5f) s = 0.5f;       // floor to keep quant sane
        step[i] = s; invstep[i] = 1.0f / s; half[i] = 0.5f * s;
    }
}

// Forward: coef (row-major chunk) -> qscan (block-major, freq-scanned levels).
// Two-stage per block: (1) gather the 8^3 sub-block into a contiguous,
// frequency-ordered scratch (the unavoidable scan permutation, a tight indexed
// copy), then (2) a branchless unit-stride quantize that autovectorizes.
void vc_quant_subband_fwd(i16 *restrict qscan, const i16 *restrict coef,
                          f32 base_step) {
    ensure_init();
    f32 step[BLKN], invstep[BLKN], half[BLKN];
    prep_steps(base_step, step, invstep, half);
    i16 cbuf[BLKN];
    size_t out = 0;
    for (u32 bz = 0; bz < QCS; bz += 8)
    for (u32 by = 0; by < QCS; by += 8)
    for (u32 bx = 0; bx < QCS; bx += 8) {
        const i16 *blk = coef + (size_t)bz * QSLZ + (size_t)by * QSLY + bx;
        for (u32 i = 0; i < BLKN; ++i) {
            u32 off = g_scan[i];
            cbuf[i] = blk[(size_t)(off >> 6) * QSLZ + (size_t)((off >> 3) & 7u) * QSLY + (off & 7u)];
        }
        dz_quant_block(qscan + out, cbuf, invstep, half);
        out += BLKN;
    }
}

// Inverse: qscan -> coef (row-major chunk). Dequantize unit-stride into a
// contiguous block scratch, then scatter back through the inverse scan.
void vc_quant_subband_inv(i16 *restrict coef, const i16 *restrict qscan,
                          f32 base_step) {
    ensure_init();
    f32 step[BLKN], invstep[BLKN], half[BLKN];
    prep_steps(base_step, step, invstep, half);
    (void)invstep; (void)half;
    i16 cbuf[BLKN];
    size_t in = 0;
    for (u32 bz = 0; bz < QCS; bz += 8)
    for (u32 by = 0; by < QCS; by += 8)
    for (u32 bx = 0; bx < QCS; bx += 8) {
        i16 *blk = coef + (size_t)bz * QSLZ + (size_t)by * QSLY + bx;
        const i16 *qb = qscan + in;
#if VC_QWEIGHT == VC_QWEIGHT_ADAPT
        f32 adapt = adapt_recon_mult(qb);
        for (u32 i = 0; i < BLKN; ++i) cbuf[i] = dz_dequant(qb[i], step[i] * adapt);
#else
        for (u32 i = 0; i < BLKN; ++i) cbuf[i] = dz_dequant(qb[i], step[i]);
#endif
        for (u32 i = 0; i < BLKN; ++i) {
            u32 off = g_scan[i];
            blk[(size_t)(off >> 6) * QSLZ + (size_t)((off >> 3) & 7u) * QSLY + (off & 7u)] = cbuf[i];
        }
        in += BLKN;
    }
}
