// Per-coefficient-position quantization MATRIX for the 16^3 integer DCT atom
// (PLAN §2 "Quantizer" + "Distortion metric" rows). This is the proper JPEG/
// JPEG-XL-style quant-matrix form of the distortion objective: instead of one
// scalar dead-zone step per block (or the 8^3 per-subband scalar in
// quant/subband.c, which is mismatched to the WON DCT-16 transform), every one
// of the 4096 coefficients (u,v,w) in a 16^3 block has its OWN step, taken from
// a 16^3 weight matrix multiplied by the rate-control base step.
//
// THREE distortion/quant objectives, selected at runtime by vc_qm_mode:
//   QM_FLAT  (a) — flat matrix (weight 1 everywhere): pure-MSE quant, the
//                  baseline the rate-control experiment showed is perceptually
//                  flat. Minimizing MSE provably starves HF (= blur).
//   QM_HF    (b) — HF-PROTECTING matrix: finer step (weight<1) as the frequency
//                  index u+v+w rises, so edge/ink/fiber high-frequency
//                  coefficients survive aggressive quant. (Scroll ink IS high
//                  frequency, so we protect — the OPPOSITE of the HVS-tuned JPEG
//                  luminance table that coarsens HF.) This silently SWAPS the
//                  objective away from MSE toward edge preservation; that is the
//                  point of the experiment.
//   QM_ADAPT (b') — content-adaptive: the HF matrix, additionally scaled per
//                  16^3 block by local EDGE ENERGY (source AC variance). Air/
//                  flat blocks get a COARSER step (crush HF -> ratio); busy ink/
//                  fiber blocks get a FINER step (preserve HF -> quality). This
//                  is exact + side-info-free: the encoder writes one extra byte
//                  per block (a quantized step-scale index) the decoder reads, so
//                  the dead-zone is reproduced identically.
//
// Objective (c), true perceptual-in-loop, is NOT a matrix — it searches the
// scalar step per block to minimize a perceptual metric on the DECODED block; it
// lives in the bench (it needs the inverse transform per candidate). This file
// gives it the matrix primitives it reuses.
//
// Self-contained: depends only on types.h. No codec.c / config.h coupling, so
// the bench can #include it directly (like the transform bake-off files). Pure
// C23, libc/libm; the per-coefficient quant loop is unit-stride + branchless ->
// autovectorizable (PLAN §7).
#include "../../include/vc/types.h"
#include <math.h>

#ifndef VC_QM16_H_INLINE
#define VC_QM16_H_INLINE

#define QM_B    16u
#define QM_BLKN 4096u            // 16*16*16 coefficients per atom

typedef enum {
    QM_FLAT  = 0,   // (a) pure-MSE flat quant
    QM_HF    = 1,   // (b) HF-protecting fixed matrix
    QM_ADAPT = 2,   // (b') HF matrix + per-block content-adaptive step scale
} vc_qm_mode;

// HF-protecting weight as a function of the 3D frequency index sum s=u+v+w in
// 0..45 (max 15+15+15). weight 1 at DC, falling smoothly to ~0.4 at the highest
// band: finer step => more bits => HF/edges preserved. Tuned (sweep in the bench)
// so the ratio cost vs flat is small while edge metrics lift.
// HF slope is a tunable so the bench can SWEEP the protection strength without
// recompiling (0 == flat; 0.60 default; up to ~0.9). Encoder+decoder must use
// the same value (it is a build/run constant, not per-block side info).
#ifndef QM_HF_SLOPE_DEFAULT
#define QM_HF_SLOPE_DEFAULT 0.60f
#endif
static f32 g_qm_hf_slope = QM_HF_SLOPE_DEFAULT;
static inline void qm_set_hf_slope(f32 s) { g_qm_hf_slope = s; }

static inline f32 qm_hf_weight(u32 s) {
    f32 t = (f32)s / 45.0f;                  // 0..1
    return 1.0f - g_qm_hf_slope * t;         // 1.00 (DC) .. (1-slope) (max freq)
}

// Build the 16^3 per-coefficient step matrix for a given mode + base step into
// `step[4096]` (row-major z*256+y*16+x, matching the DCT-16 coef layout). Steps
// are floored at 0.5 to keep the dead-zone sane. For QM_FLAT the matrix is
// uniform; otherwise it is the HF-protecting curve. `blk_scale` (1.0 unless
// content-adaptive supplies it) multiplies the whole block.
static inline void qm_build_step(f32 *restrict step, vc_qm_mode mode,
                                 f32 base_step, f32 blk_scale) {
    for (u32 z = 0; z < QM_B; ++z)
    for (u32 y = 0; y < QM_B; ++y)
    for (u32 x = 0; x < QM_B; ++x) {
        f32 w = (mode == QM_FLAT) ? 1.0f : qm_hf_weight(z + y + x);
        f32 s = base_step * w * blk_scale;
        if (s < 0.5f) s = 0.5f;
        step[(size_t)z * 256u + (size_t)y * 16u + x] = s;
    }
}

// Dead-zone quantize one 16^3 coefficient block in place of layout into qb,
// unit-stride, branchless (autovectorizes with 16-bit output lanes).
static inline void qm_quant_block(i16 *restrict qb, const i16 *restrict coef,
                                  const f32 *restrict step) {
    for (u32 i = 0; i < QM_BLKN; ++i) {
        f32 c = (f32)coef[i];
        f32 a = c < 0.f ? -c : c;
        f32 inv = 1.0f / step[i];
        i32 m = (a >= 0.5f * step[i]);
        i32 lvl = m * ((i32)(a * inv - 0.5f) + 1);
        qb[i] = (i16)(c < 0.f ? -lvl : lvl);
    }
}

static inline void qm_dequant_block(i16 *restrict coef, const i16 *restrict qb,
                                    const f32 *restrict step) {
    for (u32 i = 0; i < QM_BLKN; ++i) {
        i32 l = (i32)qb[i];
        i32 al = l < 0 ? -l : l;
        f32 r = (f32)al * step[i];
        i32 v = (i32)lrintf(l < 0 ? -r : r);
        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
        coef[i] = (i16)v;
    }
}

// Content-adaptive per-block step scale from the SOURCE 16^3 voxel block's AC
// energy (variance). High-variance (edge/ink/fiber) blocks -> scale<1 (finer);
// flat/air blocks -> scale>1 (coarser). Quantized to a small integer index (5
// bits) so encoder+decoder agree exactly via one side byte per block. Returns
// both the float scale and the index written to the stream.
//   index 0..NS-1 maps to scale in [QM_ADAPT_MIN, QM_ADAPT_MAX] log-spaced.
#define QM_ADAPT_NS   16
// Range tuned in the bake-off: an AGGRESSIVE range (e.g. 0.55..1.80) over-
// crushes flat regions and worsens GLOBAL perceptual metrics (the global metric
// does not reward trading flat-region for busy-region quality). A gentle range
// is strictly closer to the fixed HF matrix, which wins — see HFDIST_RESULTS.md.
#ifndef QM_ADAPT_MIN
#define QM_ADAPT_MIN  0.80f
#endif
#ifndef QM_ADAPT_MAX
#define QM_ADAPT_MAX  1.30f
#endif

static inline f32 qm_adapt_scale_from_index(u32 idx) {
    if (idx >= QM_ADAPT_NS) idx = QM_ADAPT_NS - 1;
    f32 t = (f32)idx / (f32)(QM_ADAPT_NS - 1);            // 0..1
    f32 lo = logf(QM_ADAPT_MIN), hi = logf(QM_ADAPT_MAX);
    return expf(lo + (hi - lo) * t);
}

// Compute the adaptive index for a source 16^3 u8 block from its std-dev.
// Air (std small) -> high index (coarse); busy (std large) -> low index (fine).
static inline u32 qm_adapt_index(const u8 *restrict vox, u32 sly, u32 slz,
                                 u32 bz, u32 by, u32 bx) {
    f64 s = 0.0, sq = 0.0;
    for (u32 z = 0; z < QM_B; ++z)
    for (u32 y = 0; y < QM_B; ++y) {
        const u8 *p = vox + (size_t)(bz + z) * slz + (size_t)(by + y) * sly + bx;
        for (u32 x = 0; x < QM_B; ++x) { f64 v = p[x]; s += v; sq += v * v; }
    }
    f64 n = (f64)QM_BLKN;
    f64 var = sq / n - (s / n) * (s / n);
    if (var < 0.0) var = 0.0;
    f64 sd = sqrt(var);
    // Map std-dev (0..~70 typical for ink/fiber) to an index. Low sd -> coarse
    // (high index); high sd -> fine (low index). Smooth ramp around sd~18.
    f64 t = sd / 36.0; if (t > 1.0) t = 1.0;              // 0 (flat) .. 1 (busy)
    f64 fidx = (1.0 - t) * (QM_ADAPT_NS - 1);             // busy -> 0, flat -> NS-1
    i32 idx = (i32)(fidx + 0.5);
    if (idx < 0) idx = 0;
    if (idx >= QM_ADAPT_NS) idx = QM_ADAPT_NS - 1;
    return (u32)idx;
}

#endif // VC_QM16_H_INLINE
