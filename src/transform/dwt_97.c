// Separable 3D CDF 9/7 lifting wavelet over the WHOLE chunk (PLAN §2 "Transform"
// RD-leader candidate). Pure C23 port of c4d/include/c4d/dwt.hpp +
// dwt_tables.hpp: the standard four-lift Daubechies-Sweldens scheme with K
// scaling, whole-sample symmetric (mirror) boundary extension, applied across
// the entire VC_CHUNK_SIDE^3 cube, recursing DWT_LEVELS levels into the LLL
// octant (Mallat pyramid layout).
//
// Contract (same signature as dct_int.c):
//   void vc_dwt97_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc);
//   void vc_dwt97_inv(u8 *restrict vox,  const i16 *restrict coef, i32 dc);
//
// The transform is intrinsically float; we keep a chunk-sized f32 scratch on the
// HEAP (never the stack — a 128^3 f32 buffer is 8 MiB; PLAN §7), run the lifting,
// then pack the float subband coefficients into the i16 `coef` the codec stores.
//
// Per-subband synthesis-L2 weighting: 9/7 lifting is only *approximately*
// orthonormal, so each subband's synthesis basis has a different L2 gain. To make
// the codec's single global dead-zone step near-MSE-optimal across subbands, the
// forward MULTIPLIES each coefficient by its subband synthesis-L2 gain (so a
// subband that contributes more reconstruction energy carries proportionally
// larger coefficients and is quantized relatively finer); the inverse divides it
// back out before the inverse lifting. Net round-trip (no quant) is near-lossless.
//
// SIMD note: c4d used std::experimental::simd + an AVX-512 transpose. This port
// is straight-line C over contiguous scratch lines (the lift passes are
// branch-free interior loops over a gathered contiguous line) so the COMPILER
// autovectorizes the interior; no hand intrinsics (PLAN §7).
#include "../../include/vc/types.h"
#include "../config.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CS   ((u32)VC_CHUNK_SIDE)
#define SLY  (CS)
#define SLZ  ((size_t)CS * CS)
#define CVOX ((size_t)CS * CS * CS)

#ifndef VC_DWT_LEVELS
#define VC_DWT_LEVELS 3
#endif
#if (VC_CHUNK_SIDE >> VC_DWT_LEVELS) < 1
#error "VC_CHUNK_SIDE too small for VC_DWT_LEVELS"
#endif

// CDF 9/7 lifting constants (JPEG2000 irreversible filter).
#define ALPHA (-1.586134342059924)
#define BETA  (-0.052980118572961)
#define GAMMA ( 0.882911075530934)
#define DELTA ( 0.443506852043971)
#define KAPPA ( 1.230174104914001)

// 1D synthesis L2 gain per orientation bit: low synthesis gain = KAPPA, high =
// 1/KAPPA (inverse of the analysis scaling 1/KAPPA on even, KAPPA on odd).
static inline f64 axis_synth_gain(int high) { return high ? (1.0 / KAPPA) : KAPPA; }

// Per-subband synthesis L2 weight: product over 3 axes of this level's
// orientation gain, times the accumulated LLL synthesis gain of every coarser
// level above it. orient bit0=x, bit1=y, bit2=z (set => high on that axis).
static inline f32 subband_synth_l2(u32 level, u32 orient) {
    f64 g = axis_synth_gain((int)(orient & 1u))
          * axis_synth_gain((int)(orient & 2u))
          * axis_synth_gain((int)(orient & 4u));
    const f64 lll = axis_synth_gain(0) * axis_synth_gain(0) * axis_synth_gain(0);
    for (u32 l = 0; l < level; ++l) g *= lll;
    return (f32)g;
}

// --- One forward lifting step over a CONTIGUOUS line v of length n (n even) ---
// `c * (left + right)` accumulated into the target parity; interior branch-free,
// boundaries use mirror (x[-1]=x[1], x[n]=x[n-2], whole-sample symmetric).
static inline void lift_odd(f32 *restrict v, u32 n, f32 c) {
    for (u32 i = 1; i + 1 < n; i += 2) v[i] += c * (v[i - 1] + v[i + 1]);
    u32 last = n - 1;                                   // odd (n even)
    v[last] += c * (v[last - 1] + v[last - 1]);         // mirror
}
static inline void lift_even(f32 *restrict v, u32 n, f32 c) {
    v[0] += c * (v[1] + v[1]);                          // mirror
    for (u32 i = 2; i < n; i += 2) v[i] += c * (v[i - 1] + v[i + 1]);
}

// Forward 1D 9/7 on a strided line: gather contiguous, lift, scatter even (low)
// to first half and odd (high) to second half (Mallat deinterleave folded in).
static inline void fwd_1d(f32 *restrict base, u32 n, size_t stride, f32 *restrict v) {
    for (u32 i = 0; i < n; ++i) v[i] = base[i * stride];
    lift_odd (v, n, (f32)ALPHA);
    lift_even(v, n, (f32)BETA);
    lift_odd (v, n, (f32)GAMMA);
    lift_even(v, n, (f32)DELTA);
    const f32 lo = (f32)(1.0 / KAPPA), hi = (f32)KAPPA;
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        base[i * stride]       = v[2 * i]     * lo;
        base[(h + i) * stride] = v[2 * i + 1] * hi;
    }
}

static inline void inv_1d(f32 *restrict base, u32 n, size_t stride, f32 *restrict v) {
    const f32 lo = (f32)KAPPA, hi = (f32)(1.0 / KAPPA);
    const u32 h = n / 2;
    for (u32 i = 0; i < h; ++i) {
        v[2 * i]     = base[i * stride]       * lo;
        v[2 * i + 1] = base[(h + i) * stride] * hi;
    }
    lift_even(v, n, -(f32)DELTA);
    lift_odd (v, n, -(f32)GAMMA);
    lift_even(v, n, -(f32)BETA);
    lift_odd (v, n, -(f32)ALPHA);
    for (u32 i = 0; i < n; ++i) base[i * stride] = v[i];
}

// One separable forward DWT step over the leading s^3 sub-cube; transforms all
// three axes, packing into Mallat octant layout. `line` is a scratch line >= CS.
static void fwd_step(f32 *restrict vol, u32 s, f32 *restrict line) {
    for (u32 z = 0; z < s; ++z)                         // X (stride 1)
    for (u32 y = 0; y < s; ++y)
        fwd_1d(vol + (size_t)z * SLZ + (size_t)y * SLY, s, 1, line);
    for (u32 z = 0; z < s; ++z)                         // Y (stride SLY)
    for (u32 x = 0; x < s; ++x)
        fwd_1d(vol + (size_t)z * SLZ + x, s, SLY, line);
    for (u32 y = 0; y < s; ++y)                         // Z (stride SLZ)
    for (u32 x = 0; x < s; ++x)
        fwd_1d(vol + (size_t)y * SLY + x, s, SLZ, line);
}

static void inv_step(f32 *restrict vol, u32 s, f32 *restrict line) {
    for (u32 y = 0; y < s; ++y)                         // Z
    for (u32 x = 0; x < s; ++x)
        inv_1d(vol + (size_t)y * SLY + x, s, SLZ, line);
    for (u32 z = 0; z < s; ++z)                         // Y
    for (u32 x = 0; x < s; ++x)
        inv_1d(vol + (size_t)z * SLZ + x, s, SLY, line);
    for (u32 z = 0; z < s; ++z)                         // X
    for (u32 y = 0; y < s; ++y)
        inv_1d(vol + (size_t)z * SLZ + (size_t)y * SLY, s, 1, line);
}

// --- Subband weight field: for each voxel, which (level, orient) subband it is
// in after the multi-level transform, mapped to its synth-L2 gain. Built once
// per call (cheap vs the transform). A voxel at (z,y,x): walk levels coarse->fine
// (largest sub-cube down) — equivalently, find the finest level L at which the
// coordinate's halving lands in a high band on some axis.
static f32 subband_weight_at(u32 z, u32 y, u32 x) {
    // Determine, per axis, the level index at which this coordinate first falls
    // in the "high" half, scanning from finest (level 0 = outer half) inward.
    // After `levels` steps the LLL octant is [0,s>>levels). A coordinate c with
    // s = CS: at level l the active size is CS>>l; its low half is [0,(CS>>l)/2).
    // The subband level of an axis = number of leading low-halves before its
    // first high-half (capped at levels-1 for the LLL approximation band).
    const u32 levels = VC_DWT_LEVELS;
    u32 lvl[3]; int hi[3];
    u32 coord[3] = { x, y, z };
    for (int a = 0; a < 3; ++a) {
        u32 c = coord[a]; u32 size = CS; u32 l = 0; int high = 0;
        for (; l < levels; ++l) {
            u32 half = size / 2;
            if (c >= half) { high = 1; break; }   // high band at this level
            size = half;                          // descend into low half
        }
        lvl[a] = (l >= levels) ? (levels - 1) : l; // pinned to coarsest if all-low
        hi[a]  = high;
    }
    // The subband level is the COARSEST (max) level at which any axis went high;
    // for the all-low LLL approximation, level = levels-1, orient = 0.
    int any_high = hi[0] | hi[1] | hi[2];
    u32 level; u32 orient;
    if (!any_high) { level = levels - 1; orient = 0; }
    else {
        level = 0;
        for (int a = 0; a < 3; ++a) if (hi[a] && lvl[a] > level) level = lvl[a];
        orient = (u32)((hi[0] ? 1 : 0) | (hi[1] ? 2 : 0) | (hi[2] ? 4 : 0));
    }
    return subband_synth_l2(level, orient);
}

void vc_dwt97_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc) {
    f32 *vol  = (f32 *)malloc(CVOX * sizeof(f32));
    f32 *line = (f32 *)malloc((size_t)CS * sizeof(f32));
    if (!vol || !line) { free(vol); free(line); return; }
    const f32 dcf = (f32)dc;
    for (size_t i = 0; i < CVOX; ++i) vol[i] = (f32)vox[i] - dcf;

    u32 s = CS;
    for (u32 l = 0; l < VC_DWT_LEVELS; ++l) { fwd_step(vol, s, line); s /= 2; }

    // Pack: scale each coefficient by its subband synthesis-L2 gain, round to i16.
    for (u32 z = 0; z < CS; ++z)
    for (u32 y = 0; y < CS; ++y)
    for (u32 x = 0; x < CS; ++x) {
        size_t idx = (size_t)z * SLZ + (size_t)y * SLY + x;
        f32 w = subband_weight_at(z, y, x);
        i32 v = (i32)lrintf(vol[idx] * w);
        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
        coef[idx] = (i16)v;
    }
    free(vol); free(line);
}

void vc_dwt97_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc) {
    f32 *vol  = (f32 *)malloc(CVOX * sizeof(f32));
    f32 *line = (f32 *)malloc((size_t)CS * sizeof(f32));
    if (!vol || !line) { free(vol); free(line); return; }

    // Unpack: undo the per-subband synthesis-L2 scaling.
    for (u32 z = 0; z < CS; ++z)
    for (u32 y = 0; y < CS; ++y)
    for (u32 x = 0; x < CS; ++x) {
        size_t idx = (size_t)z * SLZ + (size_t)y * SLY + x;
        f32 w = subband_weight_at(z, y, x);
        vol[idx] = (f32)coef[idx] / w;
    }

    u32 sizes[16]; u32 s = CS;
    for (u32 l = 0; l < VC_DWT_LEVELS; ++l) { sizes[l] = s; s /= 2; }
    for (i32 l = (i32)VC_DWT_LEVELS - 1; l >= 0; --l) inv_step(vol, sizes[l], line);

    const f32 dcf = (f32)dc;
    for (size_t i = 0; i < CVOX; ++i) {
        i32 v = (i32)lrintf(vol[i] + dcf);
        v = v < 0 ? 0 : (v > 255 ? 255 : v);
        vox[i] = (u8)v;
    }
    free(vol); free(line);
}
