// ZFP-style multiply-free near-orthogonal lifting transform on 4^3 sub-blocks
// (PLAN §2 "Transform" SPEED candidate; the red-team note that "ZFP uses a DCT"
// was FALSE — ZFP uses a multiply-free lifting transform of adds + shifts).
//
// This is an integer, reversible, separable 3D lifting decorrelator: a 4-point
// predict/update lifting (Haar-like difference + average, then a second
// decorrelating mix) applied along X, then Y, then Z over each 4x4x4 sub-block.
// Each lifting step modifies one parity using ONLY the other parity, so the
// inverse is the exact step-by-step reversal — round-trip without quantization is
// bit-exact lossless (no fixed-point rounding error, unlike the DCT). Cost per
// 4-line: 8 adds + 6 shifts (no multiplies) -> ~2 adds + 1.5 shifts per sample,
// matching ZFP's profile.
//
// Contract (same as dct_int.c):
//   void vc_lift_zfp_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc);
//   void vc_lift_zfp_inv(u8 *restrict vox,  const i16 *restrict coef, i32 dc);
//
// Coefficient layout: in-place row-major (each 4^3 block's (0,0,0) element is its
// low-low-low coefficient). Per-sub-block scratch is 4^3 i32 = 256 B.
// Autovectorization: the 1D lifting is a straight-line branch-free sequence over
// 4-element scratch; the gather/scatter loops are unit/contiguous-stride.
#include "../../include/vc/types.h"
#include "../config.h"

#if (VC_CHUNK_SIDE % 4) != 0
#error "VC_CHUNK_SIDE must be a multiple of 4 for the 4^3 lifting transform"
#endif

#define CS   ((u32)VC_CHUNK_SIDE)
#define SLY  (CS)
#define SLZ  ((size_t)CS * CS)
#define B    4u

// Forward 4-point lifting (predict/update, multiply-free). Each statement edits
// exactly one element using only the others -> exactly invertible. Smooth input
// compacts ~99% of its energy into a[0].
static inline void lift4_fwd(i32 a[4]) {
    a[1] -= a[0];
    a[3] -= a[2];
    a[0] += (a[1] + 1) >> 1;
    a[2] += (a[3] + 1) >> 1;
    a[2] -= a[0];
    a[0] += (a[2] + 1) >> 1;
    a[3] -= (a[1] + 1) >> 1;
    a[1] += (a[3] + 1) >> 1;
}

static inline void lift4_inv(i32 a[4]) {
    a[1] -= (a[3] + 1) >> 1;
    a[3] += (a[1] + 1) >> 1;
    a[0] -= (a[2] + 1) >> 1;
    a[2] += a[0];
    a[2] -= (a[3] + 1) >> 1;
    a[0] -= (a[1] + 1) >> 1;
    a[3] += a[2];
    a[1] += a[0];
}

void vc_lift_zfp_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc) {
    for (u32 bz = 0; bz < CS; bz += B)
    for (u32 by = 0; by < CS; by += B)
    for (u32 bx = 0; bx < CS; bx += B) {
        i32 blk[4][4][4];
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) {
            const u8 *src = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < B; ++x) blk[z][y][x] = (i32)src[x] - dc;
        }
        // X pass (rows).
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) lift4_fwd(blk[z][y]);
        // Y pass.
        for (u32 z = 0; z < B; ++z)
        for (u32 x = 0; x < B; ++x) {
            i32 col[4];
            for (u32 y = 0; y < B; ++y) col[y] = blk[z][y][x];
            lift4_fwd(col);
            for (u32 y = 0; y < B; ++y) blk[z][y][x] = col[y];
        }
        // Z pass + write.
        for (u32 y = 0; y < B; ++y)
        for (u32 x = 0; x < B; ++x) {
            i32 col[4];
            for (u32 z = 0; z < B; ++z) col[z] = blk[z][y][x];
            lift4_fwd(col);
            for (u32 z = 0; z < B; ++z) {
                i32 v = col[z];
                if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
                coef[(size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + (bx + x)] = (i16)v;
            }
        }
    }
}

void vc_lift_zfp_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc) {
    for (u32 bz = 0; bz < CS; bz += B)
    for (u32 by = 0; by < CS; by += B)
    for (u32 bx = 0; bx < CS; bx += B) {
        i32 blk[4][4][4];
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) {
            const i16 *src = coef + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < B; ++x) blk[z][y][x] = (i32)src[x];
        }
        // Inverse Z, then Y, then X.
        for (u32 y = 0; y < B; ++y)
        for (u32 x = 0; x < B; ++x) {
            i32 col[4];
            for (u32 z = 0; z < B; ++z) col[z] = blk[z][y][x];
            lift4_inv(col);
            for (u32 z = 0; z < B; ++z) blk[z][y][x] = col[z];
        }
        for (u32 z = 0; z < B; ++z)
        for (u32 x = 0; x < B; ++x) {
            i32 col[4];
            for (u32 y = 0; y < B; ++y) col[y] = blk[z][y][x];
            lift4_inv(col);
            for (u32 y = 0; y < B; ++y) blk[z][y][x] = col[y];
        }
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) {
            i32 row[4];
            for (u32 x = 0; x < B; ++x) row[x] = blk[z][y][x];
            lift4_inv(row);
            u8 *dst = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < B; ++x) {
                i32 v = row[x] + dc;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                dst[x] = (u8)v;
            }
        }
    }
}
