// Integer separable DCT-II in 16^3 sub-blocks (PLAN §2 "Transform", Phase-1
// bake-off vs the Phase-0 8^3 DCT). Larger blocks = better energy compaction at
// high ratios, but coarser blocking faces (16-voxel grid instead of 8).
//
// Same contract as dct_int.c:
//   void vc_dct_int16_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc);
//   void vc_dct_int16_inv(u8 *restrict vox,  const i16 *restrict coef, i32 dc);
// Operates on one VC_CHUNK_SIDE^3 cube, tiled into 16x16x16 sub-blocks. Output
// coefficients stay in the SAME row-major chunk layout (each sub-block's (0,0,0)
// element is its DC). The forward subtracts the per-chunk DC bias first.
//
// The transform is an orthonormal scaled DCT baked as a Q14 integer matrix; the
// inverse is the transpose. Round-trip without quant is near-lossless (only
// fixed-point rounding). Hot kernels are straight-line matrix-vector products
// with compile-time trip count 16 -> autovectorizable. Per-sub-block scratch is
// 16^3 i32 = 16KiB on stack: acceptable (NOT chunk-sized; PLAN §7 bans only
// chunk/tile-sized stack buffers — a 16^3 block is the fixed transform unit).
#include "../../include/vc/types.h"
#include "../config.h"

#if (VC_CHUNK_SIDE % 16) != 0
#error "VC_CHUNK_SIDE must be a multiple of 16 for the 16^3 integer DCT"
#endif

#define CS   ((u32)VC_CHUNK_SIDE)
#define SLY  (CS)            // stride between rows (Y)
#define SLZ  (CS * CS)       // stride between slices (Z)
#define B    16u            // sub-block edge
#define Q    14u

// 16-point orthonormal DCT-II, scaled cosines in Q14 fixed point. Inverse uses
// the same matrix transposed (the basis is orthonormal => inverse == transpose).
static const i32 CMAT16[16][16] = {
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

static inline void dct16_fwd(const i32 in[16], i32 out[16]) {
    const i32 rnd = (i32)1 << (Q - 1);
    for (u32 k = 0; k < 16; ++k) {
        i32 acc = 0;
        for (u32 n = 0; n < 16; ++n) acc += CMAT16[k][n] * in[n];
        out[k] = (acc + rnd) >> Q;
    }
}

static inline void idct16(const i32 in[16], i32 out[16]) {
    const i32 rnd = (i32)1 << (Q - 1);
    for (u32 n = 0; n < 16; ++n) {
        i32 acc = 0;
        for (u32 k = 0; k < 16; ++k) acc += CMAT16[k][n] * in[k];
        out[n] = (acc + rnd) >> Q;
    }
}

void vc_dct_int16_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc) {
    for (u32 bz = 0; bz < CS; bz += B)
    for (u32 by = 0; by < CS; by += B)
    for (u32 bx = 0; bx < CS; bx += B) {
        i32 blk[16][16][16];
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) {
            const u8 *src = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < B; ++x) blk[z][y][x] = (i32)src[x] - dc;
        }
        // Pass X (rows, unit stride).
        i32 tx[16][16][16];
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) dct16_fwd(blk[z][y], tx[z][y]);
        // Pass Y.
        i32 ty[16][16][16];
        for (u32 z = 0; z < B; ++z)
        for (u32 x = 0; x < B; ++x) {
            i32 col[16], outc[16];
            for (u32 y = 0; y < B; ++y) col[y] = tx[z][y][x];
            dct16_fwd(col, outc);
            for (u32 y = 0; y < B; ++y) ty[z][y][x] = outc[y];
        }
        // Pass Z, write out clamped to i16.
        for (u32 y = 0; y < B; ++y)
        for (u32 x = 0; x < B; ++x) {
            i32 col[16], outc[16];
            for (u32 z = 0; z < B; ++z) col[z] = ty[z][y][x];
            dct16_fwd(col, outc);
            for (u32 z = 0; z < B; ++z) {
                i32 v = outc[z];
                if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
                coef[(size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + (bx + x)] = (i16)v;
            }
        }
    }
}

void vc_dct_int16_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc) {
    for (u32 bz = 0; bz < CS; bz += B)
    for (u32 by = 0; by < CS; by += B)
    for (u32 bx = 0; bx < CS; bx += B) {
        i32 blk[16][16][16];
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) {
            const i16 *src = coef + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < B; ++x) blk[z][y][x] = (i32)src[x];
        }
        // Inverse Z, then Y, then X.
        i32 tz[16][16][16];
        for (u32 y = 0; y < B; ++y)
        for (u32 x = 0; x < B; ++x) {
            i32 col[16], outc[16];
            for (u32 z = 0; z < B; ++z) col[z] = blk[z][y][x];
            idct16(col, outc);
            for (u32 z = 0; z < B; ++z) tz[z][y][x] = outc[z];
        }
        i32 ty[16][16][16];
        for (u32 z = 0; z < B; ++z)
        for (u32 x = 0; x < B; ++x) {
            i32 col[16], outc[16];
            for (u32 y = 0; y < B; ++y) col[y] = tz[z][y][x];
            idct16(col, outc);
            for (u32 y = 0; y < B; ++y) ty[z][y][x] = outc[y];
        }
        for (u32 z = 0; z < B; ++z)
        for (u32 y = 0; y < B; ++y) {
            i32 row[16], outc[16];
            for (u32 x = 0; x < B; ++x) row[x] = ty[z][y][x];
            idct16(row, outc);
            u8 *dst = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < B; ++x) {
                i32 v = outc[x] + dc;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                dst[x] = (u8)v;
            }
        }
    }
}
