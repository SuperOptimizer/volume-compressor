// Float separable DCT-II in 8^3 sub-blocks (PLAN §2 optional quality reference
// for the integer 8^3 DCT). Same block geometry & coefficient layout as
// dct_int.c; the only difference is the arithmetic is f32 (orthonormal basis,
// inverse = transpose) so it isolates the integer-rounding quality loss.
//
// Contract:
//   void vc_dct_float8_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc);
//   void vc_dct_float8_inv(u8 *restrict vox,  const i16 *restrict coef, i32 dc);
//
// Coefficients are rounded to i16 on write (the codec stores i16); that rounding
// is the only lossy step here, so this is the "as good as 8^3 DCT can be" ref.
// Hot kernels are 8x8 matrix-vector products with -ffast-math fmas, fully
// autovectorizable. Per-sub-block scratch is small (8^3 f32 = 2KiB).
#include "../../include/vc/types.h"
#include "../config.h"
#include <math.h>

#if (VC_CHUNK_SIDE % 8) != 0
#error "VC_CHUNK_SIDE must be a multiple of 8 for the 8^3 float DCT"
#endif

#define CS   ((u32)VC_CHUNK_SIDE)
#define SLY  (CS)
#define SLZ  (CS * CS)

static const f32 CF[8][8] = {
  {0.353553391f,0.353553391f,0.353553391f,0.353553391f,0.353553391f,0.353553391f,0.353553391f,0.353553391f},
  {0.490392640f,0.415734806f,0.277785117f,0.097545161f,-0.097545161f,-0.277785117f,-0.415734806f,-0.490392640f},
  {0.461939766f,0.191341716f,-0.191341716f,-0.461939766f,-0.461939766f,-0.191341716f,0.191341716f,0.461939766f},
  {0.415734806f,-0.097545161f,-0.490392640f,-0.277785117f,0.277785117f,0.490392640f,0.097545161f,-0.415734806f},
  {0.353553391f,-0.353553391f,-0.353553391f,0.353553391f,0.353553391f,-0.353553391f,-0.353553391f,0.353553391f},
  {0.277785117f,-0.490392640f,0.097545161f,0.415734806f,-0.415734806f,-0.097545161f,0.490392640f,-0.277785117f},
  {0.191341716f,-0.461939766f,0.461939766f,-0.191341716f,-0.191341716f,0.461939766f,-0.461939766f,0.191341716f},
  {0.097545161f,-0.277785117f,0.415734806f,-0.490392640f,0.490392640f,-0.415734806f,0.277785117f,-0.097545161f},
};

static inline void dct8f_fwd(const f32 in[8], f32 out[8]) {
    for (u32 k = 0; k < 8; ++k) {
        f32 acc = 0.f;
        for (u32 n = 0; n < 8; ++n) acc += CF[k][n] * in[n];
        out[k] = acc;
    }
}
static inline void idct8f(const f32 in[8], f32 out[8]) {
    for (u32 n = 0; n < 8; ++n) {
        f32 acc = 0.f;
        for (u32 k = 0; k < 8; ++k) acc += CF[k][n] * in[k];
        out[n] = acc;
    }
}

void vc_dct_float8_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc) {
    const f32 dcf = (f32)dc;
    for (u32 bz = 0; bz < CS; bz += 8)
    for (u32 by = 0; by < CS; by += 8)
    for (u32 bx = 0; bx < CS; bx += 8) {
        f32 blk[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) {
            const u8 *src = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < 8; ++x) blk[z][y][x] = (f32)src[x] - dcf;
        }
        f32 tx[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) dct8f_fwd(blk[z][y], tx[z][y]);
        f32 ty[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 x = 0; x < 8; ++x) {
            f32 col[8], outc[8];
            for (u32 y = 0; y < 8; ++y) col[y] = tx[z][y][x];
            dct8f_fwd(col, outc);
            for (u32 y = 0; y < 8; ++y) ty[z][y][x] = outc[y];
        }
        for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x) {
            f32 col[8], outc[8];
            for (u32 z = 0; z < 8; ++z) col[z] = ty[z][y][x];
            dct8f_fwd(col, outc);
            for (u32 z = 0; z < 8; ++z) {
                i32 v = (i32)lrintf(outc[z]);
                if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
                coef[(size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + (bx + x)] = (i16)v;
            }
        }
    }
}

void vc_dct_float8_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc) {
    const f32 dcf = (f32)dc;
    for (u32 bz = 0; bz < CS; bz += 8)
    for (u32 by = 0; by < CS; by += 8)
    for (u32 bx = 0; bx < CS; bx += 8) {
        f32 blk[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) {
            const i16 *src = coef + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < 8; ++x) blk[z][y][x] = (f32)src[x];
        }
        f32 tz[8][8][8];
        for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x) {
            f32 col[8], outc[8];
            for (u32 z = 0; z < 8; ++z) col[z] = blk[z][y][x];
            idct8f(col, outc);
            for (u32 z = 0; z < 8; ++z) tz[z][y][x] = outc[z];
        }
        f32 ty[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 x = 0; x < 8; ++x) {
            f32 col[8], outc[8];
            for (u32 y = 0; y < 8; ++y) col[y] = tz[z][y][x];
            idct8f(col, outc);
            for (u32 y = 0; y < 8; ++y) ty[z][y][x] = outc[y];
        }
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) {
            f32 row[8], outc[8];
            for (u32 x = 0; x < 8; ++x) row[x] = ty[z][y][x];
            idct8f(row, outc);
            u8 *dst = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < 8; ++x) {
                i32 v = (i32)lrintf(outc[x] + dcf);
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                dst[x] = (u8)v;
            }
        }
    }
}
