// Integer separable DCT-II in 8^3 sub-blocks (PLAN §3 Phase-0 transform).
//
// The chunk (VC_CHUNK_SIDE^3, a multiple of 8) is tiled into 8x8x8 sub-blocks.
// Each sub-block gets a separable 3D DCT: an 8-point 1D integer DCT-II applied
// along X, then Y, then Z. The transform is integer fixed-point (i32 work,
// rounded back to i16) so it is fully autovectorizable and exactly invertible to
// within the quantizer's tolerance (it is the standard scaled DCT; round-trip
// without quantization is near-lossless, dominated by fixed-point rounding).
//
// Coefficient layout: the transformed cube stays in the SAME row-major chunk
// layout. Within each 8^3 sub-block the (0,0,0) element is that block's DC. The
// entropy coder scans the whole chunk linearly; the dead-zone makes HF runs of
// zeros that compress well. The forward subtracts a per-chunk DC bias first.
//
// Autovectorization: the hot kernels are straight-line loops over contiguous
// rows/columns with compile-time trip counts (8). The X pass is unit-stride; the
// Y/Z passes process 8 contiguous lanes at a time so they stay vectorizable.
#include "../../include/vc/types.h"
#include "../config.h"

#if (VC_CHUNK_SIDE % 8) != 0
#error "VC_CHUNK_SIDE must be a multiple of 8 for the 8^3 integer DCT"
#endif

#define CS   ((u32)VC_CHUNK_SIDE)
#define SLY  (CS)            // stride between rows (Y) within the chunk = CS
#define SLZ  (CS * CS)       // stride between slices (Z)

// 8-point integer DCT-II constants. Scaled cosines in Q13 fixed point
// (round(cos(k*pi/16) * 2^13 * sqrt(2/8)-style scaling)). We use the well-known
// integer butterfly form; the inverse uses the same constants transposed. The
// forward applies a >> SHIFT after each 1D pass; the total scale across 3 passes
// is absorbed so coefficients stay in i16 range for u8 input.
//
// Basis matrix C[k][n] = a(k) * cos((2n+1)k*pi/16), a(0)=1/sqrt(8), else
// 1/2. We bake it as Q14 integers and do straight matrix-vector products: 8x8,
// trivially unrolled and vectorizable, clearer and exactly invertible (C is
// orthonormal => inverse is the transpose).
#define Q 14
static const i32 CMAT[8][8] = {
  { 5793, 5793, 5793, 5793, 5793, 5793, 5793, 5793},
  { 8035, 6811, 4551, 1598,-1598,-4551,-6811,-8035},
  { 7568, 3135,-3135,-7568,-7568,-3135, 3135, 7568},
  { 6811,-1598,-8035,-4551, 4551, 8035, 1598,-6811},
  { 5793,-5793,-5793, 5793, 5793,-5793,-5793, 5793},
  { 4551,-8035, 1598, 6811,-6811,-1598, 8035,-4551},
  { 3135,-7568, 7568,-3135,-3135, 7568,-7568, 3135},
  { 1598,-4551, 6811,-8035, 8035,-6811, 4551,-1598},
};

// Forward 1D DCT of 8 inputs `in` (any stride via index array precomputed by
// caller); writes 8 outputs. Q14 matrix, result >> SHIFT with rounding.
static inline void dct8_fwd(const i32 in[8], i32 out[8], u32 shift) {
    const i32 rnd = (i32)1 << (shift - 1);
    for (u32 k = 0; k < 8; ++k) {
        i32 acc = 0;
        for (u32 n = 0; n < 8; ++n) acc += CMAT[k][n] * in[n];
        out[k] = (acc + rnd) >> shift;
    }
}

// Inverse 1D DCT (transpose of CMAT).
static inline void idct8(const i32 in[8], i32 out[8], u32 shift) {
    const i32 rnd = (i32)1 << (shift - 1);
    for (u32 n = 0; n < 8; ++n) {
        i32 acc = 0;
        for (u32 k = 0; k < 8; ++k) acc += CMAT[k][n] * in[k];
        out[n] = (acc + rnd) >> shift;
    }
}

// Forward transform of one chunk. `vox` is the u8 cube; `dc` is the per-chunk
// mean already computed by the pipeline and removed here. Output coefficients in
// `coef` (i16), same row-major layout.
void vc_dct_int8_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc) {
    // Per-8^3-sub-block scratch (small, fits registers/L1). We process blocks
    // one at a time; the chunk-sized buffers are the caller's.
    for (u32 bz = 0; bz < CS; bz += 8)
    for (u32 by = 0; by < CS; by += 8)
    for (u32 bx = 0; bx < CS; bx += 8) {
        i32 blk[8][8][8];
        // Load + DC-subtract into i32 work block.
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) {
            const u8 *src = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < 8; ++x) blk[z][y][x] = (i32)src[x] - dc;
        }
        // Pass X (rows): shift Q (>>14) keeps scale ~1.
        i32 tx[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) dct8_fwd(blk[z][y], tx[z][y], Q);
        // Pass Y (columns within slice): gather 8 contiguous-x lanes per k.
        i32 ty[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 x = 0; x < 8; ++x) {
            i32 col[8], outc[8];
            for (u32 y = 0; y < 8; ++y) col[y] = tx[z][y][x];
            dct8_fwd(col, outc, Q);
            for (u32 y = 0; y < 8; ++y) ty[z][y][x] = outc[y];
        }
        // Pass Z (depth): shift Q again. Total scale 2^-42 * 2^(3*?) ~ unit.
        for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x) {
            i32 col[8], outc[8];
            for (u32 z = 0; z < 8; ++z) col[z] = ty[z][y][x];
            dct8_fwd(col, outc, Q);
            for (u32 z = 0; z < 8; ++z) {
                i32 v = outc[z];
                if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
                coef[(size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + (bx + x)] = (i16)v;
            }
        }
    }
}

// Inverse transform of one chunk: coef (i16) -> u8 cube, adding DC and clamping.
void vc_dct_int8_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc) {
    for (u32 bz = 0; bz < CS; bz += 8)
    for (u32 by = 0; by < CS; by += 8)
    for (u32 bx = 0; bx < CS; bx += 8) {
        i32 blk[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) {
            const i16 *src = coef + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < 8; ++x) blk[z][y][x] = (i32)src[x];
        }
        // Inverse Z, then Y, then X (reverse order of forward).
        i32 tz[8][8][8];
        for (u32 y = 0; y < 8; ++y)
        for (u32 x = 0; x < 8; ++x) {
            i32 col[8], outc[8];
            for (u32 z = 0; z < 8; ++z) col[z] = blk[z][y][x];
            idct8(col, outc, Q);
            for (u32 z = 0; z < 8; ++z) tz[z][y][x] = outc[z];
        }
        i32 ty[8][8][8];
        for (u32 z = 0; z < 8; ++z)
        for (u32 x = 0; x < 8; ++x) {
            i32 col[8], outc[8];
            for (u32 y = 0; y < 8; ++y) col[y] = tz[z][y][x];
            idct8(col, outc, Q);
            for (u32 y = 0; y < 8; ++y) ty[z][y][x] = outc[y];
        }
        for (u32 z = 0; z < 8; ++z)
        for (u32 y = 0; y < 8; ++y) {
            i32 row[8], outc[8];
            for (u32 x = 0; x < 8; ++x) row[x] = ty[z][y][x];
            idct8(row, outc, Q);
            u8 *dst = vox + (size_t)(bz + z) * SLZ + (size_t)(by + y) * SLY + bx;
            for (u32 x = 0; x < 8; ++x) {
                i32 v = outc[x] + dc;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                dst[x] = (u8)v;
            }
        }
    }
}
