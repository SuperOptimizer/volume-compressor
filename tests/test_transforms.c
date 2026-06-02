// Round-trip self-tests for the Phase-1 transform candidates (PLAN §2). Each
// transform must invert correctly WITHOUT quantization (the codec's quantizer is
// the only intended lossy step). The integer DCTs and the float DCT round-trip to
// within their fixed-point/rounding tolerance; the ZFP-style integer lifting is
// bit-EXACT lossless; the 9/7 wavelet round-trips to small float tolerance.
//
// Includes the transform .c files directly under renamed symbols (does not touch
// codec.c / config.h pipeline selection), exactly like bench_transforms.c.
#include "../include/vc/types.h"
#include "../src/config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define BCS   ((u32)VC_CHUNK_SIDE)
#define BCVOX ((size_t)BCS * BCS * BCS)

#undef CS
#undef SLY
#undef SLZ
#undef CVOX
#undef B
#undef Q
#define vc_dct_int8_fwd   T_dct8_fwd
#define vc_dct_int8_inv   T_dct8_inv
#include "../src/transform/dct_int.c"
#undef vc_dct_int8_fwd
#undef vc_dct_int8_inv
#undef CS
#undef SLY
#undef SLZ
#undef Q
#define vc_dct_int16_fwd  T_dct16_fwd
#define vc_dct_int16_inv  T_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv
#undef CS
#undef SLY
#undef SLZ
#undef B
#undef Q
#define vc_dct_float8_fwd T_dctf8_fwd
#define vc_dct_float8_inv T_dctf8_inv
#include "../src/transform/dct_float.c"
#undef vc_dct_float8_fwd
#undef vc_dct_float8_inv
#undef CS
#undef SLY
#undef SLZ
#define vc_dwt97_fwd      T_dwt97_fwd
#define vc_dwt97_inv      T_dwt97_inv
#include "../src/transform/dwt_97.c"
#undef vc_dwt97_fwd
#undef vc_dwt97_inv
#undef CS
#undef SLY
#undef SLZ
#undef CVOX
#define vc_lift_zfp_fwd   T_zfp_fwd
#define vc_lift_zfp_inv   T_zfp_inv
#include "../src/transform/lift_zfp.c"
#undef vc_lift_zfp_fwd
#undef vc_lift_zfp_inv
#undef CS
#undef SLY
#undef SLZ
#undef B

#define CS   BCS
#define CVOX BCVOX

typedef void (*fwd_fn)(i16 *restrict, const u8 *restrict, i32);
typedef void (*inv_fn)(u8 *restrict, const i16 *restrict, i32);

static int fail = 0;

static double rt(fwd_fn fwd, inv_fn inv, const u8 *vol, double *linf_out) {
    i16 *coef = malloc(CVOX * sizeof(i16));
    u8  *rec  = malloc(CVOX);
    u64 sum = 0; for (size_t i = 0; i < CVOX; ++i) sum += vol[i];
    i32 dc = (i32)((sum + CVOX / 2) / CVOX);
    fwd(coef, vol, dc);
    inv(rec, coef, dc);
    double se = 0.0, linf = 0.0;
    for (size_t i = 0; i < CVOX; ++i) {
        double e = (double)rec[i] - vol[i];
        se += e * e; if (fabs(e) > linf) linf = fabs(e);
    }
    free(coef); free(rec);
    double mse = se / CVOX;
    *linf_out = linf;
    return mse <= 0 ? 999.0 : 10.0 * log10(255.0 * 255.0 / mse);
}

static u8 *gen(int kind) {
    u8 *v = malloc(CVOX);
    for (u32 z = 0; z < CS; ++z)
    for (u32 y = 0; y < CS; ++y)
    for (u32 x = 0; x < CS; ++x) {
        int val;
        if (kind == 0) val = (int)(128 + 60 * sin(x * 0.1) * cos(y * 0.07 + z * 0.05));
        else if (kind == 1) val = 100;
        else val = (int)((x * 7 + y * 13 + z * 3) & 0xff);
        v[((size_t)z * CS + y) * CS + x] = (u8)(val < 0 ? 0 : (val > 255 ? 255 : val));
    }
    return v;
}

static void check(const char *name, fwd_fn fwd, inv_fn inv,
                  double psnr_floor, int lossless) {
    const char *kinds[] = { "smooth", "uniform", "high-freq" };
    for (int k = 0; k < 3; ++k) {
        u8 *v = gen(k);
        double linf; double psnr = rt(fwd, inv, v, &linf);
        int ok = lossless ? (linf == 0.0)
                          : (psnr >= psnr_floor || (k == 1 && linf <= 1.0));
        printf("  %-11s %-9s PSNR=%7.2f Linf=%5.0f  %s\n",
               name, kinds[k], psnr, linf, ok ? "ok" : "FAIL");
        if (!ok) fail = 1;
        free(v);
    }
}

int main(void) {
    printf("transform round-trip (no quant), chunk=%u\n", CS);
    check("dct-int8",  T_dct8_fwd,  T_dct8_inv,  40.0, 0);
    check("dct-int16", T_dct16_fwd, T_dct16_inv, 40.0, 0);
    check("dct-float8",T_dctf8_fwd, T_dctf8_inv, 45.0, 0);
    check("dwt-9/7",   T_dwt97_fwd, T_dwt97_inv, 45.0, 0);
    check("zfp-lift4", T_zfp_fwd,   T_zfp_inv,    0.0, 1);  // bit-exact lossless
    printf(fail ? "\nSOME TRANSFORM TESTS FAILED\n" : "\nALL TRANSFORM TESTS PASSED\n");
    return fail;
}
