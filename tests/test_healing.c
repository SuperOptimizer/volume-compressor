// Unit tests for the boundary-healing blocks (src/healing/). Two mandated
// properties (PLAN §7 healing tests):
//   1. IDENTITY-ISH on a smooth volume: deblock/cdef must barely touch a volume
//      with no seams (it must not invent distortion in flat/smooth structure).
//   2. SEAM-REDUCING on a synthetic blocky volume: a volume with deliberate DC
//      steps at the 16^3 sub-block / chunk faces must have its mean face-step
//      magnitude DROP after deblocking, while staying close to the clean source.
#include "../src/healing/healing.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define BLK 16u

static double mean_abs_diff(const u8 *a, const u8 *b, size_t n) {
    double s = 0; for (size_t i = 0; i < n; ++i) s += fabs((double)a[i] - b[i]);
    return s / (double)n;
}

// Mean |step| across every interior 16-grid face along all three axes.
static double seam_step(const u8 *v, u32 dz, u32 dy, u32 dx) {
    double s = 0; u64 c = 0;
    size_t sX = 1, sY = dx, sZ = (size_t)dy * dx;
    for (u32 n = BLK; n < dz; n += BLK)
        for (u32 y = 0; y < dy; ++y) for (u32 x = 0; x < dx; ++x) {
            size_t i0 = (size_t)(n - 1) * sZ + y * sY + x * sX;
            s += fabs((double)v[i0] - v[i0 + sZ]); c++;
        }
    for (u32 n = BLK; n < dy; n += BLK)
        for (u32 z = 0; z < dz; ++z) for (u32 x = 0; x < dx; ++x) {
            size_t i0 = (size_t)z * sZ + (n - 1) * sY + x * sX;
            s += fabs((double)v[i0] - v[i0 + sY]); c++;
        }
    for (u32 n = BLK; n < dx; n += BLK)
        for (u32 z = 0; z < dz; ++z) for (u32 y = 0; y < dy; ++y) {
            size_t i0 = (size_t)z * sZ + y * sY + (n - 1) * sX;
            s += fabs((double)v[i0] - v[i0 + sX]); c++;
        }
    return c ? s / (double)c : 0;
}

int main(void) {
    const u32 D = 64, H = 64, W = 64;
    const size_t N = (size_t)D * H * W;
    u8 *smooth = malloc(N), *work = malloc(N);
    int fail = 0;

    // --- 1. smooth volume: a low-frequency gradient, no seams. -------------
    for (u32 z = 0; z < D; ++z) for (u32 y = 0; y < H; ++y) for (u32 x = 0; x < W; ++x) {
        double f = 120 + 30 * sin(x * 0.05) + 20 * cos(y * 0.04) + 15 * sin(z * 0.03);
        smooth[((size_t)z * H + y) * W + x] = (u8)(f < 0 ? 0 : f > 255 ? 255 : f);
    }
    double seam_smooth = seam_step(smooth, D, H, W);

    memcpy(work, smooth, N);
    vc_heal_deblock(work, D, H, W, 16.0f, 0.5f);
    double mad_deblock = mean_abs_diff(smooth, work, N);
    printf("[smooth] deblock mean|delta| = %.4f (seam before=%.3f)\n", mad_deblock, seam_smooth);
    if (mad_deblock > 1.0) { printf("FAIL: deblock perturbs smooth volume too much\n"); fail = 1; }

    memcpy(work, smooth, N);
    vc_heal_cdef(work, D, H, W, 16.0f, 0.5f);
    double mad_cdef = mean_abs_diff(smooth, work, N);
    printf("[smooth] cdef    mean|delta| = %.4f\n", mad_cdef);
    if (mad_cdef > 2.0) { printf("FAIL: cdef perturbs smooth volume too much\n"); fail = 1; }

    // none must be exact identity.
    memcpy(work, smooth, N);
    vc_heal_none(work, D, H, W, 16.0f, 1.0f);
    if (mean_abs_diff(smooth, work, N) != 0.0) { printf("FAIL: none not identity\n"); fail = 1; }

    // --- 2. blocky volume: smooth source + a DC step injected per 16-block. -
    u8 *blocky = malloc(N);
    for (u32 z = 0; z < D; ++z) for (u32 y = 0; y < H; ++y) for (u32 x = 0; x < W; ++x) {
        int bias = (((x / BLK) + (y / BLK) + (z / BLK)) & 1) ? 10 : -10; // +-10 DC step
        int v = smooth[((size_t)z * H + y) * W + x] + bias;
        blocky[((size_t)z * H + y) * W + x] = (u8)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    double seam_blocky = seam_step(blocky, D, H, W);
    memcpy(work, blocky, N);
    vc_heal_deblock(work, D, H, W, 24.0f, 1.0f);
    double seam_healed = seam_step(work, D, H, W);
    double mad_to_clean_before = mean_abs_diff(smooth, blocky, N);
    double mad_to_clean_after  = mean_abs_diff(smooth, work, N);
    printf("[blocky] seam-step before=%.3f after=%.3f | dist-to-clean before=%.3f after=%.3f\n",
           seam_blocky, seam_healed, mad_to_clean_before, mad_to_clean_after);
    if (!(seam_healed < seam_blocky * 0.9)) {
        printf("FAIL: deblock did not reduce blocky seam-step by >=10%%\n"); fail = 1;
    }
    if (!(mad_to_clean_after < mad_to_clean_before)) {
        printf("FAIL: deblock did not move blocky volume closer to the clean source\n"); fail = 1;
    }

    free(smooth); free(work); free(blocky);
    printf(fail ? "\nHEALING TESTS FAILED\n" : "\nall healing tests passed\n");
    return fail;
}
