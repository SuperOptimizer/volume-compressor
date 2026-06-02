// Self-test for the edge-sensitive perceptual metrics (PLAN §4): GMSD + HaarPSI.
// Known inputs -> expected ~values:
//   identical volumes        -> GMSD == 0,        HaarPSI == 1
//   degraded (blurred/noisy)  -> GMSD > 0,         HaarPSI < 1
//   monotonicity: more distortion -> larger GMSD, smaller HaarPSI
#include "../src/metrics/metrics.h"
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail = 1; } \
                              else printf("ok: %s\n", msg); } while (0)

// Structured volume: gradients + sinusoidal "fibers" + sharp block edges so the
// edge-sensitive metrics have real gradients/Haar energy to measure.
static u8 *gen(u32 d, u32 h, u32 w) {
    u8 *v = (u8 *)malloc((size_t)d * h * w);
    for (u32 z = 0; z < d; ++z)
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        double f = 110.0 + 50.0 * sin(x * 0.20) * cos(y * 0.15)
                 + 35.0 * (((x / 8 + y / 8 + z / 8) & 1) ? 1 : 0);
        int iv = (int)(f + 0.5);
        v[((size_t)z * h + y) * w + x] = (u8)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
    }
    return v;
}

// 3x3x3 box blur (smears edges -> classic c3d artifact the user reported).
static u8 *blur(const u8 *s, u32 d, u32 h, u32 w) {
    u8 *o = (u8 *)malloc((size_t)d * h * w);
    for (u32 z = 0; z < d; ++z)
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        int acc = 0, n = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int zz = (int)z + dz, yy = (int)y + dy, xx = (int)x + dx;
            if (zz < 0 || yy < 0 || xx < 0 || zz >= (int)d || yy >= (int)h || xx >= (int)w) continue;
            acc += s[((size_t)zz * h + yy) * w + xx]; ++n;
        }
        o[((size_t)z * h + y) * w + x] = (u8)(acc / n);
    }
    return o;
}

// Add uniform noise of amplitude amp.
static u8 *noisy(const u8 *s, u32 d, u32 h, u32 w, int amp) {
    u8 *o = (u8 *)malloc((size_t)d * h * w);
    unsigned st = 12345;
    for (size_t i = 0; i < (size_t)d * h * w; ++i) {
        st = st * 1103515245u + 12345u;
        int n = (int)((st >> 16) % (2u * amp + 1u)) - amp;
        int v = (int)s[i] + n;
        o[i] = (u8)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    return o;
}

int main(void) {
    const u32 D = 48, H = 48, W = 48;
    u8 *ref = gen(D, H, W);

    // 1) Identical -> GMSD 0, HaarPSI 1.
    double g0 = vc_gmsd(ref, ref, D, H, W);
    double h0 = vc_haarpsi(ref, ref, D, H, W);
    printf("identical : GMSD=%.6f  HaarPSI=%.6f\n", g0, h0);
    CHECK(fabs(g0) < 1e-6, "GMSD == 0 for identical");
    CHECK(fabs(h0 - 1.0) < 1e-6, "HaarPSI == 1 for identical");

    // 2) Blurred (smear) -> degraded.
    u8 *bl = blur(ref, D, H, W);
    double gb = vc_gmsd(ref, bl, D, H, W);
    double hb = vc_haarpsi(ref, bl, D, H, W);
    printf("blurred   : GMSD=%.6f  HaarPSI=%.6f\n", gb, hb);
    CHECK(gb > 0.0, "GMSD > 0 for blurred");
    CHECK(hb < 1.0 && hb > 0.0, "HaarPSI in (0,1) for blurred");

    // 3) Light vs heavy noise -> monotonic.
    u8 *nl = noisy(ref, D, H, W, 8);
    u8 *nh = noisy(ref, D, H, W, 40);
    double gnl = vc_gmsd(ref, nl, D, H, W), gnh = vc_gmsd(ref, nh, D, H, W);
    double hnl = vc_haarpsi(ref, nl, D, H, W), hnh = vc_haarpsi(ref, nh, D, H, W);
    printf("noise  8  : GMSD=%.6f  HaarPSI=%.6f\n", gnl, hnl);
    printf("noise 40  : GMSD=%.6f  HaarPSI=%.6f\n", gnh, hnh);
    CHECK(gnh > gnl, "GMSD monotone increasing with noise");
    CHECK(hnh < hnl, "HaarPSI monotone decreasing with noise");

    // 4) Range sanity.
    CHECK(g0 >= 0.0 && gb >= 0.0 && gnh >= 0.0, "GMSD >= 0");
    CHECK(h0 <= 1.0001 && hb <= 1.0001 && hnh <= 1.0001, "HaarPSI <= 1");

    free(ref); free(bl); free(nl); free(nh);
    printf(fail ? "\nSOME METRIC TESTS FAILED\n" : "\nALL METRIC TESTS PASSED\n");
    return fail;
}
