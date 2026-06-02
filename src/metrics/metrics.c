// Metric implementations (PLAN §4). Cheap, pure C23.
#include "metrics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Error-magnitude histogram for percentiles (errors are 0..255).
static f64 hist_percentile(const u64 *hist, u64 total, f64 pct) {
    if (total == 0) return 0.0;
    u64 target = (u64)(pct * (f64)total);
    u64 acc = 0;
    for (int e = 0; e <= 255; ++e) {
        acc += hist[e];
        if (acc >= target) return (f64)e;
    }
    return 255.0;
}

// --- SSIM on a single 2D slice using 8x8 non-overlapping windows -----------
// Cheap MS-SSIM proxy: mean SSIM over 8x8 windows of each axis-orthogonal slice,
// averaged over the three axes (PLAN §4 "per-axis-slice mean is fine for now").
#define SSIM_WIN 8
static f64 ssim_slice(const u8 *a, const u8 *b, u32 h, u32 w) {
    const f64 C1 = (0.01 * 255) * (0.01 * 255);
    const f64 C2 = (0.03 * 255) * (0.03 * 255);
    f64 acc = 0.0; u64 cnt = 0;
    for (u32 yy = 0; yy + SSIM_WIN <= h; yy += SSIM_WIN)
    for (u32 xx = 0; xx + SSIM_WIN <= w; xx += SSIM_WIN) {
        f64 sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
        for (u32 j = 0; j < SSIM_WIN; ++j)
        for (u32 i = 0; i < SSIM_WIN; ++i) {
            f64 va = a[(yy + j) * w + (xx + i)];
            f64 vb = b[(yy + j) * w + (xx + i)];
            sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb;
        }
        const f64 N = SSIM_WIN * SSIM_WIN;
        f64 ma = sa / N, mb = sb / N;
        f64 va = saa / N - ma * ma, vb = sbb / N - mb * mb;
        f64 cov = sab / N - ma * mb;
        f64 s = ((2 * ma * mb + C1) * (2 * cov + C2)) /
                ((ma * ma + mb * mb + C1) * (va + vb + C2));
        acc += s; cnt++;
    }
    return cnt ? acc / (f64)cnt : 1.0;
}

void vc_compute_metrics(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx,
                        vc_metrics *m) {
    const size_t N = (size_t)dz * dy * dx;
    u64 hist[256]; memset(hist, 0, sizeof(hist));
    f64 se = 0.0, sae = 0.0; u32 linf = 0;
    for (size_t i = 0; i < N; ++i) {
        i32 d = (i32)rec[i] - (i32)ref[i];
        u32 ad = d < 0 ? (u32)(-d) : (u32)d;
        se += (f64)d * d;
        sae += ad;
        if (ad > linf) linf = ad;
        hist[ad]++;
    }
    f64 mse = se / (f64)N;
    m->psnr = mse > 0.0 ? 10.0 * log10(255.0 * 255.0 / mse) : 99.0;
    m->mae  = sae / (f64)N;
    m->linf = (f64)linf;
    m->p95  = hist_percentile(hist, N, 0.95);
    m->p99  = hist_percentile(hist, N, 0.99);
    m->p999 = hist_percentile(hist, N, 0.999);

    // MS-SSIM proxy: average 8x8-window SSIM over slices along each of 3 axes.
    f64 ssim_sum = 0.0; u64 ssim_cnt = 0;
    // Z slices (dy x dx)
    for (u32 z = 0; z < dz; ++z) {
        ssim_sum += ssim_slice(ref + (size_t)z * dy * dx, rec + (size_t)z * dy * dx, dy, dx);
        ssim_cnt++;
    }
    // Y slices (dz x dx) and X slices (dz x dy) need gathered planes.
    u8 *pa = (u8 *)malloc((size_t)dz * (dx > dy ? dx : dy));
    u8 *pb = (u8 *)malloc((size_t)dz * (dx > dy ? dx : dy));
    if (pa && pb) {
        for (u32 y = 0; y < dy; ++y) {
            for (u32 z = 0; z < dz; ++z) {
                memcpy(pa + (size_t)z * dx, ref + ((size_t)z * dy + y) * dx, dx);
                memcpy(pb + (size_t)z * dx, rec + ((size_t)z * dy + y) * dx, dx);
            }
            ssim_sum += ssim_slice(pa, pb, dz, dx); ssim_cnt++;
        }
        for (u32 x = 0; x < dx; ++x) {
            for (u32 z = 0; z < dz; ++z)
            for (u32 y = 0; y < dy; ++y) {
                pa[(size_t)z * dy + y] = ref[((size_t)z * dy + y) * dx + x];
                pb[(size_t)z * dy + y] = rec[((size_t)z * dy + y) * dx + x];
            }
            ssim_sum += ssim_slice(pa, pb, dz, dy); ssim_cnt++;
        }
    }
    free(pa); free(pb);
    m->ms_ssim = ssim_cnt ? ssim_sum / (f64)ssim_cnt : 1.0;
}

double vc_gmsd(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx) {
    (void)ref; (void)rec; (void)dz; (void)dy; (void)dx;
    return -1.0; // TODO: gradient-magnitude similarity deviation
}
double vc_haarpsi(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx) {
    (void)ref; (void)rec; (void)dz; (void)dy; (void)dx;
    return -1.0; // TODO: Haar perceptual similarity index
}
