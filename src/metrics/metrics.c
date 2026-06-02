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

// ===========================================================================
// Edge-sensitive perceptual metrics (PLAN §4). Pure C23, compiler-vectorizable:
// straight-line loops over contiguous f32 SoA scratch, restrict pointers, no
// data-dependent branches in inner kernels, heap scratch (never stacked cubes).
//
// 2.5D CHOICE (documented): both GMSD and HaarPSI are natively 2D perceptual
// metrics built on 2D separable filters. For a 3D volume we compute the metric
// independently on every axis-orthogonal slice along ALL THREE axes (Z, Y, X)
// and average. This matches the MS-SSIM "per-axis-slice mean" choice already in
// this file, so the perceptual bundle is internally consistent. It captures
// in-plane blocking/smear on every face orientation (so chunk-face seams on any
// axis are seen) at ~3x the cost of a single-axis pass — cheap, and far cheaper
// (and better autovectorizing) than a true 3D-gradient/3D-Haar formulation,
// which the literature does not define anyway. The Y/X faces are gathered into a
// contiguous plane buffer first (same gather the SSIM path uses) so the inner
// filter loops stay unit-stride and vectorizable.
// ===========================================================================

// --- GMSD ------------------------------------------------------------------
// Gradient Magnitude Similarity Deviation (Xue et al. 2014). Prewitt gradients
// (h + v), gradient magnitude m = sqrt(gh^2 + gv^2), GMS map
//   GMS = (2*mr*md + c) / (mr^2 + md^2 + c),
// then GMSD = std-dev of the GMS map over the slice. 0 == identical; larger ==
// more structural distortion. Edge-sensitive: directly penalizes the gradient
// disruption that blocking/smear produce but MSE/PSNR average away.
#define GMSD_C 170.0f   // stabilizer in u8 gradient-magnitude units (~T*255^2 scaled; std GMSD T=0.0026 on [0,1] -> ~170 on [0,255])

// Compute sum and sum-of-squares of the GMS map for one 2D slice; accumulate
// into *acc_sum / *acc_sqr / *acc_cnt. `gr`,`gd` are scratch holding the
// gradient magnitude of ref/rec respectively (size h*w).
static void gmsd_slice(const u8 *restrict a, const u8 *restrict b,
                       u32 h, u32 w,
                       f32 *restrict gr, f32 *restrict gd,
                       f64 *acc_sum, f64 *acc_sqr, u64 *acc_cnt) {
    if (h < 3 || w < 3) return;
    // Prewitt gradient magnitude for both images. Interior pixels only (the
    // 3x3 kernel needs a 1-pixel border); branchless straight-line loop.
    for (u32 y = 1; y + 1 < h; ++y) {
        const u8 *r0 = a + (size_t)(y - 1) * w;
        const u8 *r1 = a + (size_t)(y    ) * w;
        const u8 *r2 = a + (size_t)(y + 1) * w;
        const u8 *s0 = b + (size_t)(y - 1) * w;
        const u8 *s1 = b + (size_t)(y    ) * w;
        const u8 *s2 = b + (size_t)(y + 1) * w;
        f32 *grr = gr + (size_t)y * w;
        f32 *gdd = gd + (size_t)y * w;
        for (u32 x = 1; x + 1 < w; ++x) {
            // Prewitt: horizontal = (right cols - left cols), vertical = (bottom rows - top rows)
            i32 ah = (i32)r0[x+1] + r1[x+1] + r2[x+1] - r0[x-1] - r1[x-1] - r2[x-1];
            i32 av = (i32)r2[x-1] + r2[x] + r2[x+1] - r0[x-1] - r0[x] - r0[x+1];
            i32 bh = (i32)s0[x+1] + s1[x+1] + s2[x+1] - s0[x-1] - s1[x-1] - s2[x-1];
            i32 bv = (i32)s2[x-1] + s2[x] + s2[x+1] - s0[x-1] - s0[x] - s0[x+1];
            grr[x] = sqrtf((f32)(ah * ah + av * av));
            gdd[x] = sqrtf((f32)(bh * bh + bv * bv));
        }
    }
    // GMS map + running moments over interior pixels.
    f64 s = 0.0, sq = 0.0; u64 c = 0;
    for (u32 y = 1; y + 1 < h; ++y) {
        const f32 *grr = gr + (size_t)y * w;
        const f32 *gdd = gd + (size_t)y * w;
        for (u32 x = 1; x + 1 < w; ++x) {
            f32 mr = grr[x], md = gdd[x];
            f32 gms = (2.0f * mr * md + GMSD_C) / (mr * mr + md * md + GMSD_C);
            s += gms; sq += (f64)gms * gms; ++c;
        }
    }
    *acc_sum += s; *acc_sqr += sq; *acc_cnt += c;
}

double vc_gmsd(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx) {
    const size_t pmax = (size_t)dz * (dx > dy ? dx : dy);
    const size_t smax = (size_t)((dy > dz ? dy : dz)) * (dx > dy ? dx : dy);
    // gradient scratch large enough for the biggest slice across all 3 axes.
    size_t gn = (size_t)dy * dx;
    if ((size_t)dz * dx > gn) gn = (size_t)dz * dx;
    if ((size_t)dz * dy > gn) gn = (size_t)dz * dy;
    f32 *gr = (f32 *)calloc(gn, sizeof(f32));
    f32 *gd = (f32 *)calloc(gn, sizeof(f32));
    u8  *pa = (u8 *)malloc(pmax ? pmax : 1);
    u8  *pb = (u8 *)malloc(pmax ? pmax : 1);
    (void)smax;
    if (!gr || !gd || !pa || !pb) { free(gr); free(gd); free(pa); free(pb); return -1.0; }

    f64 sum = 0.0, sqr = 0.0; u64 cnt = 0;
    // Z slices (dy x dx) — already contiguous.
    for (u32 z = 0; z < dz; ++z)
        gmsd_slice(ref + (size_t)z * dy * dx, rec + (size_t)z * dy * dx,
                   dy, dx, gr, gd, &sum, &sqr, &cnt);
    // Y slices (dz x dx) — gather.
    for (u32 y = 0; y < dy; ++y) {
        for (u32 z = 0; z < dz; ++z) {
            memcpy(pa + (size_t)z * dx, ref + ((size_t)z * dy + y) * dx, dx);
            memcpy(pb + (size_t)z * dx, rec + ((size_t)z * dy + y) * dx, dx);
        }
        gmsd_slice(pa, pb, dz, dx, gr, gd, &sum, &sqr, &cnt);
    }
    // X slices (dz x dy) — gather.
    for (u32 x = 0; x < dx; ++x) {
        for (u32 z = 0; z < dz; ++z)
        for (u32 y = 0; y < dy; ++y) {
            pa[(size_t)z * dy + y] = ref[((size_t)z * dy + y) * dx + x];
            pb[(size_t)z * dy + y] = rec[((size_t)z * dy + y) * dx + x];
        }
        gmsd_slice(pa, pb, dz, dy, gr, gd, &sum, &sqr, &cnt);
    }
    free(gr); free(gd); free(pa); free(pb);
    if (cnt == 0) return 0.0;
    f64 mean = sum / (f64)cnt;
    f64 var  = sqr / (f64)cnt - mean * mean;
    if (var < 0.0) var = 0.0;           // guard fp rounding
    return sqrt(var);                    // GMSD = std-dev of GMS map
}

// --- HaarPSI ----------------------------------------------------------------
// Haar Perceptual Similarity Index (Reisenhofer et al. 2018). Uses a 2-scale
// separable Haar wavelet (6 filter responses: horizontal & vertical high-pass
// at scales 1 and 2 form the similarity; the coarsest low-pass forms the weight).
// Local similarity per orientation:
//   S = (2|c_ref||c_rec| + C) / (|c_ref|^2 + |c_rec|^2 + C)   (on scale-2 HF)
// combined across the 2 orientations with a logistic, weighted by the max HF
// magnitude (low-freq-weighted pooling per the paper's weight map), pooled, then
// inverse-logistic. 1 == identical; lower == more perceptual distortion.
#define HAARPSI_C     30.0f    // similarity stabilizer (u8 scale; paper C=30 on [0,255])
#define HAARPSI_ALPHA 4.2f     // logistic slope (paper alpha=4.2)

static inline f32 hpsi_logistic(f32 x) { return 1.0f / (1.0f + expf(-HAARPSI_ALPHA * x)); }

// One 2D slice. Computes the Haar HF responses at two scales for both images
// via box partial sums, accumulates weighted logistic similarity. Scratch:
// hr/vr/hd/vd hold scale-2 HF magnitudes; wr/wd hold scale-1 HF (weight).
static void haarpsi_slice(const u8 *restrict a, const u8 *restrict b,
                          u32 h, u32 w,
                          f64 *acc_num, f64 *acc_den) {
    if (h < 4 || w < 4) return;
    // Scale-1 Haar HF (2x2): horizontal = (right - left) avg, vertical = (bottom - top).
    // Scale-2 Haar HF (4x4): same over 4x4 blocks. We evaluate at every interior
    // pixel using a sliding 4x4 window (top-left anchored), giving dense maps.
    // To keep it cheap and vectorizable we compute per-pixel response with a
    // straight-line kernel over the 4x4 neighbourhood.
    f64 num = 0.0, den = 0.0;
    const u32 W = 4; // scale-2 window
    for (u32 y = 0; y + W <= h; ++y) {
        for (u32 x = 0; x + W <= w; ++x) {
            // Scale-2: split 4x4 into halves. Horizontal HF = (left 4x2) - (right 4x2).
            i32 al = 0, ar = 0, at = 0, ab = 0;
            i32 bl = 0, br = 0, bt = 0, bb = 0;
            for (u32 j = 0; j < W; ++j) {
                const u8 *ra = a + (size_t)(y + j) * w + x;
                const u8 *rb = b + (size_t)(y + j) * w + x;
                i32 alj = (i32)ra[0] + ra[1];
                i32 arj = (i32)ra[2] + ra[3];
                i32 blj = (i32)rb[0] + rb[1];
                i32 brj = (i32)rb[2] + rb[3];
                al += alj; ar += arj; bl += blj; br += brj;
                int top = (j < 2);
                at += top ? (alj + arj) : 0; ab += top ? 0 : (alj + arj);
                bt += top ? (blj + brj) : 0; bb += top ? 0 : (blj + brj);
            }
            f32 ah2 = (f32)(al - ar), av2 = (f32)(at - ab);   // ref scale-2 HF
            f32 bh2 = (f32)(bl - br), bv2 = (f32)(bt - bb);    // rec scale-2 HF
            // Scale-1 (top-left 2x2) HF as the weight source (high-freq energy).
            const u8 *r0 = a + (size_t)y * w + x, *r1 = a + (size_t)(y + 1) * w + x;
            const u8 *q0 = b + (size_t)y * w + x, *q1 = b + (size_t)(y + 1) * w + x;
            f32 ah1 = (f32)((i32)r0[0] + r1[0] - r0[1] - r1[1]);
            f32 av1 = (f32)((i32)r0[0] + r0[1] - r1[0] - r1[1]);
            f32 bh1 = (f32)((i32)q0[0] + q1[0] - q0[1] - q1[1]);
            f32 bv1 = (f32)((i32)q0[0] + q0[1] - q1[0] - q1[1]);

            // Per-orientation similarity on scale-2 HF.
            f32 sh = (2.0f * fabsf(ah2) * fabsf(bh2) + HAARPSI_C) /
                     (ah2 * ah2 + bh2 * bh2 + HAARPSI_C);
            f32 sv = (2.0f * fabsf(av2) * fabsf(bv2) + HAARPSI_C) /
                     (av2 * av2 + bv2 * bv2 + HAARPSI_C);
            f32 lh = hpsi_logistic(sh), lv = hpsi_logistic(sv);
            // Weight = max HF magnitude across images/scales (low-freq-weighted
            // pooling: edges dominate, flats contribute little).
            f32 wh = fmaxf(fabsf(ah1), fabsf(bh1));
            f32 wv = fmaxf(fabsf(av1), fabsf(bv1));
            num += (f64)lh * wh + (f64)lv * wv;
            den += (f64)wh + (f64)wv;
        }
    }
    *acc_num += num; *acc_den += den;
}

double vc_haarpsi(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx) {
    const size_t pmax = (size_t)dz * (dx > dy ? dx : dy);
    u8 *pa = (u8 *)malloc(pmax ? pmax : 1);
    u8 *pb = (u8 *)malloc(pmax ? pmax : 1);
    if (!pa || !pb) { free(pa); free(pb); return -1.0; }

    f64 num = 0.0, den = 0.0;
    for (u32 z = 0; z < dz; ++z)
        haarpsi_slice(ref + (size_t)z * dy * dx, rec + (size_t)z * dy * dx, dy, dx, &num, &den);
    for (u32 y = 0; y < dy; ++y) {
        for (u32 z = 0; z < dz; ++z) {
            memcpy(pa + (size_t)z * dx, ref + ((size_t)z * dy + y) * dx, dx);
            memcpy(pb + (size_t)z * dx, rec + ((size_t)z * dy + y) * dx, dx);
        }
        haarpsi_slice(pa, pb, dz, dx, &num, &den);
    }
    for (u32 x = 0; x < dx; ++x) {
        for (u32 z = 0; z < dz; ++z)
        for (u32 y = 0; y < dy; ++y) {
            pa[(size_t)z * dy + y] = ref[((size_t)z * dy + y) * dx + x];
            pb[(size_t)z * dy + y] = rec[((size_t)z * dy + y) * dx + x];
        }
        haarpsi_slice(pa, pb, dz, dy, &num, &den);
    }
    free(pa); free(pb);
    if (den <= 0.0) return 1.0;          // featureless volume -> perfect
    // Inverse logistic of the weighted-mean similarity, squared (paper form).
    f64 mean = num / den;                // in (0,1)
    // map back through inverse logistic and square so identical -> 1.0
    f64 invlog = (f64)hpsi_logistic(1.0f); // logistic(alpha) at S=1 (max similarity)
    f64 hp = mean / invlog;
    if (hp > 1.0) hp = 1.0;
    return hp * hp;
}
