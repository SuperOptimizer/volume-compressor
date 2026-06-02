// THROWAWAY Phase-1 transform bake-off bench (new file; does NOT touch codec.c,
// entropy/, quant/, or the compare harness). It includes each candidate
// transform .c directly under a renamed symbol set, then runs a self-contained
// pipeline (per-chunk DC removal -> transform -> dead-zone quant -> count
// nonzeros as a rate proxy AND a real Rice-byte estimate -> dequant -> inverse)
// so all transforms are compared on identical data, quant, and metrics.
//
// Rate model: we entropy-code the quantized i16 levels with a simple order-0
// estimate (sum of per-symbol ceil(log2) magnitude bits + sign + a 0-run model),
// which tracks the real Rice/rANS ordering closely enough to RANK transforms at
// matched quality. Ratio is reported against this estimate; PSNR/SSIM/GMSD and
// the boundary-step blocking metric are computed on the true reconstructed u8.
//
// Build: see CMakeLists (target bench_transforms). Each transform is compiled in
// its own translation unit-like include block with macro-renamed entry points so
// the four candidates coexist in one binary.
#include "../include/vc/types.h"
#include "../src/config.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>

#define BCS   ((u32)VC_CHUNK_SIDE)
#define BCVOX ((size_t)BCS * BCS * BCS)

// --- Pull in each transform implementation, renaming its public symbols so the
// candidates coexist in one TU. Each .c only depends on types.h + config.h. The
// per-file local macros (CS/SLY/SLZ/B/Q/CVOX) are #undef'd between includes so
// each file's own definitions take effect (they differ: 8/16/4 block edges).
#define TU_UNDEF()  do{}while(0)
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

// Bench-local geometry (restored after the includes scrubbed CS/CVOX).
#define CS   BCS
#define CVOX BCVOX

typedef void (*fwd_fn)(i16 *restrict, const u8 *restrict, i32);
typedef void (*inv_fn)(u8 *restrict, const i16 *restrict, i32);

typedef struct { const char *name; fwd_fn fwd; inv_fn inv; } transform_t;
static const transform_t TFORMS[] = {
    { "dct-int8",  T_dct8_fwd,  T_dct8_inv  },
    { "dct-int16", T_dct16_fwd, T_dct16_inv },
    { "dct-float8",T_dctf8_fwd, T_dctf8_inv },
    { "dwt-9/7",   T_dwt97_fwd, T_dwt97_inv },
    { "zfp-lift4", T_zfp_fwd,   T_zfp_inv   },
};
#define NT ((int)(sizeof(TFORMS)/sizeof(TFORMS[0])))

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static u8 *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    u8 *b = (u8 *)malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); *len = (size_t)sz; return b;
}

static inline void quantize(const i16 *restrict c, i16 *restrict q, size_t n, f32 step) {
    const f32 inv = 1.f / step;
    for (size_t i = 0; i < n; ++i) {
        f32 a = fabsf((f32)c[i]); i32 lvl = 0;
        if (a >= 0.5f * step) lvl = (i32)(a * inv - 0.5f) + 1;
        q[i] = (i16)(c[i] < 0 ? -lvl : lvl);
    }
}
static inline void dequantize(const i16 *restrict q, i16 *restrict c, size_t n, f32 step) {
    for (size_t i = 0; i < n; ++i) {
        i32 l = q[i], a = l < 0 ? -l : l;
        f32 r = (a == 0) ? 0.f : (f32)a * step;
        i32 v = (i32)lrintf(l < 0 ? -r : r);
        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
        c[i] = (i16)v;
    }
}

// Order-0 rate estimate (bits): magnitude bits + sign + a cheap run-length model
// for zeros (mirrors what Rice/rANS achieve on quantized DCT/DWT coefficients;
// used only to RANK transforms, not as the shipped coder).
static double rate_bits(const i16 *q, size_t n) {
    double bits = 0.0; size_t zrun = 0;
    for (size_t i = 0; i < n; ++i) {
        i32 v = q[i];
        if (v == 0) { zrun++; continue; }
        if (zrun) { bits += 2.0 + 2.0 * floor(log2((double)zrun + 1.0)); zrun = 0; }
        u32 a = (u32)(v < 0 ? -v : v);
        bits += 1.0;                                   // sign
        bits += 2.0 + 2.0 * floor(log2((double)a));    // exp-Golomb-ish magnitude
    }
    if (zrun) bits += 2.0 + 2.0 * floor(log2((double)zrun + 1.0));
    return bits;
}

// Max chunk-boundary step: mean |delta| across the planes where adjacent chunks
// meet (x,y,z = multiples of CS), measured on the RECONSTRUCTION. PSNR hides this
// per-chunk DC/seam discontinuity; this surfaces it directly. Also a simple
// gradient-fidelity number: mean |grad_rec - grad_ref| over the volume (edge MAE).
static void blocking_metrics(const u8 *ref, const u8 *rec, u32 D, u32 H, u32 W,
                             double *seam_step, double *edge_mae) {
    double sstep = 0.0; size_t sn = 0;
    // X seams (x = k*CS): |rec[x] - rec[x-1]| compared to ref's own step.
    for (u32 z = 0; z < D; ++z)
    for (u32 y = 0; y < H; ++y)
    for (u32 x = CS; x < W; x += CS) {
        size_t i = ((size_t)z * H + y) * W + x;
        double dr = fabs((double)rec[i] - rec[i - 1]);
        double dref = fabs((double)ref[i] - ref[i - 1]);
        sstep += fabs(dr - dref); sn++;
    }
    *seam_step = sn ? sstep / sn : 0.0;
    // Edge fidelity: forward-x gradient MAE over the interior.
    double e = 0.0; size_t en = 0;
    for (u32 z = 0; z < D; ++z)
    for (u32 y = 0; y < H; ++y)
    for (u32 x = 1; x < W; ++x) {
        size_t i = ((size_t)z * H + y) * W + x;
        double gr = (double)rec[i] - rec[i - 1];
        double gf = (double)ref[i] - ref[i - 1];
        e += fabs(gr - gf); en++;
    }
    *edge_mae = en ? e / en : 0.0;
}

// Run one transform over a volume at a given quant step; fill metrics + timing.
static void run_transform(const transform_t *t, const u8 *vol, u32 D, u32 H, u32 W,
                          f32 step, double *ratio, vc_metrics *m,
                          double *enc_mbs, double *dec_mbs,
                          double *seam, double *edge, double *gmsd) {
    const u32 ncz = (D + CS - 1) / CS, ncy = (H + CS - 1) / CS, ncx = (W + CS - 1) / CS;
    u8  *cv = malloc(CVOX);
    i16 *cc = malloc(CVOX * sizeof(i16));
    i16 *cq = malloc(CVOX * sizeof(i16));
    u8  *rec = calloc((size_t)D * H * W, 1);
    double bits = 0.0;
    const size_t raw = (size_t)D * H * W;

    double tenc = 0.0, tdec = 0.0;
    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx) {
        // gather (zero-pad edges)
        memset(cv, 0, CVOX);
        for (u32 z = 0; z < CS; ++z) {
            u32 vz = cz * CS + z; if (vz >= D) break;
            for (u32 y = 0; y < CS; ++y) {
                u32 vy = cy * CS + y; if (vy >= H) break;
                u32 vx0 = cx * CS; u32 w = (vx0 + CS <= W) ? CS : (vx0 < W ? W - vx0 : 0);
                if (w) memcpy(cv + ((size_t)z * CS + y) * CS,
                              vol + (((size_t)vz * H + vy) * W + vx0), w);
            }
        }
        u64 sum = 0; for (size_t i = 0; i < CVOX; ++i) sum += cv[i];
        i32 dc = (i32)((sum + CVOX / 2) / CVOX);

        double a0 = now_sec();
        t->fwd(cc, cv, dc);
        quantize(cc, cq, CVOX, step);
        double a1 = now_sec();
        bits += rate_bits(cq, CVOX);

        double b0 = now_sec();
        dequantize(cq, cc, CVOX, step);
        t->inv(cv, cc, dc);
        double b1 = now_sec();
        tenc += a1 - a0; tdec += b1 - b0;

        // scatter
        for (u32 z = 0; z < CS; ++z) {
            u32 vz = cz * CS + z; if (vz >= D) break;
            for (u32 y = 0; y < CS; ++y) {
                u32 vy = cy * CS + y; if (vy >= H) break;
                u32 vx0 = cx * CS; u32 w = (vx0 + CS <= W) ? CS : (vx0 < W ? W - vx0 : 0);
                if (w) memcpy(rec + (((size_t)vz * H + vy) * W + vx0),
                              cv + ((size_t)z * CS + y) * CS, w);
            }
        }
    }
    *ratio = (double)(raw * 8) / bits;
    vc_compute_metrics(vol, rec, D, H, W, m);
    blocking_metrics(vol, rec, D, H, W, seam, edge);
    if (gmsd) *gmsd = vc_gmsd(vol, rec, D, H, W);
    *enc_mbs = raw / 1e6 / (tenc > 0 ? tenc : 1e-9);
    *dec_mbs = raw / 1e6 / (tdec > 0 ? tdec : 1e-9);
    free(cv); free(cc); free(cq); free(rec);
}

// Binary-search a quant step that hits a target compression ratio (+/- 5%) for a
// given transform, so all transforms are compared at MATCHED ratio.
static f32 step_for_ratio(const transform_t *t, const u8 *vol, u32 D, u32 H, u32 W,
                          double target) {
    f32 lo = 0.5f, hi = 4096.f;
    double ratio; vc_metrics m; double e, d, s, eg;
    for (int it = 0; it < 22; ++it) {
        f32 mid = sqrtf(lo * hi);
        run_transform(t, vol, D, H, W, mid, &ratio, &m, &e, &d, &s, &eg, NULL);
        if (ratio < target) lo = mid; else hi = mid;
    }
    return sqrtf(lo * hi);
}

static void bake_off(const char *label, const u8 *vol, u32 D, u32 H, u32 W) {
    const size_t raw = (size_t)D * H * W;
    printf("\n############ %s : %ux%ux%u (%.1f MB, chunk=%u) ############\n",
           label, D, H, W, raw / 1e6, CS);
    const double targets[] = { 10.0, 20.0, 50.0, 100.0 };
    for (int ti = 0; ti < 4; ++ti) {
        double tg = targets[ti];
        printf("\n--- target ratio ~%.0fx ---\n", tg);
        printf("%-11s | %7s | %8s | %6s | %5s | %8s | %8s | %9s | %9s | %8s | %8s\n",
               "transform","ratio","PSNR","MAE","Linf","MS-SSIM","GMSD","seamStep","edgeMAE","encMB/s","decMB/s");
        for (int i = 0; i < NT; ++i) {
            f32 step = step_for_ratio(&TFORMS[i], vol, D, H, W, tg);
            double ratio, e, d, s, eg, gmsd; vc_metrics m;
            run_transform(&TFORMS[i], vol, D, H, W, step, &ratio, &m, &e, &d, &s, &eg, &gmsd);
            printf("%-11s | %6.1fx | %8.2f | %6.3f | %5.0f | %8.4f | %8.4f | %8.3f | %8.3f | %8.0f | %8.0f\n",
                   TFORMS[i].name, ratio, m.psnr, m.mae, m.linf, m.ms_ssim,
                   gmsd, s, eg, e, d);
        }
    }
}

int main(int argc, char **argv) {
    // --chunks <dir> [N C] : tile N C^3 raw chunks along Z (default C=128).
    if (argc >= 3 && strcmp(argv[1], "--chunks") == 0) {
        const char *dir = argv[2];
        int maxn = (argc >= 4) ? atoi(argv[3]) : 1000000;
        u32 C = (argc >= 5) ? (u32)atoi(argv[4]) : 128;
        DIR *dp = opendir(dir); if (!dp) { fprintf(stderr,"no dir %s\n",dir); return 1; }
        char paths[4096][512]; int np = 0; struct dirent *de;
        while ((de = readdir(dp)) && np < 4096 && np < maxn) {
            size_t l = strlen(de->d_name);
            if (l > 3 && strcmp(de->d_name + l - 3, ".u8") == 0)
                snprintf(paths[np++], 512, "%s/%s", dir, de->d_name);
        }
        closedir(dp);
        if (!np) { fprintf(stderr,"no .u8 in %s\n",dir); return 1; }
        u8 *vol = malloc((size_t)np * C * C * C);
        for (int i = 0; i < np; ++i) {
            size_t len; u8 *b = read_file(paths[i], &len);
            if (!b || len != (size_t)C*C*C) { fprintf(stderr,"bad chunk %s\n",paths[i]); return 1; }
            memcpy(vol + (size_t)i*C*C*C, b, len); free(b);
        }
        char lab[96]; snprintf(lab,sizeof lab,"%d x %u^3 chunks",np,C);
        bake_off(lab, vol, (u32)np*C, C, C); free(vol); return 0;
    }
    if (argc >= 5 && argv[1][0] != '-') {
        u32 D = atoi(argv[2]), H = atoi(argv[3]), W = atoi(argv[4]);
        size_t len; u8 *vol = read_file(argv[1], &len);
        if (!vol || len < (size_t)D*H*W) { fprintf(stderr,"bad raw\n"); return 1; }
        bake_off(argv[1], vol, D, H, W); free(vol); return 0;
    }
    fprintf(stderr, "usage: bench_transforms <raw.u8> D H W | --chunks <dir> [N C]\n");
    return 1;
}
