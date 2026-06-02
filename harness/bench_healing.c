// Boundary-healing bake-off harness (PLAN §2 healing row + §4 metrics). On real
// PHerc Paris 4 hires-256 + coarse-256 (assembled from the cached chunks, same
// as compare_c3d_c4d.py), it encodes/decodes our winning stack at a target
// ratio, then applies each healing config to the DECODED volume as a pure
// post-filter and reports, vs the ORIGINAL:
//
//   GMSD       (blocking/smear metric -- should DROP with healing)
//   HaarPSI    (edge perceptual -- should rise/hold)
//   seam-step  (mean |delta| across 16^3 sub-block + chunk faces -- should DROP)
//   PSNR, MS-SSIM (must NOT drop materially -- verify no over-smoothing)
//   heal MB/s  (decode-side filter throughput cost)
//
// The healing filter runs on the reconstructed volume, so we exercise it as a
// post-process here without touching the codec internals.
//
// Usage:
//   bench_healing <raw.u8> <dz> <dy> <dx> <label> [q ...]
//   bench_healing               # synthetic fallback (no data)
#include "../include/vc/vc.h"
#include "../src/metrics/metrics.h"
#include "../src/healing/healing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static u8 *read_file(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    u8 *b = malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); *len = (size_t)sz; return b;
}

#define BLK 16u
// Mean |step| across every interior 16-grid (= sub-block AND chunk) face, and
// separately across chunk faces only (multiples of the codec chunk side).
static void seam_steps(const u8 *v, u32 dz, u32 dy, u32 dx, u32 cs,
                       double *all16, double *chunk) {
    double s16 = 0, sc = 0; u64 c16 = 0, cc = 0;
    size_t sX = 1, sY = dx, sZ = (size_t)dy * dx;
    for (u32 n = BLK; n < dz; n += BLK)
        for (u32 y = 0; y < dy; ++y) for (u32 x = 0; x < dx; ++x) {
            size_t i = (size_t)(n-1)*sZ + y*sY + x*sX;
            double d = fabs((double)v[i] - v[i+sZ]);
            s16 += d; c16++; if (n % cs == 0) { sc += d; cc++; }
        }
    for (u32 n = BLK; n < dy; n += BLK)
        for (u32 z = 0; z < dz; ++z) for (u32 x = 0; x < dx; ++x) {
            size_t i = (size_t)z*sZ + (n-1)*sY + x*sX;
            double d = fabs((double)v[i] - v[i+sY]);
            s16 += d; c16++; if (n % cs == 0) { sc += d; cc++; }
        }
    for (u32 n = BLK; n < dx; n += BLK)
        for (u32 z = 0; z < dz; ++z) for (u32 y = 0; y < dy; ++y) {
            size_t i = (size_t)z*sZ + y*sY + (n-1)*sX;
            double d = fabs((double)v[i] - v[i+sX]);
            s16 += d; c16++; if (n % cs == 0) { sc += d; cc++; }
        }
    *all16 = c16 ? s16/c16 : 0; *chunk = cc ? sc/cc : 0;
}

typedef enum { H_NONE, H_DEBLOCK, H_CDEF } heal_kind;

static void apply_heal(heal_kind k, u8 *v, u32 d, u32 h, u32 w, f32 step, f32 str) {
    switch (k) {
        case H_NONE:    vc_heal_none(v, d, h, w, step, str); break;
        case H_DEBLOCK: vc_heal_deblock(v, d, h, w, step, str); break;
        case H_CDEF:    vc_heal_cdef(v, d, h, w, step, str); break;
    }
}

static void run_one(const char *label, const u8 *vol, u32 d, u32 h, u32 w) {
    const size_t raw = (size_t)d * h * w;
    const u32 cs = vc_chunk_side();
    printf("\n=== %s : %ux%ux%u (chunk=%u, sub-block=%u) ===\n", label, d, h, w, cs, BLK);

    // Pick q values spanning the target ~20-50x ratio band.
    const float qs[] = {16.f, 32.f, 64.f};
    for (int qi = 0; qi < (int)(sizeof(qs)/sizeof(qs[0])); ++qi) {
        float q = qs[qi];
        u8 *arc = NULL; size_t alen = 0;
        if (vc_encode_volume(vol, d, h, w, q, &arc, &alen)) { printf("encode fail\n"); return; }
        u8 *base = NULL; u32 rd, rh, rw;
        if (vc_decode_volume(arc, alen, &base, &rd, &rh, &rw)) { printf("decode fail\n"); free(arc); return; }
        double ratio = (double)raw / (double)alen;
        // The dead-zone step the chunks were coded at == q (fixed-q RC).
        f32 step = q;

        printf("\n-- q=%.0f  ratio=%.1fx --\n", q, ratio);
        printf("  %-16s | PSNR(dB) | MS-SSIM |   GMSD  | HaarPSI | seam16 | seamChk | heal MB/s\n", "config");
        printf("  -----------------+----------+---------+---------+---------+--------+---------+----------\n");

        struct { const char *name; heal_kind k; float str; } cfgs[] = {
            { "none",          H_NONE,    0.0f },
            { "deblock@0.3",   H_DEBLOCK, 0.3f },
            { "deblock@0.5",   H_DEBLOCK, 0.5f },
            { "deblock@0.8",   H_DEBLOCK, 0.8f },
            { "deblock@1.0",   H_DEBLOCK, 1.0f },
            { "cdef@0.5",      H_CDEF,    0.5f },
            { "cdef@1.0",      H_CDEF,    1.0f },
        };
        u8 *work = malloc(raw);
        for (int ci = 0; ci < (int)(sizeof(cfgs)/sizeof(cfgs[0])); ++ci) {
            memcpy(work, base, raw);
            double t0 = now_sec();
            apply_heal(cfgs[ci].k, work, d, h, w, step, cfgs[ci].str);
            double t1 = now_sec();
            vc_metrics m; vc_compute_metrics(vol, work, d, h, w, &m);
            double gmsd = vc_gmsd(vol, work, d, h, w);
            double haar = vc_haarpsi(vol, work, d, h, w);
            double s16, sc16; seam_steps(work, d, h, w, cs, &s16, &sc16);
            double heal_mbs = (cfgs[ci].k == H_NONE) ? 0.0 : raw/1e6/(t1-t0 > 1e-9 ? t1-t0 : 1e-9);
            printf("  %-16s | %8.2f | %7.4f | %7.4f | %7.4f | %6.3f | %7.3f | %8.0f\n",
                   cfgs[ci].name, m.psnr, m.ms_ssim, gmsd, haar, s16, sc16, heal_mbs);
        }
        free(work); free(base); free(arc);
    }
}

static u8 *make_synth(u32 d, u32 h, u32 w) {
    u8 *v = malloc((size_t)d*h*w);
    for (u32 z=0; z<d; ++z) for (u32 y=0; y<h; ++y) for (u32 x=0; x<w; ++x) {
        double f = 80 + 40*sin(x*0.12)*cos(y*0.09) + 25*sin((x+y+z)*0.05);
        int iv = (int)(f+0.5);
        v[((size_t)z*h+y)*w+x] = (u8)(iv<0?0:iv>255?255:iv);
    }
    return v;
}

int main(int argc, char **argv) {
    if (argc >= 6 && argv[1][0] != '-') {
        u32 d = atoi(argv[2]), h = atoi(argv[3]), w = atoi(argv[4]);
        size_t len; u8 *vol = read_file(argv[1], &len);
        if (!vol || len < (size_t)d*h*w) { fprintf(stderr, "bad raw\n"); return 1; }
        run_one(argv[5], vol, d, h, w);
        free(vol); return 0;
    }
    fprintf(stderr, "[no input -> synthetic]\n");
    u32 d=256,h=256,w=256; u8 *v = make_synth(d,h,w);
    run_one("synthetic", v, d, h, w); free(v); return 0;
}
