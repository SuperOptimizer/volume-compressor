// Chunk-size sweep worker (PLAN §5 Phase-1 item 3, §2 "Chunk size").
//
// Loads a real PHerc Paris 4 u8 volume (high-res 128^3 chunks tiled, OR a coarse
// downscaled sub-region), encodes->decodes at a list of q values with the
// COMPILE-TIME-selected pipeline (chunk side = VC_CHUNK_SIDE), and prints one TSV
// row per (dataset,chunk,q) with: ratio, PSNR, MS-SSIM, GMSD, enc MB/s, dec MB/s,
// chunk-face count, and mean boundary-step magnitude (the blocking story).
//
// This file is OWNED by the metrics/sweep agent; it does NOT touch bench.c. It is
// recompiled-and-relinked per chunk size by harness/sweep_chunksize.sh.
#define _POSIX_C_SOURCE 199309L   // clock_gettime / CLOCK_MONOTONIC under -std=c23
#include "../include/vc/vc.h"
#include "../src/metrics/metrics.h"
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
    u8 *b = (u8 *)malloc(sz);
    if (!b || fread(b, 1, sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); *len = (size_t)sz; return b;
}

// EXCESS boundary step = the codec-INDUCED blocking/DC-banding at chunk faces,
// isolated from any discontinuity already present in the original at the same
// plane. For each chunk-face plane (at multiples of `side` along each axis) we
// average |step_rec - step_orig|, where step_* = |voxel(k) - voxel(k-1)|. A
// perfectly seam-free codec yields 0 here even where the *source* has real edges
// landing on a face; a DC-banding codec adds a positive excess step. This is the
// honest "blocking story" number (raw rec-step alone is confounded by where faces
// happen to land vs. true content edges — critical on this data, whose 8 source
// chunks have ~25-level DC offsets at the 128-boundaries).
static double boundary_step(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx,
                            u32 side, u64 *out_faces) {
    f64 acc = 0.0; u64 cnt = 0; u64 faces = 0;
    for (u32 z = side; z < dz; z += side) {           // Z faces
        faces++;
        const u8 *ra = ref + (size_t)(z - 1) * dy * dx, *rb = ref + (size_t)z * dy * dx;
        const u8 *ca = rec + (size_t)(z - 1) * dy * dx, *cb = rec + (size_t)z * dy * dx;
        for (size_t i = 0; i < (size_t)dy * dx; ++i) {
            i32 so = (i32)rb[i] - (i32)ra[i]; if (so < 0) so = -so;
            i32 sr = (i32)cb[i] - (i32)ca[i]; if (sr < 0) sr = -sr;
            i32 e = sr - so; acc += e < 0 ? -e : e; cnt++;
        }
    }
    for (u32 y = side; y < dy; y += side) {           // Y faces
        faces++;
        for (u32 z = 0; z < dz; ++z) {
            const u8 *r0 = ref + ((size_t)z * dy + (y - 1)) * dx, *r1 = ref + ((size_t)z * dy + y) * dx;
            const u8 *c0 = rec + ((size_t)z * dy + (y - 1)) * dx, *c1 = rec + ((size_t)z * dy + y) * dx;
            for (u32 x = 0; x < dx; ++x) {
                i32 so = (i32)r1[x] - (i32)r0[x]; if (so < 0) so = -so;
                i32 sr = (i32)c1[x] - (i32)c0[x]; if (sr < 0) sr = -sr;
                i32 e = sr - so; acc += e < 0 ? -e : e; cnt++;
            }
        }
    }
    for (u32 x = side; x < dx; x += side) {           // X faces
        faces++;
        for (u32 z = 0; z < dz; ++z)
        for (u32 y = 0; y < dy; ++y) {
            const u8 *rr = ref + ((size_t)z * dy + y) * dx;
            const u8 *cc = rec + ((size_t)z * dy + y) * dx;
            i32 so = (i32)rr[x] - (i32)rr[x - 1]; if (so < 0) so = -so;
            i32 sr = (i32)cc[x] - (i32)cc[x - 1]; if (sr < 0) sr = -sr;
            i32 e = sr - so; acc += e < 0 ? -e : e; cnt++;
        }
    }
    *out_faces = faces;
    return cnt ? acc / (f64)cnt : 0.0;
}

// Total number of chunk faces in the (dz,dy,dx) volume gridded at `side`.
static u64 chunk_face_count(u32 dz, u32 dy, u32 dx, u32 side) {
    u64 nz = (dz + side - 1) / side, ny = (dy + side - 1) / side, nx = (dx + side - 1) / side;
    // internal faces along each axis = (n-1) per the other two axes' chunk grid
    return (nz - 1) * ny * nx + (ny - 1) * nz * nx + (nx - 1) * nz * ny;
}

static void run(const char *dataset, const u8 *vol, u32 d, u32 h, u32 w,
                const float *qs, int nq) {
    const size_t raw = (size_t)d * h * w;
    const u32 side = vc_chunk_side();
    for (int i = 0; i < nq; ++i) {
        float q = qs[i];
        u8 *arc = NULL; size_t alen = 0;
        double t0 = now_sec();
        if (vc_encode_volume(vol, d, h, w, q, &arc, &alen)) { fprintf(stderr, "encode fail\n"); return; }
        double t1 = now_sec();
        u8 *rec = NULL; u32 rd, rh, rw;
        if (vc_decode_volume(arc, alen, &rec, &rd, &rh, &rw)) { fprintf(stderr, "decode fail\n"); free(arc); return; }
        double t2 = now_sec();

        vc_metrics m; vc_compute_metrics(vol, rec, d, h, w, &m);
        double gmsd = vc_gmsd(vol, rec, d, h, w);
        u64 faces = 0; double bstep = boundary_step(vol, rec, d, h, w, side, &faces);
        (void)chunk_face_count;
        double ratio = (double)raw / (double)alen;
        double enc = raw / 1e6 / (t1 - t0);
        double dec = raw / 1e6 / (t2 - t1);
        // TSV: dataset chunk q ratio psnr msssim gmsd encMBs decMBs faces bstep
        printf("%s\t%u\t%.0f\t%.2f\t%.2f\t%.4f\t%.5f\t%.0f\t%.0f\t%llu\t%.3f\n",
               dataset, side, q, ratio, m.psnr, m.ms_ssim, gmsd, enc, dec,
               (unsigned long long)faces, bstep);
        free(arc); free(rec);
    }
}

int main(int argc, char **argv) {
    // argv[1] = mode: "hires" <dir>  | "coarse" <file> <d> <h> <w>
    // remaining args = q values to sweep.
    if (argc < 3) { fprintf(stderr, "usage: sweep_metrics hires <dir> q...| coarse <file> d h w q...\n"); return 2; }
    const char *mode = argv[1];
    if (strcmp(mode, "hires") == 0) {
        const char *dir = argv[2];
        static const char *names[8] = {
            "hr_296_127_127.u8","hr_296_127_128.u8","hr_296_128_127.u8","hr_296_128_128.u8",
            "hr_297_127_127.u8","hr_297_127_128.u8","hr_297_128_127.u8","hr_297_128_128.u8"};
        const u32 C = 128;
        u8 *vol = (u8 *)malloc((size_t)8 * C * C * C);
        for (int i = 0; i < 8; ++i) {
            char p[1024]; snprintf(p, sizeof(p), "%s/%s", dir, names[i]);
            size_t l; u8 *b = read_file(p, &l);
            if (!b || l != (size_t)C * C * C) { fprintf(stderr, "bad hires chunk %s\n", p); return 1; }
            memcpy(vol + (size_t)i * C * C * C, b, l); free(b);
        }
        int nq = argc - 3; float *qs = malloc(sizeof(float) * (nq > 0 ? nq : 1));
        for (int i = 0; i < nq; ++i) qs[i] = (float)atof(argv[3 + i]);
        run("hires", vol, 8 * C, C, C, qs, nq);
        free(vol); free(qs);
        return 0;
    }
    if (strcmp(mode, "coarse") == 0) {
        const char *file = argv[2];
        u32 d = (u32)atoi(argv[3]), h = (u32)atoi(argv[4]), w = (u32)atoi(argv[5]);
        size_t l; u8 *vol = read_file(file, &l);
        if (!vol || l < (size_t)d * h * w) { fprintf(stderr, "bad coarse file\n"); return 1; }
        int nq = argc - 6; float *qs = malloc(sizeof(float) * (nq > 0 ? nq : 1));
        for (int i = 0; i < nq; ++i) qs[i] = (float)atof(argv[6 + i]);
        run("coarse", vol, d, h, w, qs, nq);
        free(vol); free(qs);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
