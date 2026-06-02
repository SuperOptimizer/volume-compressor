// compare_c3d_c4d — the validation backbone (PLAN §0.5). Runs the REFERENCE
// codec c3d and OUR configured codec on the SAME raw u8 volume across a q sweep,
// emitting one machine-readable row per (codec,q) with ratio / PSNR / MS-SSIM /
// GMSD / enc-MB/s / dec-MB/s. c4d is driven separately by the Python wrapper
// (it has a clean CLI); this binary covers OURS + c3d (a C library, no CLI).
//
// Build: linked against libvc (our codec) and libc3d.a (the reference). The
// CMake/standalone build passes -DHAVE_C3D when c3d is available; without it,
// only our codec is benchmarked (graceful per PLAN: "if a reference won't build,
// note it and continue").
//
// Usage: compare_c3d_c4d <raw.u8> <dz> <dy> <dx> <label> <q...>
// Output (stdout), one row per line, TSV:
//   codec<TAB>label<TAB>q<TAB>ratio<TAB>psnr<TAB>ms_ssim<TAB>gmsd<TAB>enc_mbs<TAB>dec_mbs<TAB>tag
// where `tag` identifies the active config (entropy+qweight) for our codec.
#define _POSIX_C_SOURCE 199309L   // clock_gettime / CLOCK_MONOTONIC under -std=c23
#include "../include/vc/vc.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef HAVE_C3D
#include "c3d.h"
#endif

#ifndef VC_TAG_STR
#define VC_TAG_STR "ours"
#endif

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

static void emit(const char *codec, const char *label, double q, double ratio,
                 const u8 *ref, const u8 *rec, u32 d, u32 h, u32 w,
                 double enc_mbs, double dec_mbs, const char *tag) {
    vc_metrics m; vc_compute_metrics(ref, rec, d, h, w, &m);
    double gmsd = vc_gmsd(ref, rec, d, h, w);
    printf("%s\t%s\t%.4g\t%.2f\t%.3f\t%.4f\t%.4f\t%.1f\t%.1f\t%s\n",
           codec, label, q, ratio, m.psnr, m.ms_ssim, gmsd, enc_mbs, dec_mbs, tag);
    fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <raw.u8> <dz> <dy> <dx> <label> <q...>\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    u32 d = (u32)atoi(argv[2]), h = (u32)atoi(argv[3]), w = (u32)atoi(argv[4]);
    const char *label = argv[5];
    size_t len; u8 *vol = read_file(path, &len);
    if (!vol || len < (size_t)d * h * w) { fprintf(stderr, "bad raw %s\n", path); return 1; }
    const size_t raw = (size_t)d * h * w;

    for (int qi = 6; qi < argc; ++qi) {
        double q = atof(argv[qi]);

        // --- OUR codec ---
        {
            u8 *arc = NULL; size_t alen = 0;
            double t0 = now_sec();
            if (vc_encode_volume(vol, d, h, w, (f32)q, &arc, &alen)) { fprintf(stderr, "ours enc fail\n"); continue; }
            double t1 = now_sec();
            u8 *rec = NULL; u32 rd, rh, rw;
            if (vc_decode_volume(arc, alen, &rec, &rd, &rh, &rw)) { fprintf(stderr, "ours dec fail\n"); free(arc); continue; }
            double t2 = now_sec();
            emit("ours", label, q, (double)raw / (double)alen, vol, rec, d, h, w,
                 raw / 1e6 / (t1 - t0), raw / 1e6 / (t2 - t1), VC_TAG_STR);
            free(arc); free(rec);
        }

#ifdef HAVE_C3D
        // --- c3d reference (256^3 chunk atom; encode_at_q over chunk grid) ---
        // c3d's codec atom is a fixed 256^3 chunk. We tile the volume into 256^3
        // chunks (zero-padded), encode each at q, decode, scatter back, then
        // score the full volume. NOTE: c3d's q is an INVERSE quantizer scalar
        // (small q == high quality, [2^-6, 2^12]) on a different scale than our
        // dead-zone step. We therefore sweep c3d over its OWN native q grid
        // (mapped from our q index) so its rate-distortion curve is sampled
        // across the same 5-200x band, then compare ratio-at-PSNR (not at equal
        // q). The mapping below was calibrated on PHerc Paris 4 (see report).
        {
            // map our sweep index to a c3d-native q spanning ~0.015..0.5
            static const double C3DQ[] = {0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0};
            int idx = qi - 6; if (idx > 6) idx = 6;
            double cq = C3DQ[idx];
            const u32 C = C3D_CHUNK_SIDE;            // 256
            u32 ncz = (d + C - 1) / C, ncy = (h + C - 1) / C, ncx = (w + C - 1) / C;
            // c3d requires 32-byte-aligned 256^3 voxel buffers (C3D_ALIGN).
            u8 *cin  = (u8 *)aligned_alloc(32, (size_t)C * C * C);
            u8 *cout = (u8 *)aligned_alloc(32, (size_t)C * C * C);
            u8 *enc  = (u8 *)malloc(c3d_chunk_encode_max_size());
            u8 *rec  = (u8 *)calloc(raw, 1);
            size_t total_enc = 0;
            double enc_t = 0, dec_t = 0;
            for (u32 cz = 0; cz < ncz; ++cz)
            for (u32 cy = 0; cy < ncy; ++cy)
            for (u32 cx = 0; cx < ncx; ++cx) {
                memset(cin, 0, (size_t)C * C * C);
                for (u32 z = 0; z < C && cz*C+z < d; ++z)
                for (u32 y = 0; y < C && cy*C+y < h; ++y) {
                    u32 vz = cz*C+z, vy = cy*C+y, ox = cx*C;
                    u32 cw = (ox + C <= w) ? C : (ox < w ? w - ox : 0);
                    if (cw) memcpy(cin + ((size_t)z*C+y)*C,
                                   vol + (((size_t)vz*h+vy)*w+ox), cw);
                }
                double a = now_sec();
                size_t el = c3d_chunk_encode_at_q(cin, (float)cq, enc, c3d_chunk_encode_max_size());
                double b = now_sec();
                c3d_chunk_decode(enc, el, cout);
                double c = now_sec();
                enc_t += b - a; dec_t += c - b; total_enc += el;
                for (u32 z = 0; z < C && cz*C+z < d; ++z)
                for (u32 y = 0; y < C && cy*C+y < h; ++y) {
                    u32 vz = cz*C+z, vy = cy*C+y, ox = cx*C;
                    u32 cw = (ox + C <= w) ? C : (ox < w ? w - ox : 0);
                    if (cw) memcpy(rec + (((size_t)vz*h+vy)*w+ox),
                                   cout + ((size_t)z*C+y)*C, cw);
                }
            }
            emit("c3d", label, cq, (double)raw / (double)total_enc, vol, rec, d, h, w,
                 raw / 1e6 / enc_t, raw / 1e6 / dec_t, "ref-c3d");
            free(cin); free(cout); free(enc); free(rec);
        }
#endif
    }
    free(vol);
    return 0;
}
