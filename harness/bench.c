// Phase-0 benchmark harness (PLAN §3 item 6). Loads a raw u8 volume (or a set of
// 128^3 chunks), encodes -> decodes at several q values, and prints a table of
// compression ratio, PSNR/MAE/L-inf/p99/MS-SSIM, and encode/decode MB/s.
//
// Usage:
//   bench <raw.u8> <dz> <dy> <dx>           # one raw volume of given shape
//   bench --chunks <dir> [N]                # N (default 128^3) chunks tiled in
//                                           # a row along Z from *.u8 in <dir>
// If no args, a synthetic structured volume is generated so the pipeline still
// demonstrates end-to-end without network/data.
#include "../include/vc/vc.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dirent.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static u8 *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    u8 *b = (u8 *)malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); *len = (size_t)sz; return b;
}

// Synthetic structured volume: smooth gradients + sinusoidal "fibers" + a few
// sharp planes, so the codec exercises both low- and high-frequency content.
static u8 *make_synthetic(u32 d, u32 h, u32 w) {
    u8 *v = (u8 *)malloc((size_t)d * h * w);
    for (u32 z = 0; z < d; ++z)
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        double f = 80.0
            + 40.0 * sin(x * 0.12) * cos(y * 0.09)
            + 25.0 * sin((x + y + z) * 0.05)
            + 30.0 * ((((x / 13) + (y / 11) + (z / 17)) & 1) ? 1 : 0);
        int iv = (int)(f + 0.5);
        v[((size_t)z * h + y) * w + x] = (u8)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
    }
    return v;
}

static void run_one(const char *label, const u8 *vol, u32 d, u32 h, u32 w) {
    const size_t raw = (size_t)d * h * w;
    printf("\n== %s : %ux%ux%u (%.1f MB raw, chunk=%u) ==\n",
           label, d, h, w, raw / 1e6, vc_chunk_side());
    printf("   q |   ratio |  PSNR(dB) |    MAE |  Linf |  p99 | MS-SSIM | enc MB/s | dec MB/s\n");
    printf("-----+---------+-----------+--------+-------+------+---------+----------+---------\n");
    const float qs[] = {2.f, 4.f, 8.f, 16.f, 32.f, 64.f};
    for (int qi = 0; qi < (int)(sizeof(qs)/sizeof(qs[0])); ++qi) {
        float q = qs[qi];
        u8 *arc = NULL; size_t alen = 0;
        double t0 = now_sec();
        if (vc_encode_volume(vol, d, h, w, q, &arc, &alen)) { printf("encode failed\n"); return; }
        double t1 = now_sec();
        u8 *rec = NULL; u32 rd, rh, rw;
        if (vc_decode_volume(arc, alen, &rec, &rd, &rh, &rw)) { printf("decode failed\n"); free(arc); return; }
        double t2 = now_sec();

        vc_metrics m; vc_compute_metrics(vol, rec, d, h, w, &m);
        double ratio = (double)raw / (double)alen;
        double enc_mbs = raw / 1e6 / (t1 - t0);
        double dec_mbs = raw / 1e6 / (t2 - t1);
        printf("%4.0f | %7.1f | %9.2f | %6.3f | %5.0f | %4.0f | %7.4f | %8.0f | %8.0f\n",
               q, ratio, m.psnr, m.mae, m.linf, m.p99, m.ms_ssim, enc_mbs, dec_mbs);
        free(arc); free(rec);
    }
}

int main(int argc, char **argv) {
    if (argc >= 5 && argv[1][0] != '-') {
        u32 d = (u32)atoi(argv[2]), h = (u32)atoi(argv[3]), w = (u32)atoi(argv[4]);
        size_t len; u8 *vol = read_file(argv[1], &len);
        if (!vol || len < (size_t)d * h * w) { fprintf(stderr, "bad raw file/size\n"); return 1; }
        run_one(argv[1], vol, d, h, w);
        free(vol);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "--chunks") == 0) {
        const char *dir = argv[2];
        int maxn = (argc >= 4) ? atoi(argv[3]) : 1000000;
        // Collect *.u8 files (each 128^3), tile along Z into one volume.
        DIR *dp = opendir(dir);
        if (!dp) { fprintf(stderr, "cannot open dir %s\n", dir); return 1; }
        char paths[4096][512]; int np = 0;
        struct dirent *de;
        while ((de = readdir(dp)) && np < 4096 && np < maxn) {
            size_t l = strlen(de->d_name);
            if (l > 3 && strcmp(de->d_name + l - 3, ".u8") == 0)
                snprintf(paths[np++], 512, "%s/%s", dir, de->d_name);
        }
        closedir(dp);
        if (np == 0) { fprintf(stderr, "no .u8 chunks in %s\n", dir); return 1; }
        const u32 C = 128;
        u8 *vol = (u8 *)malloc((size_t)np * C * C * C);
        for (int i = 0; i < np; ++i) {
            size_t len; u8 *b = read_file(paths[i], &len);
            if (!b || len != (size_t)C * C * C) { fprintf(stderr, "bad chunk %s (%zu)\n", paths[i], b?len:0); return 1; }
            memcpy(vol + (size_t)i * C * C * C, b, len);
            free(b);
        }
        char label[64]; snprintf(label, sizeof(label), "%d high-res 128^3 chunks", np);
        run_one(label, vol, (u32)np * C, C, C);
        free(vol);
        return 0;
    }
    // No data: synthetic fallback.
    fprintf(stderr, "[no input given -> synthetic structured volume]\n");
    u32 d = 256, h = 256, w = 256;
    u8 *vol = make_synthetic(d, h, w);
    run_one("synthetic", vol, d, h, w);
    free(vol);
    return 0;
}
