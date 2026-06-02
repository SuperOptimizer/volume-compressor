// Round-trip + archive integrity self-test (PLAN §3 item 3 "must round-trip").
#include "../include/vc/vc.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fail = 1; } \
                              else printf("ok: %s\n", msg); } while (0)

static u8 *gen(u32 d, u32 h, u32 w, int kind) {
    u8 *v = (u8 *)malloc((size_t)d * h * w);
    for (u32 z = 0; z < d; ++z)
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        int val;
        if (kind == 0) val = (int)(128 + 60 * sin(x * 0.1) * cos(y * 0.07 + z * 0.05));
        else if (kind == 1) val = 42;                        // uniform
        else val = (int)((x * 7 + y * 13 + z * 3) & 0xff);   // high-frequency
        v[((size_t)z * h + y) * w + x] = (u8)(val < 0 ? 0 : (val > 255 ? 255 : val));
    }
    return v;
}

static int roundtrip(u32 d, u32 h, u32 w, int kind, float q, const char *name) {
    u8 *vol = gen(d, h, w, kind);
    u8 *arc = NULL; size_t alen = 0;
    if (vc_encode_volume(vol, d, h, w, q, &arc, &alen)) { printf("FAIL encode %s\n", name); free(vol); return 1; }
    u8 *rec = NULL; u32 rd, rh, rw;
    int rc = vc_decode_volume(arc, alen, &rec, &rd, &rh, &rw);
    if (rc) { printf("FAIL decode %s rc=%d\n", name, rc); free(vol); free(arc); return 1; }
    int shape_ok = (rd == d && rh == h && rw == w);
    vc_metrics m; vc_compute_metrics(vol, rec, d, h, w, &m);
    printf("  %-22s q=%4.0f shape_ok=%d PSNR=%.2f Linf=%.0f ratio=%.1f\n",
           name, q, shape_ok, m.psnr, m.linf, (double)((size_t)d*h*w)/alen);
    // Uniform must be lossless; others must clear a sane PSNR floor.
    int qual_ok = (kind == 1) ? (m.linf == 0.0) : (m.psnr > 25.0);
    free(vol); free(arc); free(rec);
    return (shape_ok && qual_ok) ? 0 : 1;
}

int main(void) {
    printf("chunk side = %u\n", vc_chunk_side());
    // Exact multiple of chunk side.
    CHECK(roundtrip(128, 128, 128, 0, 8.f, "smooth-aligned") == 0, "smooth aligned roundtrip");
    // Non-multiple shape -> edge padding path.
    CHECK(roundtrip(100, 70, 53, 0, 8.f, "smooth-unaligned") == 0, "edge-padding roundtrip");
    // Uniform chunk fast-path (lossless).
    CHECK(roundtrip(64, 64, 64, 1, 16.f, "uniform") == 0, "uniform lossless");
    // High-frequency stress.
    CHECK(roundtrip(96, 96, 96, 2, 4.f, "high-freq") == 0, "high-freq roundtrip");
    // Tiny volume (smaller than a chunk).
    CHECK(roundtrip(10, 12, 9, 0, 8.f, "tiny") == 0, "sub-chunk roundtrip");

    printf(fail ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return fail;
}
