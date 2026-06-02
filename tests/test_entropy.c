// Entropy-coder round-trip unit tests (PLAN §7 "add tests for new coders").
// Exercises rice, rlgr, and rans on signed-level arrays designed to stress the
// zero-run, escape, and adaptation paths. Each coder must reproduce the input
// EXACTLY. Independent of the DCT/quant pipeline.
#include "../include/vc/types.h"
#include "../src/blocks.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;

typedef size_t (*enc_fn)(u8 *restrict, size_t, const i16 *restrict, size_t);
typedef void   (*dec_fn)(i16 *restrict, size_t, const u8 *restrict, size_t);

static void check_q(const char *coder, const char *what,
                    enc_fn enc, dec_fn dec, const i16 *q, size_t n, int quiet) {
    size_t cap = n * 8 + 4096;
    u8 *buf = (u8 *)malloc(cap);
    i16 *out = (i16 *)malloc(n * sizeof(i16) + 16);
    size_t len = enc(buf, cap, q, n);
    if (len > cap) { printf("FAIL %s/%s: encode overflow\n", coder, what); fail = 1; goto done; }
    memset(out, 0xAB, n * sizeof(i16));
    dec(out, n, buf, len);
    for (size_t i = 0; i < n; ++i)
        if (out[i] != q[i]) {
            printf("FAIL %s/%s: mismatch at %zu (got %d want %d), len=%zu\n",
                   coder, what, i, out[i], q[i], len);
            fail = 1; goto done;
        }
    if (!quiet)
        printf("ok   %-5s %-18s n=%-6zu bytes=%-6zu (%.3f b/sym)\n",
               coder, what, n, len, 8.0 * len / (double)n);
done:
    free(buf); free(out);
}
static void check(const char *coder, const char *what,
                  enc_fn enc, dec_fn dec, const i16 *q, size_t n) {
    check_q(coder, what, enc, dec, q, n, 0);
}

static void run_coder(const char *name, enc_fn enc, dec_fn dec) {
    const size_t N = 8192;
    i16 *q = (i16 *)malloc(N * sizeof(i16));

    // 1) all zeros (the extreme zero-run case)
    memset(q, 0, N * sizeof(i16));
    check(name, "all-zero", enc, dec, q, N);

    // 2) long zero runs with sparse spikes (HF-after-scan profile)
    memset(q, 0, N * sizeof(i16));
    for (size_t i = 0; i < N; i += 137) q[i] = (i16)((i % 7) - 3);
    check(name, "sparse-spikes", enc, dec, q, N);

    // 3) dense small magnitudes (DC/low-freq profile)
    for (size_t i = 0; i < N; ++i) q[i] = (i16)(((int)(i * 2654435761u >> 24) % 5) - 2);
    check(name, "dense-small", enc, dec, q, N);

    // 4) full-range incl. large magnitudes (rANS escape path)
    for (size_t i = 0; i < N; ++i) {
        int v = (int)((i * 1103515245u + 12345u) >> 16) % 2000 - 1000;
        q[i] = (i16)v;
    }
    check(name, "full-range", enc, dec, q, N);

    // 5) extremes at the i16 boundaries
    for (size_t i = 0; i < N; ++i) q[i] = (i & 1) ? (i16)32767 : (i16)-32768;
    check(name, "i16-extremes", enc, dec, q, N);

    // 6) tiny lengths and odd boundaries
    for (size_t n = 1; n <= 70; ++n) {
        for (size_t i = 0; i < n; ++i) q[i] = (i16)(((int)(i * 7 + 1) % 11) - 5);
        char w[32]; snprintf(w, sizeof(w), "tiny-n%zu", n);
        check_q(name, w, enc, dec, q, n, 1);
    }
    printf("ok   %-5s tiny-n1..70        (all exact)\n", name);
    free(q);
}

int main(void) {
    run_coder("rice", vc_rice_encode, vc_rice_decode);
    run_coder("rlgr", vc_rlgr_encode, vc_rlgr_decode);
    run_coder("rans", vc_rans_encode, vc_rans_decode);
    printf(fail ? "\nSOME ENTROPY TESTS FAILED\n" : "\nALL ENTROPY TESTS PASSED\n");
    return fail;
}
