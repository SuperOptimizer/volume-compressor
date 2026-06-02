// Rate-control allocator self-test (PLAN §2 rate-control row). Verifies:
//   1. The allocator HITS a requested global ratio within tolerance (per-chunk).
//   2. The allocator HITS a requested global ratio (per-16^3-block q-field).
//   3. The per-unit step DISTRIBUTION varies on a mixed (air + structure)
//      volume — easy units get coarse steps, busy units stay fine (the headline
//      "easy chunks 20x / hard chunks 5x" behaviour).
//   4. Parseval and true-MSE distortion estimators both produce valid hits and
//      the divergence is reported (sanity bound, not exactness).
//   5. The analytical Laplacian model also converges to a target.
#include "../include/vc/vc.h"
#include "../src/ratectrl/ratectrl.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int fail = 0;
#define CHECK(c, m) do { if (!(c)) { printf("FAIL: %s\n", m); fail = 1; } else printf("ok: %s\n", m); } while (0)

// A mixed volume: roughly half "air" (near-constant) + half structured
// (gradients, fibers, sharp planes) so the allocator faces both easy and hard
// units — the air+ink mix the q-field is meant to exploit.
static u8 *make_mixed(u32 d, u32 h, u32 w) {
    u8 *v = (u8 *)malloc((size_t)d * h * w);
    for (u32 z = 0; z < d; ++z)
    for (u32 y = 0; y < h; ++y)
    for (u32 x = 0; x < w; ++x) {
        double val;
        if (x < w / 2) {
            // air: flat with a tiny bit of noise
            val = 12.0 + ((x * 3 + y * 5 + z * 7) % 3);
        } else {
            // structured: gradients + fibers + sharp planes
            val = 110.0 + 50.0 * sin(x * 0.20) * cos(y * 0.15)
                + 30.0 * sin((x + y + z) * 0.08)
                + 40.0 * ((((x / 5) + (z / 7)) & 1) ? 1 : 0);
        }
        int iv = (int)(val + 0.5);
        v[((size_t)z * h + y) * w + x] = (u8)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
    }
    return v;
}

static int hits(const u8 *vol, u32 d, u32 h, u32 w, vc_rc_gran gran,
                vc_rc_dist dist, vc_rc_model model, f64 target, const char *tag) {
    vc_rc_config cfg = { gran, dist, model, target, 0 };
    u32 nu = vc_rc_count_units(d, h, w, gran);
    vc_rc_unit *units = (vc_rc_unit *)malloc((size_t)nu * sizeof(vc_rc_unit));
    vc_rc_result res;
    int rc = vc_rc_allocate(vol, d, h, w, &cfg, units, &res);
    if (rc) { printf("FAIL allocate %s rc=%d\n", tag, rc); free(units); return 0; }

    // step spread
    f32 smin = 1e9f, smax = 0.f; f64 ssum = 0;
    for (u32 i = 0; i < nu; ++i) { if (units[i].step < smin) smin = units[i].step; if (units[i].step > smax) smax = units[i].step; ssum += units[i].step; }
    printf("  %-26s target=%.0fx achieved=%.2fx lambda=%.3g units=%u step[min=%.1f mean=%.1f max=%.1f] parMSE=%.2f truMSE=%.2f\n",
           tag, target, res.achieved_ratio, res.lambda, nu, smin, ssum / nu, smax,
           res.parseval_mse, res.true_mse);
    // The achieved ratio should be within ~25% of target (probe grid is coarse).
    int ok = fabs(res.achieved_ratio - target) <= target * 0.25;
    free(units);
    return ok;
}

int main(void) {
    // 2x2x4 chunks @64 -> 16 chunks, mixed air|structure: enough chunks that
    // per-chunk granularity has headroom to hit moderate targets, while per-16^3
    // can push further (the whole point of the q-field).
    const u32 D = 128, H = 128, W = 256;
    u8 *vol = make_mixed(D, H, W);

    printf("== per-chunk, Parseval, probe ==\n");
    CHECK(hits(vol, D, H, W, VC_RC_PER_CHUNK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 10.0, "perchunk/parseval/probe 10x"), "per-chunk hits 10x");
    CHECK(hits(vol, D, H, W, VC_RC_PER_CHUNK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 20.0, "perchunk/parseval/probe 20x"), "per-chunk hits 20x");

    printf("== per-16^3-block q-field, Parseval, probe ==\n");
    CHECK(hits(vol, D, H, W, VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 10.0, "perblock/parseval/probe 10x"), "per-block hits 10x");
    CHECK(hits(vol, D, H, W, VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 20.0, "perblock/parseval/probe 20x"), "per-block hits 20x");

    printf("== true-MSE distortion ==\n");
    CHECK(hits(vol, D, H, W, VC_RC_PER_BLOCK, VC_RC_DIST_TRUEMSE, VC_RC_MODEL_PROBE, 10.0, "perblock/truemse/probe 10x"), "per-block true-MSE hits 10x");

    printf("== analytical Laplacian model ==\n");
    CHECK(hits(vol, D, H, W, VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_LAPLACIAN, 10.0, "perblock/parseval/laplacian 10x"), "Laplacian model hits 10x");

    // Step-distribution spread: on a mixed air|structure volume, per-block must
    // assign a RANGE of steps (air coarse, structure fine), not one uniform step.
    {
        vc_rc_config cfg = { VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 15.0, 0 };
        u32 nu = vc_rc_count_units(D, H, W, VC_RC_PER_BLOCK);
        vc_rc_unit *u = (vc_rc_unit *)malloc((size_t)nu * sizeof(vc_rc_unit));
        vc_rc_result res; vc_rc_allocate(vol, D, H, W, &cfg, u, &res);
        f32 smin = 1e9f, smax = 0.f;
        for (u32 i = 0; i < nu; ++i) { if (u[i].step < smin) smin = u[i].step; if (u[i].step > smax) smax = u[i].step; }
        printf("  per-block step spread: min=%.1f max=%.1f ratio=%.1fx\n", smin, smax, smax/smin);
        CHECK(smax > smin * 1.9f, "per-block assigns a varied q-field (air coarse, structure fine)");
        free(u);
    }

    free(vol);
    printf(fail ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return fail;
}
