// RLGR — Run-Length Golomb-Rice entropy coder (Malvar, DCC 2006), adapted to
// our signed-quantized-level array contract (VC_ENTROPY_ENC/DEC, same signature
// as rice.c). Table-free, backward-adaptive.
//
// The two-mode adaptive scheme:
//   * Mode 1 (RUN mode, kRP large): the source is dominated by zeros. We code a
//     run of zeros as a Golomb-Rice run: emit (run >> kR) "complete" units of
//     2^kR zeros each (one '0' bit per unit), then a '1' bit, then the low kR
//     bits of the residual run length, then the terminating nonzero value coded
//     in Golomb-Rice with parameter kGR. kR adapts up when runs are long.
//   * Mode 2 (NO-RUN / low-order mode, kRP==0): zeros are not dominant. Each
//     value is coded directly: a zero is a single '0' bit; a nonzero is a '1'
//     bit followed by its (magnitude-1) in Golomb-Rice(kGR) plus a sign bit.
//
// Backward adaptation: both kRP (run/no-run state, fixed-point with L=4 shift)
// and kGR (the Golomb-Rice parameter, fixed-point) are updated from the data
// just coded, so the decoder reproduces them with no side information. This is
// the standard RLGR adaptation rule; constants follow Malvar's reference.
//
// Round-trip exact. The hot work is bit IO; the adaptation is a handful of
// integer ops per symbol (branchy by nature — RLGR is inherently serial, so we
// do not claim vectorization here; the vectorizable stages are transform/quant).
#include "../core/bitio.h"
#include "../../include/vc/types.h"

// Adaptation constants (Malvar DCC2006 reference values).
#define RLGR_L      4u            // fixed-point shift for kRP and kGR
#define RLGR_U0     3u            // kRP up-step on long run
#define RLGR_D0     1u            // kRP down-step on broken run
#define RLGR_U1     2u            // kGR up-step
#define RLGR_D1     1u            // kGR down-step
#define RLGR_KRP_MAX (16u << RLGR_L)
#define RLGR_KGR_MAX (16u << RLGR_L)

// Reconstruct a signed level from its magnitude (>=1) and sign bit (1 => neg).
static inline i16 unzz_mag(u32 mag, u32 sgn) {
    i32 m = (i32)mag;
    return (i16)(sgn ? -m : m);
}

// Golomb-Rice code of nonnegative u with parameter k into the bit writer.
static inline void gr_put(vc_bitwriter *w, u32 u, u32 k) {
    vc_bw_put_unary(w, u >> k);
    if (k) vc_bw_put(w, u & ((1u << k) - 1u), k);
}
static inline u32 gr_get(vc_bitreader *r, u32 k) {
    u32 hi = vc_br_get_unary(r);
    u32 lo = k ? vc_br_get(r, k) : 0u;
    return (hi << k) | lo;
}

size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n) {
    vc_bitwriter w; vc_bw_init(&w, out, cap);
    u32 kRP = 0;             // run-mode state (fixed point); 0 => no-run mode
    u32 kGR = 2u << RLGR_L;  // Golomb-Rice param (fixed point), start at k=2

    size_t i = 0;
    while (i < n) {
        u32 k  = kRP >> RLGR_L;     // current run-mode order
        u32 kg = kGR >> RLGR_L;     // current GR order
        if (k == 0) {
            // NO-RUN mode: one symbol.
            i16 v = q[i++];
            if (v == 0) {
                vc_bw_put(&w, 0u, 1);                 // zero flag
                // adapt: zeros push us toward run mode
                kRP += RLGR_U0;
                if (kRP > RLGR_KRP_MAX) kRP = RLGR_KRP_MAX;
            } else {
                vc_bw_put(&w, 1u, 1);                 // nonzero flag
                u32 mag = (u32)(v < 0 ? -v : v);
                gr_put(&w, mag - 1u, kg);
                vc_bw_put(&w, v < 0 ? 1u : 0u, 1);    // sign
                // adapt kRP down (nonzero seen) and kGR toward this magnitude
                kRP = (kRP > RLGR_D0) ? kRP - RLGR_D0 : 0u;
                u32 p = mag - 1u;
                if ((p >> kg) != 0) { kGR += RLGR_U1; if (kGR > RLGR_KGR_MAX) kGR = RLGR_KGR_MAX; }
                else if (p == 0)    { kGR = (kGR > RLGR_D1) ? kGR - RLGR_D1 : 0u; }
            }
        } else {
            // RUN mode: count the zero run, then code it + terminating value.
            u32 runmax = 1u << k;
            u32 run = 0;
            while (i < n && q[i] == 0 && run < runmax) { run++; i++; }
            if (run == runmax) {
                // A full run unit: emit a single '0' (complete unit), adapt up.
                vc_bw_put(&w, 0u, 1);
                kRP += RLGR_U0;
                if (kRP > RLGR_KRP_MAX) kRP = RLGR_KRP_MAX;
            } else {
                // Partial run of `run` zeros (0..runmax-1) terminated by either a
                // nonzero value or end-of-array. Emit '1', then k low bits of run.
                vc_bw_put(&w, 1u, 1);
                vc_bw_put(&w, run, k);
                if (i < n) {
                    i16 v = q[i++];                   // the terminating nonzero
                    u32 mag = (u32)(v < 0 ? -v : v);
                    gr_put(&w, mag - 1u, kg);
                    vc_bw_put(&w, v < 0 ? 1u : 0u, 1);
                    u32 p = mag - 1u;
                    if ((p >> kg) != 0) { kGR += RLGR_U1; if (kGR > RLGR_KGR_MAX) kGR = RLGR_KGR_MAX; }
                    else if (p == 0)    { kGR = (kGR > RLGR_D1) ? kGR - RLGR_D1 : 0u; }
                }
                // broken run => adapt kRP down
                kRP = (kRP > RLGR_D0) ? kRP - RLGR_D0 : 0u;
            }
        }
        if (w.overflow) return cap + 1;
    }
    return vc_bw_finish(&w);
}

void vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len) {
    vc_bitreader r; vc_br_init(&r, in, len);
    u32 kRP = 0;
    u32 kGR = 2u << RLGR_L;

    size_t i = 0;
    while (i < n) {
        u32 k  = kRP >> RLGR_L;
        u32 kg = kGR >> RLGR_L;
        if (k == 0) {
            u32 flag = vc_br_get(&r, 1);
            if (!flag) {
                q[i++] = 0;
                kRP += RLGR_U0;
                if (kRP > RLGR_KRP_MAX) kRP = RLGR_KRP_MAX;
            } else {
                u32 mag = gr_get(&r, kg) + 1u;
                u32 sgn = vc_br_get(&r, 1);
                q[i++] = unzz_mag(mag, sgn);
                kRP = (kRP > RLGR_D0) ? kRP - RLGR_D0 : 0u;
                u32 p = mag - 1u;
                if ((p >> kg) != 0) { kGR += RLGR_U1; if (kGR > RLGR_KGR_MAX) kGR = RLGR_KGR_MAX; }
                else if (p == 0)    { kGR = (kGR > RLGR_D1) ? kGR - RLGR_D1 : 0u; }
            }
        } else {
            u32 runmax = 1u << k;
            u32 flag = vc_br_get(&r, 1);
            if (!flag) {
                // full run unit of `runmax` zeros
                for (u32 j = 0; j < runmax && i < n; ++j) q[i++] = 0;
                kRP += RLGR_U0;
                if (kRP > RLGR_KRP_MAX) kRP = RLGR_KRP_MAX;
            } else {
                u32 run = vc_br_get(&r, k);
                for (u32 j = 0; j < run && i < n; ++j) q[i++] = 0;
                if (i < n) {
                    u32 mag = gr_get(&r, kg) + 1u;
                    u32 sgn = vc_br_get(&r, 1);
                    q[i++] = unzz_mag(mag, sgn);
                    u32 p = mag - 1u;
                    if ((p >> kg) != 0) { kGR += RLGR_U1; if (kGR > RLGR_KGR_MAX) kGR = RLGR_KGR_MAX; }
                    else if (p == 0)    { kGR = (kGR > RLGR_D1) ? kGR - RLGR_D1 : 0u; }
                }
                kRP = (kRP > RLGR_D0) ? kRP - RLGR_D0 : 0u;
            }
        }
    }
}
