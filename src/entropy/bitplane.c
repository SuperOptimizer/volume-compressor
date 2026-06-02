// Bit-plane / SNR-scalable embedded coder for quantized DCT-16^3 coefficient
// atoms (PARKED axis re-examination, PLAN §6 last bullet + §2 dropped row).
//
// PURPOSE: measure the RATIO COST of embedded, truncatable coding versus the
// non-scalable table-free RLGR baseline, on real PHerc Paris 4 data, so we can
// confirm (or overturn) the 2026-06-02 decision to DROP SNR scalability in
// favour of the resolution pyramid for progressive quality.
//
// DESIGN (EZW/SPIHT/JPEG2000-EBCOT family, simplified, table-free so it is a
// fair head-to-head with RLGR — no rANS table side cost):
//   Per atom we receive the SAME freq-scanned signed quantized level array the
//   RLGR baseline codes (n levels, low->high frequency). We code it MSB->LSB:
//     * header: nplanes P = bit-position of the most-significant set bit over
//       all |level| (0 => empty atom).
//     * for plane p = P-1 .. 0:
//         - SIGNIFICANCE pass: for every coefficient still INSIGNIFICANT (no
//           bit >= p+1 seen yet), emit one significance bit "does |level| have
//           its bit p set as the FIRST set bit" — i.e. did it become newly
//           significant on this plane. Newly-significant coefficients are sparse
//           on high planes, so we code the GAPS between them in scan order with
//           an adaptive Golomb-Rice run code (same kR adaptation idea as RLGR),
//           which is the table-free analogue of EBCOT's context model. When a
//           coefficient becomes significant we emit its SIGN bit immediately.
//         - REFINEMENT pass: for every ALREADY-significant coefficient, emit its
//           bit at plane p verbatim (these bits are ~equiprobable -> bypass).
//   Truncating the stream at ANY byte yields a valid lower-quality reconstruction
//   (each plane halves the max quant error) — that is the scalability this buys.
//
// This file is self-contained (types.h + bitio.h only), like the transform/quant
// bake-off blocks, so the bench #includes it directly. Round-trip exact when the
// full stream is kept. Pure C23, libc only.
#ifndef VC_BITPLANE_C_INLINE
#define VC_BITPLANE_C_INLINE
#include "../core/bitio.h"
#include "../../include/vc/types.h"

// Adaptive run parameter for the significance-gap code (mirrors RLGR kR idea).
#define BP_L   4u
#define BP_U0  3u
#define BP_D0  1u
#define BP_KR_MAX (14u << BP_L)

// Encode one signed level array MSB->LSB. Returns bytes written (cap+1 on
// overflow). `nplanes_out` receives P (so a truncator knows the plane count).
static size_t vc_bitplane_encode(u8 *restrict out, size_t cap,
                                 const i16 *restrict q, size_t n,
                                 u32 *restrict nplanes_out) {
    vc_bitwriter w; vc_bw_init(&w, out, cap);

    // P = position of highest set bit across all magnitudes.
    u32 maxmag = 0;
    for (size_t i = 0; i < n; ++i) {
        i32 v = q[i]; u32 m = (u32)(v < 0 ? -v : v);
        if (m > maxmag) maxmag = m;
    }
    u32 P = 0; { u32 t = maxmag; while (t) { P++; t >>= 1; } }
    if (nplanes_out) *nplanes_out = P;
    // header: 5 bits of P (P <= 16 for i16 magnitudes).
    vc_bw_put(&w, P, 5);
    if (P == 0) return vc_bw_finish(&w);

    // significance state: 0 until the coefficient's leading bit is coded.
    // We avoid a separate alloc by using the sign of a small heap flag array
    // supplied via static thread-local-free scratch: encode is given n<=4096,
    // so a fixed stack-free path uses a caller buffer. Keep it simple: a VLA is
    // disallowed (chunk-sized stack banned), but n here is one ATOM (4096) which
    // is the codec atom, not a chunk -> a 4096-byte local is the atom working
    // set, acceptable (same size the transform already uses on stack-free heap).
    // To honour "no chunk-sized stack", we mark significance in the high bit of
    // a parallel array the CALLER owns is overkill; instead we recompute leading
    // bit per coefficient cheaply (n*P bit tests, P<=16) — fully branchless.

    u32 kR = 0; // adaptive run param (fixed point) for significance gaps

    for (i32 p = (i32)P - 1; p >= 0; --p) {
        // --- significance pass: walk scan order, code GAPS to next newly-sig ---
        // A coefficient is "already significant" if it has any set bit above p.
        // It becomes "newly significant" on plane p if its leading set bit == p.
        size_t i = 0;
        while (i < n) {
            // skip coefficients already significant (their refinement handled below)
            // and count a run of "not-newly-significant-on-this-plane" coefficients.
            u32 k = kR >> BP_L;
            u32 runmax = 1u << k;
            u32 run = 0;
            size_t startscan = i;
            i16 termval = 0; int have_term = 0;
            while (i < n) {
                i32 v = q[i];
                u32 m = (u32)(v < 0 ? -v : v);
                u32 above = m >> (p + 1);          // bits above plane p
                if (above != 0) { i++; continue; } // already significant: skip (refine later)
                u32 bitp = (m >> p) & 1u;
                int newly = (bitp != 0);           // leading bit is exactly p
                if (!newly) {
                    run++; i++;
                    if (run == runmax) break;      // emit a complete run unit
                } else {
                    termval = (i16)v; have_term = 1; i++;
                    break;
                }
            }
            (void)startscan;
            if (run == runmax && !have_term) {
                vc_bw_put(&w, 0u, 1);              // complete run unit of insig
                kR += BP_U0; if (kR > BP_KR_MAX) kR = BP_KR_MAX;
            } else {
                vc_bw_put(&w, 1u, 1);              // run terminated early
                vc_bw_put(&w, run, k);             // low k bits of the run length
                if (have_term) {
                    vc_bw_put(&w, termval < 0 ? 1u : 0u, 1); // sign of newly-sig
                }
                kR = (kR > BP_D0) ? kR - BP_D0 : 0u;
            }
            if (w.overflow) return cap + 1;
        }
        // --- refinement pass: bit p of every ALREADY-significant coefficient ---
        for (size_t j = 0; j < n; ++j) {
            i32 v = q[j]; u32 m = (u32)(v < 0 ? -v : v);
            u32 above = m >> (p + 1);
            if (above != 0) {                      // significant before this plane
                vc_bw_put(&w, (m >> p) & 1u, 1);
            }
        }
        if (w.overflow) return cap + 1;
    }
    return vc_bw_finish(&w);
}

// Full (non-truncated) decode -> exact inverse of encode. q must hold n levels.
static void vc_bitplane_decode(i16 *restrict q, size_t n,
                               const u8 *restrict in, size_t len) {
    vc_bitreader r; vc_br_init(&r, in, len);
    for (size_t i = 0; i < n; ++i) q[i] = 0;
    u32 P = vc_br_get(&r, 5);
    if (P == 0) return;

    // magnitude accumulator + sign per coefficient. We reconstruct |level| bit by
    // bit; sign captured when a coefficient becomes significant.
    // (decode-side significance == magnitude already nonzero.)
    // n is one atom (4096) -> atom working set, not chunk-sized.
    u32 kR = 0;
    for (i32 p = (i32)P - 1; p >= 0; --p) {
        // significance pass
        size_t i = 0;
        while (i < n) {
            u32 k = kR >> BP_L;
            u32 runmax = 1u << k;
            // skip already-significant coefficients (handled in refinement)
            // mirror encoder: we consume run/term for INSIGNIFICANT coefficients only.
            // gather insignificant positions implicitly by walking.
            // First skip leading already-significant.
            u32 flag;
            // We must reproduce the encoder's interleave: it skipped significant
            // coeffs WITHIN the run scan. So here we read one run record, then
            // distribute it over the next insignificant coeffs.
            flag = vc_br_get(&r, 1);
            u32 run; int have_term;
            if (!flag) { run = runmax; have_term = 0; kR += BP_U0; if (kR > BP_KR_MAX) kR = BP_KR_MAX; }
            else {
                run = vc_br_get(&r, k);
                have_term = 1; // a newly-significant terminator unless we hit n
                kR = (kR > BP_D0) ? kR - BP_D0 : 0u;
            }
            // place `run` insignificant coeffs (skipping already-significant), then
            // one newly-significant terminator (if have_term and not past n).
            u32 placed = 0;
            while (i < n && placed < run) {
                u32 m = (u32)(q[i] < 0 ? -q[i] : q[i]);
                if (m != 0) { i++; continue; }     // already significant: skip
                placed++; i++;                     // stays zero this plane
            }
            if (have_term) {
                // advance to next insignificant coeff and make it significant
                while (i < n) {
                    u32 m = (u32)(q[i] < 0 ? -q[i] : q[i]);
                    if (m != 0) { i++; continue; }
                    u32 sgn = vc_br_get(&r, 1);
                    i32 val = (1 << p);
                    q[i] = (i16)(sgn ? -val : val);
                    i++;
                    break;
                }
            }
            if (r.byte > r.len && r.nbits == 0) return; // truncated stream guard
        }
        // refinement pass
        for (size_t j = 0; j < n; ++j) {
            i32 v = q[j]; u32 m = (u32)(v < 0 ? -v : v);
            // significant BEFORE this plane means it had a bit above p, i.e. its
            // current magnitude >= 2^(p+1). Newly-significant-on-this-plane coeffs
            // currently equal exactly 2^p and must NOT be refined here.
            if (m >= (2u << p)) {
                u32 b = vc_br_get(&r, 1);
                u32 nm = (m & ~(1u << p)) | (b << p);
                q[j] = (i16)(v < 0 ? -(i32)nm : (i32)nm);
            }
        }
    }
}

#endif // VC_BITPLANE_C_INLINE
