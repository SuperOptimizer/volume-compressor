// VC_HEALING = deblock. Adaptive decode-side deblocking filter across the 16^3
// sub-block faces AND the VC_CHUNK_SIDE chunk faces, on all three axes (PLAN §2
// "Boundary healing"). Pure post-filter on the reconstructed u8 volume: no
// bitstream change, chunk independence + 16^3 random access preserved.
//
// Model (HEVC deblock + AV1-CDEF-style boundary-strength gating, adapted to a
// 1D normal-direction stencil per face):
//
// Across a boundary, six samples straddle the face along the normal axis:
//      p2 p1 p0 | q0 q1 q2
// We decide per line whether the step at the face (|p0-q0|) is a *quantization
// seam* (worth smoothing) or a *real edge* (must be preserved):
//
//   - beta  ~ step  : a flatness gate. The two sides must be locally flat
//                     (|p2-p0|, |q2-q0| small AND the cross-face step small
//                     RELATIVE to beta). A real edge has a large cross-face step
//                     that dwarfs beta -> gate fails -> no filtering.
//   - tc    ~ step  : a clip bound on every per-sample delta, so even when we do
//                     filter we can never move a sample more than a quantizer
//                     could plausibly have shifted it. This is what stops the
//                     filter eating thin ink / fibers.
//
// When the gate passes we apply a graded 3-2-1 smoothing of p0/p1/p2 and
// q0/q1/q2 toward the local ramp, each delta clipped to +-tc. `strength` in
// [0,1] scales beta, tc, and the applied delta (0 == identity).
//
// Hot loops: for the Z- and Y-normal faces the stride-1 (X) dimension is the
// in-plane sweep, so the inner loop is a straight-line branchless stencil over
// contiguous samples -> autovectorizes (verify -fopt-info-vec). The X-normal
// faces sweep X as the normal (unit stride along the face position) but iterate
// the contiguous-free (z,y) plane; we keep the same branchless kernel so it
// still vectorizes over the plane.
#include "healing.h"
#include <stdlib.h>
#include <string.h>

#define VC_HEAL_BLK 16u   // 16^3 integer-DCT sub-block grid

static inline i32 iclamp(i32 v, i32 lo, i32 hi) {
    v = v < lo ? lo : v;
    return v > hi ? hi : v;
}
static inline i32 iabs32(i32 v) { return v < 0 ? -v : v; }

// Deblock one face-normal direction. The volume is addressed via three strides:
//   sN : stride along the face NORMAL (the axis whose faces we filter)
//   sA, sB : strides along the two in-plane axes, with extents nA, nB; B is the
//            stride-1 (innermost, vectorizable) axis when present.
// `nN` is the extent along the normal; faces sit at every multiple of 16 and of
// VC_CHUNK_SIDE inside (0,nN). beta/tc are the precomputed gates (already
// strength-scaled). The kernel is branchless: the gate becomes a 0/1 mask that
// multiplies the (clipped) deltas, so no data-dependent branch sits in the
// vectorized inner loop.
static void deblock_axis(u8 *restrict vol, u32 nN, u32 nA, u32 nB,
                         size_t sN, size_t sA, size_t sB,
                         i32 beta, i32 tc, u32 cs) {
    for (u32 n = VC_HEAL_BLK; n < nN; n += VC_HEAL_BLK) {
        // Only act where a sub-block OR chunk face actually lies (every 16; the
        // chunk faces are a subset and use the same kernel). Need 3 samples each
        // side: skip faces too close to the volume edge.
        if (n < 3 || n + 3 > nN) continue;
        (void)cs;
        const size_t base_p0 = (size_t)(n - 1) * sN;
        const size_t base_q0 = (size_t)(n    ) * sN;
        for (u32 a = 0; a < nA; ++a) {
            u8 *restrict line = vol + (size_t)a * sA;
            // Inner loop over the contiguous in-plane axis B -> vectorizable.
            for (u32 b = 0; b < nB; ++b) {
                u8 *restrict c = line + (size_t)b * sB;
                i32 p2 = c[base_p0 - 2 * sN];
                i32 p1 = c[base_p0 - 1 * sN];
                i32 p0 = c[base_p0];
                i32 q0 = c[base_q0];
                i32 q1 = c[base_q0 + 1 * sN];
                i32 q2 = c[base_q0 + 2 * sN];

                // Flatness on each side and the cross-face step.
                i32 dp = iabs32(p2 - 2 * p1 + p0);   // 2nd diff (curvature) left
                i32 dq = iabs32(q2 - 2 * q1 + q0);   // 2nd diff right
                i32 dstep = iabs32(p0 - q0);
                // Gate: both sides locally flat (curvature < beta/8 each) AND the
                // cross-face step is in the "quant seam" band (< beta). A real
                // edge has dstep >> beta -> mask 0. Branchless mask.
                i32 flat = ((dp + dq) * 4 < beta) & (dstep < beta);
                i32 mask = -flat;                    // 0x0 or 0xFFFFFFFF

                // Smooth toward the local ramp. Target deltas (HEVC-like 3-2-1):
                //   dP0 = (q0 - p0)/2 nudged by curvature; graded into p1,p2.
                i32 d0 = (q0 - p0);
                i32 dP0 = ( 4 * d0 + 4) >> 3;         // ~ d0/2
                i32 dQ0 = (-4 * d0 + 4) >> 3;
                i32 dP1 = ( 2 * d0 + 4) >> 3;         // ~ d0/4
                i32 dQ1 = (-2 * d0 + 4) >> 3;
                i32 dP2 = ( 1 * d0 + 4) >> 3;         // ~ d0/8
                i32 dQ2 = (-1 * d0 + 4) >> 3;

                // Clip every delta to +-tc (quant-derived bound) then mask.
                dP0 = iclamp(dP0, -tc, tc) & mask;
                dQ0 = iclamp(dQ0, -tc, tc) & mask;
                dP1 = iclamp(dP1, -(tc >> 1), tc >> 1) & mask;
                dQ1 = iclamp(dQ1, -(tc >> 1), tc >> 1) & mask;
                dP2 = iclamp(dP2, -(tc >> 2), tc >> 2) & mask;
                dQ2 = iclamp(dQ2, -(tc >> 2), tc >> 2) & mask;

                c[base_p0]          = (u8)iclamp(p0 + dP0, 0, 255);
                c[base_p0 - 1 * sN] = (u8)iclamp(p1 + dP1, 0, 255);
                c[base_p0 - 2 * sN] = (u8)iclamp(p2 + dP2, 0, 255);
                c[base_q0]          = (u8)iclamp(q0 + dQ0, 0, 255);
                c[base_q0 + 1 * sN] = (u8)iclamp(q1 + dQ1, 0, 255);
                c[base_q0 + 2 * sN] = (u8)iclamp(q2 + dQ2, 0, 255);
            }
        }
    }
}

void vc_heal_deblock(u8 *restrict vol, u32 dz, u32 dy, u32 dx,
                     f32 step, f32 strength) {
    if (strength <= 0.0f) return;
    if (strength > 1.0f) strength = 1.0f;

    // Quant-derived gates. The dead-zone step bounds the worst-case error a
    // single coefficient quant could inject; a face step up to ~step is plausibly
    // a quant seam. beta is the flatness/seam band; tc the per-sample clip.
    i32 beta = (i32)(step * 2.0f * strength + 0.5f);
    i32 tc   = (i32)(step * 0.5f * strength + 0.5f);
    if (beta < 1) beta = 1;
    if (tc   < 1) tc   = 1;

    const size_t sX = 1, sY = (size_t)dx, sZ = (size_t)dy * dx;
    const u32 cs = (u32)VC_HEAL_BLK;

    // Z-normal faces: normal = Z (stride sZ); in-plane (Y outer, X inner=stride1).
    deblock_axis(vol, dz, dy, dx, sZ, sY, sX, beta, tc, cs);
    // Y-normal faces: normal = Y (stride sY); in-plane (Z outer, X inner=stride1).
    deblock_axis(vol, dy, dz, dx, sY, sZ, sX, beta, tc, cs);
    // X-normal faces: normal = X (stride sX); in-plane (Z outer, Y inner).
    deblock_axis(vol, dx, dz, dy, sX, sZ, sY, beta, tc, cs);
}
