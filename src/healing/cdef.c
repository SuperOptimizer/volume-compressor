// VC_HEALING = cdef. Directional Constrained Enhancement Filter (AV1-CDEF
// style), adapted to 3D as a per-slice 2D pass on all three orientations
// (PLAN §2 "Boundary healing", optional CDEF row). Cleans transform ringing
// that runs ALONG fiber directions without smearing edges across them.
//
// Per 8x8 in-plane block we pick 1 of 8 directions by minimizing the
// sum-of-squared-error of the block's samples to a per-direction line-average
// model (the AV1 direction search), then run the constrained filter ALONG that
// direction. The CDEF constraint function bounds how much each tap pulls the
// centre toward a neighbour:
//   constrain(diff, str, damp) = sign(diff) * clip(|diff|, 0,
//                                  max(0, str - (|diff| >> damp)))
// so large differences (edges) contribute ~0 (the str - |diff|>>damp term goes
// negative -> clipped to 0) while small differences (ringing) are smoothed. str
// is derived from the quant step; `strength` in [0,1] scales it (0 == identity).
//
// Pure post-filter: operates on the decoded u8 volume, no bitstream/codec state.
// Hot kernel is a branchless stencil over an 8x8 block; the per-slice plane is
// processed contiguously (X stride 1) for the Z faces, and gathered into a
// contiguous scratch plane for the Y/X orientations so inner loops stay
// unit-stride and autovectorize (verify -fopt-info-vec).
#include "healing.h"
#include <stdlib.h>
#include <string.h>

static inline i32 iclamp(i32 v, i32 lo, i32 hi) {
    v = v < lo ? lo : v; return v > hi ? hi : v;
}
static inline i32 iabs32(i32 v) { return v < 0 ? -v : v; }

// CDEF constraint: smooth small diffs, ignore large ones (edges). Branchless.
static inline i32 constrain(i32 diff, i32 str, i32 damp) {
    i32 a = iabs32(diff);
    i32 lim = str - (a >> damp);
    if (lim < 0) lim = 0;
    i32 mag = a < lim ? a : lim;          // clip magnitude
    return diff < 0 ? -mag : mag;
}

// The 4 in-plane direction tap offsets (dy,dx) for a 2D slice. We use the 4
// principal/diagonal axes (each filtered symmetrically +/-), which covers the
// 8 CDEF directions as +/- pairs of these 4.
static const int DIRDY[4] = { 0, 1, 1,  1 };
static const int DIRDX[4] = { 1, 0, 1, -1 };

// Direction search over an 8x8 block at (by,bx) of a (h,w) plane: pick the
// direction whose perpendicular partial sums explain the most variance (min
// residual SSE), i.e. the direction the structure runs along. Cheap proxy:
// pick the direction with the LARGEST summed squared first-difference ALONG it
// being small (smoothest along) -> equivalently smallest along-direction
// gradient energy. Returns 0..3.
static int dir_search(const u8 *restrict pl, u32 w, u32 by, u32 bx,
                      u32 h, u32 wd) {
    int best = 0; i64 bestcost = (i64)1 << 62;
    for (int d = 0; d < 4; ++d) {
        int dy = DIRDY[d], dx = DIRDX[d];
        i64 cost = 0;
        for (u32 j = 0; j < 8; ++j) {
            u32 y = by + j; if (y + 1 >= h) continue;
            for (u32 i = 0; i < 8; ++i) {
                u32 x = bx + i;
                int ny = (int)y + dy, nx = (int)x + dx;
                if (ny < 0 || nx < 0 || (u32)ny >= h || (u32)nx >= wd) continue;
                i32 diff = (i32)pl[(size_t)y * w + x] - (i32)pl[(size_t)ny * w + nx];
                cost += (i64)diff * diff;   // along-direction gradient energy
            }
        }
        if (cost < bestcost) { bestcost = cost; best = d; }
    }
    return best;
}

// Filter one (h,wd) plane stored row-major with row stride `w` (>=wd), in place,
// using `out` scratch (h*w) so the filter reads the unmodified plane.
static void cdef_plane(u8 *restrict pl, u8 *restrict out, u32 h, u32 wd, u32 w,
                       i32 str, i32 damp) {
    if (h < 3 || wd < 3) { return; }
    memcpy(out, pl, (size_t)h * w);
    for (u32 by = 0; by < h; by += 8)
    for (u32 bx = 0; bx < wd; bx += 8) {
        int d = dir_search(pl, w, by, bx, h, wd);
        int dy = DIRDY[d], dx = DIRDX[d];
        for (u32 j = 0; j < 8; ++j) {
            u32 y = by + j; if (y >= h) break;
            const u8 *restrict row = pl + (size_t)y * w;
            u8 *restrict orow = out + (size_t)y * w;
            for (u32 i = 0; i < 8; ++i) {
                u32 x = bx + i; if (x >= wd) break;
                i32 cen = row[x];
                i32 acc = 0;
                // Two symmetric taps along the chosen direction (+/-).
                int yp = (int)y + dy, xp = (int)x + dx;
                int ym = (int)y - dy, xm = (int)x - dx;
                int okp = (yp >= 0 && xp >= 0 && (u32)yp < h && (u32)xp < wd);
                int okm = (ym >= 0 && xm >= 0 && (u32)ym < h && (u32)xm < wd);
                i32 vp = okp ? (i32)pl[(size_t)yp * w + xp] : cen;
                i32 vm = okm ? (i32)pl[(size_t)ym * w + xm] : cen;
                acc += constrain(vp - cen, str, damp);
                acc += constrain(vm - cen, str, damp);
                // Round-to-nearest of the half-strength average nudge.
                orow[x] = (u8)iclamp(cen + ((acc + 2) >> 2), 0, 255);
            }
        }
    }
    memcpy(pl, out, (size_t)h * w);
}

void vc_heal_cdef(u8 *restrict vol, u32 dz, u32 dy, u32 dx,
                  f32 step, f32 strength) {
    if (strength <= 0.0f) return;
    if (strength > 1.0f) strength = 1.0f;
    i32 str = (i32)(step * 1.5f * strength + 0.5f);
    if (str < 1) str = 1;
    i32 damp = 3;   // CDEF damping: tap weight decays as |diff| grows.

    size_t pmax = (size_t)dz * (dx > dy ? dx : dy);
    if ((size_t)dy * dx > pmax) pmax = (size_t)dy * dx;
    u8 *out = (u8 *)malloc(pmax ? pmax : 1);
    u8 *pa  = (u8 *)malloc(pmax ? pmax : 1);
    if (!out || !pa) { free(out); free(pa); return; }

    const size_t sX = 1, sY = (size_t)dx, sZ = (size_t)dy * dx;
    // Z slices (dy x dx): contiguous, filter in place.
    for (u32 z = 0; z < dz; ++z)
        cdef_plane(vol + (size_t)z * sZ, out, dy, dx, dx, str, damp);
    // Y slices (dz x dx): gather, filter, scatter.
    for (u32 y = 0; y < dy; ++y) {
        for (u32 z = 0; z < dz; ++z)
            memcpy(pa + (size_t)z * dx, vol + (size_t)z * sZ + (size_t)y * sY, dx);
        cdef_plane(pa, out, dz, dx, dx, str, damp);
        for (u32 z = 0; z < dz; ++z)
            memcpy(vol + (size_t)z * sZ + (size_t)y * sY, pa + (size_t)z * dx, dx);
    }
    // X slices (dz x dy): gather, filter, scatter.
    for (u32 x = 0; x < dx; ++x) {
        for (u32 z = 0; z < dz; ++z)
        for (u32 yy = 0; yy < dy; ++yy)
            pa[(size_t)z * dy + yy] = vol[(size_t)z * sZ + (size_t)yy * sY + (size_t)x * sX];
        cdef_plane(pa, out, dz, dy, dy, str, damp);
        for (u32 z = 0; z < dz; ++z)
        for (u32 yy = 0; yy < dy; ++yy)
            vol[(size_t)z * sZ + (size_t)yy * sY + (size_t)x * sX] = pa[(size_t)z * dy + yy];
    }
    free(out); free(pa);
}
