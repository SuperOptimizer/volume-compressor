// Boundary healing (PLAN §2 "Boundary healing" row, §3 healing/). DECODE-SIDE
// post-filters that run on the fully reconstructed u8 volume to suppress the
// blocking / seam / smear artifacts the transform+quant pipeline leaves at
// sub-block faces (the 16^3 integer-DCT grid) and chunk faces (VC_CHUNK_SIDE
// grid). These are PURE post-filters: they touch no bitstream, change no codec
// state, and preserve chunk independence + 16^3 random access (a chunk can be
// decoded and healed entirely from its own samples plus, at chunk faces, its
// already-decoded neighbours — the same data a halo decode would have).
//
// Block contract (the VC_HEALING lever, selected in config.h):
//   void VC_HEALING(u8 *restrict vol, u32 dz, u32 dy, u32 dx,
//                   f32 step, f32 strength);
// In-place heal of a row-major (dz,dy,dx) u8 volume. `step` is the dead-zone
// quantizer step the chunks were coded at (the decoder knows it per chunk; the
// volume-level pass uses a representative step) — it bounds how large a true
// quantization step can be, so the filter only smooths deltas a quantizer could
// have produced and clips everything else (never eats a real edge). `strength`
// in [0,1] scales the filter aggressiveness (0 == identity passthrough).
//
// All hot loops are straight-line stencils over contiguous planes with
// compile-time-friendly bounds, branchless clamps, restrict pointers, heap
// scratch — autovectorizable per PLAN §7.
#ifndef VC_HEALING_H
#define VC_HEALING_H

#include "../../include/vc/types.h"

// none: identity passthrough (Phase-0 default).
void vc_heal_none(u8 *restrict vol, u32 dz, u32 dy, u32 dx, f32 step, f32 strength);

// deblock: adaptive HEVC/AV1-CDEF-style deblocking across the 16^3 sub-block
// faces AND the VC_CHUNK_SIDE chunk faces on all three axes. Boundary strength
// is keyed on the quantizer step + local gradient: flat regions (small gradient
// relative to the step) get filtered, real edges (gradient >> step) do not, and
// the per-sample delta is clipped by a step-derived bound. Modifies <=3 samples
// per side of each boundary.
void vc_heal_deblock(u8 *restrict vol, u32 dz, u32 dy, u32 dx, f32 step, f32 strength);

// cdef: directional constrained filter (CDEF-style). Per 8x8 in-plane block,
// pick 1 of (a reduced set of) directions by min-SSE and filter along it with a
// step-derived strength/damping bound. Cleans ringing along fiber directions
// without smearing edges. Applied on all three slice orientations.
void vc_heal_cdef(u8 *restrict vol, u32 dz, u32 dy, u32 dx, f32 step, f32 strength);

#endif // VC_HEALING_H
