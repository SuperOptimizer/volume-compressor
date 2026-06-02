// Chunk gather/scatter (PLAN §3 core). The chunk side is a COMPILE-TIME CONSTANT
// (VC_CHUNK_SIDE). gather copies a VC_CHUNK_SIDE^3 cube out of a larger volume,
// zero-padding edge chunks that run past the volume bounds; scatter writes a
// decoded cube back, clipping to the volume bounds. Dimension-agnostic of any
// block logic.
#ifndef VC_CHUNK_H
#define VC_CHUNK_H

#include "../../include/vc/types.h"
#include "../config.h"
#include <string.h>

#define VC_CS   ((u32)VC_CHUNK_SIDE)
#define VC_CVOX ((size_t)VC_CS * VC_CS * VC_CS)

// Number of chunks along an axis of length `n` (ceil division).
static inline u32 vc_nchunks(u32 n) { return (n + VC_CS - 1u) / VC_CS; }

// Gather chunk at chunk-grid coord (cz,cy,cx) from volume `vol` of shape
// (dz,dy,dx) into `out` (VC_CHUNK_SIDE^3, row-major). Edge chunks are zero-padded.
static inline void vc_chunk_gather(u8 *restrict out, const u8 *restrict vol,
                                   u32 dz, u32 dy, u32 dx,
                                   u32 cz, u32 cy, u32 cx) {
    const u32 oz = cz * VC_CS, oy = cy * VC_CS, ox = cx * VC_CS;
    // Rows that are fully in-bounds copy directly; partial/over-edge rows zero-pad.
    const u32 cw = (ox + VC_CS <= dx) ? VC_CS : (ox < dx ? dx - ox : 0u);
    for (u32 z = 0; z < VC_CS; ++z) {
        const u32 vz = oz + z;
        u8 *orow_base = out + (size_t)z * VC_CS * VC_CS;
        if (vz >= dz) { memset(orow_base, 0, (size_t)VC_CS * VC_CS); continue; }
        for (u32 y = 0; y < VC_CS; ++y) {
            const u32 vy = oy + y;
            u8 *orow = orow_base + (size_t)y * VC_CS;
            if (vy >= dy || cw == 0) { memset(orow, 0, VC_CS); continue; }
            const u8 *irow = vol + (((size_t)vz * dy + vy) * dx + ox);
            memcpy(orow, irow, cw);
            if (cw < VC_CS) memset(orow + cw, 0, VC_CS - cw);
        }
    }
}

// Scatter a decoded VC_CHUNK_SIDE^3 cube `in` back into volume `vol`, clipping
// rows that run past the volume bounds.
static inline void vc_chunk_scatter(u8 *restrict vol, const u8 *restrict in,
                                    u32 dz, u32 dy, u32 dx,
                                    u32 cz, u32 cy, u32 cx) {
    const u32 oz = cz * VC_CS, oy = cy * VC_CS, ox = cx * VC_CS;
    const u32 cw = (ox + VC_CS <= dx) ? VC_CS : (ox < dx ? dx - ox : 0u);
    if (cw == 0) return;
    for (u32 z = 0; z < VC_CS; ++z) {
        const u32 vz = oz + z;
        if (vz >= dz) break;
        const u8 *irow_base = in + (size_t)z * VC_CS * VC_CS;
        for (u32 y = 0; y < VC_CS; ++y) {
            const u32 vy = oy + y;
            if (vy >= dy) break;
            u8 *orow = vol + (((size_t)vz * dy + vy) * dx + ox);
            memcpy(orow, irow_base + (size_t)y * VC_CS, cw);
        }
    }
}

#endif // VC_CHUNK_H
