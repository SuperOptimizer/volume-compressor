// Public API for the volume-compressor toolkit (PLAN §3).
// Encode/decode a dense u8 3D volume to/from a single-file archive using the
// compile-time-selected pipeline. Single-threaded, reentrant; the caller
// parallelizes across volumes/regions.
#ifndef VC_VC_H
#define VC_VC_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Encode a row-major u8 volume of shape (dz,dy,dx) at quality `q` (dead-zone
// step in coefficient units; larger => more compression). Writes a heap archive
// buffer to *out / *out_len (caller frees). Returns 0 on success.
int vc_encode_volume(const u8 *vol, u32 dz, u32 dy, u32 dx, f32 q,
                     u8 **out, size_t *out_len);

// Decode an archive into a freshly-allocated row-major u8 volume. Writes the
// volume to *vol (caller frees) and the shape to *dz/*dy/*dx. Returns 0 on
// success, nonzero on corruption / config mismatch.
int vc_decode_volume(const u8 *archive, size_t len,
                     u8 **vol, u32 *dz, u32 *dy, u32 *dx);

// The compile-time chunk side this build was configured with.
u32 vc_chunk_side(void);

#ifdef __cplusplus
}
#endif

#endif // VC_VC_H
