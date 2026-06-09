// ============================================================================
// v3archive_api.h — FROZEN v3 archive container API (2026-06-09). See codec-freeze.md.
//
// The v3 archive is a sparse multi-level node tree (region -> subregion -> dense
// shard -> sparse-block-dir chunk) of dense 256^3 chunks, each a grid of 16^3 DCT
// blocks coded by the v2 codec. 8 independent LODs (LOD0 = full res). LODs are
// INDEPENDENTLY fetchable + decodable (hard constraint — no cross-LOD dependency).
//
// DEPENDS ON v2codec (one direction): the archive gathers voxels + the per-chunk
// air mask from the volume, hands gathered 16^3 blocks to v2_enc_block, and lays
// out the coded payloads in the sparse tree. The low-level format constants live in
// v3archive.h; this is the high-level build/decode entry points.
// ============================================================================
#ifndef V3ARCHIVE_API_H
#define V3ARCHIVE_API_H
#include <stdint.h>
#include <stddef.h>
#include "v2codec.h"

// Build a .v3 archive from a zarr volume directory (u8, 128^3 C-order chunks, "/" sep).
// `dim` must be a multiple of 256 (chunk-aligned; padding is the export pipeline's job —
// this errors out otherwise). `quality` is the v2 base_q dial. Writes `outpath`.
// Returns 0 on success. Memory-lean: LOD0 is read via chunk-mmap (working-set resident).
int v3_build_from_zarr(const char *zarr_root, const char *outpath, int dim, float quality);

// ---- random-access decode from a built/mmap'd archive ----
// (decode a single 16^3 block, or a whole 256^3 chunk; the viewer path.)
// arc = pointer to the mmap'd archive bytes. lod = 0..7. Returns 0 if absent (all zero).
typedef struct v3_reader v3_reader;
v3_reader *v3_open(const uint8_t *arc, size_t len, float quality);
void       v3_close(v3_reader *r);
// resolve the file offset of chunk (cz,cy,cx) at `lod` (0 = empty/absent).
uint64_t   v3_chunk_offset(v3_reader *r, int lod, int cz,int cy,int cx);
// decode block (bz,by,bx) within the chunk at `chunk_off` into dst (16^3). Zero if absent.
void       v3_decode_block(v3_reader *r, uint64_t chunk_off, int bz,int by,int bx, v2_u8 *dst);

#endif
