// v3 on-disk archive: sparse multi-level node map + dense contiguous chunks.
//
// SCALE: voxels up to (2^20)^3, 8 LODs. Real extents usually <=2^16 (~65k), rarely
// up to 2^20 (2 volumes ~80k). So the index must be SPARSE (a flat 2^48-chunk array
// is impossible) but cheap for the common ~2^16 case, and download/disk-friendly.
//
// HIERARCHY (per LOD), all grids are 16x16x16 (4 bits/axis), built on the proven v1
// v2-container directory entry [u64 off][u32 len][u32 flags]:
//   voxel(20b/axis) = [ region 4b | subregion 4b | shard 4b | chunkInShard 4b | intra-chunk 8b ]
//   wait: chunk = 256^3 = 8 bits/axis intra. above it 12 bits/axis = 3 levels of 4b.
//   So: CHUNK(256^3 voxels = 16^3 DCT blocks) is the dense leaf; above it 3 SPARSE
//   levels of 16^3 grids: shard(4096^3 vox) -> subregion(65536^3) -> region(1M^3).
//   3 sparse levels x 16x = 16^3 = 4096-wide each, 4096^3 = covers 2^12 chunks/axis
//   = 2^20 voxels/axis. Exact.
//
// NODE = a 16^3 (=4096) array of child entries, but stored SPARSELY: only present
// children listed, with a 4096-bit (512B) occupancy bitmap + a packed array of the
// present children's entries. A "0" child (empty region/all-air) is simply absent
// from the bitmap -> the pointer-to-0 the user asked for is "bit not set" (free).
//
// CHUNK (the dense leaf): 256^3 voxels = 16^3 = 4096 DCT-blocks of 16^3 each, packed
// CONTIGUOUSLY: [chunk header][block directory: 4096 x (u32 off,u32 len/flags)][block
// payloads...]. One range-GET fetches the whole chunk. Blocks within are ZERO (flag)
// or DCT (offset+len into the payload area).
#ifndef V3_ARCHIVE_H
#define V3_ARCHIVE_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

// ---- node (sparse 16^3 child map) ----
#define V3_GRID    16              // children per axis at each sparse level
#define V3_GRID3   4096            // 16^3
#define V3_BITMAP_BYTES 512        // 4096 bits
// child entry (8B): [u64 off] where off points to the child node OR (leaf level) the
// chunk blob. len + flags live in the child's own header (saves directory bytes; the
// occupancy bitmap already encodes present/absent). off==0 reserved (header).

// chunk block entry (8B): [u32 off][u32 len_flags]  (flags in top 2 bits of len_flags)
#define V3_BLK_ZERO  0x80000000u   // top bit: this 16^3 block is all-zero (no payload)
#define V3_BLK_LENMASK 0x7FFFFFFFu

// ---- header (128B, v3) ----
#define V3_MAGIC 0x00335643u       // "VC3\0"
#define V3H_MAGIC 0
#define V3H_VER   4
#define V3H_NLOD  8
#define V3H_NX    12
#define V3H_NY    16
#define V3H_NZ    20
#define V3H_ROOTOFF 24             // u64[8]: per-LOD root-node file offset (0 = empty LOD)
#define V3H_TOTLEN  88             // u64 total archive length
#define V3_HDR 128u
#define V3_VERSION 3u

// occupancy bitmap helpers
static inline int v3_bit_get(const uint8_t*bm,int i){ return (bm[i>>3]>>(i&7))&1; }
static inline void v3_bit_set(uint8_t*bm,int i){ bm[i>>3]|=(uint8_t)(1u<<(i&7)); }
// local child index within a 16^3 node from the 3 nibbles at level L of a chunk coord.
static inline int v3_child_idx(int cz,int cy,int cx){ return ((cz&15)*V3_GRID + (cy&15))*V3_GRID + (cx&15); }
// LEVEL DENSITY (user 2026-06-08): chunk(2^8) + shard(2^12, dense direct table) are
// DENSE; the two levels above (2^16 subregion, 2^20 region) are SPARSE. chunk coord
// = 12 bits/axis = 3 nibbles: nibble0 = chunk-within-shard (DENSE 16^3 table),
// nibbles 1,2 = the 2 sparse levels. A dense shard = 16^3=4096 chunk-offset slots
// (32KB); occupied shards are few (sparse scroll) so total shard tables are small MB.
#define V3_SPARSE_LEVELS 2          // sparse levels above the dense shard
// SHARD = dense 16^3 chunk-offset table: [4096 x u64 chunk-blob-off]. Offset 0 = EMPTY
// (zero-padding) chunk — SAFE sentinel: real data never starts at file offset 0 (the
// 128B header is there), exactly like v1's block_ref==0. So denoting a totally-zero
// chunk = leave its slot 0 (free); a whole-zero shard = unset bit in the parent sparse
// node (no table allocated at all); whole-zero block = unset bit in the chunk bitmap.
// "Totally zero padding" is thus free/near-free at every level.
#define V3_SHARD_BYTES (V3_GRID3*8)
// chunk coord = voxel>>8 (256^3 chunk). nibble L (0=finest above chunk) of a chunk coord.
static inline int v3_nib(int chunkcoord,int level){ return (chunkcoord>>(4*level))&15; }
#endif
