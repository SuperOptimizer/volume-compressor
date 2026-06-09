// v3 archive reader: resolve a chunk by coord through the sparse node tree, then
// access a block within the dense chunk. The "pointer to 0" is a clear bitmap bit:
// absent child = empty region (decodes to zero), present child = popcount-ranked offset.
#ifndef V3_READ_H
#define V3_READ_H
#include "v3archive.h"
#include <stdint.h>

// rank: number of set bits in bm[0..idx) (exclusive) — gives the packed slot of child idx.
static inline int v3_rank(const uint8_t*bm,int idx){
    int r=0, full=idx>>3;
    for(int i=0;i<full;++i) r+=__builtin_popcount(bm[i]);
    int rem=idx&7; if(rem) r+=__builtin_popcount(bm[full] & ((1u<<rem)-1));
    return r;
}

// Resolve chunk (cz,cy,cx) to its chunk-blob offset; 0 if empty/absent. Walks the 2
// SPARSE levels (chunk-coord nibbles 2 then 1) to find the shard, then DIRECT-INDEXES
// the dense shard table by nibble 0. Sparse miss (unset bit) or empty shard slot = the
// "pointer to 0" (empty region/chunk → decodes to zeros).
static uint64_t v3_resolve_chunk(const uint8_t*arc, uint64_t root_off,int cz,int cy,int cx){
    if(!root_off) return 0;
    uint64_t node = root_off;
    // sparse levels: nibble index = level+1 (level1 uses nibble2, level0 uses nibble1)
    for(int level=V3_SPARSE_LEVELS-1; level>=0; --level){
        const uint8_t*bm = arc + node;
        int nib=level+1;   // chunk-coord nibble for this sparse level (nibble0 = dense shard)
        int dz=v3_nib(cz,nib), dy=v3_nib(cy,nib), dx=v3_nib(cx,nib);
        int idx=(dz*16+dy)*16+dx;
        if(!v3_bit_get(bm,idx)) return 0;            // sparse pointer-to-0
        int slot=v3_rank(bm,idx);
        uint64_t childoff; memcpy(&childoff, arc+node+V3_BITMAP_BYTES + (size_t)slot*8, 8);
        node = childoff;
    }
    // node = dense shard table; direct-index by nibble 0
    int si=((v3_nib(cz,0))*16 + v3_nib(cy,0))*16 + v3_nib(cx,0);
    uint64_t chunk; memcpy(&chunk, arc+node + (size_t)si*8, 8);
    return chunk;   // 0 = empty chunk in this shard
}

// Chunk blob layout: [u32 masklen][masklen chunk-mask bytes][512B block-bitmap]
// [present lens u32...][payloads]. masklen==0 = no chunk-mask (legacy per-block path).
// Returns pointer to the chunk-mask stream (or NULL) and its length.
static const uint8_t* v3_chunk_mask(const uint8_t*arc, uint64_t chunk_off, uint32_t *mlen){
    uint32_t ml; memcpy(&ml, arc+chunk_off, 4); *mlen=ml;
    return ml ? arc+chunk_off+4 : 0;
}
// block (bz,by,bx) present? if so return 1 + its payload range (offset = cumulative
// len of present blocks before it). 0 = ZERO block (decode to zeros). The block
// directory is sparse — offsets implicit, ZERO blocks cost only their bitmap bit.
static int v3_block_range(const uint8_t*arc, uint64_t chunk_off, int bz,int by,int bx,
                          uint64_t *abs_off, uint32_t *len){
    uint32_t ml; memcpy(&ml, arc+chunk_off, 4);
    uint64_t bm_off = chunk_off + 4 + ml;             // skip [u32 masklen][mask]
    const uint8_t*bm = arc + bm_off;
    int bi=(bz*16+by)*16+bx;
    if(!v3_bit_get(bm,bi)) return 0;                 // ZERO block
    int npresent=0; for(int i=0;i<V3_BITMAP_BYTES;++i) npresent+=__builtin_popcount(bm[i]);
    const uint8_t*lens = arc + bm_off + V3_BITMAP_BYTES;     // u32 per present block
    uint64_t pay_base = bm_off + V3_BITMAP_BYTES + (uint64_t)npresent*4;
    int slot = v3_rank(bm,bi);                        // which present block this is
    uint64_t cum=0; for(int s=0;s<slot;++s){ uint32_t l; memcpy(&l,lens+(size_t)s*4,4); cum+=l; }
    uint32_t mylen; memcpy(&mylen, lens+(size_t)slot*4, 4);
    *abs_off = pay_base + cum; *len = mylen;
    return 1;
}
#endif
