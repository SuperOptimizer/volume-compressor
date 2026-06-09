// v3 archive writer: encode a volume (one LOD) into the sparse-tree + dense-chunk
// format. Uses a growable byte buffer; lays out chunks contiguously, then sparse
// node levels, then header. Self-contained for the lab (encodes blocks via a caller
// callback so it stays decoupled from the codec internals).
#ifndef V3_WRITE_H
#define V3_WRITE_H
#include "v3archive.h"
#include <stdlib.h>
#include <stdio.h>

typedef struct { uint8_t *p; size_t len, cap; } v3buf;
static void v3_reserve(v3buf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
static size_t v3_put(v3buf*b,const void*s,size_t n){ v3_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t v3_zero(v3buf*b,size_t n){ v3_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
static void v3_u64at(v3buf*b,size_t at,uint64_t v){ memcpy(b->p+at,&v,8); }
static void v3_u32at(v3buf*b,size_t at,uint32_t v){ memcpy(b->p+at,&v,4); }

// Caller provides: encode one 16^3 block at chunk-local block coords -> returns 1 if
// nonzero (payload appended to `out` via v3_put), 0 if all-zero (no payload).
typedef int (*v3_enc_block_fn)(void *ud, int lod, int bz,int by,int bx,
                               int blkz,int blky,int blkx, v3buf *out, uint32_t *len_out);
// Caller provides chunk occupancy: is chunk (cz,cy,cx) at lod present (has any nonzero)?
typedef int (*v3_chunk_present_fn)(void *ud, int lod, int cz,int cy,int cx);

// Write one dense chunk (256^3 = 16^3 blocks) contiguously: [blkdir 4096*8B][payloads].
// Returns the chunk blob's file offset. Blocks coded via enc_block; all-zero -> flag.
// SPARSE chunk: encode all 16^3 blocks to a temp, then write
// [512B block-occupancy bitmap][present blocks' u32 len, in idx order][payloads].
// Block offsets are IMPLICIT (cumulative len of present blocks) — saves the 4096×8B
// dense directory. ZERO blocks = unset bitmap bit (the per-block "pointer to 0").
// Returns chunk blob offset (start = the bitmap).
// Optional CHUNK-MASK PROLOGUE: code the whole-chunk (256^3) air mask ONCE as a coherent
// surface (context crossing block edges) before the blocks. Returns bytes written to out.
// When set, per-block enc skips its own mask. NULL = legacy per-block mask.
// PREPARE: build (gather+dilate) the chunk mask into shared state BEFORE blocks encode.
typedef void (*v3_chunk_mask_prep_fn)(void *ud, int lod, int cz,int cy,int cx);
// EMIT: range-code the already-prepared chunk mask into out; returns bytes written.
typedef uint32_t (*v3_chunk_mask_emit_fn)(v3buf *out);
static v3_chunk_mask_prep_fn g_v3_chunk_mask_prep = 0;
static v3_chunk_mask_emit_fn g_v3_chunk_mask_emit = 0;
// STRUCTURE AUDIT byte counters (set g_v3_audit=1): where the non-coefficient bytes go.
static int  g_v3_audit = 0;
static long g_v3_by_blkdir, g_v3_by_shard, g_v3_by_node;   // bitmap+lens / shard tables / sparse nodes

static uint64_t v3_write_chunk(v3buf*b, void*ud, v3_enc_block_fn enc, int lod,int cz,int cy,int cx){
    // CHUNK-MASK: prepare the shared chunk mask FIRST so block encodes see the correct mask.
    if(g_v3_chunk_mask_prep) g_v3_chunk_mask_prep(ud,lod,cz,cy,cx);
    // pass 1: encode into a scratch buffer, record which blocks are nonzero + their len
    static _Thread_local v3buf tmp; tmp.len=0;          // reused scratch (payloads)
    uint8_t bm[V3_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint32_t blen[V3_GRID3]; int npresent=0;
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx; uint32_t len=0;
        int nz=enc(ud,lod,bz,by,bx,cz,cy,cx,&tmp,&len);
        if(nz){ v3_bit_set(bm,bi); blen[bi]=len; npresent++; }
    }
    // chunk blob: [u32 masklen][mask bytes][bitmap][present lens][payloads].
    // masklen is ALWAYS written (0 when no chunk-mask) so the reader knows the layout.
    size_t at=b->len;
    if(g_v3_chunk_mask_emit){
        size_t lenpos=v3_put(b,&(uint32_t){0},4);       // placeholder u32 masklen
        uint32_t ml=g_v3_chunk_mask_emit(b);            // codes the prepared mask
        memcpy(b->p+lenpos,&ml,4);
    } else { v3_put(b,&(uint32_t){0},4); }              // masklen=0 sentinel
    v3_put(b,bm,V3_BITMAP_BYTES);
    for(int bi=0;bi<V3_GRID3;++bi) if(v3_bit_get(bm,bi)) v3_put(b,&blen[bi],4);
    if(npresent) v3_put(b,tmp.p,tmp.len);
    if(g_v3_audit) g_v3_by_blkdir += V3_BITMAP_BYTES + (long)npresent*4;  // per-chunk block directory
    return (uint64_t)at;
}

// A pending sparse node during build: maps present child local-idx -> child file off.
// We build bottom-up: collect present chunks, group into shard nodes, etc.
// For the lab we build per-LOD recursively over the chunk-coord space.
// node level L (L=0 is the SHARD level just above chunks; L=V3_SPARSE_LEVELS-1 = root).
// Returns file offset of the written node, or 0 if the whole subtree is empty.
// nchunks = chunk extent per axis (volume voxels/256, ceil). Recursion CLIPS to
// [0,nchunks) so it never scans the full 2^20 space (OOM). base_* are CHUNK coords.
static int g_v3_nchunks=0;

// DENSE shard: a 16^3 direct table of chunk-blob offsets (0 = empty chunk). base_c* =
// shard-origin chunk coords (multiples of 16). Returns shard table offset, 0 if empty.
static uint64_t v3_write_shard(v3buf*b, void*ud, v3_enc_block_fn enc, v3_chunk_present_fn pres,
                               int lod, int base_cz,int base_cy,int base_cx){
    uint64_t tbl[V3_GRID3]; memset(tbl,0,sizeof tbl); int any=0;
    for(int dz=0;dz<16;++dz){ int cz=base_cz+dz; if(cz>=g_v3_nchunks)break;
    for(int dy=0;dy<16;++dy){ int cy=base_cy+dy; if(cy>=g_v3_nchunks)break;
    for(int dx=0;dx<16;++dx){ int cx=base_cx+dx; if(cx>=g_v3_nchunks)break;
        if(pres(ud,lod,cz,cy,cx)){ uint64_t off=v3_write_chunk(b,ud,enc,lod,cz,cy,cx);
            tbl[(dz*16+dy)*16+dx]=off; any=1; } }}}
    if(!any) return 0;
    if(g_v3_audit) g_v3_by_shard += V3_SHARD_BYTES;
    return (uint64_t)v3_put(b,tbl,V3_SHARD_BYTES);   // dense direct table
}

// SPARSE node (the 2 levels above shards). level: 0 = just-above-shards (children are
// shards), 1 = top (children are level-0 nodes). base_* = CHUNK coords. childspan in
// chunks = 16^(level+1) (a shard spans 16 chunks; each higher level ×16).
static uint64_t v3_write_node(v3buf*b, void*ud, v3_enc_block_fn enc, v3_chunk_present_fn pres,
                              int lod, int level, int base_cz,int base_cy,int base_cx){
    int childspan = 16; for(int i=0;i<level;++i) childspan*=16;   // shard=16 chunks, ×16/level
    uint8_t bitmap[V3_BITMAP_BYTES]; memset(bitmap,0,sizeof bitmap);
    uint64_t childoff[V3_GRID3]; int any=0;
    for(int dz=0;dz<16;++dz){ int ccz=base_cz+dz*childspan; if(ccz>=g_v3_nchunks)break;
    for(int dy=0;dy<16;++dy){ int ccy=base_cy+dy*childspan; if(ccy>=g_v3_nchunks)break;
    for(int dx=0;dx<16;++dx){ int ccx=base_cx+dx*childspan; if(ccx>=g_v3_nchunks)break;
        int idx=(dz*16+dy)*16+dx;
        uint64_t off = (level==0) ? v3_write_shard(b,ud,enc,pres,lod,ccz,ccy,ccx)
                                  : v3_write_node(b,ud,enc,pres,lod,level-1,ccz,ccy,ccx);
        if(off){ v3_bit_set(bitmap,idx); childoff[idx]=off; any=1; }
    }}}
    if(!any) return 0;
    size_t nat=v3_put(b,bitmap,V3_BITMAP_BYTES);
    int nch=0; for(int idx=0;idx<V3_GRID3;++idx) if(v3_bit_get(bitmap,idx)){ v3_put(b,&childoff[idx],8); nch++; }
    if(g_v3_audit) g_v3_by_node += V3_BITMAP_BYTES + (long)nch*8;
    return (uint64_t)nat;
}
#endif
