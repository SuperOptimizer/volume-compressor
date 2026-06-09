// v3 archive test: build a sparse-tree+dense-chunk archive from a synthetic sparse
// volume, verify resolve round-trips (present chunks found, empty regions -> 0), and
// report index overhead. Validates the multi-level map + pointer-to-0 + chunk packing.
#include "v3write.h"
#include "v3read.h"
#include <stdio.h>
#include <stdlib.h>

// Synthetic test "volume": a set of present chunks (sparse). Model a scroll: a
// diagonal band of present chunks in a large coord space (rare high coords).
#define NCHUNKS_AXIS 256   // 2^16 voxels / 256 = 256 chunks/axis (common case)
static int is_present(void*ud,int lod,int cz,int cy,int cx){
    (void)ud;(void)lod;
    if(cz<0||cy<0||cx<0||cz>=NCHUNKS_AXIS||cy>=NCHUNKS_AXIS||cx>=NCHUNKS_AXIS) return 0;
    // sparse "scroll": present where the 3 coords are near a diagonal sheet
    int d = (cz+cy+cx) % 7;
    return (d==0) && ((cz^cy^cx)&1)==0;   // ~7% of chunks present, spatially structured
}
// fake block encoder: ~60% of blocks nonzero, ~40 bytes each (mimics real material)
static int enc_block(void*ud,int lod,int bz,int by,int bx,int cz,int cy,int cx,v3buf*out,uint32_t*len){
    (void)ud;(void)lod;(void)cz;(void)cy;(void)cx;
    int nz = ((bz*7+by*13+bx*17)%5)!=0;   // ~80% nonzero blocks
    if(!nz){*len=0;return 0;}
    uint32_t L=32+((bz+by+bx)%24);        // 32-55 byte payload
    for(uint32_t i=0;i<L;++i){ uint8_t v=(uint8_t)(i+bz+by+bx); v3_put(out,&v,1); }
    *len=L; return 1;
}

int main(void){
    v3buf b={0};
    // header placeholder
    v3_zero(&b, V3_HDR);
    // count present chunks
    long present=0;
    for(int cz=0;cz<NCHUNKS_AXIS;++cz)for(int cy=0;cy<NCHUNKS_AXIS;++cy)for(int cx=0;cx<NCHUNKS_AXIS;++cx)
        if(is_present(0,0,cz,cy,cx)) present++;
    printf("present chunks: %ld of %ld (%.2f%%)\n", present,(long)NCHUNKS_AXIS*NCHUNKS_AXIS*NCHUNKS_AXIS,
           100.0*present/((double)NCHUNKS_AXIS*NCHUNKS_AXIS*NCHUNKS_AXIS));
    size_t payload_start=b.len; extern int g_v3_nchunks; g_v3_nchunks=NCHUNKS_AXIS;
    // build the tree (root at top sparse level), base coord 0
    uint64_t root = v3_write_node(&b,NULL,enc_block,is_present,0,V3_SPARSE_LEVELS-1,0,0,0);
    // header: magic, ver, root[0]
    v3_u32at(&b,V3H_MAGIC,V3_MAGIC); v3_u32at(&b,V3H_VER,V3_VERSION);
    v3_u64at(&b,V3H_ROOTOFF+0*8, root);
    v3_u64at(&b,V3H_TOTLEN, b.len);
    printf("archive: %.2f MB total\n", b.len/1e6);

    // verify resolve round-trips
    long ok=0,bad=0,absent_ok=0;
    for(int cz=0;cz<NCHUNKS_AXIS;cz+=3)for(int cy=0;cy<NCHUNKS_AXIS;cy+=3)for(int cx=0;cx<NCHUNKS_AXIS;cx+=3){
        uint64_t off=v3_resolve_chunk(b.p,root,cz,cy,cx);
        int should=is_present(0,0,cz,cy,cx);
        if(should && off) ok++;
        else if(!should && !off) absent_ok++;
        else bad++;
    }
    printf("resolve: %ld present-found, %ld absent-correct, %ld WRONG\n",ok,absent_ok,bad);

    // verify a block read in a present chunk
    int fcz=-1,fcy=-1,fcx=-1;
    for(int cz=0;cz<NCHUNKS_AXIS&&fcz<0;++cz)for(int cy=0;cy<NCHUNKS_AXIS&&fcz<0;++cy)for(int cx=0;cx<NCHUNKS_AXIS;++cx)
        if(is_present(0,0,cz,cy,cx)){fcz=cz;fcy=cy;fcx=cx;break;}
    uint64_t coff=v3_resolve_chunk(b.p,root,fcz,fcy,fcx);
    uint64_t boff; uint32_t blen; int isd=v3_block_range(b.p,coff,1,2,3,&boff,&blen);
    printf("chunk(%d,%d,%d) blob@%lu; block(1,2,3): %s len=%u\n",fcz,fcy,fcx,(unsigned long)coff, isd?"DCT":"ZERO",blen);

    // index overhead = total - payload bytes
    // (payload bytes = sum of chunk dirs + block payloads; rough: total - sparse nodes)
    printf("payload region starts @%zu; index (nodes) is interleaved bottom-up\n",payload_start);
    return 0;
}
