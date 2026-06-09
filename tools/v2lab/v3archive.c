// ============================================================================
// v3archive.c — FROZEN v3 archive container (2026-06-09). See v3archive_api.h.
// Depends on v2codec (one direction). Format constants in v3archive.h; sparse-tree
// reader in v3read.h. This file owns: zarr chunk-mmap source, LOD decimation, block
// gather, chunk-mask gather, the sparse-tree writer, and the build/decode drivers.
// ============================================================================
#include "v3archive_api.h"
#include "v3archive.h"
#include "v3read.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef uint8_t u8;
#define V3_CHUNK_ALIGN 256

// ---------------------------------------------------------------- volume source
// LOD0: chunk-mmap'd zarr (working-set resident). LOD1+: contiguous decimated buffers.
typedef struct {
    int V, CH, G; const u8 **chunk; size_t *clen;
} vsrc;
static inline u8 vsrc_get(const vsrc *s, int z,int y,int x){
    if((unsigned)z>=(unsigned)s->V||(unsigned)y>=(unsigned)s->V||(unsigned)x>=(unsigned)s->V) return 0;
    int CH=s->CH; int ci=((z/CH)*s->G + (y/CH))*s->G + (x/CH);
    const u8 *c=s->chunk[ci]; if(!c) return 0;
    return c[(((size_t)(z%CH))*CH + (y%CH))*CH + (x%CH)];
}
typedef struct { const u8 *vol; int dim; const vsrc *src; } v3vol;  // vol!=NULL: contiguous; else vsrc
static inline u8 v3vol_get(const v3vol *V, int z,int y,int x){
    if(V->vol){ if((unsigned)z>=(unsigned)V->dim||(unsigned)y>=(unsigned)V->dim||(unsigned)x>=(unsigned)V->dim) return 0;
                return V->vol[((size_t)z*V->dim+y)*V->dim+x]; }
    return vsrc_get(V->src,z,y,x);
}
static vsrc *load_zarr_vsrc(const char *root, int V, int CH){
    if(V % V3_CHUNK_ALIGN != 0){
        fprintf(stderr,"FATAL: volume dim %d is not a multiple of the v3 chunk size %d.\n"
                       "       Pad the volume to a %d-multiple in the EXPORT pipeline before encoding.\n",
                V,V3_CHUNK_ALIGN,V3_CHUNK_ALIGN); exit(2);
    }
    int G=(V+CH-1)/CH; vsrc *s=calloc(1,sizeof *s); s->V=V; s->CH=CH; s->G=G;
    s->chunk=calloc((size_t)G*G*G,sizeof(const u8*)); s->clen=calloc((size_t)G*G*G,sizeof(size_t));
    size_t clen=(size_t)CH*CH*CH; char p[1024]; int present=0;
    for(int c0=0;c0<G;++c0)for(int c1=0;c1<G;++c1)for(int c2=0;c2<G;++c2){
        int ci=(c0*G+c1)*G+c2;
        snprintf(p,sizeof p,"%s/0/%d/%d/%d",root,c0,c1,c2);
        int fd=open(p,O_RDONLY); if(fd<0) continue;
        struct stat st; if(fstat(fd,&st)!=0||(size_t)st.st_size<clen){ close(fd); continue; }
        const u8 *m=mmap(NULL,clen,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
        if(m==MAP_FAILED) continue;
        s->chunk[ci]=m; s->clen[ci]=clen; present++;
    }
    fprintf(stderr,"vsrc: %d/%d chunks mmap'd (CH=%d, V=%d)\n",present,G*G*G,CH,V);
    return s;
}
static void free_vsrc(vsrc *s){ if(!s)return; for(size_t i=0;i<(size_t)s->G*s->G*s->G;++i)
    if(s->chunk[i]) munmap((void*)s->chunk[i],s->clen[i]); free(s->chunk); free(s->clen); free(s); }
// 2x box-decimate (mean of nonzero children; all-zero stays 0). vsrc and contiguous variants.
static u8 *decimate(const u8 *src,int D){ int H=D/2; u8 *o=calloc((size_t)H*H*H,1);
    for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){ int s=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            u8 v=src[(((size_t)(2*z+dz))*D+(2*y+dy))*D+(2*x+dx)]; if(v){s+=v;c++;}}
        o[((size_t)z*H+y)*H+x]=c?(u8)((s+c/2)/c):0; } return o; }
static u8 *decimate_vsrc(const vsrc *s,int D){ int H=D/2; u8 *o=calloc((size_t)H*H*H,1);
    for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){ int sum=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            u8 v=vsrc_get(s,2*z+dz,2*y+dy,2*x+dx); if(v){sum+=v;c++;}}
        o[((size_t)z*H+y)*H+x]=c?(u8)((sum+c/2)/c):0; } return o; }

// gather a 16^3 block from the LOD volume (zero-padded at edges).
static int gather_blk(const v3vol *V,int cz,int cy,int cx,int bz,int by,int bx,u8 *dst){
    int z0=(cz*16+bz)*V2_BLK,y0=(cy*16+by)*V2_BLK,x0=(cx*16+bx)*V2_BLK,any=0;
    for(int z=0;z<V2_BLK;++z)for(int y=0;y<V2_BLK;++y)for(int x=0;x<V2_BLK;++x){
        u8 v=v3vol_get(V,z0+z,y0+y,x0+x); dst[(z*V2_BLK+y)*V2_BLK+x]=v; any|=v; }
    return any;
}
static int chunk_present(const v3vol *V,int cz,int cy,int cx){
    int D=V->dim,z0=cz*256,y0=cy*256,x0=cx*256;
    if(z0>=D||y0>=D||x0>=D) return 0;
    int z1=z0+256<D?z0+256:D,y1=y0+256<D?y0+256:D,x1=x0+256<D?x0+256:D;
    for(int z=z0;z<z1;++z)for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x) if(v3vol_get(V,z,y,x)) return 1;
    return 0;
}

// ---------------------------------------------------------------- archive writer
typedef struct { u8 *p; size_t len, cap; } abuf;
static void a_reserve(abuf*b,size_t n){ if(b->len+n<=b->cap)return; size_t nc=b->cap?b->cap*2:1<<20; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
static size_t a_put(abuf*b,const void*s,size_t n){ a_reserve(b,n); size_t at=b->len; memcpy(b->p+at,s,n); b->len+=n; return at; }
static size_t a_zero(abuf*b,size_t n){ a_reserve(b,n); size_t at=b->len; memset(b->p+at,0,n); b->len+=n; return at; }
static void a_u32(abuf*b,size_t at,uint32_t v){ memcpy(b->p+at,&v,4); }
static void a_u64(abuf*b,size_t at,uint64_t v){ memcpy(b->p+at,&v,8); }

static int g_nchunks=0;
// thread-local prepared chunk air-mask (256^3, air=1) for the chunk being encoded.
static _Thread_local u8 *g_cmask=0;

// gather the 256^3 air mask for chunk (cz,cy,cx) from the volume into g_cmask.
static void prep_chunk_mask(const v3vol *V,int cz,int cy,int cx){
    if(!g_cmask) g_cmask=malloc((size_t)V2_CHUNK*V2_CHUNK*V2_CHUNK);
    int z0=cz*256,y0=cy*256,x0=cx*256;
    for(int z=0;z<V2_CHUNK;++z)for(int y=0;y<V2_CHUNK;++y)for(int x=0;x<V2_CHUNK;++x)
        g_cmask[((size_t)z*V2_CHUNK+y)*V2_CHUNK+x] = v3vol_get(V,z0+z,y0+y,x0+x)?0:1;  // air=1
}

// write one dense 256^3 chunk: [u32 masklen][mask][512B block-bitmap][present lens][payloads].
static uint64_t write_chunk(abuf*b,const v3vol *V,int cz,int cy,int cx){
    prep_chunk_mask(V,cz,cy,cx);
    static _Thread_local v2_buf tmp; tmp.len=0;        // payloads scratch
    uint8_t bm[V3_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint32_t blen[V3_GRID3]; int npresent=0;
    static _Thread_local u8 vox[V2_BLK*V2_BLK*V2_BLK], rair[V2_BLK*V2_BLK*V2_BLK];
    for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
        int bi=(bz*16+by)*16+bx;
        if(!gather_blk(V,cz,cy,cx,bz,by,bx,vox)) continue;            // all-zero -> absent
        // slice this block's reconstructed air mask from the prepared chunk mask
        for(int z=0;z<V2_BLK;++z)for(int y=0;y<V2_BLK;++y)for(int x=0;x<V2_BLK;++x){
            size_t ci=((size_t)(bz*V2_BLK+z)*V2_CHUNK+(by*V2_BLK+y))*V2_CHUNK+(bx*V2_BLK+x);
            rair[(z*V2_BLK+y)*V2_BLK+x]=g_cmask[ci];
        }
        uint32_t len=0; int nzb=v2_enc_block(vox,rair,&tmp,&len);
        if(nzb){ v3_bit_set(bm,bi); blen[bi]=len; npresent++; }
    }
    size_t at=b->len;
    // chunk-mask prologue: [u32 masklen][mask bytes]
    static _Thread_local v2_u8 *mbuf=0; if(!mbuf) mbuf=malloc((size_t)V2_CHUNK*V2_CHUNK*V2_CHUNK/4+1024);
    uint32_t mlen=v2_enc_chunkmask(g_cmask,mbuf,(size_t)V2_CHUNK*V2_CHUNK*V2_CHUNK/4+1024);
    a_put(b,&mlen,4); a_put(b,mbuf,mlen);
    a_put(b,bm,V3_BITMAP_BYTES);
    for(int bi=0;bi<V3_GRID3;++bi) if(v3_bit_get(bm,bi)) a_put(b,&blen[bi],4);
    if(npresent) a_put(b,tmp.p,tmp.len);
    return (uint64_t)at;
}
// dense shard: 16^3 direct table of chunk-blob offsets (0 = empty). base_c* = shard origin.
static uint64_t write_shard(abuf*b,const v3vol *V,int bz,int by,int bx){
    uint64_t tbl[V3_GRID3]; memset(tbl,0,sizeof tbl); int any=0;
    for(int dz=0;dz<16;++dz){ int cz=bz+dz; if(cz>=g_nchunks)break;
    for(int dy=0;dy<16;++dy){ int cy=by+dy; if(cy>=g_nchunks)break;
    for(int dx=0;dx<16;++dx){ int cx=bx+dx; if(cx>=g_nchunks)break;
        if(chunk_present(V,cz,cy,cx)){ tbl[(dz*16+dy)*16+dx]=write_chunk(b,V,cz,cy,cx); any=1; } }}}
    if(!any) return 0;
    return (uint64_t)a_put(b,tbl,V3_SHARD_BYTES);
}
// sparse node (2 levels above shards). level 0 children=shards, level1 children=level0 nodes.
static uint64_t write_node(abuf*b,const v3vol *V,int level,int bz,int by,int bx){
    int span=16; for(int i=0;i<level;++i) span*=16;
    uint8_t bm[V3_BITMAP_BYTES]; memset(bm,0,sizeof bm);
    uint64_t coff[V3_GRID3]; int any=0;
    for(int dz=0;dz<16;++dz){ int cz=bz+dz*span; if(cz>=g_nchunks)break;
    for(int dy=0;dy<16;++dy){ int cy=by+dy*span; if(cy>=g_nchunks)break;
    for(int dx=0;dx<16;++dx){ int cx=bx+dx*span; if(cx>=g_nchunks)break;
        int idx=(dz*16+dy)*16+dx;
        uint64_t off = level==0 ? write_shard(b,V,cz,cy,cx) : write_node(b,V,level-1,cz,cy,cx);
        if(off){ v3_bit_set(bm,idx); coff[idx]=off; any=1; } }}}
    if(!any) return 0;
    size_t nat=a_put(b,bm,V3_BITMAP_BYTES);
    for(int idx=0;idx<V3_GRID3;++idx) if(v3_bit_get(bm,idx)) a_put(b,&coff[idx],8);
    return (uint64_t)nat;
}

int v3_build_from_zarr(const char *root, const char *outpath, int dim, float quality){
    v2_codec_init(); v2_set_quality(quality);
    int V=dim;
    vsrc *vs=load_zarr_vsrc(root,V,128);
    abuf b={0}; a_zero(&b,V3_HDR);
    uint64_t roots[8]={0};
    const u8 *lodvol=NULL; u8 *owned=NULL; int d=V;
    for(int lod=0; lod<8 && d>=V2_BLK; ++lod){
        v3vol vv = lodvol ? (v3vol){lodvol,d,NULL} : (v3vol){NULL,d,vs};
        g_nchunks=(d+255)/256;
        roots[lod]=write_node(&b,&vv,V3_SPARSE_LEVELS-1,0,0,0);
        if(d/2<V2_BLK){ ++lod; break; }
        u8 *next = lodvol ? decimate(lodvol,d) : decimate_vsrc(vs,d);
        if(owned) free(owned); owned=next; lodvol=next; d/=2;
    }
    if(owned) free(owned);
    a_u32(&b,V3H_MAGIC,V3_MAGIC); a_u32(&b,V3H_VER,V3_VERSION);
    a_u32(&b,V3H_NX,V); a_u32(&b,V3H_NY,V); a_u32(&b,V3H_NZ,V);
    for(int l=0;l<8;++l) a_u64(&b,V3H_ROOTOFF+l*8,roots[l]);
    a_u64(&b,V3H_TOTLEN,b.len);
    FILE *of=fopen(outpath,"wb"); if(!of){ perror("fopen out"); free_vsrc(vs); return 1; }
    fwrite(b.p,1,b.len,of); fclose(of);
    free(b.p); free_vsrc(vs);
    return 0;
}

// ---------------------------------------------------------------- reader
struct v3_reader { const uint8_t *arc; size_t len; uint64_t roots[8];
    u8 *cmask; uint64_t cmask_key; };     // cached decoded chunk mask
v3_reader *v3_open(const uint8_t *arc, size_t len, float quality){
    v2_codec_init(); v2_set_quality(quality);
    v3_reader *r=calloc(1,sizeof *r); r->arc=arc; r->len=len; r->cmask_key=~0ull;
    for(int l=0;l<8;++l) memcpy(&r->roots[l], arc+V3H_ROOTOFF+l*8, 8);
    return r;
}
void v3_close(v3_reader *r){ if(!r)return; free(r->cmask); free(r); }
uint64_t v3_chunk_offset(v3_reader *r,int lod,int cz,int cy,int cx){
    if(lod<0||lod>7) return 0; return v3_resolve_chunk(r->arc,r->roots[lod],cz,cy,cx);
}
void v3_decode_block(v3_reader *r, uint64_t chunk_off, int bz,int by,int bx, v2_u8 *dst){
    if(!chunk_off){ memset(dst,0,V2_BLK*V2_BLK*V2_BLK); return; }
    // decode the 256^3 chunk mask once per chunk (cached), slice this block's air.
    if(r->cmask_key!=chunk_off){
        if(!r->cmask) r->cmask=malloc((size_t)V2_CHUNK*V2_CHUNK*V2_CHUNK);
        uint32_t cml; const u8 *cmb=v3_chunk_mask(r->arc,chunk_off,&cml);
        if(cmb) v2_dec_chunkmask(cmb,cml,r->cmask); else memset(r->cmask,0,(size_t)V2_CHUNK*V2_CHUNK*V2_CHUNK);
        r->cmask_key=chunk_off;
    }
    static _Thread_local v2_u8 rair[V2_BLK*V2_BLK*V2_BLK];
    for(int z=0;z<V2_BLK;++z)for(int y=0;y<V2_BLK;++y)for(int x=0;x<V2_BLK;++x){
        size_t ci=((size_t)(bz*V2_BLK+z)*V2_CHUNK+(by*V2_BLK+y))*V2_CHUNK+(bx*V2_BLK+x);
        rair[(z*V2_BLK+y)*V2_BLK+x]=r->cmask[ci];
    }
    uint64_t boff; uint32_t blen;
    if(!v3_block_range(r->arc,chunk_off,bz,by,bx,&boff,&blen)){ memset(dst,0,V2_BLK*V2_BLK*V2_BLK); return; }
    v2_dec_block(r->arc+boff,rair,dst);
}
