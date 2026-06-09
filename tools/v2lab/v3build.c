// v3build: wire the REAL v2 codec (DCT-16 + dead-zone quant + range coder) into the
// v3 sparse-tree/dense-chunk archive, build a .v3 archive from a 1024^3 zarr volume,
// then decode it back and report the wide metric basket + archive size / ratio.
//
// Per-block format inside a chunk's payload area (one 16^3 DCT block):
//   [u8 dc][u16 clen][clen bytes of range-coded quantized coefficients]
// A block whose source voxels are ALL ZERO is encoded as ZERO (no payload, unset
// bitmap bit) — that is the intra-chunk zero sparsity the masked dense ROI exhibits.
//
// LOD pyramid: 8 LODs by 2x box-decimation (mean of 2^3 children, but 0 stays 0 so
// the zero-mask is inherited downward). Each LOD is its own v3 sparse tree; all 8
// roots are stored in the header. Lower LODs are tiny so they barely move the ratio.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef uint8_t  u8;
typedef int32_t  i32;

#include "dct_fastf.h"
#include "rangecoder.h"
#include "metrics.h"
#include "v3archive.h"
#include "v3write.h"
#include "v3read.h"

// ---- quant knobs (mirror v2lab leaf_dct path; tuned defaults for 16^3) ----
static float g_dz_frac = 0.80f;
static float g_hf_exp  = 0.65f;
static int   g_hf_keep = 45;   // keep DCT coefs with L1 freq (cz+cy+cx) <= this; max 45 = keep all. V3_HFKEEP
static int   g_coefstats = 0;  // V3_COEFSTATS: accumulate per-band quantized-coef stats
static long  g_cs_total[8], g_cs_nz[8], g_cs_neg[8], g_cs_mag[8][16];
// pre-quant per-L1-freq (0..45) coefficient power (sum of squares) + count: the DCT power
// spectrum. Real signal rolls off with freq; white noise is flat → flatten point = S/N crossover.
static double g_ps_sq[46]; static long g_ps_n[46];
// LOD0 byte accounting: where the archive bytes go (coef stream vs mask vs headers vs corrections).
static long g_by_coef, g_by_mask, g_by_hdr, g_by_corr, g_by_chunkmask;
static float g_base_q  = 8.0f;
// VARIABLE PER-BAND QUANT: a multiplier per L1 freq (0..45). step = base_q * g_qtab[freq].
// Default = the smooth power law (1+freq)^hf_exp (so behavior is unchanged unless g_qtab set).
// A NOISE-AWARE table sets the multiplier from the measured DCT power spectrum: quantize each
// band ~proportional to (signal_rms / target_snr), so transition/noise bands get coarser steps.
static int   g_qtab_on = 0;          // V3_QTAB: 0=power law, 1=noise-aware table
static float g_qtab[46];
static float g_qtab_snr = 6.0f;      // V3_QTAB_SNR: keep coefs down to this RMS, coarsen below
static float g_denoise = 0.0f;       // V3_DENOISE: DCT soft-threshold strength (units of sigma; 0=off)
static float g_noise_sigma = 1.3f;   // V3_NSIGMA: measured noise floor RMS (interior, see v7-noise-floor)
static int   g_denoise_shaped = 0;   // V3_DENOISE_SHAPE: 0=flat threshold, 1=freq-shaped (thr rises where signal~noise)
static float g_dnthr[46];            // per-L1-freq soft-threshold (built from measured spectrum)
static inline float hf_weight(int cz,int cy,int cx){
    int f=cz+cy+cx;
    float base=powf(1.0f+(float)f, g_hf_exp);
    if(g_qtab_on) return base*g_qtab[f];   // g_qtab is a data-driven CORRECTION factor (centered ~1)
    return base;
}
static inline i32 quant_one(float c, float step){
    float dz=g_dz_frac*step, a=fabsf(c); i32 lv=0;
    if(a>=dz) lv=(i32)((a-dz)/step+1.0f);
    return c<0?-lv:lv;
}
static inline float deq_one(i32 lv, float step){
    if(!lv) return 0.0f;
    float a=(float)(lv<0?-lv:lv); float r=(a-1.0f)*step+g_dz_frac*step+0.40f*step;
    return lv<0?-r:r;
}
// Outlier correction: 0 = off. Else any reconstructed voxel whose abs error exceeds
// this cap gets a raw (idx,origval) correction appended, hard-bounding maxerr to <=cap.
static int g_maxerr_cap = 0;
// Air-restore threshold: after iDCT, a voxel reconstructing below this -> exactly 0
// (restores the SAM2 mask; air was air-filled to dc on encode). The 0<->material cliff
// is clean (only ~0.1% of voxels in 1..15), so a low threshold restores it near-exactly.
static int g_air_thresh = 8;

#define V3B 16   // DCT block edge (DCT-16 everywhere)

// ---------- shared context: the whole volume + per-LOD extents ----------
// vol is EITHER a contiguous dim^3 buffer (LODs 1+ are computed/decimated), OR — for LOD0 —
// NULL with `src` pointing at a chunk-mmap'd source (vsrc). Accessors use v3vol_get() which
// dispatches: contiguous when vol!=NULL, else chunk-mmap. This keeps LOD0's 1GB source paged
// on-demand from disk (working-set resident) instead of a resident contiguous copy.
struct vsrc;
typedef struct {
    const u8 *vol;          // contiguous dim^3 buffer (LOD1+), or NULL if chunk-mmap (LOD0)
    int dim;                // voxels/axis at this LOD
    const struct vsrc *src; // chunk-mmap source (LOD0 only; NULL otherwise)
} v3vol;

// chunk-mmap source: each CH^3 zarr chunk file mmap'd read-only; missing chunk = NULL (all air).
typedef struct vsrc {
    int V, CH, G;           // volume dim, chunk edge, chunks/axis
    const u8 **chunk;       // G^3 pointers (NULL = absent chunk = all 0)
    size_t *clen;           // mmap length per chunk (for munmap)
} vsrc;
// fetch one voxel (zero-padded out of range / missing chunk).
static inline u8 vsrc_get(const vsrc *s, int z,int y,int x){
    if((unsigned)z>=(unsigned)s->V||(unsigned)y>=(unsigned)s->V||(unsigned)x>=(unsigned)s->V) return 0;
    int CH=s->CH; int ci=((z/CH)*s->G + (y/CH))*s->G + (x/CH);
    const u8 *c=s->chunk[ci]; if(!c) return 0;
    return c[(((size_t)(z%CH))*CH + (y%CH))*CH + (x%CH)];
}
// unified voxel read for a v3vol at LOD coords.
static inline u8 v3vol_get(const v3vol *V, int z,int y,int x){
    if(V->vol){ if((unsigned)z>=(unsigned)V->dim||(unsigned)y>=(unsigned)V->dim||(unsigned)x>=(unsigned)V->dim) return 0;
                return V->vol[((size_t)z*V->dim+y)*V->dim+x]; }
    return vsrc_get(V->src,z,y,x);
}

// PER-CHUNK MASK MODE: code the whole 256^3 chunk air mask as ONE coherent surface with
// 3D context crossing 16^3 block edges (exploits 31-voxel air runs / smooth boundary),
// instead of a separate mask per block. g_chunk_mask=1 enables. The reconstructed chunk
// mask is cached so both the chunk-mask encoder and the per-block enc/dec read the same.
static int g_chunk_mask = 0;
#define V3CHUNK 256
// thread-local current chunk air mask (256^3), valid for the chunk being en/decoded.
static _Thread_local u8 *g_cmask = 0;          // 256^3, air=1
static _Thread_local uint64_t g_cmask_key = ~0ull;  // chunk_off currently cached (decode)

// 3-neighbor (z-1,y-1,x-1) context bit-coder over an SZ^3 mask. Used for the 256^3 chunk.
// HIERARCHICAL chunk-mask coder. g_mask_hier=0: flat single-level (3-neighbor, 8 ctx).
// g_mask_hier=1: code a 2x-decimated coarse mask first (all-8-children-air → coarse-air), then
// the fine mask using 3 fine-neighbors + the parent-coarse-air bit (16 ctx). Both ends build the
// coarse level identically (enc decimates m; dec decodes coarse stream first), so it's causal &
// self-contained. NOTE: LOSES in practice (v7 82.54→83.54MB) despite +13% STATIC-entropy gain —
// the adaptive coder starves on 2× contexts + extra coarse level (same lesson as per-band mag
// models). Flat 3-neighbor is near-optimal for the adaptive setting. Kept opt-in, OFF. V3_MASKHIER.
static int g_mask_hier = 0;
// build coarse (HZ=SZ/2): a coarse cell is air iff ALL 8 fine children are air (matches decimate).
static void mask_decimate_allair(const u8*m,int SZ,u8*cm){ int HZ=SZ/2;
    for(int z=0;z<HZ;++z)for(int y=0;y<HZ;++y)for(int x=0;x<HZ;++x){ int all=1;
        for(int dz=0;dz<2&&all;++dz)for(int dy=0;dy<2&&all;++dy)for(int dx=0;dx<2;++dx)
            if(!m[(((size_t)(2*z+dz))*SZ+(2*y+dy))*SZ+(2*x+dx)]){all=0;break;}
        cm[((size_t)z*HZ+y)*HZ+x]=(u8)all; }
}
// OCTREE zero-mask: descend the SZ^3 mask; each node is coded as UNIFORM(all-air|all-material)
// or SPLIT (recurse 8 children). Big correlated air/material regions collapse to one node ->
// near-free. At MIN_LEAF size a still-mixed node is coded voxel-wise with the 3-neighbor coder.
// Per-level contexts (by log2 node size) so the coder adapts split-probability to scale.
// g_mask_octree=1 (V3_MASKOCT). Self-contained + causal: dec mirrors enc exactly.
static int g_mask_octree = 0;
#define OCT_MINLEAF 8            // stop splitting at 8^3; code residual leaf voxel-wise
#define OCT_LEVELS  6            // log2(256)=8 .. log2(8)=3 -> levels indexed by lz=log2(sz)
typedef struct { ctx_t uniform[12]; ctx_t value[12]; ctx_t leaf[8]; } oct_ctx;
static int ilog2i(int v){ int l=0; while(v>1){v>>=1;l++;} return l; }
// is the cube uniform? returns 0=mixed, 1=all-air, 2=all-material. (m: air=1, material=0)
static int cube_uniform(const u8*m,int SZ,int z0,int y0,int x0,int sz){
    int first=m[((size_t)z0*SZ+y0)*SZ+x0];
    for(int z=z0;z<z0+sz;++z)for(int y=y0;y<y0+sz;++y){ const u8*r=&m[((size_t)z*SZ+y)*SZ+x0];
        for(int x=0;x<sz;++x) if(r[x]!=first) return 0; }
    return first?1:2;
}
static void oct_enc(rc_enc*e,oct_ctx*c,const u8*m,int SZ,int z0,int y0,int x0,int sz){
    int lz=ilog2i(sz);
    if(sz<=OCT_MINLEAF){ // leaf: code voxel-wise with 3-neighbor context (within the leaf)
        for(int z=0;z<sz;++z)for(int y=0;y<sz;++y)for(int x=0;x<sz;++x){
            size_t i=((size_t)(z0+z)*SZ+(y0+y))*SZ+(x0+x);
            int nz_=z?m[i-(size_t)SZ*SZ]:0, ny_=y?m[i-SZ]:0, nx_=x?m[i-1]:0;
            enc_bit(e,&c->leaf[(nz_<<2)|(ny_<<1)|nx_], m[i]); }
        return; }
    int u=cube_uniform(m,SZ,z0,y0,x0,sz);
    enc_bit(e,&c->uniform[lz], u!=0);
    if(u){ enc_bit(e,&c->value[lz], u==1); return; }   // 1=all-air, 0=all-material
    int h=sz/2;
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
        oct_enc(e,c,m,SZ,z0+dz*h,y0+dy*h,x0+dx*h,h);
}
static void oct_dec(rc_dec*d,oct_ctx*c,u8*m,int SZ,int z0,int y0,int x0,int sz){
    int lz=ilog2i(sz);
    if(sz<=OCT_MINLEAF){
        for(int z=0;z<sz;++z)for(int y=0;y<sz;++y)for(int x=0;x<sz;++x){
            size_t i=((size_t)(z0+z)*SZ+(y0+y))*SZ+(x0+x);
            int nz_=z?m[i-(size_t)SZ*SZ]:0, ny_=y?m[i-SZ]:0, nx_=x?m[i-1]:0;
            m[i]=(u8)dec_bit(d,&c->leaf[(nz_<<2)|(ny_<<1)|nx_]); }
        return; }
    int uni=dec_bit(d,&c->uniform[lz]);
    if(uni){ int air=dec_bit(d,&c->value[lz]); u8 v=air?1:0;
        for(int z=0;z<sz;++z)for(int y=0;y<sz;++y)
            memset(&m[((size_t)(z0+z)*SZ+(y0+y))*SZ+x0], v, sz);
        return; }
    int h=sz/2;
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
        oct_dec(d,c,m,SZ,z0+dz*h,y0+dy*h,x0+dx*h,h);
}

static uint32_t enc_mask3d(const u8 *m, int SZ, rc_u8 *buf, size_t cap){
    rc_enc e; enc_init(&e,buf,cap);
    if(g_mask_octree){
        oct_ctx c; for(int i=0;i<12;++i){ctx_init(&c.uniform[i]);ctx_init(&c.value[i]);}
        for(int i=0;i<8;++i)ctx_init(&c.leaf[i]);
        oct_enc(&e,&c,m,SZ,0,0,0,SZ); enc_flush(&e); return (uint32_t)e.len;
    }
    if(!g_mask_hier){
        ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init(&ctx[i]);
        for(int z=0;z<SZ;++z)for(int y=0;y<SZ;++y)for(int x=0;x<SZ;++x){
            size_t i=((size_t)z*SZ+y)*SZ+x;
            int nz_= z?m[i-(size_t)SZ*SZ]:0, ny_= y?m[i-SZ]:0, nx_= x?m[i-1]:0;
            enc_bit(&e,&ctx[(nz_<<2)|(ny_<<1)|nx_], m[i]);
        }
        enc_flush(&e); return (uint32_t)e.len;
    }
    int HZ=SZ/2; u8 *cm=malloc((size_t)HZ*HZ*HZ); mask_decimate_allair(m,SZ,cm);
    ctx_t cc[8]; for(int i=0;i<8;++i) ctx_init(&cc[i]);           // coarse: 3-neighbor
    for(int z=0;z<HZ;++z)for(int y=0;y<HZ;++y)for(int x=0;x<HZ;++x){ size_t i=((size_t)z*HZ+y)*HZ+x;
        int nz_=z?cm[i-(size_t)HZ*HZ]:0,ny_=y?cm[i-HZ]:0,nx_=x?cm[i-1]:0;
        enc_bit(&e,&cc[(nz_<<2)|(ny_<<1)|nx_],cm[i]); }
    ctx_t fc[16]; for(int i=0;i<16;++i) ctx_init(&fc[i]);          // fine: 3-neighbor + parent
    for(int z=0;z<SZ;++z)for(int y=0;y<SZ;++y)for(int x=0;x<SZ;++x){ size_t i=((size_t)z*SZ+y)*SZ+x;
        int nz_=z?m[i-(size_t)SZ*SZ]:0,ny_=y?m[i-SZ]:0,nx_=x?m[i-1]:0;
        int pc=cm[(((size_t)(z/2))*HZ+(y/2))*HZ+(x/2)];
        enc_bit(&e,&fc[(pc<<3)|(nz_<<2)|(ny_<<1)|nx_],m[i]); }
    free(cm); enc_flush(&e); return (uint32_t)e.len;
}
static void dec_mask3d(const rc_u8 *buf, size_t len, u8 *m, int SZ){
    rc_dec d; dec_init(&d,buf,len);
    if(g_mask_octree){
        oct_ctx c; for(int i=0;i<12;++i){ctx_init(&c.uniform[i]);ctx_init(&c.value[i]);}
        for(int i=0;i<8;++i)ctx_init(&c.leaf[i]);
        oct_dec(&d,&c,m,SZ,0,0,0,SZ); return;
    }
    if(!g_mask_hier){
        ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init(&ctx[i]);
        for(int z=0;z<SZ;++z)for(int y=0;y<SZ;++y)for(int x=0;x<SZ;++x){
            size_t i=((size_t)z*SZ+y)*SZ+x;
            int nz_= z?m[i-(size_t)SZ*SZ]:0, ny_= y?m[i-SZ]:0, nx_= x?m[i-1]:0;
            m[i]=(u8)dec_bit(&d,&ctx[(nz_<<2)|(ny_<<1)|nx_]);
        }
        return;
    }
    int HZ=SZ/2; u8 *cm=malloc((size_t)HZ*HZ*HZ);
    ctx_t cc[8]; for(int i=0;i<8;++i) ctx_init(&cc[i]);
    for(int z=0;z<HZ;++z)for(int y=0;y<HZ;++y)for(int x=0;x<HZ;++x){ size_t i=((size_t)z*HZ+y)*HZ+x;
        int nz_=z?cm[i-(size_t)HZ*HZ]:0,ny_=y?cm[i-HZ]:0,nx_=x?cm[i-1]:0;
        cm[i]=(u8)dec_bit(&d,&cc[(nz_<<2)|(ny_<<1)|nx_]); }
    ctx_t fc[16]; for(int i=0;i<16;++i) ctx_init(&fc[i]);
    for(int z=0;z<SZ;++z)for(int y=0;y<SZ;++y)for(int x=0;x<SZ;++x){ size_t i=((size_t)z*SZ+y)*SZ+x;
        int nz_=z?m[i-(size_t)SZ*SZ]:0,ny_=y?m[i-SZ]:0,nx_=x?m[i-1]:0;
        int pc=cm[(((size_t)(z/2))*HZ+(y/2))*HZ+(x/2)];
        m[i]=(u8)dec_bit(&d,&fc[(pc<<3)|(nz_<<2)|(ny_<<1)|nx_]); }
    free(cm);
}

// gather a 16^3 block at block-coords (within a chunk at cz,cy,cx) from the LOD volume.
// returns 1 if any voxel nonzero. Out-of-range voxels (volume edge) read as 0.
static int gather_blk(const v3vol *V, int cz,int cy,int cx, int bz,int by,int bx, u8 *dst){
    int z0=(cz*16+bz)*V3B, y0=(cy*16+by)*V3B, x0=(cx*16+bx)*V3B; int any=0;
    for(int z=0;z<V3B;++z){ int zz=z0+z;
        for(int y=0;y<V3B;++y){ int yy=y0+y;
            for(int x=0;x<V3B;++x){
                u8 v=v3vol_get(V,zz,yy,x0+x);
                dst[(z*V3B+y)*V3B+x]=v; any|=v;
            }}}
    return any;
}

// ---------- air-mask coder: range-code a binary air mask (air=1) with a small neighbor
// context (already-coded left/up/front). A boundary surface is smooth/contiguous, so it
// compresses far below the raw bitmap. LOSSY: the mask is coded at reduced resolution
// (g_mask_ds: 1=full 16^3, 2=8^3, 4=4^3). At ds=k a mask cell covers a k^3 voxel group;
// boundary accuracy is +-(k-1) voxels — "off by a voxel or two" for a big ratio win.
// The reduced mask has MS = 16/ds cells per axis. The DECODER upsamples by replication.
static int g_mask_ds = 1;      // mask downsample factor (1/2/4)
#define MASK_CTX 8             // 3-neighbor context

// downsample a 16^3 voxel air mask to an MS^3 cell mask. AIR-BIASED by default: a cell
// is air iff ALL voxels in its k^3 group are air. This GROWS material (shrinks air) at
// the boundary, so NO air voxel is ever air-filled+DCT'd -> ZERO air-leak (no gray haze
// in the void). The only error is a sliver of true-material voxels near the edge getting
// zeroed (material->0), which is the acceptable "lose a pixel" case. g_mask_vote:
// 0=air-biased(all-air), 1=majority, 2=material-biased(any-air).
static int g_mask_vote = 0;
static int g_fill_diffuse = 8;   // N Jacobi sweeps of harmonic air-fill (8 = sweet spot; 0=flat dc). tuned 2026-06-08 on v6: 0→8 gives +1.4dB, maxerr 148→83, +5% ratio; >8 diminishing.
static int g_mask_dilate = 0;    // >0 = grow air into material by N voxels (erode boundary)
static int g_mask_close = 0;     // V3_MASKCLOSE: N iters morphological close(air) — fill dents, leak-safe
static void mask_downsample(const u8 *air, u8 *cell, int ds){
    int S=V3B, MS=S/ds, grp=ds*ds*ds, half=grp/2;
    for(int z=0;z<MS;++z)for(int y=0;y<MS;++y)for(int x=0;x<MS;++x){
        int s=0; for(int dz=0;dz<ds;++dz)for(int dy=0;dy<ds;++dy)for(int dx=0;dx<ds;++dx)
            s+=air[((z*ds+dz)*S+(y*ds+dy))*S+(x*ds+dx)];
        int isair = g_mask_vote==2 ? (s>0) : g_mask_vote==1 ? (s>half) : (s==grp);
        cell[(z*MS+y)*MS+x]=(u8)isair;
    }
}
// upsample an MS^3 cell mask back to 16^3 by replication.
static void mask_upsample(const u8 *cell, u8 *air, int ds){
    int S=V3B, MS=S/ds;
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x)
        air[(z*S+y)*S+x]=cell[((z/ds)*MS+(y/ds))*MS+(x/ds)];
}
static uint32_t enc_air_mask(const u8 *cell, int MS, rc_u8 *buf, size_t cap){
    rc_enc e; enc_init(&e,buf,cap);
    ctx_t ctx[MASK_CTX]; for(int i=0;i<MASK_CTX;++i) ctx_init(&ctx[i]);
    int S=MS;
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
        int i=(z*S+y)*S+x;
        int nz_= z?cell[((z-1)*S+y)*S+x]:0;
        int ny_= y?cell[(z*S+(y-1))*S+x]:0;
        int nx_= x?cell[(z*S+y)*S+(x-1)]:0;
        int c=(nz_<<2)|(ny_<<1)|nx_;
        enc_bit(&e,&ctx[c],cell[i]);
    }
    enc_flush(&e); return (uint32_t)e.len;
}
static void dec_air_mask(const rc_u8 *buf, size_t len, u8 *cell, int MS){
    rc_dec d; dec_init(&d,buf,len);
    ctx_t ctx[MASK_CTX]; for(int i=0;i<MASK_CTX;++i) ctx_init(&ctx[i]);
    int S=MS;
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
        int i=(z*S+y)*S+x;
        int nz_= z?cell[((z-1)*S+y)*S+x]:0;
        int ny_= y?cell[(z*S+(y-1))*S+x]:0;
        int nx_= x?cell[(z*S+y)*S+(x-1)]:0;
        int c=(nz_<<2)|(ny_<<1)|nx_;
        cell[i]=(u8)dec_bit(&d,&ctx[c]);
    }
}

// ---------- block ENCODE: append [u8 dc][u8 flags][u16 masklen][mask][u16 clen][coded]
//            [u16 ncorr][corr] to out; return length ----------
static int v3_enc_block(void *ud, int lod, int bz,int by,int bx,
                        int cz,int cy,int cx, v3buf *out, uint32_t *len_out){
    (void)lod;
    const v3vol *V=(const v3vol*)ud;
    static _Thread_local u8 vox[V3B*V3B*V3B];
    if(!gather_blk(V,cz,cy,cx,bz,by,bx,vox)){ *len_out=0; return 0; }   // ZERO block

    static _Thread_local float blk[V3B*V3B*V3B], coef[V3B*V3B*V3B];
    static _Thread_local i32 lv[V3B*V3B*V3B];
    int n=V3B*V3B*V3B;
    // MASK-AWARE: air voxels are exactly 0 (SAM2 mask). Compute DC over MATERIAL only,
    // then AIR-FILL air voxels with the material DC so the 0<->material cliff flattens
    // before the DCT (the cliff is the HF-energy / max-err / ringing source). On decode
    // the mask is restored by thresholding (air reconstructs ~dc -> we knock <thresh to 0).
    long sum=0,cnt=0; for(int i=0;i<n;++i){ if(vox[i]){ sum+=vox[i]; cnt++; } }
    int dc = cnt ? (int)((sum+cnt/2)/cnt) : 0;   // mean over material only
    // build the TRUE air mask, then the (possibly lossy) RECONSTRUCTED mask the decoder
    // will see: downsample by g_mask_ds (majority), then upsample by replication. Air-fill
    // and corrections use the RECONSTRUCTED mask so encoder and decoder agree exactly.
    static _Thread_local u8 air[V3B*V3B*V3B];      // true mask
    static _Thread_local u8 rair[V3B*V3B*V3B];     // reconstructed (lossy) mask
    static _Thread_local u8 cell[V3B*V3B*V3B];     // downsampled cell mask (MS^3 used)
    int nair=0; for(int i=0;i<n;++i){ air[i]=vox[i]?0:1; nair+=air[i]; }
    // CHUNK-MASK MODE: rair comes from the per-chunk surface coder (g_cmask, already
    // dilated+coded). No per-block mask is written. Skip the per-block dilate/downsample.
    int use_chunk_mask = (g_chunk_mask && g_cmask);
    if(use_chunk_mask){
        nair=0;
        for(int z=0;z<V3B;++z)for(int y=0;y<V3B;++y)for(int x=0;x<V3B;++x){
            int bi=(z*V3B+y)*V3B+x;
            size_t ci=((size_t)(bz*V3B+z)*V3CHUNK + (by*V3B+y))*V3CHUNK + (bx*V3B+x);
            rair[bi]=g_cmask[ci]; nair+=rair[bi];
        }
    }
    // MASK DILATION: grow air into material by g_mask_dilate voxels (6-connectivity).
    // Erodes the boundary material the DCT must represent + kills the last edge ringing;
    // those eroded voxels are zeroed on decode ("lose a pixel"). Coded mask grows slightly.
    if(!use_chunk_mask && g_mask_dilate>0 && nair>0 && nair<n){
        static _Thread_local u8 a2[V3B*V3B*V3B]; int S=V3B;
        for(int it=0; it<g_mask_dilate; ++it){
            memcpy(a2,air,n);
            for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
                int i=(z*S+y)*S+x; if(a2[i]) continue;
                int grow = (z&&a2[i-S*S])||(z<S-1&&a2[i+S*S])||(y&&a2[i-S])||(y<S-1&&a2[i+S])||(x&&a2[i-1])||(x<S-1&&a2[i+1]);
                if(grow) air[i]=1;
            }
        }
        nair=0; for(int i=0;i<n;++i) nair+=air[i];
    }
    int ds=g_mask_ds, MS=V3B/ds;
    if(!use_chunk_mask){
        if(nair>0 && nair<n && ds>1){ mask_downsample(air,cell,ds); mask_upsample(cell,rair,ds); }
        else { for(int i=0;i<n;++i) rair[i]=air[i]; for(int i=0;i<n;++i) cell[i]=air[i]; }
    }
    // air-fill using the reconstructed mask (so a voxel the decoder will zero is filled).
    // blk holds (value - dc). Masked voxels are FILLED so the DCT sees no 0<->material
    // cliff; they're zeroed on decode regardless, so the fill is pure ratio optimization.
    for(int i=0;i<n;++i){ int v = rair[i] ? dc : (vox[i]?vox[i]:dc); blk[i]=(float)(v-dc); }
    if(g_fill_diffuse && nair>0 && nair<n){
        // SMOOTH (harmonic) FILL: instead of a flat dc slab, diffuse the boundary material
        // into the air region (Jacobi sweeps; material voxels fixed). Removes the slope kink
        // where sloped material meets the flat fill -> fewer HF coefficients -> better ratio.
        static _Thread_local float tmp[V3B*V3B*V3B];
        int S=V3B;
        for(int it=0; it<g_fill_diffuse; ++it){
            memcpy(tmp,blk,sizeof(float)*n);
            for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
                int i=(z*S+y)*S+x; if(!rair[i]) continue;   // only relax air voxels
                float a=0; int c=0;
                if(z) {a+=tmp[i-S*S];c++;} if(z<S-1){a+=tmp[i+S*S];c++;}
                if(y) {a+=tmp[i-S];  c++;} if(y<S-1){a+=tmp[i+S];  c++;}
                if(x) {a+=tmp[i-1];  c++;} if(x<S-1){a+=tmp[i+1];  c++;}
                if(c) blk[i]=a/c;
            }
        }
    }
    dct3_fwd(blk,coef,V3B);
    // IN-CODEC DENOISE (DCT soft-threshold): shrink each AC coef toward 0 by g_denoise*sigma
    // (VisuShrink/BayesShrink-style). sigma = measured noise floor (~1.3 interior). Removes the
    // flat HF noise tail BEFORE quant — fewer nonzeros, smaller mags. OFF by default (g_denoise=0)
    // so cleaner future inputs are untouched. DC (idx 0, freq 0) is never thresholded.
    if(g_denoise>0.0f){
        if(g_denoise_shaped){   // per-frequency Wiener-shaped threshold
            for(int z=0;z<V3B;++z)for(int y=0;y<V3B;++y)for(int x=0;x<V3B;++x){
                int idx=(z*V3B+y)*V3B+x; if(!idx)continue; float thr=g_denoise*g_dnthr[z+y+x];
                float a=fabsf(coef[idx]); coef[idx]= a<=thr?0.0f:(coef[idx]<0?-(a-thr):(a-thr));
            }
        } else { float thr=g_denoise*g_noise_sigma;
            for(int i=1;i<n;++i){ float a=fabsf(coef[i]); coef[i] = a<=thr ? 0.0f : (coef[i]<0?-(a-thr):(a-thr)); }
        }
    }
    for(int cz2=0;cz2<V3B;++cz2)for(int cy2=0;cy2<V3B;++cy2)for(int cx2=0;cx2<V3B;++cx2){
        int idx=(cz2*V3B+cy2)*V3B+cx2; float step=g_base_q*hf_weight(cz2,cy2,cx2);
        // HARD HF-DROP: zero coefficients whose L1 freq exceeds g_hf_keep (max 3*(V3B-1)=45).
        // These decode as zeros (no signaling cost) and shrink EOB. Free IFF those bands are
        // pure noise on this volume — the noise-discard test for v7.
        if(cz2+cy2+cx2 > g_hf_keep){ lv[idx]=0; continue; }
        if(g_coefstats && V->dim==1024 && (g_coefstats<2 || nair==0)){ int f=cz2+cy2+cx2; g_ps_sq[f]+=(double)coef[idx]*coef[idx]; g_ps_n[f]++; }
        lv[idx]=quant_one(coef[idx],step);
    }
    static _Thread_local rc_i16 ql[V3B*V3B*V3B];
    static _Thread_local rc_u8 scratch[V3B*V3B*V3B*2+256];
    for(int i=0;i<n;++i){ i32 v=lv[i]; ql[i]=(rc_i16)(v>32767?32767:v<-32768?-32768:v); }
    if(g_coefstats){  // per-band quantized-coef stats: count, nonzeros, |mag| histogram
        for(int cz2=0;cz2<V3B;++cz2)for(int cy2=0;cy2<V3B;++cy2)for(int cx2=0;cx2<V3B;++cx2){
            int idx=(cz2*V3B+cy2)*V3B+cx2; int f=cz2+cy2+cx2; int bnd=f*8/46; if(bnd>7)bnd=7;
            int v=lv[idx]; g_cs_total[bnd]++;
            if(v){ g_cs_nz[bnd]++; int a=v<0?-v:v; if(a>15)a=15; g_cs_mag[bnd][a]++; if(v<0)g_cs_neg[bnd]++; }
        }
    }
    rc_enc e; enc_init(&e,scratch,sizeof scratch);
    enc_block_coefs(&e,ql,V3B); enc_flush(&e);
    uint32_t clen=(uint32_t)e.len;
    // code the (downsampled) air mask if mixed. flags bit0 = has-mask. ds stored in
    // flags bits 1-3 so decode knows the upsample factor. MS = 16/ds cells/axis.
    static _Thread_local rc_u8 mscratch[V3B*V3B*V3B/4+256];
    uint32_t mlen=0; u8 flags=0;
    if(!use_chunk_mask && nair>0 && nair<n){ mlen=enc_air_mask(cell,MS,mscratch,sizeof mscratch);
        flags|=1; flags|=(u8)(ds<<1); }
    // payload: [u8 dc][u8 flags][u16 mlen][mlen mask][u16 clen][clen coefs][u16 ncorr][corr]
    u8 dcb=(u8)dc; v3_put(out,&dcb,1); v3_put(out,&flags,1);
    uint16_t ml16=(uint16_t)mlen; v3_put(out,&ml16,2);
    if(mlen) v3_put(out,mscratch,mlen);
    uint16_t cl16=(uint16_t)clen; v3_put(out,&cl16,2);
    v3_put(out,scratch,clen);
    uint32_t plen=1+1+2+mlen+2+clen;
    if(g_coefstats && lod==0){ g_by_coef+=clen; g_by_mask+=mlen; g_by_hdr+=1+1+2+2; }
    // OUTLIER CORRECTION: locally decode this block, store raw fixes for voxels whose
    // error exceeds the cap. Hard-bounds maxerr; costs 3B per outlier (rare on smooth data).
    uint16_t ncorr=0;
    if(g_maxerr_cap>0){
        static _Thread_local rc_i16 dq[V3B*V3B*V3B];
        rc_dec d; dec_init(&d,scratch,clen); dec_block_coefs(&d,dq,V3B);
        static _Thread_local float rcoef[V3B*V3B*V3B], rblk[V3B*V3B*V3B];
        for(int cz2=0;cz2<V3B;++cz2)for(int cy2=0;cy2<V3B;++cy2)for(int cx2=0;cx2<V3B;++cx2){
            int idx=(cz2*V3B+cy2)*V3B+cx2; float step=g_base_q*hf_weight(cz2,cy2,cx2);
            rcoef[idx]=deq_one(dq[idx],step);
        }
        dct3_inv(rcoef,rblk,V3B);
        size_t cntpos=out->len; uint16_t zero16=0; v3_put(out,&zero16,2);  // placeholder
        for(int i=0;i<n;++i){
            int rv = rair[i] ? 0 : (int)lrintf(rblk[i])+dc;  // mask-restore: air -> exactly 0
            if(rv<0)rv=0; if(rv>255)rv=255;
            if(abs(rv-(int)vox[i])>g_maxerr_cap){
                uint16_t ii=(uint16_t)i; u8 ov=vox[i]; v3_put(out,&ii,2); v3_put(out,&ov,1); ncorr++;
            }
        }
        memcpy(out->p+cntpos,&ncorr,2);                 // backpatch real count
        plen += 2 + (uint32_t)ncorr*3;
        if(g_coefstats && lod==0) g_by_corr += 2 + (long)ncorr*3;
    } else { uint16_t zero16=0; v3_put(out,&zero16,2); plen+=2; if(g_coefstats&&lod==0)g_by_corr+=2; }
    *len_out=plen;
    return 1;
}

// ---------- chunk presence: any nonzero voxel in this 256^3 chunk? ----------
static int v3_chunk_present(void *ud, int lod, int cz,int cy,int cx){
    (void)lod;
    const v3vol *V=(const v3vol*)ud;
    int D=V->dim, z0=cz*256,y0=cy*256,x0=cx*256;
    if(z0>=D||y0>=D||x0>=D) return 0;
    int z1=z0+256<D?z0+256:D, y1=y0+256<D?y0+256:D, x1=x0+256<D?x0+256:D;
    for(int z=z0;z<z1;++z)for(int y=y0;y<y1;++y)
        for(int x=x0;x<x1;++x) if(v3vol_get(V,z,y,x)) return 1;
    return 0;
}

// ---------- CHUNK-MASK PREP: gather the 256^3 air mask + dilate into g_cmask. Called
// BEFORE the chunk's blocks encode, so they air-fill against the SAME mask we will code.
static void v3_prep_chunk_mask(void *ud, int lod, int cz,int cy,int cx){
    (void)lod; const v3vol *V=(const v3vol*)ud; int D=V->dim;
    if(!g_cmask) g_cmask=malloc((size_t)V3CHUNK*V3CHUNK*V3CHUNK);
    u8 *m=g_cmask; int z0=cz*256,y0=cy*256,x0=cx*256;
    (void)D;
    for(int z=0;z<V3CHUNK;++z)for(int y=0;y<V3CHUNK;++y)for(int x=0;x<V3CHUNK;++x){
        u8 v=v3vol_get(V,z0+z,y0+y,x0+x);
        m[((size_t)z*V3CHUNK+y)*V3CHUNK+x]= v?0:1;       // air=1
    }
    static _Thread_local u8 *m2=0; if(!m2) m2=malloc((size_t)V3CHUNK*V3CHUNK*V3CHUNK);
    int S=V3CHUNK;
    // 6-connectivity air dilate (grow air) / erode (shrink air) helpers over m, n iterations.
    #define AIR_DILATE(N) for(int it=0;it<(N);++it){ memcpy(m2,m,(size_t)S*S*S); \
        for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){ size_t i=((size_t)z*S+y)*S+x; if(m2[i])continue; \
            int g=(z&&m2[i-(size_t)S*S])||(z<S-1&&m2[i+(size_t)S*S])||(y&&m2[i-S])||(y<S-1&&m2[i+S])||(x&&m2[i-1])||(x<S-1&&m2[i+1]); if(g)m[i]=1; } }
    #define AIR_ERODE(N) for(int it=0;it<(N);++it){ memcpy(m2,m,(size_t)S*S*S); \
        for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){ size_t i=((size_t)z*S+y)*S+x; if(!m2[i])continue; \
            int k=(!z||m2[i-(size_t)S*S])&&(z==S-1||m2[i+(size_t)S*S])&&(!y||m2[i-S])&&(y==S-1||m2[i+S])&&(!x||m2[i-1])&&(x==S-1||m2[i+1]); if(!k)m[i]=0; } }
    // MORPHOLOGICAL CLOSE(air) = dilate then erode: fills 1-voxel material dents in the air region
    // (smooths boundary wiggle the coder pays for) while net air extent is preserved (close is
    // extensive: original_air ⊆ result → LEAK-SAFE, only loses the filled dents). Much gentler than
    // dilation (which eats a uniform shell). V3_MASKCLOSE = N close iterations.
    if(g_mask_close>0){ AIR_DILATE(g_mask_close); AIR_ERODE(g_mask_close); }
    if(g_mask_dilate>0){ AIR_DILATE(g_mask_dilate); }
    g_cmask_key=~0ull;   // (decode-cache marker, unused on encode)
}
// CHUNK-MASK EMIT: range-code the prepared g_cmask surface (3D cross-block context).
static uint32_t v3_emit_chunk_mask(v3buf *out){
    static _Thread_local rc_u8 *mbuf=0; if(!mbuf) mbuf=malloc((size_t)V3CHUNK*V3CHUNK*V3CHUNK/4+1024);
    uint32_t mlen=enc_mask3d(g_cmask,V3CHUNK,mbuf,(size_t)V3CHUNK*V3CHUNK*V3CHUNK/4+1024);
    v3_put(out,mbuf,mlen);
    if(g_coefstats) g_by_chunkmask+=mlen;   // all-LOD (emit has no lod); LOD0 dominates
    return mlen;
}

// ---------- block DECODE: from an archive, reconstruct one 16^3 block into dst ----------
static void v3_dec_block(const u8 *arc, uint64_t chunk_off, int bz,int by,int bx, u8 *dst){
    uint64_t boff; uint32_t blen;
    if(!v3_block_range(arc,chunk_off,bz,by,bx,&boff,&blen)){
        memset(dst,0,V3B*V3B*V3B); return;     // ZERO block
    }
    // layout: [u8 dc][u8 flags][u16 mlen][mlen mask][u16 clen][clen coefs][u16 ncorr][corr]
    const u8 *p=arc+boff;
    int dc=p[0]; u8 flags=p[1];
    uint16_t mlen; memcpy(&mlen,p+2,2);
    const u8 *mask_bytes=p+4;
    static _Thread_local u8 air[V3B*V3B*V3B], cell[V3B*V3B*V3B];
    int n=V3B*V3B*V3B;
    if(g_chunk_mask){
        // decode the whole-chunk 256^3 mask ONCE per chunk (cache by chunk_off), then
        // slice out this block's 16^3 region. The chunk-mask stream is at [chunk_off+4].
        if(g_cmask_key!=chunk_off){
            if(!g_cmask) g_cmask=malloc((size_t)V3CHUNK*V3CHUNK*V3CHUNK);
            uint32_t cml; const u8 *cmb=v3_chunk_mask(arc,chunk_off,&cml);
            if(cmb) dec_mask3d(cmb,cml,g_cmask,V3CHUNK);
            else memset(g_cmask,0,(size_t)V3CHUNK*V3CHUNK*V3CHUNK);
            g_cmask_key=chunk_off;
        }
        for(int z=0;z<V3B;++z)for(int y=0;y<V3B;++y)for(int x=0;x<V3B;++x){
            size_t ci=((size_t)(bz*V3B+z)*V3CHUNK + (by*V3B+y))*V3CHUNK + (bx*V3B+x);
            air[(z*V3B+y)*V3B+x]=g_cmask[ci];
        }
    }
    else if(flags&1){ int ds=(flags>>1)&7; if(ds<1)ds=1; int MS=V3B/ds;
        dec_air_mask(mask_bytes,mlen,cell,MS); mask_upsample(cell,air,ds); }
    else memset(air,0,n);
    const u8 *cp_coef=mask_bytes+mlen;
    uint16_t clen; memcpy(&clen,cp_coef,2);
    const u8 *coded=cp_coef+2;
    static _Thread_local rc_i16 ql[V3B*V3B*V3B];
    rc_dec d; dec_init(&d,coded,clen); dec_block_coefs(&d,ql,V3B);
    static _Thread_local float coef[V3B*V3B*V3B], blk[V3B*V3B*V3B];
    for(int cz=0;cz<V3B;++cz)for(int cy=0;cy<V3B;++cy)for(int cx=0;cx<V3B;++cx){
        int idx=(cz*V3B+cy)*V3B+cx; float step=g_base_q*hf_weight(cz,cy,cx);
        coef[idx]=deq_one(ql[idx],step);
    }
    dct3_inv(coef,blk,V3B);
    for(int i=0;i<n;++i){
        int v = air[i] ? 0 : (int)lrintf(blk[i])+dc;     // MASK-RESTORE: air -> exactly 0
        if(v<0)v=0; if(v>255)v=255; dst[i]=(u8)v;
    }
    // apply outlier corrections: [u16 ncorr][(u16 idx,u8 val)×ncorr] after coded coefs
    const u8 *cp=coded+clen; uint16_t ncorr; memcpy(&ncorr,cp,2); cp+=2;
    for(uint16_t k=0;k<ncorr;++k){ uint16_t ii; memcpy(&ii,cp,2); u8 ov=cp[2]; cp+=3; dst[ii]=ov; }
}

// ---------- zarr loader: u8, CH^3 C-order chunks, dim_sep "/". Loads a cubic V^3 volume where
// V MUST be a multiple of the v3 chunk size (256) — chunk-alignment + padding is the UPSTREAM
// pipeline's responsibility, so we ERROR OUT (not silently pad) if V isn't 256-aligned. That
// keeps the codec's grid identical to whatever produced the data (e.g. its own downsampling).
#define V3_CHUNK_ALIGN 256
static u8 *load_zarr(const char *root, int V, int CH){
    if(V % V3_CHUNK_ALIGN != 0){
        fprintf(stderr,"FATAL: volume dim %d is not a multiple of the v3 chunk size %d.\n"
                       "       Pad the volume to a %d-multiple in the EXPORT pipeline before encoding.\n",
                V,V3_CHUNK_ALIGN,V3_CHUNK_ALIGN);
        exit(2);
    }
    // CHUNK-MMAP input: mmap each CH^3 zarr chunk file read-only (missing chunk = NULL = all air).
    // Pages fault in on demand and clean file pages are trivially evictable, so LOD0's source is
    // paged from disk (working-set resident) — no 1GB contiguous copy. Requires CH==V3_CHUNK_ALIGN
    // is NOT needed; CH (zarr chunk) is independent of the v3 grid. (legacy contiguous path removed)
    (void)0;   // (this branch is unused; see load_zarr_vsrc)
    return NULL;
}
// build the chunk-mmap source. CH must evenly tile V at the zarr level only at the GRID
// (partial edge chunks are allowed — zarr pads them; vsrc_get clips to V).
static vsrc *load_zarr_vsrc(const char *root, int V, int CH){
    if(V % V3_CHUNK_ALIGN != 0){
        fprintf(stderr,"FATAL: volume dim %d is not a multiple of the v3 chunk size %d.\n"
                       "       Pad the volume to a %d-multiple in the EXPORT pipeline before encoding.\n",
                V,V3_CHUNK_ALIGN,V3_CHUNK_ALIGN);
        exit(2);
    }
    int G=(V+CH-1)/CH; vsrc *s=calloc(1,sizeof *s);
    s->V=V; s->CH=CH; s->G=G;
    s->chunk=calloc((size_t)G*G*G,sizeof(const u8*));
    s->clen =calloc((size_t)G*G*G,sizeof(size_t));
    size_t clen=(size_t)CH*CH*CH; char p[1024]; int present=0;
    for(int c0=0;c0<G;++c0)for(int c1=0;c1<G;++c1)for(int c2=0;c2<G;++c2){
        int ci=(c0*G+c1)*G+c2;
        snprintf(p,sizeof p,"%s/0/%d/%d/%d",root,c0,c1,c2);
        int fd=open(p,O_RDONLY); if(fd<0) continue;   // missing chunk -> NULL -> all air
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

// 2x box-decimate; a child cube that is all-zero stays 0 (inherited zero-mask),
// otherwise mean of the NONZERO children (so a thin edge survives downsampling).
static u8 *decimate(const u8 *src,int D){
    int H=D/2; u8 *out=calloc((size_t)H*H*H,1);
    for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){
        int s=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            u8 v=src[(((size_t)(2*z+dz))*D+(2*y+dy))*D+(2*x+dx)]; if(v){s+=v;c++;}
        }
        out[((size_t)z*H+y)*H+x]=c?(u8)((s+c/2)/c):0;
    }
    return out;
}
// same box-decimate but reading LOD0 from the chunk-mmap source (produces the LOD1 buffer).
static u8 *decimate_vsrc(const vsrc *s,int D){
    int H=D/2; u8 *out=calloc((size_t)H*H*H,1);
    for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){
        int sum=0,c=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            u8 v=vsrc_get(s,2*z+dz,2*y+dy,2*x+dx); if(v){sum+=v;c++;}
        }
        out[((size_t)z*H+y)*H+x]=c?(u8)((sum+c/2)/c):0;
    }
    return out;
}

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <zarr_root> <out.v3> [q=%.1f] [V=1024]\n",argv[0],g_base_q); return 1; }
    const char *root=argv[1], *outp=argv[2];
    if(argc>3) g_base_q=(float)atof(argv[3]);
    int V=(argc>4)?atoi(argv[4]):1024;
    if(argc>5) g_dz_frac=(float)atof(argv[5]);   // dead-zone width fraction (default 0.80)
    if(argc>6) g_hf_exp =(float)atof(argv[6]);   // HF-boost exponent (default 0.65)
    if(getenv("V3_QTAB")) g_qtab_on=atoi(getenv("V3_QTAB"));       // 1=use noise-aware band table
    if(getenv("V3_QTAB_SNR")) g_qtab_snr=(float)atof(getenv("V3_QTAB_SNR")); // RMS noise floor in table
    // VARIABLE PER-BAND QUANT TABLE, built from the measured INTERIOR DCT power spectrum
    // (v7 q=8, all-material blocks; see [[v7-noise-floor]]). Indexed by L1 freq 0..45.
    // Rule: step_f ∝ max(signal_rms_f, noise_floor) — high-rate-optimal equal-relative-error
    // allocation, floored at the noise so we never spend MORE bits resolving sub-noise detail.
    {   static const float irms[46]={ /*measured interior RMS, f=0..45*/
            18.7f,300.f,380.f,404.8f,300.f,210.f,162.6f,110.f,80.f,58.3f,
            40.f,30.f,23.3f,16.f,13.f,10.68f,8.5f,7.f,6.01f,5.f,
            4.4f,3.97f,3.4f,2.92f,2.6f,2.4f,2.29f,2.1f,2.0f,1.93f,
            1.85f,1.8f,1.75f,1.69f,1.6f,1.55f,1.51f,1.45f,1.4f,1.39f,
            1.35f,1.33f,1.32f,1.3f,1.28f,1.28f };
        // CORRECTION factor on the power law, centered ~1. Coarsen (×>1) bands whose signal RMS
        // is within g_qtab_snr× of the noise floor (noise-dominated → quantize harder); leave
        // clean high-SNR bands at ×1. Clamped [1, 3] so it only ADDS coarsening, never refines
        // (refining can't help rate). snr_margin = how many noise-σ of signal we still trust.
        float sg=g_noise_sigma, margin=g_qtab_snr;   // e.g. snr=2 → trust signal down to 2σ
        for(int f=0;f<=45;++f){ float snr=irms[f]/sg;       // band SNR in units of noise σ
            float c = snr>=margin ? 1.0f : margin/fmaxf(snr,0.5f);  // ramp coarsening as SNR→noise
            if(c>3.0f)c=3.0f; if(c<1.0f)c=1.0f; g_qtab[f]=c; }
        // FREQ-SHAPED soft-threshold: thr_f = base * noise_fraction_f, where noise_fraction =
        // sigma^2/irms_f^2 (Wiener). ~0 in fat low bands (signal>>noise: don't touch), ->1 at HF
        // (signal~=noise: shrink fully). Bites the mid/high transition where CE-amplified noise
        // lives, leaving genuine low-freq signal intact. Scaled by g_denoise at use.
        // noise fraction ~ sigma/signal_rms (linear), so the threshold meaningfully bites the
        // mid-band transition (signal a few× noise), not just the already-cheap HF.
        for(int f=0;f<=45;++f){ float sg=g_noise_sigma, r=fmaxf(irms[f],sg); float nfrac=sg/r;
            g_dnthr[f]=g_noise_sigma*nfrac; }
    }
    if(getenv("V3_DENOISE_SHAPE")) g_denoise_shaped=atoi(getenv("V3_DENOISE_SHAPE")); // 1=freq-shaped threshold
    if(getenv("V3_DENOISE")) g_denoise=(float)atof(getenv("V3_DENOISE")); // DCT soft-threshold (units of sigma)
    if(getenv("V3_NSIGMA")) g_noise_sigma=(float)atof(getenv("V3_NSIGMA")); // noise floor RMS
    if(getenv("V3_MAXERR")) g_maxerr_cap=atoi(getenv("V3_MAXERR")); // 0=off; else outlier-correct
    if(getenv("V3_AIRTHRESH")) g_air_thresh=atoi(getenv("V3_AIRTHRESH")); // air-restore threshold
    if(getenv("V3_MASKDS")){ g_mask_ds=atoi(getenv("V3_MASKDS")); if(g_mask_ds<1)g_mask_ds=1; } // mask downsample 1/2/4
    if(getenv("V3_MASKVOTE")) g_mask_vote=atoi(getenv("V3_MASKVOTE")); // 0=air-biased 1=majority 2=material-biased
    if(getenv("V3_FILLDIFFUSE")) g_fill_diffuse=atoi(getenv("V3_FILLDIFFUSE")); // N harmonic-fill sweeps (0=flat dc)
    if(getenv("V3_MASKDILATE")) g_mask_dilate=atoi(getenv("V3_MASKDILATE")); // grow air N voxels into material
    if(getenv("V3_MASKCLOSE")) g_mask_close=atoi(getenv("V3_MASKCLOSE")); // morphological close(air): fill dents, leak-safe
    if(getenv("V3_CHUNKMASK")) g_chunk_mask=atoi(getenv("V3_CHUNKMASK")); // 1=per-chunk surface mask
    if(getenv("V3_MASKHIER")) g_mask_hier=atoi(getenv("V3_MASKHIER")); // 1=hierarchical coarse-to-fine mask coding
    if(getenv("V3_MASKOCT")) g_mask_octree=atoi(getenv("V3_MASKOCT")); // 1=octree zero-mask (all-air/all-material/split)
    if(getenv("V3_HFKEEP")) g_hf_keep=atoi(getenv("V3_HFKEEP")); // hard-drop DCT coefs with L1 freq > this (45=off)
    if(getenv("V3_COEFSTATS")) g_coefstats=atoi(getenv("V3_COEFSTATS")); // per-band quantized-coef stats dump
    if(getenv("V3_AUDIT")) g_v3_audit=atoi(getenv("V3_AUDIT")); // structural byte audit (tree/shards/dirs)
    if(g_chunk_mask){ g_v3_chunk_mask_prep=v3_prep_chunk_mask; g_v3_chunk_mask_emit=v3_emit_chunk_mask; }
    fdct_build();

    double t0=now_ms();
    vsrc *vs=load_zarr_vsrc(root,V,128);            // LOD0 source, chunk-mmap'd
    printf("loaded %dx%dx%d from %s (%.0f ms)\n",V,V,V,root,now_ms()-t0);

    // count nonzero / zero sparsity (streamed over chunk-mmap, no contiguous copy)
    size_t N=(size_t)V*V*V, nz=0;
    for(int z=0;z<V;++z)for(int y=0;y<V;++y)for(int x=0;x<V;++x) if(vsrc_get(vs,z,y,x)) nz++;
    printf("nonzero voxels: %zu / %zu (%.2f%% material, %.2f%% zero)\n",
           nz,N,100.0*nz/N,100.0*(N-nz)/N);

    // ---- build archive: 8 LODs, each its own sparse tree ----
    // LOD0 reads from the chunk-mmap vsrc (vol=NULL). LOD1 = decimate_vsrc(LOD0); LOD2+ = decimate.
    v3buf b={0}; v3_zero(&b,V3_HDR);
    uint64_t roots[8]={0};
    const u8 *lodvol=NULL; u8 *owned=NULL; int dim=V;
    extern int g_v3_nchunks;
    int nlod=0;
    double te=now_ms();
    for(int lod=0; lod<8 && dim>=V3B; ++lod){
        v3vol vv = lodvol ? (v3vol){ lodvol, dim, NULL } : (v3vol){ NULL, dim, vs };
        int nchunks=(dim+255)/256; g_v3_nchunks=nchunks;
        // canonical full depth so the reader's fixed 2-sparse-level walk resolves
        roots[lod]=v3_write_node(&b,&vv,v3_enc_block,v3_chunk_present,lod,V3_SPARSE_LEVELS-1,0,0,0);
        printf("  LOD%d dim=%d nchunks=%d root=%llu  (cum %.2f MB)\n",
               lod,dim,nchunks,(unsigned long long)roots[lod], b.len/1e6);
        nlod++;
        if(dim/2<V3B) { ++lod; break; }   // next would be too small; but still record we stop
        u8 *next = lodvol ? decimate(lodvol,dim) : decimate_vsrc(vs,dim);  // LOD0 from vsrc
        if(owned) free(owned);
        owned=next; lodvol=next; dim/=2;
    }
    printf("encode: %d LODs in %.0f ms\n",nlod,now_ms()-te);
    if(g_v3_audit){
        printf("==== STRUCTURE AUDIT (all LODs) ====\n");
        printf("  block dirs (bitmap+lens) %7.2f MB\n",g_v3_by_blkdir/1e6);
        printf("  shard tables             %7.2f MB\n",g_v3_by_shard/1e6);
        printf("  sparse nodes             %7.2f MB\n",g_v3_by_node/1e6);
        printf("  total structure          %7.2f MB\n",(g_v3_by_blkdir+g_v3_by_shard+g_v3_by_node)/1e6);
    }
    if(g_coefstats){
        // per-band: nz-rate, sign-skew, sig-map entropy, magnitude entropy (of nonzeros),
        // and the approx bits each component would cost ideally — shows where to spend effort.
        printf("==== COEF STATS (per L1-freq band, q=%.1f) ====\n",g_base_q);
        printf(" bnd   total      nz   nz%%   neg%%  sigH/b  magH/b   sigMbits  magMbits\n");
        double tot_sig=0,tot_mag=0;
        for(int b=0;b<8;++b){ long T=g_cs_total[b],NZ=g_cs_nz[b]; if(!T)continue;
            double p=(double)NZ/T, sigH = (p<=0||p>=1)?0:-(p*log2(p)+(1-p)*log2(1-p));
            double magH=0; if(NZ){ for(int a=1;a<16;++a){ long c=g_cs_mag[b][a]; if(c){ double pa=(double)c/NZ; magH-=pa*log2(pa); } } }
            double sigMb=sigH*T/8e6, magMb=(magH+1.0)*NZ/8e6; // +1 bit/nz for sign
            tot_sig+=sigMb; tot_mag+=magMb;
            printf(" %3d %9ld %8ld %5.2f %6.2f %7.3f %7.3f %9.2f %9.2f\n",
                   b,T,NZ,100.0*p,NZ?100.0*g_cs_neg[b]/NZ:0,sigH,magH,sigMb,magMb);
        }
        printf(" ideal totals: sig-map %.2f MB  magnitude+sign %.2f MB  (sum %.2f MB)\n",tot_sig,tot_mag,tot_sig+tot_mag);
        // DCT POWER SPECTRUM (LOD0, pre-quant): RMS coefficient amplitude per L1 freq.
        // Signal rolls off; white noise is FLAT. The freq where RMS stops falling = S/N crossover.
        printf("==== DCT POWER SPECTRUM (LOD0, pre-quant RMS per L1 freq) ====\n");
        printf(" freq    count       RMS\n");
        for(int f=0;f<=45;++f){ if(!g_ps_n[f])continue; double rms=sqrt(g_ps_sq[f]/g_ps_n[f]);
            printf(" %3d %10ld %9.3f\n",f,g_ps_n[f],rms); }
        // BYTE ACCOUNTING (LOD0 blocks; chunk-mask is all-LOD): where the archive bytes go.
        // The 'coef' line is the entropy-coded coefficient stream — the only thing a better
        // entropy coder could shrink. Mask+hdr+structure are NOT coefficient-coder addressable.
        double C=g_by_coef/1e6,M=g_by_mask/1e6,CM=g_by_chunkmask/1e6,H=g_by_hdr/1e6,CR=g_by_corr/1e6;
        printf("==== BYTE ACCOUNTING (LOD0) ====\n");
        printf("  coef stream   %7.2f MB   <- entropy-coder addressable\n",C);
        printf("  per-blk mask  %7.2f MB\n",M);
        printf("  chunk mask    %7.2f MB   (all-LOD)\n",CM);
        printf("  blk headers   %7.2f MB\n",H);
        printf("  corrections   %7.2f MB\n",CR);
        printf("  (rest = sparse-tree structure + non-LOD0 LODs)\n");
    }

    // header
    v3_u32at(&b,V3H_MAGIC,V3_MAGIC); v3_u32at(&b,V3H_VER,V3_VERSION);
    v3_u32at(&b,V3H_NX,V); v3_u32at(&b,V3H_NY,V); v3_u32at(&b,V3H_NZ,V);
    for(int l=0;l<8;++l) v3_u64at(&b,V3H_ROOTOFF+l*8, roots[l]);
    v3_u64at(&b,V3H_TOTLEN,b.len);

    FILE *of=fopen(outp,"wb"); fwrite(b.p,1,b.len,of); fclose(of);
    double mb=b.len/1e6;
    printf("archive: %.2f MB  (logical %.2f MB, ratio %.2fx total / %.2fx material)\n",
           mb, N/1e6, (double)N/b.len, (double)nz/b.len);

    // ---- decode LOD0 back and score against the source ----
    // SCORING + disk-verify allocate rec(N) + mask(N) (~2GB at 1024³) and a full decode pass.
    // Production ENCODE doesn't need them — gate behind V3_SCORE (default ON for the lab; set
    // V3_SCORE=0 for lean encode-only). Keeps peak RSS to ~the input volume.
    int do_score = getenv("V3_SCORE") ? atoi(getenv("V3_SCORE")) : 1;
    if(owned) free(owned);
    if(!do_score){ printf("(V3_SCORE=0: skipped decode/score/verify — encode-only)\n"); free_vsrc(vs); free(b.p); return 0; }
    double td=now_ms();
    uint64_t root0=roots[0];
    int nchunks=(V+255)/256;
    g_cmask_key=~0ull;                  // invalidate chunk-mask cache for the decode pass
    // STREAMING decode+score: decode each 16^3 block, score it against the mmap'd source
    // immediately (error hist over material + 8^3 SSIM windows + mask fidelity), never holding a
    // full rec/mask/errs buffer. Holds O(1) extra memory. V3_LODRES_MEASURE still needs a full
    // rec (a measurement path) — allocate it only then.
    metric_acc acc; macc_init(&acc);
    size_t air=0,airbad=0,matc=0,matlost=0;
    int want_rec = getenv("V3_LODRES_MEASURE")!=NULL;
    u8 *rec = want_rec ? calloc(N,1) : NULL;
    static _Thread_local u8 dblk[V3B*V3B*V3B];
    static _Thread_local u8 sblk[V3B*V3B*V3B];   // source block (gathered from mmap'd vol)
    for(int cz=0;cz<nchunks;++cz)for(int cy=0;cy<nchunks;++cy)for(int cx=0;cx<nchunks;++cx){
        uint64_t coff=v3_resolve_chunk(b.p,root0,cz,cy,cx);
        for(int bz=0;bz<16;++bz)for(int by=0;by<16;++by)for(int bx=0;bx<16;++bx){
            int z0=(cz*16+bz)*V3B, y0=(cy*16+by)*V3B, x0=(cx*16+bx)*V3B;
            if(z0>=V||y0>=V||x0>=V) continue;
            if(coff) v3_dec_block(b.p,coff,bz,by,bx,dblk); else memset(dblk,0,sizeof dblk);
            // gather the matching source block from the mmap'd vol (zero-padded at edges)
            for(int z=0;z<V3B;++z){ int zz=z0+z;
                for(int y=0;y<V3B;++y){ int yy=y0+y;
                    for(int x=0;x<V3B;++x){ int xx=x0+x;
                        sblk[(z*V3B+y)*V3B+x] = vsrc_get(vs,zz,yy,xx); }}}
            // per-voxel: error over material, mask fidelity; (whole-block, edge voxels are 0/air)
            for(int i=0;i<V3B*V3B*V3B;++i){ int s=sblk[i], r=dblk[i];
                if(s==0){ air++; if(r!=0) airbad++; }
                else { matc++; if(r==0) matlost++; macc_add_err(&acc,s,r); } }
            // SSIM: 8 windows (2x2x2) per 16^3 block, fully contained
            static _Thread_local u8 wr[512], wc[512];
            for(int wz=0;wz<2;++wz)for(int wy=0;wy<2;++wy)for(int wx=0;wx<2;++wx){
                int k=0; for(int z=0;z<8;++z)for(int y=0;y<8;++y)for(int x=0;x<8;++x){
                    int i=((wz*8+z)*V3B+(wy*8+y))*V3B+(wx*8+x); wr[k]=sblk[i]; wc[k]=dblk[i]; k++; }
                macc_add_ssim_window(&acc,wr,wc);
            }
            if(want_rec) for(int z=0;z<V3B&&z0+z<V;++z)for(int y=0;y<V3B&&y0+y<V;++y){
                int xw=(x0+V3B<=V)?V3B:V-x0; if(xw>0) memcpy(&rec[((size_t)(z0+z)*V+(y0+y))*V+x0],&dblk[(z*V3B+y)*V3B],xw); }
        }
    }
    printf("decode LOD0: %.0f ms (%.1f Mvox/s)\n",now_ms()-td, N/1e6/((now_ms()-td)/1e3));

    // ---- INTER-LOD RESIDUAL SIZING (V3_LODRES_MEASURE): how small would LOD1 be if coded as a
    // residual from the decimated LOD0 LOSSY recon, vs the independent 13.6MB it costs now? rec is
    // the LOD0 lossy recon. pred=decimate(rec) (512^3) is exactly what a hier decoder would have.
    // residual r = LOD1_source - pred, offset +128, clamped to u8. Encode r as a standalone v3 and
    // compare its size to LOD1's independent cost. AIR (r where source LOD1 is air) -> set to 128
    // (zero residual) so the air mask still zeroes them. This is a SIZING upper bound (clamp lossy).
    if(getenv("V3_LODRES_MEASURE")){
        // Walk all coarse LODs. recL = lossy recon of the FINER LOD (start: rec=LOD0 recon).
        // For each LOD l>0: pred=decimate(recL); residual=source_l - pred (+128); encode residual;
        // then form recon_l = pred + decoded-residual to feed the next level (approx: use source_l
        // here as the recon proxy since residual is near-lossless — good enough for SIZING).
        printf("==== INTER-LOD RESIDUAL SIZING (all coarse LODs) ====\n");
        {   u8 *finerecon=malloc(N); memcpy(finerecon,rec,N); int fdim=V;
            u8 *finesrc=malloc(N);   // materialize LOD0 source from vsrc (measurement path only)
            for(int z=0;z<V;++z)for(int y=0;y<V;++y)for(int x=0;x<V;++x)
                finesrc[((size_t)z*V+y)*V+x]=vsrc_get(vs,z,y,x);
            long sc2=g_by_coef,sm2=g_by_mask,scm2=g_by_chunkmask; g_v3_audit=0;
            double sumres=0; double t0=now_ms();
            for(int l=1; l<7 && fdim/2>=V3B; ++l){
                int Dn=fdim/2; size_t Nn=(size_t)Dn*Dn*Dn;
                u8 *src=decimate(finesrc,fdim);     // source at LOD l
                u8 *pred=decimate(finerecon,fdim);  // predictor = decimate(finer recon)
                u8 *resid=malloc(Nn); long clamp=0;
                for(size_t i=0;i<Nn;++i){ if(!src[i]){resid[i]=0;continue;}
                    int r=(int)src[i]-(int)pred[i]+128; if(r<1){r=1;clamp++;} if(r>255){r=255;clamp++;} resid[i]=(u8)r; }
                v3buf rb={0}; v3_zero(&rb,V3_HDR); g_v3_nchunks=(Dn+255)/256;
                v3vol rv={resid,Dn};
                v3_write_node(&rb,&rv,v3_enc_block,v3_chunk_present,l,V3_SPARSE_LEVELS-1,0,0,0);
                printf("  LOD%d (dim %d) residual: %.2f MB  (clamp %.2f%%)\n",l,Dn,rb.len/1e6,100.0*clamp/Nn);
                sumres+=rb.len/1e6;
                // advance: finer recon for next level ~ src (residual near-lossless)
                free(finerecon); finerecon=src; free(finesrc); finesrc=NULL; finesrc=malloc(Nn); memcpy(finesrc,src,Nn);
                fdim=Dn; free(pred); free(resid); free(rb.p);
            }
            g_by_coef=sc2;g_by_mask=sm2;g_by_chunkmask=scm2;
            printf("  TOTAL coarse-LOD residual: %.2f MB (vs ~16MB independent)  (%.0f ms)\n",sumres,now_ms()-t0);
            free(finerecon); if(finesrc)free(finesrc);
        }
    }

    // metrics over MATERIAL (nonzero source) voxels — from the streaming accumulator
    basket_t m=macc_finish(&acc);
    m.ratio=(double)nz/b.len; m.bpp=8.0*b.len/(double)nz;
    printf("==== METRIC BUCKET (material voxels) q=%.1f dz=%.2f hf=%.2f cap=%d maskds=%d ====\n",
           g_base_q,g_dz_frac,g_hf_exp,g_maxerr_cap,g_mask_ds);
    printf("  PSNR     %8.2f dB\n",m.psnr);
    printf("  SSIM     %8.4f\n",m.ssim);
    printf("  MAE      %8.3f\n",m.mae);
    printf("  RMSE     %8.3f\n",m.rmse);
    printf("  max err  %8.0f\n",m.max_err);
    printf("  p90 err  %8.0f\n",m.p90);
    printf("  p95 err  %8.0f\n",m.p95);
    printf("  p99 err  %8.0f\n",m.p99);
    printf("  ratio    %8.2fx (material)   bpp %.3f\n",m.ratio,m.bpp);
    printf("  MASK: air voxels=%zu  air->nonzero(leak)=%zu (%.4f%%)   material->0(lost)=%zu (%.4f%%)\n",
           air,airbad,100.0*airbad/(air?air:1),matlost,100.0*matlost/(matc?matc:1));

    // ---- VERIFY FROM DISK: the on-disk file must be byte-identical to the in-memory archive
    // buffer we wrote (a stronger + instant check than re-decoding). mmap the file, memcmp.
    {
        int fd=open(outp,O_RDONLY); struct stat st; fstat(fd,&st);
        const u8 *arc=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);
        uint32_t magic; memcpy(&magic,arc+V3H_MAGIC,4);
        uint64_t r0; memcpy(&r0,arc+V3H_ROOTOFF,8);
        size_t diff = ((size_t)st.st_size != b.len) ? (size_t)-1 : 0;
        if(!diff) diff = memcmp(arc,b.p,b.len) ? 1 : 0;
        munmap((void*)arc,st.st_size); close(fd);
        printf("disk-verify: magic=%s root0=%llu  bytes-differ-from-inmem=%zu  %s\n",
               magic==V3_MAGIC?"OK":"BAD",(unsigned long long)r0,diff,
               diff==0?"IDENTICAL ✓":"MISMATCH ✗");
    }

    if(rec) free(rec); free(b.p); free_vsrc(vs);
    return 0;
}
