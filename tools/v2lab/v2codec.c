// ============================================================================
// v2codec.c — FROZEN v2 block codec implementation. See v2codec.h / codec-freeze.md.
// ============================================================================
#include "v2codec.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dct_fastf.h"     // dct3_fwd / dct3_inv / fdct_build (integer separable DCT-16)
#include "rangecoder.h"    // enc_*/dec_* CABAC range coder + block-coef + bit coders

#define N3 (V2_BLK*V2_BLK*V2_BLK)

// ---- the one runtime parameter ----
static float g_base_q = 8.0f;
void  v2_set_quality(float q){ g_base_q = q; }
float v2_get_quality(void){ return g_base_q; }

void v2_codec_init(void){ fdct_build(); }

// ---- growable output buffer ----
void v2_buf_put(v2_buf *b, const void *s, size_t n){
    if(b->len+n > b->cap){ size_t nc=b->cap?b->cap*2:1<<16; while(nc<b->len+n)nc*=2; b->p=realloc(b->p,nc); b->cap=nc; }
    memcpy(b->p+b->len,s,n); b->len+=n;
}

// ---- frozen quant: dead-zone, step = base_q*(1+L1freq)^HF_EXP ----
static inline float hf_weight(int cz,int cy,int cx){ return powf(1.0f+(float)(cz+cy+cx), V2_HF_EXP); }
static inline v2_i32 quant_one(float c, float step){
    float dz=V2_DZ_FRAC*step, a=fabsf(c); v2_i32 lv=0;
    if(a>=dz) lv=(v2_i32)((a-dz)/step+1.0f);
    return c<0?-lv:lv;
}
static inline float deq_one(v2_i32 lv, float step){
    if(!lv) return 0.0f;
    float a=(float)(lv<0?-lv:lv); float r=(a-1.0f)*step+V2_DZ_FRAC*step+0.40f*step;
    return lv<0?-r:r;
}

// ---- chunk-mask surface coder: 3-neighbor (z-1,y-1,x-1) context bit coder over a 256^3
// air mask. A boundary surface is smooth/contiguous so this compresses far below the raw
// bitmap. FROZEN: flat single-level 3-neighbor — beat octree + hierarchical in testing
// (the adaptive coder pools statistics; structural fragmentation loses).
uint32_t v2_enc_chunkmask(const v2_u8 *m, v2_u8 *buf, size_t cap){
    rc_enc e; enc_init(&e,buf,cap);
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init(&ctx[i]);
    int SZ=V2_CHUNK;
    for(int z=0;z<SZ;++z)for(int y=0;y<SZ;++y)for(int x=0;x<SZ;++x){
        size_t i=((size_t)z*SZ+y)*SZ+x;
        int nz_= z?m[i-(size_t)SZ*SZ]:0, ny_= y?m[i-SZ]:0, nx_= x?m[i-1]:0;
        enc_bit(&e,&ctx[(nz_<<2)|(ny_<<1)|nx_], m[i]);
    }
    enc_flush(&e); return (uint32_t)e.len;
}
void v2_dec_chunkmask(const v2_u8 *buf, size_t len, v2_u8 *m){
    rc_dec d; dec_init(&d,buf,len);
    ctx_t ctx[8]; for(int i=0;i<8;++i) ctx_init(&ctx[i]);
    int SZ=V2_CHUNK;
    for(int z=0;z<SZ;++z)for(int y=0;y<SZ;++y)for(int x=0;x<SZ;++x){
        size_t i=((size_t)z*SZ+y)*SZ+x;
        int nz_= z?m[i-(size_t)SZ*SZ]:0, ny_= y?m[i-SZ]:0, nx_= x?m[i-1]:0;
        m[i]=(v2_u8)dec_bit(&d,&ctx[(nz_<<2)|(ny_<<1)|nx_]);
    }
}

// ---- block encode ----
// Payload layout (FROZEN, byte-identical to the original chunk-mask format):
//   [u8 dc][u8 flags=0][u16 mlen=0][u16 clen][clen coef bytes][u16 ncorr=0]
// flags/mlen/ncorr are always 0 in the frozen format (chunk-mask carries the air
// surface; per-block mask + outlier-correction were experimental and are removed).
int v2_enc_block(const v2_u8 *vox, const v2_u8 *rair, v2_buf *out, uint32_t *len_out){
    int n=N3, any=0; for(int i=0;i<n;++i) any|=vox[i];
    if(!any){ *len_out=0; return 0; }                 // all-zero block: no payload

    static _Thread_local float blk[N3], coef[N3];
    static _Thread_local v2_i32 lv[N3];
    // DC over MATERIAL only; air-fill air voxels to DC so the 0<->material cliff flattens.
    long sum=0,cnt=0; for(int i=0;i<n;++i){ if(vox[i]){ sum+=vox[i]; cnt++; } }
    int dc = cnt ? (int)((sum+cnt/2)/cnt) : 0;
    int nair=0; for(int i=0;i<n;++i) nair+=rair[i];
    for(int i=0;i<n;++i){ int v = rair[i] ? dc : (vox[i]?vox[i]:dc); blk[i]=(float)(v-dc); }
    // harmonic (Jacobi) air-fill: relax air voxels to the 6-neighbor mean (material fixed).
    if(nair>0 && nair<n){
        static _Thread_local float tmp[N3]; int S=V2_BLK;
        for(int it=0; it<V2_FILL_SWEEPS; ++it){
            memcpy(tmp,blk,sizeof(float)*n);
            for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
                int i=(z*S+y)*S+x; if(!rair[i]) continue;
                float a=0; int c=0;
                if(z){a+=tmp[i-S*S];c++;} if(z<S-1){a+=tmp[i+S*S];c++;}
                if(y){a+=tmp[i-S];c++;}   if(y<S-1){a+=tmp[i+S];c++;}
                if(x){a+=tmp[i-1];c++;}   if(x<S-1){a+=tmp[i+1];c++;}
                if(c) blk[i]=a/c;
            }
        }
    }
    dct3_fwd(blk,coef,V2_BLK);
    for(int cz=0;cz<V2_BLK;++cz)for(int cy=0;cy<V2_BLK;++cy)for(int cx=0;cx<V2_BLK;++cx){
        int idx=(cz*V2_BLK+cy)*V2_BLK+cx; float step=g_base_q*hf_weight(cz,cy,cx);
        lv[idx]=quant_one(coef[idx],step);
    }
    static _Thread_local rc_i16 ql[N3];
    static _Thread_local rc_u8 scratch[N3*2+256];
    for(int i=0;i<n;++i){ v2_i32 v=lv[i]; ql[i]=(rc_i16)(v>32767?32767:v<-32768?-32768:v); }
    rc_enc e; enc_init(&e,scratch,sizeof scratch);
    enc_block_coefs(&e,ql,V2_BLK); enc_flush(&e);
    uint32_t clen=(uint32_t)e.len;

    v2_u8 dcb=(v2_u8)dc, flags=0; v2_buf_put(out,&dcb,1); v2_buf_put(out,&flags,1);
    uint16_t z16=0; v2_buf_put(out,&z16,2);            // mlen=0 (chunk-mask carries the mask)
    uint16_t cl16=(uint16_t)clen; v2_buf_put(out,&cl16,2);
    v2_buf_put(out,scratch,clen);
    v2_buf_put(out,&z16,2);                            // ncorr=0
    *len_out = 1+1+2+2+clen+2;
    return 1;
}

// ---- block decode ----
void v2_dec_block(const v2_u8 *p, const v2_u8 *rair, v2_u8 *dst){
    int n=N3, dc=p[0];
    uint16_t mlen; memcpy(&mlen,p+2,2);                // =0 in frozen format
    const v2_u8 *cp_coef=p+4+mlen;
    uint16_t clen; memcpy(&clen,cp_coef,2);
    const v2_u8 *coded=cp_coef+2;
    static _Thread_local rc_i16 ql[N3];
    rc_dec d; dec_init(&d,coded,clen); dec_block_coefs(&d,ql,V2_BLK);
    static _Thread_local float coef[N3], blk[N3];
    for(int cz=0;cz<V2_BLK;++cz)for(int cy=0;cy<V2_BLK;++cy)for(int cx=0;cx<V2_BLK;++cx){
        int idx=(cz*V2_BLK+cy)*V2_BLK+cx; float step=g_base_q*hf_weight(cz,cy,cx);
        coef[idx]=deq_one(ql[idx],step);
    }
    dct3_inv(coef,blk,V2_BLK);
    for(int i=0;i<n;++i){
        int v = rair[i] ? 0 : (int)lrintf(blk[i])+dc;  // mask-restore: air -> exactly 0
        if(v<0)v=0; if(v>255)v=255; dst[i]=(v2_u8)v;
    }
}
