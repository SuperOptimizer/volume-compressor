// v1's real CABAC-style binary range coder + coefficient context coder,
// generalized to variable block size S (4/8/16/32). Lifted from src/vc/vc.c.
// Replaces v2lab's estimate_bits with actual coded bytes so ratios are REAL.
#ifndef V2_RANGECODER_H
#define V2_RANGECODER_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef uint8_t  rc_u8;  typedef int16_t rc_i16; typedef int32_t rc_i32;
typedef uint32_t rc_u32; typedef uint64_t rc_u64;

typedef struct { rc_u8 *buf; size_t cap, len; rc_u64 low; rc_u32 range; rc_u8 cache; rc_u64 cache_size; } rc_enc;
typedef struct { const rc_u8 *buf; size_t len, pos; rc_u32 code, range; } rc_dec;
typedef struct { uint16_t p0; } ctx_t;
static inline void ctx_init(ctx_t *c){ c->p0 = 1u<<11; }
#define RC_TOP (1u<<24)

static void enc_init(rc_enc *e, rc_u8 *buf, size_t cap){ e->buf=buf;e->cap=cap;e->len=0;e->low=0;e->range=0xFFFFFFFFu;e->cache=0;e->cache_size=1; }
static void enc_putbyte(rc_enc *e, rc_u8 b){ if(e->len<e->cap) e->buf[e->len++]=b; else e->len++; /*count overflow*/ }
static void enc_shift_low(rc_enc *e){
    if((rc_u32)(e->low>>32)!=0 || e->low<0xFF000000ull){
        rc_u8 carry=(rc_u8)(e->low>>32);
        do{ enc_putbyte(e,(rc_u8)(e->cache+carry)); e->cache=0xFF; }while(--e->cache_size);
        e->cache=(rc_u8)(e->low>>24);
    }
    e->cache_size++; e->low=(e->low<<8)&0xFFFFFFFFull;
}
static void enc_bit(rc_enc *e, ctx_t *c, int bit){
    rc_u32 r0=(rc_u32)(((rc_u64)e->range*c->p0)>>12);
    if(bit==0){ e->range=r0; c->p0=(uint16_t)(c->p0+((4096-c->p0)>>5)); }
    else { e->low+=r0; e->range-=r0; c->p0=(uint16_t)(c->p0-(c->p0>>5)); }
    while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; }
}
static void enc_bypass(rc_enc *e, int bit){ e->range>>=1; if(bit) e->low+=e->range; while(e->range<RC_TOP){ enc_shift_low(e); e->range<<=8; } }
static void enc_flush(rc_enc *e){ for(int i=0;i<5;++i) enc_shift_low(e); }

static void dec_init(rc_dec *d,const rc_u8*buf,size_t len){ d->buf=buf;d->len=len;d->pos=0;d->code=0;d->range=0xFFFFFFFFu; for(int i=0;i<5;++i){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; } }
static int dec_bit(rc_dec *d, ctx_t *c){
    rc_u32 r0=(rc_u32)(((rc_u64)d->range*c->p0)>>12); int bit;
    if(d->code<r0){ d->range=r0;bit=0; c->p0=(uint16_t)(c->p0+((4096-c->p0)>>5)); }
    else { d->code-=r0; d->range-=r0; bit=1; c->p0=(uint16_t)(c->p0-(c->p0>>5)); }
    while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; }
    return bit;
}
static int dec_bypass(rc_dec *d){ d->range>>=1; int bit=(d->code>=d->range); if(bit)d->code-=d->range; while(d->range<RC_TOP){ rc_u8 b=(d->pos<d->len)?d->buf[d->pos++]:0; d->code=(d->code<<8)|b; d->range<<=8; } return bit; }

// --- coefficient context coder, parameterized by block size S ---
#define NB_BANDS 8
#define MAGCTX   12
#define NMAGBAND 3      // per-frequency-band magnitude models (c3d lift): low/mid/high
typedef struct { ctx_t sig[NB_BANDS]; ctx_t mag[NMAGBAND][MAGCTX]; } atom_ctx;
static void atom_ctx_init(atom_ctx *a){ for(int i=0;i<NB_BANDS;++i)ctx_init(&a->sig[i]);
    for(int b=0;b<NMAGBAND;++b)for(int i=0;i<MAGCTX;++i)ctx_init(&a->mag[b][i]); }
static inline int magband_of(int sigband){ (void)sigband; return 0; } // per-band gave no gain (retested on v7 too: 82.54->83.27MB worse — adaptive learning cost > distribution gain); collapsed to 1 model

// per-size scan tables (ascending L1 freq) built lazily for S in {4,8,16,32}
static uint16_t *g_scanS[6]; static int g_scanS_ready[6];
static int scanS_cmp_S; // block size for the comparator
static int scanS_cmp(const void*pa,const void*pb){
    rc_u32 a=*(const rc_u32*)pa,b=*(const rc_u32*)pb; int S=scanS_cmp_S;
    rc_u32 fa=(a/(S*S))+((a/S)%S)+(a%S), fb=(b/(S*S))+((b/S)%S)+(b%S);
    if(fa!=fb) return (int)fa-(int)fb; return (int)a-(int)b;
}
static void scanS_build(int S){
    int l=0,t=S; while(t>1){t>>=1;l++;}
    if(g_scanS_ready[l]) return;
    int n=S*S*S; rc_u32 *ord=malloc(n*sizeof(rc_u32)); for(int i=0;i<n;++i)ord[i]=i;
    scanS_cmp_S=S; qsort(ord,n,sizeof(rc_u32),scanS_cmp);
    g_scanS[l]=malloc(n*sizeof(uint16_t)); for(int i=0;i<n;++i)g_scanS[l][i]=(uint16_t)ord[i];
    free(ord); g_scanS_ready[l]=1;
}
static inline int band_of_S(rc_u32 idx,int S){
    rc_u32 cz=idx/(S*S),cy=(idx/S)%S,cx=idx%S, freq=cz+cy+cx;
    int b=(int)(freq*NB_BANDS/(3u*S)); if(b>=NB_BANDS)b=NB_BANDS-1; return b;
}
static void enc_magnitude(rc_enc*e,atom_ctx*ac,rc_u32 m,int mb){
    ctx_t*mag=ac->mag[mb]; rc_u32 v=m-1,k=0;
    while(k<(rc_u32)(MAGCTX-1)&&v>0){ enc_bit(e,&mag[k],1); v-=1;k++; if(v==0){enc_bit(e,&mag[k],0);return;} }
    if(v==0){ enc_bit(e,&mag[k],0); return; }
    enc_bit(e,&mag[k],1); rc_u32 x=v,nbits=0,tt=x+1; while(tt>1){tt>>=1;nbits++;}
    for(rc_u32 i=0;i<nbits;++i)enc_bypass(e,1); enc_bypass(e,0);
    for(rc_i32 i=(rc_i32)nbits-1;i>=0;--i)enc_bypass(e,((x+1)>>i)&1);
}
static rc_u32 dec_magnitude(rc_dec*d,atom_ctx*ac,int mb){
    ctx_t*mag=ac->mag[mb]; rc_u32 v=0,k=0;
    while(k<(rc_u32)(MAGCTX-1)){ if(dec_bit(d,&mag[k])){v+=1;k++;} else return v+1; }
    if(!dec_bit(d,&mag[k])) return v+1;
    rc_u32 nbits=0; while(dec_bypass(d))nbits++; rc_u32 x=1; for(rc_u32 i=0;i<nbits;++i)x=(x<<1)|(rc_u32)dec_bypass(d); x-=1; return v+x+1;
}
static int eob_bits_S(int n){ int b=0; while((1<<b) <= n) b++; return b; }  // ceil(log2(n+1))
static void enc_eob(rc_enc*e,rc_u32 eob,int n){ int bits=eob_bits_S(n); for(int i=bits-1;i>=0;--i)enc_bypass(e,(eob>>i)&1); }
static rc_u32 dec_eob(rc_dec*d,int n){ int bits=eob_bits_S(n); rc_u32 v=0; for(int i=0;i<bits;++i)v=(v<<1)|(rc_u32)dec_bypass(d); return v; }

// Encode quantized levels q[S^3] (raster) -> writes into e. Returns nothing.
static void enc_block_coefs(rc_enc*e,const rc_i16*q,int S){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac);
    rc_u32 eob=0; for(rc_u32 p=n;p-->0;){ if(q[scan[p]]!=0){eob=p+1;break;} }
    enc_eob(e,eob,n);
    for(rc_u32 p=0;p<eob;++p){ rc_u32 idx=scan[p]; int b=band_of_S(idx,S); rc_i16 v=q[idx];
        if(v==0){ enc_bit(e,&ac.sig[b],0); continue; }
        enc_bit(e,&ac.sig[b],1); rc_u32 m=(rc_u32)(v<0?-v:v); enc_magnitude(e,&ac,m,magband_of(b)); enc_bypass(e,v<0?1:0);
    }
}
static void dec_block_coefs(rc_dec*d,rc_i16*q,int S){
    scanS_build(S); int l=0,t=S; while(t>1){t>>=1;l++;} const uint16_t*scan=g_scanS[l];
    int n=S*S*S; atom_ctx ac; atom_ctx_init(&ac); memset(q,0,n*sizeof(rc_i16));
    rc_u32 eob=dec_eob(d,n); if(eob>(rc_u32)n)eob=n;
    for(rc_u32 p=0;p<eob;++p){ rc_u32 idx=scan[p]; int b=band_of_S(idx,S);
        if(!dec_bit(d,&ac.sig[b]))continue; rc_u32 m=dec_magnitude(d,&ac,magband_of(b)); int neg=dec_bypass(d);
        q[idx]=(rc_i16)(neg?-(rc_i32)m:(rc_i32)m);
    }
}
#endif
