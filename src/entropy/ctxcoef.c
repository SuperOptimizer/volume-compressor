// Coefficient CONTEXT modeling for the 16^3 DCT atom (PLAN §2 entropy / EXP #13c).
// JPEG-XL / HEVC-style: code each quantized coefficient's SIGNIFICANCE and
// MAGNITUDE conditioned on ALREADY-DECODED neighbor coefficients WITHIN THE SAME
// 16^3 atom (never crossing the atom boundary, so 16^3 random access is intact).
//
// Engine: a small, table-free binary RANGE CODER (carryless, 32-bit, byte-wise
// renorm) driving adaptive bit models (each a 12-bit probability updated with a
// shift). This replaces RLGR for the per-atom coefficient stream so we can
// measure whether context modeling buys ratio over RLGR's backward-adaptive RLG.
//
// Coding order is the SAME increasing-frequency scan the quantizer emits (so the
// causal neighbors used as context are the lower-frequency coefficients already
// decoded). Contract MIRRORS the RLGR per-atom call but is atom-aware: it needs
// the atom side A so it can derive the 3D position of each scanned coefficient.
//
// Per coefficient i at 3D pos (z,y,x) in the atom we code, in this order:
//   1. significance bit  sig = (level != 0), context =
//        band(s=z+y+x) x  saturated count of significant CAUSAL neighbors
//        (the three already-decoded neighbors at x-1,y-1,z-1 within the atom).
//      This is the key dependency: a coefficient near already-significant
//      low-frequency energy is itself more likely significant.
//   2. if significant: magnitude m=|level|-1 via a context-coded EXP-GOLOMB-ish
//        unary prefix (each "continue" bit has its own band-conditioned model,
//        first few bins separated) + bypass mantissa, then a bypass SIGN bit.
//
// Pure C23, libc only. The range coder is inherently serial (like RLGR); the
// vectorizable stages remain transform/quant. Round-trip exact (see test).
#include "../../include/vc/types.h"
#include <string.h>

#ifndef VC_CTXCOEF_INLINE
#define VC_CTXCOEF_INLINE

// ---- binary range coder (64-bit low with explicit carry propagation) ------
// Encoder keeps a 64-bit `low` so the carry from `low += r0` is visible in the
// high bits; bytes are shifted out from bit 32..39 when range < 2^24. The cache/
// carry-count scheme defers a pending byte until we know whether a carry hits it.
typedef struct {
    u8 *buf; size_t cap, pos; int overflow;
    u64 low; u32 range; u8 cache; u64 carry_count; int started;
} rc_enc;
typedef struct { const u8 *buf; size_t len, pos; u32 range, code; } rc_dec;

static inline void rc_emit(rc_enc *e, u8 b){ if(e->pos>=e->cap){e->overflow=1;return;} e->buf[e->pos++]=b; }
static inline void rc_enc_init(rc_enc *e, u8 *buf, size_t cap){
    e->buf=buf; e->cap=cap; e->pos=0; e->overflow=0;
    e->low=0; e->range=0xFFFFFFFFu; e->cache=0; e->carry_count=0; e->started=0;
}
// LZMA-style ShiftLow: `low` is 33-bit (carry in bit 32). Defer the cache byte
// and a run of 0xFF bytes until we know whether a carry propagates into them.
static inline void rc_shift_low(rc_enc *e){
    if((u32)(e->low >> 32) != 0 || e->low < 0xFF000000ull){
        u8 carry = (u8)(e->low >> 32);
        if(e->started) rc_emit(e, (u8)(e->cache + carry));
        while(e->carry_count){ rc_emit(e,(u8)(0xFF + carry)); e->carry_count--; }
        e->cache = (u8)((e->low >> 24) & 0xFF);
        e->started = 1;
    } else {
        e->carry_count++;
    }
    e->low = (e->low << 8) & 0xFFFFFFFFull;
}
static inline void rc_enc_renorm(rc_enc *e){
    while(e->range < (1u<<24)){ rc_shift_low(e); e->range <<= 8; }
}
static inline void rc_enc_bit(rc_enc *e, u32 *p0, u32 bit){
    u32 r0 = (u32)(((u64)e->range * (*p0)) >> 12);
    if(!bit){ e->range = r0; *p0 += (4096u - *p0) >> 5; }
    else    { e->low += r0; e->range -= r0; *p0 -= (*p0) >> 5; }
    if(*p0 < 32) *p0 = 32; else if(*p0 > 4064) *p0 = 4064;
    rc_enc_renorm(e);
}
static inline void rc_enc_bypass(rc_enc *e, u32 bit){
    u32 r0 = e->range >> 1;
    if(!bit){ e->range = r0; } else { e->low += r0; e->range -= r0; }
    rc_enc_renorm(e);
}
static inline size_t rc_enc_finish(rc_enc *e){
    for(int i=0;i<5;++i) rc_shift_low(e);
    return e->pos;
}

static inline void rc_dec_init(rc_dec *d, const u8 *buf, size_t len){
    d->buf=buf; d->len=len; d->pos=0; d->range=0xFFFFFFFFu; d->code=0;
    for(int i=0;i<4;++i) d->code=(d->code<<8) | (d->pos<d->len?d->buf[d->pos++]:0);
}
static inline void rc_dec_renorm(rc_dec *d){
    while(d->range < (1u<<24)){
        d->code = (d->code<<8) | (d->pos<d->len?d->buf[d->pos++]:0);
        d->range <<= 8;
    }
}
static inline u32 rc_dec_bit(rc_dec *d, u32 *p0){
    u32 r0 = (u32)(((u64)d->range * (*p0)) >> 12);
    u32 bit;
    if(d->code < r0){ bit=0; d->range=r0; *p0 += (4096u - *p0) >> 5; }
    else { bit=1; d->code-=r0; d->range-=r0; *p0 -= (*p0) >> 5; }
    if(*p0 < 32) *p0 = 32; else if(*p0 > 4064) *p0 = 4064;
    rc_dec_renorm(d);
    return bit;
}
static inline u32 rc_dec_bypass(rc_dec *d){
    u32 r0 = d->range >> 1; u32 bit;
    if(d->code < r0){ bit=0; d->range=r0; }
    else { bit=1; d->code-=r0; d->range-=r0; }
    rc_dec_renorm(d);
    return bit;
}

// ---- context model layout -------------------------------------------------
// significance contexts: NBAND frequency bands x NNB (0..3 sig causal neighbors)
#define CC_A      16u
#define CC_AVOX   4096u
#define CC_NBAND  6u           // band from s=z+y+x (0..45) bucketed
#define CC_NNB    4u           // 0,1,2,3 significant causal neighbors
#define CC_NSIG   (CC_NBAND*CC_NNB)
#define CC_NMAGC  (CC_NBAND*3u) // magnitude continue-bit contexts: band x position(0,1,2+)

static inline u32 cc_band(u32 s){
    // s in 0..45 -> 0..5
    if(s==0) return 0;
    if(s<=4) return 1;
    if(s<=9) return 2;
    if(s<=16) return 3;
    if(s<=27) return 4;
    return 5;
}

typedef struct {
    u32 sig[CC_NSIG];
    u32 magc[CC_NMAGC];   // continue bits of the unary magnitude prefix
    u32 mag1;             // is-magnitude-greater-than-1 (single ctx, refined below)
} cc_model;

static inline void cc_model_init(cc_model *m){
    for(u32 i=0;i<CC_NSIG;++i) m->sig[i]=2048;
    for(u32 i=0;i<CC_NMAGC;++i) m->magc[i]=2048;
    m->mag1=2048;
}

// number of significant CAUSAL neighbors of (z,y,x) using a `sigmap` of 0/1 in
// atom layout z*256+y*16+x. Causal = already-scanned (x-1,y-1,z-1).
static inline u32 cc_nb(const u8 *sigmap, u32 z, u32 y, u32 x){
    u32 c=0;
    if(x>0) c += sigmap[(size_t)z*256u + y*16u + (x-1)];
    if(y>0) c += sigmap[(size_t)z*256u + (y-1)*16u + x];
    if(z>0) c += sigmap[(size_t)(z-1)*256u + y*16u + x];
    return c>3?3:c;
}

// Encode one 16^3 atom's quantized levels `q` (atom layout) in scan order `scan`
// (the increasing-frequency order; q is in atom layout, scan[k] gives the layout
// index of the k-th coefficient). Returns bytes written.
size_t vc_ctxcoef_encode_atom(u8 *restrict out, size_t cap,
                              const i16 *restrict q,
                              const u16 *restrict scan, u8 *restrict sigmap){
    rc_enc e; rc_enc_init(&e, out, cap);
    cc_model m; cc_model_init(&m);
    memset(sigmap,0,CC_AVOX);
    for(u32 k=0;k<CC_AVOX;++k){
        u32 idx = scan[k];
        u32 x = idx & 15u, y=(idx>>4)&15u, z=(idx>>8)&15u;
        u32 s = z+y+x;
        u32 b = cc_band(s);
        u32 nb = cc_nb(sigmap,z,y,x);
        u32 sc = b*CC_NNB + nb;
        i32 v = q[idx];
        u32 sig = v!=0;
        rc_enc_bit(&e,&m.sig[sc],sig);
        if(sig){
            sigmap[idx]=1;
            u32 mag = (u32)(v<0?-v:v);          // >=1
            u32 t = mag-1u;                      // 0,1,2,...
            // unary prefix with band-conditioned continue contexts (first 3 split)
            // unary prefix: emit min(t,11) continue bits; if t>=12, the 12th
            // continue bit signals ESCAPE and the remainder (t-12) follows as a
            // 16-bit bypass field (no stop bit in that case). Decoder mirrors.
            u32 j=0; int escaped=0;
            while(t>0 && j<12){
                u32 cpos = j<2?j:2;
                rc_enc_bit(&e,&m.magc[b*3u+cpos],1);  // continue
                t--; j++;
            }
            if(j==12){
                escaped=1;
                for(int bi=15;bi>=0;--bi) rc_enc_bypass(&e,(t>>bi)&1u);  // remaining t
            }
            if(!escaped){
                u32 cpos = j<2?j:2;
                rc_enc_bit(&e,&m.magc[b*3u+cpos],0);  // stop
            }
            rc_enc_bypass(&e, v<0?1u:0u);             // sign
        }
        if(e.overflow) return cap+1;
    }
    return rc_enc_finish(&e);
}

void vc_ctxcoef_decode_atom(i16 *restrict q, const u8 *restrict in, size_t len,
                            const u16 *restrict scan, u8 *restrict sigmap){
    rc_dec d; rc_dec_init(&d,in,len);
    cc_model m; cc_model_init(&m);
    memset(sigmap,0,CC_AVOX);
    memset(q,0,CC_AVOX*sizeof(i16));
    for(u32 k=0;k<CC_AVOX;++k){
        u32 idx = scan[k];
        u32 x = idx & 15u, y=(idx>>4)&15u, z=(idx>>8)&15u;
        u32 s = z+y+x;
        u32 b = cc_band(s);
        u32 nb = cc_nb(sigmap,z,y,x);
        u32 sc = b*CC_NNB + nb;
        u32 sig = rc_dec_bit(&d,&m.sig[sc]);
        if(sig){
            sigmap[idx]=1;
            u32 t=0; u32 j=0;
            while(j<12){
                u32 cpos = j<2?j:2;
                u32 cont = rc_dec_bit(&d,&m.magc[b*3u+cpos]);
                if(!cont) break;
                t++; j++;
            }
            if(j==12){
                u32 extra=0; for(int bi=0;bi<16;++bi) extra=(extra<<1)|rc_dec_bypass(&d);
                t+=extra;   // escaped: no stop bit
            }
            u32 mag = t+1u;
            u32 sgn = rc_dec_bypass(&d);
            q[idx] = (i16)(sgn? -(i32)mag : (i32)mag);
        }
    }
}

#endif // VC_CTXCOEF_INLINE
