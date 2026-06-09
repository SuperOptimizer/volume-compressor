// i16-THROUGHOUT DCT-16 with staged scaling (HEVC-style). Both data AND coefs are i16
// → i16×i16→i32 widening MAC (vpmaddwd) packs 2× per SIMD lane (the win mixed-width
// i16×i32 couldn't get). Q8 coefs + shift-by-Q each pass keeps intermediates in i16
// (~10.5 bits, verified). Accumulator i32 within a pass. Lower coef precision (Q8 vs
// v1's Q14) = some transform noise — MEASURE quality. Portable C, autovectorizable.
#ifndef V2_DCT_I16_H
#define V2_DCT_I16_H
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
typedef int16_t s16; typedef int32_t s32;
#define IQ 8                      // coef fixed-point bits
#define I16ALIGN __attribute__((aligned(64)))
static s16 i16_cm[17][16] I16ALIGN;   // [k][n] Q8 DCT-16 matrix (only N=16 here)
static int i16_ready=0;
static void i16_build(void){ if(i16_ready)return; int S=16; double sc=(double)(1<<IQ);
    for(int k=0;k<S;++k){ double ck=(k==0)?sqrt(1.0/S):sqrt(2.0/S);
        for(int n=0;n<S;++n) i16_cm[k][n]=(s16)lround(ck*cos(M_PI*(2*n+1)*k/(2.0*S))*sc); }
    i16_ready=1; }

// 1D inverse over contiguous lines, i16 in -> i16 out, shift by IQ. (in place ok)
static inline void i16_lines_inv(s16 *restrict blk){
    const int S=16,H=8; const s32 rnd=1<<(IQ-1);
    for(int off=0;off<S*S;++off){ s16 *restrict v=blk+(size_t)off*S; s16 ol[16];
        for(int n=0;n<H;++n){ s32 evn=rnd,odd=0;
            for(int k=0;k<S;k+=2) evn+=(s32)i16_cm[k][n]*v[k];
            for(int k=1;k<S;k+=2) odd+=(s32)i16_cm[k][n]*v[k];
            ol[n]=(s16)((evn+odd)>>IQ); ol[S-1-n]=(s16)((evn-odd)>>IQ); }
        for(int i=0;i<S;++i) v[i]=ol[i]; }
}
static inline void i16_lines_fwd(s16 *restrict blk){
    const int S=16,H=8; const s32 rnd=1<<(IQ-1);
    for(int off=0;off<S*S;++off){ s16 *restrict v=blk+(size_t)off*S; s16 s[8],d[8],ol[16];
        for(int n=0;n<H;++n){ s[n]=v[n]+v[S-1-n]; d[n]=v[n]-v[S-1-n]; }
        for(int k=0;k<S;k+=2){ s32 a=rnd; for(int n=0;n<H;++n) a+=(s32)i16_cm[k][n]*s[n]; ol[k]=(s16)(a>>IQ); }
        for(int k=1;k<S;k+=2){ s32 a=rnd; for(int n=0;n<H;++n) a+=(s32)i16_cm[k][n]*d[n]; ol[k]=(s16)(a>>IQ); }
        for(int i=0;i<S;++i) v[i]=ol[i]; }
}
static inline void i16_rot(const s16 *restrict src, s16 *restrict dst){
    const int S=16;
    for(int zt=0;zt<S;zt+=4)for(int xt=0;xt<S;xt+=4)
      for(int z=zt;z<zt+4;++z)for(int x=xt;x<xt+4;++x){
        const s16 *sp=src+((size_t)z*S)*S+x; s16 *dp=dst+((size_t)x*S+z)*S;
        for(int y=0;y<S;++y) dp[y]=sp[(size_t)y*S]; }
}
static void i16_inv16(const s16 *restrict coef, s16 *restrict out){
    static _Thread_local s16 a[16*16*16] I16ALIGN, b[16*16*16] I16ALIGN;
    for(int i=0;i<16*16*16;++i) a[i]=coef[i];
    i16_lines_inv(a); i16_rot(a,b);
    i16_lines_inv(b); i16_rot(b,a);
    i16_lines_inv(a); i16_rot(a,out);
}
static void i16_fwd16(const s16 *restrict in, s16 *restrict coef){
    static _Thread_local s16 a[16*16*16] I16ALIGN, b[16*16*16] I16ALIGN;
    for(int i=0;i<16*16*16;++i) a[i]=in[i];
    i16_lines_fwd(a); i16_rot(a,b);
    i16_lines_fwd(b); i16_rot(b,a);
    i16_lines_fwd(a); i16_rot(a,coef);
}
#endif
