// FAST integer DCT — v1's even/odd partial-butterfly (vc.c), generalized to
// variable size S in {4,8,16,32}, ALL sizes equally optimized (fair speed compare).
// Q14 fixed-point. Partial butterfly = ~2x fewer MACs than naive matmul. The 3D
// transform is 3 separable 1D passes + a streaming rotate (as in v1).
// Range-safe in i32 per v1's analysis for S<=32. Use -ffast-math irrelevant (integer).
#ifndef V2_DCT_FAST_H
#define V2_DCT_FAST_H
#include <stdint.h>
#include <math.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
typedef int32_t fi32;
#define FQ14 14
#define FMAXN 32
#define FALIGN 64
#define FHOT __attribute__((hot))
#define FASSUME_ALIGNED(p) __builtin_assume_aligned((p),FALIGN)
// per-size Q14 cosine matrix as i16: max |coef|=5765 < 32767 (verified S<=32) → fits
// i16, halves the matrix footprint AND lets i16×i16→i32 widening MAC (vpmaddwd) pack
// 2× coefficients per SIMD reg. Accumulator STAYS i32 (the product needs >16 bits).

static fi32 g_cm[6][FMAXN][FMAXN] __attribute__((aligned(FALIGN)));
static int g_cm_ready=0;
static void fdct_build(void){
    if(g_cm_ready) return;
    for(int S=4;S<=FMAXN;S<<=1){ int l=0,t=S; while(t>1){t>>=1;l++;}
        double scale=(double)((int64_t)1<<FQ14);
        for(int k=0;k<S;++k){ double ck=(k==0)?sqrt(1.0/S):sqrt(2.0/S);
            for(int n=0;n<S;++n){ double v=ck*cos(M_PI*(2.0*n+1.0)*k/(2.0*S)); g_cm[l][k][n]=(fi32)llround(v*scale);} } }
    g_cm_ready=1;
}
static inline int fdct_log2(int S){ int l=0; while(S>1){S>>=1;l++;} return l; }

// 1D forward DCT-II, even/odd partial butterfly, K-PARALLEL form: outputs (k) are
// the vectorized axis, s[n]/d[n] broadcast — NO per-output reduction (the reduction
// form `for k: for n: acc+=...` only vectorizes width-8 per the LLVM report; this
// k-parallel form hits width-16 + avoids the horizontal reduction). v1 uses this on
// clang/aarch64 for the same reason (4.3× there). i32 accumulators (range-safe S<=32).
// FORM CHOICE is platform-dependent (v1's exact finding): gcc/x86 is FASTER with the
// REDUCTION form (for k: for n: acc+=...); clang & aarch64 are faster with the
// K-PARALLEL form. We measured gcc/x86: reduction 362 vs k-parallel 291 Mvox/s.
// i32 accumulators throughout (range-safe S<=32, v1 +61% vs i64).
#if defined(__clang__) || defined(__aarch64__)
#define FDCT_KPARALLEL 1
#else
#define FDCT_KPARALLEL 0
#endif
FHOT static inline void fdct1d_fwd(const fi32(*restrict C)[FMAXN], const fi32 *restrict in, fi32 *restrict out, int S){
    const fi32 rnd=(fi32)1<<(FQ14-1); int H=S/2; fi32 s[FMAXN/2] __attribute__((aligned(FALIGN))), d[FMAXN/2] __attribute__((aligned(FALIGN)));
    for(int n=0;n<H;++n){ s[n]=in[n]+in[S-1-n]; d[n]=in[n]-in[S-1-n]; }
#if FDCT_KPARALLEL
    fi32 acc[FMAXN]; for(int k=0;k<S;++k) acc[k]=rnd;
    for(int n=0;n<H;++n){ fi32 sn=s[n], dn=d[n];
        for(int k=0;k<S;k+=2) acc[k]+=C[k][n]*sn;
        for(int k=1;k<S;k+=2) acc[k]+=C[k][n]*dn; }
    for(int k=0;k<S;++k) out[k]=acc[k]>>FQ14;
#else
    for(int k=0;k<S;k+=2){ fi32 a=rnd; for(int n=0;n<H;++n) a+=C[k][n]*s[n]; out[k]=a>>FQ14; }
    for(int k=1;k<S;k+=2){ fi32 a=rnd; for(int n=0;n<H;++n) a+=C[k][n]*d[n]; out[k]=a>>FQ14; }
#endif
}
FHOT static inline void fdct1d_inv(const fi32(*restrict C)[FMAXN], const fi32 *restrict in, fi32 *restrict out, int S){
    const fi32 rnd=(fi32)1<<(FQ14-1); int H=S/2;
    for(int n=0;n<H;++n){ fi32 evn=rnd,odd=0;
        for(int k=0;k<S;k+=2) evn+=C[k][n]*in[k];
        for(int k=1;k<S;k+=2) odd+=C[k][n]*in[k];
        out[n]=(evn+odd)>>FQ14; out[S-1-n]=(evn-odd)>>FQ14; }
}
// rotate (z,y,x)->(x,z,y): dst[(x*S+z)*S+y] = src[(z*S+y)*S+x]. Contiguous-read
// streaming pass (vectorizes the write side). 3 rotations return to start.
static inline void frot(const fi32 *restrict src, fi32 *restrict dst, int S){
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x)
        dst[((size_t)x*S+z)*S+y]=src[((size_t)z*S+y)*S+x];
}
// Cache-blocked rotate: tile the (z,x) plane in TILE×TILE blocks so reads & writes
// stay within a few cache lines (fixes the strided-transpose L1 thrash). Same map
// dst[(x*S+z)*S+y]=src[(z*S+y)*S+x], just tiled iteration order.
#define FROT_TILE 4
static inline void frot_blocked(const fi32 *restrict src, fi32 *restrict dst, int S){
    for(int zt=0; zt<S; zt+=FROT_TILE)
    for(int xt=0; xt<S; xt+=FROT_TILE)
        for(int z=zt; z<zt+FROT_TILE; ++z)
        for(int x=xt; x<xt+FROT_TILE; ++x){
            const fi32 *sp = src + ((size_t)z*S)*S + x;       // src[(z,*,x)] stride S in y
            fi32 *dp = dst + ((size_t)x*S+z)*S;               // dst[(x,z,*)] contiguous in y
            for(int y=0;y<S;++y) dp[y]=sp[(size_t)y*S];
        }
}
// All-contiguous-lines forward pass with pruning, in place over blk (S^3).
static inline void flines_fwd(const fi32(*C)[FMAXN], fi32 *restrict blk, int S){
    fi32 ol[FMAXN];
    for(int off=0;off<S*S;++off){ fi32 *v=blk+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;} if(!nz) continue;
        fdct1d_fwd(C,v,ol,S); for(int i=0;i<S;++i) v[i]=ol[i]; }
}
static inline void flines_fwd_to(const fi32(*C)[FMAXN], const fi32 *restrict src, fi32 *restrict dst, int S){
    fi32 ol[FMAXN];
    for(int off=0;off<S*S;++off){ const fi32 *v=src+(size_t)off*S; fi32 *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0; continue; }
        fdct1d_fwd(C,v,ol,S); for(int i=0;i<S;++i) o[i]=ol[i]; }
}
static inline void flines_inv(const fi32(*C)[FMAXN], fi32 *restrict blk, int S){
    fi32 ol[FMAXN];
    for(int off=0;off<S*S;++off){ fi32 *v=blk+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;} if(!nz) continue;
        fdct1d_inv(C,v,ol,S); for(int i=0;i<S;++i) v[i]=ol[i]; }
}
// Out-of-place inverse lines: read src, transform, write dst. Fuses the copy-in into
// the first pass (saves one full memory sweep; the transform is memory-bound). v1 does this.
static inline void flines_inv_to(const fi32(*C)[FMAXN], const fi32 *restrict src, fi32 *restrict dst, int S){
    fi32 ol[FMAXN];
    for(int off=0;off<S*S;++off){ const fi32 *v=src+(size_t)off*S; fi32 *o=dst+(size_t)off*S;
        int nz=0; for(int i=0;i<S;++i) if(v[i]){nz=1;break;}
        if(!nz){ for(int i=0;i<S;++i) o[i]=0; continue; }
        fdct1d_inv(C,v,ol,S); for(int i=0;i<S;++i) o[i]=ol[i]; }
}
// 3D forward: 3× (transform-contiguous-lines + rotate). Every pass is unit-stride →
// all 3 vectorize (vs the strided y/z passes which only got 8-byte vectors).
FHOT static void fdct3_fwd(const fi32 *blk_in, fi32 *coef, int S){
    const fi32(*C)[FMAXN]=g_cm[fdct_log2(S)];
    static _Thread_local fi32 a[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN))), b[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN)));
    flines_fwd_to(C,blk_in,a,S); frot_blocked(a,b,S);   // fused copy-in
    flines_fwd(C,b,S); frot_blocked(b,a,S);
    flines_fwd(C,a,S); frot_blocked(a,coef,S);
}
FHOT static void fdct3_inv(const fi32 *coef, fi32 *blk_out, int S){
    const fi32(*C)[FMAXN]=g_cm[fdct_log2(S)];
    static _Thread_local fi32 a[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN))), b[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN)));
    flines_inv_to(C,coef,a,S); frot_blocked(a,b,S);  // fused copy-in (no pre-copy sweep)
    flines_inv(C,b,S); frot_blocked(b,a,S);
    flines_inv(C,a,S); frot_blocked(a,blk_out,S);
}
// Compile-time-S=16 specializations: passing the LITERAL 16 lets the compiler
// constant-propagate into the inlined flines/frot (trip-16 loops fully unroll +
// schedule). No hand-written products — the compiler does the unroll safely.
FHOT static void fdct3_inv16(const fi32 *coef, fi32 *blk_out){
    const fi32(*C)[FMAXN]=g_cm[fdct_log2(16)];
    static _Thread_local fi32 a[16*16*16] __attribute__((aligned(FALIGN))), b[16*16*16] __attribute__((aligned(FALIGN)));
    flines_inv_to(C,coef,a,16); frot_blocked(a,b,16);
    flines_inv(C,b,16); frot_blocked(b,a,16);
    flines_inv(C,a,16); frot_blocked(a,blk_out,16);
}
FHOT static void fdct3_fwd16(const fi32 *blk_in, fi32 *coef){
    const fi32(*C)[FMAXN]=g_cm[fdct_log2(16)];
    static _Thread_local fi32 a[16*16*16] __attribute__((aligned(FALIGN))), b[16*16*16] __attribute__((aligned(FALIGN)));
    flines_fwd_to(C,blk_in,a,16); frot_blocked(a,b,16);
    flines_fwd(C,b,16); frot_blocked(b,a,16);
    flines_fwd(C,a,16); frot_blocked(a,coef,16);
}
#endif
