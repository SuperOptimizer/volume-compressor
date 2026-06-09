// Float-signature drop-in wrappers over the fast i32 DCT (dct_fast.h). The codec's
// leaf_dct works in float; these convert float->i32, run the optimized integer core,
// convert back. Same function names as the old naive dct.h, so the codec swaps its
// include from "dct.h" to "dct_fastf.h" and nothing else changes. DCT is integer/lossy.
#ifndef V2_DCT_FASTF_H
#define V2_DCT_FASTF_H
#include <math.h>
#include "dct_fast.h"
// compat aliases for codec code written against the old dct.h
#define DCT_MAXN FMAXN
#define dct_log2 fdct_log2
static void dct3_fwd(const float *restrict blk, float *restrict coef, int S){
    static _Thread_local fi32 bi[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN)));
    static _Thread_local fi32 ci[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN)));
    int n=S*S*S; for(int i=0;i<n;++i) bi[i]=(fi32)lrintf(blk[i]);
    if(S==16) fdct3_fwd16(bi,ci); else fdct3_fwd(bi,ci,S);
    for(int i=0;i<n;++i) coef[i]=(float)ci[i];
}
static void dct3_inv(const float *restrict coef, float *restrict blk, int S){
    static _Thread_local fi32 ci[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN)));
    static _Thread_local fi32 bi[FMAXN*FMAXN*FMAXN] __attribute__((aligned(FALIGN)));
    int n=S*S*S; for(int i=0;i<n;++i) ci[i]=(fi32)lrintf(coef[i]);
    if(S==16) fdct3_inv16(ci,bi); else fdct3_inv(ci,bi,S);
    for(int i=0;i<n;++i) blk[i]=(float)bi[i];
}
// build the matrix (codec must call dct_build() — alias to fdct_build)
#define dct_build fdct_build
#endif
