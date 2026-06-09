// BLOCK-LEVEL SIMD DCT-16: transform W=16 blocks at once. Layout is block-interleaved
// — bt[i*W + w] = value at position i of block w. For a fixed position i, the W blocks
// occupy one contiguous W-vector → the 1D transform `out[k][:] = sum_n C[k][n]*in[n][:]`
// is a scalar(coef) × W-vector MAC, accumulated into a W-vector. NO lane reduction, NO
// shuffles → the small-N (16) problem vanishes; every op is naturally W-wide SIMD.
// Portable C (compiler autovectorizes the inner W loop); W=16 matches AVX-512 i32 lanes.
#ifndef V2_DCT_BATCH_H
#define V2_DCT_BATCH_H
#include "dct_fast.h"   // reuse g_cm, FQ14, fi32, FALIGN
#define BW 16           // blocks per batch (AVX-512 i32 lane count)

// One batched 1D inverse over the contiguous axis: S lines of W lanes each.
// in/out: [S][W] for ONE (z,y) line position, but we process all S^2 lines.
// Here we do the whole 3D inverse batched. Buffers are [S*S*S][W].
FHOT static void fdct3_inv_batch(const fi32 *restrict coef /*[S*S*S][BW]*/,
                                 fi32 *restrict out /*[S*S*S][BW]*/, int S){
    const fi32(*restrict C)[FMAXN]=g_cm[fdct_log2(S)];
    int n3=S*S*S; const fi32 rnd=(fi32)1<<(FQ14-1);
    static _Thread_local fi32 a[16*16*16*BW] __attribute__((aligned(FALIGN)));
    static _Thread_local fi32 b[16*16*16*BW] __attribute__((aligned(FALIGN)));
    // helper: batched 1D inverse along contiguous lines of src->dst ([S][W] per line)
    #define BATCH_LINES_INV(SRC,DST) do{ \
        for(int off=0; off<S*S; ++off){ const fi32 *restrict sv=(SRC)+(size_t)off*S*BW; \
            fi32 *restrict dv=(DST)+(size_t)off*S*BW; int H=S/2; \
            for(int n=0;n<H;++n){ fi32 evn[BW],odd[BW]; \
                for(int w=0;w<BW;++w){evn[w]=rnd;odd[w]=0;} \
                for(int k=0;k<S;k+=2){ fi32 c=C[k][n]; const fi32*ik=sv+(size_t)k*BW; for(int w=0;w<BW;++w) evn[w]+=c*ik[w]; } \
                for(int k=1;k<S;k+=2){ fi32 c=C[k][n]; const fi32*ik=sv+(size_t)k*BW; for(int w=0;w<BW;++w) odd[w]+=c*ik[w]; } \
                fi32 *o0=dv+(size_t)n*BW, *o1=dv+(size_t)(S-1-n)*BW; \
                for(int w=0;w<BW;++w){ o0[w]=(evn[w]+odd[w])>>FQ14; o1[w]=(evn[w]-odd[w])>>FQ14; } } } }while(0)
    // batched rotate (z,y,x)->(x,z,y): each "element" is a W-vector
    #define BATCH_ROT(SRC,DST) do{ \
        for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){ \
            const fi32*sp=(SRC)+(((size_t)z*S+y)*S+x)*BW; fi32*dp=(DST)+(((size_t)x*S+z)*S+y)*BW; \
            for(int w=0;w<BW;++w) dp[w]=sp[w]; } }while(0)
    // copy coef->a then 3×(inv lines + rotate)
    for(int i=0;i<n3*BW;++i) a[i]=coef[i];
    BATCH_LINES_INV(a,b); BATCH_ROT(b,a);
    BATCH_LINES_INV(a,b); BATCH_ROT(b,a);
    BATCH_LINES_INV(a,b); BATCH_ROT(b,out);
    #undef BATCH_LINES_INV
    #undef BATCH_ROT
}
#endif
