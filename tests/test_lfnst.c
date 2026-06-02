// Round-trip + block-locality (touched=1) tests for the LFNST low-frequency
// secondary transform (harness/bench_lfnst experiment). Verifies:
//   1. ORTHONORMALITY: a random orthonormal DxD matrix's fwd then inv (= transpose)
//      recovers the input corner to float precision -> the secondary transform is
//      invertible and does not by itself add loss (loss is only at quant).
//   2. BLOCK-LOCALITY (touched=1): lfnst_fwd/lfnst_inv read & write ONLY the
//      low-freq N^3 corner of a single 16^3 atom; every coefficient outside the
//      corner is byte-identical before/after. Decoding one atom never reads or
//      writes any neighbour atom's data -> 16^3 random access preserved.
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define A 16u
#define AVOX 4096u

// ---- replicate the bench's LFNST corner ops (must match bench_lfnst.c) ----
static u32 g_N=4, g_DIM=64; static float *g_LF=NULL;
static inline void corner_gather(const i16 *coef,float *v){
    u32 N=g_N; for(u32 z=0;z<N;++z)for(u32 y=0;y<N;++y)for(u32 x=0;x<N;++x)
        v[(z*N+y)*N+x]=(float)coef[(size_t)z*256u+(size_t)y*16u+x];
}
static inline void corner_scatter(i16 *coef,const float *v){
    u32 N=g_N; for(u32 z=0;z<N;++z)for(u32 y=0;y<N;++y)for(u32 x=0;x<N;++x){
        i32 q=(i32)lrintf(v[(z*N+y)*N+x]);
        if(q>32767)q=32767; else if(q<-32768)q=-32768;
        coef[(size_t)z*256u+(size_t)y*16u+x]=(i16)q; }
}
static inline void lfnst_fwd(i16 *coef){ u32 D=g_DIM; float in[512],out[512];
    corner_gather(coef,in);
    for(u32 k=0;k<D;++k){float a=0;const float*r=g_LF+(size_t)k*D;for(u32 j=0;j<D;++j)a+=r[j]*in[j];out[k]=a;}
    corner_scatter(coef,out); }
static inline void lfnst_inv(i16 *coef){ u32 D=g_DIM; float in[512],out[512];
    corner_gather(coef,in);
    for(u32 j=0;j<D;++j){float a=0;for(u32 k=0;k<D;++k)a+=g_LF[(size_t)k*D+j]*in[k];out[j]=a;}
    corner_scatter(coef,out); }

static int fails=0;
#define CHECK(c,m) do{ if(!(c)){printf("FAIL: %s\n",m);fails++;} }while(0)

// build an orthonormal DxD matrix via Gram-Schmidt on a random matrix
static void rand_orthonormal(u32 D){
    g_LF=malloc((size_t)D*D*sizeof(float));
    double *v=malloc((size_t)D*D*sizeof(double));
    unsigned s=12345u;
    for(u32 i=0;i<D*D;++i){ s=s*1103515245u+12345u; v[i]=((double)(s>>9)/8388608.0)-1.0; }
    for(u32 i=0;i<D;++i){
        for(u32 k=0;k<i;++k){ double d=0; for(u32 j=0;j<D;++j)d+=v[i*D+j]*v[k*D+j];
            for(u32 j=0;j<D;++j)v[i*D+j]-=d*v[k*D+j]; }
        double n=0; for(u32 j=0;j<D;++j)n+=v[i*D+j]*v[i*D+j]; n=sqrt(n);
        for(u32 j=0;j<D;++j)v[i*D+j]/=n;
    }
    for(u32 i=0;i<D*D;++i) g_LF[i]=(float)v[i];
    free(v);
}

int main(void){
    for(u32 N=4;N<=8;N+=4){
        g_N=N; g_DIM=N*N*N; rand_orthonormal(g_DIM);

        // build a 16^3 coef block with a recognizable pattern
        i16 *coef=calloc(AVOX,sizeof(i16));
        i16 *orig=calloc(AVOX,sizeof(i16));
        for(u32 i=0;i<AVOX;++i){ coef[i]=(i16)(((i*37)%513)-256); }
        // keep corner magnitudes modest so int rounding round-trips cleanly
        for(u32 z=0;z<N;++z)for(u32 y=0;y<N;++y)for(u32 x=0;x<N;++x)
            coef[(size_t)z*256u+y*16u+x]=(i16)(((z*7+y*3+x)%40)-20);
        memcpy(orig,coef,AVOX*sizeof(i16));

        // (2) block-locality: snapshot everything OUTSIDE the corner
        i16 *snap=malloc(AVOX*sizeof(i16)); memcpy(snap,coef,AVOX*sizeof(i16));
        lfnst_fwd(coef);
        int outside_changed=0;
        for(u32 zz=0;zz<A;++zz)for(u32 yy=0;yy<A;++yy)for(u32 xx=0;xx<A;++xx){
            int in_corner=(zz<N&&yy<N&&xx<N);
            size_t idx=(size_t)zz*256u+yy*16u+xx;
            if(!in_corner && coef[idx]!=snap[idx]) outside_changed=1;
        }
        CHECK(!outside_changed,"LFNST fwd touched coefficients OUTSIDE the low-freq corner");

        // (1) orthonormal round-trip: inv(fwd(x)) == x for the corner (int recon
        // exact because magnitudes small and matrix orthonormal)
        lfnst_inv(coef);
        int rt_ok=1;
        for(u32 z=0;z<N;++z)for(u32 y=0;y<N;++y)for(u32 x=0;x<N;++x){
            size_t idx=(size_t)z*256u+y*16u+x;
            if(abs((int)coef[idx]-(int)orig[idx])>1) rt_ok=0;
        }
        CHECK(rt_ok,"LFNST fwd then inv (transpose) did not recover the corner");

        printf("N=%u: locality+round-trip checked (D=%u)\n",N,g_DIM);
        free(coef);free(orig);free(snap);free(g_LF);g_LF=NULL;
    }
    if(fails){ printf("LFNST TESTS FAILED: %d\n",fails); return 1; }
    printf("all LFNST tests passed\n"); return 0;
}
