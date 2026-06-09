// Fair decode-throughput bench: optimized partial-butterfly inverse DCT (v1-class),
// 16³ vs 32³, EQUAL optimization. Replaces the bogus naive-DCT numbers.
#include "dct_fast.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
int main(void){
    fdct_build();
    printf("# FAIR inverse-DCT throughput (v1 partial-butterfly + pruning, i32), per size\n");
    for(int S=16;S<=32;S<<=1){
        int n=S*S*S;
        fi32 *coef=malloc(n*sizeof(fi32)),*blk=malloc(n*sizeof(fi32));
        // realistic sparse coef block (high-compression): DC + few low-freq, rest 0
        for(int i=0;i<n;++i) coef[i]=0;
        coef[0]=8000; coef[1]=2000; coef[S]=1800; coef[S*S]=1500; coef[2]=900; coef[S+1]=700;
        long iters=(long)(4e8/n);
        // warm
        for(int w=0;w<3;++w) fdct3_inv(coef,blk,S);
        double t0=now_ms();
        for(long it=0;it<iters;++it){ fdct3_inv(coef,blk,S); coef[0]+=(blk[0]&0); }
        double t1=now_ms();
        double vox=(double)n*iters, mvps=vox/1e6/((t1-t0)/1e3);
        printf("  %2d^3: %ld blocks, %.0fms -> %.0f Mvox/s   (per-block %.2f us)\n",
               S, iters, t1-t0, mvps, (t1-t0)*1000.0/iters);
        // dense (worst case, no pruning benefit): fill all coefs
        for(int i=0;i<n;++i) coef[i]=(i*37%512)-256;
        for(int w=0;w<3;++w) fdct3_inv(coef,blk,S);
        t0=now_ms(); for(long it=0;it<iters;++it){ fdct3_inv(coef,blk,S); coef[0]+=(blk[0]&0); } t1=now_ms();
        mvps=vox/1e6/((t1-t0)/1e3);
        printf("        DENSE (no prune): %.0f Mvox/s   (per-block %.2f us)\n", mvps,(t1-t0)*1000.0/iters);
        free(coef); free(blk);
    }
    return 0;
}
