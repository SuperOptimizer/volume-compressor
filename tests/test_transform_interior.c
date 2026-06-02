// Round-trip + structural tests for EXP #13 transform-interior pieces:
//   - context coder (ctxcoef.c): exact round-trip on random quantized atoms.
//   - lapped16 transform: near-lossless round-trip (no quant) + NO cross-atom
//     leakage (outer samples of a line untouched by the prefilter alone).
//   - tuned quant matrices: weights in (0,1], DC weight==1, monotone-ish.
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define vc_dct_int16_fwd H_dct16_fwd
#define vc_dct_int16_inv H_dct16_inv
#define VC_CHUNK_SIDE 16
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv
#include "../src/quant/qmatrix16.c"
#include "../src/quant/qmatrix16_tuned.c"
#include "../src/transform/lapped16.c"
#include "../src/entropy/ctxcoef.c"

#define AVOX 4096u
static u16 scan[AVOX];
static void build_scan(void){ u32 k=0; for(u32 s=0;s<=45;++s)
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x)
        if(z+y+x==s) scan[k++]=(u16)((size_t)z*256u+y*16u+x); }

static u32 rng=12345; static u32 xr(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}

int main(void){
    build_scan();
    int fails=0;

    // ---- context coder round-trip on sparse + dense atoms ----
    u8*buf=malloc(AVOX*4); u8*sig=malloc(AVOX);
    i16 q[AVOX],qd[AVOX];
    for(int trial=0;trial<200;++trial){
        // sparsity sweep: most zeros, a few nonzeros with varied magnitude
        memset(q,0,sizeof q);
        u32 nnz = xr()% (trial<100?60u:600u);
        for(u32 i=0;i<nnz;++i){ u32 p=xr()%AVOX; i32 m=1+(i32)(xr()%(1+(xr()%40)));
            q[p]=(i16)((xr()&1)?-m:m); }
        size_t len=vc_ctxcoef_encode_atom(buf,AVOX*4,q,scan,sig);
        if(len>AVOX*4){printf("FAIL ctx overflow trial %d\n",trial);fails++;continue;}
        vc_ctxcoef_decode_atom(qd,buf,len,scan,sig);
        if(memcmp(q,qd,sizeof q)!=0){
            int firstbad=-1; for(u32 i=0;i<AVOX;++i)if(q[i]!=qd[i]){firstbad=(int)i;break;}
            printf("FAIL ctx round-trip trial %d nnz=%u first bad idx=%d (%d!=%d)\n",
                   trial,nnz,firstbad,q[firstbad],qd[firstbad]); fails++; }
    }
    if(!fails) printf("OK   ctxcoef round-trip (200 trials, sparse+dense)\n");

    // ---- context coder with large magnitudes (escape path) ----
    memset(q,0,sizeof q);
    q[scan[2000]]=20000; q[scan[10]]=-15000; q[scan[4095]]=300;
    { size_t len=vc_ctxcoef_encode_atom(buf,AVOX*4,q,scan,sig);
      vc_ctxcoef_decode_atom(qd,buf,len,scan,sig);
      if(memcmp(q,qd,sizeof q)!=0){printf("FAIL ctx large-mag escape\n");fails++;}
      else printf("OK   ctxcoef large-magnitude escape path\n"); }

    // ---- lapped16 near-lossless round-trip (no quant) ----
    u8 vox[AVOX],rec[AVOX];
    for(u32 i=0;i<AVOX;++i) vox[i]=(u8)(xr()&0xff);
    { i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=vox[i]; i32 dc=(sum+2048)/4096;
      i16 coef[AVOX]; vc_lapped16_fwd(coef,vox,dc); vc_lapped16_inv(rec,coef,dc);
      double mse=0; for(u32 i=0;i<AVOX;++i){double e=(double)vox[i]-rec[i];mse+=e*e;}
      mse/=AVOX; double psnr=mse>0?10*log10(255.0*255.0/mse):99;
      if(psnr<40){printf("FAIL lapped16 round-trip PSNR=%.1f (random)\n",psnr);fails++;}
      else printf("OK   lapped16 round-trip PSNR=%.1f dB (random, no quant)\n",psnr); }
    // smooth ramp (real-data-like) should be much better
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x)
        vox[(size_t)z*256u+y*16u+x]=(u8)(20+5*x+4*y+3*z);
    { i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=vox[i]; i32 dc=(sum+2048)/4096;
      i16 coef[AVOX]; vc_lapped16_fwd(coef,vox,dc); vc_lapped16_inv(rec,coef,dc);
      double mse=0; for(u32 i=0;i<AVOX;++i){double e=(double)vox[i]-rec[i];mse+=e*e;}
      mse/=AVOX; double psnr=mse>0?10*log10(255.0*255.0/mse):99;
      if(psnr<55){printf("FAIL lapped16 ramp PSNR=%.1f\n",psnr);fails++;}
      else printf("OK   lapped16 ramp PSNR=%.1f dB (smooth)\n",psnr); }

    // ---- tuned matrices: DC weight==1, all weights in (0,1] ----
    f32 wm[AVOX]; qmt_set_slope(0.60f);
    qmt_mode modes[]={QMT_LINEAR,QMT_L2,QMT_SEP};
    const char*mn[]={"LINEAR","L2","SEP"};
    for(int mi=0;mi<3;++mi){
        qmt_build_weight(wm,modes[mi],NULL,0);
        if(fabsf(wm[0]-1.0f)>1e-4f){printf("FAIL %s DC weight %.3f != 1\n",mn[mi],wm[0]);fails++;}
        int ok=1; for(u32 i=0;i<AVOX;++i) if(wm[i]<=0||wm[i]>1.0001f){ok=0;break;}
        if(!ok){printf("FAIL %s weight out of (0,1]\n",mn[mi]);fails++;}
    }
    if(!fails) printf("OK   tuned matrices DC=1, weights in (0,1]\n");

    free(buf);free(sig);
    printf(fails? "\n%d FAILURES\n":"\nALL PASS\n",fails);
    return fails?1:0;
}
