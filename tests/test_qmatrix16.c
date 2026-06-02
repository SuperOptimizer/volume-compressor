// Round-trip / exactness tests for the 16^3 per-coefficient quant MATRIX
// (src/quant/qmatrix16.c) and its three objectives (FLAT/HF/ADAPT). Verifies:
//   1. quant -> RLGR encode -> RLGR decode -> recovers the EXACT level array
//      (the entropy stage is lossless on the quantized levels), for all modes.
//   2. dequant(quant(coef)) is deterministic and idempotent on re-quant (the
//      reconstruction point is stable), so the dead-zone is reproducible.
//   3. at a tiny step the full DCT-16 -> quant -> dequant -> IDCT-16 pipeline is
//      near-lossless on a structured 16^3 block (only fixed-point + quant floor).
//   4. the content-adaptive scale index maps the same in encoder and decoder
//      (qm_adapt_scale_from_index is a pure function of the byte) -> exact.
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  H_dct16_fwd
#define vc_dct_int16_inv  H_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv
#include "../src/quant/qmatrix16.c"

size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

static int fails = 0;
#define CHECK(c,msg) do{ if(!(c)){ printf("FAIL: %s\n",msg); fails++; } }while(0)

int main(void){
    const u32 N = QM_BLKN;
    i16 *coef = malloc(N*sizeof(i16));
    i16 *qb   = malloc(N*sizeof(i16));
    i16 *qb2  = malloc(N*sizeof(i16));
    i16 *dec  = malloc(N*sizeof(i16));
    i16 *cdq  = malloc(N*sizeof(i16));
    u8  *tmp  = malloc(N*4);
    f32 *step = malloc(N*sizeof(f32));

    // a pseudo-random-ish coefficient block with realistic energy compaction
    for(u32 i=0;i<N;++i){
        // low-freq large, high-freq small + noise
        int z=i/256,y=(i/16)%16,x=i%16; int s=z+y+x;
        double v = 800.0*exp(-s*0.35)*cos(i*0.7) + ((int)(i*2654435761u>>20)%7 - 3);
        coef[i]=(i16)(v<-32000?-32000:(v>32000?32000:v));
    }

    vc_qm_mode modes[3]={QM_FLAT,QM_HF,QM_ADAPT};
    const char *mn[3]={"FLAT","HF","ADAPT"};
    for(int m=0;m<3;++m){
        f32 sc = (modes[m]==QM_ADAPT)? qm_adapt_scale_from_index(5) : 1.0f;
        vc_qm_mode bm = (modes[m]==QM_ADAPT)?QM_HF:modes[m];
        qm_build_step(step, bm, 8.0f, sc);
        qm_quant_block(qb, coef, step);
        // (1) RLGR exact round-trip of the levels
        size_t by = vc_rlgr_encode(tmp, N*4, qb, N);
        vc_rlgr_decode(dec, N, tmp, by);
        int ok=1; for(u32 i=0;i<N;++i) if(dec[i]!=qb[i]){ok=0;break;}
        char b[64]; snprintf(b,64,"RLGR exact round-trip (%s)",mn[m]); CHECK(ok,b);
        // (2) dequant deterministic + re-quant idempotent
        qm_dequant_block(cdq, qb, step);
        qm_quant_block(qb2, cdq, step);
        int idem=1; for(u32 i=0;i<N;++i) if(qb2[i]!=qb[i]){idem=0;break;}
        snprintf(b,64,"re-quant idempotent (%s)",mn[m]); CHECK(idem,b);
    }

    // (3) near-lossless at tiny step through the FULL pipeline on a real block.
    u8 src[QM_BLKN], rec[QM_BLKN];
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x){
        double f = 110 + 40*sin(x*0.6)*cos(y*0.5) + 25*((x+z)&1) + 15*sin(z*0.9);
        int iv=(int)(f+0.5); src[(z*16+y)*16+x]=(u8)(iv<0?0:(iv>255?255:iv));
    }
    int sum=0; for(u32 i=0;i<QM_BLKN;++i) sum+=src[i];
    i32 dc=(sum+ (i32)(QM_BLKN/2))/(i32)QM_BLKN;
    H_dct16_fwd(coef, src, dc);
    qm_build_step(step, QM_FLAT, 1.0f, 1.0f);   // step ~1: near-lossless
    qm_quant_block(qb, coef, step);
    qm_dequant_block(cdq, qb, step);
    H_dct16_inv(rec, cdq, dc);
    double se=0; int linf=0;
    for(u32 i=0;i<QM_BLKN;++i){ int e=(int)rec[i]-(int)src[i]; se+=e*e; int a=e<0?-e:e; if(a>linf)linf=a; }
    double rmse=sqrt(se/QM_BLKN);
    printf("near-lossless: rmse=%.3f linf=%d\n",rmse,linf);
    CHECK(rmse < 1.5, "near-lossless pipeline rmse < 1.5 at step=1");

    // (4) adaptive index/scale is a pure function (decoder reproduces it).
    for(u32 i=0;i<QM_ADAPT_NS;++i){
        f32 s1=qm_adapt_scale_from_index(i), s2=qm_adapt_scale_from_index(i);
        CHECK(s1==s2, "adapt scale deterministic");
        CHECK(s1>=QM_ADAPT_MIN-1e-3f && s1<=QM_ADAPT_MAX+1e-3f, "adapt scale in range");
    }

    free(coef);free(qb);free(qb2);free(dec);free(cdq);free(tmp);free(step);
    if(fails){ printf("%d CHECK(s) FAILED\n",fails); return 1; }
    printf("all qmatrix16 round-trip tests passed\n");
    return 0;
}
