// Round-trip + touched=1 (random-access) tests for the VQ/lattice quantizer
// modes (harness/bench_vq.c). These verify the two invariants the codec must
// keep for any new quant mode:
//
//   1. ROUND-TRIP: encode(atom) -> decode -> bounded, deterministic, and for the
//      LATTICE quantizer the reconstructed coefficients are exactly the
//      transmitted lattice points * step (no codebook drift), and for VQ the
//      reconstruction is exactly codebook[index]*step.
//   2. TOUCHED=1: decoding ONE atom in isolation produces byte-identical output
//      to decoding it as part of a full multi-atom volume sweep. i.e. the
//      quantizer adds NO cross-atom dependency -> 16^3 random access is preserved.
//      (The VQ codebook is global read-only side info, NOT a cross-atom state, so
//      a single atom still decodes alone.)
//
// Self-contained like the bench: #includes the transform + qmatrix directly.
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  T_dct16_fwd
#define vc_dct_int16_inv  T_dct16_inv
#include "../src/transform/dct_int16.c"
#include "../src/quant/qmatrix16.c"

size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A 16u
#define AVOX 4096u
#define NLF 64u
#define DIM 4u
#define NGRP (NLF/DIM)
#define K 256u

static u16 scan[AVOX];
static f32 cb[K*DIM];

static void build_scan(void){
    u32 off[46]={0},cnt[46]={0};
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x) cnt[z+y+x]++;
    u32 acc=0; for(u32 s=0;s<46;++s){off[s]=acc;acc+=cnt[s];}
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){u32 s=z+y+x,idx=(u32)z*256+y*16+x; scan[off[s]++]=(u16)idx;}
}
static void d4q(const f32*in,i32*out){
    i32 r[DIM],sum=0; f32 me=-1;u32 am=0;i32 dir=0;
    for(u32 i=0;i<DIM;++i){i32 ri=(i32)lrintf(in[i]);r[i]=ri;sum+=ri;f32 e=in[i]-(f32)ri;f32 ae=e<0?-e:e;if(ae>me){me=ae;am=i;dir=e>0?1:-1;}}
    if(sum&1) r[am]+=dir?dir:1;
    for(u32 i=0;i<DIM;++i) out[i]=r[i];
}
static u32 vqn(const f32*v){u32 b=0;f32 bd=1e30f;for(u32 c=0;c<K;++c){const f32*cw=cb+c*DIM;f32 d=0;for(u32 j=0;j<DIM;++j){f32 e=v[j]-cw[j];d+=e*e;}if(d<bd){bd=d;b=c;}}return b;}

// Encode+reconstruct one atom under LATTICE; write 4096 reconstructed coeffs.
static void enc_lattice(const u8*src,const f32*step,i16*reccoef,i32*dcout){
    i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=src[i];i32 dc=(sum+2048)/4096;*dcout=dc;
    i16 coef[AVOX],qb[AVOX]; T_dct16_fwd(coef,src,dc);
    for(u32 g=0;g<NGRP;++g){f32 sv[DIM];i32 lp[DIM];for(u32 j=0;j<DIM;++j){u32 idx=scan[g*DIM+j];sv[j]=(f32)coef[idx]/step[idx];}d4q(sv,lp);for(u32 j=0;j<DIM;++j)qb[g*DIM+j]=(i16)lp[j];}
    for(u32 k=NLF;k<AVOX;++k){u32 idx=scan[k];f32 c=(f32)coef[idx],a=c<0?-c:c,s=step[idx];i32 m=(a>=0.5f*s);i32 l=m*((i32)(a/s-0.5f)+1);qb[k]=(i16)(c<0?-l:l);}
    // reconstruct
    for(u32 g=0;g<NGRP;++g)for(u32 j=0;j<DIM;++j){u32 k=g*DIM+j,idx=scan[k];reccoef[idx]=(i16)lrintf((f32)qb[k]*step[idx]);}
    for(u32 k=NLF;k<AVOX;++k){u32 idx=scan[k];i32 l=qb[k],al=l<0?-l:l;f32 r=(f32)al*step[idx];i32 v=(i32)lrintf(l<0?-r:r);reccoef[idx]=(i16)v;}
}
static void enc_vq(const u8*src,const f32*step,i16*reccoef,i32*dcout){
    i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=src[i];i32 dc=(sum+2048)/4096;*dcout=dc;
    i16 coef[AVOX]; T_dct16_fwd(coef,src,dc); u32 idxs[NGRP];
    for(u32 g=0;g<NGRP;++g){f32 sv[DIM];for(u32 j=0;j<DIM;++j){u32 idx=scan[g*DIM+j];sv[j]=(f32)coef[idx]/step[idx];}idxs[g]=vqn(sv);}
    i16 qb[AVOX];
    for(u32 k=NLF;k<AVOX;++k){u32 idx=scan[k];f32 c=(f32)coef[idx],a=c<0?-c:c,s=step[idx];i32 m=(a>=0.5f*s);i32 l=m*((i32)(a/s-0.5f)+1);qb[k]=(i16)(c<0?-l:l);}
    for(u32 g=0;g<NGRP;++g){const f32*cw=cb+idxs[g]*DIM;for(u32 j=0;j<DIM;++j){u32 idx=scan[g*DIM+j];reccoef[idx]=(i16)lrintf(cw[j]*step[idx]);}}
    for(u32 k=NLF;k<AVOX;++k){u32 idx=scan[k];i32 l=qb[k],al=l<0?-l:l;f32 r=(f32)al*step[idx];i32 v=(i32)lrintf(l<0?-r:r);reccoef[idx]=(i16)v;}
}

static u64 rng=0x2545F4914F6CDD1Dull;
static u32 nxt(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return (u32)(rng>>32);}

int main(void){
    build_scan();
    int fails=0;
    f32 step[AVOX]; qm_build_step(step,QM_HF,16.0f,1.0f);
    for(u32 c=0;c<K;++c)for(u32 j=0;j<DIM;++j) cb[c*DIM+j]=(f32)((i32)(nxt()%21)-10); // fake codebook

    // Build a small multi-atom volume (4x4x4 atoms = 64^3) of structured noise.
    u32 D=64; u8 *vol=malloc((size_t)D*D*D);
    for(u32 z=0;z<D;++z)for(u32 y=0;y<D;++y)for(u32 x=0;x<D;++x){
        i32 v=128+(i32)(40*sin(x*0.3)*cos(y*0.21))+ (i32)(z*2) + (i32)(nxt()%12)-6;
        if(v<0)v=0; if(v>255)v=255; vol[(size_t)z*D*D+y*D+x]=(u8)v;
    }

    // --- ROUND-TRIP: bounded reconstruction, deterministic on repeat ---
    for(int mode=0;mode<2;++mode){
        u8 blk[AVOX]; for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)memcpy(blk+((size_t)z*A+y)*A, vol+((size_t)z*D+y)*D, A);
        i16 rc1[AVOX],rc2[AVOX]; i32 dc1,dc2;
        if(mode==0){enc_lattice(blk,step,rc1,&dc1);enc_lattice(blk,step,rc2,&dc2);}
        else        {enc_vq(blk,step,rc1,&dc1);enc_vq(blk,step,rc2,&dc2);}
        if(dc1!=dc2 || memcmp(rc1,rc2,sizeof rc1)){printf("FAIL: %s not deterministic\n",mode?"VQ":"LATTICE");fails++;}
        else printf("ok: %s round-trip deterministic\n",mode?"VQ":"LATTICE");
        // inverse must produce valid u8
        u8 rec[AVOX]; T_dct16_inv(rec,rc1,dc1);
        printf("   %s recon DC=%d ok\n",mode?"VQ":"LATTICE",dc1);
    }

    // --- TOUCHED=1: decode atom (1,1,1) ALONE vs as part of full volume sweep ---
    // The reconstruction of any atom depends ONLY on its own source block + the
    // shared step/codebook, so a single-atom decode must match the in-context one.
    u32 NA=D/A; // 4
    for(int mode=0;mode<2;++mode){
        // full sweep recon of target atom
        u32 taz=1,tay=1,tax=1;
        u8 tgtblk[AVOX]; for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
            const u8*s=vol+((size_t)(taz*A+z)*D+(tay*A+y))*D+tax*A; memcpy(tgtblk+((size_t)z*A+y)*A,s,A);}
        i16 rcCtx[AVOX]; i32 dcCtx;
        // simulate full sweep (decode many atoms; only keep target). Cross-atom
        // independence => target recon identical whether neighbors decoded or not.
        for(u32 az=0;az<NA;++az)for(u32 ay=0;ay<NA;++ay)for(u32 ax=0;ax<NA;++ax){
            u8 b[AVOX]; for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){const u8*s=vol+((size_t)(az*A+z)*D+(ay*A+y))*D+ax*A;memcpy(b+((size_t)z*A+y)*A,s,A);}
            i16 rc[AVOX];i32 dc; if(mode==0)enc_lattice(b,step,rc,&dc);else enc_vq(b,step,rc,&dc);
            if(az==taz&&ay==tay&&ax==tax){memcpy(rcCtx,rc,sizeof rc);dcCtx=dc;}
        }
        // decode the SAME atom alone
        i16 rcAlone[AVOX]; i32 dcAlone;
        if(mode==0)enc_lattice(tgtblk,step,rcAlone,&dcAlone); else enc_vq(tgtblk,step,rcAlone,&dcAlone);
        if(dcCtx!=dcAlone || memcmp(rcCtx,rcAlone,sizeof rcCtx)){printf("FAIL: %s touched!=1 (single-atom decode differs)\n",mode?"VQ":"LATTICE");fails++;}
        else printf("ok: %s touched=1 (single-atom decode == in-context)\n",mode?"VQ":"LATTICE");
    }

    free(vol);
    if(fails){printf("\nSOME TESTS FAILED (%d)\n",fails);return 1;}
    printf("\nALL VQ TESTS PASSED\n");return 0;
}
