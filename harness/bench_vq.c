// Vector / lattice quantization bake-off (PLAN §2 "Quantizer" row, parked-but-look).
//
// QUESTION: can vector/lattice quantization of small LOW-FREQUENCY coefficient
// groups pack rate-distortion tighter than the chosen per-coefficient SCALAR
// dead-zone (qmatrix16 HF matrix), and at what DECODE cost? VQ decode is a table
// lookup which can be heavy; this is a decode-speed-sensitive random-access codec,
// so the honest verdict weighs ratio gain against decode cost.
//
// THROWAWAY self-contained bench (does NOT touch codec.c / ratectrl / chunkmodel),
// mirroring bench_hfdist.c: it #includes the WON transform (integer DCT-16^3) and
// the per-coefficient 16^3 quant matrix directly, links the WON entropy coder
// (RLGR) + the metric bundle, and runs at EQUAL RATIO on real PHerc Paris 4.
//
// Pipeline atom = 16^3 (won transform + random-access unit). Per-atom mean removed
// before DCT, stored as a side i16 (counted). The 4096 coefficients per atom are
// split into:
//   - LOW-FREQUENCY group: the first NLF coefficients in a low-freq-first scan
//     (by u+v+w then raster). THIS is where VQ/lattice operates.
//   - HIGH-FREQUENCY remainder: always scalar dead-zone (HF matrix) + RLGR, as the
//     baseline. (HF coefficients are sparse/near-zero; VQ buys nothing there and
//     the entropy coder already handles the long zero runs.)
//
// CANDIDATES (all hit the SAME global target ratio via base-step bisection):
//   SCALAR  — baseline: HF-matrix scalar dead-zone over ALL 4096 coeffs + RLGR.
//   LATTICE — D4 lattice quantization of the low-freq group in 4-D subvectors;
//             lattice point index entropy-coded. Decode = arithmetic (cheap).
//   VQ256   — trained k-means codebook (256 entries, dim D=4) on low-freq sub-
//             vectors of THIS volume; per-group index (1 byte) RLGR-coded; the
//             codebook is shared/global side info (counted once). Decode = LUT.
//
// We report, at the matched ratio: PSNR, MS-SSIM, GMSD, HaarPSI, edgeMAE, seam16,
// ENCODE MB/s and DECODE MB/s (the load-bearing number for VQ). Pure C23, libc/libm.
#include "../include/vc/types.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- the won transform, renamed into this TU ------------------------------
#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  H_dct16_fwd
#define vc_dct_int16_inv  H_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv

// --- the per-coefficient 16^3 quant matrix --------------------------------
#include "../src/quant/qmatrix16.c"

// --- the won entropy coder (RLGR) -----------------------------------------
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A   16u
#define AVOX 4096u

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}

static inline void gather_atom(const u8 *vol,u32 d,u32 h,u32 w,u32 az,u32 ay,u32 ax,u8 *blk){
    (void)d;
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const u8 *src = vol + ((size_t)(az*A+z)*h + (ay*A+y))*w + ax*A;
        memcpy(blk + ((size_t)z*A+y)*A, src, A);
    }
}
static inline void scatter_atom(u8 *vol,u32 d,u32 h,u32 w,u32 az,u32 ay,u32 ax,const u8 *blk){
    (void)d;
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        u8 *dst = vol + ((size_t)(az*A+z)*h + (ay*A+y))*w + ax*A;
        memcpy(dst, blk + ((size_t)z*A+y)*A, A);
    }
}

// ---------------------------------------------------------------------------
// Low-freq-first scan order over the 4096 coefficients: sort positions by
// frequency index s=u+v+w (then raster) so the first NLF entries are the
// lowest-frequency coefficients (where most signal energy lives -> the only
// place VQ can plausibly help). Built once.
// ---------------------------------------------------------------------------
static u16 g_scan[AVOX];        // scan[k] = coef index of the k-th lowest-freq coef
static void build_scan(void){
    // stable sort positions by (s, index) via counting on s then preserving order.
    u32 cnt[46]={0}, off[46]={0};
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x) cnt[z+y+x]++;
    u32 acc=0; for(u32 s=0;s<46;++s){ off[s]=acc; acc+=cnt[s]; }
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        u32 s=z+y+x; u32 idx=(u32)z*256u+(u32)y*16u+x;
        g_scan[off[s]++]=(u16)idx;
    }
}

// ---- group geometry: low-freq group = first NLF coeffs, in D-dim subvectors ----
#ifndef VQ_NLF
#define VQ_NLF 64u              // # low-freq coefficients per atom handled by VQ/lattice
#endif
#define VQ_DIM 4u               // subvector dimension (D4 lattice / VQ dim)
#define VQ_NGRP (VQ_NLF/VQ_DIM) // groups per atom

// ===========================================================================
// SCALAR baseline: HF-matrix dead-zone over all 4096, RLGR.
// ===========================================================================
static size_t code_atom_scalar(const u8 *src, const f32 *step, u8 *rec, u8 *scratch){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc = (sum + (i32)(AVOX/2)) / (i32)AVOX;
    i16 *coef=(i16*)scratch, *qb=coef+AVOX;
    H_dct16_fwd(coef,src,dc);
    qm_quant_block(qb,coef,step);
    u8 *tmp=(u8*)(qb+AVOX);
    size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    qm_dequant_block(coef,qb,step);
    H_dct16_inv(rec,coef,dc);
    return bytes+2;
}
// decode-only path (scalar): given the stored qb levels, dequant + inverse.
static void decode_atom_scalar(const i16 *qb,const f32 *step,i32 dc,u8 *rec,i16 *coefscratch){
    qm_dequant_block(coefscratch,qb,step);
    H_dct16_inv(rec,coefscratch,dc);
}

// ===========================================================================
// LATTICE (D4): quantize low-freq subvectors to the D4 lattice.
//   D4 = integer points whose coordinate sum is even. The quantizer rounds to
//   the nearest integer point, and if the rounded sum is odd, flips the single
//   coordinate with the largest rounding error to the next integer (Conway &
//   Sloane Alg. 2 / "fast quantizer"). The lattice point coordinates are the
//   transmitted integers (then RLGR-coded like scalar levels). Distortion per
//   group is lower than scalar at the same average integer magnitude because the
//   D4 Voronoi cell packs better than the cubic Z^4 cell (gain ~ +1.5 dB / 0.66
//   bit packing gain in theory).
// The lattice operates on coefficients PRE-DIVIDED by their per-coef step (so the
// HF matrix shape is preserved); the dequant multiplies the step back.
// ===========================================================================
static inline void d4_quantize(const f32 *restrict in, i32 *restrict out){
    // round each to nearest int
    i32 r[VQ_DIM]; i32 sum=0;
    f32 maxerr=-1.f; u32 argmax=0; i32 dir=0;
    for(u32 i=0;i<VQ_DIM;++i){
        f32 v=in[i];
        i32 ri=(i32)lrintf(v);
        r[i]=ri; sum+=ri;
        f32 err=v-(f32)ri;            // in [-0.5,0.5]
        f32 ae = err<0?-err:err;
        if(ae>maxerr){ maxerr=ae; argmax=i; dir=err>0?1:-1; }
    }
    if(sum & 1){ r[argmax]+=dir?dir:1; } // sum odd -> move worst coord toward its target
    for(u32 i=0;i<VQ_DIM;++i) out[i]=r[i];
}

// Encode one atom with LATTICE on the low-freq group, scalar on the HF remainder.
// Returns total bytes; writes reconstruction. quant levels for the WHOLE 4096 are
// assembled in scan layout then RLGR-coded as one stream (the lattice points sit
// in the low-freq slots, scalar levels in the rest).
static size_t code_atom_lattice(const u8 *src,const f32 *step,u8 *rec,u8 *scratch){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
    i16 *coef=(i16*)scratch, *qb=coef+AVOX;     // qb in SCAN order
    H_dct16_fwd(coef,src,dc);
    // low-freq group via D4 lattice
    for(u32 g=0;g<VQ_NGRP;++g){
        f32 sv[VQ_DIM]; i32 lp[VQ_DIM];
        for(u32 j=0;j<VQ_DIM;++j){
            u32 k=g*VQ_DIM+j; u32 idx=g_scan[k];
            sv[j]=(f32)coef[idx]/step[idx];      // normalize by per-coef step
        }
        d4_quantize(sv,lp);
        for(u32 j=0;j<VQ_DIM;++j) qb[g*VQ_DIM+j]=(i16)lp[j];
    }
    // HF remainder via scalar dead-zone (scan order)
    for(u32 k=VQ_NLF;k<AVOX;++k){
        u32 idx=g_scan[k];
        f32 c=(f32)coef[idx]; f32 a=c<0?-c:c; f32 s=step[idx];
        i32 m=(a>=0.5f*s); i32 lvl=m*((i32)(a/s-0.5f)+1);
        qb[k]=(i16)(c<0?-lvl:lvl);
    }
    u8 *tmp=(u8*)(qb+AVOX);
    size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    // reconstruct: lattice points * step in low-freq slots, scalar dequant elsewhere
    for(u32 g=0;g<VQ_NGRP;++g)for(u32 j=0;j<VQ_DIM;++j){
        u32 k=g*VQ_DIM+j; u32 idx=g_scan[k];
        coef[idx]=(i16)lrintf((f32)qb[k]*step[idx]);
    }
    for(u32 k=VQ_NLF;k<AVOX;++k){
        u32 idx=g_scan[k]; i32 l=qb[k]; i32 al=l<0?-l:l;
        f32 r=(f32)al*step[idx]; i32 v=(i32)lrintf(l<0?-r:r);
        if(v>32767)v=32767; else if(v<-32768)v=-32768; coef[idx]=(i16)v;
    }
    H_dct16_inv(rec,coef,dc);
    return bytes+2;
}

// ===========================================================================
// VQ256: trained codebook on normalized low-freq subvectors.
//   Codebook: K=256 entries of dim D=4, k-means on the (normalized-by-step)
//   low-freq subvectors sampled from the whole volume at a TRAINING step. At
//   encode each group picks nearest codeword (exhaustive: K*D mults). The index
//   (1 byte) goes into the qb stream in the low-freq slot (one i16 per group),
//   the remaining low-freq slots are zeroed (so they cost ~nothing in RLGR), HF
//   remainder scalar as before. Decode = LUT (codebook[index]) * step -> the
//   load-bearing cost we measure.
// The codebook is global side info: K*D*sizeof(i16) bytes counted ONCE.
// ===========================================================================
#define VQ_K 256u
static f32 g_cb[VQ_K*VQ_DIM];   // codebook, normalized-coef space

// collect training vectors (normalized low-freq subvectors) into buf, return count
static u32 collect_training(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,
                            f32 *buf,u32 maxv,u8 *scratch){
    f32 step[AVOX]; qm_build_step(step,QM_HF,base_step,1.0f);
    u32 naz=d/A,nay=h/A,nax=w/A; u32 cnt=0;
    u8 blk[AVOX];
    // sample every Nth atom to bound training cost
    u32 stride=3;
    for(u32 az=0;az<naz;az+=stride)for(u32 ay=0;ay<nay;ay+=stride)for(u32 ax=0;ax<nax;ax+=stride){
        gather_atom(vol,d,h,w,az,ay,ax,blk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=blk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
        i16 *coef=(i16*)scratch; H_dct16_fwd(coef,blk,dc);
        for(u32 g=0;g<VQ_NGRP;++g){
            if(cnt>=maxv) return cnt;
            for(u32 j=0;j<VQ_DIM;++j){ u32 idx=g_scan[g*VQ_DIM+j]; buf[cnt*VQ_DIM+j]=(f32)coef[idx]/step[idx]; }
            cnt++;
        }
    }
    return cnt;
}

static inline u32 vq_nearest(const f32 *v){
    u32 best=0; f32 bd=1e30f;
    for(u32 c=0;c<VQ_K;++c){
        const f32 *cw=g_cb+c*VQ_DIM; f32 dd=0;
        for(u32 j=0;j<VQ_DIM;++j){ f32 e=v[j]-cw[j]; dd+=e*e; }
        if(dd<bd){bd=dd;best=c;}
    }
    return best;
}

static void train_codebook(const f32 *train,u32 nt){
    // k-means++ light: init by sampling spread, then 8 Lloyd iterations.
    for(u32 c=0;c<VQ_K;++c){ u32 s=(u32)((u64)c*nt/VQ_K); memcpy(g_cb+c*VQ_DIM,train+s*VQ_DIM,VQ_DIM*sizeof(f32)); }
    f32 *acc=malloc(VQ_K*VQ_DIM*sizeof(f32)); u32 *na=malloc(VQ_K*sizeof(u32));
    for(int it=0;it<8;++it){
        memset(acc,0,VQ_K*VQ_DIM*sizeof(f32)); memset(na,0,VQ_K*sizeof(u32));
        for(u32 i=0;i<nt;++i){ u32 b=vq_nearest(train+i*VQ_DIM); na[b]++; for(u32 j=0;j<VQ_DIM;++j) acc[b*VQ_DIM+j]+=train[i*VQ_DIM+j]; }
        for(u32 c=0;c<VQ_K;++c) if(na[c]){ for(u32 j=0;j<VQ_DIM;++j) g_cb[c*VQ_DIM+j]=acc[c*VQ_DIM+j]/(f32)na[c]; }
    }
    free(acc); free(na);
}

static size_t code_atom_vq(const u8 *src,const f32 *step,u8 *rec,u8 *scratch){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
    i16 *coef=(i16*)scratch, *qb=coef+AVOX;
    H_dct16_fwd(coef,src,dc);
    memset(qb,0,AVOX*sizeof(i16));
    // low-freq groups -> codebook indices, stored one i16 per group at slot g
    u32 idxs[VQ_NGRP];
    for(u32 g=0;g<VQ_NGRP;++g){
        f32 sv[VQ_DIM];
        for(u32 j=0;j<VQ_DIM;++j){ u32 idx=g_scan[g*VQ_DIM+j]; sv[j]=(f32)coef[idx]/step[idx]; }
        u32 ci=vq_nearest(sv); idxs[g]=ci; qb[g]=(i16)ci;
    }
    // HF remainder scalar, written from slot VQ_NLF on (low-freq slots VQ_NGRP..VQ_NLF stay 0)
    for(u32 k=VQ_NLF;k<AVOX;++k){
        u32 idx=g_scan[k]; f32 c=(f32)coef[idx]; f32 a=c<0?-c:c; f32 s=step[idx];
        i32 m=(a>=0.5f*s); i32 lvl=m*((i32)(a/s-0.5f)+1); qb[k]=(i16)(c<0?-lvl:lvl);
    }
    u8 *tmp=(u8*)(qb+AVOX);
    size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    // reconstruct (this IS the decode work: LUT + step)
    for(u32 g=0;g<VQ_NGRP;++g){
        const f32 *cw=g_cb+idxs[g]*VQ_DIM;
        for(u32 j=0;j<VQ_DIM;++j){ u32 idx=g_scan[g*VQ_DIM+j]; coef[idx]=(i16)lrintf(cw[j]*step[idx]); }
    }
    for(u32 k=VQ_NLF;k<AVOX;++k){
        u32 idx=g_scan[k]; i32 l=qb[k]; i32 al=l<0?-l:l; f32 r=(f32)al*step[idx];
        i32 v=(i32)lrintf(l<0?-r:r); if(v>32767)v=32767; else if(v<-32768)v=-32768; coef[idx]=(i16)v;
    }
    H_dct16_inv(rec,coef,dc);
    return bytes+2;
}

// ===========================================================================
// Decode-cost micro-bench: pure dequant+inverse for each mode (no entropy, no
// fwd transform) over all atoms, to isolate the VQ LUT cost vs scalar.
// ===========================================================================
typedef enum { M_SCALAR, M_LATTICE, M_VQ } vqmode;

static double decode_speed(const u8 *vol,u32 d,u32 h,u32 w,vqmode mode,f32 base_step,u8 *scratch){
    f32 step[AVOX]; qm_build_step(step,QM_HF,base_step,1.0f);
    u32 naz=d/A,nay=h/A,nax=w/A;
    u8 blk[AVOX], rec[AVOX]; i16 *coef=(i16*)scratch, *qb=coef+AVOX;
    // pre-encode all atoms' qb into a big store so the decode loop is pure-decode.
    u32 natoms=naz*nay*nax;
    i16 *store=malloc((size_t)natoms*AVOX*sizeof(i16));
    i32 *dcs=malloc(natoms*sizeof(i32));
    u32 *vqidx=malloc((size_t)natoms*VQ_NGRP*sizeof(u32));
    u32 ai=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax,++ai){
        gather_atom(vol,d,h,w,az,ay,ax,blk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=blk[i]; i32 dc=(sum+2048)/4096; dcs[ai]=dc;
        H_dct16_fwd(coef,blk,dc);
        i16 *st=store+(size_t)ai*AVOX;
        if(mode==M_SCALAR){ qm_quant_block(st,coef,step); }
        else if(mode==M_LATTICE){
            for(u32 g=0;g<VQ_NGRP;++g){ f32 sv[VQ_DIM]; i32 lp[VQ_DIM];
                for(u32 j=0;j<VQ_DIM;++j){u32 idx=g_scan[g*VQ_DIM+j]; sv[j]=(f32)coef[idx]/step[idx];}
                d4_quantize(sv,lp); for(u32 j=0;j<VQ_DIM;++j) st[g*VQ_DIM+j]=(i16)lp[j]; }
            for(u32 k=VQ_NLF;k<AVOX;++k){u32 idx=g_scan[k]; f32 c=(f32)coef[idx];f32 a=c<0?-c:c;f32 s=step[idx];
                i32 m=(a>=0.5f*s);i32 lvl=m*((i32)(a/s-0.5f)+1); st[k]=(i16)(c<0?-lvl:lvl);}
        } else {
            for(u32 g=0;g<VQ_NGRP;++g){ f32 sv[VQ_DIM];
                for(u32 j=0;j<VQ_DIM;++j){u32 idx=g_scan[g*VQ_DIM+j]; sv[j]=(f32)coef[idx]/step[idx];}
                vqidx[(size_t)ai*VQ_NGRP+g]=vq_nearest(sv); }
            for(u32 k=VQ_NLF;k<AVOX;++k){u32 idx=g_scan[k]; f32 c=(f32)coef[idx];f32 a=c<0?-c:c;f32 s=step[idx];
                i32 m=(a>=0.5f*s);i32 lvl=m*((i32)(a/s-0.5f)+1); st[k]=(i16)(c<0?-lvl:lvl);}
        }
    }
    // timed pure-decode pass
    double t0=now_sec();
    ai=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax,++ai){
        i16 *st=store+(size_t)ai*AVOX;
        if(mode==M_SCALAR){ qm_dequant_block(coef,st,step); }
        else if(mode==M_LATTICE){
            for(u32 g=0;g<VQ_NGRP;++g)for(u32 j=0;j<VQ_DIM;++j){u32 k=g*VQ_DIM+j;u32 idx=g_scan[k]; coef[idx]=(i16)lrintf((f32)st[k]*step[idx]);}
            for(u32 k=VQ_NLF;k<AVOX;++k){u32 idx=g_scan[k];i32 l=st[k];i32 al=l<0?-l:l;f32 r=(f32)al*step[idx];i32 v=(i32)lrintf(l<0?-r:r);if(v>32767)v=32767;else if(v<-32768)v=-32768;coef[idx]=(i16)v;}
        } else {
            for(u32 g=0;g<VQ_NGRP;++g){const f32*cw=g_cb+vqidx[(size_t)ai*VQ_NGRP+g]*VQ_DIM;
                for(u32 j=0;j<VQ_DIM;++j){u32 idx=g_scan[g*VQ_DIM+j]; coef[idx]=(i16)lrintf(cw[j]*step[idx]);}}
            for(u32 k=VQ_NLF;k<AVOX;++k){u32 idx=g_scan[k];i32 l=st[k];i32 al=l<0?-l:l;f32 r=(f32)al*step[idx];i32 v=(i32)lrintf(l<0?-r:r);if(v>32767)v=32767;else if(v<-32768)v=-32768;coef[idx]=(i16)v;}
        }
        H_dct16_inv(rec,coef,dcs[ai]);
    }
    double dt=now_sec()-t0;
    free(store);free(dcs);free(vqidx);
    return ((size_t)d*h*w)/1e6/dt;
}

// ---------------------------------------------------------------------------
typedef struct { double psnr,ms_ssim,gmsd,haarpsi,edge_mae,seam,ratio,enc_mbs,dec_mbs; } row_t;

static size_t pass(const u8 *vol,u32 d,u32 h,u32 w,vqmode mode,f32 base_step,u8 *rec,u8 *scratch){
    f32 step[AVOX]; qm_build_step(step,QM_HF,base_step,1.0f);
    u32 naz=d/A,nay=h/A,nax=w/A;
    u8 srcblk[AVOX],recblk[AVOX]; size_t total=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,d,h,w,az,ay,ax,srcblk);
        if(mode==M_SCALAR) total+=code_atom_scalar(srcblk,step,recblk,scratch);
        else if(mode==M_LATTICE) total+=code_atom_lattice(srcblk,step,recblk,scratch);
        else total+=code_atom_vq(srcblk,step,recblk,scratch);
        scatter_atom(rec,d,h,w,az,ay,ax,recblk);
    }
    return total;
}

static f32 find_step(const u8 *vol,u32 d,u32 h,u32 w,vqmode mode,double tgt,u8 *rec,u8 *scratch,size_t cb_bytes){
    size_t raw=(size_t)d*h*w; f32 lo=0.5f,hi=400.f;
    for(int it=0;it<24;++it){ f32 mid=sqrtf(lo*hi);
        size_t by=pass(vol,d,h,w,mode,mid,rec,scratch)+cb_bytes;
        double r=(double)raw/(double)by;
        if(r<tgt) lo=mid; else hi=mid;
        if(fabs(r-tgt)<tgt*0.01) return mid; }
    return sqrtf(lo*hi);
}

static void measure(const u8*vol,const u8*rec,u32 d,u32 h,u32 w,size_t bytes,double enc,double dec,row_t*o){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    o->psnr=m.psnr;o->ms_ssim=m.ms_ssim;o->gmsd=vc_gmsd(vol,rec,d,h,w);
    o->haarpsi=vc_haarpsi(vol,rec,d,h,w);o->edge_mae=vc_edge_mae(vol,rec,d,h,w);
    o->seam=vc_seam_step(vol,rec,d,h,w,16);
    o->ratio=(double)((size_t)d*h*w)/(double)bytes;
    o->enc_mbs=enc>0?((size_t)d*h*w)/1e6/enc:0;
    o->dec_mbs=dec;
}

static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w;
    u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);
    printf("\n## %s  (%ux%ux%u, %.1f MB, atom=16^3, NLF=%u dim=%u)\n",label,d,h,w,raw/1e6,VQ_NLF,VQ_DIM);
    size_t cb_bytes=(size_t)VQ_K*VQ_DIM*sizeof(i16);  // codebook side info (counted for VQ)

    const double targets[]={10.0,20.0,50.0,100.0};
    for(int ti=0;ti<4;++ti){
        double tgt=targets[ti];
        printf("\n### target %.0fx\n",tgt);
        printf("quantizer         | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE |  seam16 | enc MB/s | dec MB/s\n");
        printf("------------------+--------+-------+---------+---------+---------+---------+---------+----------+---------\n");

        struct { const char*name; vqmode mode; size_t side; } mm[3]={
            {"SCALAR (baseline) ",M_SCALAR,0},{"LATTICE D4        ",M_LATTICE,0},{"VQ256 codebook    ",M_VQ,cb_bytes}};
        for(int k=0;k<3;++k){
            if(mm[k].mode==M_VQ){
                // train codebook at the operating step (bisect a coarse step first for training)
                f32 tstep=find_step(vol,d,h,w,M_SCALAR,tgt,rec,scratch,0);
                u32 maxv=200000; f32 *tr=malloc((size_t)maxv*VQ_DIM*sizeof(f32));
                u32 nt=collect_training(vol,d,h,w,tstep,tr,maxv,scratch);
                train_codebook(tr,nt); free(tr);
            }
            f32 st=find_step(vol,d,h,w,mm[k].mode,tgt,rec,scratch,mm[k].side);
            double t0=now_sec(); size_t by=pass(vol,d,h,w,mm[k].mode,st,rec,scratch)+mm[k].side;
            double enc=now_sec()-t0;
            double dec=decode_speed(vol,d,h,w,mm[k].mode,st,scratch);
            row_t r; measure(vol,rec,d,h,w,by,enc,dec,&r);
            printf("%-17s | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %7.3f | %8.0f | %7.0f\n",
                   mm[k].name,r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam,r.enc_mbs,r.dec_mbs);
        }
    }
    free(rec);free(scratch);
}

int main(int argc,char**argv){
    build_scan();
    if(argc>=5 && argv[1][0]!='-'){
        u32 d=atoi(argv[2]),h=atoi(argv[3]),w=atoi(argv[4]); size_t len;
        u8 *v=read_file(argv[1],&len);
        if(!v||len<(size_t)d*h*w){fprintf(stderr,"bad raw\n");return 1;}
        run(argv[1],v,d,h,w); free(v); return 0;
    }
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    for(int i=0;i<2;++i){ size_t len; u8 *v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
