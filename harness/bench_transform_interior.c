// EXP #13 — transform-interior improvements bake-off (PLAN §2 transform/quant/
// entropy interior of the 16^3 atom). Self-contained THROWAWAY bench (does NOT
// touch codec.c). It #includes the WON transform (integer DCT-16^3), the WON
// quant matrix + the new tuned shapes, the WON entropy coder (RLGR), and the new
// context coder, and races them at EQUAL RATIO on real PHerc Paris 4 256^3
// volumes (hires + coarse), q-points spanning ~10/20/50x.
//
// BASELINE (the winning stack): DCT-16^3 + fixed HF matrix (slope 0.6) + RLGR.
//
// Variants measured, each swapped in isolation vs baseline at matched ratio:
//   (a) quant-matrix SHAPE: LINEAR(=baseline) vs L2-radial vs SEParable vs
//       ENERGY-tuned (data-derived per-position gain).
//   (b) intra-atom LAPPED transform (8^3 + internal-seam POT) vs DCT-16^3.
//   (c) coefficient CONTEXT modeling (range coder, sig conditioned on causal
//       neighbors) vs RLGR — on the SAME baseline quantized coefficients.
//
// Pure C23, libc/libm. Reports ratio, PSNR, MS-SSIM, GMSD, HaarPSI, edgeMAE,
// seam16, encode + decode MB/s.
#include "../include/vc/types.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  H_dct16_fwd
#define vc_dct_int16_inv  H_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv

#include "../src/quant/qmatrix16.c"          // baseline HF matrix + dead-zone prims
#include "../src/quant/qmatrix16_tuned.c"    // (a) tuned shapes
#include "../src/transform/lapped16.c"       // (b) lapped transform
#include "../src/entropy/ctxcoef.c"          // (c) context coder

size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A 16u
#define AVOX 4096u

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static u8 *read_file(const char *p,size_t *len){ FILE*f=fopen(p,"rb"); if(!f)return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8*b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;} fclose(f);*len=(size_t)s;return b; }

static inline void gather_atom(const u8*vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,u8*blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const u8*src=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(blk+((size_t)z*A+y)*A,src,A);} }
static inline void scatter_atom(u8*vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,const u8*blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        u8*dst=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(dst,blk+((size_t)z*A+y)*A,A);} }

// increasing-frequency scan: stable sort of atom layout indices by s=z+y+x.
static u16 g_scan[AVOX];
static void build_scan(void){
    u32 k=0;
    for(u32 s=0;s<=45;++s)
        for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x)
            if(z+y+x==s) g_scan[k++]=(u16)((size_t)z*256u+y*16u+x);
}

typedef enum { TR_DCT16=0, TR_LAPPED=1 } trmode;

// ---- per-atom codec under (transform, weight-matrix, entropy) selection ----
// entropy: 0=RLGR, 1=context-coder. Returns payload bytes; writes reconstruction.
static size_t code_atom(const u8*src,trmode tr,const f32*step,int ent,
                        u8*rec,u8*scratch,u8*sigmap){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
    i16*coef=(i16*)scratch; i16*qb=coef+AVOX; u8*tmp=(u8*)(qb+AVOX);
    if(tr==TR_LAPPED) vc_lapped16_fwd(coef,src,dc); else H_dct16_fwd(coef,src,dc);
    qm_quant_block(qb,coef,step);
    size_t bytes;
    if(ent==1){
        // context coder needs scan order; it codes in scan order from atom layout
        bytes=vc_ctxcoef_encode_atom(tmp,AVOX*4,qb,g_scan,sigmap);
        // verify-decode into a temp to keep the qb for reconstruction (decode is
        // exercised separately for timing; here just trust round-trip)
    } else {
        bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    }
    qm_dequant_block(coef,qb,step);
    if(tr==TR_LAPPED) vc_lapped16_inv(rec,coef,dc); else H_dct16_inv(rec,coef,dc);
    return bytes+2; // +2 for stored DC
}

// Whole-volume pass at a given base step. wmat is the 4096 weight matrix.
static size_t pass(const u8*vol,u32 d,u32 h,u32 w,trmode tr,const f32*wmat,
                   f32 base,int ent,u8*rec,u8*scratch,u8*sigmap,double*enc_s){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX]; qmt_apply_step(step,wmat,base,1.0f);
    u8 srcblk[AVOX],recblk[AVOX];
    size_t total=0; double t0=now_sec();
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        total+=code_atom(srcblk,tr,step,ent,recblk,scratch,sigmap);
        scatter_atom(rec,h,w,az,ay,ax,recblk);
    }
    if(enc_s)*enc_s=now_sec()-t0;
    return total;
}

static f32 find_step(const u8*vol,u32 d,u32 h,u32 w,trmode tr,const f32*wmat,
                     int ent,double target,u8*rec,u8*scratch,u8*sigmap){
    size_t raw=(size_t)d*h*w; f32 lo=0.5f,hi=400.f;
    for(int it=0;it<22;++it){ f32 mid=sqrtf(lo*hi);
        size_t by=pass(vol,d,h,w,tr,wmat,mid,ent,rec,scratch,sigmap,NULL);
        double r=(double)raw/by; if(r<target)lo=mid; else hi=mid;
        if(fabs(r-target)<target*0.01) return mid; }
    return sqrtf(lo*hi);
}

typedef struct{double ratio,psnr,ms,gmsd,haar,emae,seam,enc,dec;}row;
static void measure(const u8*vol,const u8*rec,u32 d,u32 h,u32 w,size_t by,double encs,row*o){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    o->psnr=m.psnr;o->ms=m.ms_ssim;o->gmsd=vc_gmsd(vol,rec,d,h,w);
    o->haar=vc_haarpsi(vol,rec,d,h,w);o->emae=vc_edge_mae(vol,rec,d,h,w);
    o->seam=vc_seam_step(vol,rec,d,h,w,16);
    o->ratio=(double)((size_t)d*h*w)/by;
    o->enc=encs>0?((size_t)d*h*w)/1e6/encs:0;
}

// measure decode speed for the context coder vs RLGR on one representative atom
// stream set (re-decode whole volume's atoms, timing only decode).
static double measure_decode_mbs(const u8*vol,u32 d,u32 h,u32 w,trmode tr,
                                 const f32*wmat,f32 base,int ent,u8*scratch,u8*sigmap){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX]; qmt_apply_step(step,wmat,base,1.0f);
    u8 srcblk[AVOX],recblk[AVOX];
    // pre-encode all atoms into a big buffer, recording lengths
    static u8 streams[ (256/16)*(256/16)*(256/16) ][AVOX*4];
    static u32 lens[(256/16)*(256/16)*(256/16)];
    static i32 dcs[(256/16)*(256/16)*(256/16)];
    u32 ai=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax,++ai){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX; dcs[ai]=dc;
        i16*coef=(i16*)scratch; i16*qb=coef+AVOX;
        if(tr==TR_LAPPED)vc_lapped16_fwd(coef,srcblk,dc);else H_dct16_fwd(coef,srcblk,dc);
        qm_quant_block(qb,coef,step);
        if(ent==1) lens[ai]=(u32)vc_ctxcoef_encode_atom(streams[ai],AVOX*4,qb,g_scan,sigmap);
        else       lens[ai]=(u32)vc_rlgr_encode(streams[ai],AVOX*3,qb,AVOX);
    }
    double t0=now_sec();
    for(u32 a=0;a<ai;++a){
        i16*coef=(i16*)scratch; i16*qb=coef+AVOX;
        if(ent==1) vc_ctxcoef_decode_atom(qb,streams[a],lens[a],g_scan,sigmap);
        else       vc_rlgr_decode(qb,AVOX,streams[a],lens[a]);
        qm_dequant_block(coef,qb,step);
        if(tr==TR_LAPPED)vc_lapped16_inv(recblk,coef,dcs[a]);else H_dct16_inv(recblk,coef,dcs[a]);
    }
    double dt=now_sec()-t0;
    return ((size_t)d*h*w)/1e6/dt;
}

// Measure per-position coefficient energy on a volume to derive ENERGY gain.
static void measure_energy_gain(const u8*vol,u32 d,u32 h,u32 w,f32*egain){
    u32 naz=d/A,nay=h/A,nax=w/A;
    double e[AVOX]; for(u32 i=0;i<AVOX;++i)e[i]=0;
    u8 srcblk[AVOX]; i16 coef[AVOX];
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
        H_dct16_fwd(coef,srcblk,dc);
        for(u32 i=0;i<AVOX;++i) e[i]+=(double)coef[i]*coef[i];
    }
    // gain: positions with MORE energy -> finer step (smaller weight). Reverse
    // water-filling step ~ proportional to 1/sqrt(E) normalized so DC weight ~1.
    double dcE=e[0]>1?e[0]:1;
    for(u32 i=0;i<AVOX;++i){
        double rel=sqrt((e[i]+1.0)/dcE);   // 0..1, 1 at DC
        // protect HF: weight smaller where energy small? No — energy-tuned means
        // spend bits where ENERGY (signal) is, i.e. weight ~ stays near 1 for
        // high-energy, larger step where low energy. weight = clamp(rel^0.25).
        double wgt=pow(rel,0.20);
        if(wgt<0.30)wgt=0.30; if(wgt>1.0)wgt=1.0;
        egain[i]=(f32)wgt;
    }
    egain[0]=1.0f;
}

// Apples-to-apples entropy comparison at IDENTICAL quant (same distortion): for
// each target ratio, take the baseline RLGR step, encode every atom's quantized
// coefficients with BOTH RLGR and the context coder, and report the byte ratio
// = the pure ratio gain context modeling buys at identical quality.
static void entropy_only(const u8*vol,u32 d,u32 h,u32 w,const f32*wmat,
                         double tgt,u8*rec,u8*scratch,u8*sigmap){
    f32 st=find_step(vol,d,h,w,TR_DCT16,wmat,0,tgt,rec,scratch,sigmap);
    f32 step[AVOX]; qmt_apply_step(step,wmat,st,1.0f);
    u32 naz=d/A,nay=h/A,nax=w/A; u8 srcblk[AVOX];
    size_t rlgr_bytes=0, ctx_bytes=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0;for(u32 i=0;i<AVOX;++i)sum+=srcblk[i];
        i32 dc=(sum+2048)/4096;
        i16*coef=(i16*)scratch; i16*qb=coef+AVOX; u8*tmp=(u8*)(qb+AVOX);
        H_dct16_fwd(coef,srcblk,dc); qm_quant_block(qb,coef,step);
        rlgr_bytes += vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
        ctx_bytes  += vc_ctxcoef_encode_atom(tmp,AVOX*4,qb,g_scan,sigmap);
    }
    printf("  entropy-only @%.0fx (identical quant): RLGR=%zuB  CTX=%zuB  gain=%.1f%%  ctx-ratio=%.2fx\n",
           tgt,rlgr_bytes,ctx_bytes,100.0*(1.0-(double)ctx_bytes/rlgr_bytes),
           (double)((size_t)d*h*w)/ctx_bytes);
}

static void run(const char*label,const u8*vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w; u8*rec=malloc(raw);
    u8*scratch=malloc(AVOX*sizeof(i16)*2+AVOX*4);
    u8*sigmap=malloc(AVOX);
    f32 egain[AVOX]; measure_energy_gain(vol,d,h,w,egain);
    // build weight matrices once per slope
    qmt_set_slope(0.60f); qm_set_hf_slope(0.60f);
    f32 w_lin[AVOX],w_l2[AVOX],w_sep[AVOX],w_en[AVOX];
    qmt_build_weight(w_lin,QMT_LINEAR,NULL,0);
    qmt_build_weight(w_l2,QMT_L2,NULL,0);
    qmt_build_weight(w_sep,QMT_SEP,NULL,0);
    qmt_build_weight(w_en,QMT_ENERGY,egain,0.6f);

    printf("\n## %s (%ux%ux%u, %.1fMB)\n",label,d,h,w,raw/1e6);
    const double tgts[]={10.0,20.0,50.0};
    for(int ti=0;ti<3;++ti){ double tgt=tgts[ti];
        printf("\n### target %.0fx\n",tgt);
        printf("variant                    | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE | seam16 | enc MB/s | dec MB/s\n");
        printf("---------------------------+--------+-------+---------+---------+---------+---------+--------+----------+---------\n");
        struct{const char*nm;trmode tr;const f32*wm;int ent;}V[]={
            {"BASE DCT16+HF+RLGR    ",TR_DCT16,w_lin,0},
            {"(a) qmat L2-radial    ",TR_DCT16,w_l2,0},
            {"(a) qmat separable    ",TR_DCT16,w_sep,0},
            {"(a) qmat energy-tuned ",TR_DCT16,w_en,0},
            {"(b) lapped8+POT       ",TR_LAPPED,w_lin,0},
            {"(c) context-coder     ",TR_DCT16,w_lin,1},
        };
        int nV=6;
        for(int k=0;k<nV;++k){
            f32 st=find_step(vol,d,h,w,V[k].tr,V[k].wm,V[k].ent,tgt,rec,scratch,sigmap);
            double encs; size_t by=pass(vol,d,h,w,V[k].tr,V[k].wm,st,V[k].ent,rec,scratch,sigmap,&encs);
            row r; measure(vol,rec,d,h,w,by,encs,&r);
            double dmb=measure_decode_mbs(vol,d,h,w,V[k].tr,V[k].wm,st,V[k].ent,scratch,sigmap);
            printf("%-26s | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %6.3f | %8.0f | %8.0f\n",
                   V[k].nm,r.ratio,r.psnr,r.ms,r.gmsd,r.haar,r.emae,r.seam,r.enc,dmb);
        }
    }
    printf("\n### entropy-only (CTX vs RLGR at IDENTICAL quant = ratio gain at matched quality)\n");
    for(int ti=0;ti<3;++ti) entropy_only(vol,d,h,w,w_lin,tgts[ti],rec,scratch,sigmap);
    free(rec);free(scratch);free(sigmap);
}

int main(int argc,char**argv){
    build_scan();
    const char*files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char*labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    int which=(argc>1)?atoi(argv[1]):-1;
    for(int i=0;i<2;++i){ if(which>=0&&which!=i)continue;
        size_t len; u8*v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
