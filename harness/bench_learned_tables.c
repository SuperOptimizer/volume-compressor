// Learned-tables experiment (PLAN §6 "learned components AS COMPILE-TIME TABLES
// ONLY"). THROWAWAY self-contained bench (does NOT touch codec.c). Question:
// does a quant matrix TRAINED on PHerc Paris 4 scroll coefficient statistics
// (offline, baked as a static const table) beat the GENERIC hand-tuned
// HF-protecting matrix (qmatrix16.c, slope ~0.5/0.6) at equal ratio?
//
// Constraint (hard): NO per-voxel neural inference. The learned artifact is a
// static 16^3 const float table derived ONCE from corpus statistics; at codec
// time it is a plain table multiply, identical hot-path cost to the generic
// matrix. We measure ratio-at-quality gain from the scroll-tuned static table,
// and explicitly cross-validate (train hires -> test coarse and vice-versa) to
// expose OVERFITTING to one scroll.
//
// Pipeline atom = 16^3 (won transform + access unit), exactly the bench_hfdist
// recipe: per-atom DC removed, integer DCT-16^3, per-coefficient quant matrix,
// RLGR entropy. Baselines + learned table all driven to the SAME global target
// ratio by bisecting the base step. Pure C23, libc/libm.
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

// --- the generic per-coefficient 16^3 quant matrix (HF curve) -------------
#include "../src/quant/qmatrix16.c"

// --- the won entropy coder (RLGR) -----------------------------------------
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A    16u
#define AVOX 4096u

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}

static inline void gather_atom(const u8 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,u8 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const u8 *src = vol + ((size_t)(az*A+z)*h + (ay*A+y))*w + ax*A;
        memcpy(blk + ((size_t)z*A+y)*A, src, A);
    }
}
static inline void scatter_atom(u8 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,const u8 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        u8 *dst = vol + ((size_t)(az*A+z)*h + (ay*A+y))*w + ax*A;
        memcpy(dst, blk + ((size_t)z*A+y)*A, A);
    }
}

// ===========================================================================
//  LEARNED TABLE: per-coefficient relative WEIGHT trained on a corpus.
//  We collect mean-abs DCT magnitude E[|c_i|] over all atoms of the training
//  volume, then form a per-coefficient quant WEIGHT (a multiplier on the base
//  step, exactly like qm_hf_weight but per-position not per-band):
//      w_i = clamp( (E_i / E_ref)^(-p) ,  wmin, 1.0 )
//  i.e. coefficients that carry MORE energy across the scroll get a FINER step
//  (smaller weight => more bits => preserved). E_ref is a high percentile of the
//  energy profile so the curve normalizes near 1.0 at the strongest AC bands and
//  crushes the (near-zero-energy) highest 3D frequencies. This is the proper
//  variance-aware static allocation, learned from THIS data instead of the
//  hand-drawn s=u+v+w line. p and wmin are the only knobs; both baked once.
// ===========================================================================
static f32 g_learn_weight[AVOX];   // the baked static table (here trained at runtime, then frozen)

static void learn_weight_table(const u8 *vol,u32 d,u32 h,u32 w,f32 p,f32 wmin){
    static f64 acc[AVOX];
    memset(acc,0,sizeof acc);
    u32 naz=d/A,nay=h/A,nax=w/A; u64 cnt=0;
    u8 srcblk[AVOX]; i16 coef[AVOX];
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
        H_dct16_fwd(coef,srcblk,dc);
        for(u32 i=0;i<AVOX;++i){ i32 c=coef[i]; acc[i]+=(f64)(c<0?-c:c); }
        ++cnt;
    }
    for(u32 i=0;i<AVOX;++i) acc[i]/=(f64)cnt;   // mean |coef| per position
    // reference energy = 90th percentile of the AC profile (exclude DC at i=0)
    f64 prof[AVOX]; for(u32 i=0;i<AVOX;++i) prof[i]=acc[i];
    // simple selection of ~p90 via sort of a copy
    static f64 tmp[AVOX]; memcpy(tmp,prof,sizeof tmp);
    for(u32 a=1;a<AVOX;++a){ f64 k=tmp[a]; i32 b=(i32)a-1; while(b>=0&&tmp[b]>k){tmp[b+1]=tmp[b];--b;} tmp[b+1]=k; }
    f64 eref=tmp[(u32)(0.90*(AVOX-1))]; if(eref<1e-6) eref=1e-6;
    g_learn_weight[0]=1.0f;  // DC: always finest (weight 1)
    for(u32 i=1;i<AVOX;++i){
        f64 e=acc[i]; if(e<1e-9) e=1e-9;
        f64 ratio=e/eref;
        f64 ww=pow(ratio, -(f64)p);          // high energy -> small weight (finer step)
        if(ww<wmin) ww=wmin; if(ww>1.0) ww=1.0;
        g_learn_weight[i]=(f32)ww;
    }
}

static inline void learned_build_step(f32 *restrict step,f32 base_step,f32 blk_scale){
    for(u32 i=0;i<AVOX;++i){ f32 s=base_step*g_learn_weight[i]*blk_scale; if(s<0.5f)s=0.5f; step[i]=s; }
}

// ---------------------------------------------------------------------------
static size_t code_atom(const u8 *src,const f32 *step,u8 *rec,u8 *scratch){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
    i16 *coef=(i16*)scratch; i16 *qb=coef+AVOX;
    H_dct16_fwd(coef,src,dc);
    qm_quant_block(qb,coef,step);
    u8 *tmp=(u8*)(qb+AVOX);
    size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    qm_dequant_block(coef,qb,step);
    H_dct16_inv(rec,coef,dc);
    return bytes+2;
}

typedef struct { double psnr,ms_ssim,gmsd,haarpsi,edge_mae,seam,ratio; } row_t;
typedef enum { MODE_FLAT, MODE_HF, MODE_LEARN } qmode;

static size_t pass(const u8 *vol,u32 d,u32 h,u32 w,qmode mode,f32 base_step,u8 *rec,u8 *scratch){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    if(mode==MODE_LEARN) learned_build_step(step,base_step,1.0f);
    else qm_build_step(step,(mode==MODE_FLAT)?QM_FLAT:QM_HF,base_step,1.0f);
    u8 srcblk[AVOX],recblk[AVOX]; size_t total=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        total+=code_atom(srcblk,step,recblk,scratch);
        scatter_atom(rec,h,w,az,ay,ax,recblk);
    }
    return total;
}

static f32 find_step(const u8 *vol,u32 d,u32 h,u32 w,qmode mode,double target,u8 *rec,u8 *scratch){
    size_t raw=(size_t)d*h*w; f32 lo=0.5f,hi=400.f;
    for(int it=0;it<26;++it){
        f32 mid=sqrtf(lo*hi);
        size_t by=pass(vol,d,h,w,mode,mid,rec,scratch);
        double r=(double)raw/(double)by;
        if(r<target) lo=mid; else hi=mid;
        if(fabs(r-target)<target*0.008) return mid;
    }
    return sqrtf(lo*hi);
}

static void measure(const u8 *vol,const u8 *rec,u32 d,u32 h,u32 w,size_t bytes,row_t *o){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    o->psnr=m.psnr; o->ms_ssim=m.ms_ssim;
    o->gmsd=vc_gmsd(vol,rec,d,h,w); o->haarpsi=vc_haarpsi(vol,rec,d,h,w);
    o->edge_mae=vc_edge_mae(vol,rec,d,h,w); o->seam=vc_seam_step(vol,rec,d,h,w,16);
    o->ratio=(double)((size_t)d*h*w)/(double)bytes;
}

// Run FLAT / HF(generic) / LEARN at matched ratio on a TEST volume.
static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w,const double*targets,int nt){
    size_t raw=(size_t)d*h*w; u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2+AVOX*3);
    printf("\n## %s  (%ux%ux%u, atom=16^3)\n",label,d,h,w);
    for(int ti=0;ti<nt;++ti){
        double tgt=targets[ti];
        printf("\n### target %.0fx\n",tgt);
        printf("matrix              | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE | seam16\n");
        printf("--------------------+--------+-------+---------+---------+---------+---------+-------\n");
        struct { const char*name; qmode m; } cells[3]={
            {"FLAT (pure-MSE)   ",MODE_FLAT},
            {"HF generic (slope)",MODE_HF},
            {"LEARNED (scroll)  ",MODE_LEARN}};
        for(int k=0;k<3;++k){
            f32 st=find_step(vol,d,h,w,cells[k].m,tgt,rec,scratch);
            size_t by=pass(vol,d,h,w,cells[k].m,st,rec,scratch);
            row_t r; measure(vol,rec,d,h,w,by,&r);
            printf("%-19s | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %6.3f\n",
                   cells[k].name,r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam);
        }
    }
    free(rec); free(scratch);
}

// p-sweep diagnostic: train on hires, test on hires at 50x, sweep allocation
// exponent p (incl. opposite sign) and wmin to see if ANY learned setting beats
// generic HF. If none does, the verdict is robust (not a single-knob miss).
static void psweep(const u8 *hires){
    u8 *rec=malloc((size_t)256*256*256);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2+AVOX*3);
    double tgt=50.0;
    printf("\n## p-sweep (train hires / test hires @ %.0fx) — can ANY learned setting beat generic HF?\n",tgt);
    printf(" p     wmin  | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE\n");
    printf("------------+--------+-------+---------+---------+---------+--------\n");
    // generic HF reference row
    { qm_set_hf_slope(QM_HF_SLOPE_DEFAULT);
      f32 st=find_step(hires,256,256,256,MODE_HF,tgt,rec,scratch);
      size_t by=pass(hires,256,256,256,MODE_HF,st,rec,scratch);
      row_t r; measure(hires,rec,256,256,256,by,&r);
      printf("HF generic  | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f\n",
             r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae); }
    const f32 ps[]={ -0.5f,-0.25f,0.25f,0.5f,1.0f };
    const f32 wm[]={0.20f,0.40f};
    for(int a=0;a<5;++a)for(int b=0;b<2;++b){
        learn_weight_table(hires,256,256,256,ps[a],wm[b]);
        f32 st=find_step(hires,256,256,256,MODE_LEARN,tgt,rec,scratch);
        size_t by=pass(hires,256,256,256,MODE_LEARN,st,rec,scratch);
        row_t r; measure(hires,rec,256,256,256,by,&r);
        printf("%+5.2f  %.2f  | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f\n",
               (double)ps[a],(double)wm[b],r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae);
    }
    free(rec);free(scratch);
}

int main(int argc,char**argv){
    (void)argc;(void)argv;
    size_t lh,lc;
    u8 *hires=read_file("harness/refbuild/hires_256.u8",&lh);
    u8 *coarse=read_file("harness/refbuild/coarse_256.u8",&lc);
    if(!hires||!coarse){fprintf(stderr,"missing data (run from repo root)\n");return 1;}
    const double targets[]={16.0/2,16.0,32.0,64.0,128.0}; // q-like ladder mapped to ratios
    // Map the requested q in {16,32,64,128} to target RATIOS empirically: we just
    // sweep a ratio ladder that brackets the operating points the codec hits at
    // those q (from prior results q16~10x, q32~20x, q64~50x, q128~100x-ish).
    const double tg[]={10.0,20.0,50.0,100.0};
    int nt=4;

    // Generic HF slope: the chosen default in qmatrix16.c is 0.60; the stack note
    // says slope ~0.5. Use the file default (0.60) as the generic baseline.
    qm_set_hf_slope(QM_HF_SLOPE_DEFAULT);

    // Learned-table knobs (baked once). p=allocation exponent, wmin=floor weight.
    f32 P=0.5f, WMIN=0.30f;

    printf("# Learned quant-table experiment — PHerc Paris 4\n");
    printf("generic HF slope=%.2f ; learned p=%.2f wmin=%.2f ; DC weight forced 1.0\n",
           (double)g_qm_hf_slope, (double)P,(double)WMIN);

    // ---- IN-DOMAIN: train on hires, test on hires ----
    learn_weight_table(hires,256,256,256,P,WMIN);
    run("[train hires / test HIRES]  in-domain",hires,256,256,256,tg,nt);
    // cross: same hires-trained table on coarse (overfitting check #1)
    run("[train hires / test COARSE] cross-domain (overfit check)",coarse,256,256,256,tg,nt);

    // ---- train on coarse, test both ----
    learn_weight_table(coarse,256,256,256,P,WMIN);
    run("[train coarse / test COARSE] in-domain",coarse,256,256,256,tg,nt);
    run("[train coarse / test HIRES]  cross-domain (overfit check)",hires,256,256,256,tg,nt);

    psweep(hires);

    free(hires); free(coarse);
    return 0;
}
