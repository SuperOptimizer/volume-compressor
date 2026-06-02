// HF / perceptual distortion-path bake-off (PLAN §2 "Distortion metric" row).
// THROWAWAY self-contained bench (does NOT touch codec.c / ratectrl / chunkmodel):
// it #includes the WON transform (integer DCT-16^3) and the new per-coefficient
// 16^3 quant matrix directly, links the WON entropy coder (RLGR) + the metric
// bundle, and runs the three distortion/quant OBJECTIVES end to end at EQUAL
// RATIO on real PHerc Paris 4 sub-volumes:
//
//   (a) FLAT  — pure-MSE flat quant. The baseline.
//   (b) HF    — HF-protecting per-coefficient quant matrix (edge/ink protection).
//   (b') ADAPT— HF matrix + per-16^3 content-adaptive step scale (1 side byte/atom).
//   (c) PERC  — true perceptual-in-loop: per-atom step chosen to minimize a
//               perceptual cost (GMSD/edge-MAE) on the DECODED block. Measures
//               the SPEED COST (needs the inverse transform per candidate).
//
// All four are driven to the SAME global target ratio by bisecting the base step
// (for ADAPT/PERC the per-block modulation rides on top of the global step). At
// the matched ratio we report PSNR, MS-SSIM, GMSD, HaarPSI, edge-MAE, 16-grid
// seam-step, and encode MB/s. Pure C23, libc/libm.
//
// Pipeline atom = 16^3 (the won transform + access unit). DC = per-atom mean,
// removed before DCT and stored as an extra level prepended to the atom's RLGR
// stream (so the rate counts it). Edges: volume dims are multiples of 16 here
// (256^3), no padding needed.
#include "../include/vc/types.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- the won transform, renamed into this TU ------------------------------
#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16          // dct_int16.c tiles a CHUNK into 16^3; here chunk==atom
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

#define A   16u                            // atom edge
#define AVOX 4096u

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}

// Gather a 16^3 atom from the volume at atom coord (az,ay,ax).
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

// Encode-and-reconstruct one atom under a given quant matrix `step[4096]`.
// Returns RLGR payload bytes for the atom; writes the reconstructed u8 atom to
// `rec`. The DC mean is prepended as level qscan[0..]? -> we instead remove the
// per-atom mean, DCT, quant; store the mean as a side level (counted as 2 bytes).
static size_t code_atom(const u8 *src, const f32 *step, u8 *rec, u8 *scratch){
    // per-atom DC bias = rounded mean
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc = (sum + (i32)(AVOX/2)) / (i32)AVOX;
    i16 *coef = (i16*)scratch;                 // 4096 i16
    i16 *qb   = coef + AVOX;                    // 4096 i16
    H_dct16_fwd(coef, src, dc);
    qm_quant_block(qb, coef, step);
    // entropy size (RLGR) of the 4096 levels
    u8 *tmp = (u8*)(qb + AVOX);                 // cap 4096*3
    size_t bytes = vc_rlgr_encode(tmp, AVOX*3, qb, AVOX);
    // reconstruct
    qm_dequant_block(coef, qb, step);
    H_dct16_inv(rec, coef, dc);
    return bytes + 2;                           // +2 bytes for the stored DC (i16)
}

typedef struct { double psnr,ms_ssim,gmsd,haarpsi,edge_mae,seam; double ratio,enc_mbs; } row_t;

// Run one global-step pass for FLAT/HF/ADAPT over the whole volume; returns
// total payload bytes and fills the reconstruction. mode picks the matrix.
static size_t pass_matrix(const u8 *vol,u32 d,u32 h,u32 w,vc_qm_mode mode,
                          f32 base_step,u8 *rec,u8 *scratch){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    if(mode!=QM_ADAPT) qm_build_step(step,mode,base_step,1.0f);
    u8 srcblk[AVOX], recblk[AVOX];
    size_t total=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,d,h,w,az,ay,ax,srcblk);
        if(mode==QM_ADAPT){
            // content-adaptive: per-block step scale from source variance; cost = 1 side byte.
            u32 idx = qm_adapt_index(vol,w,(u32)((size_t)w*h),az*A,ay*A,ax*A);
            f32 sc = qm_adapt_scale_from_index(idx);
            qm_build_step(step,QM_HF,base_step,sc);
            total += 1;                          // the side byte
        }
        total += code_atom(srcblk,step,recblk,scratch);
        scatter_atom(rec,d,h,w,az,ay,ax,recblk);
    }
    return total;
}

// (c) perceptual-in-loop: per-atom, try a small ladder of step scales around the
// global step and pick the scale minimizing a PERCEPTUAL cost on the DECODED
// block, subject to a soft per-atom rate ceiling (so it can't just spend bits
// everywhere). The chosen scale index is a side byte (counted). Uses the HF
// matrix as the base shape (so we measure perceptual-search ON TOP of the best
// fixed objective). The perceptual cost is local GMSD + edge-MAE on the 16^3
// reconstructed block vs source (cheap 2.5D over the atom).
#define PERC_NS 7
static const f32 perc_scales[PERC_NS] = {0.5f,0.7f,0.85f,1.0f,1.25f,1.6f,2.1f};

static double local_perc_cost(const u8 *src,const u8 *rec){
    // GMSD + 0.25*edge-MAE on the 16^3 block (one atom, all 3 axes via metric fns)
    double g = vc_gmsd(src,rec,A,A,A);
    double e = vc_edge_mae(src,rec,A,A,A);
    return g + 0.02*e;                          // weights: GMSD dominates, edge-MAE nudge
}

static size_t pass_perc(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,
                        double rate_ceiling_bytes,u8 *rec,u8 *scratch,double *enc_sec){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    u8 srcblk[AVOX], recblk[AVOX], bestrec[AVOX];
    size_t total=0;
    double t0=now_sec();
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,d,h,w,az,ay,ax,srcblk);
        double bestcost=1e30; size_t bestbytes=0; int chose=0;
        for(int si=0;si<PERC_NS;++si){
            qm_build_step(step,QM_HF,base_step,perc_scales[si]);
            size_t b = code_atom(srcblk,step,recblk,scratch);
            if((double)b > rate_ceiling_bytes && si>0) continue;   // rate cap (always allow coarsest-first)
            double c = local_perc_cost(srcblk,recblk);
            // tie-break: prefer fewer bytes when perceptual cost within 1%
            if(c < bestcost*0.99 || (c < bestcost*1.01 && b < bestbytes)){
                bestcost=c; bestbytes=b; chose=si; memcpy(bestrec,recblk,AVOX);
            }
        }
        if(!bestbytes){ // nothing passed the cap (all too big) -> take coarsest
            qm_build_step(step,QM_HF,base_step,perc_scales[PERC_NS-1]);
            bestbytes = code_atom(srcblk,step,bestrec,scratch); chose=PERC_NS-1;
        }
        (void)chose;
        total += bestbytes + 1;                  // +1 side byte for the chosen scale index
        scatter_atom(rec,d,h,w,az,ay,ax,bestrec);
    }
    *enc_sec = now_sec()-t0;
    return total;
}

// Bisect the global base step so a matrix-mode pass hits target ratio.
static f32 find_step_for_ratio(const u8 *vol,u32 d,u32 h,u32 w,vc_qm_mode mode,
                               double target,u8 *rec,u8 *scratch){
    size_t raw=(size_t)d*h*w;
    f32 lo=0.5f, hi=400.f;
    for(int it=0;it<24;++it){
        f32 mid=sqrtf(lo*hi);
        size_t by=pass_matrix(vol,d,h,w,mode,mid,rec,scratch);
        double r=(double)raw/(double)by;
        if(r<target) lo=mid; else hi=mid;
        if(fabs(r-target)<target*0.01) return mid;
    }
    return sqrtf(lo*hi);
}

static void measure(const u8 *vol,const u8 *rec,u32 d,u32 h,u32 w,size_t bytes,double enc_sec,row_t *out){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    out->psnr=m.psnr; out->ms_ssim=m.ms_ssim;
    out->gmsd=vc_gmsd(vol,rec,d,h,w);
    out->haarpsi=vc_haarpsi(vol,rec,d,h,w);
    out->edge_mae=vc_edge_mae(vol,rec,d,h,w);
    out->seam=vc_seam_step(vol,rec,d,h,w,16);
    out->ratio=(double)((size_t)d*h*w)/(double)bytes;
    out->enc_mbs = enc_sec>0 ? ((size_t)d*h*w)/1e6/enc_sec : 0;
}

static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w;
    u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);   // coef + qb + rlgr tmp
    printf("\n## %s  (%ux%ux%u, %.1f MB, atom=16^3)\n",label,d,h,w,raw/1e6);

    const double targets[]={10.0,20.0,50.0};
    for(int ti=0;ti<3;++ti){
        double tgt=targets[ti];
        printf("\n### target %.0fx\n",tgt);
        printf("objective         | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE |  seam16 | enc MB/s\n");
        printf("------------------+--------+-------+---------+---------+---------+---------+---------+---------\n");

        // (a) FLAT, (b) HF, (b') ADAPT — matrix passes, equal-ratio via bisection.
        struct { const char*name; vc_qm_mode mode; } mm[3]={
            {"(a) FLAT  pure-MSE",QM_FLAT},{"(b) HF-protecting ",QM_HF},{"(b') HF+adaptive  ",QM_ADAPT}};
        double base_hf=0; size_t bytes_hf=0;
        for(int k=0;k<3;++k){
            f32 st=find_step_for_ratio(vol,d,h,w,mm[k].mode,tgt,rec,scratch);
            double t0=now_sec();
            size_t by=pass_matrix(vol,d,h,w,mm[k].mode,st,rec,scratch);
            double enc=now_sec()-t0;
            row_t r; measure(vol,rec,d,h,w,by,enc,&r);
            printf("%-17s | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %7.3f | %7.0f\n",
                   mm[k].name,r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam,r.enc_mbs);
            if(mm[k].mode==QM_HF){ base_hf=st; bytes_hf=by; }
        }

        // (c) PERC — perceptual-in-loop on top of the HF base step. Rate ceiling
        // = the HF pass's mean per-atom bytes (so it matches ratio in aggregate).
        {
            u32 natoms=(d/A)*(h/A)*(w/A);
            double ceil_bytes = (double)bytes_hf/(double)natoms * 1.35; // soft, +35% headroom
            // bisect the perc base step to hit target (rate ceiling scales with it loosely)
            f32 lo=0.5f,hi=400.f,st=base_hf>0?base_hf:8.f; double enc=0; size_t by=0;
            for(int it=0;it<18;++it){
                f32 mid=sqrtf(lo*hi);
                double ceil2=ceil_bytes; // keep ceiling proportional handled by step
                by=pass_perc(vol,d,h,w,mid,ceil2,rec,scratch,&enc);
                double r=(double)raw/(double)by;
                if(r<tgt) lo=mid; else hi=mid;
                st=mid;
                if(fabs(r-tgt)<tgt*0.015) break;
            }
            by=pass_perc(vol,d,h,w,st,ceil_bytes,rec,scratch,&enc);
            row_t r; measure(vol,rec,d,h,w,by,enc,&r);
            printf("%-17s | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %7.3f | %7.0f\n",
                   "(c) PERC in-loop ",r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam,r.enc_mbs);
        }
    }
    free(rec); free(scratch);
}

// HF-slope sweep diagnostic: at a fixed target ratio, sweep the HF-protection
// strength and watch the perceptual metrics, to see if ANY slope beats flat.
static void hfsweep(const u8 *vol,u32 d,u32 h,u32 w,double tgt){
    size_t raw=(size_t)d*h*w; u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);
    printf("\n## HF-slope sweep @ %.0fx (hires) — does protecting HF help edges?\n",tgt);
    printf("hf_slope | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE |  seam16\n");
    printf("---------+--------+-------+---------+---------+---------+---------+--------\n");
    const f32 slopes[]={0.0f,0.2f,0.4f,0.6f,0.8f,0.9f};
    for(int s=0;s<6;++s){
        qm_set_hf_slope(slopes[s]);
        f32 st=find_step_for_ratio(vol,d,h,w,QM_HF,tgt,rec,scratch);
        size_t by=pass_matrix(vol,d,h,w,QM_HF,st,rec,scratch);
        row_t r; measure(vol,rec,d,h,w,by,1.0,&r);
        printf("  %.2f   | %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %7.3f\n",
               slopes[s],r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam);
    }
    qm_set_hf_slope(QM_HF_SLOPE_DEFAULT);
    free(rec);free(scratch);
}

int main(int argc,char**argv){
    if(argc>=2 && strcmp(argv[1],"--hfsweep")==0){
        size_t len; u8 *v=read_file("harness/refbuild/hires_256.u8",&len);
        if(!v){fprintf(stderr,"missing hires\n");return 1;}
        hfsweep(v,256,256,256,20.0); hfsweep(v,256,256,256,50.0);
        free(v); return 0;
    }
    if(argc>=5 && argv[1][0]!='-'){
        u32 d=atoi(argv[2]),h=atoi(argv[3]),w=atoi(argv[4]); size_t len;
        u8 *v=read_file(argv[1],&len);
        if(!v||len<(size_t)d*h*w){fprintf(stderr,"bad raw\n");return 1;}
        run(argv[1],v,d,h,w); free(v); return 0;
    }
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    for(int i=0;i<2;++i){ size_t len; u8 *v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s (run from repo root)\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
