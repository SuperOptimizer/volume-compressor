// Z-DIRECTIONAL ANISOTROPY bake-off (task r2-z-anisotropy). Self-contained
// THROWAWAY bench, same skeleton as bench_hfdist.c: integer DCT-16^3 atom +
// per-coefficient quant matrix + RLGR entropy, run end-to-end at EQUAL RATIO on
// real PHerc Paris 4 sub-volumes. Volume is ZYX, z = axis0 = outermost = scroll
// page top->bottom (verified most-correlated axis by harness/axis_corr.c).
//
// Two experiments, both vs the ISOTROPIC HF matrix baseline (the won stack):
//
//  (a) ANISOTROPIC QUANT MATRIX: instead of the isotropic HF weight
//      w(s)=1-slope*(uz+uy+ux)/45, use a per-axis weight
//        w = 1 - slope*( az*fz + ay*fy + ax*fx )/(az+ay+ax)/15
//      where fz,fy,fx are the 1D frequency indices (0..15) and az,ay,ax are
//      per-axis protection coefficients. SMALL a = protect that axis (finer step
//      at high freq of that axis). The isotropic baseline is az=ay=ax=1. We
//      sweep "Z-protect" configs that shrink az (protect Z) and grow ay,ax
//      (coarsen XY freqs). Cost: ZERO side info, just a different fixed matrix.
//
//  (b) Z-ONLY inter-atom DC prediction: predict an atom's DC (per-atom mean)
//      from the DC of its Z-neighbor atom ONLY (az-1). Encode the DC RESIDUAL.
//      (NOT symmetric 6/18/26-conn: those averaged in dead X/Y dirs and came
//      back ~0.) We also test predicting the low-freq Z-column coefficients
//      (the (fz,0,0) band) from the Z-neighbor. Measures bit savings on the DC
//      stream at equal recon (DC pred is lossless wrt the value coded).
//
// Pure C23, libc/libm. Atom = 16^3. DC = per-atom rounded mean, removed before
// DCT, stored (residual-coded in mode b).
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

// --- quant matrices --------------------------------------------------------
// Isotropic HF (baseline): w = 1 - slope*(fz+fy+fx)/45.
// Anisotropic: w = 1 - slope*(az*fz+ay*fy+ax*fx)/((az+ay+ax)*15).
// Both reduce to the SAME when az=ay=ax=1. Coef layout z*256+y*16+x.
static void build_step_iso(f32 *restrict step, f32 base, f32 slope){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        f32 t=(f32)(z+y+x)/45.0f;
        f32 wt=1.0f-slope*t;
        f32 s=base*wt; if(s<0.5f)s=0.5f;
        step[(size_t)z*256u+(size_t)y*16u+x]=s;
    }
}
static void build_step_aniso(f32 *restrict step, f32 base, f32 slope,
                             f32 az_,f32 ay_,f32 ax_){
    f32 denom=(az_+ay_+ax_)*15.0f;
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        f32 t=(az_*(f32)z+ay_*(f32)y+ax_*(f32)x)/denom;
        f32 wt=1.0f-slope*t;
        f32 s=base*wt; if(s<0.5f)s=0.5f;
        step[(size_t)z*256u+(size_t)y*16u+x]=s;
    }
}

static inline void quant_block(i16 *restrict qb,const i16 *restrict coef,const f32 *restrict step){
    for(u32 i=0;i<AVOX;++i){
        f32 c=(f32)coef[i]; f32 a=c<0.f?-c:c; f32 inv=1.0f/step[i];
        i32 m=(a>=0.5f*step[i]); i32 lvl=m*((i32)(a*inv-0.5f)+1);
        qb[i]=(i16)(c<0.f?-lvl:lvl);
    }
}
static inline void dequant_block(i16 *restrict coef,const i16 *restrict qb,const f32 *restrict step){
    for(u32 i=0;i<AVOX;++i){
        i32 l=(i32)qb[i]; i32 al=l<0?-l:l; f32 r=(f32)al*step[i];
        i32 v=(i32)lrintf(l<0?-r:r);
        if(v>32767)v=32767; else if(v<-32768)v=-32768; coef[i]=(i16)v;
    }
}

// Encode-and-reconstruct one atom. Returns RLGR payload bytes for the 4096
// levels (DC handled separately by caller). Writes recon u8.
static size_t code_atom(const u8 *src,const f32 *step,u8 *rec,u8 *scratch,i32 *dc_out){
    i32 sum=0; for(u32 i=0;i<AVOX;++i)sum+=src[i];
    i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
    i16 *coef=(i16*)scratch; i16 *qb=coef+AVOX;
    H_dct16_fwd(coef,src,dc);
    quant_block(qb,coef,step);
    u8 *tmp=(u8*)(qb+AVOX);
    size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    dequant_block(coef,qb,step);
    H_dct16_inv(rec,coef,dc);
    *dc_out=dc;
    return bytes;
}

typedef struct { double psnr,ms_ssim,gmsd,haarpsi,edge_mae,seam,ratio,enc_mbs; } row_t;

// One full pass. iso? uses isotropic HF; else anisotropic (az,ay,ax). DC stored
// as 2 bytes/atom (mode_dcpred==0) or Z-predicted residual coded compactly
// (mode_dcpred==1: residual vs Z-neighbor atom DC, varint-ish 1-2 bytes).
static size_t pass(const u8 *vol,u32 d,u32 h,u32 w,int iso,
                   f32 base,f32 slope,f32 az_,f32 ay_,f32 ax_,
                   int dcpred,u8 *rec,u8 *scratch,double *enc_sec,
                   size_t *dc_bytes_out){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    if(iso) build_step_iso(step,base,slope); else build_step_aniso(step,base,slope,az_,ay_,ax_);
    u8 srcblk[AVOX],recblk[AVOX];
    // store per-atom DC for Z-prediction (indexed [az][ay][ax])
    i32 *dcgrid=malloc((size_t)naz*nay*nax*sizeof(i32));
    size_t total=0, dcbytes=0;
    double t0=now_sec();
    // We must process in Z-major order so the Z-neighbor (az-1) DC is available.
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 dc;
        total += code_atom(srcblk,step,recblk,scratch,&dc);
        dcgrid[((size_t)az*nay+ay)*nax+ax]=dc;
        scatter_atom(rec,h,w,az,ay,ax,recblk);
        // DC coding cost
        if(dcpred==0){
            dcbytes += 2;                          // raw i16 DC
        } else {
            i32 pred = (az>0)? dcgrid[(((size_t)(az-1))*nay+ay)*nax+ax] : 0;
            i32 resid = dc - pred;
            i32 ar = resid<0?-resid:resid;
            dcbytes += (ar<64)?1:2;                // tiny residual 1 byte, else 2 (sign+mag varint)
        }
    }
    free(dcgrid);
    *enc_sec=now_sec()-t0;
    *dc_bytes_out=dcbytes;
    return total + dcbytes;
}

static void measure(const u8 *vol,const u8 *rec,u32 d,u32 h,u32 w,size_t bytes,double enc_sec,row_t *o){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    o->psnr=m.psnr; o->ms_ssim=m.ms_ssim;
    o->gmsd=vc_gmsd(vol,rec,d,h,w); o->haarpsi=vc_haarpsi(vol,rec,d,h,w);
    o->edge_mae=vc_edge_mae(vol,rec,d,h,w); o->seam=vc_seam_step(vol,rec,d,h,w,16);
    o->ratio=(double)((size_t)d*h*w)/(double)bytes;
    o->enc_mbs=enc_sec>0?((size_t)d*h*w)/1e6/enc_sec:0;
}

// Bisect base step to hit target ratio for a given config.
static f32 find_step(const u8 *vol,u32 d,u32 h,u32 w,int iso,f32 slope,
                     f32 az_,f32 ay_,f32 ax_,int dcpred,double tgt,u8 *rec,u8 *scratch){
    size_t raw=(size_t)d*h*w; f32 lo=0.5f,hi=400.f; double es; size_t db;
    for(int it=0;it<26;++it){ f32 mid=sqrtf(lo*hi);
        size_t by=pass(vol,d,h,w,iso,mid,slope,az_,ay_,ax_,dcpred,rec,scratch,&es,&db);
        double r=(double)raw/(double)by;
        if(r<tgt)lo=mid; else hi=mid;
        if(fabs(r-tgt)<tgt*0.01) return mid;
    }
    return sqrtf(lo*hi);
}

typedef struct { const char*name; int iso; f32 az,ay,ax; int dcpred; } cfg_t;

static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w;
    u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);
    const f32 SLOPE=0.5f;                          // task says ~0.5
    printf("\n## %s  (%ux%ux%u, %.1f MB, atom=16^3, hf_slope=%.2f)\n",label,d,h,w,raw/1e6,SLOPE);

    // configs: baseline isotropic HF (=az=ay=ax=1), then Z-protect anisotropic
    // sweeps (shrink Z weight => protect Z low/mid bands; grow Y,X => coarsen).
    // Recall axis_corr: Z most correlated (low energy at HF -> we can coarsen Z
    // HF cheaply? NO -- correlated means Z low-freq carries the signal, protect
    // the Z DIRECTION structure. We sweep both directions to find the truth.)
    cfg_t cfgs[]={
        {"iso HF (baseline)",   1, 1,1,1, 0},
        {"aniso Zprot .5/1/1",  0, 0.5f,1.0f,1.0f, 0},
        {"aniso Zprot .3/1/1",  0, 0.3f,1.0f,1.0f, 0},
        {"aniso Zprot .5/1.3/1.3",0,0.5f,1.3f,1.3f,0},
        {"aniso Zprot .3/1.5/1.5",0,0.3f,1.5f,1.5f,0},
        {"aniso Zcoarse 2/1/1", 0, 2.0f,1.0f,1.0f, 0},  // opposite: coarsen Z HF
        {"aniso Yonly-coarse 1/2/1",0,1.0f,2.0f,1.0f,0},// coarsen weak Y axis
        {"iso HF + Zpred DC",   1, 1,1,1, 1},           // (b) Z-only DC prediction
    };
    int nc=(int)(sizeof(cfgs)/sizeof(cfgs[0]));
    const double targets[]={ 8.0, 16.0, 32.0, 64.0 };  // q-ish via ratio anchors

    for(int ti=0;ti<4;++ti){
        double tgt=targets[ti];
        printf("\n### target %.0fx\n",tgt);
        printf("config              | ratio  |  PSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE |  seam16 | dcKB | enc MB/s\n");
        printf("--------------------+--------+-------+---------+---------+---------+---------+---------+------+---------\n");
        for(int c=0;c<nc;++c){
            f32 st=find_step(vol,d,h,w,cfgs[c].iso,SLOPE,cfgs[c].az,cfgs[c].ay,cfgs[c].ax,cfgs[c].dcpred,tgt,rec,scratch);
            double es; size_t db;
            size_t by=pass(vol,d,h,w,cfgs[c].iso,st,SLOPE,cfgs[c].az,cfgs[c].ay,cfgs[c].ax,cfgs[c].dcpred,rec,scratch,&es,&db);
            row_t r; measure(vol,rec,d,h,w,by,es,&r);
            printf("%-20s| %5.2fx | %5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %7.3f | %4.0f | %7.0f\n",
                   cfgs[c].name,r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam,db/1024.0,r.enc_mbs);
        }
    }
    free(rec); free(scratch);
}

int main(int argc,char**argv){
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    if(argc>=2){ files[0]=argv[1]; }
    for(int i=0;i<2;++i){ size_t len; u8 *v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
