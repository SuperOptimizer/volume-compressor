// EXPERIMENT #17 — inter-LOD (Laplacian-pyramid) prediction bake-off.
//
// THROWAWAY self-contained bench (does NOT touch codec.c). It #includes the WON
// transform (integer DCT-16^3) and the WON HF-protecting 16^3 quant matrix, links
// the WON entropy coder (RLGR) + the metric bundle, and answers ONE question:
//
//   We store the resolution pyramid 0/..N/ as INDEPENDENT members (each LOD coded
//   from scratch). LOD k+1 is a 2x downsample of LOD k, so the coarse level
//   predicts the fine one. Does coding LOD k as a RESIDUAL against the 2x-
//   upsampled DECODED LOD k+1 (Laplacian-pyramid style) buy total-pyramid ratio
//   at matched per-LOD quality — and does it keep each LOD independently
//   decodable?
//
// Pyramid construction (PLAN §3 mandate): every LOD is box-downsampled FROM THE
// ORIGINAL hires volume (never from our own lossy output -> no error
// accumulation). The residual PREDICTION, however, upsamples the DECODED coarse
// level (that is what a real decoder has).
//
//   LOD0 = orig 256^3, LOD1 = orig downsampled to 128^3, LOD2 = 64^3, LOD3 = 32^3.
//
// Two coders, BOTH driven to the SAME per-LOD base quant step (matched detail
// fidelity), and ALSO compared at matched per-LOD PSNR (bisected):
//   INDEP    : code each LOD from scratch (current archive behaviour).
//   RESIDUAL : code coarsest LOD from scratch; for each finer LOD, decode the
//              coarser one, trilinear-upsample it 2x, and code the (fine - pred)
//              residual with a SIGNED DCT-16 atom path.
//
// Atom = 16^3 (won transform + random-access unit). Per-atom DC mean removed
// (counted as 2 side bytes/atom). RLGR over the 4096 quantized levels.
//
// Pure C23, libc/libm. Hot loops unit-stride/branchless (autovectorizable, §7).
#include "../include/vc/types.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// --- the won transform, renamed into this TU; exposes the internal 1D kernels
//     dct16_fwd / idct16 (static inline) for the SIGNED residual path ----------
#undef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 16
#define vc_dct_int16_fwd  H_dct16_fwd
#define vc_dct_int16_inv  H_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv

// --- the won HF-protecting per-coefficient 16^3 quant matrix -----------------
#include "../src/quant/qmatrix16.c"

// --- the won entropy coder (RLGR) --------------------------------------------
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

// ---------------------------------------------------------------------------
// Pyramid: 2x box-downsample FROM THE ORIGINAL (no error accumulation).
// in is (d,h,w) -> out is (d/2,h/2,w/2). Both u8.
static void downsample2x(const u8 *in,u32 d,u32 h,u32 w,u8 *out){
    u32 od=d/2,oh=h/2,ow=w/2;
    for(u32 z=0;z<od;++z)for(u32 y=0;y<oh;++y)for(u32 x=0;x<ow;++x){
        u32 sz=z*2,sy=y*2,sx=x*2; u32 s=0;
        for(u32 dz=0;dz<2;++dz)for(u32 dy=0;dy<2;++dy)for(u32 dx=0;dx<2;++dx)
            s += in[((size_t)(sz+dz)*h + (sy+dy))*w + (sx+dx)];
        out[((size_t)z*oh + y)*ow + x] = (u8)((s+4)/8);
    }
}

// 2x trilinear upsample of the DECODED coarse level. cin (cd,ch,cw) -> out
// (2cd,2ch,2cw) i16 (kept as i16 so the residual stays exact, though values are
// 0..255). Half-pixel-aligned linear interp with edge clamp.
static void upsample2x(const u8 *cin,u32 cd,u32 ch,u32 cw,i16 *out){
    u32 od=cd*2,oh=ch*2,ow=cw*2;
    for(u32 z=0;z<od;++z){
        // source coord in coarse grid (half-pixel centers): (z+0.5)/2 - 0.5
        f32 fz=((f32)z+0.5f)*0.5f-0.5f; if(fz<0)fz=0; if(fz>cd-1)fz=cd-1;
        u32 z0=(u32)fz; u32 z1=z0+1<cd?z0+1:z0; f32 wz=fz-(f32)z0;
        for(u32 y=0;y<oh;++y){
            f32 fy=((f32)y+0.5f)*0.5f-0.5f; if(fy<0)fy=0; if(fy>ch-1)fy=ch-1;
            u32 y0=(u32)fy; u32 y1=y0+1<ch?y0+1:y0; f32 wy=fy-(f32)y0;
            for(u32 x=0;x<ow;++x){
                f32 fx=((f32)x+0.5f)*0.5f-0.5f; if(fx<0)fx=0; if(fx>cw-1)fx=cw-1;
                u32 x0=(u32)fx; u32 x1=x0+1<cw?x0+1:x0; f32 wx=fx-(f32)x0;
                #define C(zz,yy,xx) ((f32)cin[((size_t)(zz)*ch+(yy))*cw+(xx)])
                f32 c00=C(z0,y0,x0)*(1-wx)+C(z0,y0,x1)*wx;
                f32 c01=C(z0,y1,x0)*(1-wx)+C(z0,y1,x1)*wx;
                f32 c10=C(z1,y0,x0)*(1-wx)+C(z1,y0,x1)*wx;
                f32 c11=C(z1,y1,x0)*(1-wx)+C(z1,y1,x1)*wx;
                #undef C
                f32 c0=c00*(1-wy)+c01*wy, c1=c10*(1-wy)+c11*wy;
                f32 v=c0*(1-wz)+c1*wz;
                out[((size_t)z*oh+y)*ow+x]=(i16)lrintf(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SIGNED DCT-16 atom: same int kernels, but reads i16 residual (no -dc bias,
// no u8 clamp) and writes i16 reconstruction (no [0,255] clamp). Used for the
// residual path. coef/out are 16^3 contiguous.
static void sdct16_fwd(i16 *restrict coef,const i16 *restrict res){
    i32 blk[16][16][16],tx[16][16][16],ty[16][16][16];
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x)
        blk[z][y][x]=(i32)res[((size_t)z*A+y)*A+x];
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y) dct16_fwd(blk[z][y],tx[z][y]);
    for(u32 z=0;z<A;++z)for(u32 x=0;x<A;++x){
        i32 col[16],oc[16]; for(u32 y=0;y<A;++y)col[y]=tx[z][y][x];
        dct16_fwd(col,oc); for(u32 y=0;y<A;++y)ty[z][y][x]=oc[y]; }
    for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        i32 col[16],oc[16]; for(u32 z=0;z<A;++z)col[z]=ty[z][y][x];
        dct16_fwd(col,oc);
        for(u32 z=0;z<A;++z){ i32 v=oc[z]; if(v>32767)v=32767; else if(v<-32768)v=-32768;
            coef[((size_t)z*A+y)*A+x]=(i16)v; } }
}
static void sdct16_inv(i16 *restrict res,const i16 *restrict coef){
    i32 blk[16][16][16],tz[16][16][16],ty[16][16][16];
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x)
        blk[z][y][x]=(i32)coef[((size_t)z*A+y)*A+x];
    for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        i32 col[16],oc[16]; for(u32 z=0;z<A;++z)col[z]=blk[z][y][x];
        idct16(col,oc); for(u32 z=0;z<A;++z)tz[z][y][x]=oc[z]; }
    for(u32 z=0;z<A;++z)for(u32 x=0;x<A;++x){
        i32 col[16],oc[16]; for(u32 y=0;y<A;++y)col[y]=tz[z][y][x];
        idct16(col,oc); for(u32 y=0;y<A;++y)ty[z][y][x]=oc[y]; }
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        i32 row[16],oc[16]; for(u32 x=0;x<A;++x)row[x]=ty[z][y][x];
        idct16(row,oc);
        for(u32 x=0;x<A;++x){ i32 v=oc[x]; if(v>32767)v=32767; else if(v<-32768)v=-32768;
            res[((size_t)z*A+y)*A+x]=(i16)v; } }
}

// ---------------------------------------------------------------------------
// Gather/scatter a 16^3 atom.
static inline void gather_u8(const u8 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,u8 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const u8 *src=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(blk+((size_t)z*A+y)*A,src,A); }
}
static inline void scatter_u8(u8 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,const u8 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        u8 *dst=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(dst,blk+((size_t)z*A+y)*A,A); }
}
static inline void gather_i16(const i16 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,i16 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        const i16 *src=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(blk+((size_t)z*A+y)*A,src,A*sizeof(i16)); }
}
static inline void scatter_i16(i16 *vol,u32 h,u32 w,u32 az,u32 ay,u32 ax,const i16 *blk){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
        i16 *dst=vol+((size_t)(az*A+z)*h+(ay*A+y))*w+ax*A;
        memcpy(dst,blk+((size_t)z*A+y)*A,A*sizeof(i16)); }
}

// ---------------------------------------------------------------------------
// INDEP: code one u8 LOD from scratch (DCT-16 + HF matrix + RLGR). Reconstructs
// into rec, returns total payload bytes. step matrix built once (QM_HF).
static size_t code_lod_indep(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,u8 *rec,u8 *scratch){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX]; qm_build_step(step,QM_HF,base_step,1.0f);
    u8 srcblk[AVOX],recblk[AVOX];
    size_t total=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_u8(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i)sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
        i16 *coef=(i16*)scratch; i16 *qb=coef+AVOX;
        H_dct16_fwd(coef,srcblk,dc);
        qm_quant_block(qb,coef,step);
        u8 *tmp=(u8*)(qb+AVOX);
        size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
        qm_dequant_block(coef,qb,step);
        H_dct16_inv(recblk,coef,dc);
        scatter_u8(rec,h,w,az,ay,ax,recblk);
        total += bytes + 2;   // +2 for stored DC (i16)
    }
    return total;
}

// RESIDUAL: code one u8 LOD as residual vs an upsampled-decoded prediction
// `pred` (i16, same shape). Signed DCT-16 + HF matrix + RLGR. Reconstructs the
// absolute u8 LOD into rec (= clamp(pred + decoded_residual)). Returns bytes.
static size_t code_lod_residual(const u8 *vol,const i16 *pred,u32 d,u32 h,u32 w,
                                f32 base_step,u8 *rec,u8 *scratch){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX]; qm_build_step(step,QM_HF,base_step,1.0f);
    i16 resblk[AVOX],predblk[AVOX],recres[AVOX]; u8 srcblk[AVOX],recblk[AVOX];
    // residual whole-volume buffers are avoided; we work per-atom. pred is i16 vol.
    size_t total=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_u8(vol,h,w,az,ay,ax,srcblk);
        gather_i16(pred,h,w,az,ay,ax,predblk);
        for(u32 i=0;i<AVOX;++i) resblk[i]=(i16)((i32)srcblk[i]-(i32)predblk[i]);
        // per-atom residual DC (mean), removed before DCT, stored as side level.
        i32 sum=0; for(u32 i=0;i<AVOX;++i)sum+=resblk[i];
        i32 dc=(i32)lrintf((f32)sum/(f32)AVOX);
        i16 zeromean[AVOX]; for(u32 i=0;i<AVOX;++i) zeromean[i]=(i16)((i32)resblk[i]-dc);
        i16 *coef=(i16*)scratch; i16 *qb=coef+AVOX;
        sdct16_fwd(coef,zeromean);
        qm_quant_block(qb,coef,step);
        u8 *tmp=(u8*)(qb+AVOX);
        size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
        qm_dequant_block(coef,qb,step);
        sdct16_inv(recres,coef);
        for(u32 i=0;i<AVOX;++i){
            i32 v=(i32)predblk[i]+(i32)recres[i]+dc;
            v=v<0?0:(v>255?255:v); recblk[i]=(u8)v;
        }
        scatter_u8(rec,h,w,az,ay,ax,recblk);
        total += bytes + 2;   // +2 for stored residual DC
    }
    return total;
}

// Bisect base step to hit a target ratio for INDEP coding of a LOD.
static f32 step_for_ratio_indep(const u8 *vol,u32 d,u32 h,u32 w,double target,u8 *rec,u8 *scratch){
    size_t raw=(size_t)d*h*w; f32 lo=0.3f,hi=600.f;
    for(int it=0;it<26;++it){ f32 mid=sqrtf(lo*hi);
        size_t by=code_lod_indep(vol,d,h,w,mid,rec,scratch);
        double r=(double)raw/(double)by;
        if(r<target)lo=mid; else hi=mid;
        if(fabs(r-target)<target*0.01) return mid; }
    return sqrtf(lo*hi);
}
// Bisect base step so RESIDUAL coding of a LOD hits a target PSNR (matched quality).
static f32 step_for_psnr_residual(const u8 *vol,const i16 *pred,u32 d,u32 h,u32 w,
                                  double target_psnr,u8 *rec,u8 *scratch){
    f32 lo=0.3f,hi=600.f;
    for(int it=0;it<26;++it){ f32 mid=sqrtf(lo*hi);
        code_lod_residual(vol,pred,d,h,w,mid,rec,scratch);
        vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
        // higher step -> lower psnr
        if(m.psnr>target_psnr) lo=mid; else hi=mid;
        if(fabs(m.psnr-target_psnr)<0.05) return mid; }
    return sqrtf(lo*hi);
}
static f32 step_for_psnr_indep(const u8 *vol,u32 d,u32 h,u32 w,double target_psnr,u8 *rec,u8 *scratch){
    f32 lo=0.3f,hi=600.f;
    for(int it=0;it<26;++it){ f32 mid=sqrtf(lo*hi);
        code_lod_indep(vol,d,h,w,mid,rec,scratch);
        vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
        if(m.psnr>target_psnr) lo=mid; else hi=mid;
        if(fabs(m.psnr-target_psnr)<0.05) return mid; }
    return sqrtf(lo*hi);
}

#define NLOD 4
static const char *qlabel="";

static void run(const char *label,const u8 *vol0){
    // Build the pyramid from the ORIGINAL (box-downsample, never lossy output).
    u32 dims[NLOD]={256,128,64,32};
    u8 *lod[NLOD]; lod[0]=(u8*)vol0;
    for(int k=1;k<NLOD;++k){
        size_t n=(size_t)dims[k]*dims[k]*dims[k];
        lod[k]=malloc(n);
        downsample2x(lod[k-1],dims[k-1],dims[k-1],dims[k-1],lod[k]);
    }
    size_t maxraw=(size_t)256*256*256;
    u8 *rec=malloc(maxraw), *rec_coarse=malloc(maxraw);
    i16 *pred=malloc(maxraw*sizeof(i16));
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);

    printf("\n## %s  (pyramid LOD0=256^3 .. LOD3=32^3, atom=16^3, HF qmatrix slope=%.2f)\n",
           label, QM_HF_SLOPE_DEFAULT);

    // The base-step "q" knobs requested: q in {16,32,64,128}. We use these as the
    // dead-zone base step directly (same knob the codec's rate-control maps q->step
    // near-linearly). At each q, code every LOD INDEP and RESIDUAL at the SAME
    // step, and report total-pyramid ratio + per-LOD PSNR.
    const f32 qs[]={16.f,32.f,64.f,128.f};
    for(int qi=0;qi<4;++qi){
        f32 q=qs[qi];
        printf("\n### q=%.0f (base dead-zone step), matched-step per LOD\n",q);
        printf("LOD  dim    | INDEP bytes  PSNR  | RESID bytes  PSNR  | byte delta\n");
        printf("------------+--------------------+--------------------+-----------\n");
        size_t tot_indep=0, tot_resid=0;
        // coarsest LOD: both schemes code from scratch (identical).
        for(int k=NLOD-1;k>=0;--k){
            u32 d=dims[k];
            size_t bi=code_lod_indep(lod[k],d,d,d,q,rec,scratch);
            vc_metrics mi; vc_compute_metrics(lod[k],rec,d,d,d,&mi);
            size_t br; vc_metrics mr;
            if(k==NLOD-1){
                // coarsest: residual scheme == indep (no coarser to predict from)
                br=bi; mr=mi;
                memcpy(rec_coarse,rec,(size_t)d*d*d); // decoded coarsest for next finer
            } else {
                // predict from the DECODED coarser LOD (rec_coarse holds LOD k+1 decode)
                u32 cd=dims[k+1];
                upsample2x(rec_coarse,cd,cd,cd,pred);
                br=code_lod_residual(lod[k],pred,d,d,d,q,rec,scratch);
                vc_compute_metrics(lod[k],rec,d,d,d,&mr);
                memcpy(rec_coarse,rec,(size_t)d*d*d); // this LOD's decode feeds the next
            }
            tot_indep+=bi; tot_resid+=br;
            printf("LOD%d %3u^3 | %9zu %6.2f | %9zu %6.2f | %+8zd\n",
                   k,d,bi,mi.psnr,br,mr.psnr,(ssize_t)br-(ssize_t)bi);
        }
        size_t pyr_raw=0; for(int k=0;k<NLOD;++k){u32 d=dims[k]; pyr_raw+=(size_t)d*d*d;}
        printf("------------+--------------------+--------------------+-----------\n");
        printf("PYRAMID     | %9zu %6.3fx| %9zu %6.3fx|  resid/indep = %.3f\n",
               tot_indep,(double)pyr_raw/tot_indep,
               tot_resid,(double)pyr_raw/tot_resid,
               (double)tot_resid/(double)tot_indep);
    }

    // --- matched-PSNR comparison: drive INDEP to a target ratio per LOD, then
    //     match RESIDUAL to the SAME PSNR per LOD, and compare bytes. This is the
    //     fair "ratio gain at matched quality" the experiment asks for. -------
    printf("\n### matched-PSNR (INDEP target ratio -> RESID bisected to same PSNR)\n");
    const double tgts[]={20.0,50.0};
    for(int ti=0;ti<2;++ti){
        double tgt=tgts[ti];
        printf("\n  -- INDEP coarsest@%.0fx, finer LODs matched to indep-PSNR --\n",tgt);
        printf("  LOD  dim    | PSNR target | INDEP bytes | RESID bytes | gain\n");
        printf("  -----------+-------------+-------------+-------------+------\n");
        size_t tot_indep=0,tot_resid=0;
        for(int k=NLOD-1;k>=0;--k){
            u32 d=dims[k];
            // INDEP: hit the target ratio at this LOD (gives us a reference PSNR).
            f32 si=step_for_ratio_indep(lod[k],d,d,d,tgt,rec,scratch);
            size_t bi=code_lod_indep(lod[k],d,d,d,si,rec,scratch);
            vc_metrics mi; vc_compute_metrics(lod[k],rec,d,d,d,&mi);
            double target_psnr=mi.psnr;
            size_t br;
            if(k==NLOD-1){
                br=bi; memcpy(rec_coarse,rec,(size_t)d*d*d);
            } else {
                u32 cd=dims[k+1]; upsample2x(rec_coarse,cd,cd,cd,pred);
                f32 sr=step_for_psnr_residual(lod[k],pred,d,d,d,target_psnr,rec,scratch);
                br=code_lod_residual(lod[k],pred,d,d,d,sr,rec,scratch);
                memcpy(rec_coarse,rec,(size_t)d*d*d);
            }
            tot_indep+=bi; tot_resid+=br;
            printf("  LOD%d %3u^3 | %10.2f  | %10zu  | %10zu  | %.3f\n",
                   k,d,target_psnr,bi,br,(double)br/(double)bi);
        }
        size_t pyr_raw=0; for(int k=0;k<NLOD;++k){u32 d=dims[k]; pyr_raw+=(size_t)d*d*d;}
        printf("  -----------+-------------+-------------+-------------+------\n");
        printf("  PYRAMID    | (matched)   | %10zu  | %10zu  | resid/indep=%.3f (ratio %.2fx -> %.2fx)\n",
               tot_indep,tot_resid,(double)tot_resid/tot_indep,
               (double)pyr_raw/tot_indep,(double)pyr_raw/tot_resid);
    }

    // --- speed: full-pyramid encode+reconstruct time, indep vs residual ----
    {
        size_t pyr_raw=0; for(int k=0;k<NLOD;++k){u32 d=dims[k]; pyr_raw+=(size_t)d*d*d;}
        // INDEP timing
        double t0=now_sec();
        for(int k=NLOD-1;k>=0;--k){u32 d=dims[k]; code_lod_indep(lod[k],d,d,d,32.f,rec,scratch);}
        double t_ind=now_sec()-t0;
        // RESIDUAL timing (with the cascade: re-decode coarse to predict)
        t0=now_sec();
        for(int k=NLOD-1;k>=0;--k){u32 d=dims[k];
            if(k==NLOD-1){ code_lod_indep(lod[k],d,d,d,32.f,rec,scratch); memcpy(rec_coarse,rec,(size_t)d*d*d); }
            else { u32 cd=dims[k+1]; upsample2x(rec_coarse,cd,cd,cd,pred);
                   code_lod_residual(lod[k],pred,d,d,d,32.f,rec,scratch); memcpy(rec_coarse,rec,(size_t)d*d*d); } }
        double t_res=now_sec()-t0;
        printf("\n  full-pyramid enc+reconstruct @q32: INDEP %.0f MB/s | RESIDUAL %.0f MB/s (%.2fx slower)\n",
               pyr_raw/1e6/t_ind, pyr_raw/1e6/t_res, t_res/t_ind);
    }

    // --- random-access / independent-decodability cost ---------------------
    // To touch ONE 16^3 atom of LOD0 in the RESIDUAL pyramid, a decoder must also
    // decode the predicting region of LOD1, which needs LOD2, etc. Because the
    // upsample is 2x trilinear (3-tap), one fine atom depends on a 8^3 region of
    // the coarse level = ONE coarse 16^3 atom (8^3 < 16^3, aligned), which in turn
    // depends on one atom of the next-coarser level. So the cascade is exactly ONE
    // atom per level: touched LOD0 atom => +1 LOD1 atom +1 LOD2 atom +1 LOD3 atom.
    {
        printf("\n  random-access (touched=1) atom-decode cost:\n");
        printf("    INDEP   : 1 atom decode (LOD0 self-contained).\n");
        printf("    RESIDUAL: 1 + 1 + 1 + 1 = %d atom decodes (cascade LOD0->1->2->3).\n", NLOD);
        printf("    => residual keeps each atom decodable but NOT independently per-LOD;\n");
        printf("       cost = %dx atom decodes at touched=1 (still O(1), bounded by pyramid depth).\n", NLOD);
    }

    for(int k=1;k<NLOD;++k) free(lod[k]);
    free(rec); free(rec_coarse); free(pred); free(scratch);
}

int main(int argc,char**argv){
    (void)qlabel;
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    if(argc>=2){ files[0]=argv[1]; files[1]=NULL; }
    for(int i=0;i<2;++i){ if(!files[i])break; size_t len; u8 *v=read_file(files[i],&len);
        if(!v||len<256u*256u*256u){fprintf(stderr,"missing %s (run from repo root)\n",files[i]);continue;}
        run(labels[i],v); free(v); }
    return 0;
}
