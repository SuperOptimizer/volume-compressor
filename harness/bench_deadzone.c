// EXPERIMENT #18 — dead-zone ZERO-BIN WIDTH + dequant RECONSTRUCTION-OFFSET sweep
// (PLAN §2 "Quantizer" row). THROWAWAY self-contained bench (does NOT touch
// codec.c / ratectrl / chunkmodel). It #includes the WON transform (integer
// DCT-16^3) and the WON 16^3 quant matrix (qmatrix16.c, HF-protecting, slope in
// the mandated 0.4-0.6 band) directly, links the WON entropy coder (RLGR) + the
// metric bundle, and sweeps two scalar knobs of the dead-zone scalar quantizer
// END TO END at EQUAL RATIO on real PHerc Paris 4 sub-volumes:
//
//   (a) DEAD-ZONE WIDTH  dz  — the zero-bin half-width as a multiple of step.
//       encode: lvl = floor(|c|/step + (1 - dz)) for |c| >= dz*step, else 0.
//       dz=0.5 == the CURRENT/baseline quantizer (qmatrix16.c qm_quant_block).
//       Wider dz (>0.5) widens the kill-zone around 0 -> more HF coeffs -> 0 ->
//       more ratio, at some quality cost. Narrower (<0.5) keeps more, less ratio.
//
//   (b) RECONSTRUCTION OFFSET  ro  — where inside each non-zero bin we dequant.
//       dequant: |rec| = (lvl - 1 + dz + ro) * step.
//       For a UNIFORM-in-bin source the optimum is the bin CENTRE
//       (ro = (1 - dz)/2, giving rec at the bin midpoint). DCT/DWT AC coefficients
//       are ~LAPLACIAN: density falls off across the bin, so the conditional mean
//       sits BELOW the bin centre -> the Laplacian-optimal reconstruction point is
//       a biased offset (the classic ~0.375-vs-0.5 "uniform-quantizer-with-
//       reconstruction-bias" that c2d/c3d used for +0.2-0.5 dB free). We sweep ro
//       to find that point empirically on real scroll coefficients.
//
//   BASELINE (the current codec quantizer) == dz=0.5, ro=0.5
//     (qm_quant_block: a>=0.5*step; lvl=(i32)(a*inv-0.5)+1; rec=lvl*step).
//     Verify: with dz=0.5, lvl = floor(|c|/step + 0.5) for |c|>=0.5*step, which
//     equals (i32)(a/step - 0.5)+1 for a>=0.5*step. rec=(lvl-1+0.5+0.5)*step=
//     lvl*step. So (dz=0.5, ro=0.5) reproduces qm_quant_block/qm_dequant_block
//     exactly — it is the baseline row.
//
// Both knobs are GLOBAL build/run constants (encoder+decoder agree), NO side
// info. The per-coefficient quant loops are unit-stride + branchless ->
// autovectorizable (PLAN §7). The HF matrix shape is unchanged; we only move the
// dead-zone width and the in-bin reconstruction point. Pure C23, libc/libm.
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

// --- the won per-coefficient 16^3 quant matrix (HF-protecting) ------------
#include "../src/quant/qmatrix16.c"

// --- the won entropy coder (RLGR) -----------------------------------------
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A    16u
#define AVOX 4096u

// Use the mandated fixed HF matrix (slope mid of 0.4-0.6).
#define HF_SLOPE 0.5f

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

// Generalized dead-zone quantize: zero-bin half-width = dz*step.
// lvl = floor(|c|/step + (1 - dz)) for |c| >= dz*step, else 0. Branchless.
static inline void dz_quant_block(i16 *restrict qb, const i16 *restrict coef,
                                  const f32 *restrict step, f32 dz){
    f32 bias = 1.0f - dz;
    for (u32 i = 0; i < QM_BLKN; ++i) {
        f32 c = (f32)coef[i];
        f32 a = c < 0.f ? -c : c;
        f32 inv = 1.0f / step[i];
        i32 m = (a >= dz * step[i]);                 // 0/1 mask
        i32 lvl = m * (i32)(a * inv + bias);
        qb[i] = (i16)(c < 0.f ? -lvl : lvl);
    }
}

// Generalized dequant with in-bin reconstruction offset ro:
// |rec| = (lvl - 1 + dz + ro) * step for lvl != 0. Branchless-ish.
static inline void dz_dequant_block(i16 *restrict coef, const i16 *restrict qb,
                                    const f32 *restrict step, f32 dz, f32 ro){
    f32 add = dz + ro - 1.0f;                         // (lvl + add) * step
    for (u32 i = 0; i < QM_BLKN; ++i) {
        i32 l = (i32)qb[i];
        i32 al = l < 0 ? -l : l;
        f32 r = ((f32)al + add) * step[i];            // al==0 -> negative? guard:
        i32 nz = (al != 0);
        f32 rr = nz ? r : 0.f;
        i32 v = (i32)lrintf(l < 0 ? -rr : rr);
        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
        coef[i] = (i16)v;
    }
}

static size_t code_atom(const u8 *src, const f32 *step, f32 dz, f32 ro,
                        u8 *rec, u8 *scratch){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc = (sum + (i32)(AVOX/2)) / (i32)AVOX;
    i16 *coef = (i16*)scratch;
    i16 *qb   = coef + AVOX;
    H_dct16_fwd(coef, src, dc);
    dz_quant_block(qb, coef, step, dz);
    u8 *tmp = (u8*)(qb + AVOX);
    size_t bytes = vc_rlgr_encode(tmp, AVOX*3, qb, AVOX);
    dz_dequant_block(coef, qb, step, dz, ro);
    H_dct16_inv(rec, coef, dc);
    return bytes + 2;
}

typedef struct { double psnr,ms_ssim,gmsd,haarpsi,edge_mae,seam,ratio,enc_mbs,dec_mbs; } row_t;

static size_t pass(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,f32 dz,f32 ro,
                   u8 *rec,u8 *scratch){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    qm_build_step(step,QM_HF,base_step,1.0f);
    u8 srcblk[AVOX], recblk[AVOX];
    size_t total=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        total += code_atom(srcblk,step,dz,ro,recblk,scratch);
        scatter_atom(rec,h,w,az,ay,ax,recblk);
    }
    return total;
}

// Time a decode-only pass: pre-encode all atoms once, then time the dequant +
// inverse-DCT over the whole volume (decode MB/s, independent of ro/dz choice
// of the recon math but measured for the chosen one).
static double time_decode(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,f32 dz,f32 ro,
                          u8 *rec,u8 *scratch){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    qm_build_step(step,QM_HF,base_step,1.0f);
    u8 srcblk[AVOX], recblk[AVOX];
    // encode once into a big buffer of qb levels + dc
    size_t natoms=(size_t)naz*nay*nax;
    i16 *qall = malloc(natoms*AVOX*sizeof(i16));
    i32 *dcall = malloc(natoms*sizeof(i32));
    size_t ai=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax,++ai){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX; dcall[ai]=dc;
        i16 *coef=(i16*)scratch; i16 *qb=coef+AVOX;
        H_dct16_fwd(coef,srcblk,dc);
        dz_quant_block(qb,coef,step,dz);
        memcpy(qall+ai*AVOX,qb,AVOX*sizeof(i16));
    }
    double t0=now_sec();
    ai=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax,++ai){
        i16 *coef=(i16*)scratch;
        dz_dequant_block(coef,qall+ai*AVOX,step,dz,ro);
        H_dct16_inv(recblk,coef,dcall[ai]);
        scatter_atom(rec,h,w,az,ay,ax,recblk);
    }
    double dt=now_sec()-t0;
    free(qall); free(dcall);
    return ((size_t)d*h*w)/1e6/dt;
}

static f32 find_step_for_ratio(const u8 *vol,u32 d,u32 h,u32 w,f32 dz,f32 ro,
                               double target,u8 *rec,u8 *scratch){
    size_t raw=(size_t)d*h*w;
    f32 lo=0.5f, hi=400.f;
    for(int it=0;it<26;++it){
        f32 mid=sqrtf(lo*hi);
        size_t by=pass(vol,d,h,w,mid,dz,ro,rec,scratch);
        double r=(double)raw/(double)by;
        if(r<target) lo=mid; else hi=mid;
        if(fabs(r-target)<target*0.005) return mid;
    }
    return sqrtf(lo*hi);
}

static void measure(const u8 *vol,const u8 *rec,u32 d,u32 h,u32 w,size_t bytes,
                    double enc_sec,row_t *out){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    out->psnr=m.psnr; out->ms_ssim=m.ms_ssim;
    out->gmsd=vc_gmsd(vol,rec,d,h,w);
    out->haarpsi=vc_haarpsi(vol,rec,d,h,w);
    out->edge_mae=vc_edge_mae(vol,rec,d,h,w);
    out->seam=vc_seam_step(vol,rec,d,h,w,16);
    out->ratio=(double)((size_t)d*h*w)/(double)bytes;
    out->enc_mbs = enc_sec>0 ? ((size_t)d*h*w)/1e6/enc_sec : 0;
}

// Sweep a grid of (dz, ro) at each target ratio. Hold ratio constant by
// re-bisecting base_step for EACH (dz,ro) cell, so improvements show up as
// quality deltas at equal ratio (the ratio-at-quality objective).
static const f32 g_dz[] = {0.40f,0.45f,0.50f,0.55f,0.60f,0.65f,0.70f,0.80f};
static const f32 g_ro[] = {0.30f,0.375f,0.40f,0.45f,0.50f};
#define NDZ (sizeof(g_dz)/sizeof(g_dz[0]))
#define NRO (sizeof(g_ro)/sizeof(g_ro[0]))

static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w;
    u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);
    qm_set_hf_slope(HF_SLOPE);
    printf("\n## %s  (%ux%ux%u, %.1f MB, atom=16^3, HF slope %.2f)\n",label,d,h,w,raw/1e6,HF_SLOPE);

    const double targets[]={16.0,32.0,64.0,128.0};
    for(int ti=0;ti<4;++ti){
        double tgt=targets[ti];
        printf("\n### target %.0fx  (q-knob proxy ~%.0f)\n",tgt,tgt);
        printf("  dz  |  ro   | ratio  |  PSNR | dPSNR | MS-SSIM |   GMSD  | HaarPSI | edgeMAE |  seam16\n");
        printf("------+-------+--------+-------+-------+---------+---------+---------+---------+--------\n");
        double base_psnr=0;
        for(unsigned i=0;i<NDZ;++i){
          for(unsigned j=0;j<NRO;++j){
            f32 dz=g_dz[i], ro=g_ro[j];
            f32 st=find_step_for_ratio(vol,d,h,w,dz,ro,tgt,rec,scratch);
            double t0=now_sec();
            size_t by=pass(vol,d,h,w,st,dz,ro,rec,scratch);
            double enc=now_sec()-t0;
            row_t r; measure(vol,rec,d,h,w,by,enc,&r);
            if(dz==0.50f && ro==0.50f) base_psnr=r.psnr;   // baseline row
            int isbase=(dz==0.50f && ro==0.50f);
            printf("%5.2f | %5.3f | %5.2fx | %5.2f | %+5.2f | %7.4f | %7.5f | %7.4f | %7.3f | %7.3f%s\n",
                   dz,ro,r.ratio,r.psnr, base_psnr>0?r.psnr-base_psnr:0.0,
                   r.ms_ssim,r.gmsd,r.haarpsi,r.edge_mae,r.seam, isbase?"  <-baseline":"");
          }
        }
    }
    free(rec); free(scratch);
}

// Focused: pick the winning (dz,ro), report enc+dec MB/s vs baseline at each q.
static void winner_speed(const char *label,const u8 *vol,u32 d,u32 h,u32 w,
                         f32 wdz,f32 wro){
    size_t raw=(size_t)d*h*w; u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);
    qm_set_hf_slope(HF_SLOPE);
    printf("\n## SPEED: %s — baseline(dz0.50,ro0.50) vs winner(dz%.2f,ro%.3f)\n",label,wdz,wro);
    printf("  q  | cfg      | ratio  |  PSNR | MS-SSIM |  GMSD  | enc MB/s | dec MB/s\n");
    printf("-----+----------+--------+-------+---------+--------+----------+---------\n");
    const double targets[]={16.0,32.0,64.0,128.0};
    struct { const char*n; f32 dz,ro; } cfg[2]={{"baseline",0.50f,0.50f},{"winner ",wdz,wro}};
    for(int ti=0;ti<4;++ti){
        for(int c=0;c<2;++c){
            f32 st=find_step_for_ratio(vol,d,h,w,cfg[c].dz,cfg[c].ro,targets[ti],rec,scratch);
            double t0=now_sec();
            size_t by=pass(vol,d,h,w,st,cfg[c].dz,cfg[c].ro,rec,scratch);
            double enc=now_sec()-t0; row_t r; measure(vol,rec,d,h,w,by,enc,&r);
            double decmbs=time_decode(vol,d,h,w,st,cfg[c].dz,cfg[c].ro,rec,scratch);
            printf("%4.0f | %-8s | %5.2fx | %5.2f | %7.4f | %6.5f | %8.0f | %8.0f\n",
                   targets[ti],cfg[c].n,r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.enc_mbs,decmbs);
        }
    }
    free(rec); free(scratch);
}

int main(int argc,char**argv){
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    // optional: --speed dz ro  -> only the speed table for the chosen winner
    if(argc>=4 && strcmp(argv[1],"--speed")==0){
        f32 wdz=atof(argv[2]), wro=atof(argv[3]);
        for(int i=0;i<2;++i){ size_t len; u8*v=read_file(files[i],&len);
            if(!v){fprintf(stderr,"missing %s\n",files[i]);continue;}
            winner_speed(labels[i],v,256,256,256,wdz,wro); free(v); }
        return 0;
    }
    for(int i=0;i<2;++i){ size_t len; u8 *v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s (run from repo root)\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
