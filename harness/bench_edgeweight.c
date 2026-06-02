// Edge/gradient importance-weight rate-allocation experiment (r3-edge-weight).
//
// LITERATURE: cheap precompute-once analog of Feature-Preserving RDO. ONE Sobel/
// gradient pass over the volume yields a per-16^3-block importance weight from
// local edge energy. The rate-control / quantization then ALLOCATES MORE BITS
// (finer dead-zone step) to high-gradient atoms (ink strokes, sheet/fiber
// boundaries) and coarser steps to flat regions, at a FIXED TOTAL RATIO. It
// COMPOSES with the existing HF-protecting per-coef quant matrix (qmatrix16,
// slope 0.6): the matrix protects HF WITHIN a block; the edge weight decides
// WHICH blocks get the bits.
//
// The codec atom is faithfully reproduced here: standalone 16^3 integer DCT (same
// Q14 matrix as transform/dct_int16.c and ratectrl/lagrangian.c) + qmatrix16
// dead-zone quant + RLGR rate (VC_ENTROPY_ENC). Per-block step is ALREADY a
// supported lever (the q-field), so touched=1 / 16^3 random access is preserved:
// each atom still decodes from its own step, independently.
//
// EXPERIMENT (all at EQUAL TOTAL BYTES, i.e. equal ratio):
//   UNIFORM  : every block the same base step (the current-stack default).
//   MSE-LAG  : per-block Lagrangian on plain MSE (the existing rate-control;
//              known to MATCH uniform PSNR, "bang-bang").
//   EDGEW    : per-block Lagrangian on edge-WEIGHTED distortion D' = w * D, where
//              w is the precomputed gradient importance. Protects edges.
//
// We reconstruct the WHOLE volume for each policy and report PSNR + the edge/
// perceptual proxies GMSD, edge-MAE, HaarPSI, seam-step. HONEST: these are
// proxies for ink fidelity; true ink-detection AUC needs a downstream harness we
// do not have.
//
// Usage:
//   bench_edgeweight --refbuild        # hires_256 + coarse_256
//   bench_edgeweight <raw.u8> dz dy dx
#include "../include/vc/vc.h"
#include "../src/metrics/metrics.h"
#include "../src/quant/qmatrix16.c"   // header-only inline quant primitives
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// RLGR entropy coder (rate proxy). Same signature the codec uses.
size_t VC_ENTROPY_ENC(u8 *restrict out, size_t cap, const i16 *restrict in, size_t n);

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}

// --- standalone 16-point orthonormal integer DCT (Q14), codec-matched ----------
#define RB   16u
#define RBV  4096u
#define RQ   14u
static const i32 C16[16][16] = {
  {  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096},
  {  5765,  5543,  5109,  4478,  3675,  2731,  1682,   568,  -568, -1682, -2731, -3675, -4478, -5109, -5543, -5765},
  {  5681,  4816,  3218,  1130, -1130, -3218, -4816, -5681, -5681, -4816, -3218, -1130,  1130,  3218,  4816,  5681},
  {  5543,  3675,   568, -2731, -5109, -5765, -4478, -1682,  1682,  4478,  5765,  5109,  2731,  -568, -3675, -5543},
  {  5352,  2217, -2217, -5352, -5352, -2217,  2217,  5352,  5352,  2217, -2217, -5352, -5352, -2217,  2217,  5352},
  {  5109,   568, -4478, -5543, -1682,  3675,  5765,  2731, -2731, -5765, -3675,  1682,  5543,  4478,  -568, -5109},
  {  4816, -1130, -5681, -3218,  3218,  5681,  1130, -4816, -4816,  1130,  5681,  3218, -3218, -5681, -1130,  4816},
  {  4478, -2731, -5543,   568,  5765,  1682, -5109, -3675,  3675,  5109, -1682, -5765,  -568,  5543,  2731, -4478},
  {  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096},
  {  3675, -5109, -1682,  5765,  -568, -5543,  2731,  4478, -4478, -2731,  5543,   568, -5765,  1682,  5109, -3675},
  {  3218, -5681,  1130,  4816, -4816, -1130,  5681, -3218, -3218,  5681, -1130, -4816,  4816,  1130, -5681,  3218},
  {  2731, -5765,  3675,  1682, -5543,  4478,   568, -5109,  5109,  -568, -4478,  5543, -1682, -3675,  5765, -2731},
  {  2217, -5352,  5352, -2217, -2217,  5352, -5352,  2217,  2217, -5352,  5352, -2217, -2217,  5352, -5352,  2217},
  {  1682, -4478,  5765, -5109,  2731,   568, -3675,  5543, -5543,  3675,  -568, -2731,  5109, -5765,  4478, -1682},
  {  1130, -3218,  4816, -5681,  5681, -4816,  3218, -1130, -1130,  3218, -4816,  5681, -5681,  4816, -3218,  1130},
  {   568, -1682,  2731, -3675,  4478, -5109,  5543, -5765,  5765, -5543,  5109, -4478,  3675, -2731,  1682,  -568},
};
static inline void dct1(const i32 in[16], i32 out[16]){
    const i32 rnd=(i32)1<<(RQ-1);
    for(u32 k=0;k<16;++k){ i32 a=0; for(u32 n=0;n<16;++n) a+=C16[k][n]*in[n]; out[k]=(a+rnd)>>RQ; }
}
static inline void idct1(const i32 in[16], i32 out[16]){
    const i32 rnd=(i32)1<<(RQ-1);
    for(u32 n=0;n<16;++n){ i32 a=0; for(u32 k=0;k<16;++k) a+=C16[k][n]*in[k]; out[n]=(a+rnd)>>RQ; }
}
static void blk_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc){
    static _Thread_local i32 a[16][16][16], b[16][16][16];
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x) a[z][y][x]=(i32)vox[z*256u+y*16u+x]-dc;
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y) dct1(a[z][y], b[z][y]);
    for(u32 z=0;z<16;++z)for(u32 x=0;x<16;++x){ i32 col[16],oc[16]; for(u32 y=0;y<16;++y)col[y]=b[z][y][x]; dct1(col,oc); for(u32 y=0;y<16;++y)a[z][y][x]=oc[y]; }
    for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x){ i32 col[16],oc[16]; for(u32 z=0;z<16;++z)col[z]=a[z][y][x]; dct1(col,oc);
        for(u32 z=0;z<16;++z){ i32 v=oc[z]; v=v>32767?32767:(v<-32768?-32768:v); coef[z*256u+y*16u+x]=(i16)v; } }
}
static void blk_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc){
    static _Thread_local i32 a[16][16][16], b[16][16][16];
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x) a[z][y][x]=(i32)coef[z*256u+y*16u+x];
    for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x){ i32 col[16],oc[16]; for(u32 z=0;z<16;++z)col[z]=a[z][y][x]; idct1(col,oc); for(u32 z=0;z<16;++z)b[z][y][x]=oc[z]; }
    for(u32 z=0;z<16;++z)for(u32 x=0;x<16;++x){ i32 col[16],oc[16]; for(u32 y=0;y<16;++y)col[y]=b[z][y][x]; idct1(col,oc); for(u32 y=0;y<16;++y)a[z][y][x]=oc[y]; }
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y){ i32 row[16],oc[16]; for(u32 x=0;x<16;++x)row[x]=a[z][y][x]; idct1(row,oc);
        for(u32 x=0;x<16;++x){ i32 v=oc[x]+dc; v=v<0?0:(v>255?255:v); vox[z*256u+y*16u+x]=(u8)v; } }
}

// --- per-block geometry over the whole volume ----------------------------------
typedef struct { u32 nbz,nby,nbx,nb; u32 d,h,w; } geom;
static geom make_geom(u32 d,u32 h,u32 w){
    geom g; g.d=d; g.h=h; g.w=w;
    g.nbz=(d+RB-1)/RB; g.nby=(h+RB-1)/RB; g.nbx=(w+RB-1)/RB;
    g.nb=g.nbz*g.nby*g.nbx; return g;
}
// gather a 16^3 block (zero-pad edges) into contiguous bvox, return dc.
static i32 gather_block(u8 *restrict bvox, const u8 *restrict vol, const geom *g,
                        u32 bz,u32 by,u32 bx){
    u64 sum=0;
    for(u32 z=0;z<RB;++z)for(u32 y=0;y<RB;++y)for(u32 x=0;x<RB;++x){
        u32 vz=bz*RB+z, vy=by*RB+y, vx=bx*RB+x;
        u8 v=(vz<g->d&&vy<g->h&&vx<g->w)?vol[((size_t)vz*g->h+vy)*g->w+vx]:0;
        bvox[z*256u+y*16u+x]=v; sum+=v;
    }
    return (i32)((sum+RBV/2)/RBV);
}
static void scatter_block(u8 *restrict vol, const geom *g, u32 bz,u32 by,u32 bx,
                          const u8 *restrict bvox){
    for(u32 z=0;z<RB;++z)for(u32 y=0;y<RB;++y)for(u32 x=0;x<RB;++x){
        u32 vz=bz*RB+z, vy=by*RB+y, vx=bx*RB+x;
        if(vz<g->d&&vy<g->h&&vx<g->w) vol[((size_t)vz*g->h+vy)*g->w+vx]=bvox[z*256u+y*16u+x];
    }
}

// --- ONE cheap precompute: per-block edge/gradient importance weight ------------
// Sobel-magnitude mean over the block (3D central differences). Normalized across
// the volume to a multiplicative importance in [1, WMAX]: flat blocks -> 1, the
// busiest edge blocks -> WMAX. Returns mean gradient per block in `gradmean`.
static void compute_edge_weights(const u8 *restrict vol, const geom *g,
                                 f64 *restrict gradmean, f64 wgamma, f64 wmax){
    const u32 d=g->d,h=g->h,w=g->w;
    // per-block mean gradient magnitude
    for(u32 bi=0;bi<g->nb;++bi) gradmean[bi]=0.0;
    u32 bx,by,bz;
    for(bz=0;bz<g->nbz;++bz)for(by=0;by<g->nby;++by)for(bx=0;bx<g->nbx;++bx){
        f64 acc=0.0; u32 cnt=0;
        for(u32 z=0;z<RB;++z)for(u32 y=0;y<RB;++y)for(u32 x=0;x<RB;++x){
            u32 vz=bz*RB+z, vy=by*RB+y, vx=bx*RB+x;
            if(vz==0||vy==0||vx==0||vz>=d-1||vy>=h-1||vx>=w-1) continue;
            const u8 *p=vol+((size_t)vz*h+vy)*w+vx;
            i32 gx=(i32)p[1]-(i32)p[-1];
            i32 gy=(i32)p[w]-(i32)p[-(i64)w];
            i32 gz=(i32)p[(size_t)h*w]-(i32)p[-(i64)h*w];
            acc += sqrt((f64)(gx*gx+gy*gy+gz*gz)); cnt++;
        }
        gradmean[bz*g->nby*g->nbx + by*g->nbx + bx] = cnt? acc/cnt : 0.0;
    }
    // normalize: weight = 1 + (wmax-1) * (g/gmax)^gamma  (monotone, smooth)
    f64 gmax=1e-9; for(u32 i=0;i<g->nb;++i) if(gradmean[i]>gmax) gmax=gradmean[i];
    (void)wgamma; (void)wmax; // weights applied later from gradmean
}
static inline f64 weight_of(f64 gradmean, f64 gmax, f64 wgamma, f64 wmax){
    f64 t = gmax>0 ? gradmean/gmax : 0.0; if(t>1.0)t=1.0;
    return 1.0 + (wmax-1.0)*pow(t, wgamma);
}

// --- per-block R-D curve over a dense step grid, codec-faithful -----------------
#define NSTEP 16
static const f32 STEP_GRID[NSTEP] = {1.f,1.5f,2.f,3.f,4.f,6.f,8.f,12.f,16.f,24.f,32.f,48.f,64.f,90.f,128.f,180.f};
typedef struct { f64 r[NSTEP], d[NSTEP]; } block_rd;  // bytes, MSE per step

// Build R-D for every block: for each step, qmatrix16 HF quant -> RLGR bytes,
// and true decode-MSE via inverse DCT. coef cached per block.
static void build_block_rd(const u8 *restrict vol, const geom *g,
                           block_rd *restrict rd){
    u8 *bvox=malloc(RBV); i16 *coef=malloc(RBV*2); i16 *qb=malloc(RBV*2);
    i16 *cbq=malloc(RBV*2); u8 *v0=malloc(RBV); u8 *v1=malloc(RBV);
    u8 *ebuf=malloc(RBV*4+64); f32 *step=malloc(RBV*sizeof(f32));
    u32 bi=0;
    for(u32 bz=0;bz<g->nbz;++bz)for(u32 by=0;by<g->nby;++by)for(u32 bx=0;bx<g->nbx;++bx,++bi){
        i32 dc=gather_block(bvox,vol,g,bz,by,bx);
        blk_fwd(coef,bvox,dc);
        blk_inv(v0,coef,128);   // unquantized recon reference (isolate quant error)
        for(int gi=0;gi<NSTEP;++gi){
            qm_build_step(step, QM_HF, STEP_GRID[gi], 1.0f);
            qm_quant_block(qb, coef, step);
            size_t bytes=VC_ENTROPY_ENC(ebuf, RBV*4+64, qb, RBV);
            // marginal content rate: subtract the shared-per-chunk RLGR header (~4B)
            f64 r=(f64)bytes-4.0; if(r<1.0)r=1.0;
            qm_dequant_block(cbq, qb, step);
            blk_inv(v1,cbq,128);
            f64 s=0; for(u32 i=0;i<RBV;++i){ f64 e=(f64)v1[i]-(f64)v0[i]; s+=e*e; }
            rd[bi].r[gi]=r; rd[bi].d[gi]=s/RBV;
        }
    }
    free(bvox);free(coef);free(qb);free(cbq);free(v0);free(v1);free(ebuf);free(step);
}

// Pick, per block, the step index minimizing D*w + lambda*R (edge-weighted
// Lagrangian). w==1 for all blocks reduces to plain MSE-Lagrangian.
static void pick_lambda(const block_rd *rd, const f64 *wt, u32 nb, f64 lambda,
                        int *idx, f64 *total_bytes){
    f64 tb=0;
    for(u32 b=0;b<nb;++b){
        int best=0; f64 bc=1e300;
        for(int gi=0;gi<NSTEP;++gi){
            f64 c = wt[b]*rd[b].d[gi] + lambda*rd[b].r[gi];
            if(c<bc){bc=c;best=gi;}
        }
        idx[b]=best; tb+=rd[b].r[best];
    }
    *total_bytes=tb;
}
// Bisect lambda so total bytes ~= target.
static f64 alloc_to_target(const block_rd *rd, const f64 *wt, u32 nb,
                           f64 target_bytes, int *idx){
    f64 lo=1e-9, hi=1e12, lam=1.0, tb=0;
    for(int it=0;it<80;++it){ lam=sqrt(lo*hi); pick_lambda(rd,wt,nb,lam,idx,&tb);
        if(tb>target_bytes) lo=lam; else hi=lam;
        if(fabs(tb-target_bytes)<target_bytes*0.004) break; }
    return tb;
}

// Reconstruct the whole volume given a per-block step index, codec-faithful.
static void reconstruct(const u8 *restrict vol, const geom *g, const int *idx,
                        u8 *restrict out){
    u8 *bvox=malloc(RBV); i16 *coef=malloc(RBV*2); i16 *qb=malloc(RBV*2);
    i16 *cbq=malloc(RBV*2); u8 *rec=malloc(RBV); f32 *step=malloc(RBV*sizeof(f32));
    u32 bi=0;
    for(u32 bz=0;bz<g->nbz;++bz)for(u32 by=0;by<g->nby;++by)for(u32 bx=0;bx<g->nbx;++bx,++bi){
        i32 dc=gather_block(bvox,vol,g,bz,by,bx);
        blk_fwd(coef,bvox,dc);
        qm_build_step(step, QM_HF, STEP_GRID[idx[bi]], 1.0f);
        qm_quant_block(qb, coef, step);
        qm_dequant_block(cbq, qb, step);
        blk_inv(rec,cbq,dc);
        scatter_block(out,g,bz,by,bx,rec);
    }
    free(bvox);free(coef);free(qb);free(cbq);free(rec);free(step);
}

typedef struct { f64 ratio,psnr,gmsd,edge,haar,seam; } scoreset;
static void score(const u8 *vol, const u8 *rec, const geom *g, f64 total_bytes, scoreset *s){
    size_t raw=(size_t)g->d*g->h*g->w;
    f64 m=0; for(size_t i=0;i<raw;++i){ f64 e=(f64)vol[i]-(f64)rec[i]; m+=e*e; } m/=raw;
    s->ratio = (f64)raw/total_bytes;
    s->psnr  = m<=0?99.0:10.0*log10(255.0*255.0/m);
    s->gmsd  = vc_gmsd(vol,rec,g->d,g->h,g->w);
    s->edge  = vc_edge_mae(vol,rec,g->d,g->h,g->w);
    s->haar  = vc_haarpsi(vol,rec,g->d,g->h,g->w);
    s->seam  = vc_seam_step(vol,rec,g->d,g->h,g->w,16);
}

static void run(const char *label, const u8 *vol, u32 d, u32 h, u32 w){
    geom g=make_geom(d,h,w);
    size_t raw=(size_t)d*h*w;
    printf("\n#### %s  (%ux%ux%u, %.1f MB, %u blocks of 16^3)\n", label,d,h,w,raw/1e6,g.nb);

    // ONE precompute: edge weights.
    f64 *gradmean=malloc(g.nb*sizeof(f64));
    double t0=now_sec(); compute_edge_weights(vol,&g,gradmean,1.0,1.0); double t1=now_sec();
    f64 gmax=1e-9; for(u32 i=0;i<g.nb;++i) if(gradmean[i]>gmax) gmax=gradmean[i];
    printf("edge-weight precompute: %.1f ms (one Sobel pass, %.3f ns/vox)\n",
           (t1-t0)*1e3, (t1-t0)*1e9/raw);

    // R-D curves (shared by all policies).
    block_rd *rd=malloc((size_t)g.nb*sizeof(block_rd));
    double tb0=now_sec(); build_block_rd(vol,&g,rd); double tb1=now_sec();
    printf("per-block R-D build: %.1f ms (%.1f blocks/ms)\n", (tb1-tb0)*1e3, g.nb/((tb1-tb0)*1e3));

    int *idx=malloc(g.nb*sizeof(int));
    int *idxu=malloc(g.nb*sizeof(int));
    u8 *rec=malloc(raw);
    f64 *w_flat=malloc(g.nb*sizeof(f64)); for(u32 i=0;i<g.nb;++i) w_flat[i]=1.0;

    // Edge-weight strengths to sweep (wmax = max multiplier on high-grad blocks).
    const f64 WMAX[]={2.0,4.0,8.0}; const f64 WGAMMA=0.7;

    const f64 q_targets[]={16,32,64,128};  // map to a uniform base step => sets ratio
    printf("\nAt EQUAL TOTAL BYTES per q-target. Metrics: PSNR(dB,up) GMSD(down) edgeMAE(down) HaarPSI(up,*1e3) seam(down)\n");
    for(int qi=0;qi<4;++qi){
        f32 ustep=(f32)q_targets[qi];
        // UNIFORM baseline: every block at ustep -> defines the target byte budget.
        for(u32 b=0;b<g.nb;++b){ int best=0; f32 bd=1e9f;
            for(int gi=0;gi<NSTEP;++gi){ f32 dd=fabsf(STEP_GRID[gi]-ustep); if(dd<bd){bd=dd;best=gi;} }
            idxu[b]=best; }
        f64 tb_uni=0; for(u32 b=0;b<g.nb;++b) tb_uni+=rd[b].r[idxu[b]];
        reconstruct(vol,&g,idxu,rec);
        scoreset su; score(vol,rec,&g,tb_uni,&su);
        printf("\n  q=%-4.0f target ratio=%.1fx (uniform budget=%.0f B)\n", q_targets[qi], su.ratio, tb_uni);
        printf("    %-14s %5.2fx  PSNR %6.2f  GMSD %.4f  edgeMAE %6.3f  Haar %6.2f  seam %6.3f\n",
               "UNIFORM", su.ratio, su.psnr, su.gmsd, su.edge, su.haar*1000, su.seam);

        // MSE-LAG control (plain MSE Lagrangian at same budget).
        alloc_to_target(rd,w_flat,g.nb,tb_uni,idx);
        f64 tbm=0; for(u32 b=0;b<g.nb;++b) tbm+=rd[b].r[idx[b]];
        reconstruct(vol,&g,idx,rec);
        scoreset sm; score(vol,rec,&g,tbm,&sm);
        printf("    %-14s %5.2fx  PSNR %6.2f  GMSD %.4f  edgeMAE %6.3f  Haar %6.2f  seam %6.3f   (dPSNR %+.2f)\n",
               "MSE-LAG", sm.ratio, sm.psnr, sm.gmsd, sm.edge, sm.haar*1000, sm.seam, sm.psnr-su.psnr);

        // EDGEW: edge-weighted Lagrangian, swept over wmax, same budget.
        for(int wi=0;wi<3;++wi){
            f64 *wt=malloc(g.nb*sizeof(f64));
            for(u32 b=0;b<g.nb;++b) wt[b]=weight_of(gradmean[b],gmax,WGAMMA,WMAX[wi]);
            alloc_to_target(rd,wt,g.nb,tb_uni,idx);
            f64 tbe=0; for(u32 b=0;b<g.nb;++b) tbe+=rd[b].r[idx[b]];
            reconstruct(vol,&g,idx,rec);
            scoreset se; score(vol,rec,&g,tbe,&se);
            char tag[24]; snprintf(tag,sizeof tag,"EDGEW w=%.0f",WMAX[wi]);
            // diagnostic: how many blocks moved to a FINER (lower) step vs uniform,
            // and the mean step on the top-gradient decile vs the flat decile.
            u32 finer=0,coarser=0; for(u32 b=0;b<g.nb;++b){ if(idx[b]<idxu[b])finer++; else if(idx[b]>idxu[b])coarser++; }
            printf("    %-14s %5.2fx  PSNR %6.2f  GMSD %.4f  edgeMAE %6.3f  Haar %6.2f  seam %6.3f   (dGMSD %+.4f dEdge %+.3f dHaar %+.2f) [%u finer/%u coarser]\n",
                   tag, se.ratio, se.psnr, se.gmsd, se.edge, se.haar*1000, se.seam,
                   se.gmsd-su.gmsd, se.edge-su.edge, (se.haar-su.haar)*1000, finer, coarser);
            free(wt);
        }
    }
    free(gradmean);free(rd);free(idx);free(idxu);free(rec);free(w_flat);
}

int main(int argc, char **argv){
    if(argc>=5 && argv[1][0]!='-'){
        u32 d=atoi(argv[2]),h=atoi(argv[3]),w=atoi(argv[4]); size_t len;
        u8 *v=read_file(argv[1],&len);
        if(!v||len<(size_t)d*h*w){fprintf(stderr,"bad raw\n");return 1;}
        run(argv[1],v,d,h,w); free(v); return 0;
    }
    if(argc>=2 && strcmp(argv[1],"--refbuild")==0){
        const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
        const char *labels[]={"PHerc Paris 4 hires-256","PHerc Paris 4 coarse-256"};
        for(int i=0;i<2;++i){ size_t len; u8 *v=read_file(files[i],&len);
            if(!v||len<256*256*256){fprintf(stderr,"missing %s\n",files[i]);continue;}
            run(labels[i],v,256,256,256); free(v); }
        return 0;
    }
    fprintf(stderr,"usage: bench_edgeweight --refbuild | <raw.u8> dz dy dx\n");
    return 1;
}
