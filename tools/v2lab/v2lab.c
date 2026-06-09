// v2lab — experiment rig for the v2 volume-compressor codec (NOT a frozen spec).
// Total rewrite, C only. Compares the v2 octree/variable-DCT/ZERO-mask design
// against a v1-equivalent fixed-32^3 DCT, on real masked scroll chunks, over the
// wide metric basket. Parameterized over the open design forks so we can sweep.
//
// Pipeline per TRANSFORM leaf: DC-subtract -> separable DCT(S) -> dead-zone quant
// (HF-boost curve) -> [optional HF-zeroing] -> bit-cost estimate. ZERO leaves
// (all-air) cost ~0 (one prune bit). Subdivision: a node splits if it is MIXED
// (contains air) OR its activity exceeds a threshold (small DCT on busy data),
// down to a min transform size; sub-min nonzero -> coded raw.
//
// build: cc -O3 -march=native -ffast-math -o v2lab v2lab.c -lm
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "dct_fastf.h"
#include "metrics.h"
#include "rangecoder.h"

typedef uint8_t u8; typedef int32_t i32; typedef uint32_t u32;

// ---- quant: dead-zone + HF-boost curve (v1's knobs as a starting point) ----
// Quant knobs — RUNTIME (v1's 0.80/0.40/0.65 were tuned for 32³; sweep for 16³).
float g_dz_frac = 0.80f;     // dead-zone width fraction of step
float g_dq_offset = 0.40f;   // dequant reconstruction sub-center offset
float g_hf_exp = 0.65f;      // HF-boost curve exponent (<1 keeps HF relatively finer)
#define DZ_FRAC g_dz_frac
#define DQ_OFFSET g_dq_offset
#define HF_EXP g_hf_exp

static float hf_weight(int cz,int cy,int cx){ return powf(1.0f+(float)(cz+cy+cx), HF_EXP); }
int g_charge_leaf_mask = 1;   // 0 when a single GLOBAL mask is accounted instead
int g_airfill = 0;            // 1 = fill air voxels with material mean before DCT (flatten)
int g_perblock_q = 0;         // 1 = per-block adaptive q (constant-quality rate alloc)
int g_twophase_lf = 0;        // >0 = 2-phase 32³ leaves with this LF-cube edge (e.g. 8)
int g_residual = 0;           // 1 = code DCT of (vox - upsampled-coarse predictor) [full]
int g_dcpred = 0;             // 1 = DC-only prediction from coarse-LOD block-mean (surgical)
const u8 *g_pred = 0;         // predictor (upsampled coarse LOD), whole volume
int g_predZ=0,g_predY=0,g_predX=0;

// Quantize one coefficient; returns level. Dequant reconstruction in deq().
static inline i32 quant_one(float c, float step){
    float dz = DZ_FRAC*step;
    float a = fabsf(c);
    i32 lv = 0;
    if (a >= dz) lv = (i32)((a-dz)/step + 1.0f);
    return c<0 ? -lv : lv;
}
static inline float deq_one(i32 lv, float step){
    if (lv==0) return 0;
    float r = (fabsf((float)lv) - 1.0f + DQ_OFFSET + DZ_FRAC)*step;
    return lv<0 ? -r : r;
}

// Estimate bits for a block of quantized levels via an order-0 entropy model of
// the magnitudes + sign + a last-significant(EOB) scan termination. This is a
// CHEAP proxy for the real range coder (good enough to rank design choices).
// Returns estimated bits.
static double estimate_bits(const i32 *lv, int S){
    int n=S*S*S;
    // histogram of |lv| (capped), plus sign bits for nonzeros, plus EOB cost.
    // entropy of significance + magnitude. We use a simple split: P(zero) and
    // a geometric-ish magnitude code (exp-golomb length) — mirrors v1's coder.
    long nz=0; double mag_bits=0, sign_bits=0;
    long last_sig=-1;
    for (int i=0;i<n;++i){ if(lv[i]) { nz++; last_sig=i; } }
    // significance: binary entropy over n positions up to EOB
    long upto = last_sig+1; if(upto<1) upto=0;
    double sig_bits=0;
    if (upto>0){
        double p = (double)nz/upto; if(p<=0)p=1e-9; if(p>=1)p=1-1e-9;
        double H = -(p*log2(p) + (1-p)*log2(1-p));
        sig_bits = H*upto;
    }
    // magnitude: exp-golomb-ish length for |lv|-1
    for (int i=0;i<upto;++i){ i32 a=lv[i]<0?-lv[i]:lv[i]; if(a){ double v=a; mag_bits += 2.0*floor(log2(v))+1.0; sign_bits+=1.0; } }
    // EOB position cost ~ log2(n)
    double eob_bits = log2((double)n+1);
    return sig_bits + mag_bits + sign_bits + eob_bits;
}

// ---- one transform leaf: encode (estimate bits) + reconstruct in place ----
// vox: S^3 u8 block. Writes reconstruction into rec (same layout). Returns bits.
// hf_zero: if >0, zero coefficients with (cz+cy+cx) >= hf_zero*S*(3/?) — i.e.
// keep only a low-frequency fraction. 0 disables.
// DCT leaf with OPTIONAL intra-leaf zero-mask (overlap/boundary handling).
// If amask!=NULL it marks air voxels (amask[i]!=0): they are FILLED with the
// mean of the unmasked (material) voxels before the DCT (no step -> no ringing at
// the boundary, smoother support for the real material = better edge quality),
// then forced back to 0 on decode. The mask itself costs ~its own entropy bits.
extern int g_perblock_q;   // 1 = per-block adaptive q (constant-quality rate alloc)
extern int g_residual;     // 1 = code DCT of (vox - predictor) instead of (vox - dc)
extern int g_dcpred;       // 1 = predict block DC from coarse-LOD block-mean, skip DC byte
extern const u8 *g_pred;   // upsampled-coarse-LOD predictor (whole volume), or NULL
extern int g_predZ,g_predY,g_predX; // predictor dims (== volume)

// Residual DCT leaf: codes DCT of (vox - upsampled-coarse-LOD predictor). The
// predictor replaces the per-block DC (it's a per-voxel coarse estimate). Air voxels
// (amask) restored to 0. Used when g_residual && g_pred. z0/y0/x0 locate the block.
static double leaf_dct_resid(const u8 *vox,const u8 *amask,u8 *rec,int S,float base_q,float hf_keep,
                             int z0,int y0,int x0){
    static _Thread_local float blk[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local float coef[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local i32 lv[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local float pblk[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    int n=S*S*S;
    // DC-PREDICT mode (g_dcpred): subtract the predictor's BLOCK-MEAN (a scalar the
    // decoder also derives from the stored coarse LOD) — removes ONLY the DC redundancy,
    // keeps all AC for the DCT to code well. Avoids the full-residual trap (which removed
    // cheap low-freq AC too). Full-residual (g_dcpred=0): subtract per-voxel predictor.
    float pmean=0;
    if(g_dcpred){ double s=0; long c=0;
        for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
            int idx=(z*S+y)*S+x; size_t gi=(((size_t)(z0+z))*g_predY+(y0+y))*g_predX+(x0+x);
            if(amask&&amask[idx])continue; s+=g_pred[gi]; c++; }
        pmean = c? (float)(s/c):0;
    }
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
        int idx=(z*S+y)*S+x; size_t gi=(((size_t)(z0+z))*g_predY+(y0+y))*g_predX+(x0+x);
        float p = g_dcpred ? pmean : (float)g_pred[gi]; pblk[idx]=p;
        float v=(amask&&amask[idx])?p:(float)vox[idx];   // air: residual 0 (matches pred)
        blk[idx]=v-p;                                    // residual from predictor (scalar or per-voxel)
    }
    dct3_fwd(blk,coef,S);
    float maxfreq=3.0f*(S-1); int do_hf=(hf_keep>0&&hf_keep<1.0f&&S>=16);
    for(int cz=0;cz<S;++cz)for(int cy=0;cy<S;++cy)for(int cx=0;cx<S;++cx){
        int idx=(cz*S+cy)*S+cx; float step=base_q*hf_weight(cz,cy,cx);
        if(do_hf&&(float)(cz+cy+cx)>hf_keep*maxfreq){lv[idx]=0;coef[idx]=0;continue;}
        lv[idx]=quant_one(coef[idx],step);
    }
    static _Thread_local rc_i16 ql[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local rc_u8 scratch[DCT_MAXN*DCT_MAXN*DCT_MAXN*2+256];
    for(int i=0;i<n;++i){i32 v=lv[i];ql[i]=(rc_i16)(v>32767?32767:v<-32768?-32768:v);}
    rc_enc e; enc_init(&e,scratch,sizeof scratch); enc_block_coefs(&e,ql,S); enc_flush(&e);
    double bits=(double)e.len*8.0;   // no DC byte: predictor IS the DC (stored as coarse LOD)
    extern int g_charge_leaf_mask;
    if(amask&&g_charge_leaf_mask){long a=0;for(int i=0;i<n;++i)a+=amask[i]?1:0;
        if(a>0&&a<n){double p=(double)a/n;bits+=-(p*log2(p)+(1-p)*log2(1-p))*n;}}
    for(int cz=0;cz<S;++cz)for(int cy=0;cy<S;++cy)for(int cx=0;cx<S;++cx){
        int idx=(cz*S+cy)*S+cx;float step=base_q*hf_weight(cz,cy,cx);coef[idx]=deq_one(lv[idx],step);}
    dct3_inv(coef,blk,S);
    for(int i=0;i<n;++i){
        if(amask&&amask[i]){rec[i]=0;continue;}
        int v=(int)lrintf(blk[i]+pblk[i]); rec[i]=(u8)(v<0?0:v>255?255:v);
    }
    return bits;
}

static double leaf_dct_m(const u8 *vox, const u8 *amask, u8 *rec, int S, float base_q, float hf_keep){
    static _Thread_local float blk[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local float coef[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local i32 lv[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    int n=S*S*S;
    long sum=0,cnt=0;
    for(int i=0;i<n;++i){ if(amask && amask[i]) continue; sum+=vox[i]; cnt++; }
    int dc = cnt? (int)((sum+cnt/2)/cnt) : 0;     // mean over MATERIAL only
    for(int i=0;i<n;++i){
        int v = (amask && amask[i]) ? dc : vox[i];  // fill air with material mean
        blk[i]=(float)(v-dc);
    }
    dct3_fwd(blk,coef,S);
    float maxfreq = 3.0f*(S-1);
    int do_hf = (hf_keep>0 && hf_keep<1.0f && S>=16);
    // PER-BLOCK ADAPTIVE Q: scale this block's q by its AC energy. Low-energy (flat)
    // blocks can use a LARGER step (compress harder, still ~lossless-looking); high-
    // energy blocks get a smaller step. Pick a q-multiplier from a small codebook and
    // store ~3 bits/block. Targets constant-quality rate allocation across blocks.
    float qmul=1.0f; int qcode=0;
    if (g_perblock_q){
        double e=0; for(int i=1;i<n;++i) e+=(double)coef[i]*coef[i]; e=sqrt(e/n); // RMS AC
        // map AC-RMS to a q-multiplier: flatter (lower RMS) -> bigger q. 8-entry codebook.
        static const float QMUL[8]={2.0f,1.6f,1.3f,1.1f,0.9f,0.75f,0.6f,0.5f};
        double t = e/ (8.0*base_q);                  // normalized activity
        qcode = (int)(t<0?0: t>7?7: 0); // placeholder; pick by thresholds below
        // higher activity -> lower index (smaller qmul). thresholds tuned roughly.
        if(e< 4) qcode=0; else if(e< 8) qcode=1; else if(e<16) qcode=2; else if(e<32) qcode=3;
        else if(e<64) qcode=4; else if(e<128) qcode=5; else if(e<256) qcode=6; else qcode=7;
        qmul=QMUL[qcode];
    }
    for(int cz=0;cz<S;++cz)for(int cy=0;cy<S;++cy)for(int cx=0;cx<S;++cx){
        int idx=(cz*S+cy)*S+cx;
        float step=base_q*qmul*hf_weight(cz,cy,cx);
        if (do_hf && (float)(cz+cy+cx) > hf_keep*maxfreq){ lv[idx]=0; coef[idx]=0; continue; }
        lv[idx]=quant_one(coef[idx],step);
    }
    // REAL range coder: encode the quantized levels, measure actual bytes.
    static _Thread_local rc_i16 ql[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local rc_u8 scratch[DCT_MAXN*DCT_MAXN*DCT_MAXN*2 + 256];
    for(int i=0;i<n;++i){ i32 v=lv[i]; ql[i]=(rc_i16)(v>32767?32767:v<-32768?-32768:v); }
    rc_enc e; enc_init(&e, scratch, sizeof scratch);
    enc_block_coefs(&e, ql, S);
    enc_flush(&e);
    double bits = (double)e.len*8.0 + 8.0; // coded coefs + DC byte
    if (g_perblock_q) bits += 3.0;         // 3 bits to store the per-block q-code
    // Per-leaf mask charge ONLY if g_charge_leaf_mask (off when a GLOBAL mask is used).
    extern int g_charge_leaf_mask;
    if (amask && g_charge_leaf_mask){ long a=0; for(int i=0;i<n;++i) a+=amask[i]?1:0;
        if(a>0 && a<n){ double p=(double)a/n; bits += -(p*log2(p)+(1-p)*log2(1-p))*n; } }
    for(int cz=0;cz<S;++cz)for(int cy=0;cy<S;++cy)for(int cx=0;cx<S;++cx){
        int idx=(cz*S+cy)*S+cx; float step=base_q*qmul*hf_weight(cz,cy,cx);
        coef[idx]=deq_one(lv[idx],step);
    }
    dct3_inv(coef,blk,S);
    for(int i=0;i<n;++i){
        if (amask && amask[i]) { rec[i]=0; continue; }   // masked air -> exactly 0
        int v=(int)lrintf(blk[i])+dc; rec[i]=(u8)(v<0?0:v>255?255:v);
    }
    return bits;
}
static double leaf_dct(const u8 *vox, u8 *rec, int S, float base_q, float hf_keep){
    return leaf_dct_m(vox,NULL,rec,S,base_q,hf_keep);
}

// gather an S^3 block from the volume at (z0,y0,x0) into dst (zero-padded if OOB)
static void gather(const u8 *vol,int Z,int Y,int X,int z0,int y0,int x0,int S,u8 *dst){
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
        int zz=z0+z,yy=y0+y,xx=x0+x;
        dst[(z*S+y)*S+x] = (zz<Z&&yy<Y&&xx<X)? vol[((size_t)zz*Y+yy)*X+xx] : 0;
    }
}
static void scatter(u8 *vol,int Z,int Y,int X,int z0,int y0,int x0,int S,const u8 *src){
    for(int z=0;z<S;++z)for(int y=0;y<S;++y)for(int x=0;x<S;++x){
        int zz=z0+z,yy=y0+y,xx=x0+x;
        if(zz<Z&&yy<Y&&xx<X) vol[((size_t)zz*Y+yy)*X+xx]=src[(z*S+y)*S+x];
    }
}

// block stats: count nonzero, variance of nonzero
static void blk_stats(const u8 *b,int S,long *nz,double *var){
    int n=S*S*S; long c=0; double s=0,s2=0;
    for(int i=0;i<n;++i){ if(b[i]){c++; s+=b[i]; s2+=(double)b[i]*b[i]; } }
    *nz=c; *var = c>1 ? (s2/c - (s/c)*(s/c)) : 0;
}

// stats directly over a sub-box of the volume (no copy) — for large nodes that
// exceed the small leaf buffer. Returns nonzero count + variance of nonzero.
static void box_stats(const u8 *vol,int Z,int Y,int X,int z0,int y0,int x0,int S,
                      long *nz,double *var){
    long c=0; double s=0,s2=0;
    for(int z=0;z<S;++z)for(int y=0;y<S;++y){
        int zz=z0+z,yy=y0+y; if(zz>=Z||yy>=Y) continue;
        const u8 *row=&vol[((size_t)zz*Y+yy)*X+x0];
        int xlim = (x0+S<=X)? S : (X-x0);
        for(int x=0;x<xlim;++x){ u8 v=row[x]; if(v){c++; s+=v; s2+=(double)v*v;} }
    }
    *nz=c; *var = c>1 ? (s2/c - (s/c)*(s/c)) : 0;
}

// ---- v2 octree codec: recursive encode of a region ----
typedef struct {
    int min_dct, max_dct;   // transform size bounds (e.g. 4..32)
    float base_q;
    float hf_keep;          // 0 = off; else keep low-freq fraction
    float var_split;        // split a homogeneous (all-nonzero) node if var>this
    float zero_eps;         // LOSSY zero mask: prune node if nonzero-fraction <= this
                            // (0 = exact all-zero only). Drops a few faint voxels.
    float overlap_air;      // Case-B trigger: if a MIXED DCT-eligible node has air-frac
                            // in (zero_eps, overlap_air], CODE A DCT HERE then RECURSE to
                            // place finer ZERO leaves OVER it (zero-as-deeper-tree). The
                            // DCT covers the whole node; deeper ZERO leaves override air
                            // sub-regions. 0 = disabled (always clean-split mixed nodes).
    float zero_child_eps;   // in Case-B recursion, a child becomes a ZERO override leaf if
                            // its air-frac >= this (else left to the base DCT). e.g. 0.85.
    float zbase_air;        // MIRROR Case-B (ZERO base + DCT refine): if a MIXED node has
                            // air-frac >= this (mostly air, sparse signal), paint ZERO base
                            // then recurse to overlay DCT pockets where signal flecks are.
                            // 0 = disabled. e.g. 0.65. Takes priority over DCT-base Case B.
    float dct_child_nz;     // in mirror recursion, a child becomes a DCT-refine leaf if its
                            // nonzero-frac >= this (else stays zero). e.g. 0.15.
    // accounting
    double bits;
    long n_zero_leaf, n_dct_leaf[6], n_caseB; // counts
    long pruned_vox;
    double bits_tree;      // structural bits (split flags + leaf tags)
    double bits_coef[6];   // coded coefficient bits per log2(size)
    // NOTE: RD-split (H.266-style trial-encode compare) was considered and REJECTED
    // — research says full RDO buys only ~1.4% over a cheap heuristic, and on this
    // smooth data even less; greedy-biggest is strictly better (simpler+faster, same
    // ratio). Speed is a hard gate. Keep greedy-biggest.
} v2ctx;

// zero a sub-box of the recon volume (for pruned ZERO leaves of any size).
static void box_zero(u8 *vol,int Z,int Y,int X,int z0,int y0,int x0,int S){
    for(int z=0;z<S;++z)for(int y=0;y<S;++y){ int zz=z0+z,yy=y0+y; if(zz>=Z||yy>=Y)continue;
        int xlim=(x0+S<=X)?S:(X-x0); if(xlim>0) memset(&vol[((size_t)zz*Y+yy)*X+x0],0,xlim); }
}

// Case-B helper: after a DCT is painted over a node, recurse to OVERRIDE air
// sub-regions with deeper ZERO leaves (zero-as-deeper-tree). Only descends into
// children that are substantially air (>= zero_child_eps); material children are
// left to the base DCT. Writes 0 into the recon over those sub-regions.
static void zero_override_rec(v2ctx *c,const u8 *vol,u8 *rec,int Z,int Y,int X,
                              int z0,int y0,int x0,int S,int min_zsplit){
    long nz; double var; box_stats(vol,Z,Y,X,z0,y0,x0,S,&nz,&var);
    long n=(long)S*S*S;
    if (nz==0){ c->bits+=1.0; c->n_zero_leaf++; c->pruned_vox+=n; box_zero(rec,Z,Y,X,z0,y0,x0,S); return; }
    double air=1.0-(double)nz/(double)n;
    if (air < c->zero_child_eps) { c->bits+=1.0; return; }   // mostly material: keep base DCT (1 'no-override' bit)
    if (S<=min_zsplit){ // small + mostly air -> override to zero (lossy: drops the few nonzero)
        c->bits+=1.0; c->n_zero_leaf++; c->pruned_vox+=nz; box_zero(rec,Z,Y,X,z0,y0,x0,S); return; }
    c->bits+=1.0; int h=S/2; // subdivide the override deeper
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
        zero_override_rec(c,vol,rec,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h,min_zsplit);
}

// Mirror Case-B helper: ZERO is the base (region painted 0 by the caller); recurse
// to find DCT pockets (children with enough signal) and overlay them ON TOP of the
// zeros. Children too sparse to bother are left zero (lossy: drops sparse flecks).
static void dct_refine_rec(v2ctx *c,const u8 *vol,u8 *rec,int Z,int Y,int X,
                           int z0,int y0,int x0,int S){
    static _Thread_local u8 buf[DCT_MAXN*DCT_MAXN*DCT_MAXN], rbuf[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    long nz; double var; box_stats(vol,Z,Y,X,z0,y0,x0,S,&nz,&var);
    long n=(long)S*S*S;
    double nzf=(double)nz/(double)n;
    if (nzf < c->dct_child_nz){ c->bits+=1.0; return; }   // too sparse: leave as zero base
    if (S > c->min_dct && nzf < 0.85){ // still mixed -> descend to localize the DCT pocket
        c->bits+=1.0; int h=S/2;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
            dct_refine_rec(c,vol,rec,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h);
        return;
    }
    // dense enough (or at min size): overlay a DCT here on top of the zero base
    gather(vol,Z,Y,X,z0,y0,x0,S,buf);
    c->bits += 1.0 + leaf_dct(buf,rbuf,S,c->base_q,c->hf_keep);
    c->n_dct_leaf[dct_log2(S)]++;
    scatter(rec,Z,Y,X,z0,y0,x0,S,rbuf);
}

// returns nothing; accumulates into ctx, writes recon into rec volume.
// Two-split-kinds model: ZERO leaf | clean recurse (Case A) | DCT-base+deeper-ZERO
// (Case B) | ZERO-base+deeper-DCT (mirror Case B). Large nodes analyzed in-place.
static void v2_rec(v2ctx *c, const u8 *vol,u8 *rec,int Z,int Y,int X,
                   int z0,int y0,int x0,int S){
    static _Thread_local u8 buf[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    static _Thread_local u8 rbuf[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    long nz; double var; box_stats(vol,Z,Y,X,z0,y0,x0,S,&nz,&var);
    long n=(long)S*S*S;
    // LOSSY ZERO leaf: prune if nonzero fraction <= zero_eps (0 => exact all-zero).
    if (nz==0 || (double)nz <= c->zero_eps*(double)n){
        c->bits += 1.0; c->bits_tree += 1.0; c->n_zero_leaf++; c->pruned_vox += n;
        box_zero(rec,Z,Y,X,z0,y0,x0,S); return; }
    int mixed = (nz < n);
    double air_frac = 1.0 - (double)nz/(double)n;
    int can_split = (S > c->min_dct);
    int too_big = (S > c->max_dct || S > DCT_MAXN);
    // MIRROR Case B (ZERO base + deeper DCT refine): mostly-air node with sparse
    // signal -> paint ZERO over the whole node, then recurse to overlay DCT pockets.
    if (!too_big && mixed && c->zbase_air>0 && air_frac>=c->zbase_air){
        c->bits += 1.0;                              // zero-base tag
        box_zero(rec,Z,Y,X,z0,y0,x0,S);             // base layer: all zero
        c->n_zero_leaf++;
        int h=S/2;                                   // recurse for DCT pockets on top
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
            dct_refine_rec(c,vol,rec,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h);
        return;
    }
    // KEY FIX: a node that is MOSTLY MATERIAL should stop splitting and use a BIG
    // DCT at its current size — the few interior air voxels ride into the DCT
    // (approximated, cheap). Only KEEP SPLITTING when there's ENOUGH air to be
    // worth carving (boundary regions). overlap_air = max air-frac tolerated in a
    // DCT leaf before we must split to carve the air. This is what lets big clean
    // material use 16/32^3 DCTs instead of fragmenting into millions of 4^3.
    int air_ok = (c->overlap_air>0) ? (air_frac <= c->overlap_air) : (!mixed);
    int use_caseB = 0;
    // Split if: too big for DCT, OR there's too much air here to tolerate (carve it),
    // OR (legacy) activity exceeds var_split.
    int want_split = too_big || (!air_ok) || (var > c->var_split);
    if (can_split && want_split){
        c->bits += 1.0; c->bits_tree += 1.0; // split flag
        int h=S/2;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
            v2_rec(c,vol,rec,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h);
        return;
    }
    // Paint a DCT over the whole node. If it contains air, fill the air voxels with
    // the material mean BEFORE the DCT (kills the air->material step -> no ringing ->
    // lower max-err) and force them back to 0 on decode. This is the leaf_dct_m mask
    // path, applied ONLY to the few air-tolerant boundary leaves (cheap with min8).
    gather(vol,Z,Y,X,z0,y0,x0,S,buf);
    c->bits += 1.0; c->bits_tree += 1.0;            // leaf/DCT tag
    // Clean path: air (if any) rides into the DCT as real 0s, approximated. Cheap;
    // max-err already matches v1. (Air-fill+mask variant tested: better max-err but
    // costs the mask -> not worth the ratio. p99 is what matters for ink and is fine.)
    double cb = leaf_dct(buf,rbuf,S,c->base_q,c->hf_keep);
    c->bits += cb; c->bits_coef[dct_log2(S)] += cb;
    c->n_dct_leaf[dct_log2(S)]++;
    scatter(rec,Z,Y,X,z0,y0,x0,S,rbuf);
    // Case B: overlay deeper ZERO leaves that override air sub-regions of this DCT.
    if (use_caseB){
        c->bits += 1.0; // 'has zero-override subtree' flag
        c->n_caseB++;
        int h=S/2;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
            zero_override_rec(c,vol,rec,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h, c->min_dct/2>0?c->min_dct/2:2);
    }
}

// ====================================================================
// DENSE-INDEX encoder (the chosen v2 structure): NO walked tree. A flat dense
// grid of cells; each cell is ZERO or a DCT of size in {min_dct..max_dct}. Greedy-
// biggest: try the largest size whose air-frac <= overlap_air; if too much air,
// drop one size and try the 8 sub-cells; ZERO if empty. The per-cell size code is
// the dense index (counted at ~log2(#sizes+1) bits/finest-cell, compresses tiny).
// Produces O(1)-addressable blocks (no descent). Same leaf_dct coding as before.
// 2-PHASE leaf on a 32³ region: phase1 = a 32³ DCT keeping only the lowest LF³ coeffs
// (the cross-block low-freq the 16³ blocks miss); phase2 = subtract phase1 recon and
// code the residual as 8× 16³ DCTs. Recovers the 32³ low-freq advantage as a real
// hierarchical TRANSFORM (not prediction). Returns bits; writes recon into rec.
extern int g_twophase_lf;   // LF cube edge kept in phase1 (e.g. 8 → 8³=512 of 32768)
static double leaf_2phase(v2ctx*c,const u8*vol,u8*rec,int Z,int Y,int X,int z0,int y0,int x0){
    const int S=32; int n=S*S*S; int LF=g_twophase_lf?g_twophase_lf:8;
    static _Thread_local u8 buf[32*32*32];
    static _Thread_local float blk[32*32*32], coef[32*32*32], lfrec[32*32*32];
    static _Thread_local rc_i16 ql[32*32*32]; static _Thread_local rc_u8 scr[32*32*32*2+256];
    gather(vol,Z,Y,X,z0,y0,x0,S,buf);
    // material mean over the 32³ (air filled to mean to flatten, mask restores 0 later)
    long sum=0,cnt=0; for(int i=0;i<n;++i){ if(buf[i]){sum+=buf[i];cnt++;} }
    int dc=cnt?(int)((sum+cnt/2)/cnt):0;
    for(int i=0;i<n;++i) blk[i]=(float)((buf[i]?buf[i]:dc)-dc);
    // PHASE 1: full fwd DCT, keep only LF^3 low-freq coeffs, quantize+code them.
    dct3_fwd(blk,coef,S);
    double bits=8.0; // dc byte
    { atom_ctx ac; atom_ctx_init(&ac); (void)ac; }  // (we reuse enc_block_coefs on the LF subcube)
    // build an LF^3 quantized subcube (coarse q for the smooth band)
    static _Thread_local rc_i16 lfq[32*32*32];
    int ln=LF*LF*LF; for(int i=0;i<ln;++i) lfq[i]=0;
    for(int cz=0;cz<LF;++cz)for(int cy=0;cy<LF;++cy)for(int cx=0;cx<LF;++cx){
        int gi=(cz*S+cy)*S+cx; float step=c->base_q*hf_weight(cz,cy,cx);
        int lv=quant_one(coef[gi],step); int li=(cz*LF+cy)*LF+cx;
        lfq[li]=(rc_i16)(lv>32767?32767:lv<-32768?-32768:lv);
    }
    rc_enc e1; enc_init(&e1,scr,sizeof scr); enc_block_coefs(&e1,lfq,LF); enc_flush(&e1);
    bits += e1.len*8.0;
    // reconstruct phase1 (low-freq only) into lfrec
    for(int i=0;i<n;++i) coef[i]=0;
    for(int cz=0;cz<LF;++cz)for(int cy=0;cy<LF;++cy)for(int cx=0;cx<LF;++cx){
        int gi=(cz*S+cy)*S+cx; float step=c->base_q*hf_weight(cz,cy,cx);
        int li=(cz*LF+cy)*LF+cx; coef[gi]=deq_one(lfq[li],step);
    }
    dct3_inv(coef,lfrec,S);   // lfrec = smooth low-freq reconstruction (zero-mean)
    // PHASE 2: residual = (block - dc) - lfrec, code as 8× 16³ DCTs.
    int H=16; double rb=0;
    for(int bz=0;bz<2;++bz)for(int by=0;by<2;++by)for(int bx=0;bx<2;++bx){
        static _Thread_local float r16[16*16*16],rc16[16*16*16]; static _Thread_local rc_i16 q16[16*16*16];
        for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){
            int gz=bz*H+z,gy=by*H+y,gx=bx*H+x; int gi=(gz*S+gy)*S+gx;
            r16[(z*H+y)*H+x]=blk[gi]-lfrec[gi];   // residual after low-freq removed
        }
        float c16[16*16*16]; dct3_fwd(r16,c16,H);
        for(int i=0;i<H*H*H;++i){ float step=c->base_q; /* flat q on residual detail */
            int cz=i/(H*H),cy=(i/H)%H,cx=i%H; step=c->base_q*hf_weight(cz,cy,cx);
            int lv=quant_one(c16[i],step); q16[i]=(rc_i16)(lv>32767?32767:lv<-32768?-32768:lv);
        }
        rc_enc e2; enc_init(&e2,scr,sizeof scr); enc_block_coefs(&e2,q16,H); enc_flush(&e2); rb+=e2.len*8.0;
        // recon this 16³ residual block and add back lfrec+dc
        for(int i=0;i<H*H*H;++i){ int cz=i/(H*H),cy=(i/H)%H,cx=i%H; float step=c->base_q*hf_weight(cz,cy,cx); c16[i]=deq_one(q16[i],step);}
        dct3_inv(c16,rc16,H);
        // write recon for this sub-block (residual + low-freq + dc; air->0)
        for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){
            int gz=bz*H+z,gy=by*H+y,gx=bx*H+x; int gi=(gz*S+gy)*S+gx;
            float val=rc16[(z*H+y)*H+x]+lfrec[gi]+dc; int iv=(int)lrintf(val);
            int zz=z0+gz,yy=y0+gy,xx=x0+gx;
            if(zz<Z&&yy<Y&&xx<X) rec[((size_t)zz*Y+yy)*X+xx] = buf[gi]? (u8)(iv<0?0:iv>255?255:iv) : 0;
        }
    }
    bits+=rb;
    c->n_dct_leaf[5]++; // count as a 32-region
    return bits;
}

static void dense_fill(v2ctx *c,const u8*vol,u8*rec,int Z,int Y,int X,int z0,int y0,int x0,int S){
    static _Thread_local u8 buf[DCT_MAXN*DCT_MAXN*DCT_MAXN], rbuf[DCT_MAXN*DCT_MAXN*DCT_MAXN];
    long nz; double var; box_stats(vol,Z,Y,X,z0,y0,x0,S,&nz,&var);
    long n=(long)S*S*S;
    double air = 1.0 - (double)nz/(double)n;
    if (nz==0 || (double)nz <= c->zero_eps*n){     // ZERO cell -> index flag only
        c->n_zero_leaf++; c->pruned_vox+=n; box_zero(rec,Z,Y,X,z0,y0,x0,S); return;
    }
    // greedy-biggest with a HARD FLOOR at min_dct: place a DCT here if clean enough,
    // OR if we're at the floor (a floor cell with any material is one DCT; air rides
    // in — it was already caught as ZERO above if empty/near-empty). NEVER a sub-min
    // DCT (that reintroduces the 4^3 fragmentation). ZERO cells (above) are the mask.
    // SIZE-SCALED air tolerance (var_split repurposed as a flag: >0 enables): bigger
    // blocks demand to be CLEANER (less air), so boundaries push down to 8^3 (good
    // worst-case) while 32^3 stays on truly-clean interior (good ratio). eff = oair *
    // (min_dct/S). var_split<=0 (default 1e9) keeps the flat threshold.
    double eff_oair = (c->var_split>0 && c->var_split<1e8) ? c->overlap_air*((double)c->min_dct/S) : c->overlap_air;
    // Must be at or below max_dct to place a DCT here; a too-big node always splits.
    if (S <= c->max_dct && (air <= eff_oair || S <= c->min_dct)){
        // 2-PHASE: a clean 32³ node → low-freq-32³ band + 8× 16³ residual (recover 32³
        // ratio at ~16³ decode latency). Only at S==32 when enabled.
        if (g_twophase_lf && S==32){
            double cb=leaf_2phase(c,vol,rec,Z,Y,X,z0,y0,x0); c->bits+=cb; c->bits_coef[5]+=cb; return;
        }
        gather(vol,Z,Y,X,z0,y0,x0,S,buf);
        double cb;
        if ((g_residual||g_dcpred) && g_pred){   // residual (full) or DC-only prediction from coarse LOD
            static _Thread_local u8 am[DCT_MAXN*DCT_MAXN*DCT_MAXN];
            u8 *amp=NULL; if(nz<n){ for(long i=0;i<n;++i) am[i]=buf[i]?0:1; amp=am; }
            cb=leaf_dct_resid(buf,amp,rbuf,S,c->base_q,c->hf_keep,z0,y0,x0);
        } else if (g_airfill && nz<n){      // mixed: AIR-FILL (flatten step) + mask restores 0
            static _Thread_local u8 am[DCT_MAXN*DCT_MAXN*DCT_MAXN];
            for(long i=0;i<n;++i) am[i]=buf[i]?0:1;
            cb=leaf_dct_m(buf,am,rbuf,S,c->base_q,c->hf_keep);  // fills air w/ material mean, rec air->0
        } else cb=leaf_dct(buf,rbuf,S,c->base_q,c->hf_keep);
        c->bits+=cb; c->bits_coef[dct_log2(S)]+=cb; c->n_dct_leaf[dct_log2(S)]++;
        scatter(rec,Z,Y,X,z0,y0,x0,S,rbuf); return;
    }
    int h=S/2;                                      // too much air: subdivide for size/ZERO
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
        dense_fill(c,vol,rec,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h);
}
// Encode a whole V^3 via the dense index, region by region (REG = a coarse cell,
// e.g. 32). Index cost = (#finest cells) * log2(nsizes+1) bits, added after.
static double mask_octree_bits(const u8*vol,int Z,int Y,int X,int z0,int y0,int x0,int S);
static double dense_encode(v2ctx *base,const u8*vol,u8*rec,int V,int REG){
    v2ctx tot=*base; tot.bits=0; tot.bits_tree=0; memset(tot.bits_coef,0,sizeof tot.bits_coef);
    tot.n_zero_leaf=0; memset(tot.n_dct_leaf,0,sizeof tot.n_dct_leaf); tot.pruned_vox=0;
    for(int z=0;z<V;z+=REG)for(int y=0;y<V;y+=REG)for(int x=0;x<V;x+=REG)
        dense_fill(&tot,vol,rec,V,V,V,z,y,x,REG);
    // dense index: 1 size-code per FINEST cell. #finest = (V/min_dct)^3. Each code
    // in {ZERO,8,16,32,COVERED} ~ log2(5)=2.32 bits, but it's RLE/range-coded to far
    // less (huge ZERO/COVERED runs). Estimate generously at 0.1 bit/finest-cell.
    double fcells = pow((double)V/base->min_dct, 3.0);
    double idx_bits = fcells * 0.1;   // compressed dense index (conservative)
    // If air-fill is on, a SINGLE global octree-compressed occupancy mask restores
    // the zeros on decode. Account it ONCE here (per-leaf charge is disabled).
    double mask_bits = 0;
    if (g_airfill){
        for(int z=0;z<V;z+=REG)for(int y=0;y<V;y+=REG)for(int x=0;x<V;x+=REG)
            mask_bits += mask_octree_bits(vol,V,V,V,z,y,x,REG);
    }
    tot.bits_tree = idx_bits + mask_bits;
    *base = tot; base->bits += idx_bits + mask_bits;
    return base->bits;
}

// ---- v1-equivalent: fixed 32^3 DCT every atom, AF_ZERO for all-air atoms ----
static void v1_encode(v2ctx *c,const u8 *vol,u8 *rec,int Z,int Y,int X){
    static _Thread_local u8 buf[32*32*32], rbuf[32*32*32];
    int S=32;
    for(int z0=0;z0<Z;z0+=S)for(int y0=0;y0<Y;y0+=S)for(int x0=0;x0<X;x0+=S){
        gather(vol,Z,Y,X,z0,y0,x0,S,buf);
        long nz; double var; blk_stats(buf,S,&nz,&var);
        if(nz==0){ c->bits+=1.0; c->n_zero_leaf++; memset(rbuf,0,S*S*S); scatter(rec,Z,Y,X,z0,y0,x0,S,rbuf); continue; }
        c->bits += 1.0 + leaf_dct(buf,rbuf,S,c->base_q,0.0f);
        c->n_dct_leaf[dct_log2(S)]++;
        scatter(rec,Z,Y,X,z0,y0,x0,S,rbuf);
    }
}

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }

static u8 *read_raw(const char *p, size_t want){
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    u8 *b=malloc(want); if(fread(b,1,want,f)!=want){fprintf(stderr,"short read %s\n",p);exit(1);} fclose(f); return b;
}

// Run one config across a fine q-sweep; return ratio interpolated at the quality
// target (PSNR>=target_psnr), plus the basket at the operating point. Reports the
// R-D point that just meets the quality bar — the decision-relevant number.
typedef struct { int v1; int mn,mx; float hf,zeps,vsplit,oair,zchild,zbase,dchild; const char*tag; } cfg_t;

static void run_cfg(const cfg_t *cf, const u8 *vol,u8 *rec,u8 *mask,int S,
                    double logical_bytes, double target_psnr){
    // sweep q geometrically; find the highest-ratio point still >= target PSNR
    double best_ratio=0, bq=0; basket_t bestb; memset(&bestb,0,sizeof bestb);
    double enc_ms_at=0;
    for (double q=0.10; q<=16.0; q*=1.15){
        v2ctx c; memset(&c,0,sizeof c);
        c.min_dct=cf->mn; c.max_dct=cf->mx; c.base_q=(float)q;
        c.hf_keep=cf->hf; c.zero_eps=cf->zeps; c.var_split=cf->vsplit;
        c.overlap_air=cf->oair; c.zero_child_eps=cf->zchild?cf->zchild:0.85f;
        c.zbase_air=cf->zbase; c.dct_child_nz=cf->dchild?cf->dchild:0.15f;
        double t0=now_ms();
        if (cf->v1) v1_encode(&c,vol,rec,S,S,S);
        else v2_rec(&c,vol,rec,S,S,S,0,0,0,S);
        double t1=now_ms();
        basket_t b=metrics_eval(vol,rec,S,S,S,mask);
        double ratio=logical_bytes/(c.bits/8.0);
        if (b.psnr>=target_psnr && ratio>best_ratio){ best_ratio=ratio; bq=q; bestb=b; bestb.ratio=ratio; bestb.bpp=c.bits/((double)S*S*S); enc_ms_at=t1-t0; }
    }
    if (best_ratio<=0){ printf("%-16s  (never reached PSNR %.0f)\n", cf->tag, target_psnr); return; }
    printf("%-16s q=%4.2f enc%5.1fms  ", cf->tag, bq, enc_ms_at);
    basket_print("", &bestb);
}

// Load a 1024^3 u8 zarr (C-order, 128^3 chunks, dim-separator '/', key root/c0/c1/c2,
// fill_value 0) into a flat (z,y,x) volume. Missing chunk files => all-zero.
static u8 *load_zarr_1024(const char *root, int V, int CH){
    int G = V/CH;                       // chunks per axis
    u8 *vol = (u8*)calloc((size_t)V*V*V, 1);
    u8 *cb = (u8*)malloc((size_t)CH*CH*CH);
    char p[1024];
    for(int c0=0;c0<G;++c0)for(int c1=0;c1<G;++c1)for(int c2=0;c2<G;++c2){
        snprintf(p,sizeof p,"%s/0/%d/%d/%d",root,c0,c1,c2);
        FILE *f=fopen(p,"rb"); if(!f) continue;           // missing => zeros
        if(fread(cb,1,(size_t)CH*CH*CH,f)!=(size_t)CH*CH*CH){fclose(f);continue;}
        fclose(f);
        // chunk (c0,c1,c2) maps to volume block at (c0*CH, c1*CH, c2*CH) in z,y,x
        for(int z=0;z<CH;++z)for(int y=0;y<CH;++y){
            u8 *dst=&vol[(((size_t)(c0*CH+z))*V + (c1*CH+y))*V + c2*CH];
            memcpy(dst, &cb[((size_t)z*CH+y)*CH], CH);
        }
    }
    free(cb);
    return vol;
}

// Encode an entire VxVxV volume by walking REG^3 regions through v2_rec, with the
// given config + q. Accumulates bits and reconstructs into rec for whole-volume
// metrics. Returns total estimated bits.
static double encode_volume(const v2ctx *base, const u8 *vol, u8 *rec, int V, int REG){
    v2ctx tot; memset(&tot,0,sizeof tot);
    for(int z0=0;z0<V;z0+=REG)for(int y0=0;y0<V;y0+=REG)for(int x0=0;x0<V;x0+=REG){
        v2ctx c=*base; c.bits=0; c.n_zero_leaf=0; memset(c.n_dct_leaf,0,sizeof c.n_dct_leaf); c.n_caseB=0; c.pruned_vox=0;
        c.bits_tree=0; memset(c.bits_coef,0,sizeof c.bits_coef);
        v2_rec(&c, vol+0, rec, V,V,V, z0,y0,x0, REG);   // region within the full volume coords
        tot.bits+=c.bits; tot.n_zero_leaf+=c.n_zero_leaf; tot.pruned_vox+=c.pruned_vox;
        tot.bits_tree+=c.bits_tree;
        for(int i=0;i<6;++i){ tot.n_dct_leaf[i]+=c.n_dct_leaf[i]; tot.bits_coef[i]+=c.bits_coef[i]; }
    }
    printf("# leaves: zero=%ld  dct 4=%ld 8=%ld 16=%ld 32=%ld  pruned_vox=%.1f%%\n",
        tot.n_zero_leaf, tot.n_dct_leaf[2],tot.n_dct_leaf[3],tot.n_dct_leaf[4],tot.n_dct_leaf[5],
        100.0*tot.pruned_vox/((double)V*V*V));
    printf("# bits: tree=%.2fMB  coef 4=%.2f 8=%.2f 16=%.2f 32=%.2f MB  (coef MB/leaf: 4=%.0fB 8=%.0fB 16=%.0fB)\n",
        tot.bits_tree/8e6, tot.bits_coef[2]/8e6, tot.bits_coef[3]/8e6, tot.bits_coef[4]/8e6, tot.bits_coef[5]/8e6,
        tot.n_dct_leaf[2]?tot.bits_coef[2]/8/tot.n_dct_leaf[2]:0,
        tot.n_dct_leaf[3]?tot.bits_coef[3]/8/tot.n_dct_leaf[3]:0,
        tot.n_dct_leaf[4]?tot.bits_coef[4]/8/tot.n_dct_leaf[4]:0);
    return tot.bits;
}

// Octree-compress a binary occupancy mask of an S^3 region: recursively, a node
// that is all-same (all air or all material) costs 1 bit; else 1 bit + 8 children.
// Returns estimated bits — the cost of a GLOBAL zero-mask stream.
static double mask_octree_bits(const u8*vol,int Z,int Y,int X,int z0,int y0,int x0,int S){
    long nz,n=(long)S*S*S; double var; box_stats(vol,Z,Y,X,z0,y0,x0,S,&nz,&var);
    if(nz==0 || nz==n) return 1.0;          // uniform node: 1 bit
    if(S<=1) return 1.0;
    double b=1.0; int h=S/2;                // split + recurse
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
        b+=mask_octree_bits(vol,Z,Y,X,z0+dz*h,y0+dy*h,x0+dx*h,h);
    return b;
}

// Fast single-region encode with full accounting (for rapid iteration).
static void run_region(const char*tag, v2ctx base, const u8*vol,u8*rec,u8*mask,int S){
    base.bits=0; base.bits_tree=0; memset(base.bits_coef,0,sizeof base.bits_coef);
    base.n_zero_leaf=0; memset(base.n_dct_leaf,0,sizeof base.n_dct_leaf); base.pruned_vox=0;
    double t0=now_ms(); v2_rec(&base,vol,rec,S,S,S,0,0,0,S); double t1=now_ms();
    double N=(double)S*S*S;
    // metrics over SIGNAL and over ALL voxels, BEFORE the global mask overwrite
    basket_t bs=metrics_eval(vol,rec,S,S,S,mask);     // signal only (air not scored)
    basket_t ba=metrics_eval(vol,rec,S,S,S,NULL);     // all voxels (incl. air recon)
    bs.ratio=N/(base.bits/8.0); bs.bpp=base.bits/N;
    // apply GLOBAL mask overwrite: air -> exactly 0 (the user's quality-fix idea)
    static u8 *recm=NULL; static size_t recm_n=0; if(recm_n<(size_t)N){ recm=realloc(recm,(size_t)N); recm_n=N; }
    for(long i=0;i<N;++i) recm[i] = mask[i]? rec[i] : 0;
    basket_t bsm=metrics_eval(vol,recm,S,S,S,mask);   // signal, after overwrite
    basket_t bam=metrics_eval(vol,recm,S,S,S,NULL);   // all, after overwrite
    double maskbits = mask_octree_bits(vol,S,S,S,0,0,0,S);
    double ratio_with_mask = N/((base.bits+maskbits)/8.0);
    printf("%-22s d8=%-6ld d16=%-5ld d32=%-4ld z=%-6ld | DCT ratio %.2fx (+mask %.2fx, maskMB %.3f)\n",
        tag, base.n_dct_leaf[3],base.n_dct_leaf[4],base.n_dct_leaf[5],base.n_zero_leaf,
        bs.ratio, ratio_with_mask, maskbits/8e6);
    printf("    signal  no-mask: PSNR %.2f max %.0f p99 %.0f | +mask: PSNR %.2f max %.0f p99 %.0f\n",
        bs.psnr,bs.max_err,bs.p99, bsm.psnr,bsm.max_err,bsm.p99);
    printf("    allvox  no-mask: PSNR %.2f max %.0f SSIM %.4f | +mask: PSNR %.2f max %.0f SSIM %.4f\n",
        ba.psnr,ba.max_err,ba.ssim, bam.psnr,bam.max_err,bam.ssim);
}

// Light decode-side deblock across leaf seams (all planes at multiples of STEP, the
// finest leaf size). For each boundary voxel pair (a|b) across the seam, if the local
// step looks like a quant artifact (small gap, flat surroundings) nudge both toward
// their mean — clamped so real edges/ink are preserved. Skips air (0) voxels.
static void deblock(u8 *v,int V,int step,float strength,int clampd){
    // X seams
    for(int z=0;z<V;++z)for(int y=0;y<V;++y)for(int x=step;x<V;x+=step){
        size_t i=((size_t)z*V+y)*V+x; u8 a=v[i-1],b=v[i]; if(!a||!b)continue;
        int d=(int)b-a; if(d>-clampd&&d<clampd){ int adj=(int)(strength*d*0.5f);
            int na=a+adj,nb=b-adj; v[i-1]=(u8)(na<0?0:na>255?255:na); v[i]=(u8)(nb<0?0:nb>255?255:nb);} }
    // Y seams
    for(int z=0;z<V;++z)for(int y=step;y<V;y+=step)for(int x=0;x<V;++x){
        size_t i=((size_t)z*V+y)*V+x,j=((size_t)z*V+(y-1))*V+x; u8 a=v[j],b=v[i]; if(!a||!b)continue;
        int d=(int)b-a; if(d>-clampd&&d<clampd){ int adj=(int)(strength*d*0.5f);
            int na=a+adj,nb=b-adj; v[j]=(u8)(na<0?0:na>255?255:na); v[i]=(u8)(nb<0?0:nb>255?255:nb);} }
    // Z seams
    for(int z=step;z<V;z+=step)for(int y=0;y<V;++y)for(int x=0;x<V;++x){
        size_t i=((size_t)z*V+y)*V+x,j=((size_t)(z-1)*V+y)*V+x; u8 a=v[j],b=v[i]; if(!a||!b)continue;
        int d=(int)b-a; if(d>-clampd&&d<clampd){ int adj=(int)(strength*d*0.5f);
            int na=a+adj,nb=b-adj; v[j]=(u8)(na<0?0:na>255?255:na); v[i]=(u8)(nb<0?0:nb>255?255:nb);} }
}

int main(int argc,char**argv){
    dct_build();
    // --decbench : measure inverse-DCT throughput (the decode hot path) per size.
    if (argc>=2 && !strcmp(argv[1],"--decbench")){
        printf("# inverse-DCT throughput (decode hot path), float separable DCT\n");
        for(int S=8;S<=32;S<<=1){
            int n=S*S*S;
            float *coef=malloc(n*sizeof(float)),*blk=malloc(n*sizeof(float));
            // realistic sparse coef block: DC + a few low-freq
            for(int i=0;i<n;++i) coef[i]=0; coef[0]=1000; coef[1]=300; coef[S]=250; coef[S*S]=200;
            long iters = (long)(2e8 / n);   // ~constant work
            double t0=now_ms();
            for(long it=0;it<iters;++it){ dct3_inv(coef,blk,S); coef[0]+=blk[0]*0+1e-6; }
            double t1=now_ms();
            double vox=(double)n*iters, mbps=vox/1e6/((t1-t0)/1e3);
            printf("  %2d^3: %ld blocks in %.0fms -> %.0f Mvox/s (%.0f MB/s decode of DCT inverse)\n",
                   S, iters, t1-t0, vox/1e6/((t1-t0)/1e3), mbps);
            free(coef); free(blk);
        }
        // reference: v1 reports ~89-139 MB/s full decode on real data (incl entropy)
        printf("# (v1 full-decode real-data: ~89-139 MB/s incl entropy+dequant+iDCT+clamp)\n");
        return 0;
    }
    // --dense <root> <q> [oair] [min] [max] : whole-volume DENSE-INDEX encode (the
    // chosen v2 structure: flat dense index, greedy-biggest DCT, ZERO cells = mask).
    if (argc>=4 && !strcmp(argv[1],"--dense")){
        const char*root=argv[2]; float q=(float)atof(argv[3]);
        float oair=argc>4?(float)atof(argv[4]):0.20f;
        int mind=argc>5?atoi(argv[5]):8, maxd=argc>6?atoi(argv[6]):32;
        float hfk=argc>7?(float)atof(argv[7]):0.0f;   // HF-zeroing: keep low-freq fraction (0=off)
        g_airfill=argc>8?atoi(argv[8]):0;             // 1 = air-fill+global-mask (flatten DCT step)
        if(g_airfill) g_charge_leaf_mask=0;           // global mask accounts it, not per-leaf
        g_perblock_q=argc>9?atoi(argv[9]):0;          // 1 = per-block adaptive q
        g_residual=argc>10?atoi(argv[10]):0;          // 1 = full residual-from-coarse-LOD coding
        g_dcpred=argc>11?atoi(argv[11]):0;            // 1 = DC-only prediction from coarse LOD
        g_twophase_lf=argc>12?atoi(argv[12]):0;       // >0 = 2-phase 32³ leaves, LF-cube edge
        if(argc>13) g_dz_frac=(float)atof(argv[13]);   // dead-zone width (default 0.80)
        if(argc>14) g_hf_exp=(float)atof(argv[14]);    // HF-boost exponent (default 0.65)
        int V=1024; size_t N=(size_t)V*V*V;
        u8*vol=load_zarr_1024(root,V,128),*rec=malloc(N),*mask=malloc(N);
        long nz=0; for(size_t i=0;i<N;++i){mask[i]=vol[i]?1:0;nz+=mask[i];}
        if(g_residual||g_dcpred){
            // predictor = upsample(downsample2x(LOD0)). LOD1 is stored anyway (8-LOD
            // spec) so this predictor is FREE. NN 2x up: each coarse voxel predicts its 8.
            int H=V/2; u8 *lo=malloc((size_t)H*H*H);
            for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){
                unsigned s=0; for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                    s+=vol[(((size_t)(2*z+dz))*V+(2*y+dy))*V+(2*x+dx)];
                lo[((size_t)z*H+y)*H+x]=(u8)((s+4)/8);
            }
            u8 *pr=malloc(N);
            for(int z=0;z<V;++z)for(int y=0;y<V;++y)for(int x=0;x<V;++x)
                pr[(((size_t)z)*V+y)*V+x]=lo[(((size_t)(z/2))*H+(y/2))*H+(x/2)];
            g_pred=pr; g_predZ=V; g_predY=V; g_predX=V; free(lo);
        }
        v2ctx base; memset(&base,0,sizeof base); base.base_q=q; base.var_split=1e9f;
        base.min_dct=mind; base.max_dct=maxd; base.overlap_air=oair; base.hf_keep=hfk;
        double t0=now_ms(); double bits=dense_encode(&base,vol,rec,V,32); double t1=now_ms();
        basket_t bs=metrics_eval(vol,rec,V,V,V,mask);
        double comp=bits/8.0;
        // TWO ratios: (1) MATERIAL-ONLY DCT ratio = the 10-100x spec target, measured
        // on the DCT'd material only (coef+index bytes / material-voxel bytes), excludes
        // pruned air. (2) TOTAL archive ratio = all logical voxels / all bytes (mask
        // savings folded in — the more air pruned, the higher).
        double coef_bytes=0; for(int i=0;i<6;++i) coef_bytes+=base.bits_coef[i]/8.0;
        double dct_bytes = coef_bytes + base.bits_tree/8.0;   // DCT coeffs + dense index
        double material_voxels=(double)nz;
        double mat_ratio = material_voxels / dct_bytes;       // 10-100x spec number
        double tot_ratio = (double)N / comp;                  // mask-inflated total
        bs.ratio=tot_ratio; bs.bpp=bits/N;
        printf("DENSE q=%.2f oair=%.2f hf=%.2f min%d max%d: d4=%ld d8=%ld d16=%ld d32=%ld z=%ld idxMB=%.2f enc%.1fs\n",
            q,oair,hfk,mind,maxd, base.n_dct_leaf[1],base.n_dct_leaf[2],base.n_dct_leaf[4],base.n_dct_leaf[5],
            base.n_zero_leaf, base.bits_tree/8e6,(t1-t0)/1e3);
        printf("  coefMB 8=%.1f 16=%.1f 32=%.1f | MATERIAL-ratio %.1fx (spec 10-100x) | TOTAL-ratio %.1fx\n",
            base.bits_coef[2]/8e6,base.bits_coef[4]/8e6,base.bits_coef[5]/8e6, mat_ratio, tot_ratio);
        printf("  PSNR %.2f SSIM %.4f MAE %.2f max %.0f p99 %.0f  (nonzero %.1f%%, mask saved %.0f%% of vol)\n",
            bs.psnr,bs.ssim,bs.mae,bs.max_err,bs.p99, 100.0*nz/N, 100.0*(1.0-(double)nz/N));
        // DEBLOCK A/B: filter leaf seams (multiples of min_dct) and re-score (quality only,
        // same bytes — recovers the 16³ seam-artifact part of the gap to 32³).
        { u8 *db=malloc(N); memcpy(db,rec,N);
          deblock(db,V,mind,0.5f,12);   // strength 0.5, clamp ±12 (protect real edges)
          basket_t bd=metrics_eval(vol,db,V,V,V,mask);
          printf("  [+deblock seams@%d str.5 clamp12] PSNR %.2f SSIM %.4f MAE %.2f max %.0f p99 %.0f  (dPSNR %+.2f)\n",
              mind, bd.psnr,bd.ssim,bd.mae,bd.max_err,bd.p99, bd.psnr-bs.psnr);
          free(db); }
        return 0;
    }
    // --chunk <raw128> [q] : fast single-128^3-region multi-config analysis.
    if (argc>=2 && !strcmp(argv[1],"--chunk")){
        const char *path=argc>2?argv[2]:"testdata/v2chunk_128.raw"; int S=128;
        float q=argc>3?(float)atof(argv[3]):2.0f;
        size_t n=(size_t)S*S*S; u8*vol=read_raw(path,n),*rec=malloc(n),*mask=malloc(n);
        long nz=0; for(size_t i=0;i<n;++i){mask[i]=vol[i]?1:0;nz+=mask[i];}
        printf("# %s 128^3 nonzero=%.1f%% q=%.2f  (DCT-size-set floor experiment)\n",path,100.0*nz/n,q);
        v2ctx base; memset(&base,0,sizeof base); base.base_q=q; base.var_split=1e9f;
        v2ctx c;
        // The architecture question: DENSE uniform 16^3 grid + overlay mask (user's
        // "no kd-tree" proposal: min16 max16 oa=1.0 => never splits, pure grid) vs
        // the kd-tree variable-size (8..32, splits on air). Both get the +mask column.
        c=base;c.min_dct=16;c.max_dct=16;c.overlap_air=1.0f; run_region("DENSE-16grid",c,vol,rec,mask,S);
        c=base;c.min_dct=8; c.max_dct=8; c.overlap_air=1.0f; run_region("DENSE-8grid",c,vol,rec,mask,S);
        c=base;c.min_dct=8; c.max_dct=32;c.overlap_air=0.15f; run_region("KD 8..32 oa.15",c,vol,rec,mask,S);
        c=base;c.min_dct=8; c.max_dct=32;c.overlap_air=0.25f; run_region("KD 8..32 oa.25",c,vol,rec,mask,S);
        return 0;
    }
    // --zarr <root> [q] : encode the whole 1024^3 volume from a zarr store.
    if (argc>=3 && !strcmp(argv[1],"--zarr")){
        const char *root=argv[2];
        float q = argc>3 ? (float)atof(argv[3]) : 1.0f;
        float oair = argc>4 ? (float)atof(argv[4]) : 0.20f;  // air tolerated in a DCT leaf
        int maxd = argc>5 ? atoi(argv[5]) : 32;              // max DCT size
        int mind = argc>6 ? atoi(argv[6]) : 8;               // min DCT size (8 = drop 4^3)
        int V=1024, CH=128, REG=128;   // encode region = one zarr chunk
        fprintf(stderr,"loading %s (1024^3)...\n", root);
        double t0=now_ms();
        u8 *vol=load_zarr_1024(root,V,CH);
        double t1=now_ms();
        size_t N=(size_t)V*V*V;
        u8 *rec=(u8*)malloc(N);
        long nz=0; for(size_t i=0;i<N;++i) if(vol[i]) nz++;
        fprintf(stderr,"loaded in %.1fs, nonzero=%.2f%%\n",(t1-t0)/1e3,100.0*nz/N);
        v2ctx base; memset(&base,0,sizeof base);
        base.min_dct=mind; base.max_dct=maxd; base.base_q=q;
        base.var_split=1e9f;        // split on air (mask), not activity
        base.overlap_air=oair;      // mostly-material nodes stop + use a big DCT
        base.zero_eps=0; base.zbase_air=0;
        fprintf(stderr,"config: min4 max%d overlap_air=%.2f q=%.2f\n", maxd, oair, q);
        double te0=now_ms();
        double bits=encode_volume(&base,vol,rec,V,REG);
        double te1=now_ms();
        // whole-volume metrics over nonzero (signal) voxels
        u8 *mask=(u8*)malloc(N); for(size_t i=0;i<N;++i) mask[i]=vol[i]?1:0;
        basket_t b=metrics_eval(vol,rec,V,V,V,mask);
        double comp_bytes=bits/8.0, logical=(double)N;
        b.ratio=logical/comp_bytes; b.bpp=bits/N;
        printf("\n=== WHOLE VOLUME 1024^3  q=%.2f ===\n", q);
        printf("compressed: %.2f MB   logical: %.0f MB   RATIO %.2fx   bpp %.4f\n",
               comp_bytes/1e6, logical/1e6, b.ratio, b.bpp);
        printf("encode: %.2fs  (%.1f MB/s logical)\n", (te1-te0)/1e3, logical/1e6/((te1-te0)/1e3));
        basket_print("v2 1024^3 (signal)", &b);
        // also whole-volume incl. air (the 'vs raw' number)
        basket_t ball=metrics_eval(vol,rec,V,V,V,NULL); ball.ratio=b.ratio; ball.bpp=b.bpp;
        basket_print("v2 1024^3 (all vox)", &ball);
        return 0;
    }
    int S = argc>2?atoi(argv[2]):128;
    const char *path = argc>1?argv[1]:"testdata/v2chunk_128.raw";
    size_t n=(size_t)S*S*S;
    u8 *vol=read_raw(path,n);
    u8 *rec=malloc(n);
    u8 *mask=malloc(n); long nzc=0; for(size_t i=0;i<n;++i){mask[i]=vol[i]?1:0; nzc+=mask[i];}
    double logical_bytes=(double)n;
    printf("# chunk %s  %d^3  nonzero=%.1f%%\n", path,S,100.0*nzc/n);
    printf("# metrics over NONZERO voxels; ratio = best (highest) at PSNR>=target\n");
    printf("# NOTE bit-cost is an ESTIMATE (order-0 + EOB), not the real range coder\n\n");

    double targets[] = {45.0, 40.0, 36.0};
    cfg_t cfgs[] = {
        // fields: v1,min,max,hf,zero_eps,var_split, oair(DCT-base CaseB),zchild, zbase(ZERO-base mirror),dchild
        {1,32,32, 0,0,1e9, 0,0, 0,0, "v1 fixed32"},
        // --- clean-partition baseline (Case A only) ---
        {0,4,16,  0,0,1e9, 0,0, 0,0, "v2 4..16 clean"},
        {0,4,32,  0,0,1e9, 0,0, 0,0, "v2 4..32 clean"},
        // --- Case B: DCT base + deeper ZERO override ---
        {0,4,16,  0,0,1e9, 0.35f,0.90f, 0,0, "v2 dctbase a.35"},
        // --- Mirror Case B: ZERO base + deeper DCT refine (the fleck case) ---
        {0,4,16,  0,0,1e9, 0,0, 0.65f,0.15f, "v2 zerobase .65"},
        {0,4,16,  0,0,1e9, 0,0, 0.80f,0.10f, "v2 zerobase .80"},
        // --- BOTH directions enabled (encoder-ish: zero-base if mostly air, dct-base if mostly material) ---
        {0,4,16,  0,0,1e9, 0.35f,0.90f, 0.70f,0.12f, "v2 both"},
    };
    for(unsigned ti=0;ti<sizeof(targets)/sizeof(targets[0]);++ti){
        printf("=== quality target: PSNR >= %.0f dB (over signal) ===\n", targets[ti]);
        for(unsigned ci=0;ci<sizeof(cfgs)/sizeof(cfgs[0]);++ci)
            run_cfg(&cfgs[ci],vol,rec,mask,S,logical_bytes,targets[ti]);
        printf("\n");
    }
    return 0;
}
