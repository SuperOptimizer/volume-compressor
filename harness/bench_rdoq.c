// RDOQ bake-off (EXPERIMENT #19). The ONLY fair RDOQ question is ratio-at-EQUAL-
// QUALITY: trading PSNR for ratio is just "use a bigger q". So we build a full
// rate-distortion CURVE for the baseline (independent rounding, finely swept q)
// and for RDOQ (swept q at a fixed lambda), then report, at each RDOQ operating
// point, the ratio gain vs the baseline INTERPOLATED to the SAME PSNR. A positive
// ratio gain at equal PSNR (== negative BD-rate) is a real RDOQ win.
//
// Stack: the won max-ratio config (DCT-16^3 + HF-quant + box-128 M1 + RLGR, no
// prediction). Decode is unchanged by RDOQ (encode-only). Real PHerc Paris 4
// 256^3 volumes (hires_256.u8 + coarse_256.u8 from harness/refbuild).
#include "../src/chunkmodel/blockgrid.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static u8 *rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f)return 0;fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);u8*b=malloc(s);if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return 0;}fclose(f);*n=s;return b;}

typedef struct { double q, ratio, psnr, msssim, gmsd, bpv, enc, dec; } pt;

static pt run(const u8*vol,u32 D,u32 H,u32 W, vc_entropy_mode en, float q, int rdoq, float lambda){
    vc_bg_cfg cfg; memset(&cfg,0,sizeof(cfg));
    cfg.stencil=VC_STENCIL_NONE; cfg.traversal=VC_TRAV_RASTER; cfg.edge=VC_EDGE_SELF;
    cfg.entropy=en; cfg.chunk_atoms=8; cfg.shared_dc=1; cfg.dc_subvol=0; cfg.step=q;
    cfg.group_mode=VC_GROUP_BOX; cfg.rdoq=rdoq; cfg.rdoq_lambda=lambda;
    pt r; memset(&r,0,sizeof(r)); r.q=q;
    vc_bg_archive *a=NULL; vc_bg_stats st;
    double t0=now_sec();
    if (vc_bg_encode(vol,D,H,W,&cfg,&a,&st)){ r.ratio=-1; return r; }
    double t1=now_sec();
    u8 *rec=calloc((size_t)D*H*W,1);
    vc_bg_decode(a,rec);
    double t2=now_sec();
    vc_metrics m; vc_compute_metrics(vol,rec,D,H,W,&m);
    double V=(double)((size_t)D*H*W), raw=V/1e6;
    r.ratio=V/st.total_bytes; r.bpv=8.0*(double)st.total_bytes/V;
    r.psnr=m.psnr; r.msssim=m.ms_ssim; r.gmsd=vc_gmsd(vol,rec,D,H,W);
    r.enc=raw/(t1-t0); r.dec=raw/(t2-t1);
    vc_bg_free(a); free(rec);
    return r;
}

// Linearly interpolate baseline bpv (bits/voxel) at target PSNR from a curve
// sorted by increasing PSNR. Returns -1 if target is out of curve range.
static double interp_bpv(const pt*c,int n,double psnr){
    for (int i=0;i+1<n;++i){
        double p0=c[i].psnr, p1=c[i+1].psnr;
        if ((psnr>=p0&&psnr<=p1)||(psnr<=p0&&psnr>=p1)){
            double t=(psnr-p0)/(p1-p0);
            return c[i].bpv + t*(c[i+1].bpv-c[i].bpv);
        }
    }
    return -1;
}

static void run_input(const char*name,const u8*vol,u32 D,u32 H,u32 W){
    // Fine baseline q grid for the RD curve (independent rounding).
    float bq[]={8,11,16,22,32,45,64,90,128,180,256};
    int nb=(int)(sizeof(bq)/sizeof(bq[0]));
    for (int e=0;e<2;++e){
        vc_entropy_mode en = e? VC_ENT_RANS_SHARED : VC_ENT_RLGR;
        const char *enm = e? "rANS" : "RLGR";
        // baseline curve, sorted by PSNR ascending (q descending)
        pt base[16];
        for (int i=0;i<nb;++i) base[i]=run(vol,D,H,W,en,bq[nb-1-i],0,0.f); // q desc -> psnr asc
        printf("# %s  %s  baseline RD curve (independent rounding):\n",name,enm);
        for (int i=0;i<nb;++i) printf("#   q=%-5.0f bpv=%6.4f PSNR=%6.2f MSSSIM=%.4f GMSD=%.4f ratio=%6.2f enc=%.1f\n",
              base[i].q,base[i].bpv,base[i].psnr,base[i].msssim,base[i].gmsd,base[i].ratio,base[i].enc);
        double avg_enc_base=0; for(int i=0;i<nb;++i) avg_enc_base+=base[i].enc; avg_enc_base/=nb;
        // RDOQ operating points at matched q's and a few lambdas
        float rq[]={16,32,64,128};
        float lam[]={0.06f,0.10f,0.15f,0.22f};
        printf("# %s  %s  RDOQ vs baseline-at-equal-PSNR (ratio gain = real RDOQ win):\n",name,enm);
        for (int li=0;li<4;++li){
            double avg_enc_rdoq=0; int cnt=0;
            for (int qi=0;qi<4;++qi){
                pt r=run(vol,D,H,W,en,rq[qi],1,lam[li]);
                double bbpv=interp_bpv(base,nb,r.psnr);
                double gain = (bbpv>0)? 100.0*(bbpv-r.bpv)/bbpv : 0; // % bitrate saved @ equal PSNR
                // equal-MS-SSIM check (base curve sorted by PSNR asc == MSSSIM asc)
                double bbpv_s=-1; for(int i=0;i+1<nb;++i){double a0=base[i].msssim,a1=base[i+1].msssim;
                    if((r.msssim>=a0&&r.msssim<=a1)||(r.msssim<=a0&&r.msssim>=a1)){double t=(r.msssim-a0)/(a1-a0);bbpv_s=base[i].bpv+t*(base[i+1].bpv-base[i].bpv);break;}}
                double gain_s=(bbpv_s>0)?100.0*(bbpv_s-r.bpv)/bbpv_s:0;
                avg_enc_rdoq+=r.enc; cnt++;
                printf("  %s %s rdoq l=%-4.2f q=%-4.0f bpv=%6.4f PSNR=%6.2f MSSSIM=%.4f GMSD=%.4f | bitrate@PSNR %+5.1f%% @MSSSIM %+5.1f%% (%s) enc=%5.1f\n",
                    name,enm,lam[li],r.q,r.bpv,r.psnr,r.msssim,r.gmsd,gain,gain_s,
                    gain>0?"SAVE":"LOSE",r.enc);
            }
            avg_enc_rdoq/=cnt;
            printf("  -> %s %s l=%.2f  avg enc baseline %.1f MB/s vs RDOQ %.1f MB/s (%.2fx slower)\n",
                   name,enm,lam[li],avg_enc_base,avg_enc_rdoq,avg_enc_base/avg_enc_rdoq);
        }
        printf("\n");
    }
}

int main(int argc,char**argv){
    const char *root=(argc>1)?argv[1]:".";
    char ph[600],pc[600];
    snprintf(ph,sizeof(ph),"%s/harness/refbuild/hires_256.u8",root);
    snprintf(pc,sizeof(pc),"%s/harness/refbuild/coarse_256.u8",root);
    size_t nh,nc; u8 *hi=rd(ph,&nh), *co=rd(pc,&nc);
    const size_t V=(size_t)256*256*256;
    if (hi&&nh!=V){free(hi);hi=0;}
    if (co&&nc!=V){free(co);co=0;}
    if(!hi)fprintf(stderr,"[no hires_256.u8 at %s]\n",ph);
    if(!co)fprintf(stderr,"[no coarse_256.u8 at %s]\n",pc);
    printf("# RDOQ bake-off (DCT-16^3 + HF-quant + box-128 M1, no prediction)\n");
    printf("# Fair metric: ratio/bitrate at EQUAL PSNR (interpolated baseline RD curve).\n\n");
    if (hi) run_input("hires-256",hi,256,256,256);
    if (co) run_input("coarse-256",co,256,256,256);
    free(hi); free(co);
    return 0;
}
