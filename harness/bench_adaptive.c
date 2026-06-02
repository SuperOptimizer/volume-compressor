// EXP#20 — per-atom adaptive transform/mode selection bake-off.
// Compares the winning box-mode stack (DCT-16^3 atom + HF-quant + shared-table
// rANS / table-free RLGR + dc_subvol, ca8 = 128^3 chunk) with adaptive OFF
// (always-DCT baseline) vs adaptive ON at several R-D lambdas. Reports ratio,
// PSNR, MS-SSIM, GMSD, enc/dec MB/s, round-trip exactness, and the per-atom mode
// histogram (DCT/SKIP/RAW %). Also runs a synthetic air+structure mixed volume to
// stress the SKIP (air) and RAW (noise) escapes.
//
// Real data: harness/refbuild/{hires,coarse}_256.u8 (256^3 PHerc Paris 4 crops).
#include "../src/chunkmodel/blockgrid.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static u8 *rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f)return 0;fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);u8*b=malloc(s);if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return 0;}fclose(f);*n=s;return b;}

// synthetic: half air (z<128 -> 0), a central textured slab, and a noisy block.
static u8 *make_mixed256(void){
    const u32 S=256; u8 *v=calloc((size_t)S*S*S,1); unsigned seed=12345;
    for (u32 z=0;z<S;++z) for (u32 y=0;y<S;++y) for (u32 x=0;x<S;++x){
        size_t i=((size_t)z*S+y)*S+x; u8 val=0;
        if (z>=96 && z<160) {        // smooth structured slab (DCT-friendly)
            val=(u8)(120+40.0*sin(x*0.05)*cos(y*0.04)+20.0*sin(z*0.1));
        }
        if (z>=200 && y>=200) {       // pure incompressible noise block (RAW escape)
            seed=seed*1103515245u+12345u; val=(u8)(seed>>24);
        }
        v[i]=val;
    }
    return v;
}

typedef struct { const char*name; vc_bg_cfg cfg; } cell;

static void run_cell(const char*input,const u8*vol,u32 D,u32 H,u32 W,
                     const char*label, vc_bg_cfg cfg){
    vc_bg_archive *a=NULL; vc_bg_stats st;
    double t0=now_sec();
    if (vc_bg_encode(vol,D,H,W,&cfg,&a,&st)){ printf("%-26s %-11s ENCODE FAIL\n",label,input); return; }
    double t1=now_sec();
    u8 *rec=calloc((size_t)D*H*W,1);
    vc_bg_decode(a,rec);
    double t2=now_sec();
    vc_metrics m; vc_compute_metrics(vol,rec,D,H,W,&m);
    double gm=vc_gmsd(vol,rec,D,H,W);
    double ratio=(double)((size_t)D*H*W)/st.total_bytes;
    double raw=(double)((size_t)D*H*W)/1e6;
    double enc=raw/(t1-t0), dec=raw/(t2-t1);
    // random-access round-trip + touched check
    u32 Az=vc_bg_natoms(D),Ay=vc_bg_natoms(H),Ax=vc_bg_natoms(W);
    u8 atom[4096]; unsigned seed=7; u32 maxtouch=0; int rt_ok=1;
    for (int t=0;t<300;++t){ seed=seed*1103515245u+12345u; u32 az=(seed>>16)%Az;
        seed=seed*1103515245u+12345u; u32 ay=(seed>>16)%Ay; seed=seed*1103515245u+12345u; u32 ax=(seed>>16)%Ax;
        u32 touched=0; vc_bg_decode_atom(a,az,ay,ax,atom,&touched); if(touched>maxtouch)maxtouch=touched;
        // verify the atom matches the full-decode reconstruction
        for (u32 z=0; z<16 && rt_ok; ++z) for (u32 y=0;y<16 && rt_ok;++y) for (u32 x=0;x<16;++x){
            u32 vz=az*16+z, vy=ay*16+y, vx=ax*16+x;
            u8 ref = (vz<D&&vy<H&&vx<W)? rec[((size_t)vz*H+vy)*W+vx] : 0;
            if (atom[(z*16+y)*16+x]!=ref){ rt_ok=0; break; }
        }
    }
    u32 nm = st.n_mode_dct+st.n_mode_skip+st.n_mode_raw;
    double pd = nm?100.0*st.n_mode_dct/nm:0, ps=nm?100.0*st.n_mode_skip/nm:0, pr=nm?100.0*st.n_mode_raw/nm:0;
    printf("%-26s %-11s %7.2f %6.2f %7.4f %7.4f %7.0f %7.0f %5.1f %5.1f %5.1f  rt=%d t<=%u\n",
        label,input,ratio,m.psnr,m.ms_ssim,gm,enc,dec,pd,ps,pr,rt_ok,maxtouch);
    vc_bg_free(a); free(rec);
}

int main(int argc,char**argv){
    const char *root=(argc>1)?argv[1]:".";
    char ph[512],pc[512];
    snprintf(ph,sizeof(ph),"%s/data/refbuild/hires_256.u8",root);
    snprintf(pc,sizeof(pc),"%s/data/refbuild/coarse_256.u8",root);
    size_t nh,nc; u8 *hi=rd(ph,&nh), *co=rd(pc,&nc);
    if(!hi||nh!=(size_t)256*256*256){fprintf(stderr,"[no hires at %s]\n",ph);hi=0;}
    if(!co||nc!=(size_t)256*256*256){fprintf(stderr,"[no coarse at %s]\n",pc);co=0;}
    u8 *mx=make_mixed256();

    float qs[]={16,32,64,128}; int nq=(argc>2)?1:4;
    if (argc>2) qs[0]=(float)atof(argv[2]);

    // winning box stack: stencil NONE, raster, self, ca8, shared_dc, dc_subvol ON.
    // fields: stencil,trav,edge,entropy,chunk_atoms,shared_dc,step,dc_subvol,
    //   group_mode,group_n,boundary,drift,table_coding,dc_pred,nested,
    //   band_split,sparse_prepass,skip_meta, adaptive, adaptive_lambda
    #define BASE(en,adp,lam) (vc_bg_cfg){VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,en,8,1,q,1,\
        VC_GROUP_BOX,0,VC_BOUND_FIXED,0,VC_TABLE_FULL,0,0, VC_BAND_NONE,0,0, adp,lam}

    printf("# EXP#20 per-atom adaptive mode  (box ca8=128^3, dc_subvol, winning stack)\n");
    printf("# cols: ratio PSNR MS-SSIM GMSD encMB/s decMB/s  DCT%% SKIP%% RAW%%  rt maxtouch\n");

    for (int qi=0; qi<nq; ++qi){
        float q=qs[qi];
        printf("\n=== q=%.0f ===\n",q);
        printf("%-26s %-11s %7s %6s %7s %7s %7s %7s %5s %5s %5s\n",
            "config","input","ratio","PSNR","MSSSIM","GMSD","enc","dec","DCT","SKIP","RAW");
        struct { const char*nm; vc_entropy_mode en; int adp; float lam; } V[]={
            {"rlgr baseline(noadapt)", VC_ENT_RLGR,        0, 0},
            {"rlgr adaptive L=0.002",  VC_ENT_RLGR,        1, 0.002f},
            {"rlgr adaptive L=0.01",   VC_ENT_RLGR,        1, 0.01f},
            {"rlgr adaptive L=0.05",   VC_ENT_RLGR,        1, 0.05f},
            {"rans baseline(noadapt)", VC_ENT_RANS_SHARED, 0, 0},
            {"rans adaptive L=0.002",  VC_ENT_RANS_SHARED, 1, 0.002f},
            {"rans adaptive L=0.01",   VC_ENT_RANS_SHARED, 1, 0.01f},
            {"rans adaptive L=0.05",   VC_ENT_RANS_SHARED, 1, 0.05f},
        };
        int nv=sizeof(V)/sizeof(V[0]);
        for (int i=0;i<nv;++i){ if(hi){ vc_bg_cfg c=BASE(V[i].en,V[i].adp,V[i].lam); run_cell("hires-256",hi,256,256,256,V[i].nm,c);} }
        printf("\n");
        for (int i=0;i<nv;++i){ if(co){ vc_bg_cfg c=BASE(V[i].en,V[i].adp,V[i].lam); run_cell("coarse-256",co,256,256,256,V[i].nm,c);} }
        printf("\n");
        for (int i=0;i<nv;++i){ vc_bg_cfg c=BASE(V[i].en,V[i].adp,V[i].lam); run_cell("mixed-256",mx,256,256,256,V[i].nm,c); }
    }
    free(hi);free(co);free(mx);
    return 0;
}
