// Block-grid (chunk-model) bake-off bench. Sweeps the three orthogonal axes
// (A stencil x B traversal x C edge) + chunk-size + entropy mode on real PHerc
// Paris 4 data, and reports per cell: ratio, PSNR, MS-SSIM, GMSD, 16-grid
// seam-step, enc/dec MB/s, and the amortized 16^3 decode cost under a
// neighborhood-sweep access trace. See docs/EXP_chunk_model.md §3.
//
// Inputs match compare_c3d_c4d.py: hires-256 (8 cached 128^3 chunks 2x2x2) and
// coarse-256 (256^3 crop of the coarse 384^3). q is the dead-zone base step.
#include "../src/chunkmodel/blockgrid.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static u8 *rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f)return 0;fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);u8*b=malloc(s);if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return 0;}fclose(f);*n=s;return b;}

// Assemble hires-256 from data/cache_hires/*.u8 (first 8, 2x2x2). Returns volume.
static u8 *load_hires256(const char *root) {
    char dir[512]; snprintf(dir,sizeof(dir),"%s/data/cache_hires",root);
    // collect up to 8 names deterministically (sorted)
    char names[64][512]; int nn=0;
    FILE *pp; char cmd[700]; snprintf(cmd,sizeof(cmd),"ls %s/*.u8 2>/dev/null | sort",dir);
    pp=popen(cmd,"r"); if(!pp) return 0;
    char line[600];
    while (nn<8 && fgets(line,sizeof(line),pp)) { line[strcspn(line,"\n")]=0; snprintf(names[nn],512,"%s",line); nn++; }
    pclose(pp);
    if (nn<8) return 0;
    const u32 C=128; u8 *vol=calloc(256u*256*256,1);
    for (int idx=0; idx<8; ++idx) {
        size_t l; u8 *ch=rd(names[idx],&l); if(!ch||l!=(size_t)C*C*C){free(vol);return 0;}
        u32 cz=(idx>>2)&1, cy=(idx>>1)&1, cx=idx&1;
        for (u32 z=0;z<C;++z) for (u32 y=0;y<C;++y) {
            u32 vz=cz*C+z, vy=cy*C+y, vx=cx*C;
            memcpy(vol+(((size_t)vz*256)+vy)*256+vx, ch+(((size_t)z*C)+y)*C, C);
        }
        free(ch);
    }
    return vol;
}
static u8 *load_coarse256(const char *root) {
    char p[512]; snprintf(p,sizeof(p),"%s/data/coarse/coarse_z15_y8_x8_3x3x3.u8",root);
    size_t l; u8 *raw=rd(p,&l); if(!raw) return 0; const u32 S=384;
    if (l<(size_t)S*S*S){free(raw);return 0;}
    const u32 C=256; u8 *vol=malloc((size_t)C*C*C);
    for (u32 z=0;z<C;++z) for (u32 y=0;y<C;++y) memcpy(vol+(((size_t)z*C)+y)*C, raw+(((size_t)z*S)+y)*S, C);
    free(raw); return vol;
}

// 16-grid seam-step: mean | |rec step| - |ref step| | across planes at x,y,z = k*16.
static double seam16(const u8*ref,const u8*rec,u32 D,u32 H,u32 W){
    double s=0; size_t n=0;
    for (u32 z=0;z<D;++z) for (u32 y=0;y<H;++y) for (u32 x=16;x<W;x+=16){
        size_t i=((size_t)z*H+y)*W+x;
        s+=fabs(fabs((double)rec[i]-rec[i-1])-fabs((double)ref[i]-ref[i-1])); n++;
    }
    return n? s/n : 0;
}

typedef struct { const char*name; vc_bg_cfg cfg; } cell;

static void run_cell(const char*input,const u8*vol,u32 D,u32 H,u32 W,
                     const char*label, vc_bg_cfg cfg) {
    vc_bg_archive *a=NULL; vc_bg_stats st;
    double t0=now_sec();
    if (vc_bg_encode(vol,D,H,W,&cfg,&a,&st)) { printf("%-22s %-11s ENCODE FAIL\n",label,input); return; }
    double t1=now_sec();
    u8 *rec=calloc((size_t)D*H*W,1);
    vc_bg_decode(a,rec);
    double t2=now_sec();
    vc_metrics m; vc_compute_metrics(vol,rec,D,H,W,&m);
    double gm=vc_gmsd(vol,rec,D,H,W);
    double seam=seam16(vol,rec,D,H,W);
    double ratio=(double)((size_t)D*H*W)/st.total_bytes;
    double raw=(double)((size_t)D*H*W)/1e6;
    // group-header overhead = (shared table + seek directory) as % of payload.
    double hdr_pct = st.payload_bytes ?
        100.0*(double)(st.table_bytes+st.directory_bytes)/(double)st.payload_bytes : 0;
    double enc=raw/(t1-t0), dec=raw/(t2-t1);

    // Single-atom random-access decode cost (avg touched + time/atom).
    u32 Az=vc_bg_natoms(D),Ay=vc_bg_natoms(H),Ax=vc_bg_natoms(W);
    u8 atom[4096]; double tt=0; u64 sumtouch=0; int N=200; unsigned seed=999;
    double ta=now_sec();
    for (int t=0;t<N;++t){ seed=seed*1103515245u+12345u; u32 az=(seed>>16)%Az;
        seed=seed*1103515245u+12345u; u32 ay=(seed>>16)%Ay; seed=seed*1103515245u+12345u; u32 ax=(seed>>16)%Ax;
        u32 touched=0; vc_bg_decode_atom(a,az,ay,ax,atom,&touched); sumtouch+=touched; }
    tt=now_sec()-ta;
    double avgtouch=(double)sumtouch/N;
    double us_per_atom=tt/N*1e6;

    // Neighborhood-sweep amortized cost: decode a 4x4x4-atom box, count unique
    // atom-decodes / atoms-in-box (the key amortized metric, spec §3).
    u64 totdec=0; u32 box=0;
    u32 bz=Az/2>2?Az/2-2:0, by=Ay/2>2?Ay/2-2:0, bx=Ax/2>2?Ax/2-2:0;
    vc_bg_decode_region(a,bz,by,bx,bz+4,by+4,bx+4,&totdec,&box);
    double amort = box? (double)totdec/box : 0;

    // table bytes as % of payload (isolates the E1/E2 table-coding effect).
    double tbl_pct = st.payload_bytes ? 100.0*(double)st.table_bytes/(double)st.payload_bytes : 0;
    printf("%-28s %-10s %7.1f %6.2f %7.4f %7.4f %7.0f %6.2f %7.2f %6.2f %6.2f %6.2f %6u\n",
           label,input,ratio,m.psnr,m.ms_ssim,gm,dec,avgtouch,us_per_atom,amort,hdr_pct,tbl_pct,st.n_chunks);
    (void)seam; (void)enc;
    vc_bg_free(a); free(rec);
}

int main(int argc,char**argv){
    const char *root = (argc>1)? argv[1] : ".";
    float q = (argc>2)? (float)atof(argv[2]) : 16.0f;
    u8 *hi=load_hires256(root), *co=load_coarse256(root);
    if (!hi) fprintf(stderr,"[no hires data under %s]\n",root);
    if (!co) fprintf(stderr,"[no coarse data under %s]\n",root);

    const char *mode = (argc>3)? argv[3] : "";
    int curve_mode = (strcmp(mode,"curve")==0);

    printf("# block-grid bake-off  q(base-step)=%.1f   atom=16^3%s\n", q,
           curve_mode? "   [CURVE-GROUP experiment]" : "");
    printf("# cols: ratio PSNR MS-SSIM GMSD decMB/s avgTouch us/atom amortDecode(4^3box) hdr%%payload tbl%%payload ngroups\n");
    printf("%-28s %-10s %7s %6s %7s %7s %7s %6s %7s %6s %6s %6s %6s\n",
           "config","input","ratio","PSNR","MSSSIM","GMSD","dec","touch","us/atom","amort","hdr%","tbl%","ngrp");

    // ----- CURVE-GROUP experiment cell set (group-mode = curve, N in {64,256,512}).
    // Baseline = box-128^3 (M1, ca8) at the SAME no-prediction winning stack, for
    // both rANS and RLGR. Then curve-group Morton/Hilbert x N for each coder.
    if (curve_mode) {
        // vc_bg_cfg fields (in order): stencil,traversal,edge,entropy,chunk_atoms,
        // shared_dc,step,dc_subvol, group_mode,group_n,
        //   boundary,drift_thresh,table_coding,dc_pred_curve,nested_sub
        #define BOX(en) (vc_bg_cfg){VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,en,8,1,q,0,VC_GROUP_BOX,0,\
            VC_BOUND_FIXED,0,VC_TABLE_FULL,0,0}
        // fixed-N curve group (the second baseline)
        #define CRV(tr,en,n) (vc_bg_cfg){VC_STENCIL_NONE,tr,VC_EDGE_SELF,en,8,1,q,0,VC_GROUP_CURVE,n,\
            VC_BOUND_FIXED,0,VC_TABLE_FULL,0,0}
        // fully-parameterised curve group
        #define CG(tr,en,n,bnd,dth,tc,dcp,nst) (vc_bg_cfg){VC_STENCIL_NONE,tr,VC_EDGE_SELF,en,8,1,q,0,\
            VC_GROUP_CURVE,n,bnd,dth,tc,dcp,nst}
        cell cc[] = {
          // ===== BASELINES =====
          {"BASE box-128 rans",          BOX(VC_ENT_RANS_SHARED)},
          {"BASE box-128 rlgr",          BOX(VC_ENT_RLGR)},
          {"BASE curveHil N64  rans",    CRV(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED, 64)},
          {"BASE curveHil N256 rans",    CRV(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED, 256)},
          {"BASE curveMor N64  rans",    CRV(VC_TRAV_MORTON, VC_ENT_RANS_SHARED, 64)},
          {"BASE curveMor N256 rans",    CRV(VC_TRAV_MORTON, VC_ENT_RANS_SHARED, 256)},
          {"BASE curveHil N256 rlgr",    CRV(VC_TRAV_HILBERT,VC_ENT_RLGR, 256)},
          // ===== C1: Hilbert vs Morton group compactness (rANS, FULL table) =====
          {"C1 Hil N512 rans",           CRV(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED, 512)},
          {"C1 Mor N512 rans",           CRV(VC_TRAV_MORTON, VC_ENT_RANS_SHARED, 512)},
          // ===== E1: group-to-group table DELTA (the small-group unlocker) =====
          {"E1 Hil N64  delta rans",     CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,64, VC_BOUND_FIXED,0,VC_TABLE_DELTA,0,0)},
          {"E1 Hil N128 delta rans",     CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,128,VC_BOUND_FIXED,0,VC_TABLE_DELTA,0,0)},
          {"E1 Hil N256 delta rans",     CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,256,VC_BOUND_FIXED,0,VC_TABLE_DELTA,0,0)},
          {"E1 Mor N64  delta rans",     CG(VC_TRAV_MORTON, VC_ENT_RANS_SHARED,64, VC_BOUND_FIXED,0,VC_TABLE_DELTA,0,0)},
          // ===== E2: coarse super-group base + per-group delta =====
          {"E2 Hil N64  base rans",      CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,64, VC_BOUND_FIXED,0,VC_TABLE_BASE,0,0)},
          {"E2 Hil N256 base rans",      CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,256,VC_BOUND_FIXED,0,VC_TABLE_BASE,0,0)},
          // ===== I1: drift-adaptive boundaries (cap=group_n) =====
          {"I1 Hil drift.03 rans",       CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.03f,VC_TABLE_FULL,0,0)},
          {"I1 Hil drift.10 rans",       CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.10f,VC_TABLE_FULL,0,0)},
          {"I1 Hil drift.15 rans",       CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.15f,VC_TABLE_FULL,0,0)},
          {"I1 Hil drift.20 rans",       CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.20f,VC_TABLE_FULL,0,0)},
          {"I1 Hil drift.15 rlgr",       CG(VC_TRAV_HILBERT,VC_ENT_RLGR,       512,VC_BOUND_DRIFT,0.15f,VC_TABLE_FULL,0,0)},
          // ===== I1 + E1 (drift boundaries + table delta = the likely winner) =====
          {"I1+E1 Hil drift.15 rans",    CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.15f,VC_TABLE_DELTA,0,0)},
          {"I1+E1 Mor drift.15 rans",    CG(VC_TRAV_MORTON, VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.15f,VC_TABLE_DELTA,0,0)},
          // ===== I2: nested sub-groups =====
          {"I2 Hil N256 nest16 rans",    CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,256,VC_BOUND_FIXED,0,VC_TABLE_FULL,0,16)},
          // ===== B1: DC-only curve-predecessor prediction (marginal-gain test) ===
          {"B1 Hil N256 dcpred rans",    CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,256,VC_BOUND_FIXED,0,VC_TABLE_FULL,1,0)},
          {"B1 Hil N256 dcpred rlgr",    CG(VC_TRAV_HILBERT,VC_ENT_RLGR,       256,VC_BOUND_FIXED,0,VC_TABLE_FULL,1,0)},
          // B2: B1 ON TOP OF the best group-model variant (drift+E1) — marginal gain
          {"B2 Hil drift.15 E1 dcpred",  CG(VC_TRAV_HILBERT,VC_ENT_RANS_SHARED,512,VC_BOUND_DRIFT,0.15f,VC_TABLE_DELTA,1,0)},
        };
        int ncc=(int)(sizeof(cc)/sizeof(cc[0]));
        for (int i=0;i<ncc;++i) if (hi) run_cell("hires-256",hi,256,256,256,cc[i].name,cc[i].cfg);
        printf("\n");
        for (int i=0;i<ncc;++i) if (co) run_cell("coarse-256",co,256,256,256,cc[i].name,cc[i].cfg);
        free(hi); free(co);
        return 0;
    }

    // The sweep matrix. base = M1 (no prediction) reference, then the axes.
    // fields: stencil, traversal, edge, entropy, chunk_atoms, shared_dc, step, dc_subvol
    #define CFG(s,t,e,en,ca,dc) (vc_bg_cfg){s,t,e,en,ca,dc,q,0}
    #define CFGD(s,t,e,en,ca,dc) (vc_bg_cfg){s,t,e,en,ca,dc,q,1}  /* dc_subvol on */
    cell cells[] = {
      // --- entropy / M0 vs M1 (no prediction), chunk 8^3 atoms (=128^3) ---
      {"M0-indep none ca8",      CFG(VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_INDEP,8,1)},
      {"M1-rans none ca8",       CFG(VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      {"M1-rlgr none ca8",       CFG(VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RLGR,8,1)},
      // --- (A) stencil sweep, raster, self, rans-shared, ca8 ---
      // NOTE (lit. review 2026-06): cross-atom AC prediction is LOW payoff
      // (HEVC inter-block coeff pred ~1.5-1.8% BD-rate; JPEG neighbor AC ~4.5%).
      // 18/26-conn predict AC terms -> measure but expect little over 6-conn.
      {"rans 6  raster self ca8",  CFG(VC_STENCIL_6, VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      {"rans 18 raster self ca8",  CFG(VC_STENCIL_18,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      {"rans 26 raster self ca8",  CFG(VC_STENCIL_26,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      // --- (B) traversal sweep, 6-conn ---
      {"rans 6  morton self ca8",  CFG(VC_STENCIL_6, VC_TRAV_MORTON, VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      {"rans 6  hilbert self ca8", CFG(VC_STENCIL_6, VC_TRAV_HILBERT,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      // --- (C) edge sweep, 6-conn raster ---
      {"rans 6  raster halo ca8",  CFG(VC_STENCIL_6, VC_TRAV_RASTER,VC_EDGE_HALO, VC_ENT_RANS_SHARED,8,1)},
      {"rans 6  raster fetch ca8", CFG(VC_STENCIL_6, VC_TRAV_RASTER,VC_EDGE_FETCH,VC_ENT_RANS_SHARED,8,1)},
      // --- chunk size knee (6-conn raster self) ---
      {"rans 6  raster self ca4",  CFG(VC_STENCIL_6, VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,4,1)},
      {"rans 6  raster self ca16", CFG(VC_STENCIL_6, VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,16,1)},
      // --- no shared DC vs shared DC (none stencil, ca8) ---
      {"M1-rans none ca8 noDC",  CFG(VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,0)},
      // --- RLGR + prediction (balanced pick) ---
      {"rlgr 6  raster self ca8",  CFG(VC_STENCIL_6, VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RLGR,8,1)},
      {"rlgr 6  hilbert self ca8", CFG(VC_STENCIL_6, VC_TRAV_HILBERT,VC_EDGE_SELF,VC_ENT_RLGR,8,1)},
      // --- DC sub-volume (global predicted DC frame), AC stencil NONE, ca8 ---
      {"rans DCsv none ca8",       CFGD(VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RANS_SHARED,8,1)},
      {"rlgr DCsv none ca8",       CFGD(VC_STENCIL_NONE,VC_TRAV_RASTER,VC_EDGE_SELF,VC_ENT_RLGR,8,1)},
    };
    int nc = (int)(sizeof(cells)/sizeof(cells[0]));
    for (int i=0;i<nc;++i) {
        if (hi) run_cell("hires-256",hi,256,256,256,cells[i].name,cells[i].cfg);
    }
    printf("\n");
    for (int i=0;i<nc;++i) {
        if (co) run_cell("coarse-256",co,256,256,256,cells[i].name,cells[i].cfg);
    }
    free(hi); free(co);
    return 0;
}
