// Rate-control allocator bench (PLAN §2 rate-control row). Drives the Lagrangian
// allocator on real PHerc Paris 4 sub-volumes (or a raw u8 file, or a synthetic
// air+structure mix) and reports everything RATECTRL_RESULTS.md needs:
//   (i)   does it HIT a target global ratio (10x / 20x / 50x)?
//   (ii)  variable-q vs fixed-q quality at the SAME achieved ratio (PSNR/MSE).
//   (iii) per-16^3-block q-field vs per-chunk-q delta.
//   (iv)  Parseval-estimate vs true-decode-MSE divergence.
//   (v)   encode-time cost: probe vs analytical-Laplacian.
//   (vi)  per-unit step distribution on a real mixed air+ink volume.
//
// Usage:
//   bench_ratectrl <raw.u8> <dz> <dy> <dx>
//   bench_ratectrl --refbuild            # uses harness/refbuild/{hires,coarse}_256.u8
//   (no args)                            # synthetic air+structure mix
#include "../include/vc/vc.h"
#include "../src/ratectrl/ratectrl.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static u8 *read_file(const char *p, size_t *len){
    FILE *f=fopen(p,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    u8 *b=malloc(s); if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return NULL;}
    fclose(f); *len=(size_t)s; return b;
}

static u8 *make_mixed(u32 d,u32 h,u32 w){
    u8 *v=malloc((size_t)d*h*w);
    for(u32 z=0;z<d;++z)for(u32 y=0;y<h;++y)for(u32 x=0;x<w;++x){
        double val = (x<w/2) ? 12.0+((x*3+y*5+z*7)%3)
            : 110.0+50.0*sin(x*0.20)*cos(y*0.15)+30.0*sin((x+y+z)*0.08)+40.0*((((x/5)+(z/7))&1)?1:0);
        int iv=(int)(val+0.5); v[((size_t)z*h+y)*w+x]=(u8)(iv<0?0:(iv>255?255:iv));
    }
    return v;
}

// MSE between two u8 volumes.
static double mse(const u8 *a, const u8 *b, size_t n){
    double s=0; for(size_t i=0;i<n;++i){ double e=(double)a[i]-(double)b[i]; s+=e*e; } return s/n;
}
static double psnr_of(double m){ return m<=0?99.0:10.0*log10(255.0*255.0/m); }

// Encode the volume with a UNIFORM step (fixed-q) via the real codec, return
// achieved ratio + true whole-volume PSNR.
static void fixed_q(const u8 *vol,u32 d,u32 h,u32 w,f32 step,double *ratio,double *psnr){
    u8 *arc=NULL; size_t alen=0;
    vc_encode_volume(vol,d,h,w,step,&arc,&alen);
    u8 *rec=NULL; u32 rd,rh,rw; vc_decode_volume(arc,alen,&rec,&rd,&rh,&rw);
    *ratio = (double)((size_t)d*h*w)/alen;
    *psnr = psnr_of(mse(vol,rec,(size_t)d*h*w));
    free(arc); free(rec);
}

// REAL per-chunk variable-q: the codec already stores one step per chunk in the
// chunk header, but vc_encode_volume drives one global step. To exercise true
// variable allocation through the unmodified codec we encode each VC_CHUNK_SIDE^3
// chunk as its OWN sub-volume at the allocator's per-chunk step, sum the chunk
// payload bytes, decode it back, and scatter into the reconstruction. This is an
// honest whole-volume PSNR at the allocator's per-chunk q-field (per-chunk
// granularity). `cs` is vc_chunk_side(). Returns achieved ratio + true PSNR.
static void variable_q_perchunk(const u8 *vol,u32 d,u32 h,u32 w,
                                const vc_rc_unit *units,double *ratio,double *psnr){
    u32 cs=vc_chunk_side();
    u32 ncz=(d+cs-1)/cs, ncy=(h+cs-1)/cs, ncx=(w+cs-1)/cs;
    u8 *rec=calloc((size_t)d*h*w,1);
    u8 *cbuf=malloc((size_t)cs*cs*cs);
    size_t total_bytes=0; u32 ui=0;
    for(u32 cz=0;cz<ncz;++cz)for(u32 cy=0;cy<ncy;++cy)for(u32 cx=0;cx<ncx;++cx,++ui){
        // gather chunk (zero-pad edges) into a cs^3 sub-volume
        for(u32 z=0;z<cs;++z)for(u32 y=0;y<cs;++y)for(u32 x=0;x<cs;++x){
            u32 vz=cz*cs+z,vy=cy*cs+y,vx=cx*cs+x;
            cbuf[(size_t)(z*cs+y)*cs+x]=(vz<d&&vy<h&&vx<w)?vol[((size_t)vz*h+vy)*w+vx]:0;
        }
        u8 *arc=NULL; size_t alen=0;
        vc_encode_volume(cbuf,cs,cs,cs,units[ui].step,&arc,&alen);
        total_bytes+=alen;
        u8 *crec=NULL; u32 rd,rh,rw; vc_decode_volume(arc,alen,&crec,&rd,&rh,&rw);
        for(u32 z=0;z<cs;++z)for(u32 y=0;y<cs;++y)for(u32 x=0;x<cs;++x){
            u32 vz=cz*cs+z,vy=cy*cs+y,vx=cx*cs+x;
            if(vz<d&&vy<h&&vx<w) rec[((size_t)vz*h+vy)*w+vx]=crec[(size_t)(z*cs+y)*cs+x];
        }
        free(arc); free(crec);
    }
    *ratio=(double)((size_t)d*h*w)/(double)total_bytes;
    *psnr=psnr_of(mse(vol,rec,(size_t)d*h*w));
    free(rec); free(cbuf);
}

// FAST-ENCODE tier bake-off: probe vs Laplacian vs closed-form vs feedback, all
// per-16^3, measured on the SAME footing — allocation wall-time (the win) AND an
// honest q-field PSNR+ratio through the real codec (eval_qfield_realcodec).
static void run_fast_tier(const u8 *vol,u32 d,u32 h,u32 w){
    const double targets[]={16.0,32.0,64.0,128.0}; // q-knobs map ~ ratio targets
    const char *mnames[]={"probe","laplacian","closed-form","feedback"};
    const vc_rc_model models[]={VC_RC_MODEL_PROBE,VC_RC_MODEL_LAPLACIAN,
                                VC_RC_MODEL_CLOSEDFORM,VC_RC_MODEL_FEEDBACK};
    printf("\n=== FAST-ENCODE TIER (per-16^3): probe vs laplacian vs closed-form vs feedback ===\n");
    printf("pred_ratio = allocator's predicted ratio (shared-table model, faithful).\n");
    printf("qfPSNR = honest whole-volume PSNR of the chosen q-field through the codec's\n");
    printf("         exact DCT+deadzone (vc_rc_qfield_truemse) — pure allocation quality.\n");
    printf("target | model       | alloc_ms | pred_ratio | ratio_err | qfPSNR_dB | dB_vs_probe | speedup\n");
    printf("-------+-------------+----------+------------+-----------+-----------+-------------+--------\n");
    for(int ti=0;ti<4;++ti){
        double probe_ms=0, probe_psnr=0;
        for(int mi=0;mi<4;++mi){
            vc_rc_config cfg={ .gran=VC_RC_PER_BLOCK,.dist=VC_RC_DIST_PARSEVAL,
                .model=models[mi],.target_ratio=targets[ti],.step_window=0.0 };
            u32 nu=vc_rc_count_units(d,h,w,VC_RC_PER_BLOCK);
            vc_rc_unit *u=malloc((size_t)nu*sizeof(vc_rc_unit)); vc_rc_result res;
            double t0=now_sec(); vc_rc_allocate(vol,d,h,w,&cfg,u,&res); double t1=now_sec();
            double ams=(t1-t0)*1e3;
            double m=vc_rc_qfield_truemse(vol,d,h,w,u);
            double qp=psnr_of(m);
            double rerr=100.0*(res.achieved_ratio-targets[ti])/targets[ti];
            if(mi==0){ probe_ms=ams; probe_psnr=qp; }
            printf("%5.0fx | %-11s | %8.1f | %9.2fx | %+7.1f%%  | %8.2f  | %+10.2f  | %6.2fx\n",
                   targets[ti],mnames[mi],ams,res.achieved_ratio,rerr,qp,
                   qp-probe_psnr, probe_ms/(ams+1e-9));
            free(u);
        }
    }
}

static void run(const char *label, const u8 *vol, u32 d, u32 h, u32 w){
    size_t raw=(size_t)d*h*w;
    printf("\n#### %s  (%ux%ux%u, %.1f MB, chunk=%u)\n", label,d,h,w,raw/1e6,vc_chunk_side());

    const double targets[] = { 10.0, 20.0, 50.0 };
    printf("\n(i) Target-ratio hit + (iii) per-16^3 vs per-chunk + (ii) PSNR vs fixed-q\n");
    printf("target | gran      | achieved | lambda | enc_ms | parMSE | truMSE | est-distortion\n");
    printf("-------+-----------+----------+--------+--------+--------+--------+---------------\n");
    for(int ti=0; ti<3; ++ti){
        for(int gi=0; gi<2; ++gi){
            vc_rc_gran gran = gi==0?VC_RC_PER_CHUNK:VC_RC_PER_BLOCK;
            vc_rc_config cfg = { .gran=gran, .dist=VC_RC_DIST_PARSEVAL, .model=VC_RC_MODEL_PROBE, .target_ratio=targets[ti], .step_window=0.0 };
            u32 nu = vc_rc_count_units(d,h,w,gran);
            vc_rc_unit *u = malloc((size_t)nu*sizeof(vc_rc_unit));
            vc_rc_result res;
            double t0=now_sec(); vc_rc_allocate(vol,d,h,w,&cfg,u,&res); double t1=now_sec();
            double td=0; for(u32 i=0;i<nu;++i) td+=u[i].dist;
            printf("%5.0fx | %-9s | %7.2fx | %6.2g | %6.1f | %6.2f | %6.2f | %.4g\n",
                   targets[ti], gi==0?"per-chunk":"per-16^3", res.achieved_ratio, res.lambda,
                   (t1-t0)*1e3, res.parseval_mse, res.true_mse, td/nu);
            free(u);
        }
    }

    // (iv) Parseval vs true-MSE divergence, per target, true-MSE allocation.
    printf("\n(iv) Parseval estimate vs true decode-MSE (per-16^3, true-MSE allocation)\n");
    printf("target | achieved | mean parMSE | mean truMSE | ratio par/tru\n");
    printf("-------+----------+-------------+-------------+--------------\n");
    for(int ti=0; ti<3; ++ti){
        vc_rc_config cfg = { .gran=VC_RC_PER_BLOCK, .dist=VC_RC_DIST_TRUEMSE, .model=VC_RC_MODEL_PROBE, .target_ratio=targets[ti], .step_window=0.0 };
        u32 nu = vc_rc_count_units(d,h,w,VC_RC_PER_BLOCK);
        vc_rc_unit *u = malloc((size_t)nu*sizeof(vc_rc_unit));
        vc_rc_result res; vc_rc_allocate(vol,d,h,w,&cfg,u,&res);
        printf("%5.0fx | %7.2fx | %11.3f | %11.3f | %.3f\n",
               targets[ti], res.achieved_ratio, res.parseval_mse, res.true_mse,
               res.true_mse>0?res.parseval_mse/res.true_mse:0.0);
        free(u);
    }

    // (v) encode-time cost: probe vs analytical Laplacian (per-16^3, 20x).
    printf("\n(v) Allocation cost: multi-q probe vs analytical Laplacian (per-16^3, 20x)\n");
    {
        u32 nu = vc_rc_count_units(d,h,w,VC_RC_PER_BLOCK);
        vc_rc_unit *u = malloc((size_t)nu*sizeof(vc_rc_unit)); vc_rc_result res;
        vc_rc_config cp = { .gran=VC_RC_PER_BLOCK, .dist=VC_RC_DIST_PARSEVAL, .model=VC_RC_MODEL_PROBE, .target_ratio=20.0, .step_window=0.0 };
        double t0=now_sec(); vc_rc_allocate(vol,d,h,w,&cp,u,&res); double t1=now_sec();
        double r_probe=res.achieved_ratio;
        vc_rc_config cl = { .gran=VC_RC_PER_BLOCK, .dist=VC_RC_DIST_PARSEVAL, .model=VC_RC_MODEL_LAPLACIAN, .target_ratio=20.0, .step_window=0.0 };
        double t2=now_sec(); vc_rc_allocate(vol,d,h,w,&cl,u,&res); double t3=now_sec();
        printf("  probe     : %.1f ms  achieved %.2fx\n", (t1-t0)*1e3, r_probe);
        printf("  laplacian : %.1f ms  achieved %.2fx  (%.2fx faster)\n",
               (t3-t2)*1e3, res.achieved_ratio, (t1-t0)/(t3-t2+1e-9));
        free(u);
    }

    // (vi) per-unit step distribution on the volume (per-16^3, 20x).
    printf("\n(vi) Per-16^3 step distribution at 20x target (histogram over step grid)\n");
    {
        vc_rc_config cfg = { .gran=VC_RC_PER_BLOCK, .dist=VC_RC_DIST_PARSEVAL, .model=VC_RC_MODEL_PROBE, .target_ratio=20.0, .step_window=0.0 };
        u32 nu = vc_rc_count_units(d,h,w,VC_RC_PER_BLOCK);
        vc_rc_unit *u = malloc((size_t)nu*sizeof(vc_rc_unit)); vc_rc_result res;
        vc_rc_allocate(vol,d,h,w,&cfg,u,&res);
        // bucket by step value (matches lagrangian.c STEP_GRID)
        const f32 grid[]={1.f,1.5f,2.f,3.f,4.f,6.f,8.f,12.f,16.f,24.f,32.f,48.f,64.f,90.f,128.f,180.f};
        const int NG=16; int *cnt=calloc(NG,sizeof(int));
        f32 smin=1e9f,smax=0; for(u32 i=0;i<nu;++i){ if(u[i].step<smin)smin=u[i].step; if(u[i].step>smax)smax=u[i].step;
            for(int g=0;g<NG;++g) if(fabsf(u[i].step-grid[g])<0.01f){cnt[g]++;break;} }
        printf("  step:");for(int g=0;g<NG;++g)printf(" %5.0f",grid[g]); printf("\n");
        printf("  cnt :");for(int g=0;g<NG;++g)printf(" %5d",cnt[g]); printf("\n  (min=%.1f max=%.1f spread=%.1fx over %u blocks)\n",smin,smax,smax/smin,nu);
        printf("  => easy/air units take coarse steps (high ratio), busy/structure units stay fine.\n");
        free(cnt); free(u);
    }

    // (ii) variable-q quality buy: fixed-q vs allocator per-chunk variable-q,
    // BOTH measured through the real codec as true whole-volume PSNR, matched at
    // ~the allocator's achieved ratio. Per-chunk variable-q is what the unmodified
    // codec can express today (one step per chunk header); per-16^3 needs the
    // q-field stored in the chunk header (future codec lever).
    printf("\n(ii) Variable-q (per-chunk, real codec) vs fixed-q at matched ratio\n");
    for(double tgt=10.0; tgt<=50.0; tgt*=sqrt(5.0)){
        // allocate per-chunk steps at this target, then encode each chunk at its step
        vc_rc_config cfg={ .gran=VC_RC_PER_CHUNK, .dist=VC_RC_DIST_TRUEMSE, .model=VC_RC_MODEL_PROBE, .target_ratio=tgt, .step_window=0.0 };
        u32 nu=vc_rc_count_units(d,h,w,VC_RC_PER_CHUNK);
        vc_rc_unit *u=malloc((size_t)nu*sizeof(vc_rc_unit)); vc_rc_result res;
        vc_rc_allocate(vol,d,h,w,&cfg,u,&res);
        double vr,vp; variable_q_perchunk(vol,d,h,w,u,&vr,&vp);
        // fixed-q uniform step matched to the SAME achieved ratio vr
        double fr=0,fp=0; f32 lo=1,hi=250,step=20;
        for(int it=0;it<28;++it){ step=sqrtf(lo*hi); fixed_q(vol,d,h,w,step,&fr,&fp); if(fr<vr) lo=step; else hi=step; if(fabs(fr-vr)<vr*0.02) break; }
        printf("  target %4.0fx | variable %.2fx PSNR=%.2f dB  vs  fixed %.2fx PSNR=%.2f dB  (delta %+.2f dB)\n",
               tgt, vr, vp, fr, fp, vp-fp);
        free(u);
    }

    run_fast_tier(vol,d,h,w);
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
    fprintf(stderr,"[no input -> synthetic air+structure mix]\n");
    u8 *v=make_mixed(128,128,256); run("synthetic air+structure",v,128,128,256); free(v);
    return 0;
}
