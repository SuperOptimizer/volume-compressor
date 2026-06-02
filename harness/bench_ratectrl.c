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

// Encode the volume with a UNIFORM step (fixed-q proxy) at the given step, return
// achieved ratio + true PSNR (uses the real codec).
static void fixed_q(const u8 *vol,u32 d,u32 h,u32 w,f32 step,double *ratio,double *psnr){
    u8 *arc=NULL; size_t alen=0;
    vc_encode_volume(vol,d,h,w,step,&arc,&alen);
    u8 *rec=NULL; u32 rd,rh,rw; vc_decode_volume(arc,alen,&rec,&rd,&rh,&rw);
    *ratio = (double)((size_t)d*h*w)/alen;
    *psnr = psnr_of(mse(vol,rec,(size_t)d*h*w));
    free(arc); free(rec);
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
            vc_rc_config cfg = { gran, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, targets[ti], 0 };
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
        vc_rc_config cfg = { VC_RC_PER_BLOCK, VC_RC_DIST_TRUEMSE, VC_RC_MODEL_PROBE, targets[ti], 0 };
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
        vc_rc_config cp = { VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 20.0, 0 };
        double t0=now_sec(); vc_rc_allocate(vol,d,h,w,&cp,u,&res); double t1=now_sec();
        double r_probe=res.achieved_ratio;
        vc_rc_config cl = { VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_LAPLACIAN, 20.0, 0 };
        double t2=now_sec(); vc_rc_allocate(vol,d,h,w,&cl,u,&res); double t3=now_sec();
        printf("  probe     : %.1f ms  achieved %.2fx\n", (t1-t0)*1e3, r_probe);
        printf("  laplacian : %.1f ms  achieved %.2fx  (%.2fx faster)\n",
               (t3-t2)*1e3, res.achieved_ratio, (t1-t0)/(t3-t2+1e-9));
        free(u);
    }

    // (vi) per-unit step distribution on the volume (per-16^3, 20x).
    printf("\n(vi) Per-16^3 step distribution at 20x target (histogram over step grid)\n");
    {
        vc_rc_config cfg = { VC_RC_PER_BLOCK, VC_RC_DIST_PARSEVAL, VC_RC_MODEL_PROBE, 20.0, 0 };
        u32 nu = vc_rc_count_units(d,h,w,VC_RC_PER_BLOCK);
        vc_rc_unit *u = malloc((size_t)nu*sizeof(vc_rc_unit)); vc_rc_result res;
        vc_rc_allocate(vol,d,h,w,&cfg,u,&res);
        // bucket by step value
        const f32 grid[]={1.5f,3.f,6.f,12.f,24.f,48.f,96.f}; int cnt[7]={0};
        f32 smin=1e9f,smax=0; for(u32 i=0;i<nu;++i){ if(u[i].step<smin)smin=u[i].step; if(u[i].step>smax)smax=u[i].step;
            for(int g=0;g<7;++g) if(fabsf(u[i].step-grid[g])<0.01f){cnt[g]++;break;} }
        printf("  step:");for(int g=0;g<7;++g)printf(" %6.1f",grid[g]); printf("\n");
        printf("  cnt :");for(int g=0;g<7;++g)printf(" %6d",cnt[g]); printf("   (min=%.1f max=%.1f spread=%.1fx)\n",smin,smax,smax/smin);
        printf("  => easy/air units take coarse steps (high ratio), busy/structure units stay fine.\n");
        free(u);
    }

    // (ii) variable-q quality buy: at the per-chunk 20x achieved ratio, compare
    // fixed-q (uniform step) PSNR to the allocator's mean true-MSE PSNR proxy.
    printf("\n(ii) Variable-q vs fixed-q at matched ratio (~20x)\n");
    {
        // find uniform step giving ~20x
        double r=0,p=0; f32 lo=1,hi=200,step=20;
        for(int it=0;it<24;++it){ step=sqrtf(lo*hi); fixed_q(vol,d,h,w,step,&r,&p); if(r<20.0) lo=step; else hi=step; if(fabs(r-20.0)<0.3) break; }
        printf("  fixed-q  : step=%.1f achieved %.2fx PSNR=%.2f dB\n", step, r, p);
        // allocator per-16^3 at 20x, report mean true-MSE -> PSNR proxy
        vc_rc_config cfg={VC_RC_PER_BLOCK,VC_RC_DIST_TRUEMSE,VC_RC_MODEL_PROBE,20.0,0};
        u32 nu=vc_rc_count_units(d,h,w,VC_RC_PER_BLOCK);
        vc_rc_unit *u=malloc((size_t)nu*sizeof(vc_rc_unit)); vc_rc_result res;
        vc_rc_allocate(vol,d,h,w,&cfg,u,&res);
        printf("  variable : per-16^3 achieved %.2fx  mean-block true-MSE=%.2f -> PSNR proxy=%.2f dB\n",
               res.achieved_ratio, res.true_mse, psnr_of(res.true_mse));
        printf("  (PSNR proxy is the mean per-block decode-MSE; allocator minimizes total distortion at rate.)\n");
        free(u);
    }
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
