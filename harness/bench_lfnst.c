// LFNST-style low-frequency secondary transform bake-off (PLAN §2 "Transform",
// round-3 lit experiment). THROWAWAY self-contained bench (does NOT touch
// codec.c / ratectrl / chunkmodel): it #includes the WON transform (integer
// DCT-16^3) and the 16^3 per-coefficient quant matrix directly, links the WON
// entropy coder (RLGR) + the metric bundle.
//
// IDEA (from VVC's LFNST). After the separable primary DCT-16^3 of each atom,
// the low-frequency CORNER of the coefficient cube still carries correlated
// energy the separable transform cannot pack (the separable basis is suboptimal
// for diagonal/oriented low-freq structure). Apply ONE small FIXED non-separable
// secondary transform to just the lowest N^3 sub-block of coefficients (N=4 ->
// 64-dim, N=8 -> 512-dim) to decorrelate it further before quant. Inverse is the
// transpose. Block-local: touches only the corner of ONE 16^3 atom -> 16^3
// random access (touched=1) is preserved exactly.
//
// MATRIX: a KLT derived from the EMPIRICAL covariance of the corner DCT coeffs on
// the hires data (the strongest possible fixed matrix; "if time permits, derive
// from covariance" — done). Orthonormal => inverse == transpose. Fixed table, no
// per-atom side info beyond an optional 1-bit on/off flag (RD-switched arm).
//
// MEASURE: ratio-at-equal-quality (PSNR-matched) gain vs LFNST-OFF, on real
// PHerc Paris 4 hires-256 + coarse-256, across q in {16,32,64,128}. Cost:
// enc/dec MB/s, random-access impact (none — block-local), GMSD/edge-MAE/seam.
//
// Pure C23, libc/libm. Hot kernels straight-line, but this is a THROWAWAY bench
// (encode-time analysis) so absolute speed is informative, not load-bearing.
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

// --- the per-coefficient 16^3 quant matrix --------------------------------
#include "../src/quant/qmatrix16.c"

// --- the won entropy coder (RLGR) -----------------------------------------
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

#define A   16u
#define AVOX 4096u

// Baseline quant tuning per the current stack description (dz=0.80,
// recon-offset=0.40, HF slope ~0.5). Applied IDENTICALLY to both arms so the
// only variable is the secondary transform.
#define DZ        0.80f
#define RECON_OFF 0.40f
#define HF_SLOPE  0.50f

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

// ---- dead-zone quant with width DZ and recon offset RECON_OFF -------------
static inline void dz_quant_block(i16 *restrict qb, const i16 *restrict coef,
                                  const f32 *restrict step){
    for(u32 i=0;i<AVOX;++i){
        f32 c=(f32)coef[i]; f32 a=c<0.f?-c:c;
        f32 inv=1.0f/step[i];
        i32 m=(a >= DZ*step[i]);
        i32 lvl = m * ((i32)((a*inv) - DZ) + 1);
        if(lvl<0) lvl=0;
        qb[i]=(i16)(c<0.f?-lvl:lvl);
    }
}
static inline void dz_dequant_block(i16 *restrict coef, const i16 *restrict qb,
                                    const f32 *restrict step){
    for(u32 i=0;i<AVOX;++i){
        i32 l=(i32)qb[i]; i32 al=l<0?-l:l;
        f32 r = al ? ((f32)al + RECON_OFF) * step[i] : 0.f;
        i32 v=(i32)lrintf(l<0?-r:r);
        if(v>32767)v=32767; else if(v<-32768)v=-32768;
        coef[i]=(i16)v;
    }
}

// Build HF-protecting step matrix with slope HF_SLOPE (no content adapt).
static void build_step(f32 *restrict step, f32 base){
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        f32 t=(f32)(z+y+x)/45.0f;
        f32 wgt=1.0f - HF_SLOPE*t;
        f32 s=base*wgt; if(s<0.5f)s=0.5f;
        step[(size_t)z*256u+(size_t)y*16u+x]=s;
    }
}

// ============================ LFNST ========================================
// N = corner edge (4 or 8). DIM = N^3. The secondary transform is a DIM x DIM
// orthonormal matrix LF[DIM][DIM] applied to the flattened corner coefficient
// vector (corner-local layout: index = (z*N+y)*N+x). Forward: y = LF * x.
// Inverse: x = LF^T * y.
static u32   g_N   = 4;
static u32   g_DIM = 64;
static int   g_lfnst_on = 0;            // 0 = off (baseline), 1 = on
static float *g_LF = NULL;              // [DIM*DIM]

// flatten the low-freq corner of a 16^3 coef block into a DIM vector
static inline void corner_gather(const i16 *coef, float *v){
    u32 N=g_N;
    for(u32 z=0;z<N;++z)for(u32 y=0;y<N;++y)for(u32 x=0;x<N;++x)
        v[(z*N+y)*N+x] = (float)coef[(size_t)z*256u + (size_t)y*16u + x];
}
static inline void corner_scatter(i16 *coef, const float *v){
    u32 N=g_N;
    for(u32 z=0;z<N;++z)for(u32 y=0;y<N;++y)for(u32 x=0;x<N;++x){
        float f=v[(z*N+y)*N+x];
        i32 q=(i32)lrintf(f);
        if(q>32767)q=32767; else if(q<-32768)q=-32768;
        coef[(size_t)z*256u + (size_t)y*16u + x]=(i16)q;
    }
}
static inline void lfnst_fwd(i16 *coef){
    u32 D=g_DIM; float in[512], out[512];
    corner_gather(coef,in);
    for(u32 k=0;k<D;++k){ float a=0.f; const float *row=g_LF+(size_t)k*D;
        for(u32 j=0;j<D;++j) a+=row[j]*in[j]; out[k]=a; }
    corner_scatter(coef,out);
}
static inline void lfnst_inv(i16 *coef){
    u32 D=g_DIM; float in[512], out[512];
    corner_gather(coef,in);
    // x = LF^T * y
    for(u32 j=0;j<D;++j){ float a=0.f;
        for(u32 k=0;k<D;++k) a+=g_LF[(size_t)k*D+j]*in[k]; out[j]=a; }
    corner_scatter(coef,out);
}

// ---- KLT derivation: collect corner covariance over all atoms, eigendecomp --
// Jacobi eigenvalue algorithm for a symmetric DxD matrix. Produces eigenvectors
// in columns of V; we then order by descending eigenvalue and store row-major as
// LF (each row = one eigenvector = one secondary "basis" component).
static void jacobi_eig(double *a, int n, double *eval, double *evec){
    for(int i=0;i<n;++i)for(int j=0;j<n;++j) evec[i*n+j]=(i==j)?1.0:0.0;
    for(int sweep=0; sweep<100; ++sweep){
        double off=0; for(int p=0;p<n;++p)for(int q=p+1;q<n;++q) off+=a[p*n+q]*a[p*n+q];
        if(off<1e-20) break;
        for(int p=0;p<n;++p)for(int q=p+1;q<n;++q){
            double apq=a[p*n+q]; if(fabs(apq)<1e-300) continue;
            double app=a[p*n+p], aqq=a[q*n+q];
            double phi=0.5*atan2(2*apq, aqq-app);
            double c=cos(phi), s=sin(phi);
            for(int k=0;k<n;++k){
                double akp=a[k*n+p], akq=a[k*n+q];
                a[k*n+p]=c*akp - s*akq; a[k*n+q]=s*akp + c*akq;
            }
            for(int k=0;k<n;++k){
                double apk=a[p*n+k], aqk=a[q*n+k];
                a[p*n+k]=c*apk - s*aqk; a[q*n+k]=s*apk + c*aqk;
            }
            for(int k=0;k<n;++k){
                double vkp=evec[k*n+p], vkq=evec[k*n+q];
                evec[k*n+p]=c*vkp - s*vkq; evec[k*n+q]=s*vkp + c*vkq;
            }
        }
    }
    for(int i=0;i<n;++i) eval[i]=a[i*n+i];
}

// Derive LFNST KLT from a volume: DCT each atom, accumulate corner covariance,
// eigendecompose, store basis (rows = eigenvectors, descending eigenvalue).
static void derive_klt(const u8 *vol,u32 d,u32 h,u32 w,u32 N){
    g_N=N; g_DIM=N*N*N;
    u32 D=g_DIM;
    double *cov = calloc((size_t)D*D,sizeof(double));
    double *mean= calloc(D,sizeof(double));
    u64 cnt=0;
    u32 naz=d/A,nay=h/A,nax=w/A;
    u8 srcblk[AVOX]; i16 coef[AVOX]; float v[512];
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
        H_dct16_fwd(coef,srcblk,dc);
        // exclude the DC coefficient (index 0) influence? keep it; the secondary
        // transform on the corner will just pass DC through largely. To avoid DC
        // dominating the covariance we zero coef[0] for the stats only.
        i16 saved=coef[0]; coef[0]=0;
        corner_gather(coef,v); coef[0]=saved;
        for(u32 i=0;i<D;++i) mean[i]+=v[i];
        for(u32 i=0;i<D;++i)for(u32 j=i;j<D;++j) cov[i*D+j]+=(double)v[i]*v[j];
        cnt++;
    }
    for(u32 i=0;i<D;++i) mean[i]/=(double)cnt;
    for(u32 i=0;i<D;++i)for(u32 j=i;j<D;++j){
        double c=cov[i*D+j]/(double)cnt - mean[i]*mean[j];
        cov[i*D+j]=c; cov[j*D+i]=c;
    }
    double *eval=malloc(D*sizeof(double));
    double *evec=malloc((size_t)D*D*sizeof(double));
    jacobi_eig(cov,(int)D,eval,evec);
    // order indices by descending eigenvalue
    int *ord=malloc(D*sizeof(int)); for(u32 i=0;i<D;++i) ord[i]=(int)i;
    for(u32 i=0;i<D;++i)for(u32 j=i+1;j<D;++j) if(eval[ord[j]]>eval[ord[i]]){int t=ord[i];ord[i]=ord[j];ord[j]=t;}
    g_LF=realloc(g_LF,(size_t)D*D*sizeof(float));
    for(u32 k=0;k<D;++k){ int e=ord[k]; for(u32 j=0;j<D;++j) g_LF[(size_t)k*D+j]=(float)evec[j*D+e]; }
    free(cov);free(mean);free(eval);free(evec);free(ord);
}

// ---- code one atom; returns RLGR payload bytes (+2 DC, +sideflag bits later) -
// lfnst_mode: 0=off, 1=always-on, 2=RD-switched (try both, keep smaller, 1 side bit)
static size_t code_atom(const u8 *src, const f32 *step, u8 *rec, u8 *scratch, int lfnst_mode){
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=src[i];
    i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX;
    i16 *coef=(i16*)scratch;
    i16 *qb  =coef+AVOX;
    u8  *tmp =(u8*)(qb+AVOX);
    H_dct16_fwd(coef,src,dc);

    if(lfnst_mode==1){
        i16 c2[AVOX]; memcpy(c2,coef,sizeof(c2));
        lfnst_fwd(c2);
        dz_quant_block(qb,c2,step);
        size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
        dz_dequant_block(c2,qb,step);
        lfnst_inv(c2);
        H_dct16_inv(rec,c2,dc);
        return bytes+2;
    }
    if(lfnst_mode==2){
        // arm A: off
        i16 qa[AVOX]; dz_quant_block(qa,coef,step);
        size_t ba=vc_rlgr_encode(tmp,AVOX*3,qa,AVOX);
        // arm B: on
        i16 c2[AVOX]; memcpy(c2,coef,sizeof(c2)); lfnst_fwd(c2);
        i16 qbb[AVOX]; dz_quant_block(qbb,c2,step);
        size_t bb=vc_rlgr_encode(tmp,AVOX*3,qbb,AVOX);
        if(bb<ba){
            dz_dequant_block(c2,qbb,step); lfnst_inv(c2); H_dct16_inv(rec,c2,dc);
            return bb+2; // (the +1 bit/atom side flag accounted in caller as bits)
        } else {
            i16 cc[AVOX]; dz_dequant_block(cc,qa,step); H_dct16_inv(rec,cc,dc);
            return ba+2;
        }
    }
    // off
    dz_quant_block(qb,coef,step);
    size_t bytes=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
    dz_dequant_block(coef,qb,step);
    H_dct16_inv(rec,coef,dc);
    return bytes+2;
}

static size_t pass(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,int lfnst_mode,
                   u8 *rec,u8 *scratch,double *enc_sec){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX]; build_step(step,base_step);
    u8 srcblk[AVOX], recblk[AVOX];
    size_t total=0; u64 natoms=0;
    double t0=now_sec();
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        total += code_atom(srcblk,step,recblk,scratch,lfnst_mode);
        scatter_atom(rec,h,w,az,ay,ax,recblk);
        natoms++;
    }
    *enc_sec=now_sec()-t0;
    if(lfnst_mode==2) total += (natoms+7)/8;   // 1 on/off bit per atom
    return total;
}

typedef struct { double psnr,gmsd,edge_mae,seam,ratio,enc_mbs,dec_mbs; } row_t;

static void measure(const u8 *vol,const u8 *rec,u32 d,u32 h,u32 w,size_t bytes,double enc,row_t *o){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    o->psnr=m.psnr;
    o->gmsd=vc_gmsd(vol,rec,d,h,w);
    o->edge_mae=vc_edge_mae(vol,rec,d,h,w);
    o->seam=vc_seam_step(vol,rec,d,h,w,16);
    o->ratio=(double)((size_t)d*h*w)/(double)bytes;
    o->enc_mbs= enc>0 ? ((size_t)d*h*w)/1e6/enc : 0;
}

// bisect base_step so a pass hits a target PSNR (equal-quality comparison).
static f32 find_step_for_psnr(const u8 *vol,u32 d,u32 h,u32 w,int lfnst_mode,
                              double target_psnr,u8 *rec,u8 *scratch){
    f32 lo=0.5f, hi=400.f; double es;
    for(int it=0;it<22;++it){
        f32 mid=sqrtf(lo*hi);
        pass(vol,d,h,w,mid,lfnst_mode,rec,scratch,&es);
        vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
        // higher step -> lower psnr
        if(m.psnr>target_psnr) lo=mid; else hi=mid;
        if(fabs(m.psnr-target_psnr)<0.03) return mid;
    }
    return sqrtf(lo*hi);
}

// measure pure decode speed for a mode at a given step
static double dec_speed(const u8 *vol,u32 d,u32 h,u32 w,f32 base_step,int lfnst_mode,u8 *scratch){
    // encode all atoms into a payload buffer, then time decode-only.
    u32 naz=d/A,nay=h/A,nax=w/A; u64 natoms=(u64)naz*nay*nax;
    f32 step[AVOX]; build_step(step,base_step);
    // store per-atom: dc(i16) + len(u32) + flag(u8) + payload
    u8 **pl=malloc(natoms*sizeof(u8*)); u32 *plen=malloc(natoms*sizeof(u32));
    i32 *dcs=malloc(natoms*sizeof(i32)); u8 *flags=malloc(natoms);
    u8 srcblk[AVOX];
    u64 k=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax,++k){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=srcblk[i];
        i32 dc=(sum+(i32)(AVOX/2))/(i32)AVOX; dcs[k]=dc;
        i16 coef[AVOX]; H_dct16_fwd(coef,srcblk,dc);
        int use=lfnst_mode;
        if(lfnst_mode==2){ use=1; } // for dec-speed just measure on-path cost
        i16 c2[AVOX]; memcpy(c2,coef,sizeof(c2));
        if(use==1) lfnst_fwd(c2);
        i16 qb[AVOX]; dz_quant_block(qb,c2,step);
        u8 tmp[AVOX*3]; size_t b=vc_rlgr_encode(tmp,AVOX*3,qb,AVOX);
        pl[k]=malloc(b); memcpy(pl[k],tmp,b); plen[k]=(u32)b; flags[k]=(u8)(use==1);
    }
    // time decode-only
    u8 recblk[AVOX]; i16 coef[AVOX], qb[AVOX];
    double t0=now_sec();
    for(u64 i=0;i<natoms;++i){
        vc_rlgr_decode(qb,AVOX,pl[i],plen[i]);
        dz_dequant_block(coef,qb,step);
        if(flags[i]) lfnst_inv(coef);
        H_dct16_inv(recblk,coef,dcs[i]);
    }
    double dt=now_sec()-t0;
    (void)recblk;
    for(u64 i=0;i<natoms;++i) free(pl[i]);
    free(pl);free(plen);free(dcs);free(flags);
    return ((size_t)d*h*w)/1e6/dt;
}

static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w;
    u8 *rec=malloc(raw);
    u8 *scratch=malloc(AVOX*sizeof(i16)*2 + AVOX*3);
    printf("\n## %s  (%ux%ux%u, atom=16^3, LFNST corner=%u^3 KLT)\n",label,d,h,w,g_N);

    // q knob -> base step (the codec's q maps roughly linearly to dead-zone step)
    const int qs[]={16,32,64,128};
    printf("\nEqual-QUALITY (PSNR-matched) ratio gain from LFNST. dz=%.2f off=%.2f hfslope=%.2f\n",DZ,RECON_OFF,HF_SLOPE);
    printf(" q  | tgtPSNR | OFF ratio | ON ratio | RDsw ratio | gain(ON) | gain(RDsw) | GMSD off->on | edgeMAE | seam\n");
    printf("----+---------+-----------+----------+------------+----------+------------+--------------+---------+------\n");
    for(int qi=0;qi<4;++qi){
        f32 base=(f32)qs[qi];
        // establish target PSNR from the OFF pass at this q
        double es;
        pass(vol,d,h,w,base,0,rec,scratch,&es);
        vc_metrics m0; vc_compute_metrics(vol,rec,d,h,w,&m0);
        double tgt=m0.psnr;
        // OFF at exactly this step:
        size_t by_off=pass(vol,d,h,w,base,0,rec,scratch,&es); row_t roff; measure(vol,rec,d,h,w,by_off,es,&roff);
        double goff=roff.gmsd;
        // ON: bisect step to match the SAME PSNR, then read ratio
        f32 son=find_step_for_psnr(vol,d,h,w,1,tgt,rec,scratch);
        size_t by_on=pass(vol,d,h,w,son,1,rec,scratch,&es); row_t ron; measure(vol,rec,d,h,w,by_on,es,&ron);
        // RDsw: bisect to same PSNR
        f32 srd=find_step_for_psnr(vol,d,h,w,2,tgt,rec,scratch);
        size_t by_rd=pass(vol,d,h,w,srd,2,rec,scratch,&es); row_t rrd; measure(vol,rec,d,h,w,by_rd,es,&rrd);
        double gain_on  = (ron.ratio/roff.ratio - 1.0)*100.0;
        double gain_rd  = (rrd.ratio/roff.ratio - 1.0)*100.0;
        printf("%3d | %7.3f | %8.2fx | %7.2fx | %9.2fx | %+6.2f%% | %+8.2f%% | %.5f->%.5f | %6.3f | %5.3f\n",
               qs[qi],tgt,roff.ratio,ron.ratio,rrd.ratio,gain_on,gain_rd,goff,ron.gmsd,ron.edge_mae,ron.seam);
    }
    // speed: enc/dec MB/s off vs on at q=32
    double es; size_t b=pass(vol,d,h,w,32.f,0,rec,scratch,&es); (void)b;
    double enc_off=((size_t)d*h*w)/1e6/es;
    b=pass(vol,d,h,w,32.f,1,rec,scratch,&es); double enc_on=((size_t)d*h*w)/1e6/es;
    double dec_off=dec_speed(vol,d,h,w,32.f,0,scratch);
    double dec_on =dec_speed(vol,d,h,w,32.f,1,scratch);
    printf("\nspeed @ q32: enc OFF %.0f MB/s -> ON %.0f MB/s | dec OFF %.0f MB/s -> ON %.0f MB/s\n",
           enc_off,enc_on,dec_off,dec_on);
    free(rec); free(scratch);
}

int main(int argc,char**argv){
    u32 N = (argc>=2)? (u32)atoi(argv[1]) : 4;
    if(N!=4 && N!=8) N=4;
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    // derive KLT from hires (the ink/fiber-rich corpus) and reuse for both.
    size_t len; u8 *hv=read_file(files[0],&len);
    if(!hv){fprintf(stderr,"missing hires (run from repo root)\n");return 1;}
    printf("# LFNST low-frequency secondary transform — KLT corner=%u^3 derived from hires\n",N);
    derive_klt(hv,256,256,256,N);
    free(hv);
    for(int i=0;i<2;++i){ u8 *v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
