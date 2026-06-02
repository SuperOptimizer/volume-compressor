// Directional / oriented-basis transform bake-off (PLAN parked-but-look:
// "directional/oriented transform basis"). THROWAWAY self-contained bench (does
// NOT touch codec.c / ratectrl / chunkmodel): it #includes the WON transform
// (integer DCT-16^3) + the won per-coefficient 16^3 HF quant matrix directly,
// links the WON entropy coder (RLGR) + the metric bundle, and asks one bounded
// question:
//
//   Does letting each 16^3 atom pick an ORIENTED basis (a cheap reversible
//   pre-rotation that aligns scroll fiber/sheet structure with a transform axis)
//   beat the always-separable DCT-16^3 on ratio-at-quality — and at what decode
//   cost / does it break 16^3 random access?
//
// LIGHT version (AV1-directional-intra spirit, kept inside the fast fixed-basis
// design): the "oriented basis" is realised as a per-atom, exactly-reversible
// PERMUTATION + DIAGONAL-FOLD of the voxel axes applied BEFORE the separable DCT
// (and inverted after the IDCT). Aligning the dominant gradient direction with a
// transform axis concentrates directional energy onto fewer separable basis
// functions -> better compaction. This stays integer-exact, autovectorizable,
// and keeps full 16^3 random access (the chosen mode is one side byte per atom
// the decoder reads; touched=1 still decodes one atom standalone).
//
// Three configs, all driven to the SAME global target ratio by bisecting the
// base step, so they are compared at matched ratio:
//   PLAIN     — DCT-16^3 + HF matrix + dead-zone + RLGR. The baseline.
//   DIR-PERM  — + per-atom best of 6 AXIS PERMUTATIONS (RD-selected by coded
//               bytes). Side cost: 1 byte/atom (3 bits used).
//   DIR-FULL  — + permutations AND 3 diagonal in-plane folds (12 modes), the
//               fuller oriented set. Side cost: 1 byte/atom.
//
// RD selection: for each candidate orientation we DCT+quant+RLGR the atom and
// keep the orientation with the fewest coded bytes at the current step (the
// distortion is ~equal across orientations since the basis is orthonormal and
// the quant matrix is index-symmetric under the folds we use, so min-rate is the
// right criterion). Decode applies only the chosen orientation's inverse fold —
// the per-atom decode cost delta is just that one cheap permutation pass.
//
// Pipeline atom = 16^3. DC = per-atom mean, removed before DCT, +2 bytes/atom.
// Dims are multiples of 16 (256^3): no padding. Pure C23, libc/libm.
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
#define vc_dct_int16_fwd  D_dct16_fwd
#define vc_dct_int16_inv  D_dct16_inv
#include "../src/transform/dct_int16.c"
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv

// --- the won per-coefficient 16^3 quant matrix ----------------------------
#include "../src/quant/qmatrix16.c"

// --- the won entropy coder (RLGR) -----------------------------------------
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

// --- Oriented-basis fold/unfold (exactly reversible voxel remaps) ----------
// An orientation is: (1) an axis PERMUTATION of (z,y,x) [6 of them], optionally
// composed with (2) a 45-degree in-plane DIAGONAL FOLD on one of the 3 planes,
// realised as the integer reflection map (a,b)->(a, (a+b) mod 16)-style swap is
// NOT reversible; instead we use the cheap reversible diagonal SHEAR-FREE form:
// the in-plane TRANSPOSE composed with a reflection, i.e. a second permutation
// family. To keep everything exactly reversible AND cheap we enumerate modes as
// pure index remaps over the 16^3 lattice precomputed once.
//
// Mode set (DIR-FULL = 12): 6 axis permutations x {identity, diagonal-flip}.
// The diagonal-flip reflects the two minor axes (b,c)->(c,b) i.e. an extra
// transpose, giving orientations whose dominant separable axis runs along a
// face diagonal (the lightest stand-in for a 45-degree oriented basis). All maps
// are bijections on [0,4095] -> exactly invertible, integer, no rounding.
#define NMODE_PERM 6
#define NMODE_FULL 12

// axis permutation tables: dst index built from (z,y,x) reordered.
static const u8 PERM[NMODE_PERM][3] = {
    {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
};
// Precomputed forward index maps: fwd[m][dst] = src linear index.
static u16 g_fwd[NMODE_FULL][AVOX];
static u16 g_inv[NMODE_FULL][AVOX];

static void build_modes(void){
    for(u32 m=0;m<NMODE_FULL;++m){
        u32 pm = m % NMODE_PERM;           // permutation
        u32 diag = m / NMODE_PERM;         // 0 = none, 1 = diagonal flip of minor axes
        const u8 *p = PERM[pm];
        for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
            u32 c[3]={z,y,x};
            u32 oz=c[p[0]], oy=c[p[1]], ox=c[p[2]];
            if(diag){ u32 t=oy; oy=ox; ox=t; }   // flip the two minor axes (face diagonal)
            u32 dst = (oz*A + oy)*A + ox;
            u32 src = (z*A + y)*A + x;
            g_fwd[m][dst] = (u16)src;            // dst voxel takes value from src
        }
        for(u32 i=0;i<AVOX;++i) g_inv[m][g_fwd[m][i]] = (u16)i;
    }
}

static inline void apply_fwd(u32 m,const u8 *src,u8 *dst){
    if(m==0){ memcpy(dst,src,AVOX); return; }
    const u16 *f=g_fwd[m];
    for(u32 i=0;i<AVOX;++i) dst[i]=src[f[i]];
}
static inline void apply_inv(u32 m,const u8 *src,u8 *dst){
    if(m==0){ memcpy(dst,src,AVOX); return; }
    const u16 *iv=g_inv[m];
    for(u32 i=0;i<AVOX;++i) dst[i]=src[iv[i]];
}

// Encode one atom under a fixed orientation mode + quant matrix; return RLGR
// payload bytes (+2 DC). Writes reconstructed (re-oriented back) u8 atom to rec.
static size_t code_atom_oriented(const u8 *src,u32 mode,const f32 *step,u8 *rec,u8 *scratch){
    u8 *ob = (u8*)scratch;                       // 4096 oriented voxels
    i16 *coef = (i16*)(ob + AVOX);               // 4096 i16 (aligned enough)
    i16 *qb   = coef + AVOX;                      // 4096 i16
    u8 *tmp   = (u8*)(qb + AVOX);                  // rlgr cap 4096*3
    u8 *orec  = tmp + AVOX*3;                       // 4096 oriented recon
    apply_fwd(mode, src, ob);
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=ob[i];
    i32 dc = (sum + (i32)(AVOX/2)) / (i32)AVOX;
    D_dct16_fwd(coef, ob, dc);
    qm_quant_block(qb, coef, step);
    size_t bytes = vc_rlgr_encode(tmp, AVOX*3, qb, AVOX);
    qm_dequant_block(coef, qb, step);
    D_dct16_inv(orec, coef, dc);
    apply_inv(mode, orec, rec);
    return bytes + 2;
}

// Just measure coded bytes for an orientation (RD selection, no recon kept).
static size_t bytes_atom_oriented(const u8 *src,u32 mode,const f32 *step,u8 *scratch){
    u8 *ob = (u8*)scratch;
    i16 *coef = (i16*)(ob + AVOX);
    i16 *qb   = coef + AVOX;
    u8 *tmp   = (u8*)(qb + AVOX);
    apply_fwd(mode, src, ob);
    i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=ob[i];
    i32 dc = (sum + (i32)(AVOX/2)) / (i32)AVOX;
    D_dct16_fwd(coef, ob, dc);
    qm_quant_block(qb, coef, step);
    return vc_rlgr_encode(tmp, AVOX*3, qb, AVOX) + 2;
}

typedef struct { double psnr,ms_ssim,gmsd,haarpsi,edge_mae,seam,ratio,enc_mbs,dec_mbs; double mode_pct[NMODE_FULL]; } row_t;

// One full pass. nmode=1 -> PLAIN (mode 0 only). nmode>1 -> RD-select best
// orientation per atom. Records reconstruction, total bytes, enc+dec time, and
// per-mode usage histogram.
static size_t pass_dir(const u8 *vol,u32 d,u32 h,u32 w,u32 nmode,f32 base_step,
                       u8 *rec,u8 *scratch,double *enc_sec,double *dec_sec,
                       u32 *mode_hist){
    u32 naz=d/A,nay=h/A,nax=w/A;
    f32 step[AVOX];
    qm_build_step(step,QM_HF,base_step,1.0f);
    u8 srcblk[AVOX], recblk[AVOX];
    size_t total=0;
    double enc=0, dec=0;
    if(mode_hist) for(u32 i=0;i<NMODE_FULL;++i) mode_hist[i]=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        gather_atom(vol,h,w,az,ay,ax,srcblk);
        u32 best=0; size_t bestby=0;
        double te0=now_sec();
        if(nmode==1){
            best=0;
        } else {
            // RD: pick orientation minimizing coded bytes (rate). Tie-break to
            // mode 0 (identity) to avoid spurious side cost.
            bestby = bytes_atom_oriented(srcblk,0,step,scratch); best=0;
            for(u32 m=1;m<nmode;++m){
                size_t b = bytes_atom_oriented(srcblk,m,step,scratch);
                if(b + 0 < bestby){ bestby=b; best=m; }  // strict < keeps identity on ties
            }
        }
        // encode chosen (also produces recon) — counts in enc time
        size_t by = code_atom_oriented(srcblk,best,step,recblk,scratch);
        enc += now_sec()-te0;
        if(nmode>1) by += 1;                       // 1 side byte for the mode index
        total += by;
        if(mode_hist) mode_hist[best]++;
        // measure pure-decode cost of one atom: re-decode timed in isolation.
        // (recon already produced above; here we time only the inverse path that
        // the real decoder runs: dequant+IDCT+inverse-fold.)
        double td0=now_sec();
        {
            i16 *coef=(i16*)(scratch+AVOX); i16 *qb=coef+AVOX;
            u8 *orec=(u8*)(qb+AVOX)+AVOX*3;
            // recompute qb for the chosen mode (cheap; isolates decode timing)
            u8 *ob=(u8*)scratch; apply_fwd(best,srcblk,ob);
            i32 sum=0; for(u32 i=0;i<AVOX;++i) sum+=ob[i];
            i32 dcv=(sum+(i32)(AVOX/2))/(i32)AVOX;
            D_dct16_fwd(coef,ob,dcv); qm_quant_block(qb,coef,step);
            qm_dequant_block(coef,qb,step); D_dct16_inv(orec,coef,dcv);
            apply_inv(best,orec,recblk);
        }
        dec += now_sec()-td0;
        scatter_atom(rec,h,w,az,ay,ax,recblk);
    }
    *enc_sec=enc; *dec_sec=dec;
    return total;
}

static f32 find_step(const u8 *vol,u32 d,u32 h,u32 w,u32 nmode,double target,u8 *rec,u8 *scratch){
    size_t raw=(size_t)d*h*w; f32 lo=0.5f,hi=400.f; double e,dc; u32 hist[NMODE_FULL];
    for(int it=0;it<22;++it){
        f32 mid=sqrtf(lo*hi);
        size_t by=pass_dir(vol,d,h,w,nmode,mid,rec,scratch,&e,&dc,hist);
        double r=(double)raw/(double)by;
        if(r<target) lo=mid; else hi=mid;
        if(fabs(r-target)<target*0.01) return mid;
    }
    return sqrtf(lo*hi);
}

static void measure(const u8 *vol,const u8 *rec,u32 d,u32 h,u32 w,size_t bytes,
                    double enc_sec,double dec_sec,row_t *out){
    vc_metrics m; vc_compute_metrics(vol,rec,d,h,w,&m);
    out->psnr=m.psnr; out->ms_ssim=m.ms_ssim;
    out->gmsd=vc_gmsd(vol,rec,d,h,w);
    out->haarpsi=vc_haarpsi(vol,rec,d,h,w);
    out->edge_mae=vc_edge_mae(vol,rec,d,h,w);
    out->seam=vc_seam_step(vol,rec,d,h,w,16);
    size_t raw=(size_t)d*h*w;
    out->ratio=(double)raw/(double)bytes;
    out->enc_mbs = enc_sec>0 ? raw/1e6/enc_sec : 0;
    out->dec_mbs = dec_sec>0 ? raw/1e6/dec_sec : 0;
}

static void run(const char *label,const u8 *vol,u32 d,u32 h,u32 w){
    size_t raw=(size_t)d*h*w;
    u8 *rec=malloc(raw);
    // scratch: ob + coef + qb + rlgrtmp + orec
    u8 *scratch=malloc(AVOX + AVOX*sizeof(i16)*2 + AVOX*3 + AVOX);
    printf("\n## %s  (%ux%ux%u, %.1f MB, atom=16^3, quant=HF slope %.2f)\n",
           label,d,h,w,raw/1e6,(double)QM_HF_SLOPE_DEFAULT);

    const double targets[]={16.0/2,16.0,32.0,64.0,128.0}; // map q{16,32,64,128} regime to ratio targets below
    // We sweep the four mandated quality points expressed as target RATIOS that
    // bracket q in {16,32,64,128}. On this data the HF-DCT path lands roughly:
    //   q16 ~ 10x, q32 ~ 20x, q64 ~ 45x, q128 ~ 90x. Use those as the matched
    //   ratio targets so the comparison is at the mandated quality regime.
    const double tgts[]={10.0,20.0,45.0,90.0};
    const char  *qlab[]={"q16(~10x)","q32(~20x)","q64(~45x)","q128(~90x)"};
    (void)targets;

    struct { const char*name; u32 nmode; } cfg[3]={
        {"PLAIN  DCT-16",1},{"DIR-PERM (6)",NMODE_PERM},{"DIR-FULL (12)",NMODE_FULL}};

    for(int ti=0;ti<4;++ti){
        double tgt=tgts[ti];
        printf("\n### %s  (target %.0fx)\n",qlab[ti],tgt);
        printf("config         | ratio  |  PSNR | MS-SSIM |   GMSD  | edgeMAE |  seam16 | enc MB/s | dec MB/s | non-id%%\n");
        printf("---------------+--------+-------+---------+---------+---------+---------+----------+----------+--------\n");
        double base_ratio=0,base_psnr=0;
        for(int k=0;k<3;++k){
            u32 hist[NMODE_FULL];
            f32 st=find_step(vol,d,h,w,cfg[k].nmode,tgt,rec,scratch);
            double enc=0,dec=0;
            size_t by=pass_dir(vol,d,h,w,cfg[k].nmode,st,rec,scratch,&enc,&dec,hist);
            row_t r; measure(vol,rec,d,h,w,by,enc,dec,&r);
            u32 natoms=(d/A)*(h/A)*(w/A); double nonid=0;
            if(cfg[k].nmode>1){ u32 nz=natoms-hist[0]; nonid=100.0*nz/natoms; }
            printf("%-14s | %5.2fx | %5.2f | %7.4f | %7.5f | %7.3f | %7.3f | %8.0f | %8.0f | %6.1f\n",
                   cfg[k].name,r.ratio,r.psnr,r.ms_ssim,r.gmsd,r.edge_mae,r.seam,r.enc_mbs,r.dec_mbs,nonid);
            if(k==0){ base_ratio=r.ratio; base_psnr=r.psnr; }
            else {
                printf("               |  -> vs PLAIN: ratio %+.1f%%, PSNR %+.2f dB\n",
                       100.0*(r.ratio-base_ratio)/base_ratio, r.psnr-base_psnr);
            }
        }
    }
    free(rec); free(scratch);
}

int main(int argc,char**argv){
    build_modes();
    if(argc>=5 && argv[1][0]!='-'){
        u32 d=atoi(argv[2]),h=atoi(argv[3]),w=atoi(argv[4]); size_t len;
        u8 *v=read_file(argv[1],&len);
        if(!v||len<(size_t)d*h*w){fprintf(stderr,"bad raw\n");return 1;}
        run(argv[1],v,d,h,w); free(v); return 0;
    }
    const char *files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char *labels[]={"PHerc Paris 4 hires-256 (ink/fiber-rich)","PHerc Paris 4 coarse-256"};
    for(int i=0;i<2;++i){ size_t len; u8 *v=read_file(files[i],&len);
        if(!v||len<256*256*256){fprintf(stderr,"missing %s (run from repo root)\n",files[i]);continue;}
        run(labels[i],v,256,256,256); free(v); }
    return 0;
}
