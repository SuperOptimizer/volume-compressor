// Bit-plane / SNR-scalable coding bake-off (PARKED-axis re-examination).
//
// Measures the RATIO COST of an embedded, truncatable bit-plane coder vs the
// non-scalable table-free RLGR baseline, on the SAME quantized DCT-16^3
// coefficient atoms, on real PHerc Paris 4 data (refbuild hires_256 + coarse_256).
//
// Pipeline per 16^3 atom (the won stack): per-axis DCT-16 -> 16^3 freq scan +
// HF-protecting dead-zone quant (qmatrix16, slope 0.5, dz 0.80, recon-offset
// 0.40) -> entropy. We code each atom's freq-scanned level array two ways and
// compare total payload bytes:
//   (A) RLGR  (vc_rlgr_encode/decode)        — the measurable non-scalable baseline
//   (B) BITPLANE embedded coder (this expt)  — MSB->LSB, truncatable to any quality
//
// Both are round-trip-verified atom by atom (full stream). We report total bytes,
// compression ratio vs the raw u8 volume, the ratio PENALTY of bit-plane vs RLGR,
// reconstruction PSNR (identical for both — same quantizer), and enc/dec MB/s.
//
// Random-access note: the bit-plane stream is per-ATOM (4096 levels), so touched=1
// still holds (one atom decodes independently); scalability is WITHIN an atom's
// stream. Reported in BITPLANE_RESULTS.md.
//
// Build: added as a harness target in CMakeLists.txt. Run from the worktree root
// so the relative refbuild paths resolve.
// Drive the won DCT-16 on exactly ONE 16^3 atom: set its chunk side to 16 so a
// "chunk" == one atom in 16^3 raster layout (z*256+y*16+x), which matches the
// qmatrix16 step indexing exactly. (The DCT block engine loops 16^3 sub-blocks
// over a CS^3 chunk; CS=16 makes it a single-atom transform — the atom IS the
// codec unit for this experiment.)
#define VC_CHUNK_SIDE 16
#include "../src/transform/dct_int16.c"      // vc_dct_int16_fwd/inv (won transform)
#include "../src/quant/qmatrix16.c"          // qm_build_step / qm_quant_block / dequant
#include "../src/entropy/bitplane.c"         // vc_bitplane_encode/decode (this expt)
#include "../src/metrics/metrics.h"
#include "../include/vc/types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// RLGR baseline lives in rlgr.c with external linkage.
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static u8 *rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f)return 0;fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);u8*b=malloc(s);if(fread(b,1,s,f)!=(size_t)s){free(b);fclose(f);return 0;}fclose(f);*n=s;return b;}

#define A 16u
#define AVOX 4096u

// Build the low->high frequency scan order for a 16^3 block: sort coefficient
// linear indices (z*256+y*16+x) by z+y+x ascending. Precompute once.
static u32 g_scan[AVOX];
static void build_scan(void){
    // counting-sort by fsum (0..45)
    u32 cnt[46]={0};
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x) cnt[z+y+x]++;
    u32 off[46]; u32 acc=0; for(int s=0;s<46;++s){off[s]=acc;acc+=cnt[s];}
    u32 cur[46]; memcpy(cur,off,sizeof off);
    for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x){
        u32 s=z+y+x; u32 lin=z*256u+y*16u+x; g_scan[cur[s]++]=lin;
    }
}

typedef struct { double bytes; double psnr; double enc_mbps, dec_mbps; int rt_fail; } res;

// Encode the whole volume atom-by-atom with the chosen coder; round-trip verify.
typedef enum { CODER_RLGR=0, CODER_BP=1 } coder;

static res run(const u8*vol,u32 D,u32 H,u32 W,float base_step,coder cd,float hf_slope){
    res r={0,0,0,0,0};
    qm_set_hf_slope(hf_slope);
    f32 step[AVOX]; qm_build_step(step, QM_HF, base_step, 1.0f);
    u32 naz=(D+A-1)/A, nay=(H+A-1)/A, nax=(W+A-1)/A;

    i16 *coef=malloc(AVOX*sizeof(i16));
    i16 *qlev=malloc(AVOX*sizeof(i16));   // raster layout
    i16 *qscan=malloc(AVOX*sizeof(i16));  // freq-scanned
    i16 *qback=malloc(AVOX*sizeof(i16));
    i16 *qrec=malloc(AVOX*sizeof(i16));   // decoded raster
    u8  *vox=malloc(AVOX);
    u8  *rec=malloc(AVOX);
    u8  *pay=malloc(AVOX*4+64);
    u8  *recvol=calloc((size_t)D*H*W,1);

    double t_enc=0, t_dec=0; size_t total_bytes=0;
    for(u32 az=0;az<naz;++az)for(u32 ay=0;ay<nay;++ay)for(u32 ax=0;ax<nax;++ax){
        // gather atom (zero-pad partial — volumes are 256^3 = exact, no pad)
        for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
            size_t src=(((size_t)(az*A+z)*H)+(ay*A+y))*W+ax*A;
            memcpy(vox+(z*A+y)*A, vol+src, A);
        }
        // per-atom DC = rounded mean
        u64 sum=0; for(u32 i=0;i<AVOX;++i) sum+=vox[i];
        i32 dc=(i32)((sum+AVOX/2)/AVOX);
        vc_dct_int16_fwd(coef, vox, dc);
        qm_quant_block(qlev, coef, step);
        // freq scan
        for(u32 i=0;i<AVOX;++i) qscan[i]=qlev[g_scan[i]];

        double e0=now_sec(); size_t plen; u32 np;
        if(cd==CODER_RLGR) plen=vc_rlgr_encode(pay, AVOX*4+64, qscan, AVOX);
        else               plen=vc_bitplane_encode(pay, AVOX*4+64, qscan, AVOX, &np);
        t_enc+=now_sec()-e0;
        if(plen>AVOX*4+64){ r.rt_fail=1; }
        total_bytes+=plen;

        double d0=now_sec();
        if(cd==CODER_RLGR) vc_rlgr_decode(qback, AVOX, pay, plen);
        else               vc_bitplane_decode(qback, AVOX, pay, plen);
        t_dec+=now_sec()-d0;
        // verify exact level round-trip
        for(u32 i=0;i<AVOX;++i){ if(qback[i]!=qscan[i]){ r.rt_fail=1; } }
        // unscan -> raster, dequant, inverse DCT
        for(u32 i=0;i<AVOX;++i) qrec[g_scan[i]]=qback[i];
        qm_dequant_block(coef, qrec, step);
        vc_dct_int16_inv(rec, coef, dc);
        for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y){
            size_t dst=(((size_t)(az*A+z)*H)+(ay*A+y))*W+ax*A;
            memcpy(recvol+dst, rec+(z*A+y)*A, A);
        }
    }
    // metrics
    r.bytes=(double)total_bytes;
    double mse=0; size_t N=(size_t)D*H*W;
    for(size_t i=0;i<N;++i){ double d=(double)vol[i]-recvol[i]; mse+=d*d; }
    mse/=(double)N;
    r.psnr= mse>0? 10.0*log10(255.0*255.0/mse) : 99.0;
    double MB=(double)N/1e6;
    r.enc_mbps=MB/t_enc; r.dec_mbps=MB/t_dec;

    free(coef);free(qlev);free(qscan);free(qback);free(qrec);free(vox);free(rec);free(pay);free(recvol);
    return r;
}

int main(int argc,char**argv){
    build_scan();
    const char*files[]={"harness/refbuild/hires_256.u8","harness/refbuild/coarse_256.u8"};
    const char*names[]={"hires-256","coarse-256"};
    float qs[]={16,32,64,128};
    float hf_slope=0.5f;
    if(argc>1) hf_slope=atof(argv[1]);

    printf("# Bit-plane vs RLGR bake-off (HF slope %.2f, dz built into qmatrix)\n",hf_slope);
    printf("%-11s %5s | %-28s | %-28s | %8s\n","input","q","RLGR (baseline)","BITPLANE (scalable)","penalty");
    printf("%-11s %5s | %10s %7s %6s | %10s %7s %6s | %7s\n",
           "","","bytes","ratio","PSNR","bytes","ratio","PSNR","ratio%");

    for(int fi=0;fi<2;++fi){
        size_t len; u8*vol=rd(files[fi],&len);
        if(!vol){ printf("MISSING %s\n",files[fi]); continue; }
        u32 S=256; // refbuild volumes are 256^3
        if(len!=(size_t)S*S*S){ printf("BAD SIZE %s (%zu)\n",files[fi],len); free(vol); continue; }
        for(int qi=0;qi<4;++qi){
            res a=run(vol,S,S,S,qs[qi],CODER_RLGR,hf_slope);
            res b=run(vol,S,S,S,qs[qi],CODER_BP,hf_slope);
            double raw=(double)len;
            double ra=raw/a.bytes, rb=raw/b.bytes;
            double penalty=(a.bytes>0)?100.0*(b.bytes-a.bytes)/a.bytes:0;
            printf("%-11s %5.0f | %10.0f %6.1fx %5.1f%s | %10.0f %6.1fx %5.1f%s | %+6.2f%%\n",
                names[fi],qs[qi],
                a.bytes,ra,a.psnr, a.rt_fail?"!":" ",
                b.bytes,rb,b.psnr, b.rt_fail?"!":" ",
                penalty);
        }
        // speed (one representative q=32) printed once per input
        res sa=run(vol,S,S,S,32,CODER_RLGR,hf_slope);
        res sb=run(vol,S,S,S,32,CODER_BP,hf_slope);
        printf("# %-9s speed @q32: RLGR enc %.0f / dec %.0f MB/s | BP enc %.0f / dec %.0f MB/s\n",
               names[fi], sa.enc_mbps, sa.dec_mbps, sb.enc_mbps, sb.dec_mbps);
        free(vol);
    }
    return 0;
}
