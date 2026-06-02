// bench_1024 — FINAL 3-way benchmark: OURS (frozen vc) vs c3d vs c4d on a real
// 1024^3 PHerc Paris 4 region. Measures 5 axes per (codec,target-ratio):
//   (1) ratio achieved   (2) PSNR/MS-SSIM/GMSD   (3) decode MB/s
//   (4) encode MB/s (OURS: end-to-end incl. rate-search AND single-pass)
//   (5) time-to-first-16^3-block (cold random-access latency, microseconds)
//
// c4d is driven separately by the Python wrapper (CLI). This binary covers
// OURS + c3d (both C libraries). Emits one TSV row per (codec,target).
//
// Build (see harness/build_bench_1024.sh).
#define _POSIX_C_SOURCE 199309L
#define VC_BENCH 1
#include "src/vc/vc.h"
#include "src/metrics/metrics.h"
#include "c3d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// bench-only hooks declared in vc.c
vc_status vc_bench_encode_full(const uint8_t*, vc_dims, float, uint8_t**, size_t*, float[VC_NLOD], int*);
vc_status vc_bench_encode_singlepass(const uint8_t*, vc_dims, const float[VC_NLOD], int, uint8_t**, size_t*);
extern int vc_bench_max_lods; // 1 = native-resolution LOD0 only (no pyramid)

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

static unsigned char *read_file(const char *p, size_t n){
    FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
    unsigned char *b=malloc(n);
    if(fread(b,1,n,f)!=n){fprintf(stderr,"short read %s\n",p);exit(1);}
    fclose(f); return b;
}

// ---- metrics on full 1024^3 are O(N) and fine; MS-SSIM/GMSD are 2.5D slice means.
static void emit(const char *codec, double target, double ratio, double lod0_ratio,
                 double psnr, double mssim, double gmsd,
                 double enc_e2e, double enc_sp, double dec_mbs, double ttf_us){
    // TSV: codec target ratio lod0ratio psnr mssim gmsd enc_e2e enc_sp dec ttf_us
    printf("%s\t%.0f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.2f\t%.1f\t%.1f\t%.2f\n",
        codec,target,ratio,lod0_ratio,psnr,mssim,gmsd,enc_e2e,enc_sp,dec_mbs,ttf_us);
    fflush(stdout);
}

// Parse OURS archive footer to get member 0 (LOD0) compressed length.
static uint64_t ours_lod0_len(const unsigned char *arc, size_t alen){
    // trailer: [dir_off:u64][dir_len:u64][ver:u32][magic:u32] at end
    if(alen<24) return 0;
    uint64_t dir_off; memcpy(&dir_off, arc+alen-24, 8);
    if(dir_off+4>alen) return 0;
    uint32_t nm; memcpy(&nm, arc+dir_off, 4);
    if(nm<1) return 0;
    uint64_t len0; memcpy(&len0, arc+dir_off+4+8, 8); // first pair: rel_off(8),length(8)
    return len0;
}

int main(int argc, char**argv){
    const char *path = argv[1];
    int N = atoi(argv[2]);
    const char *which = argc>3?argv[3]:"all"; // "ours" | "c3d" | "all"
    vc_bench_max_lods = 1; // native-resolution LOD0 ONLY — LODs are independent,
                           // c3d/c4d don't build pyramids, so compare like-for-like.
    size_t n=(size_t)N*N*N;
    unsigned char *vol = read_file(path, n);
    double targets[]={10,20,50,100};
    int ntargets = argc>4 ? atoi(argv[4]) : 4;

    fprintf(stderr,"loaded %s %dx%dx%d (%.2f GB)\n",path,N,N,N,n/1e9);

    // ================= OURS =================
    if(!strcmp(which,"ours")||!strcmp(which,"all")){
        for(int ti=0; ti<ntargets; ti++){
            float qsel[VC_NLOD]; int nlod=0;
            unsigned char *arc=NULL; size_t alen=0;
            double t0=now();
            vc_status s=vc_bench_encode_full(vol,(vc_dims){N,N,N},(float)targets[ti],&arc,&alen,qsel,&nlod);
            double enc_e2e_t=now()-t0;
            if(s){fprintf(stderr,"ours enc fail %d\n",s);continue;}
            // single-pass encode at the q already chosen
            unsigned char *arc2=NULL; size_t alen2=0;
            double t1=now();
            vc_bench_encode_singlepass(vol,(vc_dims){N,N,N},qsel,nlod,&arc2,&alen2);
            double enc_sp_t=now()-t1;
            free(arc2);
            // decode full LOD0
            vc_archive *a=vc_open(arc,alen);
            unsigned char *out=malloc(n); vc_dims od;
            double t2=now(); vc_decode_lod(a,0,out,&od); double dec_t=now()-t2;
            // quality
            vc_metrics m; vc_compute_metrics(vol,out,N,N,N,&m);
            double gmsd=vc_gmsd(vol,out,N,N,N);
            // time-to-first-16^3-block: cold open + decode ONE atom at a fixed
            // interior location (atom index 17,23,31 -> voxels ~272,368,496).
            vc_close(a); // ensure cold
            double tf0=now();
            vc_archive *ac=vc_open(arc,alen);
            unsigned char blk[VC_ATOM3];
            vc_decode_atom(ac,0,17,23,31,blk);
            double ttf_us=(now()-tf0)*1e6;
            vc_close(ac);
            double lod0_ratio = (double)n / (double)ours_lod0_len(arc,alen);
            emit("ours",targets[ti],(double)n/alen,lod0_ratio,m.psnr,m.ms_ssim,gmsd,
                 n/1e6/enc_e2e_t, n/1e6/enc_sp_t, n/1e6/dec_t, ttf_us);
            free(arc); free(out);
        }
    }

    // ================= c3d (256^3 chunk atom; native target_ratio) =================
    // We tile the 1024^3 volume into 4x4x4 = 64 chunks of 256^3, encode each at
    // the SAME target_ratio (c3d_chunk_encode takes target_ratio directly), then
    // decode+scatter and score the full volume. encode/decode timed across all
    // chunks. time-to-first-16^3-block = cold-decode ONE 256^3 chunk + extract a
    // 16^3 sub-cube (256^3 is c3d's random-access unit).
    if(!strcmp(which,"c3d")||!strcmp(which,"all")){
        const uint32_t C=C3D_CHUNK_SIDE; // 256
        uint32_t nc=(uint32_t)N/C;
        size_t csz=(size_t)C*C*C;
        for(int ti=0; ti<ntargets; ti++){
            float tr=(float)targets[ti];
            uint8_t *cin=aligned_alloc(32,csz);
            uint8_t *cout=aligned_alloc(32,csz);
            uint8_t *enc=malloc(c3d_chunk_encode_max_size());
            unsigned char *rec=calloc(n,1);
            // store one encoded chunk for the cold-access test (interior chunk)
            uint8_t *saved_enc=malloc(c3d_chunk_encode_max_size()); size_t saved_len=0;
            uint32_t save_cz=N/C/2, save_cy=N/C/2, save_cx=N/C/2;
            size_t total_enc=0; double enc_t=0, dec_t=0;
            c3d_encoder *E=c3d_encoder_new(); c3d_decoder *D=c3d_decoder_new();
            for(uint32_t cz=0; cz<nc; cz++)
            for(uint32_t cy=0; cy<nc; cy++)
            for(uint32_t cx=0; cx<nc; cx++){
                for(uint32_t z=0; z<C; z++)
                for(uint32_t y=0; y<C; y++){
                    size_t vz=(size_t)cz*C+z, vy=(size_t)cy*C+y, vx=(size_t)cx*C;
                    memcpy(cin+((size_t)z*C+y)*C, vol+((vz*N+vy)*N+vx), C);
                }
                double a=now();
                size_t el=c3d_encoder_chunk_encode(E,cin,tr,enc,c3d_chunk_encode_max_size());
                double b=now();
                c3d_decoder_chunk_decode(D,enc,el,cout);
                double c=now();
                enc_t+=b-a; dec_t+=c-b; total_enc+=el;
                if(cz==save_cz&&cy==save_cy&&cx==save_cx){memcpy(saved_enc,enc,el);saved_len=el;}
                for(uint32_t z=0; z<C; z++)
                for(uint32_t y=0; y<C; y++){
                    size_t vz=(size_t)cz*C+z, vy=(size_t)cy*C+y, vx=(size_t)cx*C;
                    memcpy(rec+((vz*N+vy)*N+vx), cout+((size_t)z*C+y)*C, C);
                }
            }
            c3d_encoder_free(E); c3d_decoder_free(D);
            vc_metrics m; vc_compute_metrics(vol,rec,N,N,N,&m);
            double gmsd=vc_gmsd(vol,rec,N,N,N);
            // time-to-first-16^3: cold decoder, decode the saved 256^3 chunk,
            // extract a 16^3 sub-cube.
            double tf0=now();
            c3d_decoder *Dc=c3d_decoder_new();
            c3d_decoder_chunk_decode(Dc,saved_enc,saved_len,cout);
            uint8_t blk[16*16*16];
            for(int z=0;z<16;z++)for(int y=0;y<16;y++)
                memcpy(blk+(z*16+y)*16, cout+(((size_t)(17*16+z)*C)+(23*16+y))*C+(31*16),16);
            double ttf_us=(now()-tf0)*1e6;
            c3d_decoder_free(Dc);
            emit("c3d",targets[ti],(double)n/total_enc,(double)n/total_enc,
                 m.psnr,m.ms_ssim,gmsd, n/1e6/enc_t, n/1e6/enc_t, n/1e6/dec_t, ttf_us);
            free(cin);free(cout);free(enc);free(rec);free(saved_enc);
        }
    }
    free(vol);
    return 0;
}
