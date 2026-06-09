// v1 whole-volume baseline: encode the 1024^3 zarr through the REAL v1 codec
// (src/vc/vc.c via libvc) and report the actual archive size — a true, non-
// estimated ratio to compare v2lab against. LOD0 only (the headline level).
// build: cc -O3 -march=native -o v1_whole v1_whole.c -I../../src/vc -L../../build -lvc -lm -lpthread
#include "vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <time.h>

typedef uint8_t u8;
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }

static u8 *load_zarr_1024(const char *root,int V,int CH){
    int G=V/CH; u8 *vol=calloc((size_t)V*V*V,1); u8 *cb=malloc((size_t)CH*CH*CH); char p[1024];
    for(int c0=0;c0<G;++c0)for(int c1=0;c1<G;++c1)for(int c2=0;c2<G;++c2){
        snprintf(p,sizeof p,"%s/0/%d/%d/%d",root,c0,c1,c2);
        FILE *f=fopen(p,"rb"); if(!f) continue;
        if(fread(cb,1,(size_t)CH*CH*CH,f)!=(size_t)CH*CH*CH){fclose(f);continue;} fclose(f);
        for(int z=0;z<CH;++z)for(int y=0;y<CH;++y){
            u8 *dst=&vol[(((size_t)(c0*CH+z))*V+(c1*CH+y))*V+c2*CH];
            memcpy(dst,&cb[((size_t)z*CH+y)*CH],CH);
        }
    } free(cb); return vol;
}

int main(int argc,char**argv){
    const char *root = argc>1?argv[1]:"/home/forrest/paris4_2um_processed";
    float q = argc>2?(float)atof(argv[2]):1.0f;
    const char *out = "/tmp/v1_whole.vca";
    int V=1024, A=32;
    fprintf(stderr,"v1: loading %s...\n",root);
    u8 *vol=load_zarr_1024(root,V,V<128?V:128);
    size_t N=(size_t)V*V*V; long nz=0; for(size_t i=0;i<N;++i) if(vol[i]) nz++;
    fprintf(stderr,"v1: nonzero=%.2f%%, encoding LOD0 at q=%.2f...\n",100.0*nz/N,q);
    vc_dims d={V,V,V};
    vc_writer *w=vc_create(out,d,10.0f);
    if(!w){fprintf(stderr,"vc_create failed\n");return 1;}
    vc_set_base_q(w,0,q);
    u8 atom[32*32*32];
    int AC=V/A;
    double t0=now_ms();
    for(int az=0;az<AC;++az)for(int ay=0;ay<AC;++ay)for(int ax=0;ax<AC;++ax){
        for(int z=0;z<A;++z)for(int y=0;y<A;++y)for(int x=0;x<A;++x)
            atom[(z*A+y)*A+x]=vol[(((size_t)(az*A+z))*V+(ay*A+y))*V+(ax*A+x)];
        vc_append_atom(w,0,az,ay,ax,atom);
    }
    vc_writer_close(w);
    double t1=now_ms();
    struct stat st; stat(out,&st);
    double comp=(double)st.st_size, logical=(double)N;
    // Decode the archive back and compute PSNR/max-err over SIGNAL (nonzero) voxels,
    // so v1 reports the same quality basket as v2 for a fair quality-matched compare.
    FILE *af=fopen(out,"rb"); u8 *arc=malloc((size_t)comp); fread(arc,1,(size_t)comp,af); fclose(af);
    vc_archive *a=vc_open(arc,(size_t)comp);
    double se=0,mx=0; long m=0;
    u8 dec[32*32*32];
    for(int az=0;az<AC;++az)for(int ay=0;ay<AC;++ay)for(int ax=0;ax<AC;++ax){
        vc_decode_atom(a,0,ax,ay,az,dec);
        for(int z=0;z<A;++z)for(int y=0;y<A;++y)for(int x=0;x<A;++x){
            u8 orig=vol[(((size_t)(az*A+z))*V+(ay*A+y))*V+(ax*A+x)];
            if(!orig) continue; double e=fabs((double)orig-dec[(z*A+y)*A+x]); se+=e*e; if(e>mx)mx=e; m++;
        }
    }
    double mse=m?se/m:0, psnr=mse<=0?99:10*log10(255.0*255.0/mse);
    // MATERIAL-ONLY ratio (the 10-100x spec): sum actual atom payload bytes (the DCT
    // data) / material voxels. Excludes the zero/absent atoms (the mask) just like v2.
    double pay_bytes=0; long matvox=m;   // m = nonzero voxels counted above
    for(int az=0;az<AC;++az)for(int ay=0;ay<AC;++ay)for(int ax=0;ax<AC;++ax){
        uint64_t off; uint32_t len;
        if(vc_atom_payload_range(a,0,az,ay,ax,&off,&len)==VC_PRESENT) pay_bytes+=len;
    }
    double mat_ratio = matvox / pay_bytes;
    printf("=== v1 REAL codec, whole 1024^3 LOD0, q=%.2f ===\n",q);
    printf("archive: %.2f MB  TOTAL-ratio %.2fx  MATERIAL-ratio %.1fx (spec 10-100x)  PSNR %.2f  max %.0f  enc %.2fs\n",
           comp/1e6, logical/comp, mat_ratio, psnr, mx, (t1-t0)/1e3);
    return 0;
}
