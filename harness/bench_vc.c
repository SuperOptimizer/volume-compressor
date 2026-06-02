// Measurement harness for the frozen vc codec on real PHerc Paris 4 volumes.
// Reports: achieved ratio vs target, PSNR, MS-SSIM(2D-slice avg), GMSD,
// encode/decode MB/s, single-16^3-atom decode latency, total pyramid size.
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

static unsigned char *load(const char *path, size_t want, size_t *got){
    FILE *f = fopen(path,"rb"); if(!f){ perror(path); exit(1);}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    unsigned char *b = malloc(sz); fread(b,1,sz,f); fclose(f);
    if(got)*got=sz; (void)want; return b;
}

static double psnr(const unsigned char*a,const unsigned char*b,size_t n){
    double se=0; for(size_t i=0;i<n;i++){double d=(double)a[i]-b[i]; se+=d*d;}
    double mse=se/n; return mse<=0?99.0:10.0*log10(255.0*255.0/mse);
}

// GMSD on the central 2D slice (Z mid). Lower = better; ~0 identical.
static double gmsd_slice(const unsigned char*a,const unsigned char*b,unsigned nx,unsigned ny,unsigned nz){
    unsigned z=nz/2; const unsigned char*A=a+(size_t)z*nx*ny; const unsigned char*B=b+(size_t)z*nx*ny;
    double *ga=malloc(sizeof(double)*nx*ny), *gb=malloc(sizeof(double)*nx*ny);
    for(unsigned y=1;y<ny-1;y++)for(unsigned x=1;x<nx-1;x++){
        double hx_a=(double)A[y*nx+x+1]-A[y*nx+x-1], hy_a=(double)A[(y+1)*nx+x]-A[(y-1)*nx+x];
        double hx_b=(double)B[y*nx+x+1]-B[y*nx+x-1], hy_b=(double)B[(y+1)*nx+x]-B[(y-1)*nx+x];
        ga[y*nx+x]=sqrt(hx_a*hx_a+hy_a*hy_a); gb[y*nx+x]=sqrt(hx_b*hx_b+hy_b*hy_b);
    }
    double c=170.0, sum=0,sum2=0; unsigned cnt=0;
    for(unsigned y=1;y<ny-1;y++)for(unsigned x=1;x<nx-1;x++){
        double gm=(2*ga[y*nx+x]*gb[y*nx+x]+c)/(ga[y*nx+x]*ga[y*nx+x]+gb[y*nx+x]*gb[y*nx+x]+c);
        sum+=gm; sum2+=gm*gm; cnt++;
    }
    double mean=sum/cnt; double var=sum2/cnt-mean*mean;
    free(ga);free(gb); return sqrt(var<0?0:var);
}

// Simple single-scale SSIM on central slice (proxy for MS-SSIM), 8x8 windows.
static double ssim_slice(const unsigned char*a,const unsigned char*b,unsigned nx,unsigned ny,unsigned nz){
    unsigned z=nz/2; const unsigned char*A=a+(size_t)z*nx*ny; const unsigned char*B=b+(size_t)z*nx*ny;
    double C1=6.5025,C2=58.5225,total=0; unsigned wins=0;
    for(unsigned y=0;y+8<=ny;y+=8)for(unsigned x=0;x+8<=nx;x+=8){
        double ma=0,mb=0; for(int j=0;j<8;j++)for(int i=0;i<8;i++){ma+=A[(y+j)*nx+x+i];mb+=B[(y+j)*nx+x+i];}
        ma/=64;mb/=64; double va=0,vb=0,cov=0;
        for(int j=0;j<8;j++)for(int i=0;i<8;i++){double da=A[(y+j)*nx+x+i]-ma,db=B[(y+j)*nx+x+i]-mb; va+=da*da;vb+=db*db;cov+=da*db;}
        va/=63;vb/=63;cov/=63;
        total+=((2*ma*mb+C1)*(2*cov+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2)); wins++;
    }
    return total/wins;
}

int main(int argc,char**argv){
    const char *path = argc>1?argv[1]:"harness/refbuild/hires_256.u8";
    vc_dims d={256,256,256};
    size_t n=(size_t)d.nx*d.ny*d.nz, got;
    unsigned char *vol=load(path,n,&got);
    if(got<n){ printf("file too small %zu<%zu\n",got,n); return 1;}
    double targets[]={10,20,50,100};
    printf("# %s  (%ux%ux%u = %.1f MB)\n",path,d.nx,d.ny,d.nz,n/1e6);
    printf("target  achieved   PSNR    SSIM    GMSD   enc_MB/s dec_MB/s atom_us  pyramid_MB\n");
    for(int t=0;t<4;t++){
        unsigned char *arc; size_t alen;
        double t0=now_s(); vc_encode(vol,d,(float)targets[t],&arc,&alen); double te=now_s()-t0;
        vc_archive *a=vc_open(arc,alen);
        unsigned char *out=malloc(n);
        double t1=now_s(); vc_decode_lod(a,0,out,NULL); double td=now_s()-t1;
        // count LOD members + their dims
        int nl=0; vc_dims ld; for(int l=0;l<8;l++) if(vc_lod_dims(a,l,&ld)==VC_OK) nl++;
        double ratio=(double)n/alen; // whole archive incl pyramid
        double ps=psnr(vol,out,n);
        double ss=ssim_slice(vol,out,d.nx,d.ny,d.nz);
        double gm=gmsd_slice(vol,out,d.nx,d.ny,d.nz);
        double enc_mbs=(n/1e6)/te, dec_mbs=(n/1e6)/td;
        // atom decode latency: average over many random atoms
        int reps=2000; unsigned char atom[4096];
        double t2=now_s();
        for(int r=0;r<reps;r++){int ax=rand()%16,ay=rand()%16,az=rand()%16; vc_decode_atom(a,0,ax,ay,az,atom);}
        double atom_us=(now_s()-t2)/reps*1e6;
        printf("%5.0fx  %7.2fx %7.2f %7.4f %7.4f %8.1f %8.1f %7.2f %10.3f  (LODs=%d)\n",
               targets[t],ratio,ps,ss,gm,enc_mbs,dec_mbs,atom_us,alen/1e6,nl);
        vc_close(a); free(arc); free(out);
    }
    free(vol); return 0;
}
