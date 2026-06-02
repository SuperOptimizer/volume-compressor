#define _POSIX_C_SOURCE 199309L
#include "src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char**argv){
  const char*path=argv[1]; int N=atoi(argv[2]); // cube side
  size_t n=(size_t)N*N*N;
  FILE*f=fopen(path,"rb"); if(!f){perror(path);return 1;}
  unsigned char*vol=malloc(n); if(fread(vol,1,n,f)!=n){printf("short read\n");return 1;} fclose(f);
  vc_dims d={(uint32_t)N,(uint32_t)N,(uint32_t)N};
  printf("== %s %dx%dx%d (raw %.1f MB) — FROZEN CODEC (src/vc) ==\n",path,N,N,N,n/1e6);
  printf(" target |  ratio | PSNR(dB) | enc MB/s | dec MB/s\n");
  float targets[]={10,20,50,100};
  for(int ti=0;ti<4;ti++){
    unsigned char*arc; size_t alen;
    double t0=now(); vc_status s=vc_encode(vol,d,targets[ti],&arc,&alen); double te=now()-t0;
    if(s){printf(" %5.0f  | ENCODE FAIL %d\n",targets[ti],s); continue;}
    vc_archive*a=vc_open(arc,alen);
    unsigned char*out=malloc(n); vc_dims od;
    double t1=now(); vc_decode_lod(a,0,out,&od); double td=now()-t1;
    double mse=0; for(size_t i=0;i<n;i++){double e=(double)vol[i]-out[i]; mse+=e*e;} mse/=n;
    double psnr= mse>0? 10*log10(255.0*255/mse): 99.0;
    printf(" %5.0f  | %6.1f | %8.2f | %8.0f | %8.0f\n",
       targets[ti],(double)n/alen,psnr, n/1e6/te, n/1e6/td);
    vc_close(a); free(arc); free(out);
  }
  return 0;
}
