// score <ref.u8> <rec.u8> <N> -> "PSNR MS-SSIM GMSD" using the shared vc metrics.
#include "src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
static unsigned char* rd(const char*p,size_t n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}
  unsigned char*b=malloc(n); if(fread(b,1,n,f)!=n){fprintf(stderr,"short\n");exit(1);} fclose(f); return b;}
int main(int c,char**v){ if(c<4)return 2;
  unsigned N=atoi(v[3]); size_t n=(size_t)N*N*N;
  unsigned char*r=rd(v[1],n),*q=rd(v[2],n);
  vc_metrics m; vc_compute_metrics(r,q,N,N,N,&m);
  double g=vc_gmsd(r,q,N,N,N);
  printf("%.2f %.4f %.4f\n",m.psnr,m.ms_ssim,g);
  return 0;}
