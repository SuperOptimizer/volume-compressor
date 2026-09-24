/* single-thread encode/decode throughput: mode-0 q8 vs surface q; usage: bench_surf chunks.u8 */
#include "volcomp.h"
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t*all=malloc(sz); if(fread(all,1,sz,f)!=(size_t)sz) return 1; size_t n=sz/VOLCOMP_CHUNK_VOXELS;
  uint8_t*enc=malloc(VOLCOMP_SURFACE_ENCODE_BOUND),*dec=malloc(VOLCOMP_CHUNK_VOXELS);
  const float qs[]={8,48,64,96}; 
  for(int k=0;k<4;k++){ int surf=k>0; double te=0,td=0; size_t bytes=0;
    for(size_t i=0;i<n;i++){ const uint8_t*c=all+i*VOLCOMP_CHUNK_VOXELS; size_t en; double t0=now();
      if(surf) volcomp_surface_encode(c,qs[k],128,enc,VOLCOMP_SURFACE_ENCODE_BOUND,&en); else volcomp_encode(c,qs[k],enc,VOLCOMP_ENCODE_BOUND,&en);
      te+=now()-t0; bytes+=en; t0=now(); for(int r=0;r<3;r++) volcomp_decode(enc,en,dec,VOLCOMP_CHUNK_VOXELS); td+=(now()-t0)/3; }
    double mb=n*2.097152; printf("%-10s q%-3.0f %8zu B/chunk  encode %6.0f MB/s  decode %6.0f MB/s\n", surf?"surface":"mode-0", qs[k], bytes/n, mb/te, mb/td);}
  printf("kernels %s, %zu chunks\n", volcomp_kernels(), n); return 0;}
