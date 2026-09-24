/* decode vs decode_smooth throughput, single thread. usage: bench_smooth chunks.u8 q surface(0/1) */
#include "volcomp.h"
#include <stdio.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t*all=malloc(sz); if(fread(all,1,sz,f)!=(size_t)sz) return 1; size_t n=sz/VOLCOMP_CHUNK_VOXELS; if(n>16)n=16;
  float q=atof(argv[2]); int surf=atoi(argv[3]);
  uint8_t*enc=malloc(VOLCOMP_SURFACE_ENCODE_BOUND),*dec=malloc(VOLCOMP_CHUNK_VOXELS);
  double t[3]={0}; 
  for(size_t i=0;i<n;i++){ size_t en; const uint8_t*c=all+i*VOLCOMP_CHUNK_VOXELS;
    if(surf) volcomp_surface_encode(c,q,128,enc,VOLCOMP_SURFACE_ENCODE_BOUND,&en); else volcomp_encode(c,q,enc,VOLCOMP_ENCODE_BOUND,&en);
    double t0=now(); volcomp_decode(enc,en,dec,VOLCOMP_CHUNK_VOXELS); t[0]+=now()-t0;
    t0=now(); volcomp_decode_smooth(enc,en,dec,VOLCOMP_CHUNK_VOXELS,2.0f,VOLCOMP_SMOOTH_GATED|VOLCOMP_DEBLOCK_ZERO_GUARD); t[1]+=now()-t0;
    t0=now(); volcomp_decode_smooth(enc,en,dec,VOLCOMP_CHUNK_VOXELS,0.6f,VOLCOMP_DEBLOCK_ZERO_GUARD); t[2]+=now()-t0;}
  double mb=n*2.097152; printf("%s q%.0f: decode %.0f MB/s, smooth gated2 zg %.0f MB/s (%.1f ms/chunk), smooth gauss0.6 zg %.0f MB/s (%.1f ms/chunk)\n", surf?"surface":"mode-0", q, mb/t[0], mb/t[1], t[1]/n*1e3, mb/t[2], t[2]/n*1e3);
  return 0;}
