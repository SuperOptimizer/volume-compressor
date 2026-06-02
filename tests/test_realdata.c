#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
int main(int c,char**v){
 FILE*f=fopen(v[1],"rb");unsigned char*b=malloc(16777216);if(fread(b,1,16777216,f)){};fclose(f);
 vc_dims d={256,256,256};unsigned char*arc;size_t l;
 vc_encode(b,d,50.0f,&arc,&l);
 vc_archive*a=vc_open(arc,l);
 if(!a){printf("open FAIL\n");return 1;}
 // decode_lod0 (deblocked) and a non-deblocked reference: compare every atom's
 // standalone decode to decode_region of exactly that atom (also deblocked-edge),
 // so instead verify standalone-atom == standalone-atom determinism + bounds.
 unsigned char*out=malloc(16777216);
 vc_decode_lod(a,0,out,NULL);
 // exhaustive single-atom decode over all atoms, interior compare to LOD
 long mism=0,checked=0; unsigned char atom[4096];
 for(int az=0;az<16;az++)for(int ay=0;ay<16;ay++)for(int ax=0;ax<16;ax++){
   if(vc_decode_atom(a,0,ax,ay,az,atom)!=VC_OK){printf("atom decode FAIL %d %d %d\n",ax,ay,az);return 1;}
   for(int z=2;z<14;z++)for(int y=2;y<14;y++)for(int x=2;x<14;x++){
     int vx=ax*16+x,vy=ay*16+y,vz=az*16+z;
     unsigned char fl=out[((size_t)vz*256+vy)*256+vx], fa=atom[(z*16+y)*16+x];
     if(fl!=fa)mism++; checked++;
   }
 }
 printf("exhaustive single-atom interior match vs decode_lod: %ld/%ld mismatch\n",mism,checked);
 // verify all 8 LODs decode without error
 for(int lod=0;lod<8;lod++){vc_dims ld; if(vc_lod_dims(a,lod,&ld)!=VC_OK){printf("LOD %d missing\n",lod);return 1;}
   size_t nn=(size_t)ld.nx*ld.ny*ld.nz; unsigned char*o=malloc(nn);
   if(vc_decode_lod(a,lod,o,NULL)!=VC_OK){printf("LOD %d decode FAIL\n",lod);return 1;}
   printf("  LOD%d %ux%ux%u decoded OK\n",lod,ld.nx,ld.ny,ld.nz); free(o);}
 // region decode bounds
 vc_box box={10,20,30,90,100,110}; unsigned char*rb=malloc(80*80*80);
 if(vc_decode_region(a,0,box,rb)!=VC_OK){printf("region FAIL\n");return 1;}
 printf("region decode OK\n");
 printf("RESULT: %s\n", mism==0?"PASS":"FAIL");
 return mism==0?0:1;}
