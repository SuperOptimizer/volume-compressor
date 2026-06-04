// vc_verify — decode PRESENT LOD0 atoms from a produced .vca and PSNR them
// against the true source zarr0 voxels fetched from S3. Validates the real
// export pipeline output (not a fixture). Samples a stride of atoms for speed.
#define _GNU_SOURCE
#include "../src/vc/vc.h"
#include "third_party/cJSON.h"
#include "third_party/libs3/libs3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
typedef uint8_t u8; typedef uint32_t u32;
static s3_client *g_s3;
static s3_status credp(void*ud,s3_credentials*o){(void)ud;return s3_credentials_load(NULL,o);}
// Fetch an object: S3 (s3://) or a local file path. Returns NULL on miss.
static u8* fetch(const char*url,size_t*len){
  if(strncmp(url,"s3://",5)==0){
    s3_response r; memset(&r,0,sizeof r);
    if(s3_get(g_s3,url,&r)!=S3_OK||r.status!=200||!r.body){s3_response_free(&r);return NULL;}
    u8*o=malloc(r.body_len); memcpy(o,r.body,r.body_len); *len=r.body_len;
    s3_response_free(&r); return o;
  }
  int fd=open(url,O_RDONLY); if(fd<0)return NULL;
  struct stat st; if(fstat(fd,&st)||!S_ISREG(st.st_mode)){close(fd);return NULL;}
  u8*o=malloc(st.st_size); ssize_t g=read(fd,o,st.st_size); close(fd);
  if(g!=(ssize_t)st.st_size){free(o);return NULL;} *len=st.st_size; return o;
}
int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"usage: %s <zarr_root> <out.vca> [stride=97] [maxatoms=4000]\n",argv[0]);return 2;}
  const char*zarr=argv[1],*vca=argv[2];
  long stride=argc>3?atol(argv[3]):97, maxa=argc>4?atol(argv[4]):4000;
  int use_s3=strncmp(zarr,"s3://",5)==0;
  s3_config cfg; memset(&cfg,0,sizeof cfg); cfg.max_retries=5;
  if(use_s3){
  s3_credentials pr; memset(&pr,0,sizeof pr);
  if(s3_credentials_load(NULL,&pr)==S3_OK){cfg.cred_provider=credp; if(pr.region&&pr.region[0])cfg.region=strdup(pr.region); s3_credentials_free(&pr);}
  g_s3=s3_client_new(&cfg);
  }
  // read zarr 0/.zarray for shape + chunk size
  char p[1200]; size_t n;
  snprintf(p,sizeof p,"%s/0/.zarray",zarr);
  u8*zj=fetch(p,&n); if(!zj){fprintf(stderr,"no .zarray\n");return 1;}
  cJSON*j=cJSON_Parse((char*)zj); cJSON*sh=cJSON_GetObjectItem(j,"shape"),*ch=cJSON_GetObjectItem(j,"chunks");
  u32 sz=cJSON_GetArrayItem(sh,0)->valueint, sy=cJSON_GetArrayItem(sh,1)->valueint, sx=cJSON_GetArrayItem(sh,2)->valueint;
  u32 cz=cJSON_GetArrayItem(ch,0)->valueint, cy=cJSON_GetArrayItem(ch,1)->valueint, cx=cJSON_GetArrayItem(ch,2)->valueint;
  printf("source %ux%ux%u chunk %ux%ux%u\n",sz,sy,sx,cz,cy,cx);
  // open vca
  int fd=open(vca,O_RDONLY); struct stat st; fstat(fd,&st);
  void*m=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
  vc_archive*a=vc_open((u8*)m,st.st_size); if(!a){fprintf(stderr,"vc_open failed\n");return 1;}
  vc_dims d0; vc_lod_dims(a,0,&d0);
  printf("vca LOD0 dims %ux%ux%u\n",d0.nx,d0.ny,d0.nz);
  u32 acx=(sx+31)/32, acy=(sy+31)/32, acz=(sz+31)/32; // atom grid over TRUE source extent
  // chunk cache: 1 entry (last chunk) — atoms within a chunk are contiguous in scan
  u8*cc=NULL; long cck=-1; u32 ccx=cx*cy*cz>0?0:0;(void)ccx;
  double sumsq=0; long npix=0,natom=0,zerr=0; long idx=0;
  u8 dec[VC_ATOM3];
  for(u32 az=0; az<acz; ++az)for(u32 ay=0; ay<acy; ++ay)for(u32 ax=0; ax<acx; ++ax){
    if((idx++ % stride)!=0) continue;
    if(natom>=maxa) goto done;
    if(vc_atom_coverage(a,0,az,ay,ax)!=VC_PRESENT) continue;
    if(vc_decode_atom(a,0,(int)ax,(int)ay,(int)az,dec)!=VC_OK){zerr++;continue;}
    // build reference 32^3 from source chunks (clamp at true extent -> 0)
    double asq=0; long an=0;
    for(u32 z=0;z<32;++z){u32 vz=az*32+z; for(u32 y=0;y<32;++y){u32 vy=ay*32+y; for(u32 x=0;x<32;++x){u32 vx=ax*32+x;
      u8 ref=0;
      if(vz<sz&&vy<sy&&vx<sx){
        long ckz=vz/cz,cky=vy/cy,ckx=vx/cx;
        long key=(ckz*((sy+cy-1)/cy)+cky)*((sx+cx-1)/cx)+ckx;
        if(key!=cck){ free(cc); char cp[1300]; snprintf(cp,sizeof cp,"%s/0/%ld/%ld/%ld",zarr,ckz,cky,ckx); size_t cn; cc=fetch(cp,&cn); cck=key; }
        if(cc){ u32 lz=vz%cz,ly=vy%cy,lx=vx%cx; ref=cc[(lz*cy+ly)*cx+lx]; }
      }
      double dv=(double)dec[(z*32+y)*32+x]-(double)ref; asq+=dv*dv; an++;
    }}}
    sumsq+=asq; npix+=an; natom++;
  }
done:
  if(npix){ double mse=sumsq/npix; double psnr=mse>0?10.0*log10(255.0*255.0/mse):99.0;
    printf("verified %ld PRESENT atoms (%ld voxels): overall PSNR %.2f dB, decode-errors %ld\n",natom,npix,psnr,zerr);}
  else printf("no present atoms sampled (stride too coarse?)\n");
  return 0;
}
