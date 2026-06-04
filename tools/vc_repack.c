// vc_repack — rewrite a v1 .vca into the v2 container: region-contiguous blobs +
// a direct-indexed region directory, ordered coarse-LOD-first. The atom codec and
// the compressed payloads are UNCHANGED (copied verbatim, no re-encode) — only the
// on-disk layout/index changes, so v2 decodes bit-identically to v1 but reads with
// locality (one contiguous span per region: great for cached/offline random access
// and ~1 range-GET per region when streaming).
//
// Usage: vc_repack <in.vca (v1)> <out.vca (v2)>
//
// Layout written:
//   [128B header (VC2)] [region directory: dircnt x 16B] [region blobs ...]
//   blob = [512KB L2 slot array][payloads, contiguous in slot order]
// Directory entry = [u64 blob_off][u32 blob_len][u32 flags{ABSENT,ZERO,DATA}].
#define _GNU_SOURCE
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef uint8_t u8; typedef uint32_t u32; typedef uint64_t u64;
#define A 32u
#define R_ATOMS 32u
#define R3 (R_ATOMS*R_ATOMS*R_ATOMS)
#define ATOM_SLOT 16u
#define L2_BLOCK_BYTES ((u64)R3*ATOM_SLOT)   // 512 KiB slot array per region
#define FILE_HDR 128u
// header field offsets (must match vc.c)
#define FH_MAGIC 0
#define FH_VER 4
#define FH_ATOM 8
#define FH_RGN 16
#define FH_NLOD 20
#define FH_NX 24
#define FH_NY 28
#define FH_NZ 32
#define FH_TRATIO 36
#define FH2_DIROFF 40
#define FH2_DIRCNT 48
#define FH2_TOTLEN 56
#define FH_BASEQ 72
#define VC_MAGIC 0x00314356u
#define VC_VERSION2 2u
#define V2_DIR_ENTRY 16u
#define V2_RGN_ABSENT 0u
#define V2_RGN_ZERO   1u
#define V2_RGN_DATA   2u
#define AF_PRESENT 1u

static void wr32(u8*p,u32 v){memcpy(p,&v,4);}
static void wr64(u8*p,u64 v){memcpy(p,&v,8);}
static u32 rd32(const u8*p){u32 v;memcpy(&v,p,4);return v;}

static void dims_lod(u32 nx,u32 ny,u32 nz,int lod,u32*ox,u32*oy,u32*oz){
    for(int i=0;i<lod;i++){nx=(nx+1)/2;ny=(ny+1)/2;nz=(nz+1)/2;} *ox=nx;*oy=ny;*oz=nz;
}
static void region_grid(u32 nx,u32 ny,u32 nz,int lod,u32*nrz,u32*nry,u32*nrx){
    u32 lx,ly,lz; dims_lod(nx,ny,nz,lod,&lx,&ly,&lz);
    *nrx=((lx+A-1)/A+R_ATOMS-1)/R_ATOMS;
    *nry=((ly+A-1)/A+R_ATOMS-1)/R_ATOMS;
    *nrz=((lz+A-1)/A+R_ATOMS-1)/R_ATOMS;
}
static u64 slot_index(u32 az,u32 ay,u32 ax){
    u32 lz=az&(R_ATOMS-1), ly=ay&(R_ATOMS-1), lx=ax&(R_ATOMS-1);
    return ((u64)lz*R_ATOMS+ly)*R_ATOMS+lx;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s <in.vca v1> <out.vca v2>\n",argv[0]);return 2;}
    const char *inpath=argv[1], *outpath=argv[2];

    // ---- map the v1 input + open it through libvc (validates + gives dims/nlod) ----
    int ifd=open(inpath,O_RDONLY); if(ifd<0){perror("open in");return 1;}
    struct stat ist; fstat(ifd,&ist); u64 ilen=ist.st_size;
    const u8 *ibuf=mmap(NULL,ilen,PROT_READ,MAP_PRIVATE,ifd,0);
    if(ibuf==MAP_FAILED){perror("mmap in");return 1;}
    vc_archive *in=vc_open(ibuf,ilen);
    if(!in){fprintf(stderr,"vc_open(in) failed (not a v1 .vca?)\n");return 1;}
    u32 nx=rd32(ibuf+FH_NX),ny=rd32(ibuf+FH_NY),nz=rd32(ibuf+FH_NZ);
    int nlod=(int)rd32(ibuf+FH_NLOD);
    printf("input v%u: %ux%ux%u (x,y,z) nlod=%d  %.1f GB\n",rd32(ibuf+FH_VER),nx,ny,nz,nlod,ilen/1e9);

    // ---- enumerate regions COARSE-LOD-FIRST; assign each a linear directory index ----
    // Directory order = the canonical region linear index (LOD-major, z,y,x). We
    // WRITE blobs in coarse-first order (LOD nlod-1 .. 0) for front-loaded overview,
    // but the directory entry index is the canonical one so the reader's regidx math
    // (prefix[lod] + (rz*nry+ry)*nrx+rx, LOD ascending) addresses it directly.
    u64 prefix[VC_NLOD+1]; u64 total_regions=0;
    for(int l=0;l<nlod;l++){ prefix[l]=total_regions; u32 rz,ry,rx; region_grid(nx,ny,nz,l,&rz,&ry,&rx); total_regions+=(u64)rz*ry*rx; }
    prefix[nlod]=total_regions;
    printf("regions: %lu across %d LODs; directory = %lu KB\n",total_regions,nlod,total_regions*V2_DIR_ENTRY/1024);

    u64 dir_off=FILE_HDR;
    u64 blobs_off=dir_off+total_regions*V2_DIR_ENTRY;
    // round blob area start up to 4KB for clean alignment
    blobs_off=(blobs_off+4095)&~4095ull;

    // ---- PASS 1: per region, classify + size its blob (slots + sum of payloads) ----
    // dir flags + each present region's payload total, computed via the v1 reader.
    u8 *dir=calloc(total_regions, V2_DIR_ENTRY); if(!dir){fprintf(stderr,"oom dir\n");return 1;}
    u64 *blob_off=calloc(total_regions,sizeof(u64));
    u64 *blob_len=calloc(total_regions,sizeof(u64));
    u64 cursor=blobs_off; long present_regions=0,zero_regions=0,present_atoms=0;
    // iterate coarse-first for layout, but index the dir by canonical regidx
    for(int l=nlod-1;l>=0;--l){
        u32 nrz,nry,nrx; region_grid(nx,ny,nz,l,&nrz,&nry,&nrx);
        for(u32 rz=0;rz<nrz;++rz)for(u32 ry=0;ry<nry;++ry)for(u32 rx=0;rx<nrx;++rx){
            u64 regidx=prefix[l]+((u64)rz*nry+ry)*nrx+rx;
            // scan the region's 32^3 atoms for present/zero/absent + payload bytes
            u64 paybytes=0; int anyPresent=0,anyZero=0;
            for(u32 dz=0;dz<R_ATOMS;++dz)for(u32 dy=0;dy<R_ATOMS;++dy)for(u32 dx=0;dx<R_ATOMS;++dx){
                u32 az=rz*R_ATOMS+dz, ay=ry*R_ATOMS+dy, ax=rx*R_ATOMS+dx;
                u64 off; u32 len;
                vc_cover c=vc_atom_payload_range(in,l,az,ay,ax,&off,&len);
                if(c==VC_PRESENT){ anyPresent=1; paybytes+=len; }
                else if(c==VC_KNOWN_ZERO) anyZero=1;
            }
            u8 *e=dir+regidx*V2_DIR_ENTRY;
            if(anyPresent){
                u64 boff=cursor; u64 blen=L2_BLOCK_BYTES+paybytes;
                wr64(e,boff); wr32(e+8,(u32)blen); wr32(e+12,V2_RGN_DATA);
                blob_off[regidx]=boff; blob_len[regidx]=blen; cursor+=blen;
                present_regions++; present_atoms+=0; // counted in pass 2
            } else if(anyZero){
                wr64(e,0); wr32(e+8,0); wr32(e+12,V2_RGN_ZERO); zero_regions++;
            } else { wr64(e,0); wr32(e+8,0); wr32(e+12,V2_RGN_ABSENT); }
        }
    }
    u64 total_len=cursor;
    printf("v2 size = %.1f GB (%ld present regions, %ld all-zero)\n",total_len/1e9,present_regions,zero_regions);

    // ---- create + map the output, sized exactly ----
    int ofd=open(outpath,O_RDWR|O_CREAT|O_TRUNC,0644); if(ofd<0){perror("open out");return 1;}
    if(ftruncate(ofd,(off_t)total_len)!=0){perror("ftruncate out");return 1;}
    u8 *obuf=mmap(NULL,total_len,PROT_READ|PROT_WRITE,MAP_SHARED,ofd,0);
    if(obuf==MAP_FAILED){perror("mmap out");return 1;}

    // ---- PASS 2: write each present region's blob: slots + payloads (verbatim) ----
    for(int l=nlod-1;l>=0;--l){
        u32 nrz,nry,nrx; region_grid(nx,ny,nz,l,&nrz,&nry,&nrx);
        for(u32 rz=0;rz<nrz;++rz)for(u32 ry=0;ry<nry;++ry)for(u32 rx=0;rx<nrx;++rx){
            u64 regidx=prefix[l]+((u64)rz*nry+ry)*nrx+rx;
            if(rd32(dir+regidx*V2_DIR_ENTRY+12)!=V2_RGN_DATA) continue;
            u64 boff=blob_off[regidx];
            u8 *slots=obuf+boff;                 // 512KB slot array (already zeroed by sparse file)
            u64 paycur=boff+L2_BLOCK_BYTES;      // payloads begin right after the slots
            for(u32 dz=0;dz<R_ATOMS;++dz)for(u32 dy=0;dy<R_ATOMS;++dy)for(u32 dx=0;dx<R_ATOMS;++dx){
                u32 az=rz*R_ATOMS+dz, ay=ry*R_ATOMS+dy, ax=rx*R_ATOMS+dx;
                u64 ioff; u32 ilen;
                vc_cover c=vc_atom_payload_range(in,l,az,ay,ax,&ioff,&ilen);
                u8 *slot=slots+slot_index(az,ay,ax)*ATOM_SLOT;
                if(c==VC_PRESENT){
                    // copy the compressed payload verbatim; slot gets ABSOLUTE v2 offset.
                    // dc is the payload's first byte (vc.c decode uses pay[0] as dc; the
                    // v1 slot's dc byte mirrors it). We re-read dc from the input slot via
                    // a decode-free path: it's pay[0], so just copy it from the payload.
                    memcpy(obuf+paycur, ibuf+ioff, ilen);
                    wr64(slot, paycur); wr32(slot+8, ilen);
                    slot[12]=(u8)AF_PRESENT; slot[13]=ibuf[ioff]; // dc == first payload byte
                    paycur+=ilen; present_atoms++;
                } else if(c==VC_KNOWN_ZERO){
                    slot[12]=2u; // AF_ZERO
                }
                // ABSENT -> slot stays all-zero (AF_ABSENT)
            }
        }
    }

    // ---- header ----
    u8 hdr[FILE_HDR]; memset(hdr,0,sizeof hdr);
    wr32(hdr+FH_MAGIC,VC_MAGIC); wr32(hdr+FH_VER,VC_VERSION2);
    wr32(hdr+FH_ATOM,A); wr32(hdr+FH_RGN,R_ATOMS); wr32(hdr+FH_NLOD,(u32)nlod);
    wr32(hdr+FH_NX,nx); wr32(hdr+FH_NY,ny); wr32(hdr+FH_NZ,nz);
    memcpy(hdr+FH_TRATIO, ibuf+FH_TRATIO, 4);
    memcpy(hdr+FH_BASEQ, ibuf+FH_BASEQ, VC_NLOD*4);
    wr64(hdr+FH2_DIROFF,dir_off); wr64(hdr+FH2_DIRCNT,total_regions); wr64(hdr+FH2_TOTLEN,total_len);
    memcpy(obuf, hdr, FILE_HDR);
    memcpy(obuf+dir_off, dir, total_regions*V2_DIR_ENTRY);

    msync(obuf,total_len,MS_SYNC);
    munmap(obuf,total_len); close(ofd);
    vc_close(in); munmap((void*)ibuf,ilen); close(ifd);
    free(dir); free(blob_off); free(blob_len);
    printf("done: %s  v2  %.1f GB  present_atoms=%ld\n",outpath,total_len/1e9,present_atoms);
    return 0;
}
