// vc_repack — rewrite a v1 .vca into the v2 container: region-contiguous blobs +
// a direct-indexed region directory, ordered coarse-LOD-first. The atom codec and
// the compressed payloads are UNCHANGED (copied verbatim, no re-encode) — only the
// on-disk layout/index changes, so v2 decodes bit-identically to v1 but reads with
// locality (one contiguous span per region: great for cached/offline random access
// and ~1 range-GET per region when streaming).
//
// Usage: vc_repack <in.vca (v1)> <out.vca (v2)> [--threads N]
//
// Parallel over regions (independent). PASS 1 (parallel) scans each region's atom
// slots to classify it + size its blob; a tiny serial prefix-sum then assigns each
// present region a contiguous blob offset in coarse-LOD-first order; PASS 2
// (parallel) copies each present region's slots+payloads verbatim into its blob.
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
#include <stdatomic.h>
#include <pthread.h>
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

// A region work-item: its canonical directory index + (lod, region coords). Built
// in coarse-LOD-first order so the blob layout is front-loaded with overview LODs.
typedef struct { u64 regidx; int lod; u32 rz,ry,rx; } ritem;

// Shared state for the parallel passes.
typedef struct {
    vc_archive *in; const u8 *ibuf; u8 *obuf;
    u32 nx,ny,nz;
    ritem *items; u64 nitems;
    u8 *dir;                      // [total_regions * 16]
    u64 *blob_off, *blob_len;     // per regidx
    atomic_ullong next;           // work counter (item index)
    atomic_llong present_regions, zero_regions, present_atoms;
} ctx;

// PASS 1: classify each region + compute its payload byte total. Writes dir flags
// and blob_len[regidx]; blob_off is assigned later by the serial prefix-sum.
static void *pass1_worker(void *arg){
    ctx *c=(ctx*)arg;
    long preg=0,zreg=0;
    for(;;){
        u64 i=atomic_fetch_add(&c->next,1);
        if(i>=c->nitems) break;
        ritem *it=&c->items[i];
        u64 paybytes=0; int anyPresent=0,anyZero=0;
        for(u32 dz=0;dz<R_ATOMS;++dz)for(u32 dy=0;dy<R_ATOMS;++dy)for(u32 dx=0;dx<R_ATOMS;++dx){
            u32 az=it->rz*R_ATOMS+dz, ay=it->ry*R_ATOMS+dy, ax=it->rx*R_ATOMS+dx;
            u64 off; u32 len;
            vc_cover cov=vc_atom_payload_range(c->in,it->lod,az,ay,ax,&off,&len);
            if(cov==VC_PRESENT){ anyPresent=1; paybytes+=len; }
            else if(cov==VC_KNOWN_ZERO) anyZero=1;
        }
        u8 *e=c->dir+it->regidx*V2_DIR_ENTRY;
        if(anyPresent){
            c->blob_len[it->regidx]=L2_BLOCK_BYTES+paybytes;
            wr32(e+12,V2_RGN_DATA); preg++;
        } else if(anyZero){
            wr64(e,0); wr32(e+8,0); wr32(e+12,V2_RGN_ZERO); zreg++;
        } else { wr64(e,0); wr32(e+8,0); wr32(e+12,V2_RGN_ABSENT); }
    }
    atomic_fetch_add(&c->present_regions,preg);
    atomic_fetch_add(&c->zero_regions,zreg);
    return NULL;
}

// one present atom's copy job: read [ioff,ioff+ilen) from v1, write at blob+ooff.
typedef struct { u64 ioff; u64 ooff; u32 ilen; } acopy;
static int acopy_cmp(const void *a,const void *b){
    u64 x=((const acopy*)a)->ioff, y=((const acopy*)b)->ioff;
    return x<y?-1:x>y?1:0;
}

// PASS 2: copy each present region's slots+payloads verbatim into its assigned blob.
// Within a region the v1 payloads are scattered across ~10-15x their own size of
// file (v1's global append cursor); reading them in atom (slot) order back-seeks all
// over that span and thrashes the page cache on a >RAM file. So we GATHER the
// region's present atoms, SORT them by input file offset, and copy in offset order —
// a near-sequential forward sweep of the span (each page read once, no re-eviction).
// The OUTPUT payload offset is still assigned in slot order (deterministic, matches
// the single-thread layout), so the result is byte-identical regardless of copy order.
static void *pass2_worker(void *arg){
    ctx *c=(ctx*)arg;
    long patoms=0;
    acopy *job=malloc(R3*sizeof(acopy));
    for(;;){
        u64 i=atomic_fetch_add(&c->next,1);
        if(i>=c->nitems) break;
        ritem *it=&c->items[i];
        if(rd32(c->dir+it->regidx*V2_DIR_ENTRY+12)!=V2_RGN_DATA) continue;
        u64 boff=c->blob_off[it->regidx];
        u8 *slots=c->obuf+boff;                 // 512KB slot array (sparse-zeroed)
        // first walk (slot order): assign each present atom its OUTPUT offset + write
        // its slot; mark zero slots; collect the copy jobs.
        u64 paycur=boff+L2_BLOCK_BYTES; u32 nj=0;
        for(u32 dz=0;dz<R_ATOMS;++dz)for(u32 dy=0;dy<R_ATOMS;++dy)for(u32 dx=0;dx<R_ATOMS;++dx){
            u32 az=it->rz*R_ATOMS+dz, ay=it->ry*R_ATOMS+dy, ax=it->rx*R_ATOMS+dx;
            u64 ioff; u32 ilen;
            vc_cover cov=vc_atom_payload_range(c->in,it->lod,az,ay,ax,&ioff,&ilen);
            u8 *slot=slots+slot_index(az,ay,ax)*ATOM_SLOT;
            if(cov==VC_PRESENT){
                wr64(slot,paycur); wr32(slot+8,ilen);
                slot[12]=(u8)AF_PRESENT; slot[13]=c->ibuf[ioff]; // dc == first payload byte
                job[nj].ioff=ioff; job[nj].ooff=paycur; job[nj].ilen=ilen;
                ++nj; paycur+=ilen;
            } else if(cov==VC_KNOWN_ZERO){
                slot[12]=2u; // AF_ZERO
            }
            // ABSENT -> slot stays all-zero (AF_ABSENT)
        }
        // sort the copy jobs by INPUT offset and copy in that order (sequential read).
        qsort(job,nj,sizeof(acopy),acopy_cmp);
        for(u32 j=0;j<nj;++j)
            memcpy(c->obuf+job[j].ooff, c->ibuf+job[j].ioff, job[j].ilen);
        patoms+=nj;
    }
    free(job);
    atomic_fetch_add(&c->present_atoms,patoms);
    return NULL;
}

int main(int argc,char**argv){
    if(argc<3){fprintf(stderr,"usage: %s <in.vca v1> <out.vca v2> [--threads N]\n",argv[0]);return 2;}
    const char *inpath=argv[1], *outpath=argv[2];
    int nthreads=0;
    for(int i=3;i<argc;++i){ if(!strcmp(argv[i],"--threads")&&i+1<argc) nthreads=atoi(argv[++i]); }
    if(nthreads<=0){ long n=sysconf(_SC_NPROCESSORS_ONLN); nthreads=n>1?(int)n:1; }
    if(nthreads>256) nthreads=256;

    // ---- map the v1 input + open it through libvc (validates + gives dims/nlod) ----
    int ifd=open(inpath,O_RDONLY); if(ifd<0){perror("open in");return 1;}
    struct stat ist; fstat(ifd,&ist); u64 ilen=ist.st_size;
    const u8 *ibuf=mmap(NULL,ilen,PROT_READ,MAP_PRIVATE,ifd,0);
    if(ibuf==MAP_FAILED){perror("mmap in");return 1;}
    madvise((void*)ibuf, ilen, MADV_RANDOM);   // scattered v1 reads: don't readahead
    vc_archive *in=vc_open(ibuf,ilen);
    if(!in){fprintf(stderr,"vc_open(in) failed (not a v1 .vca?)\n");return 1;}
    u32 nx=rd32(ibuf+FH_NX),ny=rd32(ibuf+FH_NY),nz=rd32(ibuf+FH_NZ);
    int nlod=(int)rd32(ibuf+FH_NLOD);
    printf("input v%u: %ux%ux%u (x,y,z) nlod=%d  %.1f GB  threads=%d\n",
           rd32(ibuf+FH_VER),nx,ny,nz,nlod,ilen/1e9,nthreads);

    // ---- enumerate regions COARSE-LOD-FIRST; assign canonical directory indices ----
    u64 prefix[VC_NLOD+1]; u64 total_regions=0;
    for(int l=0;l<nlod;l++){ prefix[l]=total_regions; u32 rz,ry,rx; region_grid(nx,ny,nz,l,&rz,&ry,&rx); total_regions+=(u64)rz*ry*rx; }
    prefix[nlod]=total_regions;
    printf("regions: %lu across %d LODs; directory = %lu KB\n",total_regions,nlod,total_regions*V2_DIR_ENTRY/1024);

    u64 dir_off=FILE_HDR;
    u64 blobs_off=dir_off+total_regions*V2_DIR_ENTRY;
    blobs_off=(blobs_off+4095)&~4095ull;   // 4KB-align the blob area

    // build the coarse-first region work list (this is also the blob layout order)
    ritem *items=malloc(total_regions*sizeof(ritem)); if(!items){fprintf(stderr,"oom items\n");return 1;}
    u64 ni=0;
    for(int l=nlod-1;l>=0;--l){
        u32 nrz,nry,nrx; region_grid(nx,ny,nz,l,&nrz,&nry,&nrx);
        for(u32 rz=0;rz<nrz;++rz)for(u32 ry=0;ry<nry;++ry)for(u32 rx=0;rx<nrx;++rx)
            items[ni++]=(ritem){ prefix[l]+((u64)rz*nry+ry)*nrx+rx, l, rz,ry,rx };
    }

    u8 *dir=calloc(total_regions, V2_DIR_ENTRY); if(!dir){fprintf(stderr,"oom dir\n");return 1;}
    u64 *blob_off=calloc(total_regions,sizeof(u64));
    u64 *blob_len=calloc(total_regions,sizeof(u64));
    if(!blob_off||!blob_len){fprintf(stderr,"oom blob arrays\n");return 1;}

    ctx c; memset(&c,0,sizeof c);
    c.in=in; c.ibuf=ibuf; c.nx=nx; c.ny=ny; c.nz=nz;
    c.items=items; c.nitems=total_regions; c.dir=dir; c.blob_off=blob_off; c.blob_len=blob_len;

    pthread_t th[256];
    // ---- PASS 1 (parallel): classify + size every region ----
    atomic_store(&c.next,0);
    for(int i=0;i<nthreads;++i) pthread_create(&th[i],NULL,pass1_worker,&c);
    for(int i=0;i<nthreads;++i) pthread_join(th[i],NULL);

    // ---- serial prefix-sum: assign each present region its contiguous blob offset
    // in the coarse-first item order, and fill its dir entry off/len ----
    u64 cursor=blobs_off;
    for(u64 i=0;i<ni;++i){
        u64 ridx=items[i].regidx;
        if(rd32(dir+ridx*V2_DIR_ENTRY+12)!=V2_RGN_DATA) continue;
        u64 blen=blob_len[ridx];
        blob_off[ridx]=cursor;
        u8 *e=dir+ridx*V2_DIR_ENTRY; wr64(e,cursor); wr32(e+8,(u32)blen);
        cursor+=blen;
    }
    u64 total_len=cursor;
    long preg=atomic_load(&c.present_regions), zreg=atomic_load(&c.zero_regions);
    printf("v2 size = %.1f GB (%ld present regions, %ld all-zero)\n",total_len/1e9,preg,zreg);

    // ---- create + map the output, sized exactly ----
    int ofd=open(outpath,O_RDWR|O_CREAT|O_TRUNC,0644); if(ofd<0){perror("open out");return 1;}
    if(ftruncate(ofd,(off_t)total_len)!=0){perror("ftruncate out");return 1;}
    u8 *obuf=mmap(NULL,total_len,PROT_READ|PROT_WRITE,MAP_SHARED,ofd,0);
    if(obuf==MAP_FAILED){perror("mmap out");return 1;}
    c.obuf=obuf;

    // ---- PASS 2 (parallel): copy each present region's slots+payloads verbatim ----
    atomic_store(&c.next,0);
    for(int i=0;i<nthreads;++i) pthread_create(&th[i],NULL,pass2_worker,&c);
    for(int i=0;i<nthreads;++i) pthread_join(th[i],NULL);

    // ---- header + directory ----
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
    free(items); free(dir); free(blob_off); free(blob_len);
    printf("done: %s  v2  %.1f GB  present_atoms=%lld\n",outpath,total_len/1e9,(long long)atomic_load(&c.present_atoms));
    return 0;
}
