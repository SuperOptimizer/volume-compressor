// vc_export_stream — single-pass, Z-pipelined pyramid export. Scales to 80k^3.
//
// Input: a level-0 OME-Zarr v2 (uncompressed u8 chunks). Output: one vc archive
// with ALL LODs, built in ONE streaming pass. LOD0 voxels are read once from the
// raw zarr; every coarser level is produced by 2x within-cell box downscales IN
// RAM as the stream flows up the Z axis. Because the downscale is strictly
// within-cell (each output voxel = its own 2x2x2 parents, no neighbor reach),
// there is NO halo and tiles are fully independent.
//
// Pipelining: the volume is split into XY column-tiles (full Z). Within a tile we
// march Z bottom-up; a push-driven cascade feeds every level concurrently (level
// L+1 consumes pairs of L's planes as they appear). LOD0 is decoded/read ONCE;
// no level is ever re-read. Each level's atoms are compressed+written once.
//
// Usage: vc_export_stream <zarr0> <out.vca>
//          [--ratio R] [--threads N] [--downscale box|cbox] [--alpha A]
//          [--mem MB] [--tile T]
#define _GNU_SOURCE
#include "../src/vc/vc.h"
#include "third_party/cJSON.h"
#include "third_party/libs3/libs3.h"
#include "downscale.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdatomic.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

typedef uint8_t u8; typedef uint32_t u32; typedef uint64_t u64;
#define A 32u

// ------------------------------------------------------------------ source I/O
// The source zarr lives either on a local filesystem or in S3 (s3://bucket/key).
// One abstraction: a path is an S3 URL iff it starts with "s3://". A single
// shared, thread-safe s3_client is created once if any S3 path is used (the
// vendored libs3 makes concurrent transfers safe: NOSIGNAL + curl-init mutex).
static s3_client *g_s3 = NULL;             // NULL until first S3 path seen
static int src_is_s3(const char *p){ return strncmp(p,"s3://",5)==0; }
// Per-request credential resolver: cache-served IMDSv2 (EC2 instance role),
// refreshed before expiry -> safe for multi-hour exports on rotating STS creds.
static s3_status src_cred_provider(void *ud, s3_credentials *out){
    (void)ud; return s3_credentials_load(NULL, out);
}
static void src_init_s3(void){
    if(g_s3) return;
    s3_config cfg; memset(&cfg,0,sizeof cfg);
    cfg.max_retries=5;
    // Resolve creds once up front (IMDS/SSO/env) so a misconfigured host fails
    // loudly here, not mid-stream; empty -> anonymous (public buckets).
    s3_credentials probe; memset(&probe,0,sizeof probe);
    if(s3_credentials_load(NULL,&probe)==S3_OK){
        cfg.cred_provider=src_cred_provider;
        if(probe.region&&probe.region[0]) cfg.region=strdup(probe.region);
        s3_credentials_free(&probe);
        fprintf(stderr,"s3: using resolved credentials (IMDS/SSO/env)\n");
    } else {
        fprintf(stderr,"s3: no credentials found; using anonymous access\n");
    }
    g_s3=s3_client_new(&cfg);
    if(!g_s3){ fprintf(stderr,"s3_client_new failed\n"); exit(1); }
}
// GET an object fully into a malloc'd buffer (caller frees). *len set. Returns
// NULL on 404 / error. For S3, this is a single GET; for local, a file read.
static u8 *src_get(const char *path, size_t *len){
    if(src_is_s3(path)){
        s3_response r; memset(&r,0,sizeof r);
        s3_status st=s3_get(g_s3, path, &r);
        u8 *out=NULL;
        if(st==S3_OK && r.status==200 && r.body){ out=malloc(r.body_len); if(out){ memcpy(out,r.body,r.body_len); if(len)*len=r.body_len; } }
        s3_response_free(&r);
        return out;
    }
    int fd=open(path,O_RDONLY);
    if(fd<0) return NULL;
    struct stat stt;
    if(fstat(fd,&stt)){close(fd);return NULL;}
    if(stt.st_size<0||!S_ISREG(stt.st_mode)){close(fd);return NULL;}
    size_t n=(size_t)stt.st_size; u8 *b=malloc(n); if(!b){close(fd);return NULL;}
    ssize_t g=read(fd,b,n); close(fd);
    if(g!=(ssize_t)n){free(b);return NULL;}
    if(len)*len=n;
    return b;
}
// Does an object exist? (occupancy probe.) HEAD for S3, access() for local.
static int src_exists(const char *path){
    if(src_is_s3(path)){
        s3_response r; memset(&r,0,sizeof r);
        s3_status st=s3_head(g_s3, path, &r);
        int ok=(st==S3_OK && r.status>=200 && r.status<300);
        s3_response_free(&r);
        return ok;
    }
    return access(path,F_OK)==0;
}

// Copy exactly A(=32) bytes. Intrinsic-free (like vc.c): __builtin_memcpy with a
// compile-time-constant size lowers to inline vector moves under -march — one ymm
// move on AVX2, two q-reg moves on NEON, etc. — with NO libc call/dispatch. This
// reaches ARM/AVX2/AVX512 from the same generic source via -march, no #ifdefs.
static inline void copy_row32(u8 *d, const u8 *s){ __builtin_memcpy(d,s,A); }
#define R_ATOMS 32u   // index-region edge in atoms (must match vc_region_atoms())
extern u32 vc_test_atom_rt(const u8 vox[VC_ATOM3], float q, u8 *recon);

// I/O accounting: total chunk opens, misses (absent chunks), bytes read. Lets us
// verify we read each source chunk exactly once (no extraneous re-reads).
static atomic_long g_io_open=0, g_io_miss=0, g_io_bytes=0;

// ------------------------------------------------------------------ zarr read
typedef struct { char dir[1024]; u32 sz,sy,sx, cz,cy,cx; } zlevel;

// Read a whole text resource (local file OR s3:// object) into a NUL-terminated
// malloc'd buffer for cJSON. Routes through src_get (handles both).
static char *read_text(const char *p){
    size_t n=0; u8 *raw=src_get(p,&n); if(!raw) return NULL;
    char *b=realloc(raw,n+1); if(!b){ free(raw); return NULL; }
    b[n]=0; return b;
}
// Parse the .zarray for one pyramid level L of the zarr (dir = "<zarr>/<L>").
static int parse_zarray_lvl(const char *zarr, int L, zlevel *lv){
    char p[1100]; snprintf(lv->dir,sizeof lv->dir,"%s/%d",zarr,L);
    snprintf(p,sizeof p,"%s/.zarray",lv->dir);
    char *t=read_text(p); if(!t)return -1; cJSON *j=cJSON_Parse(t); free(t); if(!j)return -1;
    cJSON *sh=cJSON_GetObjectItem(j,"shape"), *ch=cJSON_GetObjectItem(j,"chunks");
    cJSON *dt=cJSON_GetObjectItem(j,"dtype"), *co=cJSON_GetObjectItem(j,"compressor");
    cJSON *sep=cJSON_GetObjectItem(j,"dimension_separator");
    int ok = sh&&ch&&cJSON_GetArraySize(sh)==3;
    if(ok){ lv->sz=cJSON_GetArrayItem(sh,0)->valueint; lv->sy=cJSON_GetArrayItem(sh,1)->valueint; lv->sx=cJSON_GetArrayItem(sh,2)->valueint;
            lv->cz=cJSON_GetArrayItem(ch,0)->valueint; lv->cy=cJSON_GetArrayItem(ch,1)->valueint; lv->cx=cJSON_GetArrayItem(ch,2)->valueint; }
    if(ok&&dt&&dt->valuestring&&!strstr(dt->valuestring,"u1")){fprintf(stderr,"dtype %s not u8\n",dt->valuestring);ok=0;}
    if(ok&&co&&!cJSON_IsNull(co)){fprintf(stderr,"compressor not null unsupported\n");ok=0;}
    if(ok&&sep&&sep->valuestring&&strcmp(sep->valuestring,"/")){fprintf(stderr,"dim sep not '/'\n");ok=0;}
    cJSON_Delete(j); return ok?0:-1;
}
static int parse_zarray0(const char *zarr, zlevel *lv){ return parse_zarray_lvl(zarr,0,lv); }

// ------------------------------------------------------------------ helpers
static int npyr(u32 sz,u32 sy,u32 sx){ int n=1; u32 z=sz,y=sy,x=sx; for(int l=1;l<VC_NLOD;++l){z=(z+1)/2;y=(y+1)/2;x=(x+1)/2;n=l+1;if(z<=1&&y<=1&&x<=1)break;} return n; }

// A-priori occupancy: a present-chunk bitmap of the source level-0 grid. A
// MISSING source chunk means that whole 128³(or whatever cz) extent is zero —
// and since downscaling preserves zero-ness, it is zero at EVERY coarser level
// too. We build this with a cheap stat() scan (no reads), then workers use it to
// SKIP fully-empty tiles entirely (no chunk reads, no downscale, no zero-marks).
typedef struct { u8 *present; u32 ncz,ncy,ncx; } chunkmap;

int g_scan_threads = 1;   // worker count for fetching the coarse occupancy level

// Build the present-chunk bitmap from a COARSE pyramid level instead of probing
// every 0/ chunk. The zarr is box-downscaled (zero-preserving), so a coarse-level
// voxel is zero IFF all its 0/ parents are zero. We pick the coarsest level Lc
// whose 2^Lc factor divides the 0/ chunk size (so a 0/ chunk maps to a whole
// integer block of Lc voxels), download that tiny level in full, and mark a 0/
// chunk present iff its footprint at Lc has any nonzero voxel. For this volume Lc
// is 5/ -> one 128x71x71 chunk (<1MB, a single GET) replaces ~10k HEADs. Scales:
// the coarse level is 2^(3*Lc) smaller than 0/. Falls back to per-chunk probing
// only if no usable coarse level exists.
#define CM_OCC_LEVEL 4   // preferred occupancy level: 4/ (16x). Coarser (5/, 32x)
                         // averages faint edge data to zero; 4/ keeps more of the
                         // feathered surface. A 0/ chunk maps to (chunk/16)^3 voxels.
// Optional separate root for the coarse occupancy level (e.g. a local mirror of
// just N/ synced ahead of time), so the mask probe reads from fast local disk
// while voxel data still streams from the (remote) main root. NULL -> use `zarr`.
const char *g_occ_zarr = NULL;
static int cm_fill_from_coarse(chunkmap *cm, const char *zarr, const zlevel *z0){
    if(g_occ_zarr) zarr = g_occ_zarr;   // read the coarse mask level from here
    // Prefer CM_OCC_LEVEL; if it's absent or its factor doesn't tile a 0/ chunk
    // cleanly, fall back to the next-coarsest usable level (then finer).
    int bestL=-1; zlevel cl; u32 f=1;
    for(int pass=0; pass<2 && bestL<0; ++pass){
        // pass 0: try exactly CM_OCC_LEVEL. pass 1: scan coarsest..finest.
        int lo = pass==0?CM_OCC_LEVEL:1, hi = pass==0?CM_OCC_LEVEL:VC_NLOD-1;
        for(int L=hi; L>=lo; --L){
            u32 ff=1u<<L;
            if(z0->cz % ff || z0->cy % ff || z0->cx % ff) continue;  // must tile a 0/ chunk cleanly
            zlevel t; if(parse_zarray_lvl(zarr,L,&t)!=0) continue;    // level must exist
            cl=t; f=ff; bestL=L; break;
        }
    }
    if(bestL<0) return -1;
    // voxels-per-0/chunk at level Lc on each axis
    u32 vcz=z0->cz/f, vcy=z0->cy/f, vcx=z0->cx/f;
    // download the whole coarse level into a dense buffer [clz][cly][clx]
    size_t voxn=(size_t)cl.sz*cl.sy*cl.sx; u8 *buf=calloc(voxn,1); if(!buf){return -1;}
    u32 nclz=(cl.sz+cl.cz-1)/cl.cz, ncly=(cl.sy+cl.cy-1)/cl.cy, nclx=(cl.sx+cl.cx-1)/cl.cx;
    size_t cb=(size_t)cl.cz*cl.cy*cl.cx; u8 *chk=malloc(cb);
    for(u32 qz=0;qz<nclz;++qz)for(u32 qy=0;qy<ncly;++qy)for(u32 qx=0;qx<nclx;++qx){
        char p[1200]; snprintf(p,sizeof p,"%s/%u/%u/%u",cl.dir,qz,qy,qx);
        size_t gn=0; u8 *got=src_get(p,&gn);
        if(!got) continue;                                   // absent chunk -> stays zero
        if(gn>cb) gn=cb;
        memcpy(chk,got,gn);
        if(gn<cb) memset(chk+gn,0,cb-gn);
        free(got);
        // scatter this coarse chunk into buf (clamp to true coarse dims)
        u32 gz0=qz*cl.cz, gy0=qy*cl.cy, gx0=qx*cl.cx;
        for(u32 z=0; z<cl.cz && gz0+z<cl.sz; ++z)
          for(u32 y=0; y<cl.cy && gy0+y<cl.sy; ++y){
            u32 w=cl.cx; if(gx0+w>cl.sx) w=cl.sx-gx0;
            memcpy(buf+((size_t)(gz0+z)*cl.sy+(gy0+y))*cl.sx+gx0,
                   chk+((size_t)z*cl.cy+y)*cl.cx, w);
          }
    }
    free(chk);
    // mark each 0/ chunk present iff its [vcz x vcy x vcx] footprint at Lc is nonzero
    for(u32 cz=0;cz<cm->ncz;++cz)for(u32 cy=0;cy<cm->ncy;++cy)for(u32 cx=0;cx<cm->ncx;++cx){
        u32 z0v=cz*vcz, y0v=cy*vcy, x0v=cx*vcx; int nz=0;
        for(u32 z=z0v; z<z0v+vcz && z<cl.sz && !nz; ++z)
          for(u32 y=y0v; y<y0v+vcy && y<cl.sy && !nz; ++y)
            for(u32 x=x0v; x<x0v+vcx && x<cl.sx; ++x)
              if(buf[((size_t)z*cl.sy+y)*cl.sx+x]){ nz=1; break; }
        if(nz) cm->present[((size_t)cz*cm->ncy+cy)*cm->ncx+cx]=1;
    }
    free(buf);
    printf("occupancy from coarse level %d/ (%ux%ux%u, factor %u)\n", bestL, cl.sz,cl.sy,cl.sx, f);
    return 0;
}

static int cm_build(chunkmap *cm, const char *zarr, const zlevel *z0){
    cm->ncz=(z0->sz+z0->cz-1)/z0->cz; cm->ncy=(z0->sy+z0->cy-1)/z0->cy; cm->ncx=(z0->sx+z0->cx-1)/z0->cx;
    size_t n=(size_t)cm->ncz*cm->ncy*cm->ncx;
    cm->present=calloc(n,1); if(!cm->present) return -1;
    // Preferred: derive occupancy from a coarse pyramid level (one tiny download).
    if(cm_fill_from_coarse(cm, zarr, z0)==0) return 0;
    // Fallback: probe each 0/ chunk directly (access() local / HEAD on S3).
    fprintf(stderr,"no usable coarse level; falling back to per-chunk existence probe\n");
    for(u32 cz=0;cz<cm->ncz;++cz)for(u32 cy=0;cy<cm->ncy;++cy)for(u32 cx=0;cx<cm->ncx;++cx){
        char p[1200]; snprintf(p,sizeof p,"%s/%u/%u/%u",z0->dir,cz,cy,cx);
        if(src_exists(p)) cm->present[((size_t)cz*cm->ncy+cy)*cm->ncx+cx]=1;
    }
    return 0;
}
static void cm_free(chunkmap *cm){ free(cm->present); }

// ------------------------------------------------------------------ pipeline
// A SUPERCUBE COLUMN: 4096x4096 XY (SB = A*2^(nlod-1), region-aligned at EVERY
// level so nothing straddles), streamed in Z. We march Z in source-chunk bands;
// each LOD0 z-plane is pushed into a per-level cascade that holds only the few
// z-planes each level needs (a 2-plane downscale pair + a 32-plane atom slab),
// flushing atom-planes as they complete and feeding the level above. After the
// full Z of the cube, every level's atoms are written — self-contained, no tail.
// RAM ~ sum_L (SB/2^L)^2 * ~34 planes ~= 760MB per active cube. Cubes are
// independent across XY*Z grid -> parallel.

typedef struct {
    u32 nx, ny;        // XY voxel extent at this level (atom-padded)
    u32 acx, acy;      // atom counts in XY
    u32 ax0, ay0;      // global atom-coord origin at this level
    u8 *pair;          // nx*ny*2  downscale Z-pair
    int pairfill;
    u8 *slab;          // nx*ny*A  current atom-plane (32 Z deep)
    int slabfill;
    u32 zatom;         // next atom-Z index to flush
} plvl;

typedef struct {
    vc_writer *w; const zlevel *z0; int nlod;
    u32 px,py,pz;             // PADDED output dims (multiple of region; grid extent)
    ds_method method; float alpha;
    u32 SB;                   // XY tile = region-aligned cube footprint (4096)
    u32 BAND;                 // Z-band height (LOD0 voxels), multiple of A*2^FINE
    int fine;                 // deepest level a band produces alone (LOD0..fine)
    const chunkmap *cm;
    u32 ntx,nty; long nbands; // XY-tile grid + total (tile,band) units
    u32 nbz;                  // Z-bands per XY column
    atomic_long next, present, zero, skipped;
} sched;

typedef struct { sched *sc; int top; plvl lv[VC_NLOD]; u8 atom[A*A*A]; long present, zero; } cube;

// A region at level L covers REG=1024 voxels/axis at level L = REG*2^L LOD0
// voxels. It is DEFINITE-ZERO iff every source chunk overlapping that LOD0 extent
// is absent (a missing zarr chunk = no data; downscaling preserves zero). Test it.
static int region_definite_zero(const chunkmap *cm, const zlevel *z0, int L,
                                u32 rz,u32 ry,u32 rx){
    u32 REGv = (u32)R_ATOMS*A;                 // region voxels at level L (1024)
    // region's LOD0 voxel extent
    u64 vx0=(u64)rx*REGv<<L, vy0=(u64)ry*REGv<<L, vz0=(u64)rz*REGv<<L;
    u64 vx1=vx0+((u64)REGv<<L), vy1=vy0+((u64)REGv<<L), vz1=vz0+((u64)REGv<<L);
    if(vx0>=z0->sx||vy0>=z0->sy||vz0>=z0->sz) return 1;   // fully outside -> zero
    if(vx1>z0->sx)vx1=z0->sx;
    if(vy1>z0->sy)vy1=z0->sy;
    if(vz1>z0->sz)vz1=z0->sz;
    u32 cxa=vx0/z0->cx,cxb=(vx1-1)/z0->cx, cya=vy0/z0->cy,cyb=(vy1-1)/z0->cy, cza=vz0/z0->cz,czb=(vz1-1)/z0->cz;
    for(u32 qz=cza;qz<=czb;++qz)for(u32 qy=cya;qy<=cyb;++qy)for(u32 qx=cxa;qx<=cxb;++qx)
        if(cm->present[((size_t)qz*cm->ncy+qy)*cm->ncx+qx]) return 0;  // some data
    return 1;
}

static void push_plane(cube *c, int L, const u8 *plane);

static void flush_slab(cube *c, int L){
    plvl *lv=&c->lv[L];
    size_t plane=(size_t)lv->ny*lv->nx;
    for(u32 ay=0;ay<lv->acy;++ay)for(u32 ax=0;ax<lv->acx;++ax){
        // Gather the 32x32x32 atom contiguously: 32 Z-planes x 32 rows, each row a
        // contiguous 32-byte run in the slab (nx-strided). memcpy per row lets the
        // compiler use vector moves instead of the old byte-by-byte scalar loop.
        const u8 *base=lv->slab + (size_t)(ay*A)*lv->nx + (ax*A);
        u8 *o=c->atom;
        for(u32 z=0;z<A;++z){ const u8 *sp=base+(size_t)z*plane;
            for(u32 y=0;y<A;++y){ copy_row32(o, sp+(size_t)y*lv->nx); o+=A; } }
        // Nonzero test over the packed atom, 8 bytes at a time.
        const u64 *w64=(const u64*)c->atom; u64 acc=0;
        for(u32 i=0;i<(A*A*A)/8u;++i) acc|=w64[i];
        u32 gaz=lv->zatom, gay=lv->ay0+ay, gax=lv->ax0+ax;
        if(!acc){ vc_mark_zero_atom(c->sc->w,L,gaz,gay,gax); c->zero++; }
        else { vc_append_atom(c->sc->w,L,gaz,gay,gax,c->atom); c->present++; }
    }
    lv->zatom++; lv->slabfill=0;
}

// push one voxel-plane into level L: accumulate into the atom slab (flush at 32)
// and the downscale pair (on the 2nd, downscale -> 1 plane of L+1).
static void push_plane(cube *c, int L, const u8 *plane){
    plvl *lv=&c->lv[L];
    size_t pl=(size_t)lv->ny*lv->nx;
    u8 *dst=lv->slab+(size_t)lv->slabfill*pl;
    memcpy(dst, plane, pl);
    // The downscale needs 2 consecutive planes. A pair always completes on an ODD
    // slab index (pairfill hits 2 when sf is 1,3,5,...), and flush_slab only resets
    // slabfill — it never overwrites slab data — so the two planes [sf-1][sf] are
    // always still contiguous in the slab when we downscale. No separate `pair`
    // buffer, no extra plane copy: feed ds_downscale2x straight from the slab.
    int sf=lv->slabfill;            // index this plane occupies (pre-increment)
    if(++lv->slabfill==(int)A) flush_slab(c,L);
    if(L+1<=c->top){
        if(++lv->pairfill==2){
            plvl *nl=&c->lv[L+1]; int ox,oy,oz;
            static _Thread_local u8 *tmp=NULL; static _Thread_local size_t cap=0;
            size_t need=(size_t)nl->nx*nl->ny; if(need>cap){free(tmp);tmp=malloc(need);cap=need;}
            const u8 *p2=lv->slab+(size_t)(sf-1)*pl;   // [sf-1][sf], contiguous
            ds_downscale2x(p2, lv->nx, lv->ny, 2, tmp, &ox,&oy,&oz, c->sc->method, c->sc->alpha);
            push_plane(c, L+1, tmp);
            lv->pairfill=0;
        }
    }
}

// flush partial pairs/slabs at the end of the cube's Z (pad with zeros).
static void finish_cube(cube *c){
    for(int L=0; L<=c->top; ++L){
        plvl *lv=&c->lv[L];
        if(L+1<=c->top && lv->pairfill==1){
            plvl *nl=&c->lv[L+1]; int ox,oy,oz;
            size_t pl=(size_t)lv->ny*lv->nx;
            // The lone unpaired plane is the last one pushed, at slab[slabfill-1].
            // Build a 2-plane pair in lv->pair scratch: [lone][zero], downscale it.
            memcpy(lv->pair, lv->slab+(size_t)(lv->slabfill-1)*pl, pl);
            memset(lv->pair+pl, 0, pl);
            static _Thread_local u8 *tmp=NULL; static _Thread_local size_t cap=0;
            size_t need=(size_t)nl->nx*nl->ny; if(need>cap){free(tmp);tmp=malloc(need);cap=need;}
            ds_downscale2x(lv->pair, lv->nx, lv->ny, 2, tmp,&ox,&oy,&oz,c->sc->method,c->sc->alpha);
            push_plane(c,L+1,tmp); lv->pairfill=0;
        }
        if(lv->slabfill>0){
            memset(lv->slab+(size_t)lv->slabfill*lv->ny*lv->nx, 0, (size_t)(A-lv->slabfill)*lv->ny*lv->nx);
            flush_slab(c,L);
        }
    }
}

// Each work unit is one (XY-tile, Z-band): a BAND x SB x SB voxel slab, processed
// SINGLE-THREADED internally (the cascade), producing LOD0..fine atoms for that
// band's Z range. Parallelism is ACROSS units. A band is BAND voxels tall with
// BAND a multiple of A*2^fine, so it produces whole atoms at LOD0..fine at
// absolute Z offsets — disjoint from every other band, so no coordination.
static void *tile_worker(void *arg){
    sched *sc=(sched*)arg; const zlevel *z0=sc->z0; u32 SB=sc->SB, BAND=sc->BAND; int fine=sc->fine;
    size_t cb=(size_t)z0->cz*z0->cy*z0->cx; u8 *chk=malloc(cb);
    cube C; memset(&C,0,sizeof C); C.sc=sc;

    // Per-THREAD scratch, allocated ONCE and reused across every unit this thread
    // processes — not per-unit. Per-unit calloc/free of the cascade buffers forced
    // the kernel to first-touch-zero fresh pages every time (showed up as ~10% in
    // alloc_anon_folio/kernel_init_pages). All units share the same SB footprint,
    // so a single max-size allocation per level serves them all. slab is fully
    // overwritten before each flush (finish_cube zero-pads the tail), and bbuf is
    // memset per chunk-band, so reuse needs no extra clearing. pair holds 2 planes.
    size_t bbuf_cap=0; u8 *bbuf=NULL;
    for(int L=0;L<=fine;++L){
        u32 mnx=SB>>L, mny=SB>>L;
        C.lv[L].pair=malloc((size_t)mnx*mny*2);
        C.lv[L].slab=malloc((size_t)mnx*mny*A);
    }

    for(;;){
        long ti=atomic_fetch_add(&sc->next,1);
        if(ti>=sc->nbands) break;
        u32 bz   = ti % sc->nbz;
        u32 txy  = ti / sc->nbz;
        u32 ty   = txy / sc->ntx, tx = txy % sc->ntx;
        // Extents clamp to PADDED dims (so cube footprint is full SB, band full
        // BAND — no partial atoms). Source reads clamp to TRUE dims (z0->s*),
        // padding reads as zero.
        u32 vx0=tx*SB, vy0=ty*SB, vz0=bz*BAND;
        u32 vx1=vx0+SB<sc->px?vx0+SB:sc->px;
        u32 vy1=vy0+SB<sc->py?vy0+SB:sc->py;
        u32 vz1=vz0+BAND<sc->pz?vz0+BAND:sc->pz;
        u32 corex=vx1-vx0, corey=vy1-vy0;

        // FULLY-EMPTY-BAND fast path. If every source chunk covering this band is
        // absent (a-priori occupancy), the band is all zero: NO read, NO downscale,
        // NO codec. But we must NOT just `continue` — the band still OWNS LOD0..fine
        // atoms, and its coarse-level atoms live in regions shared with neighbouring
        // tiles (a LOD2 region spans 4x4 tiles), so the prepass can't have marked
        // them. So we directly STAMP every atom this band owns as KNOWN_ZERO — the
        // minimum possible work (a slot write per atom, no 32^3 data touched). This
        // is the bulk of the volume (~80% of chunks absent) and skips ALL the
        // expensive per-band work for it while keeping ABSENT=0.
        if(sc->cm){
            int any=0;
            // chunk extent covering this band (clamped to the true volume; padding
            // chunks are absent by definition).
            u32 cxa=vx0/z0->cx, cxb=(vx1-1)/z0->cx, cya=vy0/z0->cy, cyb=(vy1-1)/z0->cy;
            u32 cza=vz0/z0->cz, czb=(vz1-1)/z0->cz;
            for(u32 qz=cza;qz<=czb&&!any;++qz){ if(qz>=sc->cm->ncz)break;
              for(u32 qy=cya;qy<=cyb&&!any;++qy){ if(qy>=sc->cm->ncy)break;
                for(u32 qx=cxa;qx<=cxb;++qx){ if(qx>=sc->cm->ncx)break;
                  if(sc->cm->present[((size_t)qz*sc->cm->ncy+qy)*sc->cm->ncx+qx]){any=1;break;} }}}
            if(!any){
                // stamp KNOWN_ZERO for every owned atom at LOD0..fine
                for(int L=0;L<=fine;++L){
                    u32 az0=(vz0>>L)/A, ay0=(vy0>>L)/A, ax0=(vx0>>L)/A;
                    u32 nz=((vz1-vz0)>>L)/A, nyc=((vy1-vy0)>>L)/A, nxc=((vx1-vx0)>>L)/A;
                    for(u32 az=0;az<nz;++az)for(u32 ay=0;ay<nyc;++ay)for(u32 ax=0;ax<nxc;++ax)
                        vc_mark_zero_atom(sc->w,L,az0+az,ay0+ay,ax0+ax);
                    C.zero += (long)nz*nyc*nxc;
                }
                atomic_fetch_add(&sc->zero,C.zero);
                C.present=0; C.zero=0;
                atomic_fetch_add(&sc->skipped,1); continue;
            }
        }

        // Cascade buffers. corex/corey are multiples of the region size (1024)
        // because the OUTPUT volume is padded to 1024 — so every level halves
        // CLEANLY into whole atoms (no partials, no stride mismatch).
        for(int L=0;L<=fine;++L){
            plvl *lv=&C.lv[L];
            lv->nx = corex>>L; lv->ny = corey>>L;   // exact halving (corex %1024==0)
            lv->acx=lv->nx/A; lv->acy=lv->ny/A;
            lv->ax0=(vx0>>L)/A; lv->ay0=(vy0>>L)/A;
            lv->pairfill=0; lv->slabfill=0; lv->zatom=(vz0>>L)/A;
        }
        C.present=0; C.zero=0; C.top=fine;   // cascade stops at `fine`

        // stream this band's Z in source-chunk sub-bands; read each chunk once.
        plvl *l0=&C.lv[0];
        size_t bneed=(size_t)z0->cz*l0->nx*l0->ny;
        if(bneed>bbuf_cap){ free(bbuf); bbuf=malloc(bneed); bbuf_cap=bneed; }
        u32 cz=z0->cz,cy=z0->cy,cx=z0->cx;
        u32 cxa=vx0/cx,cxb=(vx1-1)/cx, cya=vy0/cy,cyb=(vy1-1)/cy;
        for(u32 zz0=vz0; zz0<vz1; ){
            u32 ccz=zz0/cz; u32 sz0=ccz*cz, sz1=sz0+cz<vz1?sz0+cz:vz1;
            // Zero-fill the band buffer ONLY if some covering chunk is absent or the
            // tile clips the true volume edge (those gaps need zeros). When every
            // chunk in this XY footprint is present AND the tile is fully interior,
            // the reads overwrite every byte — skip the memset (the dense core is the
            // slow path, so this is where it matters most). cm may be NULL.
            int need_zero = 1;
            if(sc->cm && vx1<=z0->sx && vy1<=z0->sy && sz1<=z0->sz){
                need_zero=0;
                for(u32 ccy=cya;ccy<=cyb && !need_zero;++ccy)for(u32 ccx=cxa;ccx<=cxb;++ccx)
                    if(!sc->cm->present[((size_t)ccz*sc->cm->ncy+ccy)*sc->cm->ncx+ccx]){ need_zero=1; break; }
            }
            if(need_zero) memset(bbuf,0,(size_t)cz*l0->nx*l0->ny);
            for(u32 ccy=cya;ccy<=cyb;++ccy)for(u32 ccx=cxa;ccx<=cxb;++ccx){
                // The a-priori occupancy map already knows which chunks exist. Never
                // open() a chunk it says is absent — bbuf is already zero there (the
                // memset above ran because need_zero is forced when any covering
                // chunk is absent). This removes a pointless syscall per absent chunk
                // (the volume is ~80% absent chunks -> the dominant open() cost).
                if(sc->cm && (ccz>=sc->cm->ncz||ccy>=sc->cm->ncy||ccx>=sc->cm->ncx ||
                              !sc->cm->present[((size_t)ccz*sc->cm->ncy+ccy)*sc->cm->ncx+ccx])){
                    atomic_fetch_add(&g_io_miss,1); continue;
                }
                char pth[1200]; snprintf(pth,sizeof pth,"%s/%u/%u/%u",z0->dir,ccz,ccy,ccx);
                atomic_fetch_add(&g_io_open,1);
                if(src_is_s3(pth)){
                    size_t gn=0; u8 *got=src_get(pth,&gn);          // S3 GET
                    if(!got){ atomic_fetch_add(&g_io_miss,1); continue; }
                    if(gn==cb) memcpy(chk,got,cb); else { memset(chk,0,cb); if(gn) memcpy(chk,got,gn<cb?gn:cb); }
                    free(got);
                }else{
                    int fd=open(pth,O_RDONLY);
                    if(fd<0){ atomic_fetch_add(&g_io_miss,1); continue; }
                    if(read(fd,chk,cb)!=(ssize_t)cb) memset(chk,0,cb);
                    close(fd);
                }
                atomic_fetch_add(&g_io_bytes,cb);
                u32 gx0=ccx*cx, gy0=ccy*cy;
                u32 ox0=gx0>vx0?gx0:vx0, ox1=(gx0+cx)<vx1?(gx0+cx):vx1;
                u32 oy0=gy0>vy0?gy0:vy0, oy1=(gy0+cy)<vy1?(gy0+cy):vy1;
                for(u32 z=sz0; z<sz1; ++z){ u32 iz=z-sz0;
                    u8 *bp=bbuf+(size_t)iz*l0->nx*l0->ny;
                    for(u32 gy=oy0;gy<oy1;++gy){
                        const u8 *s=chk+((size_t)iz*cy+(gy-gy0))*cx+(ox0-gx0);
                        memcpy(bp+(size_t)(gy-vy0)*l0->nx+(ox0-vx0), s, ox1-ox0);
                    }
                }
            }
            u32 s = zz0>sz0?zz0:sz0;
            for(u32 z=s; z<sz1; ++z)
                push_plane(&C,0, bbuf+(size_t)(z-sz0)*l0->nx*l0->ny);
            zz0=sz1;
        }
        finish_cube(&C);
        atomic_fetch_add(&sc->present,C.present); atomic_fetch_add(&sc->zero,C.zero);
        C.present=0; C.zero=0;
    }
    for(int L=0;L<=fine;++L){ free(C.lv[L].pair); free(C.lv[L].slab); }
    free(bbuf);
    free(chk);
    return NULL;
}

// ----------------------------------- coarse tail: build one level L from L-1
// (in the archive) by 2x downscale. dst atoms are independent; threads pull
// atom-Z-planes from a shared counter. Tiny (top 0.2% of atoms).
typedef struct {
    vc_writer *w; int L; u32 dacx,dacy,dacz;
    ds_method method; float alpha;
    atomic_uint_least32_t next_z; atomic_long present, zero;
} tail_ctx;
static void *tail_worker(void *arg){
    tail_ctx *c=(tail_ctx*)arg;
    u8 *src=malloc((size_t)(2*A)*(2*A)*(2*A)); u8 datom[A*A*A]; u8 blk8[8][A*A*A];
    long pp=0,zz=0;
    for(;;){
        u32 daz=atomic_fetch_add(&c->next_z,1);
        if(daz>=c->dacz) break;
        for(u32 day=0;day<c->dacy;++day)for(u32 dax=0;dax<c->dacx;++dax){
            int any=0; vc_writer_decode_2x2x2(c->w, c->L-1, daz*2, day*2, dax*2, blk8, &any);
            if(!any){ vc_mark_zero_atom(c->w,c->L,daz,day,dax); zz++; continue; }
            for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
                u8 *sa=blk8[(dz*2+dy)*2+dx];
                for(u32 z=0;z<A;++z)for(u32 y=0;y<A;++y)for(u32 x=0;x<A;++x)
                    src[((size_t)(dz*A+z)*(2*A)+(dy*A+y))*(2*A)+(dx*A+x)]=sa[(z*A+y)*A+x];
            }
            int ox,oy,oz; ds_downscale2x(src,2*A,2*A,2*A,datom,&ox,&oy,&oz,c->method,c->alpha);
            int nz=0; for(int i=0;i<(int)(A*A*A);++i) if(datom[i]){nz=1;break;}
            if(!nz){ vc_mark_zero_atom(c->w,c->L,daz,day,dax); zz++; }
            else { vc_append_atom(c->w,c->L,daz,day,dax,datom); pp++; }
        }
    }
    free(src); atomic_fetch_add(&c->present,pp); atomic_fetch_add(&c->zero,zz);
    return NULL;
}

// ------------------------------------------------ progress monitor (for long
// EC2 runs). Polls the shared counters every `interval` seconds and prints
// atoms done, %, throughput, and ETA. Runs until `done` is set.
typedef struct {
    sched *sc; long total_atoms; double interval; double t0;
    atomic_int done;
} monitor;
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static void *monitor_thread(void *arg){
    monitor *m=(monitor*)arg;
    long last=0; double lastt=m->t0;
    while(!atomic_load(&m->done)){
        struct timespec req={ (time_t)m->interval, (long)((m->interval-(long)m->interval)*1e9) };
        nanosleep(&req,NULL);
        if(atomic_load(&m->done)) break;
        long pp=atomic_load(&m->sc->present), zz=atomic_load(&m->sc->zero);
        long did=pp+zz; double t=now_s();
        double inst = (t>lastt)? (did-last)/(t-lastt) : 0;       // atoms/s now
        double avg  = (t>m->t0)? did/(t-m->t0) : 0;              // atoms/s avg
        double frac = m->total_atoms? (double)did/m->total_atoms : 0;
        double eta  = (avg>0 && m->total_atoms>did)? (m->total_atoms-did)/avg : 0;
        double gb = atomic_load(&g_io_bytes)/1e9;
        printf("[%5.0fs] %ld/%ld atoms (%.1f%%) | %.0fk now, %.0fk avg atoms/s | %.1f GB read | ETA %.0fs\n",
               t-m->t0, did, m->total_atoms, 100*frac, inst/1e3, avg/1e3, gb, eta);
        last=did; lastt=t;
    }
    return NULL;
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv){
    if(argc<3){fprintf(stderr,"usage: %s <zarr0> <out.vca> [--q Q][--q-falloff F][--threads N][--downscale box|cbox][--alpha A][--mem MB][--tile T][--occupancy-zarr ROOT][--progress S]\n",argv[0]);return 2;}
    setvbuf(stdout,NULL,_IOLBF,0);
    const char *zarr=argv[1], *out=argv[2];
    float q0=1.0f, qfall=0.5f, alpha=0.5f; int nthreads=0; long mem_mb=1024; int tile_req=0;
    int log_io=0; double progress_s=2.0;
    ds_method method=DS_CBOX;
    for(int i=3;i<argc;++i){
        if(!strcmp(argv[i],"--q")&&i+1<argc)q0=atof(argv[++i]);
        else if(!strcmp(argv[i],"--q-falloff")&&i+1<argc)qfall=atof(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)nthreads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--downscale")&&i+1<argc)method=ds_parse(argv[++i]);
        else if(!strcmp(argv[i],"--alpha")&&i+1<argc)alpha=atof(argv[++i]);
        else if(!strcmp(argv[i],"--mem")&&i+1<argc)mem_mb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--tile")&&i+1<argc)tile_req=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--log-io"))log_io=1;
        else if(!strcmp(argv[i],"--progress")&&i+1<argc)progress_s=atof(argv[++i]);
        else if(!strcmp(argv[i],"--occupancy-zarr")&&i+1<argc)g_occ_zarr=argv[++i];
    }
    if(nthreads<=0){long n=sysconf(_SC_NPROCESSORS_ONLN);nthreads=n>2?(int)n-1:1;}

    // If the source is an s3:// URL, spin up the shared thread-safe S3 client and
    // thread the occupancy sweep (HEAD per chunk would be RTT-bound otherwise).
    if(src_is_s3(zarr)){ src_init_s3(); g_scan_threads=nthreads;
        printf("source is S3; %d-way occupancy sweep\n",nthreads); }

    zlevel z0;
    if(parse_zarray0(zarr,&z0)){fprintf(stderr,"cannot read level 0 .zarray\n");return 1;}
    // PAD the output volume up to a multiple of the region size (1024) in every
    // dimension. The archive's official dims are the padded dims; padded voxels
    // are zero (the source-read returns 0 outside z0->s*), so the padded atoms are
    // KNOWN_ZERO. This guarantees EVERY dimension is a whole number of regions at
    // every level -> NO partial atoms/regions, ever. (z0->s* keep the TRUE source
    // extent so reads clamp correctly; only the OUTPUT grid is padded.)
    u32 REGv = (u32)R_ATOMS*A;                 // 1024
    u32 px=((z0.sx+REGv-1)/REGv)*REGv, py=((z0.sy+REGv-1)/REGv)*REGv, pz=((z0.sz+REGv-1)/REGv)*REGv;
    int nlod=npyr(pz,py,px);
    printf("source 0/: %ux%ux%u (z,y,x) chunks %ux%ux%u\n",z0.sz,z0.sy,z0.sx,z0.cz,z0.cy,z0.cx);
    printf("padded to %ux%ux%u (multiple of %u) -> %d LODs, no partial atoms\n",pz,py,px,REGv,nlod);
    printf("downscale=%s alpha=%.2f q0=%.3f q-falloff=%.2f threads=%d mem=%ldMB\n",ds_name(method),alpha,q0,qfall,nthreads,mem_mb);

    vc_dims d0={px,py,pz};                     // PADDED dims = archive's official dims
    vc_writer *w=vc_create(out,d0,1.0f);
    if(!w){fprintf(stderr,"vc_create failed\n");return 1;}

    // NO calibration. The caller gives q0 (LOD0 quantizer step) and a per-level
    // falloff: q[l] = q0 * qfall^l. qfall<1 makes coarser levels higher quality
    // (lower q) — cheap since coarse levels are tiny. Caller re-runs with a
    // different q0 if the achieved ratio isn't what they want.
    {
        float q=q0;
        for(int l=0;l<nlod;++l){
            if(q<0.05f)q=0.05f;
            vc_set_base_q(w,l,q);
            printf("  LOD%d base_q=%.4f\n",l,q);
            q *= qfall;
        }
    }

    // PARALLEL UNIT = (XY-tile, Z-band). XY-tile = SB (region-aligned cube
    // footprint, 4096 for 8 LODs — so coarse atoms never straddle in XY). Z-band =
    // BAND voxels tall, a multiple of A*2^fine, so a band produces WHOLE atoms at
    // LOD0..fine on its own, single-threaded, at disjoint absolute Z. Many bands
    // -> parallel even on small volumes. `fine` is the deepest per-band level; the
    // tiny coarse tail LOD(fine+1..) is built after, in parallel, from LOD(fine).
    int fine = nlod-1 < 2 ? nlod-1 : 2;             // bands produce LOD0..2 (99.8% of atoms)
    u32 BAND = A << fine;                            // 128 voxels (multiple => whole atoms LOD0..fine)
    // XY tile footprint. The canonical parallel unit is 128 x 1024 x 1024 (z,y,x):
    // BAND=128 in Z, SB=1024 in X/Y. A band only produces LOD0..fine, so a tile need
    // only be a multiple of (A<<fine)=128 in XY for those atoms not to straddle — NOT
    // the full supercube. 1024 is a clean multiple of 128, so every coarse atom's XY
    // parents stay co-tiled (the within-cell 2x2x2 downscale never reaches across a
    // tile edge) and the cascade is identical to the full-tile case. Many small units
    // -> good load balance, ~128MB/worker, no TLB-shootdown storm. The coarse tail
    // (LOD fine+1..) is built later from the archive, independent of tile size.
    // --tile overrides (clamped up to a multiple of BAND, capped at the supercube).
    u32 SBmax = A << (nlod-1);                       // full supercube (upper bound)
    u32 SB = tile_req>0 ? (u32)tile_req : 1024;      // locked default: 1024
    if(SB > SBmax) SB = SBmax;
    SB = ((SB + BAND-1)/BAND)*BAND;                  // align up to whole LOD0..fine atoms
    // grid over the PADDED dims (px,py,pz) so every tile/band is full-size.
    u32 ntx=(px+SB-1)/SB, nty=(py+SB-1)/SB;
    u32 nbz=(pz+BAND-1)/BAND;
    long nbands=(long)ntx*nty*nbz;
    double band_mem=0; for(int l=0;l<=fine;++l){ double e=(double)SB/(1<<l); band_mem += e*e*A*2.0; }
    printf("band BAND=%u x SB=%u x SB=%u -> %ux%u tiles x %u Z-bands = %ld units (per-band LOD0..%d)\n",
           BAND,SB,SB,ntx,nty,nbz,nbands,fine);
    printf("  per-band RAM ~%.0f MB; coarse tail LOD%d..%d built after\n", band_mem/1e6, fine+1, nlod-1);

    // A-priori occupancy: derived from a coarse pyramid level (one tiny download),
    // not per-chunk probing. The zarr is box-downscaled so coarse-zero => 0/-zero.
    chunkmap cm; double tcm=now_s();
    if(cm_build(&cm,zarr,&z0)!=0){ fprintf(stderr,"chunkmap alloc failed\n"); return 1; }
    long cm_present=0,cm_total=(long)cm.ncz*cm.ncy*cm.ncx;
    for(long i=0;i<cm_total;++i) cm_present+=cm.present[i];
    printf("occupancy scan: %ld/%ld source chunks present (%.0f%%) in %.2fs\n",
           cm_present,cm_total,100.0*cm_present/cm_total, now_s()-tcm);

    sched sc={ w,&z0,nlod, px,py,pz, method,alpha,SB,BAND,fine, &cm, ntx,nty,nbands,nbz, 0, 0,0,0 };
    int nt2=nthreads>256?256:nthreads;

    // PREPASS: from the a-priori chunk data, mark every DEFINITE-ZERO region
    // (all covering source chunks absent => no data) as ZERO_REGION. The region
    // grid is tiny, so a plain loop over every region at every level. The main
    // pass then skips any band whose region is already zero.
    { double tp=now_s(); long marked=0;
      u32 lx=px,ly=py,lz=pz;
      for(int L=0;L<nlod;++L){
        u32 nrz=(((lz+A-1)/A)+R_ATOMS-1)/R_ATOMS, nry=(((ly+A-1)/A)+R_ATOMS-1)/R_ATOMS, nrx=(((lx+A-1)/A)+R_ATOMS-1)/R_ATOMS;
        for(u32 rz=0;rz<nrz;++rz)for(u32 ry=0;ry<nry;++ry)for(u32 rx=0;rx<nrx;++rx)
          if(region_definite_zero(&cm,&z0,L,rz,ry,rx)){ vc_mark_zero_region(w,L,rz,ry,rx); marked++; }
        lx=(lx+1)/2; ly=(ly+1)/2; lz=(lz+1)/2;
      }
      printf("prepass: marked %ld definite-zero regions in %.2fs\n", marked, now_s()-tp);
    }

    // total atoms across all LODs (present+zero) for progress %/ETA.
    long total_atoms=0; { u32 nx=z0.sx,ny=z0.sy,nz=z0.sz;
        for(int l=0;l<nlod;++l){ total_atoms += (long)((nx+A-1)/A)*((ny+A-1)/A)*((nz+A-1)/A);
            nx=(nx+1)/2; ny=(ny+1)/2; nz=(nz+1)/2; } }
    monitor mon={ &sc, total_atoms, progress_s, now_s(), 0 };
    pthread_t montid; int have_mon = progress_s>0;
    if(have_mon) pthread_create(&montid,NULL,monitor_thread,&mon);

    pthread_t th[256];
    for(int i=0;i<nt2;++i) pthread_create(&th[i],NULL,tile_worker,&sc);
    for(int i=0;i<nt2;++i) pthread_join(th[i],NULL);
    printf("  bands skipped (a-priori empty): %ld / %ld\n", atomic_load(&sc.skipped), nbands);

    // Coarse tail: build LOD(fine+1..nlod-1) from LOD(fine), parallel over the
    // dst atom-Z-planes (decode 2x2x2 parents from the archive, downscale). These
    // levels are tiny (LOD3 of the whole volume is ~0.2% of atoms).
    for(int L=fine+1; L<nlod; ++L){
        u32 dnx=d0.nx,dny=d0.ny,dnz=d0.nz; for(int i=0;i<L;++i){dnx=(dnx+1)/2;dny=(dny+1)/2;dnz=(dnz+1)/2;}
        u32 dacx=(dnx+A-1)/A, dacy=(dny+A-1)/A, dacz=(dnz+A-1)/A;
        tail_ctx tc={ w, L, dacx,dacy,dacz, method, alpha, 0, 0, 0 };
        pthread_t tt[256];
        for(int i=0;i<nt2;++i) pthread_create(&tt[i],NULL,tail_worker,&tc);
        for(int i=0;i<nt2;++i) pthread_join(tt[i],NULL);
        long pp=atomic_load(&tc.present), zz=atomic_load(&tc.zero);
        atomic_fetch_add(&sc.present,pp); atomic_fetch_add(&sc.zero,zz);
    }

    if(have_mon){ atomic_store(&mon.done,1); pthread_join(montid,NULL); }

    vc_writer_close(w);
    struct stat st; stat(out,&st);
    long Pp=atomic_load(&sc.present), Zz=atomic_load(&sc.zero);
    printf("done: %s  %.1f MB\n",out,st.st_size/1e6);
    printf("  present=%ld zero=%ld (%.0f%% mask)\n",Pp,Zz,100.0*Zz/(double)(Pp+Zz));
    printf("  ratio over PRESENT = %.1fx\n",(double)Pp*A*A*A/st.st_size);
    printf("  ratio vs LOD0 raw  = %.1fx\n",(double)z0.sz*z0.sy*z0.sx/st.st_size);
    // I/O accounting (--log-io): verify each source chunk is read exactly once.
    if(log_io){
        long opens=atomic_load(&g_io_open), miss=atomic_load(&g_io_miss), iob=atomic_load(&g_io_bytes);
        long present_chunks=opens-miss;
        u32 ncz=(z0.sz+z0.cz-1)/z0.cz, ncy=(z0.sy+z0.cy-1)/z0.cy, ncx=(z0.sx+z0.cx-1)/z0.cx;
        long grid_chunks=(long)ncz*ncy*ncx;
        printf("  I/O: %ld chunk opens (%ld present read, %ld absent), %.1f GB read; "
               "grid has %ld chunks -> reads/chunk = %.2f%s\n",
               opens, present_chunks, miss, iob/1e9, grid_chunks,
               grid_chunks? (double)opens/grid_chunks : 0.0,
               (grid_chunks && opens>grid_chunks) ? "  <-- RE-READING!" : "  (each chunk once)");
    }
    cm_free(&cm);
    return 0;
}
