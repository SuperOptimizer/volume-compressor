// 2x volumetric downscale kernels for building the LOD pyramid.
//
// ALL methods here are STRICTLY WITHIN-CELL: each output voxel is a function of
// exactly its own 2x2x2 = 8 parent voxels and nothing else. This is a hard
// design constraint — it means a downscale never reaches across atom / chunk /
// tile boundaries, so the cascade needs NO halo, NO neighbor exchange, and tiles
// (and atoms) are fully independent. Kernels that need neighbors (gaussian,
// lanczos, ...) are deliberately NOT here; they were removed for this reason.
#ifndef VC_DOWNSCALE_H
#define VC_DOWNSCALE_H
#include <stdint.h>
#include <string.h>

typedef enum {
    DS_BOX = 0,   // 2x2x2 mean. Anti-aliases but washes thin sheets toward gray.
    DS_CBOX,      // contrast-stretched box: the 8-mean pushed toward the cell's
                  // max-deviation voxel (bright sheet OR dark gap), so thin
                  // structure stays visible at coarse zoom. DEFAULT.
} ds_method;

static inline ds_method ds_parse(const char *s) {
    if (!s) return DS_CBOX;
    if (!strcmp(s,"box")) return DS_BOX;
    return DS_CBOX;   // "cbox"/"contrast"/anything else
}
static inline const char *ds_name(ds_method m) {
    return m==DS_BOX ? "box" : "cbox";
}

static inline uint8_t ds_u8(float v){ return (uint8_t)(v<0?0:v>255?255:(v+0.5f)); }

// Downscale `in` (nx,ny,nz) by 2x into `out` (ox=ceil(nx/2), ...). out sized
// ox*oy*oz. Within-cell only: reads exactly the 2x2x2 block at (2x,2y,2z),
// edge-replicating the odd-boundary voxel. alpha (0..1) is the DS_CBOX contrast
// push strength (0 == plain box, 1 == pick the extreme voxel).
// Reduce one 2x2x2 cell (already gathered into c8, with acc = OR of all 8) to its
// output byte. Inlined; the all-zero short-circuit is handled by the caller.
static inline uint8_t ds_cell(const unsigned char c8[8], unsigned acc,
                              ds_method method, float alpha){
    if (acc == 0) return 0;
    float mean = (c8[0]+c8[1]+c8[2]+c8[3]+c8[4]+c8[5]+c8[6]+c8[7]) * 0.125f;
    if (method == DS_BOX) return ds_u8(mean);
    float best=mean, bestdev=-1.0f;            // DS_CBOX
    for(int i=0;i<8;++i){ float d=(float)c8[i]-mean; float ad=d<0?-d:d; if(ad>bestdev){bestdev=ad;best=c8[i];} }
    return ds_u8(mean + alpha*(best - mean));
}

static inline void ds_downscale2x(const uint8_t *in, int nx, int ny, int nz,
                                  uint8_t *out, int *oxp, int *oyp, int *ozp,
                                  ds_method method, float alpha) {
    int ox=(nx+1)/2, oy=(ny+1)/2, oz=(nz+1)/2;
    if(ox<1)ox=1;
    if(oy<1)oy=1;
    if(oz<1)oz=1;
    *oxp=ox; *oyp=oy; *ozp=oz;

    // FAST PATH: the cascade always calls with nz==2 and (because the output volume
    // is padded to a multiple of 1024) EVEN nx,ny — so no odd-boundary replication
    // ever happens. Drop all 6 boundary branches; gather the 2x2 XY footprint from
    // two contiguous source rows per output row. The acc==0 test stays (most cells
    // are zero mask) but the index math is now branch-free and autovectorizes.
    if (nz==2 && (nx&1)==0 && (ny&1)==0) {
        const uint8_t *z0=in, *z1=in+(size_t)ny*nx;
        for (int y=0; y<oy; ++y) {
            const uint8_t *r0a=z0+(size_t)(2*y)*nx,   *r0b=r0a+nx;   // z=0 rows y,y+1
            const uint8_t *r1a=z1+(size_t)(2*y)*nx,   *r1b=r1a+nx;   // z=1 rows y,y+1
            uint8_t *o=out+(size_t)y*ox;
            for (int x=0; x<ox; ++x) {
                int ix=2*x;
                unsigned char c8[8];
                c8[0]=r0a[ix]; c8[1]=r0a[ix+1]; c8[2]=r0b[ix]; c8[3]=r0b[ix+1];
                c8[4]=r1a[ix]; c8[5]=r1a[ix+1]; c8[6]=r1b[ix]; c8[7]=r1b[ix+1];
                unsigned acc=c8[0]|c8[1]|c8[2]|c8[3]|c8[4]|c8[5]|c8[6]|c8[7];
                o[x]=ds_cell(c8,acc,method,alpha);
            }
        }
        return;
    }

    // GENERIC PATH: any dims (odd boundaries edge-replicated). Used by the coarse
    // tail and any non-cascade caller.
    for (int z=0; z<oz; ++z)
    for (int y=0; y<oy; ++y)
    for (int x=0; x<ox; ++x) {
        int ix=x*2, iy=y*2, iz=z*2;
        unsigned char c8[8]; int n=0; unsigned acc=0;
        for(int dz=0;dz<2;++dz){ int z2=iz+dz; if(z2>=nz)z2=nz-1;
          for(int dy=0;dy<2;++dy){ int y2=iy+dy; if(y2>=ny)y2=ny-1;
            for(int dx=0;dx<2;++dx){ int x2=ix+dx; if(x2>=nx)x2=nx-1;
              unsigned char v=in[((size_t)z2*ny+y2)*nx+x2]; c8[n++]=v; acc|=v; }}}
        out[((size_t)z*oy + y)*ox + x] = ds_cell(c8,acc,method,alpha);
    }
}

#endif
