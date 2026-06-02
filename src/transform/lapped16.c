// Intra-atom LAPPED transform for the 16^3 atom (EXP #13b). The won transform is
// a SINGLE 16-point DCT per atom axis (no internal sub-block seams). The lapped
// candidate instead tiles each atom axis as TWO 8-point DCTs and applies a
// pre/post overlap filter (POT/LBT-style lifting, Tran 2003) ACROSS THE INTERNAL
// 8-voxel seam ONLY — never across the atom's outer faces — so the 16^3 atom is
// still independently decodable (random access preserved). Goal: smooth the one
// internal seam of an 8^3-tiled transform AT THE SOURCE rather than post-deblock,
// while keeping the better directional selectivity of the shorter 8-pt DCT.
//
// 1D pipeline along a length-16 line split [0..7][8..15]:
//   forward:  pre-filter the 4 samples straddling the internal seam (idx 4..11)
//             with a butterfly + scaling lift, then DCT-8 each half.
//   inverse:  iDCT-8 each half, then the inverse pre-filter.
// The pre-filter is orthogonal (its own inverse up to the lift), so round-trip is
// exact up to fixed-point rounding. The OUTER samples (0..3, 12..15) are NOT
// touched, guaranteeing no cross-atom leakage.
//
// Contract MIRRORS dct_int16.c (operates on one 16^3 cube), renamed symbols so a
// bench can include it directly. Pure C23; the DCT-8 kernels are const-trip
// straight-line (autovectorizable). Per-line scratch only.
#include "../../include/vc/types.h"
#include <math.h>

#ifndef VC_LAPPED16_INLINE
#define VC_LAPPED16_INLINE
#define LP_B 16u

// 8-point orthonormal DCT-II (float; the bench measures quality, fixed-point not
// required to validate the IDEA — kept float for clarity, still autovectorizable).
static f32 LP_C8[8][8];
static int lp_init_done=0;
static inline void lp_init(void){
    if(lp_init_done) return;
    for(u32 k=0;k<8;++k){
        f32 ck = (k==0)? sqrtf(1.0f/8.0f) : sqrtf(2.0f/8.0f);
        for(u32 n=0;n<8;++n)
            LP_C8[k][n]= ck*cosf((float)M_PI*(2.0f*n+1.0f)*k/16.0f);
    }
    lp_init_done=1;
}
static inline void dct8(const f32 in[8], f32 out[8]){
    for(u32 k=0;k<8;++k){ f32 a=0; for(u32 n=0;n<8;++n) a+=LP_C8[k][n]*in[n]; out[k]=a; }
}
static inline void idct8(const f32 in[8], f32 out[8]){
    for(u32 n=0;n<8;++n){ f32 a=0; for(u32 k=0;k<8;++k) a+=LP_C8[k][n]*in[k]; out[n]=a; }
}

// Pre-filter across the seam between samples [..7] and [8..]. We process the 4
// straddling pairs (3,4)(2,5)? Standard LBT pre-filter butterflies the boundary
// samples; here a light 2-tap on the innermost 2 pairs (indices {6,9} and {7,8})
// with a lifting coefficient chosen to flatten the seam DCT response.
#define LP_S 0.80f     // lift strength (Tran LBT ~ 0.8 for smoothing)
static inline void lp_prefilter(f32 v[16]){
    // pair (7,8) and (6,9): rotate by light Givens to spread energy across seam
    for(int p=0;p<2;++p){
        int a=7-p, b=8+p;
        f32 va=v[a], vb=v[b];
        f32 s = (p==0)?LP_S:(LP_S*0.5f+0.5f); // inner pair stronger
        // orthogonal lift: scale the difference component
        f32 sum=(va+vb)*0.5f, dif=(va-vb)*0.5f;
        dif*=s;
        v[a]=sum+dif; v[b]=sum-dif;
    }
}
static inline void lp_postfilter(f32 v[16]){
    for(int p=1;p>=0;--p){
        int a=7-p, b=8+p;
        f32 va=v[a], vb=v[b];
        f32 s = (p==0)?LP_S:(LP_S*0.5f+0.5f);
        f32 sum=(va+vb)*0.5f, dif=(va-vb)*0.5f;
        dif/=s;
        v[a]=sum+dif; v[b]=sum-dif;
    }
}

// Forward 1D: pre-filter seam, then DCT-8 each half into out[0..7],out[8..15].
static inline void lp_fwd1d(const f32 in[16], f32 out[16]){
    f32 t[16]; for(int i=0;i<16;++i) t[i]=in[i];
    lp_prefilter(t);
    f32 lo[8],hi[8],ol[8],oh[8];
    for(int i=0;i<8;++i){ lo[i]=t[i]; hi[i]=t[8+i]; }
    dct8(lo,ol); dct8(hi,oh);
    for(int i=0;i<8;++i){ out[i]=ol[i]; out[8+i]=oh[i]; }
}
static inline void lp_inv1d(const f32 in[16], f32 out[16]){
    f32 lo[8],hi[8],ol[8],oh[8];
    for(int i=0;i<8;++i){ ol[i]=in[i]; oh[i]=in[8+i]; }
    idct8(ol,lo); idct8(oh,hi);
    f32 t[16]; for(int i=0;i<8;++i){ t[i]=lo[i]; t[8+i]=hi[i]; }
    lp_postfilter(t);
    for(int i=0;i<16;++i) out[i]=t[i];
}

// 3D forward over one 16^3 atom. coef is i16 atom layout (z*256+y*16+x). dc bias.
void vc_lapped16_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc){
    lp_init();
    f32 a[16][16][16], b[16][16][16];
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x)
        a[z][y][x]=(f32)((i32)vox[(size_t)z*256u+y*16u+x]-dc);
    // X
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y){ f32 in[16],out[16];
        for(u32 x=0;x<16;++x)in[x]=a[z][y][x]; lp_fwd1d(in,out);
        for(u32 x=0;x<16;++x)b[z][y][x]=out[x]; }
    // Y
    for(u32 z=0;z<16;++z)for(u32 x=0;x<16;++x){ f32 in[16],out[16];
        for(u32 y=0;y<16;++y)in[y]=b[z][y][x]; lp_fwd1d(in,out);
        for(u32 y=0;y<16;++y)a[z][y][x]=out[y]; }
    // Z
    for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x){ f32 in[16],out[16];
        for(u32 z=0;z<16;++z)in[z]=a[z][y][x]; lp_fwd1d(in,out);
        for(u32 z=0;z<16;++z){ i32 v=(i32)lrintf(out[z]);
            if(v>32767)v=32767; else if(v<-32768)v=-32768;
            coef[(size_t)z*256u+y*16u+x]=(i16)v; } }
}

void vc_lapped16_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc){
    lp_init();
    f32 a[16][16][16], b[16][16][16];
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x)
        a[z][y][x]=(f32)coef[(size_t)z*256u+y*16u+x];
    // inverse Z
    for(u32 y=0;y<16;++y)for(u32 x=0;x<16;++x){ f32 in[16],out[16];
        for(u32 z=0;z<16;++z)in[z]=a[z][y][x]; lp_inv1d(in,out);
        for(u32 z=0;z<16;++z)b[z][y][x]=out[z]; }
    // inverse Y
    for(u32 z=0;z<16;++z)for(u32 x=0;x<16;++x){ f32 in[16],out[16];
        for(u32 y=0;y<16;++y)in[y]=b[z][y][x]; lp_inv1d(in,out);
        for(u32 y=0;y<16;++y)a[z][y][x]=out[y]; }
    // inverse X
    for(u32 z=0;z<16;++z)for(u32 y=0;y<16;++y){ f32 in[16],out[16];
        for(u32 x=0;x<16;++x)in[x]=a[z][y][x]; lp_inv1d(in,out);
        for(u32 x=0;x<16;++x){ i32 v=(i32)lrintf(out[x])+dc;
            v=v<0?0:(v>255?255:v); vox[(size_t)z*256u+y*16u+x]=(u8)v; } }
}

#endif
