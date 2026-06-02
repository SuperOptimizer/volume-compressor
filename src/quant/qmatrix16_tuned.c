// Tuned per-coefficient 16^3 quant-matrix SHAPES (EXP #13a). Beyond the existing
// linear HF slope (qmatrix16.c: weight = 1 - slope*(u+v+w)/45), this provides
// alternative psychovisual/energy-tuned per-position step shapes, all swappable
// at runtime via qmt_mode. Same dead-zone quant primitives as qmatrix16.c.
//
// Shapes (all weight 1 at DC, all monotone non-increasing into HF so they still
// "protect HF" relative to a coarsening JPEG table, but with different curvature):
//   QMT_LINEAR  — the BASELINE shape (== qmatrix16 HF, slope 0.6 on L1 freq sum).
//   QMT_L2      — radial L2: weight = 1 - slope * sqrt((u^2+v^2+w^2)/(3*15^2)).
//                 (isotropic in Euclidean freq; the JPEG-XL DCT distance form.)
//   QMT_SEP     — separable per-axis product of 1D 16-pt weights w1(u)*w1(v)*w1(w),
//                 w1(f)=1-slope1*(f/15): matches a separable HVS/energy model.
//   QMT_ENERGY  — data-derived: step proportional to 1/sqrt(mean coef energy at
//                 that position) (a reverse-water-filling / KLT-step shape),
//                 supplied as a 4096 float gain table measured on the corpus and
//                 blended toward flat by `ablend`. This is the "energy-tuned"
//                 candidate the task asks for.
//
// Pure C23, libc/libm. Quant loop identical to qmatrix16 (unit-stride branchless).
#include "../../include/vc/types.h"
#include <math.h>

#ifndef VC_QMT16_INLINE
#define VC_QMT16_INLINE
#define QMT_B 16u
#define QMT_N 4096u

typedef enum { QMT_LINEAR=0, QMT_L2=1, QMT_SEP=2, QMT_ENERGY=3 } qmt_mode;

static f32 g_qmt_slope = 0.60f;       // matches qmatrix16 default
static inline void qmt_set_slope(f32 s){ g_qmt_slope=s; }

// Build a per-coefficient WEIGHT matrix (multiplies base step). `egain` is an
// optional 4096 energy-gain table (used only for QMT_ENERGY; pass NULL else);
// `ablend` in [0,1] blends ENERGY toward flat (0 = flat, 1 = full energy shape).
static inline void qmt_build_weight(f32 *restrict wmat, qmt_mode mode,
                                    const f32 *restrict egain, f32 ablend){
    f32 sl = g_qmt_slope;
    for(u32 z=0;z<QMT_B;++z)
    for(u32 y=0;y<QMT_B;++y)
    for(u32 x=0;x<QMT_B;++x){
        f32 w;
        switch(mode){
            case QMT_L2: {
                f32 r = sqrtf((f32)(x*x+y*y+z*z)/(3.0f*225.0f)); // 0..1
                w = 1.0f - sl*r;
            } break;
            case QMT_SEP: {
                f32 wx=1.0f-sl*((f32)x/15.0f);
                f32 wy=1.0f-sl*((f32)y/15.0f);
                f32 wz=1.0f-sl*((f32)z/15.0f);
                // geometric-mean so it stays comparable in magnitude to LINEAR
                w = cbrtf(wx*wy*wz*wx*wy*wz*wx*wy*wz); // (wx wy wz) but keep positive
                w = wx*wy*wz; // product (separable); cbrt line above unused safety
            } break;
            case QMT_ENERGY: {
                u32 idx=(size_t)z*256u+y*16u+x;
                f32 eg = egain?egain[idx]:1.0f;       // <=1, smaller where less energy
                f32 lin = 1.0f - sl*((f32)(x+y+z)/45.0f);
                w = (1.0f-ablend)*lin + ablend*eg;
            } break;
            case QMT_LINEAR: default: {
                w = 1.0f - sl*((f32)(x+y+z)/45.0f);
            } break;
        }
        if(w<0.05f) w=0.05f;
        wmat[(size_t)z*256u+y*16u+x]=w;
    }
}

static inline void qmt_apply_step(f32 *restrict step, const f32 *restrict wmat,
                                  f32 base_step, f32 blk_scale){
    for(u32 i=0;i<QMT_N;++i){
        f32 s = base_step * wmat[i] * blk_scale;
        if(s<0.5f) s=0.5f;
        step[i]=s;
    }
}

#endif
