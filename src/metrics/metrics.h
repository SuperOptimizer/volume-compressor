// Quality metrics over two u8 volumes (PLAN §4). Pure C23, cheap. PSNR, MAE,
// L-inf, error percentiles (p95/p99/p99.9), and MS-SSIM (per-axis-slice mean for
// Phase 0). GMSD / HaarPSI are TODO stubs for a later agent.
#ifndef VC_METRICS_H
#define VC_METRICS_H

#include "../../include/vc/types.h"

typedef struct {
    f64 psnr;        // dB
    f64 mae;         // mean absolute error
    f64 linf;        // max absolute error
    f64 p95, p99, p999;  // absolute-error percentiles
    f64 ms_ssim;     // multi-scale SSIM (per-axis-slice mean), 0..1
} vc_metrics;

// Compute all metrics for `rec` vs reference `ref`, both shape (dz,dy,dx).
void vc_compute_metrics(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx,
                        vc_metrics *m);

// --- Edge-sensitive perceptual metrics (PLAN §4) ---------------------------
// GMSD and HaarPSI catch the blocking/seam artifacts PSNR hides. Implemented in
// metrics.c (2.5D: per-axis-slice mean over all three axes).
double vc_gmsd(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx);
double vc_haarpsi(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx);

// Edge-MAE: mean absolute error of the Prewitt gradient MAGNITUDE between ref
// and rec (2.5D, all three axes). Directly measures how much the codec disrupts
// edge/ink/fiber structure — the gradient is where ink lives. Larger = worse.
double vc_edge_mae(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx);

// Seam-step: mean absolute first-difference of the ERROR (rec-ref) across block
// faces on a `grid`-voxel lattice (e.g. 16 for DCT-16 atoms), 2.5D over all
// axes. A blocking/banding face produces a step in the error at the grid wall;
// this isolates that step. 0 = no seam discontinuity. Larger = worse blocking.
double vc_seam_step(const u8 *ref, const u8 *rec, u32 dz, u32 dy, u32 dx, u32 grid);

#endif // VC_METRICS_H
