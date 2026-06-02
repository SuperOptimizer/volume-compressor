// Rate-control allocator (PLAN §2 "Rate control" + "Quality granularity",
// §3 ratectrl/). The headline "variable quality" feature: the user sets ONE
// global target compression ratio and the allocator picks a per-UNIT dead-zone
// step so easy/air units compress hard and detailed/ink units stay fine,
// minimizing total distortion at the target rate (equal-slope Lagrangian, a
// single global lambda).
//
// This sits ABOVE the per-chunk block pipeline (transform/quant/entropy) and is
// orthogonal to it: it only chooses the scalar dead-zone STEP per unit. It does
// NOT change the per-chunk byte layout the codec already uses (each chunk header
// already carries its own f32 step), so an allocator-produced per-chunk step set
// decodes with the unmodified codec. The per-16^3-block q-field variant stores a
// compact step field in the chunk header (see VC_RC_GRANULARITY_BLOCK).
//
// Two distortion estimators are provided and compared (PLAN §2 distortion row,
// open question "how wrong is Parseval"):
//   * PARSEVAL transform-domain estimate: sum of per-coefficient quantization
//     errors in coefficient space (cheap, no inverse transform / no decode).
//   * TRUE decode-MSE: actually dequantize + inverse-transform the unit and
//     measure real spatial MSE. Exact but ~2x the cost (one extra inverse pass).
//
// Two rate estimators:
//   * PROBE: run the actual entropy coder, take its byte count (exact, slow).
//   * LAPLACIAN model: predict R (and D) from per-unit coefficient statistics in
//     ~1 pass (cheap, approximate) — the "analytical" variant.
//
// Granularity lever: per-chunk scalar step, or a per-16^3-block step field.
//
// Pure C23, single-threaded reentrant, heap scratch. The allocator is a HARNESS/
// encode-side tool; it is not on the decode hot path.
#ifndef VC_RATECTRL_H
#define VC_RATECTRL_H

#include "../../include/vc/types.h"
#include <stddef.h>

// --- existing fixed-q contract (kept; fixed.c) -----------------------------
f32 vc_rc_fixed_step(f32 q);

// --- Lagrangian allocator --------------------------------------------------

// Granularity: the quality atom the allocator assigns one step to.
typedef enum {
    VC_RC_PER_CHUNK = 0,   // one step per VC_CHUNK_SIDE^3 chunk
    VC_RC_PER_BLOCK = 1,   // one step per 16^3 sub-block (the q-field)
} vc_rc_gran;

// Distortion estimator used while building each unit's R-D curve.
typedef enum {
    VC_RC_DIST_PARSEVAL = 0,  // cheap transform-domain coefficient-error sum
    VC_RC_DIST_TRUEMSE  = 1,  // exact decode-MSE (extra inverse pass)
} vc_rc_dist;

// Rate/curve model.
typedef enum {
    VC_RC_MODEL_PROBE     = 0,  // multi-q probe through the real entropy coder
    VC_RC_MODEL_LAPLACIAN = 1,  // analytical Laplacian R-D from coef variance
    // --- FAST-ENCODE tier (no per-step grid, no hull, no per-point probing) ---
    VC_RC_MODEL_CLOSEDFORM = 2, // closed-form q-from-lambda per atom: one fwd DCT
                                // per block, derive step = base*f(variance) from a
                                // single global lambda via step ~= 2.94*sqrt(lambda),
                                // bisect lambda on the closed-form rate sum only.
    VC_RC_MODEL_FEEDBACK   = 3, // single-pass raster feedback controller: one fwd
                                // DCT per block, adjust step on the fly toward a
                                // running bytes-per-block budget (proportional).
} vc_rc_model;

typedef struct {
    vc_rc_gran  gran;
    vc_rc_dist  dist;
    vc_rc_model model;
    f64 target_ratio;       // global ratio target (e.g. 10.0 for 10x)
    // Per-unit step is clamped to [base/step_window, base*step_window] around the
    // Lagrangian base step (base = 2.94*sqrt(lambda)). This is what HEVC/JPEG-XL
    // do: QP varies in a bounded window around a frame base-QP, NOT unbounded.
    // Plain summed-MSE Lagrangian otherwise degenerates to "bang-bang" (a few
    // units near-lossless, the rest crushed) which underperforms uniform-q in
    // PSNR (PLAN §2 "MSE-optimal allocation provably starves HF"). A window of
    // ~4x reproduces the realistic "easy chunks 20x / hard chunks 5x" spread.
    // 0 or <1 => unbounded (the degenerate baseline, for comparison).
    f64 step_window;
    int verbose;
} vc_rc_config;

// One unit's chosen step plus diagnostics. For per-block granularity the codec
// stores `step` per 16^3 block; for per-chunk one step per chunk.
typedef struct {
    f32  step;              // chosen dead-zone step for this unit
    f64  bytes;             // estimated/achieved bytes at that step
    f64  dist;              // estimated distortion at that step (MSE units)
} vc_rc_unit;

// Result of an allocation pass over a whole volume.
typedef struct {
    f64 lambda;             // the global lambda the bisection settled on
    f64 achieved_ratio;     // predicted overall ratio at the allocation
    f64 total_bytes;        // predicted payload bytes
    f64 raw_bytes;          // raw volume bytes
    // Parseval-vs-true divergence accounting (filled when both are computed).
    f64 parseval_mse;       // mean Parseval-estimated MSE over units (last build)
    f64 true_mse;           // mean true decode-MSE over units (last build)
    u32 n_units;
} vc_rc_result;

// Allocate per-unit steps for a volume of shape (dz,dy,dx) to hit cfg.target_ratio.
// Writes one vc_rc_unit per unit into `units` (caller sizes it: see
// vc_rc_count_units). Returns 0 on success. The allocator is self-contained:
// it gathers chunks, runs transform+quant+entropy internally to build R-D data.
int vc_rc_allocate(const u8 *vol, u32 dz, u32 dy, u32 dx,
                   const vc_rc_config *cfg, vc_rc_unit *units, vc_rc_result *res);

// Number of allocation units for a volume at the given granularity.
u32 vc_rc_count_units(u32 dz, u32 dy, u32 dx, vc_rc_gran gran);

// Honest whole-volume MSE of a PER-BLOCK q-field: for each 16^3 block, run the
// codec's exact transform + dead-zone quant/dequant + inverse at that block's
// `units[ui].step` and accumulate spatial error. This isolates the ALLOCATION
// quality (which step each block got) from per-block-archive entropy overhead, so
// it is the fair quality metric to compare rate-control models at a matched
// predicted ratio. Block walk matches vc_rc_count_units(PER_BLOCK). Returns MSE.
f64 vc_rc_qfield_truemse(const u8 *vol, u32 dz, u32 dy, u32 dx,
                         const vc_rc_unit *units);

#endif // VC_RATECTRL_H
