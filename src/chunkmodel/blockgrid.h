// Block-grid (chunk-model) engine — PLAN §2 chunk/dependency-model axis, full
// spec in docs/EXP_chunk_model.md.
//
// FOUNDATIONAL MODEL: the 16^3 block is THE ONE TRUE ATOM (decode = transform =
// cache = access = prediction-node unit; FIXED). The codec operates on a GLOBAL
// 3D LATTICE of 16^3 atoms. Prediction & entropy context flow across the lattice
// by a STENCIL, independent of where chunk walls fall. A CHUNK = "N atoms bundled
// for I/O" (shared header + seek directory), NOT a coding wall. Each 16^3 atom
// stays individually decodable (random access): load chunk header + that atom's
// seekable byte range (+ its prediction ancestors, if a stencil is active).
//
// This is a self-contained bake-off engine (like the transform/entropy benches):
// the three orthogonal axes + chunk size are RUNTIME parameters of vc_bg_cfg so
// ONE binary sweeps the whole matrix. It does NOT touch codec.c. The winning
// config would later be frozen into a compile-time VC_CHUNKMODEL block.
//
// Pipeline per atom: DCT-16^3 (the won transform) -> per-atom DC handling (shared/
// predicted reference) -> 16^3 frequency-scan + HF-protecting dead-zone quant ->
// stencil prediction of DC/low-freq from already-coded neighbors -> entropy
// (static rANS with a per-chunk SHARED table = M1, or RLGR table-free).
//
// Pure C23, libc/libm only, single-threaded reentrant, heap/arena scratch.
#ifndef VC_BLOCKGRID_H
#define VC_BLOCKGRID_H

#include "../../include/vc/types.h"

#define VC_ATOM 16u                       // THE atom edge (fixed)
#define VC_ATOM_VOX (VC_ATOM * VC_ATOM * VC_ATOM)  // 4096

// --- (A) prediction stencil -------------------------------------------------
typedef enum {
    VC_STENCIL_NONE = 0,   // no cross-atom prediction (M1 baseline)
    VC_STENCIL_6    = 6,   // 6-connected: face neighbors (-x,-y,-z causal subset)
    VC_STENCIL_18   = 18,  // 18-connected: + edge neighbors
    VC_STENCIL_26   = 26,  // 26-connected: + corner neighbors
} vc_stencil;

// --- (B) atom traversal order ----------------------------------------------
typedef enum {
    VC_TRAV_RASTER = 0,    // raster ZYX
    VC_TRAV_MORTON = 1,    // Morton / Z-order
    VC_TRAV_HILBERT = 2,   // Hilbert curve
} vc_traversal;

// --- (C) edge data-availability --------------------------------------------
typedef enum {
    VC_EDGE_SELF   = 0,    // self-contained: boundary atoms fall back to intra
    VC_EDGE_HALO   = 1,    // halo: predict across chunk edge via stored DC shell
    VC_EDGE_FETCH  = 2,    // fetch neighbor chunk's atoms (cross-chunk dependency)
} vc_edge;

// --- entropy mode (the sharing mechanism) ----------------------------------
typedef enum {
    VC_ENT_RANS_INDEP = 0, // M0 reference cliff: each atom its own rANS table
    VC_ENT_RANS_SHARED= 1, // M1: one shared rANS table per chunk (table-amortized)
    VC_ENT_RLGR       = 2,  // table-free RLGR (shared param seed)
} vc_entropy_mode;

// --- grouping mode: the SHARING + FETCH unit (curve-group experiment) -------
// Defines WHAT set of 16^3 atoms shares one entropy table + one seek directory.
//   VC_GROUP_BOX   : the current M1 design — an axis-aligned chunk_atoms^3 box.
//   VC_GROUP_CURVE : a "curve group" = N CONSECUTIVE atoms along a global
//                    space-filling curve (Morton/Hilbert) over the whole lattice.
//                    Eliminates the chunk-box concept: the data is ONE continuous
//                    lattice of 16^3 atoms; groups are contiguous runs along the
//                    curve. The `traversal` field selects the curve; `group_n` is
//                    the run length (atoms per group). Decode one atom: compute
//                    its curve index -> find its group (group-index table) -> load
//                    that group's header (shared table + per-atom directory) ->
//                    decode just that atom's bytes. touched stays 1.
typedef enum {
    VC_GROUP_BOX   = 0,
    VC_GROUP_CURVE = 1,
} vc_group_mode;

// --- THREE-LEVEL REDUNDANCY HIERARCHY variants (curve-group experiment) ------
// All consulted ONLY in VC_GROUP_CURVE mode; ignored (0) otherwise so a zero-
// initialised cfg is exactly the old fixed-N curve-group / box behaviour.
//
// (I) INTRA-group boundary rule — how a group ENDS (how big it is):
//   FIXED  : every group is exactly group_n atoms (the original curve-group).
//   DRIFT  : a group ENDS when its running symbol histogram diverges from the
//            group's accumulated histogram past `drift_thresh` (variable-size,
//            internally-homogeneous groups). group_n becomes the MAX cap.
typedef enum {
    VC_BOUND_FIXED = 0,
    VC_BOUND_DRIFT = 1,
} vc_boundary_rule;

// (E) INTER-group table coding — how each group's rANS table is STORED:
//   FULL  : store the full NSYM*2-byte freq table per group (original).
//   DELTA : store group N's table as a zig-zag varint DELTA from group N-1's
//           table (E1). Adjacent curve-groups have similar stats -> tiny delta
//           -> cheap header -> makes small coherent groups affordable.
//   BASE  : a single coarse SUPER-GROUP base table (built from the whole
//           lattice histogram) that every group corrects against with a delta
//           (E2, two-level base+group-delta). The base is stored once.
// Table coding only affects VC_ENT_RANS_SHARED (RLGR is table-free).
typedef enum {
    VC_TABLE_FULL  = 0,
    VC_TABLE_DELTA = 1,
    VC_TABLE_BASE  = 2,
} vc_table_coding;

// --- EG2024 (Fast Compressed Segmentation Volumes) experiment knobs ----------
// (1) ★ SPLIT shared entropy tables by SYMBOL ROLE (DC vs AC bands). Currently
// ALL 4096 freq-scanned coefficients of an atom pool into ONE shared rANS table.
// DC, low-AC and high-AC have very different level distributions; pooling them
// wastes bits. Code each band of coefficients against its OWN shared table. The
// band of a coefficient is decided by its frequency-scan position's u+v+w sum
// (g_fsum): NONE = one table (old behaviour); DC_AC = {DC(fsum 0), AC(rest)};
// DC_LO_HI = {DC, low-AC (fsum 1..VC_BAND_LO), high-AC}. Only consulted in the
// VC_ENT_RANS_SHARED box path; RLGR is table-free (band split is a no-op there).
typedef enum {
    VC_BAND_NONE   = 0,    // one pooled table (baseline)
    VC_BAND_DC_AC  = 1,    // 2 tables: DC | AC
    VC_BAND_DC_LO_HI = 2,  // 3 tables: DC | low-AC | high-AC
} vc_band_split;

#define VC_BAND_LO 8u      // fsum<=VC_BAND_LO (and >0) is "low-AC"; >LO is "high-AC"
#define VC_NBAND_MAX 3u

typedef struct {
    vc_stencil      stencil;
    vc_traversal    traversal;
    vc_edge         edge;
    vc_entropy_mode entropy;
    u32             chunk_atoms;   // atoms per side of a chunk bundle (4/8/16)
    int             shared_dc;     // 1 = shared/predicted DC reference across atoms
    float           step;          // dead-zone base step (rate control knob)
    // JPEG-XL "DC frame" variant: code ALL atoms' DC together as a separate,
    // globally-predicted DC SUB-VOLUME (one DC level per atom, an Az*Ay*Ax mini-
    // volume, raster left/up/front-predicted + rANS), decoded ONCE up front. Each
    // atom then stores AC ONLY and looks up its DC from the decoded sub-volume.
    // This decouples DC prediction from the atom decode order, preserving cheap
    // 16^3 random access (touched stays 1 — no ancestor cone). When set, the
    // per-atom `stencil` applies to AC only (typically NONE). 0 = off (per-atom
    // DC threading per `stencil`, the original path).
    int             dc_subvol;
    // Curve-group experiment (see vc_group_mode). Only consulted when
    // group_mode == VC_GROUP_CURVE; ignored for VC_GROUP_BOX (back-compat: a
    // zero-initialized cfg is exactly the old box behavior).
    vc_group_mode   group_mode;
    u32             group_n;       // atoms per curve group (e.g. 64/256/512)
    // --- three-level hierarchy knobs (curve mode only; 0 = original behaviour) -
    vc_boundary_rule boundary;     // I1: FIXED vs DRIFT-adaptive group sizing
    float           drift_thresh;  // I1: histogram-divergence cut (e.g. 0.15);
                                   //     only used when boundary==DRIFT. group_n
                                   //     is then the MAX cap (and a min of 8).
    vc_table_coding table_coding;  // E1/E2: FULL / DELTA(prev) / BASE(super-group)
    int             dc_pred_curve; // B1: DC-only prediction from the CURVE
                                   //     PREDECESSOR (touched stays 1; DC residual
                                   //     in the directory). 0 = absolute DC.
    u32             nested_sub;    // I2: nested sub-group size (atoms). >0 splits a
                                   //     group into sub-runs that each carry a small
                                   //     DC-base delta on top of the group DC mean
                                   //     (intra-group two-level DC model). 0 = off.
    // --- EG2024 knobs (box rANS-shared path) ---------------------------------
    vc_band_split   band_split;    // (1) per-band (DC/AC) shared tables. 0 = pooled.
    u32             sparse_prepass;// (3) build the shared table from every Nth atom
                                   //     (sample stride). 0 or 1 = scan ALL atoms.
    int             skip_meta;     // (4) per-chunk min/max + uniform-flag skip index
                                   //     (ORC/Parquet 3-tier). Charges the tiny index
                                   //     bytes and lets near-constant (single-value)
                                   //     chunks store ONE byte instead of atom blobs.
    // --- EXP#20: per-atom ADAPTIVE transform/mode selection (box path only) ----
    // A 2-bit per-atom mode flag (stored in the seek directory) lets each 16^3 atom
    // pick the cheapest of: DCT (default), SKIP (DC-only: store mean, drop all AC),
    // RAW (incompressible escape: bypass-coded zig-zag levels, no shared table).
    // The encoder picks by trial: lowest rate + adaptive_lambda * distortion (SSD).
    // 0 = off (exactly the old DCT-only behaviour; flag costs nothing).
    int             adaptive;      // 1 = enable per-atom mode selection
    float           adaptive_lambda; // R-D Lagrange multiplier (bytes per SSD unit)
} vc_bg_cfg;

// Encoded archive (in-memory): a sequence of chunk records, each with a header
// (shared table + delta-coded seek directory) and atom payloads. Opaque to the
// caller; introspect via the stats struct.
typedef struct vc_bg_archive vc_bg_archive;

typedef struct {
    size_t total_bytes;      // whole archive
    size_t payload_bytes;    // atom payloads only
    size_t directory_bytes;  // seek directories (all chunks)
    size_t table_bytes;      // shared entropy tables (all chunks)
    size_t halo_bytes;       // stored halo shells (edge=halo)
    size_t dc_subvol_bytes;  // separate globally-predicted DC sub-volume stream
    u32    n_atoms;          // total atoms in the lattice
    u32    n_chunks;
    u32    n_absent_atoms;   // all-zero atoms (directory flag only)
    size_t skip_meta_bytes;  // (4) skip-index bytes (min/max+uniform flag per chunk)
    size_t skip_saved_bytes; // (4) payload+directory bytes skipped on uniform chunks
    u32    n_uniform_chunks;  // chunks flagged uniform (single value) and skipped
    // EXP#20: per-atom adaptive mode histogram (non-absent atoms only).
    u32    n_mode_dct;       // atoms coded as DCT
    u32    n_mode_skip;      // atoms coded as SKIP (DC-only)
    u32    n_mode_raw;       // atoms coded as RAW (bypass escape)
} vc_bg_stats;

// Encode a full u8 volume into a block-grid archive under cfg. Returns 0 on ok.
int vc_bg_encode(const u8 *vol, u32 dz, u32 dy, u32 dx,
                 const vc_bg_cfg *cfg, vc_bg_archive **out, vc_bg_stats *st);

// Full decode: reconstruct the whole volume (decodes every atom in coding order).
int vc_bg_decode(const vc_bg_archive *a, u8 *vol);

// Random-access decode of ONE 16^3 atom at lattice coord (az,ay,ax) into a 4096-
// byte buffer `atom_out`. Returns the number of atoms that had to be touched to
// produce it (= 1 + prediction-ancestor count) via *touched; this is the key
// "amortized 16^3 decode cost" measurement. Returns 0 on ok.
int vc_bg_decode_atom(const vc_bg_archive *a, u32 az, u32 ay, u32 ax,
                      u8 *atom_out, u32 *touched);

// Decode every atom whose lattice coord lies in a [lo,hi) box, sharing the
// dependency cache, and report the TOTAL number of atom-decodes performed (incl.
// re-decodes forced by prediction ancestry). This is the neighborhood-sweep
// access trace: amortized cost = total_decodes / atoms_in_box.
int vc_bg_decode_region(const vc_bg_archive *a,
                        u32 z0, u32 y0, u32 x0, u32 z1, u32 y1, u32 x1,
                        u64 *total_decodes, u32 *atoms_in_box);

void vc_bg_free(vc_bg_archive *a);

// Lattice dims (atoms per axis) for a volume shape (ceil-div by 16).
static inline u32 vc_bg_natoms(u32 n) { return (n + VC_ATOM - 1u) / VC_ATOM; }

#endif // VC_BLOCKGRID_H
