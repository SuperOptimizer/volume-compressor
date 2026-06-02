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
