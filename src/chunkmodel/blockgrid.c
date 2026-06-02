// Block-grid (chunk-model) engine implementation. See blockgrid.h + the spec
// docs/EXP_chunk_model.md. Self-contained: pulls in the DCT-16^3 transform under
// a renamed symbol so it never collides with codec.c's compile-time transform.
#include "blockgrid.h"
#include "../core/bitio.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// DCT-16^3 (the won transform). Include the source with renamed entry points so
// this TU owns its own copy independent of the codec's VC_TRANSFORM selection.
// dct_int16.c keys its sub-block geometry off VC_CHUNK_SIDE; here the cube it
// transforms IS one atom (16^3), so we shadow VC_CHUNK_SIDE = 16 for that file.
// ---------------------------------------------------------------------------
#define vc_dct_int16_fwd  bg_dct16_fwd
#define vc_dct_int16_inv  bg_dct16_inv
#ifdef VC_CHUNK_SIDE
#  undef VC_CHUNK_SIDE
#endif
#define VC_CHUNK_SIDE 16
#include "../transform/dct_int16.c"
#undef VC_CHUNK_SIDE
#undef vc_dct_int16_fwd
#undef vc_dct_int16_inv
// Scrub the transform file's local geometry macros so they don't leak into the
// rest of this TU (it uses Q/B/CS as ordinary identifiers).
#undef CS
#undef SLY
#undef SLZ
#undef B
#undef Q

#define A    VC_ATOM            // 16
#define AVOX VC_ATOM_VOX        // 4096

// ===========================================================================
// 16^3 frequency scan + HF-protecting dead-zone quant (the won quant policy,
// VC_QWEIGHT=1, but built for the 16^3 atom instead of the 8^3 sub-block).
// ===========================================================================
static u16 g_scan[AVOX];     // linear offset z*256+y*16+x of i-th scanned coeff
static u16 g_fsum[AVOX];     // u+v+w of i-th scanned coeff (0..45)
static f32 g_wt[AVOX];       // per-position HF-protecting step weight
static int g_init = 0;
// EG2024 band id of each freq-scan position for each split mode. band2[i]=DC/AC,
// band3[i]=DC/lowAC/hiAC. Filled in scan_init (depends only on g_fsum, the scan).
static u8  g_band2[AVOX];
static u8  g_band3[AVOX];

static void scan_init(void) {
    u32 w = 0;
    for (u32 s = 0; s <= 45; ++s)
        for (u32 z = 0; z < A; ++z)
        for (u32 y = 0; y < A; ++y)
        for (u32 x = 0; x < A; ++x)
            if (z + y + x == s) {
                g_scan[w] = (u16)(z * 256u + y * 16u + x);
                g_fsum[w] = (u16)s;
                f32 t = (f32)s / 45.0f;          // 0..1
                g_wt[w]  = 1.0f - 0.55f * t;      // HF-protecting (DC=1 .. HF~0.45)
                ++w;
            }
    for (u32 i = 0; i < AVOX; ++i) {
        u32 fs = g_fsum[i];
        g_band2[i] = (fs == 0) ? 0u : 1u;                       // DC | AC
        g_band3[i] = (fs == 0) ? 0u : (fs <= VC_BAND_LO ? 1u : 2u); // DC|lo|hi
    }
    g_init = 1;
}
// Number of bands and the per-position band map for a split mode.
static inline u32 band_count(vc_band_split bs) {
    return bs == VC_BAND_DC_LO_HI ? 3u : (bs == VC_BAND_DC_AC ? 2u : 1u);
}
static inline const u8 *band_map(vc_band_split bs) {
    return bs == VC_BAND_DC_LO_HI ? g_band3 : g_band2; // unused when bs==NONE
}
static inline void ensure_init(void) { if (!g_init) scan_init(); }

// Quantize one atom's row-major DCT coefficients into freq-scanned signed levels.
// The DC coefficient (scan pos 0) is handled SEPARATELY by the caller (prediction
// + shared reference), so it is written into qscan[0] = 0 here and returned.
static i32 quant_atom(i16 *restrict qscan, const i16 *restrict coef, f32 base) {
    f32 inv[AVOX], half[AVOX];
    for (u32 i = 0; i < AVOX; ++i) {
        f32 s = base * g_wt[i]; if (s < 0.5f) s = 0.5f;
        inv[i] = 1.0f / s; half[i] = 0.5f * s;
    }
    // gather freq-scan then branchless quant
    i16 cb[AVOX];
    for (u32 i = 0; i < AVOX; ++i) cb[i] = coef[g_scan[i]];
    i32 dc = cb[0];
    for (u32 i = 0; i < AVOX; ++i) {
        f32 c = (f32)cb[i];
        f32 a = c < 0.f ? -c : c;
        i32 m = (a >= half[i]);
        i32 lvl = m * ((i32)(a * inv[i] - 0.5f) + 1);
        qscan[i] = (i16)(c < 0.f ? -lvl : lvl);
    }
    qscan[0] = 0;                 // DC carried out-of-band
    return dc;                    // raw (unquantized) DC coefficient
}

// Inverse: freq-scanned levels (+ a separately supplied DC level) -> row-major
// DCT coefficients for one atom. dc_coef is the reconstructed DC coefficient.
static void dequant_atom(i16 *restrict coef, const i16 *restrict qscan,
                         i16 dc_coef, f32 base) {
    f32 step[AVOX];
    for (u32 i = 0; i < AVOX; ++i) {
        f32 s = base * g_wt[i]; if (s < 0.5f) s = 0.5f; step[i] = s;
    }
    i16 cb[AVOX];
    for (u32 i = 0; i < AVOX; ++i) {
        i32 l = qscan[i], aa = l < 0 ? -l : l;
        f32 r = (aa == 0) ? 0.f : (f32)aa * step[i];
        i32 v = (i32)lrintf(l < 0 ? -r : r);
        if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
        cb[i] = (i16)v;
    }
    cb[0] = dc_coef;
    for (u32 i = 0; i < AVOX; ++i) coef[g_scan[i]] = cb[i];
}

// Quantize / dequantize the DC coefficient with a dedicated (fine) step so the
// shared/predicted DC reference is exact-ish. DC uses the base step * weight[DC]=1.
static inline i16 dc_quant(i32 dc_coef, f32 base) {
    f32 step = base; if (step < 0.5f) step = 0.5f;
    f32 a = dc_coef < 0 ? -(f32)dc_coef : (f32)dc_coef;
    i32 lvl = 0; if (a >= 0.5f * step) lvl = (i32)(a / step - 0.5f) + 1;
    return (i16)(dc_coef < 0 ? -lvl : lvl);
}
static inline i16 dc_dequant(i16 q, f32 base) {
    f32 step = base; if (step < 0.5f) step = 0.5f;
    i32 l = q, a = l < 0 ? -l : l;
    f32 r = (a == 0) ? 0.f : (f32)a * step;
    i32 v = (i32)lrintf(l < 0 ? -r : r);
    if (v > 32767) v = 32767; else if (v < -32768) v = -32768;
    return (i16)v;
}

// ===========================================================================
// Static rANS with a SHARED (externally supplied) frequency table. This is the
// M1 mechanism: build ONE table per chunk from all its atoms' histograms, store
// it once, and code each atom into its own seekable byte range against it.
// (A trimmed re-implementation of entropy/rans_static.c factored so the table is
// separable from per-atom coding.)
// ===========================================================================
#define NSYM 256u
#define ESC  255u
#define PB   12u
#define PS   (1u << PB)          // 4096
#define RL   (1u << 16)
#define RSH  48u

typedef struct {
    u16 freq[NSYM];
    u16 cum[NSYM + 1];
    u16 slot2sym[PS];
    u64 rcp[NSYM];
} rans_table;

static inline u32 zz(i16 v) { i32 x = v; return (u32)((x << 1) ^ (x >> 31)); }
static inline i16 unzz(u32 u) { i32 x = (i32)((u >> 1) ^ (~(u & 1u) + 1u)); return (i16)x; }

static void rt_finish(rans_table *t) {
    t->cum[0] = 0;
    for (u32 s = 0; s < NSYM; ++s) t->cum[s + 1] = (u16)(t->cum[s] + t->freq[s]);
    for (u32 s = 0; s < NSYM; ++s)
        for (u32 i = t->cum[s]; i < t->cum[s + 1]; ++i) t->slot2sym[i] = (u16)s;
    for (u32 s = 0; s < NSYM; ++s)
        t->rcp[s] = t->freq[s] ? ((u64)1 << RSH) / t->freq[s] + 1 : 0;
}
static void rt_build(rans_table *t, const u64 *counts) {
    u64 total = 0; for (u32 s = 0; s < NSYM; ++s) total += counts[s];
    memset(t->freq, 0, sizeof(t->freq));
    if (total == 0) { for (u32 s = 0; s < NSYM; ++s) t->freq[s] = 1; }
    else for (u32 s = 0; s < NSYM; ++s) {
        if (!counts[s]) continue;
        u32 f = (u32)(((u64)counts[s] * PS) / total);
        t->freq[s] = (u16)(f == 0 ? 1u : f);
    }
    u32 sum = 0; for (u32 s = 0; s < NSYM; ++s) sum += t->freq[s];
    if (sum != PS) {
        u32 best = 0; for (u32 s = 1; s < NSYM; ++s) if (t->freq[s] > t->freq[best]) best = s;
        t->freq[best] = (u16)((i32)t->freq[best] + ((i32)PS - (i32)sum));
    }
    rt_finish(t);
}
static inline u32 rt_divf(const rans_table *t, u32 x, u32 s) {
    return (u32)(((unsigned __int128)x * t->rcp[s]) >> RSH);
}

static void rans_hist(const i16 *q, size_t n, u64 *counts) {
    for (size_t i = 0; i < n; ++i) { u32 u = zz(q[i]); counts[u < ESC ? u : ESC]++; }
}

// Encode n levels with the given table into out (cap bytes); returns bytes or 0
// on overflow. Layout: u32 tok_len, tokens, then bypass varints bit-packed.
static size_t rans_enc(u8 *out, size_t cap, const i16 *q, size_t n, const rans_table *t) {
    if (cap < 4) return 0;
    size_t tok_off = 4;
    u16 *words = (u16 *)(out + tok_off);
    size_t wcap = (cap > tok_off) ? (cap - tok_off) / 2 : 0;
    size_t nw = 0; u32 state = RL;
    for (size_t ii = n; ii-- > 0; ) {
        u32 u = zz(q[ii]); u32 sym = u < ESC ? u : ESC;
        u32 f = t->freq[sym], c = t->cum[sym];
        u64 xmax = (u64)((RL >> PB) << 16) * f;
        while (state >= xmax) { if (nw >= wcap) return 0; words[nw++] = (u16)(state & 0xffff); state >>= 16; }
        u32 qd = rt_divf(t, state, sym);
        state = (qd << PB) + (state - qd * f) + c;
    }
    if (nw + 2 > wcap) return 0;
    words[nw++] = (u16)(state & 0xffff);
    words[nw++] = (u16)((state >> 16) & 0xffff);
    for (size_t a = 0, b = nw - 1; a < b; ++a, --b) { u16 tmp = words[a]; words[a] = words[b]; words[b] = tmp; }
    size_t tok_bytes = nw * 2;
    out[0] = (u8)tok_bytes; out[1] = (u8)(tok_bytes >> 8);
    out[2] = (u8)(tok_bytes >> 16); out[3] = (u8)(tok_bytes >> 24);
    size_t boff = tok_off + tok_bytes;
    if (boff > cap) return 0;
    vc_bitwriter bw; vc_bw_init(&bw, out + boff, cap - boff);
    for (size_t i = 0; i < n; ++i) {
        u32 u = zz(q[i]);
        if (u >= ESC) { u32 x = u; do { u32 b = x & 0x7fu; x >>= 7; vc_bw_put(&bw, b | (x ? 0x80u : 0u), 8); } while (x); }
    }
    if (bw.overflow) return 0;
    size_t blen = vc_bw_finish(&bw);
    return boff + blen;
}
static void rans_dec(i16 *q, size_t n, const u8 *in, size_t len, const rans_table *t) {
    size_t tok_bytes = (size_t)in[0] | ((size_t)in[1] << 8) | ((size_t)in[2] << 16) | ((size_t)in[3] << 24);
    size_t tok_off = 4; const u8 *toks = in + tok_off;
    size_t boff = tok_off + tok_bytes;
    vc_bitreader br; vc_br_init(&br, in + boff, len - boff);
    size_t pos = 0;
    u32 hi = (pos + 1 < tok_bytes) ? (toks[pos] | ((u32)toks[pos+1] << 8)) : 0; pos += 2;
    u32 lo = (pos + 1 < tok_bytes) ? (toks[pos] | ((u32)toks[pos+1] << 8)) : 0; pos += 2;
    u32 state = (hi << 16) | lo;
    for (size_t i = 0; i < n; ++i) {
        u32 slot = state & (PS - 1);
        u32 sym = t->slot2sym[slot];
        state = t->freq[sym] * (state >> PB) + slot - t->cum[sym];
        while (state < RL) { u32 w = (pos + 1 < tok_bytes) ? (toks[pos] | ((u32)toks[pos+1] << 8)) : 0; pos += 2; state = (state << 16) | w; }
        u32 u;
        if (sym == ESC) { u = 0; u32 sh = 0, b; do { b = vc_br_get(&br, 8); u |= (b & 0x7fu) << sh; sh += 7; } while (b & 0x80u); }
        else u = sym;
        q[i] = unzz(u);
    }
}

// --- EG2024 (1): per-band shared-table coding. The atom's AVOX freq-scanned
// levels are split into `nb` bands by the band map; each band's subsequence is
// coded against its OWN shared table (tbl[band]). Layout: per band a u24 length
// prefix then that band's rANS stream. Decode reverses it, scattering each band's
// decoded levels back to their scan positions. Band-NONE callers use rans_enc.
static size_t rans_enc_banded(u8 *out, size_t cap, const i16 *q, size_t n,
                              const rans_table *tbl, const u8 *bmap, u32 nb) {
    i16 *scratch = (i16 *)malloc(n * sizeof(i16));
    if (!scratch) return 0;
    size_t off = 0;
    for (u32 b = 0; b < nb; ++b) {
        size_t m = 0;
        for (size_t i = 0; i < n; ++i) if (bmap[i] == b) scratch[m++] = q[i];
        if (off + 3 > cap) { free(scratch); return 0; }
        size_t plen = rans_enc(out + off + 3, cap - off - 3, scratch, m, &tbl[b]);
        if (!plen && m) { free(scratch); return 0; }
        out[off] = (u8)plen; out[off+1] = (u8)(plen>>8); out[off+2] = (u8)(plen>>16);
        off += 3 + plen;
    }
    free(scratch);
    return off;
}
static void rans_dec_banded(i16 *q, size_t n, const u8 *in, size_t len,
                            const rans_table *tbl, const u8 *bmap, u32 nb) {
    (void)len;
    i16 *scratch = (i16 *)malloc(n * sizeof(i16));
    size_t off = 0;
    for (u32 b = 0; b < nb; ++b) {
        size_t m = 0; for (size_t i = 0; i < n; ++i) if (bmap[i] == b) ++m;
        size_t plen = (size_t)in[off] | ((size_t)in[off+1]<<8) | ((size_t)in[off+2]<<16);
        off += 3;
        if (m) rans_dec(scratch, m, in + off, plen, &tbl[b]);
        off += plen;
        size_t j = 0; for (size_t i = 0; i < n; ++i) if (bmap[i] == b) q[i] = scratch[j++];
    }
    free(scratch);
}

// RLGR (table-free) for the VC_ENT_RLGR mode — declared in blocks.h, lives in
// entropy/rlgr.c (already in the build).
size_t vc_rlgr_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rlgr_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

// ===========================================================================
// Traversal orderings over a chunk's chunk_atoms^3 atom grid (B axis).
// We enumerate the atoms of ONE chunk in coding order; raster across chunks.
// ===========================================================================
static inline u32 morton3(u32 x, u32 y, u32 z) {
    u32 r = 0;
    for (u32 i = 0; i < 10; ++i) {
        r |= ((x >> i) & 1u) << (3 * i + 0);
        r |= ((y >> i) & 1u) << (3 * i + 1);
        r |= ((z >> i) & 1u) << (3 * i + 2);
    }
    return r;
}
// Hilbert d2xyz / xyz2d for a 3D cube of side 2^bits (Skilling's algorithm).
static u32 hilbert_d(u32 bits, u32 x, u32 y, u32 z) {
    u32 X[3] = { x, y, z };
    u32 M = 1u << (bits - 1), P, Q, t;
    // Inverse undo excess work
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        for (int i = 0; i < 3; ++i) {
            if (X[i] & Q) X[0] ^= P;
            else { t = (X[0] ^ X[i]) & P; X[0] ^= t; X[i] ^= t; }
        }
    }
    for (int i = 1; i < 3; ++i) X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1) if (X[2] & Q) t ^= Q - 1;
    for (int i = 0; i < 3; ++i) X[i] ^= t;
    // interleave bits of X[] -> distance (transpose)
    u32 d = 0;
    for (u32 b = 0; b < bits; ++b)
        for (int i = 0; i < 3; ++i)
            d |= ((X[i] >> b) & 1u) << (3 * b + (2 - i));
    return d;
}

// Build coding order: order[k] = linear atom index (z*ca2 + y*ca + x) within the
// chunk, for k = 0..na-1. Inverse rank[] gives coding rank of each atom.
static void build_order(vc_traversal trav, u32 ca, u32 *order, u32 *rank) {
    u32 na = ca * ca * ca;
    if (trav == VC_TRAV_RASTER) {
        for (u32 i = 0; i < na; ++i) { order[i] = i; rank[i] = i; }
        return;
    }
    // bits = ceil(log2(ca))
    u32 bits = 0; while ((1u << bits) < ca) ++bits; if (bits == 0) bits = 1;
    // pair (code, linear) then sort by code
    typedef struct { u32 code, lin; } pr;
    pr *p = (pr *)malloc(na * sizeof(pr));
    u32 m = 0;
    for (u32 z = 0; z < ca; ++z)
    for (u32 y = 0; y < ca; ++y)
    for (u32 x = 0; x < ca; ++x) {
        u32 code = (trav == VC_TRAV_MORTON) ? morton3(x, y, z) : hilbert_d(bits, x, y, z);
        p[m].code = code; p[m].lin = z * ca * ca + y * ca + x; ++m;
    }
    // insertion/qsort by code
    for (u32 i = 1; i < na; ++i) { pr key = p[i]; u32 j = i; while (j && p[j-1].code > key.code) { p[j] = p[j-1]; --j; } p[j] = key; }
    for (u32 k = 0; k < na; ++k) { order[k] = p[k].lin; rank[p[k].lin] = k; }
    free(p);
}

// ===========================================================================
// Stencil neighbor offsets (causal subset = neighbors with LOWER coding rank).
// We generate all neighbor (dz,dy,dx) in {-1,0,1}^3 \ {0} up to the connectivity,
// then at coding time keep only already-coded ones (rank lower).
// ===========================================================================
typedef struct { int dz, dy, dx; } nbr;
static u32 stencil_offsets(vc_stencil s, nbr *out) {
    u32 n = 0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int man = (dz!=0) + (dy!=0) + (dx!=0);
        if (man == 0) continue;
        int keep = 0;
        if (s == VC_STENCIL_6  && man == 1) keep = 1;
        if (s == VC_STENCIL_18 && man <= 2) keep = 1;
        if (s == VC_STENCIL_26 && man <= 3) keep = 1;
        if (keep) { out[n].dz = dz; out[n].dy = dy; out[n].dx = dx; ++n; }
    }
    return n;
}

// ===========================================================================
// Archive layout (in-memory). One member, lattice of Ax*Ay*Az atoms grouped into
// chunks of chunk_atoms^3. We keep a flat per-atom record for simplicity of the
// bake-off (the byte accounting below charges the REAL on-disk costs: shared
// table once per chunk, delta-coded directory, halo). Random access uses the
// per-atom byte ranges + the chunk table + prediction ancestors.
// ===========================================================================
typedef struct {
    u32 off, len;     // payload byte range within its chunk blob
    i16 dc_coef;      // reconstructed DC coefficient (for prediction + decode)
    i16 dc_resid_q;   // quantized DC residual level actually stored
    u8  absent;       // all-zero atom
    u8  intra;        // coded intra (no prediction) — boundary fallback
} bg_atom;

typedef struct {
    rans_table table;     // shared table (M1 / M0-per-atom unused field for RLGR)
    rans_table btable[VC_NBAND_MAX];  // EG2024 (1): per-band shared tables
    u8 uniform;           // EG2024 (4): chunk is a single value (skipped); val in uval
    u8 uval;              // the constant value when uniform
    u8 *blob;             // concatenated atom payloads
    size_t blob_len;
    u32 a0z, a0y, a0x;    // chunk's atom-origin in the lattice
    u32 caz, cay, cax;    // chunk's atom extent (<= chunk_atoms, clipped)
} bg_chunk;

struct vc_bg_archive {
    vc_bg_cfg cfg;
    u32 dz, dy, dx;
    u32 Ax, Ay, Az;       // lattice atom dims
    bg_atom *atoms;       // [Az*Ay*Ax], lattice raster index
    bg_chunk *chunks;     // [n_chunks]
    u32 ncz, ncy, ncx, n_chunks;
    // DC sub-volume variant (cfg.dc_subvol): decoded DC coefficient per atom,
    // [Az*Ay*Ax] lattice raster. Decoded ONCE up front; atom decode just looks up.
    i16 *dc_sub;          // NULL unless dc_subvol active
    // CURVE-GROUP mode (cfg.group_mode==VC_GROUP_CURVE): a global space-filling
    // curve over the lattice bounding cube. group_of[gi] = group id owning atom
    // gi (lattice raster index). Groups are contiguous runs of group_n atoms in
    // curve order; each group is stored in chunks[group_id] (reusing bg_chunk as
    // a group record: its table + blob + directory). The DC residual stays in the
    // per-atom directory (at->dc_resid_q), so random access touches only the one
    // atom's payload. group_of doubles as the "group-index table".
    u32 *group_of;        // [Az*Ay*Ax] group id per atom (NULL in box mode)
    u32  curve_bits;      // curve cube side = 2^curve_bits (>= max lattice dim)
    // Variable-size grouping (I1 drift) needs explicit boundaries: group g spans
    // curve positions [group_start[g], group_start[g+1]). For FIXED grouping this
    // is simply g*N. Length n_chunks+1. Built in curve_encode, reused by decode.
    u32 *group_start;     // [n_chunks+1] curve-order start of each group
};

static inline u32 lat_idx(const vc_bg_archive *a, u32 z, u32 y, u32 x) {
    return (z * a->Ay + y) * a->Ax + x;
}
// which chunk owns lattice atom (z,y,x), and that chunk's index
static inline u32 chunk_of(const vc_bg_archive *a, u32 z, u32 y, u32 x) {
    u32 ca = a->cfg.chunk_atoms;
    u32 cz = z / ca, cy = y / ca, cx = x / ca;
    return (cz * a->ncy + cy) * a->ncx + cx;
}

// Gather one 16^3 atom out of the volume (zero-padded at the lattice edge).
static void gather_atom(u8 *out, const u8 *vol, u32 dz, u32 dy, u32 dx,
                        u32 az, u32 ay, u32 ax) {
    u32 oz = az * A, oy = ay * A, ox = ax * A;
    for (u32 z = 0; z < A; ++z) {
        u32 vz = oz + z;
        for (u32 y = 0; y < A; ++y) {
            u32 vy = oy + y;
            u8 *o = out + (z * A + y) * A;
            if (vz >= dz || vy >= dy) { memset(o, 0, A); continue; }
            u32 cw = (ox + A <= dx) ? A : (ox < dx ? dx - ox : 0u);
            if (cw) memcpy(o, vol + ((size_t)vz * dy + vy) * dx + ox, cw);
            if (cw < A) memset(o + cw, 0, A - cw);
        }
    }
}
static void scatter_atom(u8 *vol, const u8 *in, u32 dz, u32 dy, u32 dx,
                         u32 az, u32 ay, u32 ax) {
    u32 oz = az * A, oy = ay * A, ox = ax * A;
    for (u32 z = 0; z < A; ++z) {
        u32 vz = oz + z; if (vz >= dz) break;
        for (u32 y = 0; y < A; ++y) {
            u32 vy = oy + y; if (vy >= dy) break;
            u32 cw = (ox + A <= dx) ? A : (ox < dx ? dx - ox : 0u);
            if (cw) memcpy(vol + ((size_t)vz * dy + vy) * dx + ox, in + (z * A + y) * A, cw);
        }
    }
}

// DC-prediction reference for an atom: mean of available causal stencil
// neighbors' reconstructed DC coefficients (a->atoms[].dc_coef). The "available"
// set depends on edge policy. Returns predicted DC coefficient and the list of
// ancestor lattice indices used (for the touched-atom accounting).
static i16 predict_dc(const vc_bg_archive *a, u32 z, u32 y, u32 x,
                      const nbr *st, u32 nst, const u32 *rank_of,
                      u32 my_rank, int *was_intra,
                      u32 *anc, u32 *nanc) {
    *nanc = 0;
    if (a->cfg.stencil == VC_STENCIL_NONE) { *was_intra = 1; return 0; }
    i32 acc = 0; u32 cnt = 0;
    u32 ca = a->cfg.chunk_atoms;
    u32 mycz = z / ca, mycy = y / ca, mycx = x / ca;
    for (u32 i = 0; i < nst; ++i) {
        int nz = (int)z + st[i].dz, ny = (int)y + st[i].dy, nx = (int)x + st[i].dx;
        if (nz < 0 || ny < 0 || nx < 0) continue;
        if ((u32)nz >= a->Az || (u32)ny >= a->Ay || (u32)nx >= a->Ax) continue;
        u32 ni = lat_idx(a, (u32)nz, (u32)ny, (u32)nx);
        // chunk membership for edge policy
        int same_chunk = ((u32)nz / ca == mycz && (u32)ny / ca == mycy && (u32)nx / ca == mycx);
        if (!same_chunk) {
            if (a->cfg.edge == VC_EDGE_SELF) continue;  // self-contained: ignore
            // HALO / FETCH both make the neighbor available; HALO stores its DC in
            // a shell (charged in stats), FETCH reads the neighbor chunk live.
        }
        // causality: only neighbors already coded (lower rank within same chunk,
        // or in an earlier-coded chunk). For cross-chunk we rely on raster chunk
        // order: earlier chunk index => already coded.
        u32 nchunk = chunk_of(a, (u32)nz, (u32)ny, (u32)nx);
        u32 mychunk = chunk_of(a, z, y, x);
        int causal;
        if (nchunk < mychunk) causal = 1;
        else if (nchunk > mychunk) causal = 0;
        else causal = (rank_of[ni] < my_rank);  // intra-chunk: by traversal rank
        (void)my_rank;
        if (!causal) continue;
        if (a->atoms[ni].absent && a->atoms[ni].dc_coef == 0) {
            // absent neighbor contributes DC 0 (still counts as a real neighbor)
        }
        acc += a->atoms[ni].dc_coef; ++cnt;
        anc[(*nanc)++] = ni;
    }
    if (cnt == 0) { *was_intra = 1; return 0; }
    *was_intra = 0;
    return (i16)((acc + (i32)cnt / 2) / (i32)cnt);
}

// ===========================================================================
// DC SUB-VOLUME (JPEG-XL "DC frame" trick). All atoms' quantized DC LEVELS form
// a small Az*Ay*Ax mini-volume that is globally predicted (causal raster, mean of
// available left/up/front reconstructed DC levels) and rANS-coded ONCE as a
// separate stream, independent of chunk walls and atom decode order. Decoupling
// DC from the per-atom decode keeps 16^3 random access at touched==1: resolving
// an atom's DC is an O(1) lookup into the decoded sub-volume, no ancestor cone.
//
// Predictor: causal mean of the <=3 already-coded face neighbors (-x,-y,-z) of
// the DC level grid. Reconstruction is exact (we code level residuals, not value
// residuals) so encoder/decoder stay in lockstep with integer arithmetic.
// ===========================================================================
static inline i16 dcsv_predict(const i16 *lvl, u32 Az, u32 Ay, u32 Ax,
                               u32 z, u32 y, u32 x) {
    (void)Az;                         // raster index needs only Ay,Ax
    i32 acc = 0; u32 cnt = 0;
    if (x) { acc += lvl[(z*Ay + y)*Ax + (x-1)]; ++cnt; }
    if (y) { acc += lvl[(z*Ay + (y-1))*Ax + x]; ++cnt; }
    if (z) { acc += lvl[((z-1)*Ay + y)*Ax + x]; ++cnt; }
    if (!cnt) return 0;
    return (i16)((acc + (i32)cnt/2) / (i32)cnt);
}

// Encode the DC level grid into a freshly malloc'd stream. Returns stream (caller
// frees) and sets *out_len; charges *stat_bytes. Residuals = level - causal pred.
static u8 *dcsv_encode(const i16 *lvl, u32 Az, u32 Ay, u32 Ax,
                       size_t *out_len) {
    u32 n = Az*Ay*Ax;
    i16 *resid = (i16*)malloc((size_t)n*sizeof(i16));
    for (u32 z=0; z<Az; ++z) for (u32 y=0; y<Ay; ++y) for (u32 x=0; x<Ax; ++x) {
        u32 i = (z*Ay + y)*Ax + x;
        resid[i] = (i16)(lvl[i] - dcsv_predict(lvl, Az,Ay,Ax, z,y,x));
    }
    u64 counts[NSYM]; memset(counts,0,sizeof(counts));
    rans_hist(resid, n, counts);
    rans_table t; rt_build(&t, counts);
    size_t cap = (size_t)n*4 + NSYM*2 + 64;
    u8 *out = (u8*)malloc(cap);
    // [u32 n][NSYM*2 table freqs][rANS payload]
    out[0]=(u8)n; out[1]=(u8)(n>>8); out[2]=(u8)(n>>16); out[3]=(u8)(n>>24);
    for (u32 s=0;s<NSYM;++s){ out[4+2*s]=(u8)(t.freq[s]&0xff); out[4+2*s+1]=(u8)(t.freq[s]>>8); }
    size_t hdr = 4 + NSYM*2;
    size_t plen = rans_enc(out+hdr, cap-hdr, resid, n, &t);
    free(resid);
    if (!plen) { free(out); return NULL; }
    *out_len = hdr + plen;
    return out;
}

// Decode the DC level grid (inverse of dcsv_encode), then add back the causal
// prediction to recover absolute DC levels in place.
static void dcsv_decode(const u8 *in, size_t len, i16 *lvl_out,
                        u32 Az, u32 Ay, u32 Ax) {
    u32 n = (u32)in[0] | ((u32)in[1]<<8) | ((u32)in[2]<<16) | ((u32)in[3]<<24);
    rans_table t;
    for (u32 s=0;s<NSYM;++s) t.freq[s] = (u16)(in[4+2*s] | ((u32)in[4+2*s+1]<<8));
    rt_finish(&t);
    size_t hdr = 4 + NSYM*2;
    i16 *resid = (i16*)malloc((size_t)n*sizeof(i16));
    rans_dec(resid, n, in+hdr, len-hdr, &t);
    for (u32 z=0; z<Az; ++z) for (u32 y=0; y<Ay; ++y) for (u32 x=0; x<Ax; ++x) {
        u32 i = (z*Ay + y)*Ax + x;
        lvl_out[i] = (i16)(resid[i] + dcsv_predict(lvl_out, Az,Ay,Ax, z,y,x));
    }
    free(resid);
}

// ===========================================================================
// CURVE-GROUP MODE (the curve-groups experiment). The sharing+fetch unit is a
// run of N consecutive 16^3 atoms along a GLOBAL space-filling curve over the
// whole lattice (no axis-aligned chunk boxes). Same winning stack otherwise:
// DCT-16^3 + HF-quant + shared rANS table (or RLGR), NO prediction. The DC level
// is carried per-atom in the group directory (dc_resid_q holds the absolute DC
// level, was_intra=1 always), so single-atom random access touches exactly 1
// payload after an O(1) group-index lookup.
// ===========================================================================

// Compute the global curve index of a lattice atom (gz,gy,gx) for the chosen
// curve over a cube of side 2^bits. Atoms outside the lattice are skipped by the
// caller; the curve simply orders the bounding cube and we keep the present ones.
static inline u32 curve_index(vc_traversal trav, u32 bits, u32 gz, u32 gy, u32 gx) {
    if (trav == VC_TRAV_HILBERT) return hilbert_d(bits, gx, gy, gz);
    return morton3(gx, gy, gz);     // Morton (raster falls back to Morton here)
}

typedef struct { u32 code, lin; } curve_pr;
static int curve_pr_cmp(const void *pa, const void *pb) {
    u32 ca = ((const curve_pr *)pa)->code, cb = ((const curve_pr *)pb)->code;
    return ca < cb ? -1 : (ca > cb ? 1 : 0);
}

// Build the global curve order over the lattice: order_lat[k] = lattice raster
// index of the k-th present atom in curve order (k=0..nat-1). Returns curve bits.
static u32 build_curve_order(const vc_bg_archive *a, u32 *order_lat) {
    u32 maxd = a->Az > a->Ay ? a->Az : a->Ay; if (a->Ax > maxd) maxd = a->Ax;
    u32 bits = 1; while ((1u << bits) < maxd) ++bits;
    u32 nat = a->Az * a->Ay * a->Ax;
    curve_pr *p = (curve_pr *)malloc((size_t)nat * sizeof(curve_pr));
    u32 m = 0;
    for (u32 z = 0; z < a->Az; ++z)
    for (u32 y = 0; y < a->Ay; ++y)
    for (u32 x = 0; x < a->Ax; ++x) {
        p[m].code = curve_index(a->cfg.traversal, bits, z, y, x);
        p[m].lin  = (z * a->Ay + y) * a->Ax + x;
        ++m;
    }
    qsort(p, nat, sizeof(curve_pr), curve_pr_cmp);
    for (u32 k = 0; k < nat; ++k) order_lat[k] = p[k].lin;
    free(p);
    return bits;
}

// --- varint byte cost of an unsigned value (LEB128) -------------------------
static inline size_t varint_cost(u32 v) { size_t n = 1; while (v >>= 7) ++n; return n; }

// --- I1 drift boundary: total-variation distance between two normalised hists.
// Returns 0..1. Used to decide when a curve-group has drifted enough to split.
static double hist_tv(const u64 *acc, u64 acc_tot, const u64 *win, u64 win_tot) {
    if (!acc_tot || !win_tot) return 0.0;
    double d = 0.0;
    for (u32 s = 0; s < NSYM; ++s) {
        double pa = (double)acc[s] / (double)acc_tot;
        double pw = (double)win[s] / (double)win_tot;
        d += (pa > pw ? pa - pw : pw - pa);
    }
    return 0.5 * d;            // total-variation distance in [0,1]
}

// --- E1/E2 table coding cost + EXACT round-trip of the stored freq table.
// Returns the byte cost of storing table `cur`'s quantised freqs under the chosen
// coding, AND overwrites cur->freq with the value a decoder would reconstruct
// (so decode == bytes). For FULL: raw 12-bit freqs (NSYM*2 bytes, no change).
// For DELTA: zig-zag-varint of (cur.freq[s]-ref.freq[s]); reconstruction adds the
// delta back to ref (lossless -> cur unchanged, but cost reflects the delta).
// `ref` is the previous group's already-reconstructed table (DELTA) or the global
// base table (BASE); NULL on the first DELTA group (falls back to FULL cost).
static size_t table_code_cost(rans_table *cur, const rans_table *ref,
                              vc_table_coding mode) {
    if (mode == VC_TABLE_FULL || !ref) {
        // raw store: 2 bytes/sym. Reconstruction is identity.
        return NSYM * 2;
    }
    // DELTA / BASE: zig-zag varint of per-symbol freq delta vs ref. Lossless, so
    // cur->freq is already exactly what a decoder reconstructs (ref + delta).
    size_t bytes = 0;
    for (u32 s = 0; s < NSYM; ++s) {
        i32 d = (i32)cur->freq[s] - (i32)ref->freq[s];
        u32 u = (u32)((d << 1) ^ (d >> 31));   // zig-zag
        bytes += varint_cost(u);
    }
    return bytes;
}

static int curve_encode(const u8 *vol, vc_bg_archive *a, vc_bg_stats *st) {
    const vc_bg_cfg *cfg = &a->cfg;
    u32 dz = a->dz, dy = a->dy, dx = a->dx;
    u32 nat = a->Az * a->Ay * a->Ax;
    f32 base = cfg->step;
    u32 N = cfg->group_n ? cfg->group_n : 256u;
    int drift = (cfg->boundary == VC_BOUND_DRIFT);
    int rans  = (cfg->entropy == VC_ENT_RANS_SHARED);

    u32 *order_lat = (u32 *)malloc((size_t)nat * sizeof(u32));
    a->group_of = (u32 *)malloc((size_t)nat * sizeof(u32));
    if (!order_lat || !a->group_of) { free(order_lat); return 1; }
    a->curve_bits = build_curve_order(a, order_lat);

    // ---- PHASE A: transform+quant EVERY atom in curve order. Store all levels
    // (qstore), per-atom histogram (for drift boundaries), and the absolute DC
    // level. Curve order means qstore[k] is the k-th atom along the curve.
    i16 *qstore = (i16 *)malloc((size_t)nat * AVOX * sizeof(i16));
    i16 *dc_lvl = (i16 *)malloc((size_t)nat * sizeof(i16)); // absolute DC level/atom
    u8  *avox = (u8 *)malloc(AVOX);
    i16 *coef = (i16 *)malloc(AVOX * sizeof(i16));
    i16 *qsc  = (i16 *)malloc(AVOX * sizeof(i16));
    u8  *pay  = (u8 *)malloc(AVOX * 4 + 64);
    if (!qstore||!dc_lvl||!avox||!coef||!qsc||!pay) {
        free(qstore);free(dc_lvl);free(avox);free(coef);free(qsc);free(pay);free(order_lat); return 1; }

    for (u32 k = 0; k < nat; ++k) {
        u32 gi = order_lat[k];
        u32 gz = gi / (a->Ay * a->Ax), gy = (gi / a->Ax) % a->Ay, gx = gi % a->Ax;
        bg_atom *at = &a->atoms[gi];
        i16 *qs = qstore + (size_t)k * AVOX;
        gather_atom(avox, vol, dz, dy, dx, gz, gy, gx);
        int allz = 1; for (u32 i = 0; i < AVOX; ++i) if (avox[i]) { allz = 0; break; }
        if (allz) { at->absent = 1; at->dc_coef = 0; at->intra = 1;
                    memset(qs, 0, AVOX*sizeof(i16)); dc_lvl[k] = 0; continue; }
        bg_dct16_fwd(coef, avox, 0);
        i32 dc_raw = quant_atom(qsc, coef, base);
        i16 dc_q_full = dc_quant(dc_raw, base);
        at->dc_coef = dc_dequant(dc_q_full, base);
        at->intra = 1; at->absent = 0;
        memcpy(qs, qsc, AVOX*sizeof(i16));
        qs[0] = 0;
        dc_lvl[k] = dc_q_full;            // absolute DC level (curve order)
    }

    // ---- B1: DC-only prediction from the CURVE PREDECESSOR. Stored per atom in
    // the directory is the residual (dc_lvl[k] - dc_lvl[k-1]); k=0 stores absolute.
    // touched stays 1 (the predecessor's DC is a directory read on full decode; for
    // random access we walk back along the curve reading only directory DC levels —
    // accounted as touched==1 because no AC payload of a predecessor is decoded).
    // We materialise the per-atom STORED dc level into at->dc_resid_q below per the
    // boundary pass so the chosen reset semantics (reset at group start) hold.

    // ---- PHASE B: form group boundaries. FIXED: every N atoms. DRIFT: end a group
    // when the next atom's histogram pushes the group's running histogram past the
    // TV-distance threshold, capped at [min, N].
    u32 *gstart = (u32 *)malloc((size_t)(nat + 2) * sizeof(u32));
    if (!gstart) { free(qstore);free(dc_lvl);free(avox);free(coef);free(qsc);free(pay);free(order_lat); return 1; }
    u32 ng = 0; gstart[0] = 0;
    if (!drift) {
        ng = 0; for (u32 k = 0; k < nat; k += N) gstart[ng++] = k;
        gstart[ng] = nat;
    } else {
        u32 gmin = 8; if (gmin > N) gmin = N;
        u64 acc[NSYM]; memset(acc, 0, sizeof(acc)); u64 acc_tot = 0;
        u64 win[NSYM]; u32 cur_start = 0; ng = 0; gstart[0] = 0;
        for (u32 k = 0; k < nat; ++k) {
            memset(win, 0, sizeof(win));
            rans_hist(qstore + (size_t)k*AVOX, AVOX, win);
            u64 win_tot = 0; for (u32 s=0;s<NSYM;++s) win_tot += win[s];
            u32 sz = k - cur_start;
            double d = hist_tv(acc, acc_tot, win, win_tot);
            int split = (sz >= gmin && d > (double)cfg->drift_thresh) || (sz >= N);
            if (split && sz > 0) {
                gstart[++ng] = k; cur_start = k;
                memset(acc, 0, sizeof(acc)); acc_tot = 0;
            }
            for (u32 s=0;s<NSYM;++s) acc[s] += win[s];
            acc_tot += win_tot;
        }
        gstart[++ng] = nat;
    }
    a->n_chunks = ng; st->n_chunks = ng;
    a->ncz = a->ncy = a->ncx = 0;
    a->group_start = (u32 *)malloc((size_t)(ng + 1) * sizeof(u32));
    a->chunks = (bg_chunk *)calloc(ng, sizeof(bg_chunk));
    if (!a->group_start || !a->chunks) { free(gstart);free(qstore);free(dc_lvl);
        free(avox);free(coef);free(qsc);free(pay);free(order_lat); return 1; }
    for (u32 g = 0; g <= ng; ++g) a->group_start[g] = gstart[g];

    rans_table base_tbl; int have_base = 0;        // E2 super-group base
    if (rans && cfg->table_coding == VC_TABLE_BASE) {
        u64 gc[NSYM]; memset(gc, 0, sizeof(gc));
        for (u32 k = 0; k < nat; ++k) if (!a->atoms[order_lat[k]].absent)
            rans_hist(qstore + (size_t)k*AVOX, AVOX, gc);
        rt_build(&base_tbl, gc);
        st->table_bytes += NSYM * 2;               // base stored once
        have_base = 1;
    }
    rans_table prev_tbl; int have_prev = 0;        // E1 previous-group table

    // ---- PHASE C: per group, build table (+code it E1/E2), entropy-code atoms.
    for (u32 g = 0; g < ng; ++g) {
        bg_chunk *C = &a->chunks[g];
        u32 k0 = gstart[g], k1 = gstart[g+1], cnt = k1 - k0;
        u64 counts[NSYM]; memset(counts, 0, sizeof(counts));
        for (u32 k = k0; k < k1; ++k) {
            u32 gi = order_lat[k]; a->group_of[gi] = g;
            if (!a->atoms[gi].absent) rans_hist(qstore + (size_t)k*AVOX, AVOX, counts);
        }

        if (rans) {
            rt_build(&C->table, counts);
            // E1/E2: charge the (possibly delta-coded) table cost. table_code_cost
            // also guarantees C->table.freq equals the decoder's reconstruction.
            const rans_table *ref = NULL;
            if (cfg->table_coding == VC_TABLE_DELTA && have_prev) ref = &prev_tbl;
            else if (cfg->table_coding == VC_TABLE_BASE && have_base) ref = &base_tbl;
            st->table_bytes += table_code_cost(&C->table, ref, cfg->table_coding);
            prev_tbl = C->table; have_prev = 1;
        }

        u8 *blob = (u8 *)malloc((size_t)cnt * (AVOX * 4 + 64) + 16);
        if (!blob) { free(gstart);free(qstore);free(dc_lvl);free(avox);free(coef);
            free(qsc);free(pay);free(order_lat); return 1; }
        size_t bl = 0; size_t dirbytes = 0;
        for (u32 k = k0; k < k1; ++k) {
            u32 gi = order_lat[k];
            bg_atom *at = &a->atoms[gi];
            const i16 *q = qstore + (size_t)k*AVOX;
            // --- DC stored level: B1 curve-predecessor prediction (reset at group
            // start under FIXED so the directory stays self-contained per group; the
            // predecessor at k-1 is always the spatial neighbour under the curve).
            i16 dc_store;
            if (cfg->dc_pred_curve && k > k0) dc_store = (i16)(dc_lvl[k] - dc_lvl[k-1]);
            else                              dc_store = dc_lvl[k];
            at->dc_resid_q = dc_store;
            at->dc_coef = dc_dequant(dc_lvl[k], base);
            if (at->absent) { at->off = (u32)bl; at->len = 0; st->n_absent_atoms++; }
            else {
                size_t cap = AVOX * 4 + 64, plen;
                if (cfg->entropy == VC_ENT_RLGR)        plen = vc_rlgr_encode(pay, cap, q, AVOX);
                else if (rans)                          plen = rans_enc(pay, cap, q, AVOX, &C->table);
                else { u64 c1[NSYM]; memset(c1,0,sizeof(c1)); rans_hist(q, AVOX, c1);
                       rans_table t1; rt_build(&t1, c1);
                       for (u32 s=0;s<NSYM;++s){ pay[2*s]=(u8)(t1.freq[s]&0xff); pay[2*s+1]=(u8)(t1.freq[s]>>8); }
                       plen = rans_enc(pay + NSYM*2, cap - NSYM*2, q, AVOX, &t1) + NSYM*2; }
                memcpy(blob + bl, pay, plen);
                at->off = (u32)bl; at->len = (u32)plen; bl += plen;
            }
            dirbytes += varint_cost(at->len);
            dirbytes += varint_cost(zz(at->dc_resid_q));
        }
        dirbytes += (cnt + 7) / 8;        // absent-flag bits
        // I2 nested sub-groups: tiny per-sub-group index (one boundary marker per
        // sub-run). Charged as 1 byte / sub-group. (The DC base/delta itself rides
        // in the already-counted directory DC residuals; this measures the index
        // overhead the nesting costs.)
        if (cfg->nested_sub) dirbytes += (cnt + cfg->nested_sub - 1) / cfg->nested_sub;
        C->blob = blob; C->blob_len = bl;
        st->payload_bytes += bl;
        st->directory_bytes += dirbytes;
    }

    st->total_bytes = st->payload_bytes + st->directory_bytes + st->table_bytes
                    + st->halo_bytes + st->dc_subvol_bytes + 64;
    free(gstart);free(qstore);free(dc_lvl);free(avox);free(coef);free(qsc);free(pay);free(order_lat);
    return 0;
}

// ---------------------------------------------------------------------------
// ENCODE
// ---------------------------------------------------------------------------
int vc_bg_encode(const u8 *vol, u32 dz, u32 dy, u32 dx,
                 const vc_bg_cfg *cfg, vc_bg_archive **out, vc_bg_stats *st) {
    ensure_init();
    vc_bg_archive *a = (vc_bg_archive *)calloc(1, sizeof(*a));
    if (!a) return 1;
    a->cfg = *cfg; a->dz = dz; a->dy = dy; a->dx = dx;
    a->Az = vc_bg_natoms(dz); a->Ay = vc_bg_natoms(dy); a->Ax = vc_bg_natoms(dx);
    u32 nat = a->Az * a->Ay * a->Ax;
    a->atoms = (bg_atom *)calloc(nat, sizeof(bg_atom));
    if (!a->atoms) { vc_bg_free(a); return 1; }

    // CURVE-GROUP mode: a separate, self-contained path (no chunk boxes). Forces
    // the winning stack (no prediction, no DC sub-volume; DC carried per-atom).
    if (cfg->group_mode == VC_GROUP_CURVE) {
        memset(st, 0, sizeof(*st));
        st->n_atoms = nat;
        if (curve_encode(vol, a, st)) { vc_bg_free(a); return 1; }
        *out = a; return 0;
    }

    u32 ca = cfg->chunk_atoms;
    a->ncz = (a->Az + ca - 1) / ca; a->ncy = (a->Ay + ca - 1) / ca; a->ncx = (a->Ax + ca - 1) / ca;
    a->n_chunks = a->ncz * a->ncy * a->ncx;
    a->chunks = (bg_chunk *)calloc(a->n_chunks, sizeof(bg_chunk));
    if (!a->atoms || !a->chunks) { vc_bg_free(a); return 1; }

    // scratch
    u8  *avox = (u8 *)malloc(AVOX);
    i16 *coef = (i16 *)malloc(AVOX * sizeof(i16));
    i16 *qsc  = (i16 *)malloc(AVOX * sizeof(i16));
    u8  *pay  = (u8 *)malloc(AVOX * 4 + 64);
    // store every atom's quantized levels so a chunk can build a shared histogram
    // BEFORE coding (M1 needs the whole chunk's stats first).
    u32 namax = ca * ca * ca;
    i16 *qstore = (i16 *)malloc((size_t)namax * AVOX * sizeof(i16));
    i16 *dcq    = (i16 *)malloc((size_t)namax * sizeof(i16)); // DC residual level/atom
    u8  *intraf = (u8 *)malloc((size_t)namax);
    u32 *order = (u32 *)malloc(namax * sizeof(u32));
    u32 *rankc = (u32 *)malloc(namax * sizeof(u32));
    // rank over the WHOLE lattice (per-atom) for cross-chunk causality lookups
    u32 *rank_lat = (u32 *)malloc(nat * sizeof(u32));
    // DC sub-volume variant: lattice-wide grid of quantized DC LEVELS, filled in
    // pass 1 and coded once as a separate stream after all chunks (see below).
    i16 *dc_lvl = cfg->dc_subvol ? (i16 *)calloc(nat, sizeof(i16)) : NULL;
    if (!avox||!coef||!qsc||!pay||!qstore||!dcq||!intraf||!order||!rankc||!rank_lat
        || (cfg->dc_subvol && !dc_lvl)) {
        free(avox);free(coef);free(qsc);free(pay);free(qstore);free(dcq);free(intraf);
        free(order);free(rankc);free(rank_lat);free(dc_lvl); vc_bg_free(a); return 1;
    }

    nbr stoff[26]; u32 nst = stencil_offsets(cfg->stencil, stoff);
    f32 base = cfg->step;

    memset(st, 0, sizeof(*st));
    st->n_atoms = nat; st->n_chunks = a->n_chunks;

    // Precompute per-atom traversal rank across the lattice (rank within chunk).
    {
        u32 ordg[16*16*16], rkg[16*16*16];
        build_order(cfg->traversal, ca, ordg, rkg);
        for (u32 cz = 0; cz < a->ncz; ++cz)
        for (u32 cy = 0; cy < a->ncy; ++cy)
        for (u32 cx = 0; cx < a->ncx; ++cx) {
            u32 a0z = cz*ca, a0y = cy*ca, a0x = cx*ca;
            for (u32 lz = 0; lz < ca; ++lz)
            for (u32 ly = 0; ly < ca; ++ly)
            for (u32 lx = 0; lx < ca; ++lx) {
                u32 gz = a0z+lz, gy = a0y+ly, gx = a0x+lx;
                if (gz>=a->Az||gy>=a->Ay||gx>=a->Ax) continue;
                rank_lat[lat_idx(a,gz,gy,gx)] = rkg[(lz*ca+ly)*ca+lx];
            }
        }
    }

    // Encode chunk by chunk in raster chunk order (so cross-chunk causality holds).
    u32 ci = 0;
    for (u32 cz = 0; cz < a->ncz; ++cz)
    for (u32 cy = 0; cy < a->ncy; ++cy)
    for (u32 cx = 0; cx < a->ncx; ++cx, ++ci) {
        bg_chunk *C = &a->chunks[ci];
        C->a0z = cz*ca; C->a0y = cy*ca; C->a0x = cx*ca;
        C->caz = (C->a0z + ca <= a->Az) ? ca : a->Az - C->a0z;
        C->cay = (C->a0y + ca <= a->Ay) ? ca : a->Ay - C->a0y;
        C->cax = (C->a0x + ca <= a->Ax) ? ca : a->Ax - C->a0x;
        build_order(cfg->traversal, ca, order, rankc);

        u64 counts[NSYM]; memset(counts, 0, sizeof(counts));
        // EG2024 (1): per-band histograms. bcounts[b] = histogram of band b's
        // coefficients across the chunk. EG2024 (3): sparse prepass samples every
        // `stride`-th atom for histogram building (the encoded payload still codes
        // ALL atoms — only the table-fitting sample is sparsened).
        u32 nb = band_count(cfg->band_split);
        const u8 *bmap = band_map(cfg->band_split);
        u64 (*bcounts)[NSYM] = (u64(*)[NSYM])calloc(VC_NBAND_MAX, sizeof(u64[NSYM]));
        u32 prep = cfg->sparse_prepass ? cfg->sparse_prepass : 1u;
        // Pass 1: transform+quant every atom IN CODING ORDER, predict DC, store
        // levels, accumulate the chunk histogram for the shared table (M1).
        u32 nstored = 0; u32 hist_seen = 0;
        for (u32 k = 0; k < namax; ++k) {
            u32 lin = order[k];
            u32 lz = lin / (ca*ca), ly = (lin / ca) % ca, lx = lin % ca;
            if (lz >= C->caz || ly >= C->cay || lx >= C->cax) continue;
            u32 gz = C->a0z+lz, gy = C->a0y+ly, gx = C->a0x+lx;
            u32 gi = lat_idx(a, gz, gy, gx);
            bg_atom *at = &a->atoms[gi];

            gather_atom(avox, vol, dz, dy, dx, gz, gy, gx);
            // all-zero atom => absent
            int allz = 1; for (u32 i = 0; i < AVOX; ++i) if (avox[i]) { allz = 0; break; }
            if (allz) { at->absent = 1; at->dc_coef = 0; intraf[nstored] = 1;
                        memset(qstore + (size_t)nstored*AVOX, 0, AVOX*sizeof(i16));
                        dcq[nstored] = 0; nstored++; continue; }

            bg_dct16_fwd(coef, avox, 0);     // DC bias handled via coefficient
            i32 dc_raw = quant_atom(qsc, coef, base);

            i16 dc_q_full = dc_quant(dc_raw, base);
            int was_intra; i16 dc_resid; i16 dc_recon;
            if (cfg->dc_subvol) {
                // DC SUB-VOLUME: this atom's DC level joins the global DC grid,
                // coded separately later. No per-atom DC residual in the stream
                // or directory; reconstruction looks DC up from the decoded grid.
                dc_lvl[gi] = dc_q_full;
                dc_resid = 0; was_intra = 1;
                dc_recon = dc_dequant(dc_q_full, base);
            } else {
                // DC prediction from causal stencil neighbors (threaded path)
                u32 anc[26], nanc;
                i16 pred = predict_dc(a, gz, gy, gx, stoff, nst, rank_lat,
                                      rankc[lin], &was_intra, anc, &nanc);
                i16 dc_q_pred = dc_quant((i32)pred, base);
                dc_resid  = (i16)(dc_q_full - (was_intra ? 0 : dc_q_pred));
                // reconstructed DC coefficient (for downstream prediction + decode)
                dc_recon  = dc_dequant((i16)((was_intra ? 0 : dc_q_pred) + dc_resid), base);
            }
            at->dc_coef = dc_recon; at->intra = (u8)was_intra; at->absent = 0;

            // levels to entropy-code = freq-scanned AC levels; the DC residual is
            // carried in the SEEK DIRECTORY (dc_resid_q), NOT the stream, so that
            // cross-atom DC prediction stays compatible with O(1) random access.
            // Position 0 stays 0 (a cheap zero token).
            memcpy(qstore + (size_t)nstored*AVOX, qsc, AVOX*sizeof(i16));
            qstore[(size_t)nstored*AVOX + 0] = 0;
            dcq[nstored] = dc_resid; intraf[nstored] = (u8)was_intra;
            // sparse prepass: only every prep-th non-absent atom feeds the table.
            if ((hist_seen++ % prep) == 0) {
                const i16 *qh = qstore + (size_t)nstored*AVOX;
                rans_hist(qh, AVOX, counts);
                if (nb > 1) for (u32 i = 0; i < AVOX; ++i) {
                    u32 u = zz(qh[i]); bcounts[bmap[i]][u < ESC ? u : ESC]++;
                }
            }
            nstored++;
        }

        // Build shared table once (M1). For M0 we rebuild per atom below.
        // EG2024 (3) sparse prepass: the table is fit to a SAMPLE, so a symbol that
        // occurs in an un-sampled atom could get freq 0 and break rANS. Laplace-
        // smooth (every 0..254 symbol gets a floor count) so coverage is total.
        // This is the standard fix and keeps the encode-time saving (we still only
        // histogram the sample). ESC (255) is the catch-all bypass and never zero.
        if (cfg->entropy == VC_ENT_RANS_SHARED) {
            if (prep > 1) {
                for (u32 s = 0; s < ESC; ++s) {
                    counts[s] += 1;
                    for (u32 b = 0; b < nb; ++b) bcounts[b][s] += 1;
                }
            }
            if (nb > 1) {
                for (u32 b = 0; b < nb; ++b) rt_build(&C->btable[b], bcounts[b]);
                st->table_bytes += (size_t)nb * NSYM * 2;   // one table per band
            } else {
                rt_build(&C->table, counts);
                st->table_bytes += NSYM * 2;     // serialized 12-bit freqs
            }
        }
        free(bcounts);

        // EG2024 (4): skip metadata. Charge a tiny per-chunk index (min,max =2B +
        // 1 uniform flag bit). If the whole chunk reconstructs to a single value
        // (every atom absent, OR a constant-value chunk that quantized to DC-only
        // zero-AC + identical DC), flag it uniform and store ONE value instead of
        // the atom blobs+directory. We detect the cheap, common case: all atoms
        // absent (pure air) — extended to "all atoms have zero AC and equal DC".
        if (cfg->skip_meta) {
            st->skip_meta_bytes += 3;        // min(1)+max(1)+flags(1) per chunk
            int uni = 1; i16 dc0 = 0; int have = 0;
            for (u32 s = 0; s < nstored && uni; ++s) {
                const i16 *q = qstore + (size_t)s*AVOX;
                for (u32 i = 1; i < AVOX; ++i) if (q[i]) { uni = 0; break; }
            }
            // uniform DC across the chunk's atoms (use reconstructed dc_coef)
            if (uni) {
                u32 s2 = 0;
                for (u32 k = 0; k < namax && uni; ++k) {
                    u32 lin = order[k];
                    u32 lz=lin/(ca*ca), ly=(lin/ca)%ca, lx=lin%ca;
                    if (lz>=C->caz||ly>=C->cay||lx>=C->cax) continue;
                    u32 gz=C->a0z+lz, gy=C->a0y+ly, gx=C->a0x+lx;
                    i16 d = a->atoms[lat_idx(a,gz,gy,gx)].dc_coef;
                    if (!have) { dc0 = d; have = 1; } else if (d != dc0) uni = 0;
                    s2++;
                }
            }
            if (uni && nstored > 0) {
                C->uniform = 1;
                // reconstructed constant byte value: inverse-DCT of a DC-only atom
                // is flat; recover one sample via the volume directly (origin atom).
                C->uval = vol[((size_t)(C->a0z*A)*dy + (C->a0y*A))*dx + C->a0x*A];
                C->blob = NULL; C->blob_len = 0;
                st->n_uniform_chunks++;
                continue;     // skip Pass 2 + directory entirely for this chunk
            }
        }

        // Pass 2: entropy-code each stored atom into the chunk blob; record range.
        vc_bitwriter dummy; (void)dummy;
        u8 *blob = (u8 *)malloc((size_t)nstored * (AVOX * 4 + 64) + 16);
        size_t bl = 0; u32 si = 0;
        for (u32 k = 0; k < namax; ++k) {
            u32 lin = order[k];
            u32 lz = lin / (ca*ca), ly = (lin / ca) % ca, lx = lin % ca;
            if (lz >= C->caz || ly >= C->cay || lx >= C->cax) continue;
            u32 gz = C->a0z+lz, gy = C->a0y+ly, gx = C->a0x+lx;
            bg_atom *at = &a->atoms[lat_idx(a, gz, gy, gx)];
            const i16 *q = qstore + (size_t)si*AVOX;
            if (at->absent) { at->off = (u32)bl; at->len = 0; at->dc_resid_q = 0; si++; st->n_absent_atoms++; continue; }
            at->dc_resid_q = dcq[si];
            size_t cap = AVOX * 4 + 64;
            size_t plen;
            if (cfg->entropy == VC_ENT_RLGR) {
                plen = vc_rlgr_encode(pay, cap, q, AVOX);
            } else if (cfg->entropy == VC_ENT_RANS_SHARED && nb > 1) {
                plen = rans_enc_banded(pay, cap, q, AVOX, C->btable, bmap, nb);
            } else if (cfg->entropy == VC_ENT_RANS_SHARED) {
                plen = rans_enc(pay, cap, q, AVOX, &C->table);
            } else { // M0 independent: own table per atom
                u64 c1[NSYM]; memset(c1,0,sizeof(c1)); rans_hist(q, AVOX, c1);
                rans_table t1; rt_build(&t1, c1);
                // store the table inline in the payload (NSYM*2 bytes) then stream
                for (u32 s=0;s<NSYM;++s){ pay[2*s]=(u8)(t1.freq[s]&0xff); pay[2*s+1]=(u8)(t1.freq[s]>>8); }
                plen = rans_enc(pay + NSYM*2, cap - NSYM*2, q, AVOX, &t1);
                plen += NSYM*2;
            }
            memcpy(blob + bl, pay, plen);
            at->off = (u32)bl; at->len = (u32)plen; bl += plen;
            si++;
        }
        C->blob = blob; C->blob_len = bl;
        st->payload_bytes += bl;

        // Seek directory: N x {delta-coded offset, len}. Offsets are monotonic =>
        // store first offset + per-atom length; offset reconstructs by prefix sum.
        // Cost: ~varint(len) per atom + flags. Estimate: 2 bytes/atom typical.
        u32 ndir = C->caz * C->cay * C->cax;
        // delta-coded length varints
        size_t dirbytes = 0;
        {
            // recompute lengths in coding order for a realistic varint cost
            u32 s2 = 0;
            for (u32 k = 0; k < namax; ++k) {
                u32 lin = order[k];
                u32 lz = lin/(ca*ca), ly=(lin/ca)%ca, lx=lin%ca;
                if (lz>=C->caz||ly>=C->cay||lx>=C->cax) continue;
                u32 gz=C->a0z+lz, gy=C->a0y+ly, gx=C->a0x+lx;
                const bg_atom *da = &a->atoms[lat_idx(a,gz,gy,gx)];
                u32 L = da->len;
                do { L >>= 7; dirbytes++; } while (L);   // length varint
                // DC residual (zigzag varint) carried in the directory, not the
                // stream — this is what keeps DC prediction O(1)-random-access.
                u32 dz2 = zz(da->dc_resid_q);
                do { dz2 >>= 7; dirbytes++; } while (dz2);
                s2++;
            }
            dirbytes += (ndir + 7) / 8;                  // 1 absent-flag bit/atom
        }
        st->directory_bytes += dirbytes;

        // HALO cost: edge=halo stores the DC shell of this chunk's boundary atoms
        // so neighbor chunks can predict across the edge (2 bytes/boundary atom).
        if (cfg->edge == VC_EDGE_HALO) {
            u32 face = 0;
            for (u32 lz=0; lz<C->caz; ++lz)
            for (u32 ly=0; ly<C->cay; ++ly)
            for (u32 lx=0; lx<C->cax; ++lx)
                if (lz==0||ly==0||lx==0||lz==C->caz-1||ly==C->cay-1||lx==C->cax-1) face++;
            st->halo_bytes += (size_t)face * 2;
        }
    }

    // DC SUB-VOLUME: now that every atom's DC level is in dc_lvl, code the whole
    // grid once as a separate globally-predicted stream. We keep the DECODED grid
    // (== the encoded levels here; coding is lossless) in the archive so atom
    // decode is an O(1) lookup. This captures cross-atom DC redundancy without
    // threading DC through the atom decode order (preserves cheap random access).
    if (cfg->dc_subvol) {
        size_t dlen = 0;
        u8 *dstream = dcsv_encode(dc_lvl, a->Az, a->Ay, a->Ax, &dlen);
        if (!dstream) {
            free(avox);free(coef);free(qsc);free(pay);free(qstore);free(dcq);
            free(intraf);free(order);free(rankc);free(rank_lat);free(dc_lvl);
            vc_bg_free(a); return 1;
        }
        st->dc_subvol_bytes = dlen;
        // Reconstructed DC coefficient grid for decode-time O(1) lookup. Decode
        // the stream back exactly as the real decoder would (round-trips the DC
        // levels), then dequant — this exercises dcsv_decode and guarantees the
        // encoder's a->dc_sub matches what a from-bytes decode would reconstruct.
        i16 *dc_lvl_rec = (i16 *)malloc((size_t)nat * sizeof(i16));
        dcsv_decode(dstream, dlen, dc_lvl_rec, a->Az, a->Ay, a->Ax);
        free(dstream);
        a->dc_sub = (i16 *)malloc((size_t)nat*sizeof(i16));
        for (u32 i=0;i<nat;++i) a->dc_sub[i] = dc_dequant(dc_lvl_rec[i], base);
        free(dc_lvl_rec);
    }

    // shared-table-bytes only counted for SHARED above; M0 table cost rides inside
    // payloads. RLGR has no table. Total bytes = payload + directory + table + halo
    // + DC sub-volume + a small fixed lattice header.
    st->total_bytes = st->payload_bytes + st->directory_bytes + st->table_bytes
                    + st->halo_bytes + st->dc_subvol_bytes + 64 /*lattice header*/;

    free(dc_lvl);
    free(avox);free(coef);free(qsc);free(pay);free(qstore);free(dcq);free(intraf);
    free(order);free(rankc);free(rank_lat);
    *out = a;
    return 0;
}

// ---------------------------------------------------------------------------
// DECODE (full): decode every atom in coding order, reconstruct DC by prediction,
// inverse-quant, inverse-DCT, scatter. Mirrors the encoder exactly.
// ---------------------------------------------------------------------------
static void decode_one_atom_levels(const vc_bg_archive *a, const bg_chunk *C,
                                    const bg_atom *at, i16 *qsc) {
    if (at->absent) { memset(qsc, 0, AVOX*sizeof(i16)); return; }
    const u8 *p = C->blob + at->off;
    u32 nb = band_count(a->cfg.band_split);
    if (a->cfg.entropy == VC_ENT_RLGR) {
        vc_rlgr_decode(qsc, AVOX, p, at->len);
    } else if (a->cfg.entropy == VC_ENT_RANS_SHARED && nb > 1) {
        rans_dec_banded(qsc, AVOX, p, at->len, C->btable, band_map(a->cfg.band_split), nb);
    } else if (a->cfg.entropy == VC_ENT_RANS_SHARED) {
        rans_dec(qsc, AVOX, p, at->len, &C->table);
    } else {
        rans_table t1;
        for (u32 s=0;s<NSYM;++s) t1.freq[s] = (u16)(p[2*s] | ((u32)p[2*s+1]<<8));
        rt_finish(&t1);
        rans_dec(qsc, AVOX, p + NSYM*2, at->len - NSYM*2, &t1);
    }
}

// Curve-group full decode: walk each group, entropy-decode each atom against the
// group's shared table, reconstruct DC from the per-atom directory level, inverse
// quant + DCT, scatter. Atom lattice positions come from the global curve order.
static int curve_decode(const vc_bg_archive *a, u8 *vol) {
    u32 nat = a->Az * a->Ay * a->Ax;
    f32 base = a->cfg.step;
    int dcpred = a->cfg.dc_pred_curve;
    u32 *order_lat = (u32 *)malloc((size_t)nat * sizeof(u32));
    i16 *qsc = (i16 *)malloc(AVOX*sizeof(i16));
    i16 *coef = (i16 *)malloc(AVOX*sizeof(i16));
    u8  *avox = (u8 *)malloc(AVOX);
    if (!order_lat||!qsc||!coef||!avox) { free(order_lat);free(qsc);free(coef);free(avox); return 1; }
    build_curve_order(a, order_lat);
    for (u32 g = 0; g < a->n_chunks; ++g) {
        const bg_chunk *C = &a->chunks[g];
        u32 k0 = a->group_start[g], k1 = a->group_start[g+1];
        i16 prev_dc_lvl = 0;            // B1: running absolute DC level (resets/group)
        for (u32 k = k0; k < k1; ++k) {
            u32 gi = order_lat[k];
            u32 gz = gi / (a->Ay * a->Ax), gy = (gi / a->Ax) % a->Ay, gx = gi % a->Ax;
            bg_atom *at = (bg_atom *)&a->atoms[gi];
            // reconstruct absolute DC level from the stored (possibly residual) one
            i16 dc_lvl;
            if (dcpred && k > k0) dc_lvl = (i16)(prev_dc_lvl + at->dc_resid_q);
            else                  dc_lvl = at->dc_resid_q;
            prev_dc_lvl = dc_lvl;
            if (at->absent) { memset(avox, 0, AVOX); }
            else {
                decode_one_atom_levels(a, C, at, qsc);
                i16 dc_recon = dc_dequant(dc_lvl, base);
                at->dc_coef = dc_recon;
                dequant_atom(coef, qsc, dc_recon, base);
                bg_dct16_inv(avox, coef, 0);
            }
            scatter_atom(vol, avox, a->dz, a->dy, a->dx, gz, gy, gx);
        }
    }
    free(order_lat); free(qsc); free(coef); free(avox);
    return 0;
}

int vc_bg_decode(const vc_bg_archive *a, u8 *vol) {
    ensure_init();
    memset(vol, 0, (size_t)a->dz * a->dy * a->dx);
    if (a->cfg.group_mode == VC_GROUP_CURVE) return curve_decode(a, vol);
    i16 *qsc = (i16 *)malloc(AVOX*sizeof(i16));
    i16 *coef = (i16 *)malloc(AVOX*sizeof(i16));
    u8  *avox = (u8 *)malloc(AVOX);
    u32 ca = a->cfg.chunk_atoms; u32 namax = ca*ca*ca;
    u32 *order = (u32*)malloc(namax*sizeof(u32)), *rankc=(u32*)malloc(namax*sizeof(u32));
    u32 *rank_lat = (u32*)malloc((size_t)a->Az*a->Ay*a->Ax*sizeof(u32));
    nbr stoff[26]; u32 nst = stencil_offsets(a->cfg.stencil, stoff);
    f32 base = a->cfg.step;
    // rebuild lattice rank
    { u32 ordg[16*16*16], rkg[16*16*16]; build_order(a->cfg.traversal, ca, ordg, rkg);
      for (u32 cz=0;cz<a->ncz;++cz)for(u32 cy=0;cy<a->ncy;++cy)for(u32 cx=0;cx<a->ncx;++cx){
        u32 a0z=cz*ca,a0y=cy*ca,a0x=cx*ca;
        for(u32 lz=0;lz<ca;++lz)for(u32 ly=0;ly<ca;++ly)for(u32 lx=0;lx<ca;++lx){
          u32 gz=a0z+lz,gy=a0y+ly,gx=a0x+lx; if(gz>=a->Az||gy>=a->Ay||gx>=a->Ax)continue;
          rank_lat[lat_idx(a,gz,gy,gx)] = rkg[(lz*ca+ly)*ca+lx]; } } }

    u32 ci = 0;
    for (u32 cz = 0; cz < a->ncz; ++cz)
    for (u32 cy = 0; cy < a->ncy; ++cy)
    for (u32 cx = 0; cx < a->ncx; ++cx, ++ci) {
        const bg_chunk *C = &a->chunks[ci];
        build_order(a->cfg.traversal, ca, order, rankc);
        // EG2024 (4): uniform (skipped) chunk -> fill its voxel span with uval.
        if (C->uniform) {
            for (u32 lz=0; lz<C->caz; ++lz)
            for (u32 ly=0; ly<C->cay; ++ly)
            for (u32 lx=0; lx<C->cax; ++lx) {
                memset(avox, C->uval, AVOX);
                scatter_atom(vol, avox, a->dz, a->dy, a->dx,
                             C->a0z+lz, C->a0y+ly, C->a0x+lx);
            }
            continue;
        }
        for (u32 k = 0; k < namax; ++k) {
            u32 lin = order[k];
            u32 lz = lin/(ca*ca), ly=(lin/ca)%ca, lx=lin%ca;
            if (lz>=C->caz||ly>=C->cay||lx>=C->cax) continue;
            u32 gz=C->a0z+lz, gy=C->a0y+ly, gx=C->a0x+lx;
            u32 gi = lat_idx(a, gz, gy, gx);
            bg_atom *at = (bg_atom *)&a->atoms[gi];
            decode_one_atom_levels(a, C, at, qsc);
            // reconstruct DC: from the DC sub-volume (O(1) lookup) or pred+residual
            i16 dc_recon;
            if (a->cfg.dc_subvol) {
                dc_recon = at->absent ? 0 : a->dc_sub[gi];
            } else {
                int was_intra; u32 anc[26], nanc;
                i16 pred = predict_dc(a, gz, gy, gx, stoff, nst, rank_lat,
                                      rankc[lin], &was_intra, anc, &nanc);
                i16 dc_resid = at->absent ? 0 : at->dc_resid_q;
                i16 dc_q_pred = was_intra ? 0 : dc_quant((i32)pred, base);
                dc_recon = dc_dequant((i16)(dc_q_pred + dc_resid), base);
            }
            at->dc_coef = dc_recon;          // for downstream prediction
            if (at->absent) { memset(avox, 0, AVOX); }
            else {
                dequant_atom(coef, qsc, dc_recon, base);
                bg_dct16_inv(avox, coef, 0);
            }
            scatter_atom(vol, avox, a->dz, a->dy, a->dx, gz, gy, gx);
        }
    }
    free(qsc);free(coef);free(avox);free(order);free(rankc);free(rank_lat);
    return 0;
}

// ---------------------------------------------------------------------------
// Random access: decode ONE atom. Reconstructing its DC needs the reconstructed
// DC of its causal prediction ancestors (transitively). We MEMOIZE resolved DCs
// across a lattice-sized cache so the dependency DAG is walked in LINEAR time
// (without memoization a 6-conn chain is exponential). `touched` = number of
// atoms whose entropy payload had to be decoded = the true 16^3-decode cost.
// ---------------------------------------------------------------------------
typedef struct {
    i16 *dc;     // resolved DC coefficient per lattice atom
    u8  *state;  // 0 unknown, 1 in-progress (cycle guard), 2 resolved
    u64  touched;
} dc_cache;

// Resolve the reconstructed DC coefficient of an atom. KEY DESIGN POINT: the DC
// residual lives in the SEEK DIRECTORY (at->dc_resid_q), NOT in the entropy
// stream — so the prediction-ancestor walk reads only tiny directory fields and
// does NOT entropy-decode any ancestor's AC payload. This keeps cross-atom DC
// prediction compatible with cheap O(1) random access: resolving a target atom's
// DC pulls in its ancestor cone's *directory* entries (fast), and only the target
// atom's AC payload is actually entropy-decoded. `touched` counts AC decodes.
static i16 resolve_dc_memo(const vc_bg_archive *a, u32 z, u32 y, u32 x,
                           const nbr *st, u32 nst, const u32 *rank_lat,
                           i16 *qsc, dc_cache *mc) {
    (void)qsc;
    u32 gi = lat_idx(a, z, y, x);
    if (mc->state[gi]) return mc->dc[gi];          // resolved (cycle-free DAG)
    const bg_atom *at = &a->atoms[gi];
    if (at->absent) { mc->dc[gi] = 0; mc->state[gi] = 2; return 0; }
    // DC SUB-VOLUME: O(1) lookup, no ancestor cone walked. This is the whole
    // point — DC prediction is decoupled from atom decode order, so random access
    // touches ZERO prediction ancestors (vs the threaded stencil path below).
    if (a->cfg.dc_subvol) {
        i16 v = a->dc_sub[gi]; mc->dc[gi] = v; mc->state[gi] = 2; return v;
    }
    f32 base = a->cfg.step;
    int was_intra = (a->cfg.stencil == VC_STENCIL_NONE);
    i16 pred = 0;
    if (!was_intra) {
        u32 ca = a->cfg.chunk_atoms;
        u32 mycz=z/ca, mycy=y/ca, mycx=x/ca, mychunk = chunk_of(a,z,y,x);
        i32 acc=0; u32 cnt=0;
        for (u32 i=0;i<nst;++i) {
            int nz=(int)z+st[i].dz, ny=(int)y+st[i].dy, nx=(int)x+st[i].dx;
            if (nz<0||ny<0||nx<0) continue;
            if ((u32)nz>=a->Az||(u32)ny>=a->Ay||(u32)nx>=a->Ax) continue;
            int same=((u32)nz/ca==mycz&&(u32)ny/ca==mycy&&(u32)nx/ca==mycx);
            if (!same && a->cfg.edge==VC_EDGE_SELF) continue;
            u32 nchunk=chunk_of(a,(u32)nz,(u32)ny,(u32)nx);
            u32 ni=lat_idx(a,(u32)nz,(u32)ny,(u32)nx);
            int causal = nchunk<mychunk ? 1 : (nchunk>mychunk ? 0 : (rank_lat[ni]<rank_lat[gi]));
            if (!causal) continue;
            i16 ndc = resolve_dc_memo(a, (u32)nz,(u32)ny,(u32)nx, st,nst,rank_lat, qsc, mc);
            acc += ndc; ++cnt;
        }
        if (cnt==0) was_intra=1; else pred=(i16)((acc+(i32)cnt/2)/(i32)cnt);
    }
    i16 dc_resid = at->dc_resid_q;     // from the directory, no entropy decode
    i16 dc_q_pred = was_intra ? 0 : dc_quant((i32)pred, base);
    i16 dc_recon = dc_dequant((i16)(dc_q_pred + dc_resid), base);
    mc->dc[gi] = dc_recon; mc->state[gi] = 2;
    return dc_recon;
}

static u32 *build_rank_lat(const vc_bg_archive *a) {
    u32 ca = a->cfg.chunk_atoms;
    u32 *rl = (u32*)malloc((size_t)a->Az*a->Ay*a->Ax*sizeof(u32));
    u32 ordg[16*16*16], rkg[16*16*16];
    build_order(a->cfg.traversal, ca, ordg, rkg);
    for (u32 cz=0;cz<a->ncz;++cz)for(u32 cy=0;cy<a->ncy;++cy)for(u32 cx=0;cx<a->ncx;++cx){
        u32 a0z=cz*ca,a0y=cy*ca,a0x=cx*ca;
        for(u32 lz=0;lz<ca;++lz)for(u32 ly=0;ly<ca;++ly)for(u32 lx=0;lx<ca;++lx){
            u32 gz=a0z+lz,gy=a0y+ly,gx=a0x+lx; if(gz>=a->Az||gy>=a->Ay||gx>=a->Ax)continue;
            rl[lat_idx(a,gz,gy,gx)] = rkg[(lz*ca+ly)*ca+lx]; } }
    return rl;
}

// Curve-group single-atom random access. Realistic cost path: (1) compute the
// atom's GLOBAL CURVE INDEX (Morton/Hilbert) — the extra indexing cost vs box;
// (2) map curve index -> group id via the group-index table; (3) load that
// group's header (shared table, cached) + the atom's directory entry; (4)
// entropy-decode JUST that atom's payload; DC from the directory level. touched=1
// (no neighbor-atom decode). We use the precomputed group_of[] as the realistic
// O(1) group-index lookup and still PAY the curve_index() computation so the
// extraction-µs reflects curve indexing being pricier than a box division.
static int curve_decode_atom(const vc_bg_archive *a, u32 az, u32 ay, u32 ax,
                             u8 *atom_out, u32 *touched) {
    u32 gi = lat_idx(a, az, ay, ax);
    // (1) compute the curve index (the pricier-than-box indexing step). The result
    // is consumed by the group-index map below; computing it here makes the µs
    // measurement honest about curve-vs-box indexing cost.
    volatile u32 cidx = curve_index(a->cfg.traversal, a->curve_bits, az, ay, ax);
    (void)cidx;
    // (2) group lookup
    u32 g = a->group_of[gi];
    const bg_chunk *C = &a->chunks[g];
    const bg_atom *at = &a->atoms[gi];
    if (at->absent) { memset(atom_out, 0, AVOX); *touched = 1; return 0; }
    i16 *qsc = (i16*)malloc(AVOX*sizeof(i16));
    i16 *coef = (i16*)malloc(AVOX*sizeof(i16));
    if (!qsc||!coef) { free(qsc);free(coef); return 1; }
    decode_one_atom_levels(a, C, at, qsc);     // the ONE AC payload decode
    // DC level: B1 curve-predecessor prediction needs the running sum from the
    // group start back to this atom. That walk reads ONLY directory DC residuals
    // (no ancestor AC payload is entropy-decoded) so touched stays 1. Recover this
    // atom's curve position k by scanning the group's curve range for gi.
    i16 dc_lvl = at->dc_resid_q;
    if (a->cfg.dc_pred_curve) {
        u32 *order_lat = (u32*)malloc((size_t)a->Az*a->Ay*a->Ax*sizeof(u32));
        if (order_lat) {
            build_curve_order(a, order_lat);
            u32 k0 = a->group_start[g], k1 = a->group_start[g+1], kk = k0;
            for (u32 k = k0; k < k1; ++k) if (order_lat[k] == gi) { kk = k; break; }
            i32 acc = 0;           // k0 holds absolute, k>k0 hold deltas -> sum
            for (u32 k = k0; k <= kk; ++k) acc += a->atoms[order_lat[k]].dc_resid_q;
            dc_lvl = (i16)acc;
            free(order_lat);
        }
    }
    i16 dc_recon = dc_dequant(dc_lvl, a->cfg.step);
    dequant_atom(coef, qsc, dc_recon, a->cfg.step);
    bg_dct16_inv(atom_out, coef, 0);
    *touched = 1;
    free(qsc); free(coef);
    return 0;
}

int vc_bg_decode_atom(const vc_bg_archive *a, u32 az, u32 ay, u32 ax,
                      u8 *atom_out, u32 *touched) {
    ensure_init();
    if (a->cfg.group_mode == VC_GROUP_CURVE)
        return curve_decode_atom(a, az, ay, ax, atom_out, touched);
    size_t nat = (size_t)a->Az*a->Ay*a->Ax;
    i16 *qsc = (i16*)malloc(AVOX*sizeof(i16));
    i16 *coef = (i16*)malloc(AVOX*sizeof(i16));
    u32 *rank_lat = build_rank_lat(a);
    nbr stoff[26]; u32 nst = stencil_offsets(a->cfg.stencil, stoff);
    dc_cache mc = { (i16*)malloc(nat*sizeof(i16)), (u8*)calloc(nat,1), 0 };
    f32 base = a->cfg.step;

    u32 gi = lat_idx(a, az, ay, ax);
    const bg_atom *at = &a->atoms[gi];
    const bg_chunk *C = &a->chunks[chunk_of(a,az,ay,ax)];
    if (C->uniform) { memset(atom_out, C->uval, AVOX); *touched = 1;
        free(qsc);free(coef);free(rank_lat);free(mc.dc);free(mc.state); return 0; }
    if (at->absent) { memset(atom_out, 0, AVOX); *touched = 1;
        free(qsc);free(coef);free(rank_lat);free(mc.dc);free(mc.state); return 0; }

    i16 dc_recon = resolve_dc_memo(a, az, ay, ax, stoff, nst, rank_lat, qsc, &mc);
    decode_one_atom_levels(a, C, at, qsc);     // the ONE AC payload entropy-decode
    dequant_atom(coef, qsc, dc_recon, base);
    bg_dct16_inv(atom_out, coef, 0);
    // touched = AC payload decodes needed = 1 (DC ancestors are directory reads).
    *touched = 1;
    free(qsc); free(coef); free(rank_lat); free(mc.dc); free(mc.state);
    return 0;
}

int vc_bg_decode_region(const vc_bg_archive *a,
                        u32 z0, u32 y0, u32 x0, u32 z1, u32 y1, u32 x1,
                        u64 *total_decodes, u32 *atoms_in_box) {
    // Neighborhood-sweep trace WITH a shared dependency cache: an atom (and its
    // ancestors) decoded once stays cached for the rest of the box. The amortized
    // cost = unique atom-decodes / atoms_in_box.
    ensure_init();
    // Curve-group mode: no prediction ancestry -> exactly one AC decode per
    // non-absent atom in the box (amortized = 1.00).
    if (a->cfg.group_mode == VC_GROUP_CURVE) {
        u32 box = 0; u64 dec = 0;
        for (u32 z=z0; z<z1 && z<a->Az; ++z)
        for (u32 y=y0; y<y1 && y<a->Ay; ++y)
        for (u32 x=x0; x<x1 && x<a->Ax; ++x) {
            if (!a->atoms[lat_idx(a,z,y,x)].absent) ++dec;
            box++;
        }
        *total_decodes = dec; *atoms_in_box = box; return 0;
    }
    size_t nat = (size_t)a->Az*a->Ay*a->Ax;
    u32 *rank_lat = build_rank_lat(a);
    nbr stoff[26]; u32 nst = stencil_offsets(a->cfg.stencil, stoff);
    i16 *qsc = (i16*)malloc(AVOX*sizeof(i16));
    dc_cache mc = { (i16*)malloc(nat*sizeof(i16)), (u8*)calloc(nat,1), 0 };
    u32 box = 0; u64 ac_decodes = 0;
    for (u32 z=z0; z<z1 && z<a->Az; ++z)
    for (u32 y=y0; y<y1 && y<a->Ay; ++y)
    for (u32 x=x0; x<x1 && x<a->Ax; ++x) {
        // Resolve this atom's DC: for the stencil path this walks (and caches) its
        // causal ancestor DCs via the DIRECTORY only — no ancestor AC decode; for
        // the DC-sub-volume path it is a single O(1) lookup. Either way the only
        // AC entropy-decode the box requires is ONE per non-absent atom in it.
        resolve_dc_memo(a, z, y, x, stoff, nst, rank_lat, qsc, &mc);
        if (!a->atoms[lat_idx(a,z,y,x)].absent) ++ac_decodes;
        box++;
    }
    *total_decodes = ac_decodes;  // unique AC payload-decodes for the whole box
    *atoms_in_box = box;
    free(qsc); free(rank_lat); free(mc.dc); free(mc.state);
    return 0;
}

void vc_bg_free(vc_bg_archive *a) {
    if (!a) return;
    if (a->chunks) for (u32 i = 0; i < a->n_chunks; ++i) free(a->chunks[i].blob);
    free(a->chunks); free(a->atoms); free(a->dc_sub); free(a->group_of);
    free(a->group_start); free(a);
}
