// volume-compressor — frozen single-pipeline codec implementation.
// Build order per plan.txt §7. Step 1: minimal core (DCT-16 + dead-zone quant +
// context coder + chunk/archive, fixed q, single LOD). Round-trips lossy.
#include "vc.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef int8_t   i8;
typedef uint8_t  u8;
typedef int16_t  i16;
typedef uint16_t u16;
typedef int32_t  i32;
typedef uint32_t u32;
typedef int64_t  i64;
typedef uint64_t u64;

// ---------------------------------------------------------------------------
// Fail-fast panic
// ---------------------------------------------------------------------------
static void (*g_panic)(const char *) = NULL;
static void vc_panic(const char *msg) {
    if (g_panic) g_panic(msg);
    fprintf(stderr, "vc: panic: %s\n", msg);
    abort();
}
void vc_set_panic_hook(void (*hook)(const char *)) { g_panic = hook; }

#define VC_CHECK(cond, msg) do { if (!(cond)) vc_panic(msg); } while (0)

static void *vc_xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    VC_CHECK(p, "out of memory");
    return p;
}
static void *vc_xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    VC_CHECK(p, "out of memory");
    return p;
}

// ---------------------------------------------------------------------------
// Compile-time layout constants (plan.txt §1)
// ---------------------------------------------------------------------------
#define A      16u                 // atom edge
#define A3     (A*A*A)             // 4096
#define CHUNK_ATOMS 8u             // atoms per axis per chunk
#define CHUNK_SIDE  (CHUNK_ATOMS*A) // 128 voxels

#define Q14    14u                 // DCT fixed-point shift

// ---------------------------------------------------------------------------
// Integer DCT-16 (orthonormal scaled-cosine Q14 matrix; inverse = transpose).
// plan.txt §2.3. Straight-line trip-count-16 kernels for autovectorization.
// ---------------------------------------------------------------------------
static const i32 CMAT16[16][16] = {
  {  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096,  4096},
  {  5765,  5543,  5109,  4478,  3675,  2731,  1682,   568,  -568, -1682, -2731, -3675, -4478, -5109, -5543, -5765},
  {  5681,  4816,  3218,  1130, -1130, -3218, -4816, -5681, -5681, -4816, -3218, -1130,  1130,  3218,  4816,  5681},
  {  5543,  3675,   568, -2731, -5109, -5765, -4478, -1682,  1682,  4478,  5765,  5109,  2731,  -568, -3675, -5543},
  {  5352,  2217, -2217, -5352, -5352, -2217,  2217,  5352,  5352,  2217, -2217, -5352, -5352, -2217,  2217,  5352},
  {  5109,   568, -4478, -5543, -1682,  3675,  5765,  2731, -2731, -5765, -3675,  1682,  5543,  4478,  -568, -5109},
  {  4816, -1130, -5681, -3218,  3218,  5681,  1130, -4816, -4816,  1130,  5681,  3218, -3218, -5681, -1130,  4816},
  {  4478, -2731, -5543,   568,  5765,  1682, -5109, -3675,  3675,  5109, -1682, -5765,  -568,  5543,  2731, -4478},
  {  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096,  4096, -4096, -4096,  4096},
  {  3675, -5109, -1682,  5765,  -568, -5543,  2731,  4478, -4478, -2731,  5543,   568, -5765,  1682,  5109, -3675},
  {  3218, -5681,  1130,  4816, -4816, -1130,  5681, -3218, -3218,  5681, -1130, -4816,  4816,  1130, -5681,  3218},
  {  2731, -5765,  3675,  1682, -5543,  4478,   568, -5109,  5109,  -568, -4478,  5543, -1682, -3675,  5765, -2731},
  {  2217, -5352,  5352, -2217, -2217,  5352, -5352,  2217,  2217, -5352,  5352, -2217, -2217,  5352, -5352,  2217},
  {  1682, -4478,  5765, -5109,  2731,   568, -3675,  5543, -5543,  3675,  -568, -2731,  5109, -5765,  4478, -1682},
  {  1130, -3218,  4816, -5681,  5681, -4816,  3218, -1130, -1130,  3218, -4816,  5681, -5681,  4816, -3218,  1130},
  {   568, -1682,  2731, -3675,  4478, -5109,  5543, -5765,  5765, -5543,  5109, -4478,  3675, -2731,  1682,  -568},
};

static inline void dct16_fwd(const i32 *restrict in, i32 *restrict out) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    for (u32 k = 0; k < 16; ++k) {
        i32 acc = 0;
        for (u32 n = 0; n < 16; ++n) acc += CMAT16[k][n] * in[n];
        out[k] = (acc + rnd) >> Q14;
    }
}
static inline void dct16_inv(const i32 *restrict in, i32 *restrict out) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    for (u32 n = 0; n < 16; ++n) {
        i32 acc = 0;
        for (u32 k = 0; k < 16; ++k) acc += CMAT16[k][n] * in[k];
        out[n] = (acc + rnd) >> Q14;
    }
}

// Forward 3D DCT of one atom. vox = 4096 u8 (raster z,y,x). dc subtracted first.
// Output: 4096 i32 coefficients in raster (cz,cy,cx) order, scratch caller-owned.
static void atom_dct_fwd(const u8 *restrict vox, i32 dc, i32 *restrict coef) {
    static const u32 N = A;
    i32 *t = coef; // reuse coef as working buffer between passes
    // load - dc
    for (u32 i = 0; i < A3; ++i) t[i] = (i32)vox[i] - dc;
    i32 tmp[A];
    // Pass X (unit stride within a row of 16)
    for (u32 z = 0; z < N; ++z)
    for (u32 y = 0; y < N; ++y) {
        i32 *row = t + (z*N + y)*N;
        dct16_fwd(row, tmp);
        for (u32 x = 0; x < N; ++x) row[x] = tmp[x];
    }
    // Pass Y
    i32 col[A], outc[A];
    for (u32 z = 0; z < N; ++z)
    for (u32 x = 0; x < N; ++x) {
        for (u32 y = 0; y < N; ++y) col[y] = t[(z*N + y)*N + x];
        dct16_fwd(col, outc);
        for (u32 y = 0; y < N; ++y) t[(z*N + y)*N + x] = outc[y];
    }
    // Pass Z
    for (u32 y = 0; y < N; ++y)
    for (u32 x = 0; x < N; ++x) {
        for (u32 z = 0; z < N; ++z) col[z] = t[(z*N + y)*N + x];
        dct16_fwd(col, outc);
        for (u32 z = 0; z < N; ++z) t[(z*N + y)*N + x] = outc[z];
    }
}

// Inverse 3D DCT. coef = 4096 i32, dc added back, clamp to u8 into vox.
static void atom_dct_inv(const i32 *restrict coef, i32 dc, u8 *restrict vox) {
    static const u32 N = A;
    i32 *t = (i32 *)vc_xmalloc(A3 * sizeof(i32));
    memcpy(t, coef, A3 * sizeof(i32));
    i32 tmp[A], col[A], outc[A];
    // Inverse along Z, Y, X (order does not matter for separable)
    for (u32 y = 0; y < N; ++y)
    for (u32 x = 0; x < N; ++x) {
        for (u32 z = 0; z < N; ++z) col[z] = t[(z*N + y)*N + x];
        dct16_inv(col, outc);
        for (u32 z = 0; z < N; ++z) t[(z*N + y)*N + x] = outc[z];
    }
    for (u32 z = 0; z < N; ++z)
    for (u32 x = 0; x < N; ++x) {
        for (u32 y = 0; y < N; ++y) col[y] = t[(z*N + y)*N + x];
        dct16_inv(col, outc);
        for (u32 y = 0; y < N; ++y) t[(z*N + y)*N + x] = outc[y];
    }
    for (u32 z = 0; z < N; ++z)
    for (u32 y = 0; y < N; ++y) {
        i32 *row = t + (z*N + y)*N;
        dct16_inv(row, tmp);
        for (u32 x = 0; x < N; ++x) {
            i32 v = tmp[x] + dc;
            if (v < 0) v = 0; else if (v > 255) v = 255;
            row[x] = v;
        }
    }
    for (u32 i = 0; i < A3; ++i) vox[i] = (u8)t[i];
    free(t);
}

// ---------------------------------------------------------------------------
// Quantization: dead-zone scalar quantizer + 1-parameter HF-boost curve.
// plan.txt §2.4. step(freq) = base_q * hf_weight(freq), hf_weight protects HF.
// ---------------------------------------------------------------------------
#define DZ_FRAC   0.80f   // dead-zone width fraction of step
#define DQ_OFFSET 0.40f   // dequant reconstruction sub-center offset
#define HF_EXP    0.65f   // single tuned exponent; <1 => HF kept relatively finer

// Per-coefficient quant step, precomputed for the 4096 raster positions.
// freq = cz+cy+cx (L1). We want HF to keep RELATIVELY finer steps than flat;
// i.e. step grows slower than freq. We normalize so mid frequencies ~= base_q.
static float g_step[A3];
static u16   g_scan[A3];      // scan order: index = scan position -> raster idx
static u16   g_scan_inv[A3];  // raster idx -> scan position
static int   g_tables_ready = 0;

static int scan_cmp(const void *pa, const void *pb) {
    u32 a = *(const u32 *)pa, b = *(const u32 *)pb;
    u32 fa = (a / (A*A)) + ((a / A) % A) + (a % A);
    u32 fb = (b / (A*A)) + ((b / A) % A) + (b % A);
    if (fa != fb) return (int)fa - (int)fb;
    return (int)a - (int)b;
}

static void build_tables(void) {
    if (g_tables_ready) return;
    // HF weight curve: weight(freq) = (1+freq)^HF_EXP normalized so freq=0 -> ~1.
    // The DC (freq 0) gets the base step; higher freq get a larger step but the
    // growth is sub-linear so HF is protected relative to a freq-proportional
    // matrix. Reference flat = 1 everywhere.
    for (u32 cz = 0; cz < A; ++cz)
    for (u32 cy = 0; cy < A; ++cy)
    for (u32 cx = 0; cx < A; ++cx) {
        u32 idx = (cz*A + cy)*A + cx;
        u32 freq = cz + cy + cx;
        g_step[idx] = powf(1.0f + (float)freq, HF_EXP);
    }
    // Build scan permutation by ascending L1 frequency.
    u32 *order = (u32 *)vc_xmalloc(A3 * sizeof(u32));
    for (u32 i = 0; i < A3; ++i) order[i] = i;
    qsort(order, A3, sizeof(u32), scan_cmp);
    for (u32 p = 0; p < A3; ++p) {
        g_scan[p] = (u16)order[p];
        g_scan_inv[order[p]] = (u16)p;
    }
    free(order);
    g_tables_ready = 1;
}

// Quantize coef (raster) -> q (raster). Returns nothing.
static void quantize(const i32 *restrict coef, float base_q, i16 *restrict q) {
    for (u32 i = 0; i < A3; ++i) {
        float step = base_q * g_step[i];
        float dz = DZ_FRAC * step;
        float a = (float)(coef[i] < 0 ? -coef[i] : coef[i]);
        i32 level = 0;
        if (a >= dz) level = (i32)((a - dz) / step + 1.0f);
        q[i] = (i16)(coef[i] < 0 ? -level : level);
    }
}

static void dequantize(const i16 *restrict q, float base_q, i32 *restrict coef) {
    for (u32 i = 0; i < A3; ++i) {
        i32 lv = q[i];
        if (lv == 0) { coef[i] = 0; continue; }
        float step = base_q * g_step[i];
        float dz = DZ_FRAC * step;
        float mag = (float)(lv < 0 ? -lv : lv);
        float r = (mag - 1.0f + DQ_OFFSET) * step + dz;
        coef[i] = (i32)(lv < 0 ? -r : r);
    }
}

// ---------------------------------------------------------------------------
// Binary range coder (CABAC-style) — adaptive, table-free, reset per atom.
// plan.txt §2.6. Used by the coefficient context coder.
// ---------------------------------------------------------------------------
typedef struct {
    u8 *buf; size_t cap, len;
    u64 low; u32 range; u8 cache; u64 cache_size;
} rc_enc;

typedef struct {
    const u8 *buf; size_t len, pos;
    u32 code; u32 range;
} rc_dec;

// Probability state: 12-bit prob of bit==0, adapted by shift.
typedef struct { u16 p0; } ctx_t;
static inline void ctx_init(ctx_t *c) { c->p0 = 1u << 11; } // 0.5 in Q12

#define RC_TOP (1u << 24)

static void enc_init(rc_enc *e, u8 *buf, size_t cap) {
    e->buf = buf; e->cap = cap; e->len = 0;
    e->low = 0; e->range = 0xFFFFFFFFu; e->cache = 0; e->cache_size = 1;
}
static void enc_putbyte(rc_enc *e, u8 b) {
    VC_CHECK(e->len < e->cap, "range encoder overflow");
    e->buf[e->len++] = b;
}
// LZMA-style shift-low: emits one byte, handles carry propagation.
static void enc_shift_low(rc_enc *e) {
    if ((u32)(e->low >> 32) != 0 || e->low < 0xFF000000ull) {
        u8 carry = (u8)(e->low >> 32);
        do {
            enc_putbyte(e, (u8)(e->cache + carry));
            e->cache = 0xFF;
        } while (--e->cache_size);
        e->cache = (u8)(e->low >> 24);
    }
    e->cache_size++;
    e->low = (e->low << 8) & 0xFFFFFFFFull;
}
static void enc_bit(rc_enc *e, ctx_t *c, int bit) {
    // split range by prob of 0
    u32 r0 = (u32)(((u64)e->range * c->p0) >> 12);
    if (bit == 0) {
        e->range = r0;
        c->p0 = (u16)(c->p0 + ((4096 - c->p0) >> 5));
    } else {
        e->low += r0;
        e->range -= r0;
        c->p0 = (u16)(c->p0 - (c->p0 >> 5));
    }
    while (e->range < RC_TOP) { enc_shift_low(e); e->range <<= 8; }
}
static void enc_bypass(rc_enc *e, int bit) {
    e->range >>= 1;
    if (bit) e->low += e->range;
    while (e->range < RC_TOP) { enc_shift_low(e); e->range <<= 8; }
}
static void enc_flush(rc_enc *e) {
    for (int i = 0; i < 5; ++i) enc_shift_low(e);
}

static void dec_init(rc_dec *d, const u8 *buf, size_t len) {
    d->buf = buf; d->len = len; d->pos = 0;
    d->code = 0; d->range = 0xFFFFFFFFu;
    for (int i = 0; i < 5; ++i) {
        u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0;
        d->code = (d->code << 8) | b;
    }
    // first byte produced by encoder is the initial cache (started flag), our
    // encoder emits 5 flush bytes; decoder primes with 5 reads but encoder's
    // first emitted byte corresponds to cache. We align below.
}
static int dec_bit(rc_dec *d, ctx_t *c) {
    u32 r0 = (u32)(((u64)d->range * c->p0) >> 12);
    int bit;
    if (d->code < r0) {
        d->range = r0; bit = 0;
        c->p0 = (u16)(c->p0 + ((4096 - c->p0) >> 5));
    } else {
        d->code -= r0; d->range -= r0; bit = 1;
        c->p0 = (u16)(c->p0 - (c->p0 >> 5));
    }
    while (d->range < RC_TOP) {
        u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0;
        d->code = (d->code << 8) | b;
        d->range <<= 8;
    }
    return bit;
}
static int dec_bypass(rc_dec *d) {
    d->range >>= 1;
    int bit = (d->code >= d->range);
    if (bit) d->code -= d->range;
    while (d->range < RC_TOP) {
        u8 b = (d->pos < d->len) ? d->buf[d->pos++] : 0;
        d->code = (d->code << 8) | b;
        d->range <<= 8;
    }
    return bit;
}

// ---------------------------------------------------------------------------
// Coefficient context coder. Per-atom: code each scanned coefficient up to EOB
// as significance + magnitude (exp-golomb-ish unary prefix in contexts) + sign.
// Contexts reset per atom. Step 1 codes ALL A3 coefficients (no EOB yet).
// ---------------------------------------------------------------------------
#define NB_BANDS 8          // coarse frequency-band contexts
#define MAGCTX   12         // magnitude continuation contexts

typedef struct {
    ctx_t sig[NB_BANDS];           // significance per band
    ctx_t mag[MAGCTX];             // unary magnitude continuation
    // sign + magnitude LSBs are bypass
} atom_ctx;

static void atom_ctx_init(atom_ctx *a) {
    for (int i = 0; i < NB_BANDS; ++i) ctx_init(&a->sig[i]);
    for (int i = 0; i < MAGCTX; ++i) ctx_init(&a->mag[i]);
}

static inline int band_of(u32 raster_idx) {
    u32 cz = raster_idx / (A*A), cy = (raster_idx / A) % A, cx = raster_idx % A;
    u32 freq = cz + cy + cx; // 0..45
    int b = (int)(freq * NB_BANDS / 46u);
    if (b >= NB_BANDS) b = NB_BANDS - 1;
    return b;
}

// Encode magnitude m>=1 with adaptive unary then exp-golomb bypass tail.
static void enc_magnitude(rc_enc *e, atom_ctx *ac, u32 m) {
    // m>=1. Code (m-1) as: unary prefix in mag contexts up to a cap, then EG.
    u32 v = m - 1;
    u32 k = 0;
    while (k < (u32)(MAGCTX - 1) && v > 0) {
        enc_bit(e, &ac->mag[k], 1); // continue
        v -= 1; k++;
        if (v == 0) { enc_bit(e, &ac->mag[k], 0); return; }
    }
    if (v == 0) { enc_bit(e, &ac->mag[k], 0); return; }
    // remaining v coded as exp-golomb order 0 in bypass
    enc_bit(e, &ac->mag[k], 1);
    u32 x = v; // >=1 here? v could be >0
    // EG0 of x
    u32 nbits = 0; u32 t = x + 1;
    while (t > 1) { t >>= 1; nbits++; }
    for (u32 i = 0; i < nbits; ++i) enc_bypass(e, 1);
    enc_bypass(e, 0);
    for (i32 i = (i32)nbits - 1; i >= 0; --i) enc_bypass(e, ((x + 1) >> i) & 1);
}
static u32 dec_magnitude(rc_dec *d, atom_ctx *ac) {
    u32 v = 0; u32 k = 0;
    while (k < (u32)(MAGCTX - 1)) {
        if (dec_bit(d, &ac->mag[k])) { v += 1; k++; }
        else return v + 1;
    }
    // at last context: a 1 means EG tail follows, 0 means stop
    if (!dec_bit(d, &ac->mag[k])) return v + 1;
    // EG0 decode
    u32 nbits = 0;
    while (dec_bypass(d)) nbits++;
    u32 x = 1;
    for (u32 i = 0; i < nbits; ++i) x = (x << 1) | (u32)dec_bypass(d);
    x -= 1;
    return v + x + 1;
}

// EOB = scan position of last significant coef + 1 (0 means empty atom).
// Coded as a 12-bit fixed-length value in bypass (A3=4096 fits in 12 bits).
// plan.txt §2.5: decode stops at EOB; trailing zeros cost nothing.
static void enc_eob(rc_enc *e, u32 eob) {
    for (int i = 11; i >= 0; --i) enc_bypass(e, (eob >> i) & 1);
}
static u32 dec_eob(rc_dec *d) {
    u32 v = 0;
    for (int i = 0; i < 12; ++i) v = (v << 1) | (u32)dec_bypass(d);
    return v;
}

// Encode one atom's quantized coefficients (raster order q[A3]) into encoder.
static void enc_atom_coefs(rc_enc *e, const i16 *q) {
    atom_ctx ac; atom_ctx_init(&ac);
    // find EOB: last scan position with nonzero coef
    u32 eob = 0;
    for (u32 p = A3; p-- > 0;) { if (q[g_scan[p]] != 0) { eob = p + 1; break; } }
    enc_eob(e, eob);
    for (u32 p = 0; p < eob; ++p) {
        u32 idx = g_scan[p];
        int b = band_of(idx);
        i16 v = q[idx];
        if (v == 0) { enc_bit(e, &ac.sig[b], 0); continue; }
        enc_bit(e, &ac.sig[b], 1);
        u32 m = (u32)(v < 0 ? -v : v);
        enc_magnitude(e, &ac, m);
        enc_bypass(e, v < 0 ? 1 : 0);
    }
}
static void dec_atom_coefs(rc_dec *d, i16 *q) {
    atom_ctx ac; atom_ctx_init(&ac);
    memset(q, 0, A3 * sizeof(i16));
    u32 eob = dec_eob(d);
    for (u32 p = 0; p < eob; ++p) {
        u32 idx = g_scan[p];
        int b = band_of(idx);
        if (!dec_bit(d, &ac.sig[b])) continue;
        u32 m = dec_magnitude(d, &ac);
        int neg = dec_bypass(d);
        q[idx] = (i16)(neg ? -(i32)m : (i32)m);
    }
}

// ===========================================================================
// Archive format (plan.txt §4). Step 1: single LOD member, raster atoms.
// ===========================================================================
#define VC_MAGIC   0x00314356u // "VC1\0"
#define VC_VERSION 1u
#define ABSENT     0xFFFFFFFFFFFFFFFFull

// Per-atom directory entry stored in chunk header.
typedef struct {
    u64 offset;   // byte offset of atom payload within archive (ABSENT sentinel)
    u32 length;   // payload length in bytes
    u8  flags;    // bit0 uniform, bit1 absent
    u8  uval;     // uniform value
    u8  dc;       // stored DC mean (rounded)
} atom_dir;
#define AF_UNIFORM 1u
#define AF_ABSENT  2u

// In step 1 q is fixed and global. Stored in member header.

// ---- growable byte buffer ----
typedef struct { u8 *p; size_t len, cap; } bbuf;
static void bb_reserve(bbuf *b, size_t need) {
    if (b->len + need <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 4096;
    while (nc < b->len + need) nc *= 2;
    b->p = (u8 *)realloc(b->p, nc);
    VC_CHECK(b->p, "out of memory");
    b->cap = nc;
}
static void bb_put(bbuf *b, const void *src, size_t n) {
    bb_reserve(b, n); memcpy(b->p + b->len, src, n); b->len += n;
}
static void bb_u8(bbuf *b, u8 v)  { bb_reserve(b, 1); b->p[b->len++] = v; }
static void bb_u32(bbuf *b, u32 v){ bb_put(b, &v, 4); }
static void bb_u64(bbuf *b, u64 v){ bb_put(b, &v, 8); }
static void bb_align(bbuf *b, size_t a) {
    while (b->len % a) bb_u8(b, 0);
}

// ---- little reader helpers over a borrowed buffer ----
static u32 rd_u32(const u8 *p) { u32 v; memcpy(&v, p, 4); return v; }
static u64 rd_u64(const u8 *p) { u64 v; memcpy(&v, p, 8); return v; }

// Per-atom payload: [dc u8][rc bytes...]. Uniform atoms carry no payload.
// Encode one atom from a 16^3 voxel block. Returns 0 if uniform (sets *uval),
// else appends rc-coded coefs to out and returns payload length (incl dc byte).
static u32 encode_atom(const u8 *vox, float base_q, bbuf *out,
                       int *is_uniform, u8 *uval, u8 *dcv) {
    // uniform fast-path
    u8 first = vox[0];
    int uni = 1;
    for (u32 i = 1; i < A3; ++i) if (vox[i] != first) { uni = 0; break; }
    if (uni) { *is_uniform = 1; *uval = first; *dcv = first; return 0; }
    *is_uniform = 0;

    // DC mean
    u32 sum = 0;
    for (u32 i = 0; i < A3; ++i) sum += vox[i];
    i32 dc = (i32)((sum + A3/2) / A3);
    *dcv = (u8)dc;

    i32 *coef = (i32 *)vc_xmalloc(A3 * sizeof(i32));
    atom_dct_fwd(vox, dc, coef);
    i16 q[A3];
    quantize(coef, base_q, q);
    free(coef);

    size_t start = out->len;
    bb_u8(out, (u8)dc);
    // rc encode coefs into a scratch buffer then append
    size_t cap = A3 * 4 + 64;
    u8 *scratch = (u8 *)vc_xmalloc(cap);
    rc_enc e; enc_init(&e, scratch, cap);
    enc_atom_coefs(&e, q);
    enc_flush(&e);
    bb_put(out, scratch, e.len);
    free(scratch);
    return (u32)(out->len - start);
}

// Decode one atom payload (dc byte + rc bytes) into 16^3 voxels.
static void decode_atom_payload(const u8 *pay, u32 len, float base_q, u8 *vox) {
    i32 dc = pay[0];
    rc_dec d; dec_init(&d, pay + 1, len - 1);
    i16 q[A3];
    dec_atom_coefs(&d, q);
    i32 *coef = (i32 *)vc_xmalloc(A3 * sizeof(i32));
    dequantize(q, base_q, coef);
    atom_dct_inv(coef, dc, vox);
    free(coef);
}

// ---------------------------------------------------------------------------
// Member encode. A member = one LOD volume. Layout written to bbuf `out`:
//   member header (returned via member_rec), chunks each:
//     [chunk header: per-atom dir] [atom payloads, 64B-aligned]
// The member is self-contained; offsets in the atom dir are relative to the
// member start. We collect chunk metadata, then write index + payloads.
// ---------------------------------------------------------------------------
typedef struct {
    u32 nx, ny, nz;          // logical dims
    u32 acx, acy, acz;       // atom counts per axis (ceil(dim/16))
    u32 ccx, ccy, ccz;       // chunk counts per axis
    float base_q;            // step-1 single q for whole member (per-chunk later)
    u64  pay_base;           // member-relative offset of payload region
    u64  rel_offset;         // member payload offset within archive
    u64  length;             // member length
} member_rec;

// Downsample a volume 2x per axis (box filter). Returns malloc'd buffer + dims.
static u8 *downsample2x(const u8 *vol, vc_dims in, vc_dims *out) {
    u32 ox = (in.nx + 1) / 2, oy = (in.ny + 1) / 2, oz = (in.nz + 1) / 2;
    if (ox < 1) ox = 1;
    if (oy < 1) oy = 1;
    if (oz < 1) oz = 1;
    u8 *o = (u8 *)vc_xmalloc((size_t)ox * oy * oz);
    for (u32 z = 0; z < oz; ++z)
    for (u32 y = 0; y < oy; ++y)
    for (u32 x = 0; x < ox; ++x) {
        u32 x0 = x*2, y0 = y*2, z0 = z*2;
        u32 acc = 0, n = 0;
        for (u32 dz = 0; dz < 2; ++dz) { u32 zz = z0+dz; if (zz>=in.nz) continue;
        for (u32 dy = 0; dy < 2; ++dy) { u32 yy = y0+dy; if (yy>=in.ny) continue;
        for (u32 dx = 0; dx < 2; ++dx) { u32 xx = x0+dx; if (xx>=in.nx) continue;
            acc += vol[((size_t)zz*in.ny + yy)*in.nx + xx]; n++;
        }}}
        o[((size_t)z*oy + y)*ox + x] = (u8)((acc + n/2) / (n ? n : 1));
    }
    out->nx = ox; out->ny = oy; out->nz = oz;
    return o;
}

// Gather a 16^3 atom from a volume at atom coords, padding edges with fill.
static void gather_atom(const u8 *vol, vc_dims d, u32 ax, u32 ay, u32 az,
                        u8 fill, u8 *atom) {
    for (u32 z = 0; z < A; ++z)
    for (u32 y = 0; y < A; ++y)
    for (u32 x = 0; x < A; ++x) {
        u32 vx = ax*A + x, vy = ay*A + y, vz = az*A + z;
        u8 v = fill;
        if (vx < d.nx && vy < d.ny && vz < d.nz)
            v = vol[((size_t)vz*d.ny + vy)*d.nx + vx];
        atom[(z*A + y)*A + x] = v;
    }
}

// Encode a full member (one volume) into `out`, given base_q. Fills *rec.
// Serialized member layout (all offsets relative to member start):
//   [u32 nx][u32 ny][u32 nz][u32 acx][u32 acy][u32 acz]
//   [u32 ccx][u32 ccy][u32 ccz][f32 base_q]
//   per chunk (raster ccz,ccy,ccx):
//     [u32 n_atoms]
//     per atom (raster within chunk): [u64 off][u32 len][u8 flags][u8 uval][u8 dc][u8 pad]
//   [payload region, 64B-aligned atoms]
static void encode_member(const u8 *vol, vc_dims d, float base_q, bbuf *arc,
                          member_rec *rec) {
    u32 acx = (d.nx + A - 1) / A, acy = (d.ny + A - 1) / A, acz = (d.nz + A - 1) / A;
    if (acx == 0) acx = 1;
    if (acy == 0) acy = 1;
    if (acz == 0) acz = 1;
    u32 ccx = (acx + CHUNK_ATOMS - 1) / CHUNK_ATOMS;
    u32 ccy = (acy + CHUNK_ATOMS - 1) / CHUNK_ATOMS;
    u32 ccz = (acz + CHUNK_ATOMS - 1) / CHUNK_ATOMS;

    u64 member_start = arc->len;
    rec->nx = d.nx; rec->ny = d.ny; rec->nz = d.nz;
    rec->acx = acx; rec->acy = acy; rec->acz = acz;
    rec->ccx = ccx; rec->ccy = ccy; rec->ccz = ccz;
    rec->base_q = base_q;

    // Build payloads + directory in two passes. First encode all atoms into a
    // temp payload buffer recording dir entries, then lay out header+payloads.
    bbuf pay = {0};
    // per-chunk arrays of atom dir entries
    u32 nchunks = ccx * ccy * ccz;
    atom_dir **dirs = (atom_dir **)vc_xcalloc(nchunks, sizeof(atom_dir *));
    u32 *chunk_natoms = (u32 *)vc_xcalloc(nchunks, sizeof(u32));
    u8  *chunk_absent = (u8 *)vc_xcalloc(nchunks, sizeof(u8));

    u8 atom[A3];
    for (u32 cz = 0; cz < ccz; ++cz)
    for (u32 cy = 0; cy < ccy; ++cy)
    for (u32 cx = 0; cx < ccx; ++cx) {
        u32 ci = (cz*ccy + cy)*ccx + cx;
        u32 ax0 = cx*CHUNK_ATOMS, ay0 = cy*CHUNK_ATOMS, az0 = cz*CHUNK_ATOMS;
        u32 axn = acx - ax0; if (axn > CHUNK_ATOMS) axn = CHUNK_ATOMS;
        u32 ayn = acy - ay0; if (ayn > CHUNK_ATOMS) ayn = CHUNK_ATOMS;
        u32 azn = acz - az0; if (azn > CHUNK_ATOMS) azn = CHUNK_ATOMS;
        u32 na = axn * ayn * azn;
        chunk_natoms[ci] = na;
        atom_dir *dir = (atom_dir *)vc_xcalloc(na, sizeof(atom_dir));
        dirs[ci] = dir;
        u32 ai = 0;
        for (u32 lz = 0; lz < azn; ++lz)
        for (u32 ly = 0; ly < ayn; ++ly)
        for (u32 lx = 0; lx < axn; ++lx, ++ai) {
            gather_atom(vol, d, ax0+lx, ay0+ly, az0+lz, 0, atom);
            int uni = 0; u8 uval = 0, dcv = 0;
            // Atom payloads are packed tightly (entropy decode is byte-serial;
            // 64B per-atom padding would cap ratio near 64x and buys nothing).
            u32 len = encode_atom(atom, base_q, &pay, &uni, &uval, &dcv);
            atom_dir *e = &dir[ai];
            e->flags = uni ? AF_UNIFORM : 0;
            e->uval = uval; e->dc = dcv;
            e->length = len; // offset is implicit (cumulative length within member)
        }
        // ABSENT-chunk detection (plan §4): entire chunk uniform with fill (0).
        int all_absent = 1;
        for (u32 k = 0; k < na; ++k)
            if (!((dir[k].flags & AF_UNIFORM) && dir[k].uval == 0)) { all_absent = 0; break; }
        chunk_absent[ci] = (u8)all_absent;
    }

    // Write member header
    bbuf hdr = {0};
    bb_u32(&hdr, d.nx); bb_u32(&hdr, d.ny); bb_u32(&hdr, d.nz);
    bb_u32(&hdr, acx); bb_u32(&hdr, acy); bb_u32(&hdr, acz);
    bb_u32(&hdr, ccx); bb_u32(&hdr, ccy); bb_u32(&hdr, ccz);
    bb_put(&hdr, &base_q, 4);
    u64 pay_base_slot = hdr.len; // patched after we know header size
    bb_u64(&hdr, 0);             // [u64 pay_base] member-relative payload start
    // chunk directories. Sentinel n_atoms = 0xFFFFFFFF marks an ABSENT chunk
    // (entirely fill/zero) with NO atom entries and zero payload (plan §4).
    // Each atom entry is 8 bytes: [u32 len][u8 flags][u8 uval][u8 dc][u8 pad].
    // Atom byte offsets are IMPLICIT (cumulative payload length within member).
    for (u32 ci = 0; ci < nchunks; ++ci) {
        if (chunk_absent[ci]) { bb_u32(&hdr, 0xFFFFFFFFu); continue; }
        bb_u32(&hdr, chunk_natoms[ci]);
        for (u32 ai = 0; ai < chunk_natoms[ci]; ++ai) {
            atom_dir *e = &dirs[ci][ai];
            bb_u32(&hdr, e->length);
            bb_u8(&hdr, e->flags);
            bb_u8(&hdr, e->uval);
            bb_u8(&hdr, e->dc);
            bb_u8(&hdr, 0);
        }
    }
    // Chunk directories / payload region aligned to 64B for mmap-friendliness.
    bb_align(&hdr, 64);
    { u64 pb = hdr.len; memcpy(hdr.p + pay_base_slot, &pb, 8); }

    // Append hdr then payloads to archive
    bb_put(arc, hdr.p, hdr.len);
    bb_put(arc, pay.p, pay.len);

    rec->rel_offset = member_start;
    rec->length = arc->len - member_start;

    for (u32 ci = 0; ci < nchunks; ++ci) free(dirs[ci]);
    free(dirs); free(chunk_natoms); free(chunk_absent);
    free(hdr.p); free(pay.p);
}

// Measure compressed size of a volume at a given base_q (for rate control).
static u64 measure_member_size(const u8 *vol, vc_dims d, float base_q) {
    bbuf tmp = {0}; member_rec rec;
    encode_member(vol, d, base_q, &tmp, &rec);
    u64 sz = tmp.len; free(tmp.p);
    return sz;
}

// Pick base_q to hit a target ratio for one member via encode-measure-adjust.
// plan.txt §2.7: trivial q<->ratio search, single q per member/chunk. (Step 4
// applies this per member; chunk granularity falls out since each LOD member
// has its own q.)
static float pick_q_for_ratio(const u8 *vol, vc_dims d, float target_ratio) {
    u64 raw = (u64)d.nx * d.ny * d.nz;
    if (raw == 0) return 1.0f;
    u64 target_bytes = (u64)((double)raw / target_ratio);
    float lo = 0.10f, hi = 4096.0f;
    // bisection on log(q): bigger q -> smaller size (monotone).
    float best = 8.0f; u64 best_sz = measure_member_size(vol, d, best);
    for (int it = 0; it < 16; ++it) {
        float mid = sqrtf(lo * hi);
        u64 sz = measure_member_size(vol, d, mid);
        if (sz > target_bytes) lo = mid; else hi = mid;
        // track closest to target
        u64 da = sz > target_bytes ? sz - target_bytes : target_bytes - sz;
        u64 db = best_sz > target_bytes ? best_sz - target_bytes : target_bytes - best_sz;
        if (da < db) { best = mid; best_sz = sz; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// vc_encode — top level. Builds the 8-LOD pyramid (step 6) and footer.
// ---------------------------------------------------------------------------
vc_status vc_encode(const u8 *vol, vc_dims dims, float target_ratio,
                    u8 **out_archive, size_t *out_len) {
    build_tables();
    VC_CHECK(vol && out_archive && out_len, "vc_encode null arg");
    VC_CHECK(target_ratio >= 1.0f, "target_ratio < 1");

    bbuf arc = {0};
    // file header
    bb_u32(&arc, VC_MAGIC);
    bb_u32(&arc, VC_VERSION);

    member_rec recs[VC_NLOD];
    int nmembers = 0;

    // LOD0 = original; LOD k+1 = 2x downsample of level above (plan.txt §5).
    u8 *cur = (u8 *)vol; vc_dims cd = dims;
    u8 *owned = NULL; // buffers we must free (not the caller's vol)
    for (int lod = 0; lod < VC_NLOD; ++lod) {
        float q = pick_q_for_ratio(cur, cd, target_ratio);
        encode_member(cur, cd, q, &arc, &recs[nmembers]);
        nmembers++;
        // stop if next level would be degenerate (all dims already 1)
        if (cd.nx <= 1 && cd.ny <= 1 && cd.nz <= 1) break;
        vc_dims nd; u8 *nv = downsample2x(cur, cd, &nd);
        if (owned) free(owned);
        owned = nv; cur = nv; cd = nd;
    }
    if (owned) free(owned);

    // footer: member directory
    u64 dir_off = arc.len;
    bb_u32(&arc, (u32)nmembers);
    for (int i = 0; i < nmembers; ++i) {
        bb_u64(&arc, recs[i].rel_offset);
        bb_u64(&arc, recs[i].length);
    }
    u64 dir_len = arc.len - dir_off;
    bb_align(&arc, 8);
    // trailer
    bb_u64(&arc, dir_off);
    bb_u64(&arc, dir_len);
    bb_u32(&arc, VC_VERSION);
    bb_u32(&arc, VC_MAGIC);

    *out_archive = arc.p;
    *out_len = arc.len;
    return VC_OK;
}

// ===========================================================================
// Decode side
// ===========================================================================
struct vc_archive {
    const u8 *buf; size_t len;
    u32 nmembers;
    const u8 *members[VC_NLOD]; // pointer to each member start
    member_rec recs[VC_NLOD];
};

// Parse a member header into rec (dims/grid/base_q); returns ptr past header
// data is read lazily; we only cache the rec fields.
static void parse_member_header(const u8 *m, member_rec *r) {
    r->nx = rd_u32(m+0);  r->ny = rd_u32(m+4);  r->nz = rd_u32(m+8);
    r->acx = rd_u32(m+12); r->acy = rd_u32(m+16); r->acz = rd_u32(m+20);
    r->ccx = rd_u32(m+24); r->ccy = rd_u32(m+28); r->ccz = rd_u32(m+32);
    memcpy(&r->base_q, m+36, 4);
    r->pay_base = rd_u64(m+40);
}

vc_archive *vc_open(const u8 *archive, size_t len) {
    build_tables();
    if (!archive || len < 40) return NULL;
    // verify file header
    if (rd_u32(archive) != VC_MAGIC) return NULL;
    // trailer: last 24 bytes = [u64 dir_off][u64 dir_len][u32 ver][u32 magic]
    const u8 *tr = archive + len - 24;
    u64 dir_off = rd_u64(tr);
    u64 dir_len = rd_u64(tr + 8);
    u32 magic = rd_u32(tr + 20);
    if (magic != VC_MAGIC) return NULL;
    if (dir_off + dir_len > len) return NULL;
    const u8 *dir = archive + dir_off;
    u32 nm = rd_u32(dir);
    if (nm == 0 || nm > VC_NLOD) return NULL;
    vc_archive *a = (vc_archive *)vc_xcalloc(1, sizeof(*a));
    a->buf = archive; a->len = len; a->nmembers = nm;
    const u8 *e = dir + 4;
    for (u32 i = 0; i < nm; ++i) {
        u64 ro = rd_u64(e); e += 8;
        u64 ml = rd_u64(e); e += 8;
        (void)ml;
        a->members[i] = archive + ro;
        parse_member_header(a->members[i], &a->recs[i]);
    }
    return a;
}
void vc_close(vc_archive *a) { free(a); }

vc_status vc_lod_dims(const vc_archive *a, int lod, vc_dims *out) {
    if (!a || lod < 0 || (u32)lod >= a->nmembers || !out) return VC_ERR_RANGE;
    out->nx = a->recs[lod].nx; out->ny = a->recs[lod].ny; out->nz = a->recs[lod].nz;
    return VC_OK;
}

// Locate the directory entry for an atom (member-relative). Returns offset of
// the 16-byte dir record within the member, plus parsed fields.
static const u8 *member_dir_base(const u8 *m, const member_rec *r) {
    (void)r;
    return m + 9*4 + 4 + 8; // after 9 u32 + f32 base_q + u64 pay_base
}
// Size of one chunk's dir block: 4 (n_atoms) + n_atoms*16.
static vc_status find_atom_entry(const u8 *m, const member_rec *r,
                                 u32 ax, u32 ay, u32 az,
                                 u64 *off, u32 *len, u8 *flags, u8 *uval, u8 *dc) {
    if (ax >= r->acx || ay >= r->acy || az >= r->acz) return VC_ERR_RANGE;
    u32 cx = ax / CHUNK_ATOMS, cy = ay / CHUNK_ATOMS, cz = az / CHUNK_ATOMS;
    u32 lx = ax % CHUNK_ATOMS, ly = ay % CHUNK_ATOMS, lz = az % CHUNK_ATOMS;
    // Walk chunks in raster order up to target, accumulating implicit payload
    // offset (sum of all atom lengths in earlier chunks). Each entry is 8 bytes.
    const u8 *p = member_dir_base(m, r);
    u32 target = (cz*r->ccy + cy)*r->ccx + cx;
    u64 cum = 0; // cumulative payload bytes before target chunk
    for (u32 ci = 0; ci < target; ++ci) {
        u32 na = rd_u32(p); p += 4;
        if (na == 0xFFFFFFFFu) continue; // ABSENT chunk: no entries, no payload
        for (u32 k = 0; k < na; ++k) cum += rd_u32(p + (u64)k*8);
        p += (u64)na * 8;
    }
    u32 na = rd_u32(p); p += 4;
    if (na == 0xFFFFFFFFu) { // target chunk ABSENT
        *off = ABSENT; *len = 0; *flags = AF_ABSENT; *uval = 0; *dc = 0;
        return VC_OK;
    }
    // local atom index within chunk: need chunk's atom extents
    u32 ax0 = cx*CHUNK_ATOMS, ay0 = cy*CHUNK_ATOMS;
    u32 axn = r->acx - ax0; if (axn > CHUNK_ATOMS) axn = CHUNK_ATOMS;
    u32 ayn = r->acy - ay0; if (ayn > CHUNK_ATOMS) ayn = CHUNK_ATOMS;
    u32 ai = (lz*ayn + ly)*axn + lx;
    if (ai >= na) return VC_ERR_RANGE;
    // accumulate lengths of atoms before ai within this chunk
    for (u32 k = 0; k < ai; ++k) cum += rd_u32(p + (u64)k*8);
    const u8 *e = p + (u64)ai * 8;
    *len = rd_u32(e);
    *flags = e[4]; *uval = e[5]; *dc = e[6];
    *off = (*flags & (AF_UNIFORM|AF_ABSENT)) ? ABSENT : (r->pay_base + cum);
    return VC_OK;
}

vc_status vc_decode_atom(vc_archive *a, int lod, int ax, int ay, int az,
                         u8 out[VC_ATOM3]) {
    if (!a || lod < 0 || (u32)lod >= a->nmembers || !out) return VC_ERR_RANGE;
    const member_rec *r = &a->recs[lod];
    const u8 *m = a->members[lod];
    u64 off; u32 len; u8 flags, uval, dc;
    vc_status s = find_atom_entry(m, r, (u32)ax, (u32)ay, (u32)az,
                                  &off, &len, &flags, &uval, &dc);
    if (s != VC_OK) return s;
    if (flags & AF_UNIFORM) { memset(out, uval, A3); return VC_OK; }
    if (flags & AF_ABSENT)  { memset(out, 0, A3); return VC_OK; }
    decode_atom_payload(m + off, len, r->base_q, out);
    return VC_OK;
}

// ---------------------------------------------------------------------------
// Deblock decode post-filter (plan.txt §3). Adaptive filter across atom-grid
// faces (every 16 voxels). Boundary strength keyed on quant step + local
// gradient; filter only flat regions where a step is likely a quant artifact;
// clip per-sample delta so real edges/ink survive; modify <=3 samples/side.
// Decode-side only, no bitstream change. Applied to an assembled volume buffer.
// ---------------------------------------------------------------------------
#define DEBLOCK_STRENGTH 0.4f

// Filter one boundary line of 8 samples (4 each side: p3 p2 p1 p0 | q0 q1 q2 q3)
// in-place. `tc` is the delta clip (derived from quant step). Modifies <=3/side.
static inline void deblock_line(int *p3, int *p2, int *p1, int *p0,
                                int *q0, int *q1, int *q2, int *q3, int tc) {
    int P3=*p3,P2=*p2,P1=*p1,P0=*p0,Q0=*q0,Q1=*q1,Q2=*q2,Q3=*q3;
    // flatness: only filter if both sides are locally smooth
    int dp = abs(P2 - 2*P1 + P0);
    int dq = abs(Q2 - 2*Q1 + Q0);
    int d  = dp + dq;
    int step = abs(P0 - Q0);
    if (step == 0) return;
    // don't touch real edges: if the jump dwarfs local variation it's an edge
    if (step > 2*tc + (d>>1)) return;
    if (d >= tc) return; // not flat enough -> likely real structure
    // weak filter: adjust p0,q0 (and p1,q1 if very flat)
    int delta = (9*(Q0 - P0) - 3*(Q1 - P1) + 8) >> 4;
    if (delta > tc) delta = tc; else if (delta < -tc) delta = -tc;
    int np0 = P0 + delta; if (np0<0)np0=0; else if(np0>255)np0=255;
    int nq0 = Q0 - delta; if (nq0<0)nq0=0; else if(nq0>255)nq0=255;
    *p0 = np0; *q0 = nq0;
    if (d < (tc>>1)) {
        int dp1 = ((P2 + P0 + 1) >> 1) - P1 + delta;
        int dq1 = ((Q2 + Q0 + 1) >> 1) - Q1 - delta;
        int tc1 = tc >> 1;
        if (dp1 > tc1) dp1 = tc1; else if (dp1 < -tc1) dp1 = -tc1;
        if (dq1 > tc1) dq1 = tc1; else if (dq1 < -tc1) dq1 = -tc1;
        int np1 = P1 + dp1; if(np1<0)np1=0; else if(np1>255)np1=255;
        int nq1 = Q1 + dq1; if(nq1<0)nq1=0; else if(nq1>255)nq1=255;
        *p1 = np1; *q1 = nq1;
    }
    (void)P3; (void)Q3; (void)p2; (void)q2; (void)p3; (void)q3;
}

// Deblock a volume across all atom-grid planes for one base_q.
static void deblock_volume(u8 *v, vc_dims d, float base_q) {
    // clip threshold derived from the (DC) quant step, scaled by strength.
    int tc = (int)(base_q * DEBLOCK_STRENGTH);
    if (tc < 1) tc = 1;
    size_t SX = 1, SY = d.nx, SZ = (size_t)d.nx * d.ny;
    // X-normal faces (boundary planes at x = 16,32,...)
    for (u32 bx = A; bx < d.nx; bx += A)
    for (u32 z = 0; z < d.nz; ++z)
    for (u32 y = 0; y < d.ny; ++y) {
        size_t base = (size_t)z*SZ + (size_t)y*SY + bx*SX;
        if (bx < 4 || bx+4 > d.nx) continue;
        int p3=v[base-4*SX],p2=v[base-3*SX],p1=v[base-2*SX],p0=v[base-1*SX];
        int q0=v[base],q1=v[base+1*SX],q2=v[base+2*SX],q3=v[base+3*SX];
        deblock_line(&p3,&p2,&p1,&p0,&q0,&q1,&q2,&q3,tc);
        v[base-2*SX]=(u8)p1; v[base-1*SX]=(u8)p0; v[base]=(u8)q0; v[base+1*SX]=(u8)q1;
    }
    // Y-normal faces
    for (u32 by = A; by < d.ny; by += A)
    for (u32 z = 0; z < d.nz; ++z)
    for (u32 x = 0; x < d.nx; ++x) {
        if (by < 4 || by+4 > d.ny) continue;
        size_t base = (size_t)z*SZ + (size_t)by*SY + x*SX;
        int p3=v[base-4*SY],p2=v[base-3*SY],p1=v[base-2*SY],p0=v[base-1*SY];
        int q0=v[base],q1=v[base+1*SY],q2=v[base+2*SY],q3=v[base+3*SY];
        deblock_line(&p3,&p2,&p1,&p0,&q0,&q1,&q2,&q3,tc);
        v[base-2*SY]=(u8)p1; v[base-1*SY]=(u8)p0; v[base]=(u8)q0; v[base+1*SY]=(u8)q1;
    }
    // Z-normal faces
    for (u32 bz = A; bz < d.nz; bz += A)
    for (u32 y = 0; y < d.ny; ++y)
    for (u32 x = 0; x < d.nx; ++x) {
        if (bz < 4 || bz+4 > d.nz) continue;
        size_t base = (size_t)bz*SZ + (size_t)y*SY + x*SX;
        int p3=v[base-4*SZ],p2=v[base-3*SZ],p1=v[base-2*SZ],p0=v[base-1*SZ];
        int q0=v[base],q1=v[base+1*SZ],q2=v[base+2*SZ],q3=v[base+3*SZ];
        deblock_line(&p3,&p2,&p1,&p0,&q0,&q1,&q2,&q3,tc);
        v[base-2*SZ]=(u8)p1; v[base-1*SZ]=(u8)p0; v[base]=(u8)q0; v[base+1*SZ]=(u8)q1;
    }
}

vc_status vc_decode_lod(vc_archive *a, int lod, u8 *out_vol, vc_dims *out_dims) {
    if (!a || lod < 0 || (u32)lod >= a->nmembers || !out_vol) return VC_ERR_RANGE;
    const member_rec *r = &a->recs[lod];
    if (out_dims) { out_dims->nx = r->nx; out_dims->ny = r->ny; out_dims->nz = r->nz; }
    u8 atom[A3];
    for (u32 az = 0; az < r->acz; ++az)
    for (u32 ay = 0; ay < r->acy; ++ay)
    for (u32 ax = 0; ax < r->acx; ++ax) {
        vc_status s = vc_decode_atom(a, lod, (int)ax, (int)ay, (int)az, atom);
        if (s != VC_OK) return s;
        // scatter into volume (crop to logical extent)
        for (u32 z = 0; z < A; ++z) { u32 vz = az*A + z; if (vz >= r->nz) break;
        for (u32 y = 0; y < A; ++y) { u32 vy = ay*A + y; if (vy >= r->ny) break;
        for (u32 x = 0; x < A; ++x) { u32 vx = ax*A + x; if (vx >= r->nx) break;
            out_vol[((size_t)vz*r->ny + vy)*r->nx + vx] = atom[(z*A + y)*A + x];
        }}}
    }
    { vc_dims vd = { r->nx, r->ny, r->nz }; deblock_volume(out_vol, vd, r->base_q); }
    return VC_OK;
}

vc_status vc_decode_region(vc_archive *a, int lod, vc_box box, u8 *out) {
    if (!a || lod < 0 || (u32)lod >= a->nmembers || !out) return VC_ERR_RANGE;
    const member_rec *r = &a->recs[lod];
    if (box.x1 > r->nx || box.y1 > r->ny || box.z1 > r->nz) return VC_ERR_RANGE;
    if (box.x0 >= box.x1 || box.y0 >= box.y1 || box.z0 >= box.z1) return VC_ERR_RANGE;
    u32 ox = box.x1 - box.x0, oy = box.y1 - box.y0;
    u8 atom[A3];
    u32 ax0 = box.x0 / A, ax1 = (box.x1 - 1) / A;
    u32 ay0 = box.y0 / A, ay1 = (box.y1 - 1) / A;
    u32 az0 = box.z0 / A, az1 = (box.z1 - 1) / A;
    for (u32 az = az0; az <= az1; ++az)
    for (u32 ay = ay0; ay <= ay1; ++ay)
    for (u32 ax = ax0; ax <= ax1; ++ax) {
        vc_status s = vc_decode_atom(a, lod, (int)ax, (int)ay, (int)az, atom);
        if (s != VC_OK) return s;
        for (u32 z = 0; z < A; ++z) { u32 vz = az*A + z; if (vz < box.z0 || vz >= box.z1) continue;
        for (u32 y = 0; y < A; ++y) { u32 vy = ay*A + y; if (vy < box.y0 || vy >= box.y1) continue;
        for (u32 x = 0; x < A; ++x) { u32 vx = ax*A + x; if (vx < box.x0 || vx >= box.x1) continue;
            out[((size_t)(vz-box.z0)*oy + (vy-box.y0))*ox + (vx-box.x0)] = atom[(z*A + y)*A + x];
        }}}
    }
    // Deblock atom-grid faces that fall inside the region (global grid aligned).
    {
        u32 oz = box.z1 - box.z0;
        int tc = (int)(r->base_q * DEBLOCK_STRENGTH); if (tc < 1) tc = 1;
        size_t SX=1, SY=ox, SZ=(size_t)ox*oy;
        for (u32 gx = ((box.x0 + A - 1)/A)*A; gx < box.x1; gx += A) {
            u32 lx = gx - box.x0; if (lx < 4 || lx+4 > ox) continue;
            for (u32 z=0; z<oz; ++z) for (u32 y=0; y<oy; ++y) {
                size_t base=(size_t)z*SZ+(size_t)y*SY+lx*SX;
                int p3=out[base-4*SX],p2=out[base-3*SX],p1=out[base-2*SX],p0=out[base-1*SX];
                int q0=out[base],q1=out[base+SX],q2=out[base+2*SX],q3=out[base+3*SX];
                deblock_line(&p3,&p2,&p1,&p0,&q0,&q1,&q2,&q3,tc);
                out[base-2*SX]=(u8)p1; out[base-SX]=(u8)p0; out[base]=(u8)q0; out[base+SX]=(u8)q1;
            }
        }
        for (u32 gy = ((box.y0 + A - 1)/A)*A; gy < box.y1; gy += A) {
            u32 ly = gy - box.y0; if (ly < 4 || ly+4 > oy) continue;
            for (u32 z=0; z<oz; ++z) for (u32 x=0; x<ox; ++x) {
                size_t base=(size_t)z*SZ+(size_t)ly*SY+x*SX;
                int p3=out[base-4*SY],p2=out[base-3*SY],p1=out[base-2*SY],p0=out[base-1*SY];
                int q0=out[base],q1=out[base+SY],q2=out[base+2*SY],q3=out[base+3*SY];
                deblock_line(&p3,&p2,&p1,&p0,&q0,&q1,&q2,&q3,tc);
                out[base-2*SY]=(u8)p1; out[base-SY]=(u8)p0; out[base]=(u8)q0; out[base+SY]=(u8)q1;
            }
        }
        for (u32 gz = ((box.z0 + A - 1)/A)*A; gz < box.z1; gz += A) {
            u32 lz = gz - box.z0; if (lz < 4 || lz+4 > oz) continue;
            for (u32 y=0; y<oy; ++y) for (u32 x=0; x<ox; ++x) {
                size_t base=(size_t)lz*SZ+(size_t)y*SY+x*SX;
                int p3=out[base-4*SZ],p2=out[base-3*SZ],p1=out[base-2*SZ],p0=out[base-1*SZ];
                int q0=out[base],q1=out[base+SZ],q2=out[base+2*SZ],q3=out[base+3*SZ];
                deblock_line(&p3,&p2,&p1,&p0,&q0,&q1,&q2,&q3,tc);
                out[base-2*SZ]=(u8)p1; out[base-SZ]=(u8)p0; out[base]=(u8)q0; out[base+SZ]=(u8)q1;
            }
        }
    }
    return VC_OK;
}


