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
#define A      VC_ATOM             // atom edge (derives from the public header)
#define A3     VC_ATOM3            // A*A*A
#define CHUNK_ATOMS 4u             // atoms per axis per chunk
#define CHUNK_SIDE  (CHUNK_ATOMS*A) // 128 voxels (4 * 32)

#define Q14    14u                 // DCT fixed-point shift

// ---------------------------------------------------------------------------
// Integer DCT-A (orthonormal scaled-cosine Q14 matrix; inverse = transpose).
// plan.txt §2.3. The A×A matrix is generated once at runtime (build_tables) so
// the codec is atom-size agnostic; the trip-count-A kernels autovectorize.
//   CMATA[k][n] = round( c_k * cos(pi*(2n+1)*k/(2A)) * 2^Q14 ),
//   c_0 = sqrt(1/A), c_k>0 = sqrt(2/A)  → orthonormal, inverse = transpose.
// ---------------------------------------------------------------------------
// Function multiversioning for a PORTABLE single binary: with -DVC_PORTABLE the
// hot DCT/quant kernels are cloned for AVX-512 (v4) / AVX2 (v3) / generic, and an
// ifunc resolver picks the best at runtime — one .so runs optimally on any x86,
// no -march needed (decode ~80% of native). Default build uses -march=native
// instead (fastest, build-on-target). On ARM this is a no-op (use VC_MARCH).
#if defined(VC_PORTABLE) && (defined(__x86_64__) || defined(__i386__))
#define VC_MULTIVERSION __attribute__((target_clones("arch=x86-64-v4","arch=x86-64-v3","default")))
#else
#define VC_MULTIVERSION
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static i32 CMATA[A][A] __attribute__((aligned(64)));
static int g_dct_ready = 0;
static void build_dct_matrix(void) {
    if (g_dct_ready) return;
    const double scale = (double)((i64)1 << Q14);
    for (u32 k = 0; k < A; ++k) {
        double ck = (k == 0) ? sqrt(1.0 / (double)A) : sqrt(2.0 / (double)A);
        for (u32 n = 0; n < A; ++n) {
            double v = ck * cos(M_PI * (2.0*n + 1.0) * k / (2.0 * (double)A));
            CMATA[k][n] = (i32)llround(v * scale);
        }
    }
    g_dct_ready = 1;
}

// Fast forward DCT-II via even/odd partial butterfly (HEVC-style), evaluating the
// SAME CMATA matrix → bit-identical to the naive matmul but ~2x fewer MACs and
// autovectorizable. DCT-II symmetry: C[k][A-1-n] = (-1)^k C[k][n]. So
//   even k:  out[k] = sum_{n<H} C[k][n] * (in[n] + in[A-1-n])   (sums s[])
//   odd  k:  out[k] = sum_{n<H} C[k][n] * (in[n] - in[A-1-n])   (diffs d[])
#define HALF (A/2u)
static inline void dctA_fwd(const i32 *restrict in, i32 *restrict out) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    i32 s[HALF], d[HALF];
    for (u32 n = 0; n < HALF; ++n) { s[n] = in[n] + in[A-1-n]; d[n] = in[n] - in[A-1-n]; }
    for (u32 k = 0; k < A; k += 2) {
        i64 acc = 0;
        for (u32 n = 0; n < HALF; ++n) acc += (i64)CMATA[k][n] * s[n];
        out[k] = (i32)((acc + rnd) >> Q14);
    }
    for (u32 k = 1; k < A; k += 2) {
        i64 acc = 0;
        for (u32 n = 0; n < HALF; ++n) acc += (i64)CMATA[k][n] * d[n];
        out[k] = (i32)((acc + rnd) >> Q14);
    }
}
// Fast inverse: out[n] = sum_k C[k][n]*in[k]. Split by k-parity and use the same
// symmetry to reconstruct both halves from the even/odd partial sums.
//   evn = sum_{even k} C[k][n] in[k];  odd = sum_{odd k} C[k][n] in[k]
//   out[n] = evn + odd;  out[A-1-n] = evn - odd   (n < HALF)
static inline void dctA_inv(const i32 *restrict in, i32 *restrict out) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    for (u32 n = 0; n < HALF; ++n) {
        i64 evn = 0, odd = 0;
        for (u32 k = 0; k < A; k += 2) evn += (i64)CMATA[k][n] * in[k];
        for (u32 k = 1; k < A; k += 2) odd += (i64)CMATA[k][n] * in[k];
        out[n]       = (i32)((evn + odd + rnd) >> Q14);
        out[A-1-n]   = (i32)((evn - odd + rnd) >> Q14);
    }
}

// MAC accumulators are i32, NOT i64 — and that matters a LOT. The 64-bit
// multiplies forced both gcc and clang into emulated/narrow SIMD (vpmullq is slow
// / absent); switching to i32 lets the compiler use fast 32-bit SIMD lanes →
// MEASURED +61% decode, +49% encode. RANGE-SAFE: |CMATA|≤~5800 (Q14), butterfly
// operands |s|,|d|≤510 (fwd) and intermediate coefs stay ≤~3000 across passes
// (each pass normalized by 2^14), summed over HALF=16 terms → max |acc|≈854M,
// well under INT32_MAX (2.1e9). Do NOT widen back to i64.
// 1D DCT along the contiguous (last) axis for all A*A lines, in place. Unit-stride
// → autovectorizes. Coefficient pruning: skip all-zero lines (most are zero at
// high compression). (Tried fusing the axis-rotate into this pass — the strided
// output write killed vectorization, ~5x slower — so the rotate stays a separate
// contiguous-streaming pass.) SPLIT into _fwd/_inv rather than an `inv` flag:
// gcc constant-propagated the flag and cloned (dctA_lines.constprop.0/.1) but
// CLANG did NOT — it left the per-line branch inside the loop, defeating forward
// vectorization (clang encode was 3.5x slower than gcc). Two straight-line
// kernels fix it for BOTH compilers (no reliance on the cloning pass).
VC_MULTIVERSION static void dctA_lines_fwd(i32 *restrict blk) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    static _Thread_local i32 out[A] __attribute__((aligned(64)));
    for (u32 line = 0; line < A*A; ++line) {
        i32 *restrict v = blk + (size_t)line*A;
        { int nz = 0; for (u32 i = 0; i < A; ++i) if (v[i]) { nz = 1; break; } if (!nz) continue; }
        i32 s[HALF], d[HALF];
        for (u32 n = 0; n < HALF; ++n) { s[n] = v[n] + v[A-1-n]; d[n] = v[n] - v[A-1-n]; }
#if defined(__clang__)
        // clang lowers the reduction-over-n form below into a slow horizontal
        // reduction (3.5× slower encode). The mathematically-identical k-parallel
        // form — outputs as the vectorized axis, broadcast s[n]/d[n], no per-output
        // reduction — fixes clang (63→154 MB/s). gcc, however, is FASTER with the
        // reduction form (230 vs 165), so we pick per-compiler. Same results.
        i32 acc[A]; for (u32 k = 0; k < A; ++k) acc[k] = rnd;
        for (u32 n = 0; n < HALF; ++n) {
            i32 sn = s[n], dn = d[n];
            for (u32 k = 0; k < A; k += 2) acc[k] += CMATA[k][n] * sn;
            for (u32 k = 1; k < A; k += 2) acc[k] += CMATA[k][n] * dn;
        }
        for (u32 k = 0; k < A; ++k) out[k] = acc[k] >> Q14;
#else
        for (u32 k = 0; k < A; k += 2) { i32 a=0; for (u32 n=0;n<HALF;++n) a += CMATA[k][n]*s[n]; out[k]=(a+rnd)>>Q14; }
        for (u32 k = 1; k < A; k += 2) { i32 a=0; for (u32 n=0;n<HALF;++n) a += CMATA[k][n]*d[n]; out[k]=(a+rnd)>>Q14; }
#endif
        for (u32 i=0;i<A;++i) v[i]=out[i];
    }
}
VC_MULTIVERSION static void dctA_lines_inv(i32 *restrict blk) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    static _Thread_local i32 out[A] __attribute__((aligned(64)));
    for (u32 line = 0; line < A*A; ++line) {
        i32 *restrict v = blk + (size_t)line*A;
        { int nz = 0; for (u32 i = 0; i < A; ++i) if (v[i]) { nz = 1; break; } if (!nz) continue; }
        for (u32 n = 0; n < HALF; ++n) {
            i32 evn=0, odd=0;
            for (u32 k=0;k<A;k+=2) evn += CMATA[k][n]*v[k];
            for (u32 k=1;k<A;k+=2) odd += CMATA[k][n]*v[k];
            out[n]     = (evn+odd+rnd)>>Q14;
            out[A-1-n] = (evn-odd+rnd)>>Q14;
        }
        for (u32 i=0;i<A;++i) v[i]=out[i];
    }
}
// Out-of-place inverse DCT-lines: read each contiguous src line, transform, write
// to dst. Lets atom_dct_inv fuse the coef→scratch memcpy into the first pass.
static void dctA_lines_inv_to(const i32 *restrict src, i32 *restrict dst) {
    const i32 rnd = (i32)1 << (Q14 - 1);
    for (u32 line = 0; line < A*A; ++line) {
        const i32 *restrict v = src + (size_t)line*A;
        i32 *restrict o = dst + (size_t)line*A;
        { int nz = 0; for (u32 i = 0; i < A; ++i) if (v[i]) { nz = 1; break; }
          if (!nz) { for (u32 i=0;i<A;++i) o[i]=0; continue; } }
        for (u32 n = 0; n < HALF; ++n) {
            i32 evn=0, odd=0;
            for (u32 k=0;k<A;k+=2) evn += CMATA[k][n]*v[k];
            for (u32 k=1;k<A;k+=2) odd += CMATA[k][n]*v[k];
            o[n]     = (evn+odd+rnd)>>Q14;
            o[A-1-n] = (evn-odd+rnd)>>Q14;
        }
    }
}
// Rotate axes (z,y,x)->(x,z,y): dst[(x*A+z)*A+y] = src[(z*A+y)*A+x]. 3 rotations
// return to start. Separate contiguous-read pass (streams well; see dctA_lines).
VC_MULTIVERSION static void rot_zyx_to_xzy(const i32 *restrict src, i32 *restrict dst) {
    for (u32 z=0;z<A;++z) for (u32 y=0;y<A;++y) for (u32 x=0;x<A;++x)
        dst[((size_t)x*A+z)*A+y] = src[((size_t)z*A+y)*A+x];
}
static void atom_dct_fwd(const u8 *restrict vox, i32 dc, i32 *restrict coef) {
    static _Thread_local i32 a[A3] __attribute__((aligned(64))), b[A3] __attribute__((aligned(64)));
    for (u32 i = 0; i < A3; ++i) a[i] = (i32)vox[i] - dc;
    dctA_lines_fwd(a); rot_zyx_to_xzy(a, b);
    dctA_lines_fwd(b); rot_zyx_to_xzy(b, a);
    dctA_lines_fwd(a); rot_zyx_to_xzy(a, coef);
}
static void atom_dct_inv(const i32 *restrict coef, i32 dc, u8 *restrict vox) {
    static _Thread_local i32 a[A3] __attribute__((aligned(64))), b[A3] __attribute__((aligned(64)));
    // Fuse the (former) 128KB memcpy of coef into the first transform pass: the
    // first dctA_lines_inv reads coef directly and writes a — saves one full-buffer
    // sweep per atom (memory-bandwidth is the multithread bottleneck, ~48:1
    // scratch:output, so cutting sweeps directly improves thread scaling).
    dctA_lines_inv_to(coef, a); rot_zyx_to_xzy(a, b);
    dctA_lines_inv(b); rot_zyx_to_xzy(b, a);
    dctA_lines_inv(a); rot_zyx_to_xzy(a, b);
    for (u32 i = 0; i < A3; ++i) { i32 v = b[i] + dc; vox[i] = (u8)(v < 0 ? 0 : v > 255 ? 255 : v); }
}

// ---------------------------------------------------------------------------
// Quantization: dead-zone scalar quantizer + 1-parameter HF-boost curve.
// plan.txt §2.4. step(freq) = base_q * hf_weight(freq), hf_weight protects HF.
// ---------------------------------------------------------------------------
#ifdef VC_BENCH
// Bench-only: let the harness sweep the three frozen quant knobs to check whether
// a better (free) operating point exists on the composed context-coder stack.
float vc_bench_dz  = 0.80f;
float vc_bench_dq  = 0.40f;
float vc_bench_hfe = 0.65f;
#define DZ_FRAC   vc_bench_dz
#define DQ_OFFSET vc_bench_dq
#define HF_EXP    vc_bench_hfe
#else
#define DZ_FRAC   0.80f   // dead-zone width fraction of step
#define DQ_OFFSET 0.40f   // dequant reconstruction sub-center offset
#define HF_EXP    0.65f   // single tuned exponent; <1 => HF kept relatively finer
#endif

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
    build_dct_matrix();
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
        // Levels are stored as i16 — clamp so a very small base_q (large levels)
        // can't wrap the i16 and corrupt the atom. (32767 is far beyond any useful
        // operating point; the rate-control floor keeps q well above this anyway.)
        if (level > 32767) level = 32767;
        q[i] = (i16)(coef[i] < 0 ? -level : level);
    }
}

// Dequant reconstruction: r = (|lv| - 1 + DQ_OFFSET)*step + DZ_FRAC*step, with
// step = base_q*g_step[i]. base_q is CONSTANT per member, so we cache the scaled
// per-coefficient { step, base = (DQ_OFFSET+DZ_FRAC)*step } as integer Q8 tables
// keyed on base_q — turning the per-coef float multiply into integer mul+shift.
// (Reconstruction is the lossy inverse; Q8 rounding is well below the quant step,
// so this is bit-stable vs the float path within the quantizer's own tolerance.)
#define DQ_Q 8
static _Thread_local float g_dq_base_q = -1.0f;
static _Thread_local i32   g_dq_step[A3];   // round(step * 2^DQ_Q)
static _Thread_local i32   g_dq_bias[A3];   // round((DQ_OFFSET-1+DZ_FRAC)*step * 2^DQ_Q)
static void dq_build(float base_q) {
    if (g_dq_base_q == base_q) return;
    for (u32 i = 0; i < A3; ++i) {
        float step = base_q * g_step[i];
        g_dq_step[i] = (i32)lrintf(step * (float)(1 << DQ_Q));
        g_dq_bias[i] = (i32)lrintf((DQ_OFFSET - 1.0f + DZ_FRAC) * step * (float)(1 << DQ_Q));
    }
    g_dq_base_q = base_q;
}
VC_MULTIVERSION static void dequantize(const i16 *restrict q, float base_q, i32 *restrict coef) {
    dq_build(base_q);
    const i32 rnd = 1 << (DQ_Q - 1);
    for (u32 i = 0; i < A3; ++i) {
        i32 lv = q[i];
        i32 mag = lv < 0 ? -lv : lv;
        // r = (mag*step + bias) >> Q   (mag==0 -> we force 0 below)
        i32 r = (mag * g_dq_step[i] + g_dq_bias[i] + rnd) >> DQ_Q;
        r = lv == 0 ? 0 : (lv < 0 ? -r : r);
        coef[i] = r;
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
    u32 freq = cz + cy + cx;            // 0 .. 3*(A-1)
    int b = (int)(freq * NB_BANDS / (3u*A));  // map full L1-freq range onto bands
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
// Coded as an EOB_BITS-bit fixed-length value in bypass; EOB_BITS is derived so
// the value range [0, A3] fits (A3=32768 needs 16 bits; A3=4096 needed 12).
// plan.txt §2.5: decode stops at EOB; trailing zeros cost nothing.
#define EOB_BITS ( (A3) <= 4096u ? 12 : (A3) <= 8192u ? 13 : (A3) <= 16384u ? 14 : (A3) <= 32768u ? 15 : 16 )
static void enc_eob(rc_enc *e, u32 eob) {
    // eob can equal A3 (all-significant) -> needs one extra code point above the
    // EOB_BITS range; clamp by coding min(eob, (1<<EOB_BITS)-1) is WRONG. Instead
    // use ceil(log2(A3+1)) bits: A3=32768 -> 16 bits covers 0..65535.
    int bits = (A3 < (1u<<EOB_BITS)) ? EOB_BITS : EOB_BITS + 1;
    for (int i = bits - 1; i >= 0; --i) enc_bypass(e, (eob >> i) & 1);
}
static u32 dec_eob(rc_dec *d) {
    int bits = (A3 < (1u<<EOB_BITS)) ? EOB_BITS : EOB_BITS + 1;
    u32 v = 0;
    for (int i = 0; i < bits; ++i) v = (v << 1) | (u32)dec_bypass(d);
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
    if (eob > A3) eob = A3;   // bound against a corrupt/malicious archive (EOB
                              // can never legitimately exceed A3; otherwise the
                              // g_scan[p] index below reads out of bounds)
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
    static _Thread_local i16 q[A3] __attribute__((aligned(64)));      // no per-atom stack pressure / it's 64KB
    static _Thread_local i32 coef[A3] __attribute__((aligned(64)));   // was vc_xmalloc per atom — now reentrant TLS
    dec_atom_coefs(&d, q);
    dequantize(q, base_q, coef);
    atom_dct_inv(coef, dc, vox);
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

    // Build payloads + directory. Chunks are INDEPENDENT, so encode each chunk
    // into its OWN payload buffer in parallel (coarse multithread), then
    // concatenate in chunk order. Atom offsets are implicit-cumulative within the
    // member, which the in-order concat preserves. Single-thread without OpenMP.
    u32 nchunks = ccx * ccy * ccz;
    atom_dir **dirs = (atom_dir **)vc_xcalloc(nchunks, sizeof(atom_dir *));
    u32 *chunk_natoms = (u32 *)vc_xcalloc(nchunks, sizeof(u32));
    u8  *chunk_absent = (u8 *)vc_xcalloc(nchunks, sizeof(u8));
    bbuf *chunk_pay = (bbuf *)vc_xcalloc(nchunks, sizeof(bbuf)); // per-chunk payload

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) collapse(1)
    #endif
    for (u32 ci = 0; ci < nchunks; ++ci) {
        u32 cx = ci % ccx, cy = (ci / ccx) % ccy, cz = ci / (ccx*ccy);
        u32 ax0 = cx*CHUNK_ATOMS, ay0 = cy*CHUNK_ATOMS, az0 = cz*CHUNK_ATOMS;
        u32 axn = acx - ax0; if (axn > CHUNK_ATOMS) axn = CHUNK_ATOMS;
        u32 ayn = acy - ay0; if (ayn > CHUNK_ATOMS) ayn = CHUNK_ATOMS;
        u32 azn = acz - az0; if (azn > CHUNK_ATOMS) azn = CHUNK_ATOMS;
        u32 na = axn * ayn * azn;
        chunk_natoms[ci] = na;
        atom_dir *dir = (atom_dir *)vc_xcalloc(na, sizeof(atom_dir));
        dirs[ci] = dir;
        bbuf *pay = &chunk_pay[ci];   // this chunk's own payload buffer
        u8 atom[A3];
        u32 ai = 0;
        for (u32 lz = 0; lz < azn; ++lz)
        for (u32 ly = 0; ly < ayn; ++ly)
        for (u32 lx = 0; lx < axn; ++lx, ++ai) {
            gather_atom(vol, d, ax0+lx, ay0+ly, az0+lz, 0, atom);
            int uni = 0; u8 uval = 0, dcv = 0;
            u32 len = encode_atom(atom, base_q, pay, &uni, &uval, &dcv);
            atom_dir *e = &dir[ai];
            e->flags = uni ? AF_UNIFORM : 0;
            e->uval = uval; e->dc = dcv;
            e->length = len; // offset implicit (cumulative within member)
        }
        int all_absent = 1;
        for (u32 k = 0; k < na; ++k)
            if (!((dir[k].flags & AF_UNIFORM) && dir[k].uval == 0)) { all_absent = 0; break; }
        chunk_absent[ci] = (u8)all_absent;
    }
    // Concatenate per-chunk payloads in chunk order into one member payload.
    bbuf pay = {0};
    for (u32 ci = 0; ci < nchunks; ++ci) {
        if (chunk_pay[ci].len) bb_put(&pay, chunk_pay[ci].p, chunk_pay[ci].len);
        free(chunk_pay[ci].p);
    }
    free(chunk_pay);

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

// Encode a SAMPLE of atoms (every `stride`-th, raster) at base_q and return the
// (sample_raw_bytes, sample_coded_bytes) so we can estimate ratio(q) cheaply
// without encoding the whole volume. Uniform atoms cost ~0 and are counted.
static void sample_ratio_at_q(const u8 *vol, vc_dims d, float base_q, u32 stride,
                              u64 *raw_out, u64 *coded_out) {
    u32 nax = (d.nx + A - 1) / A, nay = (d.ny + A - 1) / A, naz = (d.nz + A - 1) / A;
    u64 raw = 0, coded = 0;
    u8 *atom = (u8 *)vc_xmalloc(A3);
    u32 i = 0;
    for (u32 az = 0; az < naz; ++az)
    for (u32 ay = 0; ay < nay; ++ay)
    for (u32 ax = 0; ax < nax; ++ax) {
        if ((i++ % stride) != 0) continue;
        // gather atom (zero-pad partials)
        for (u32 z = 0; z < A; ++z) { u32 vz = az*A + z;
        for (u32 y = 0; y < A; ++y) { u32 vy = ay*A + y;
        for (u32 x = 0; x < A; ++x) { u32 vx = ax*A + x;
            u8 v = (vz < d.nz && vy < d.ny && vx < d.nx)
                 ? vol[((size_t)vz*d.ny + vy)*d.nx + vx] : 0;
            atom[(z*A + y)*A + x] = v;
        }}}
        bbuf pay = {0}; int uni; u8 uval, dcv;
        u32 len = encode_atom(atom, base_q, &pay, &uni, &uval, &dcv);
        free(pay.p);
        raw += A3;
        coded += uni ? 1 : (len + 2); // +2 ~ per-atom dir/flag overhead estimate
    }
    free(atom);
    *raw_out = raw; *coded_out = coded;
}

// Pick base_q to hit a target ratio. plan.txt §2.7: scroll data is homogeneous
// (q for a target ratio varies only ~15% across a volume — MEASURED), so instead
// of an 11x whole-volume re-encode bisection we CALIBRATE from a cheap spatial
// SAMPLE: probe 2 q values on ~1/SAMPLE_STRIDE of the atoms, fit the near-linear
// log(ratio)=m*log(q)+b relationship, and solve for the target q in closed form.
// One light-weight pass instead of 11 full encodes -> ~single-pass encode speed.
#define SAMPLE_STRIDE 64u   // sample ~1.5% of atoms for calibration
// Crop a representative sub-volume (central ≤CAL_SIDE^3 corner-aligned region)
// for calibration. q is spatially homogeneous (~15% stdev — MEASURED), so a
// sub-volume predicts the whole-member q well, and bisecting on a 128³ crop
// instead of a 1024³ member is ~512× less encode work per probe. Returns the
// crop dims; copies into `dst` (caller-owned, CAL_SIDE^3 max).
#define CAL_SIDE 128u
static vc_dims calib_crop(const u8 *vol, vc_dims d, u8 *dst) {
    vc_dims c = { d.nx < CAL_SIDE ? d.nx : CAL_SIDE,
                  d.ny < CAL_SIDE ? d.ny : CAL_SIDE,
                  d.nz < CAL_SIDE ? d.nz : CAL_SIDE };
    // center the crop so it samples interior content, not just a corner edge
    u32 ox = (d.nx - c.nx)/2, oy = (d.ny - c.ny)/2, oz = (d.nz - c.nz)/2;
    for (u32 z = 0; z < c.nz; ++z)
    for (u32 y = 0; y < c.ny; ++y)
        memcpy(dst + ((size_t)z*c.ny + y)*c.nx,
               vol + (((size_t)(z+oz)*d.ny + (y+oy))*d.nx + ox), c.nx);
    return c;
}
static float pick_q_for_ratio(const u8 *vol, vc_dims d, float target_ratio) {
    u64 raw = (u64)d.nx * d.ny * d.nz;
    if (raw == 0) return 1.0f;
    // Calibrate on a representative sub-volume if the member is large (q is
    // spatially homogeneous, so the sub-volume's q≈the member's q). Small members
    // calibrate on themselves. This keeps rate control CORRECT (real measure_member
    // bisection) while cutting its cost from O(full member) to O(128³).
    static _Thread_local u8 cbuf[CAL_SIDE*CAL_SIDE*CAL_SIDE];
    vc_dims cd = d; const u8 *cvol = vol;
    if (d.nx > CAL_SIDE || d.ny > CAL_SIDE || d.nz > CAL_SIDE) { cd = calib_crop(vol, d, cbuf); cvol = cbuf; }
    vol = cvol; d = cd;
    raw = (u64)d.nx * d.ny * d.nz;
    u64 target_bytes = (u64)((double)raw / (double)target_ratio);
    if (target_bytes < 1) target_bytes = 1;
    // Bisection on the REAL coded member size (ground truth — accounts for the
    // directory, concatenated payloads, uniform/absent atoms, everything). The
    // earlier cheap-per-atom-sample estimator was fragile (its ratio diverged
    // from the true member size on very-compressible or thin volumes → picked a
    // clamped q and crushed quality). Encode is the least-important axis and a
    // member is small (esp. coarse LODs), so a handful of real encodes is the
    // right trade for correctness. Monotone: higher q → smaller size.
    // q floor 0.5: below this the dead-zone quantizer produces levels that add no
    // real quality (already near-lossless) and only stress the i16 level range.
    // The operating range is 10-100×; if a target is so low it needs q<0.5, the
    // data simply can't be compressed that little and we return the gentlest q.
    float lo = 0.5f, hi = 4096.0f;
    float best = 8.0f; u64 best_sz = measure_member_size(vol, d, best);
    u64 bd = best_sz > target_bytes ? best_sz - target_bytes : target_bytes - best_sz;
    for (int it = 0; it < 16; ++it) {
        float mid = sqrtf(lo * hi);
        u64 sz = measure_member_size(vol, d, mid);
        u64 dd = sz > target_bytes ? sz - target_bytes : target_bytes - sz;
        if (dd < bd) { bd = dd; best = mid; }
        if (sz > target_bytes) lo = mid; else hi = mid;
    }
    return best;
}

// ---------------------------------------------------------------------------
// --- Benchmark-only hooks (not part of the frozen public API) --------------
// vc_bench_encode_full: like vc_encode but also returns the per-LOD base_q it
// chose (the result of the rate-search bisection) and the per-LOD member dims.
// vc_bench_encode_singlepass: encode all LODs with q ALREADY chosen (no
// bisection) — measures the fundamental single-pass encode throughput.
#ifdef VC_BENCH
vc_status vc_bench_encode_full(const u8 *vol, vc_dims dims, float target_ratio,
                               u8 **out_archive, size_t *out_len,
                               float qout[VC_NLOD], int *nlod_out);
vc_status vc_bench_encode_singlepass(const u8 *vol, vc_dims dims,
                                     const float qin[VC_NLOD], int nlod,
                                     u8 **out_archive, size_t *out_len);
#endif

// vc_encode — top level. Builds the 8-LOD pyramid (step 6) and footer.
// ---------------------------------------------------------------------------
vc_status vc_encode(const u8 *vol, vc_dims dims, float target_ratio,
                    u8 **out_archive, size_t *out_len) {
    build_tables();
    VC_CHECK(vol && out_archive && out_len, "vc_encode null arg");
    VC_CHECK(target_ratio >= 1.0f, "target_ratio < 1");

    bbuf arc = {0};
    // file header: magic, version, then SELF-DESCRIBING geometry so the format is
    // not silently tied to compile-time constants — vc_open validates these and
    // rejects an archive built with a different atom/chunk size. (1.0 freeze.)
    bb_u32(&arc, VC_MAGIC);
    bb_u32(&arc, VC_VERSION);
    bb_u32(&arc, VC_ATOM);         // atom edge (voxels) — decoder must match
    bb_u32(&arc, CHUNK_ATOMS);     // atoms per chunk axis

    member_rec recs[VC_NLOD];
    int nmembers = 0;

    // Rate control (plan.txt §2.7): scroll data is statistically homogeneous, so
    // the q that hits the target ratio is near-constant across chunks (measured:
    // ~15% stdev over a 1024^3 volume). We therefore calibrate ONE global q on
    // LOD0 (a single bisection) and reuse it for every LOD member — no per-chunk
    // or per-LOD search. Encode is then single-pass (~70 MB/s) instead of paying
    // an 11x re-encode bisection per member. Per-LOD ratio may wobble slightly;
    // LOD0 dominates the byte budget so the overall ratio tracks the target.
    float gq = pick_q_for_ratio(vol, dims, target_ratio);

    // LOD0 = original; LOD k+1 = 2x downsample of level above (plan.txt §5).
    u8 *cur = (u8 *)vol; vc_dims cd = dims;
    u8 *owned = NULL; // buffers we must free (not the caller's vol)
    for (int lod = 0; lod < VC_NLOD; ++lod) {
        encode_member(cur, cd, gq, &arc, &recs[nmembers]);
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
    // verify file header: magic, version, and SELF-DESCRIBING geometry. Reject an
    // archive from a different version or a different atom/chunk size (would
    // misdecode silently otherwise — EOB width, band map, atom grid all depend on
    // atom size). This is what makes the 1.0 format safely frozen.
    if (rd_u32(archive) != VC_MAGIC) return NULL;
    if (rd_u32(archive + 4) != VC_VERSION) return NULL;
    if (rd_u32(archive + 8) != VC_ATOM) return NULL;
    if (rd_u32(archive + 12) != CHUNK_ATOMS) return NULL;
    // trailer: last 24 bytes = [u64 dir_off][u64 dir_len][u32 ver][u32 magic].
    // ALL structural offsets/lengths below are validated against `len` (overflow-
    // safe) before any dereference — a malformed/malicious archive must return
    // NULL, never read out of bounds. (Hardened for untrusted input; fuzzed.)
    const u8 *tr = archive + len - 24;
    u64 dir_off = rd_u64(tr);
    u64 dir_len = rd_u64(tr + 8);
    u32 magic = rd_u32(tr + 20);
    if (magic != VC_MAGIC) return NULL;
    if (dir_off > len || dir_len > len - dir_off) return NULL;  // overflow-safe
    if (dir_len < 4) return NULL;
    const u8 *dir = archive + dir_off;
    u32 nm = rd_u32(dir);
    if (nm == 0 || nm > VC_NLOD) return NULL;
    // directory body must hold nm member entries of 16 bytes each, after the u32 count
    if ((u64)nm * 16u > dir_len - 4) return NULL;
    vc_archive *a = (vc_archive *)vc_xcalloc(1, sizeof(*a));
    a->buf = archive; a->len = len; a->nmembers = nm;
    const u8 *e = dir + 4;
    const u64 MIN_HDR = 9u*4u + 4u + 8u; // 9×u32 dims + f32 base_q + u64 pay_base
    for (u32 i = 0; i < nm; ++i) {
        u64 ro = rd_u64(e); e += 8;
        u64 ml = rd_u64(e); e += 8;
        // member must lie within the archive and have room for its header
        if (ro > len || ml > len - ro || ml < MIN_HDR) { free(a); return NULL; }
        a->members[i] = archive + ro;
        a->recs[i].rel_offset = ro;
        a->recs[i].length = ml;
        parse_member_header(a->members[i], &a->recs[i]);
        member_rec *r = &a->recs[i];
        // sanity-check parsed geometry: atom/chunk counts consistent and bounded,
        // pay_base within the member. Rejects corrupt headers before any decode.
        if (r->acx==0||r->acy==0||r->acz==0||r->ccx==0||r->ccy==0||r->ccz==0) { free(a); return NULL; }
        if ((u64)r->acx > 1u<<20 || (u64)r->acy > 1u<<20 || (u64)r->acz > 1u<<20) { free(a); return NULL; }
        if (r->pay_base > ml) { free(a); return NULL; }
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
    // offset. Every read is bounds-checked against the member end `mend` so a
    // corrupt directory (huge n_atoms, truncated member) returns an error rather
    // than reading out of bounds. Each atom entry is 8 bytes.
    const u8 *p = member_dir_base(m, r);
    const u8 *mend = m + r->length;            // one past the member's last byte
    if (p > mend) return VC_ERR_FORMAT;
    u32 target = (cz*r->ccy + cy)*r->ccx + cx;
    u64 cum = 0; // cumulative payload bytes before target chunk
    for (u32 ci = 0; ci < target; ++ci) {
        if (p + 4 > mend) return VC_ERR_FORMAT;
        u32 na = rd_u32(p); p += 4;
        if (na == 0xFFFFFFFFu) continue; // ABSENT chunk: no entries, no payload
        if ((u64)na * 8u > (u64)(mend - p)) return VC_ERR_FORMAT;
        for (u32 k = 0; k < na; ++k) cum += rd_u32(p + (u64)k*8);
        p += (u64)na * 8;
    }
    if (p + 4 > mend) return VC_ERR_FORMAT;
    u32 na = rd_u32(p); p += 4;
    if (na == 0xFFFFFFFFu) { // target chunk ABSENT
        *off = ABSENT; *len = 0; *flags = AF_ABSENT; *uval = 0; *dc = 0;
        return VC_OK;
    }
    if ((u64)na * 8u > (u64)(mend - p)) return VC_ERR_FORMAT;
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
    // Validate the payload slice lies fully within the archive before reading
    // (off/len come from a possibly-corrupt directory; decode_atom_payload reads
    // pay[0..len-1]). Reject out-of-range slices instead of dereferencing wild.
    const u8 *pp = m + off;
    if (len < 1 || pp < a->buf || pp + len > a->buf + a->len) {
        memset(out, 0, A3); return VC_ERR_FORMAT;
    }
    decode_atom_payload(pp, len, r->base_q, out);
    return VC_OK;
}


vc_status vc_decode_lod(vc_archive *a, int lod, u8 *out_vol, vc_dims *out_dims) {
    if (!a || lod < 0 || (u32)lod >= a->nmembers || !out_vol) return VC_ERR_RANGE;
    const member_rec *r = &a->recs[lod];
    if (out_dims) { out_dims->nx = r->nx; out_dims->ny = r->ny; out_dims->nz = r->nz; }
    // Coarse multithread: atoms are FULLY independent (each decodes standalone
    // into a distinct output region; vc_decode_atom uses thread-local scratch and
    // no deblock means no cross-atom dependency). One flat parallel-for over the
    // atom grid. Compiles to single-thread when OpenMP is absent (VC_OPENMP off).
    const u64 natoms = (u64)r->acz * r->acy * r->acx;
    const u32 acyx = r->acy * r->acx;
    volatile vc_status err = VC_OK;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (u64 ai = 0; ai < natoms; ++ai) {
        if (err != VC_OK) continue;
        u32 az = (u32)(ai / acyx), rem = (u32)(ai % acyx);
        u32 ay = rem / r->acx, ax = rem % r->acx;
        u8 atom[A3];
        vc_status s = vc_decode_atom(a, lod, (int)ax, (int)ay, (int)az, atom);
        if (s != VC_OK) { err = s; continue; }
        for (u32 z = 0; z < A; ++z) { u32 vz = az*A + z; if (vz >= r->nz) break;
        for (u32 y = 0; y < A; ++y) { u32 vy = ay*A + y; if (vy >= r->ny) break;
        for (u32 x = 0; x < A; ++x) { u32 vx = ax*A + x; if (vx >= r->nx) break;
            out_vol[((size_t)vz*r->ny + vy)*r->nx + vx] = atom[(z*A + y)*A + x];
        }}}
    }
    return err;
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
    // No deblock (dropped — see vc_decode_lod). Region output is exactly the
    // per-atom decode scattered into place.
    return VC_OK;
}

// ---------------------------------------------------------------------------
// Benchmark-only encode hooks (compiled only with -DVC_BENCH). Reuse the same
// internal pipeline as vc_encode; the only difference is they expose / accept
// the per-LOD base_q so the cost of the rate-search bisection can be separated
// from the cost of the actual single-pass encode.
// ---------------------------------------------------------------------------
#ifdef VC_BENCH
// Optional per-LOD progress hook (bench-only). Called after each LOD member is
// encoded: (lod, dims, chosen q, member bytes, raw voxels of that LOD).
void (*vc_bench_lod_cb)(int lod, vc_dims d, float q, u64 member_bytes, u64 raw) = NULL;
// Cap the number of LODs encoded (bench-only). 0 = all VC_NLOD. Set to 1 to
// measure native-resolution LOD0 only (LODs are independent, so this is just
// "encode the one member we score" — no pyramid).
int vc_bench_max_lods = 0;
static vc_status vc_bench_encode_impl(const u8 *vol, vc_dims dims,
                                      float target_ratio, const float *qin,
                                      u8 **out_archive, size_t *out_len,
                                      float qout[VC_NLOD], int *nlod_out) {
    build_tables();
    bbuf arc = {0};
    bb_u32(&arc, VC_MAGIC);
    bb_u32(&arc, VC_VERSION);
    bb_u32(&arc, VC_ATOM);
    bb_u32(&arc, CHUNK_ATOMS);
    member_rec recs[VC_NLOD];
    int nmembers = 0;
    u8 *cur = (u8 *)vol; vc_dims cd = dims;
    u8 *owned = NULL;
    for (int lod = 0; lod < VC_NLOD; ++lod) {
        float q = qin ? qin[nmembers] : pick_q_for_ratio(cur, cd, target_ratio);
        if (qout) qout[nmembers] = q;
        encode_member(cur, cd, q, &arc, &recs[nmembers]);
        if (vc_bench_lod_cb)
            vc_bench_lod_cb(lod, cd, q, recs[nmembers].length,
                            (u64)cd.nx * cd.ny * cd.nz);
        nmembers++;
        if (cd.nx <= 1 && cd.ny <= 1 && cd.nz <= 1) break;
        if (vc_bench_max_lods && nmembers >= vc_bench_max_lods) break;
        vc_dims nd; u8 *nv = downsample2x(cur, cd, &nd);
        if (owned) free(owned);
        owned = nv; cur = nv; cd = nd;
    }
    if (owned) free(owned);
    u64 dir_off = arc.len;
    bb_u32(&arc, (u32)nmembers);
    for (int i = 0; i < nmembers; ++i) {
        bb_u64(&arc, recs[i].rel_offset);
        bb_u64(&arc, recs[i].length);
    }
    u64 dir_len = arc.len - dir_off;
    bb_align(&arc, 8);
    bb_u64(&arc, dir_off);
    bb_u64(&arc, dir_len);
    bb_u32(&arc, VC_VERSION);
    bb_u32(&arc, VC_MAGIC);
    *out_archive = arc.p; *out_len = arc.len;
    if (nlod_out) *nlod_out = nmembers;
    return VC_OK;
}
vc_status vc_bench_encode_full(const u8 *vol, vc_dims dims, float target_ratio,
                               u8 **out_archive, size_t *out_len,
                               float qout[VC_NLOD], int *nlod_out) {
    return vc_bench_encode_impl(vol, dims, target_ratio, NULL,
                                out_archive, out_len, qout, nlod_out);
}
vc_status vc_bench_encode_singlepass(const u8 *vol, vc_dims dims,
                                     const float qin[VC_NLOD], int nlod,
                                     u8 **out_archive, size_t *out_len) {
    (void)nlod;
    return vc_bench_encode_impl(vol, dims, 0.0f, qin,
                                out_archive, out_len, NULL, NULL);
}
// Encode ONE 16^3 atom at base_q `q`; return payload bytes and (optionally)
// reconstruct it into `recon`. Used to measure per-block-q vs global-q quality
// at equal bytes, without touching the archive format. Returns the EXACT bytes
// the real codec would spend on this atom (uniform atoms cost 0 payload bytes).
void vc_bench_reset_tables(void) { g_tables_ready = 0; build_tables(); }
u32 vc_bench_atom_rt(const u8 vox[VC_ATOM3], float q, u8 recon[VC_ATOM3]) {
    build_tables();
    bbuf pay = {0}; int uni; u8 uval, dcv;
    u32 len = encode_atom(vox, q, &pay, &uni, &uval, &dcv);
    if (uni) { if (recon) memset(recon, uval, A3); free(pay.p); return 0; }
    if (recon) decode_atom_payload(pay.p, pay.len, q, recon);
    free(pay.p);
    return len;
}
#endif


