// volume-compressor — lossy 3D u8 codec with a sparse, appendable archive.
// Core: DCT-32 + dead-zone quant + context coder; archive maps (lod,az,ay,ax)
// to a self-contained atom payload via a two-level sparse index. See docs/SPEC.md.
#if defined(__linux__)
#define _GNU_SOURCE   // enables mremap on Linux (faster in-place file remap)
#endif
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
#if defined(__clang__) || defined(__aarch64__)
        // clang (any arch) and gcc-on-aarch64 lower the reduction-over-n form
        // below into a slow scalar/horizontal reduction. The mathematically-
        // identical k-parallel form — outputs as the vectorized axis, broadcast
        // s[n]/d[n], no per-output reduction — vectorizes cleanly: ~4.3× faster
        // on Neoverse-V2 (gcc) and fixes clang on x86. gcc-on-x86 is instead
        // FASTER with the reduction form (230 vs 165 MB/s) so it keeps it below.
        // Bit-identical either way.
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
// Archive format constants (docs/SPEC.md). Sparse two-level index: a region
// hash table (L1) → dense R³ atom-slot blocks (L2), mapping (lod,az,ay,ax) to a
// self-contained atom payload. See the writer/reader sections below.
// ===========================================================================
#define VC_MAGIC      0x00314356u // "VC1\0"
#define VC_VERSION    1u
#define VC_LAYOUT_SPARSE2 1u      // file-header layout word: two-level sparse
#define ABSENT        0xFFFFFFFFFFFFFFFFull

// Atom-slot flags (level-2 dense block). A zero flags byte == AF_ABSENT.
#define AF_ABSENT  0u   // unwritten / never fetched (coverage ABSENT)
#define AF_PRESENT 1u   // real payload at offset/length
#define AF_ZERO    2u   // known-zero atom, no payload (coverage KNOWN_ZERO)

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

// Encode one atom. An all-zero atom carries no payload (writer flags it
// AF_ZERO): sets *is_zero=1, returns 0. Otherwise appends the payload
// [dc u8][rc bytes...] to `out` and returns its length.
static u32 encode_atom(const u8 *vox, float base_q, bbuf *out,
                       int *is_zero, u8 *dcv) {
    // Single pass: sum the voxels. For unsigned bytes, sum==0 iff the atom is all
    // zero — so the zero-check and the DC mean fold into one scan (was two passes).
    u32 sum = 0;
    for (u32 i = 0; i < A3; ++i) sum += vox[i];
    if (sum == 0) { *is_zero = 1; *dcv = 0; return 0; }
    *is_zero = 0;
    i32 dc = (i32)((sum + A3/2) / A3);
    *dcv = (u8)dc;

    // Per-atom scratch is thread-local and reused across atoms — was malloc/free
    // per atom (2 each), which dominated encode_atom's non-DCT time at scale. The
    // decode path already does this; mirror it here. Reentrant (one set per thread).
    static _Thread_local i32 coef[A3] __attribute__((aligned(64)));
    static _Thread_local i16 q[A3]    __attribute__((aligned(64)));
    static _Thread_local u8 scratch[A3 * 4 + 64] __attribute__((aligned(64)));
    atom_dct_fwd(vox, dc, coef);
    quantize(coef, base_q, q);

    size_t start = out->len;
    bb_u8(out, (u8)dc);
    rc_enc e; enc_init(&e, scratch, sizeof scratch);
    enc_atom_coefs(&e, q);
    enc_flush(&e);
    bb_put(out, scratch, e.len);
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

// ===========================================================================
// Sparse appendable archive (docs/SPEC.md).
//
// Two-level index over a single mmap'd file, mutated in place; atom payloads
// and L2 blocks append at EOF and are never moved. One index, no footer.
//
//   [file header + index header]  (FILE_HDR bytes, 64B aligned)
//   [L1 hash table]               (l1_cap × L1_ENTRY)        ← mutated in place
//   [L2 blocks + atom payloads]   (append at EOF via cursor) ← never rewritten
//
// L1 entry: [u64 region_key][u64 block_ref]
//   block_ref == 0            empty bucket (header occupies offset 0)
//   block_ref == VC_ZERO_REGION  known all-zero region (no L2 block)
//   else                      file offset of the dense R³ L2 block
// L2 slot (ATOM_SLOT bytes): [u64 offset][u32 length][u8 flags][u8 dc][u16 pad]
//   `offset` is the release-published commit word (acquire-loaded by readers).
// ===========================================================================

#include <pthread.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define R_ATOMS    32u                 // index-region edge in atoms (1024 voxels)
#define R3         (R_ATOMS*R_ATOMS*R_ATOMS)
#define ATOM_SLOT  16u                 // bytes per L2 slot
#define L1_ENTRY   16u                 // bytes per L1 bucket
#define L2_BLOCK_BYTES ((u64)R3 * ATOM_SLOT)
#define VC_ZERO_REGION 1ull            // block_ref sentinel: whole region known-0
#define VC_CLAIMED_REGION 2ull         // block_ref sentinel: claimed, block being
                                       // allocated by the CAS winner (transient)
#define L1_INIT_CAP   1024u            // initial L1 buckets (grows by rehash)
#define L1_MAX_LOAD_N 7u               // grow when count*10 >= cap*7  (70%)
#define L1_MAX_LOAD_D 10u

uint32_t vc_region_atoms(void) { return R_ATOMS; }

// ---- file header (little-endian, packed by field offset) -------------------
//  0  u32 magic
//  4  u32 version
//  8  u32 atom            (== VC_ATOM)
// 12  u32 layout          (== VC_LAYOUT_SPARSE2)
// 16  u32 region_atoms    (== R_ATOMS)
// 20  u32 nlod
// 24  u32 nx,ny,nz        (lod0 dims)
// 36  f32 target_ratio
// 40  u64 l1_cap
// 48  u64 l1_count
// 56  u64 l1_off          (file offset of L1 table)
// 64  u64 cursor          (append EOF cursor; bytes used)
// 72  f32 base_q[8]       (per-lod, 0 == not yet learned)
// 104 ... pad to 128
#define FH_MAGIC 0
#define FH_VER   4
#define FH_ATOM  8
#define FH_LAYOUT 12
#define FH_RGN   16
#define FH_NLOD  20
#define FH_NX    24
#define FH_NY    28
#define FH_NZ    32
#define FH_TRATIO 36
#define FH_L1CAP 40
#define FH_L1CNT 48
#define FH_L1OFF 56
#define FH_CURSOR 64
#define FH_BASEQ 72
#define FILE_HDR 128u

// split-mix64 finalizer — good integer hash for the packed region key.
static inline u64 mix64(u64 x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static inline u64 region_key(int lod, u32 rz, u32 ry, u32 rx) {
    return ((u64)(lod & 7) << 45) | ((u64)(rz & 0x7FFF) << 30)
         | ((u64)(ry & 0x7FFF) << 15) | (u64)(rx & 0x7FFF);
}

// derive lod dims from lod0 via strict 2x pyramid.
static vc_dims lod_dims_of(vc_dims d0, int lod) {
    vc_dims d = d0;
    for (int i = 0; i < lod; ++i) {
        d.nx = (d.nx + 1) / 2; d.ny = (d.ny + 1) / 2; d.nz = (d.nz + 1) / 2;
    }
    return d;
}
static int dims_valid(vc_dims d) {
    return d.nx && d.ny && d.nz &&
           d.nx <= VC_MAX_DIM && d.ny <= VC_MAX_DIM && d.nz <= VC_MAX_DIM;
}

// ===========================================================================
// Writer
// ===========================================================================
// On Linux we reserve a large VIRTUAL address range once (cheap; MAP_NORESERVE
// so unwritten pages cost nothing) and grow the FILE under it with ftruncate —
// the mmap base NEVER moves, so reads need no lock and the cursor just advances.
// macOS lacks reliable map-past-EOF growth, so it falls back to remap-on-grow.
#define VC_RESERVE  (10ull << 40)   // 10 TiB virtual reservation (Linux)
#define VC_GROW_STEP (1ull << 30)   // grow the file 1 GiB at a time

struct vc_writer {
    int     fd;
    u8     *map;            // mmap base (stable for the writer's life on Linux)
    u64     map_len;        // mapped length (== reservation on Linux)
    u64     file_len;       // file length actually backed by ftruncate
    int     fixed_map;      // 1 = base never moves (Linux huge map); 0 = remap mode
    pthread_mutex_t grow_mu;// serializes ftruncate/remap growth
    pthread_rwlock_t lock;  // guards L1 structural mutations (region create/rehash)
    atomic_uint_least64_t cursor; // append EOF cursor (mirrors header FH_CURSOR)
    vc_dims dims0;
    int     nlod;
    float   target_ratio;
    // Per-LOD base_q, set explicitly by the caller via vc_set_base_q (no
    // auto-calibration). Atomic so the append hot path reads it lock-free and
    // encodes outside the structural lock. 0.0 == not set (appending errors).
    _Atomic float base_q[VC_NLOD];
};

static u32 rd_u32m(const u8 *p){ u32 v; memcpy(&v,p,4); return v; }
static u64 rd_u64m(const u8 *p){ u64 v; memcpy(&v,p,8); return v; }
static void wr_u32m(u8 *p, u32 v){ memcpy(p,&v,4); }
static void wr_u64m(u8 *p, u64 v){ memcpy(p,&v,8); }
static float rd_f32m(const u8 *p){ float v; memcpy(&v,p,4); return v; }
static void wr_f32m(u8 *p, float v){ memcpy(p,&v,4); }

// Ensure the file backs at least `need` bytes (so accesses to offset<need via
// the mapping don't SIGBUS). Thread-safe via grow_mu; does NOT require the
// rwlock. On the fixed huge map (Linux) the base never moves — just ftruncate.
// In remap mode (macOS) it remaps, which moves the base, so remap-mode callers
// must hold the rwlock exclusively around any raw-pointer use (see writer_alloc).
static int writer_ensure(vc_writer *w, u64 need) {
    if (need <= w->file_len) return 0;
    pthread_mutex_lock(&w->grow_mu);
    if (need <= w->file_len) { pthread_mutex_unlock(&w->grow_mu); return 0; }
    // grow the file in big steps to amortize the metadata ops
    u64 nl = w->file_len ? w->file_len : FILE_HDR;
    while (nl < need) nl += VC_GROW_STEP;
    if (w->fixed_map) {
        if (nl > w->map_len) nl = w->map_len;   // bounded by the reservation
        if (need > w->map_len) { pthread_mutex_unlock(&w->grow_mu); return -1; } // exceeded 10TB
        if (ftruncate(w->fd, (off_t)nl) != 0) { pthread_mutex_unlock(&w->grow_mu); return -1; }
        w->file_len = nl;                        // base pointer unchanged
    } else {
        if (ftruncate(w->fd, (off_t)nl) != 0) { pthread_mutex_unlock(&w->grow_mu); return -1; }
        munmap(w->map, w->map_len);
        void *nm = mmap(NULL, nl, PROT_READ|PROT_WRITE, MAP_SHARED, w->fd, 0);
        if (nm == MAP_FAILED) { pthread_mutex_unlock(&w->grow_mu); return -1; }
        w->map = (u8 *)nm; w->map_len = nl; w->file_len = nl;
    }
    pthread_mutex_unlock(&w->grow_mu);
    return 0;
}

// Reserve `n` bytes at the append cursor (atomic) and ensure the file backs it.
// Returns the file offset of the reserved range.
static u64 writer_alloc(vc_writer *w, u64 n, int have_excl) {
    (void)have_excl;
    u64 off = atomic_fetch_add(&w->cursor, n);
    if (off + n > w->file_len) writer_ensure(w, off + n);
    return off;
}

// L1 probe over the table at file offset l1_off with capacity cap. Returns the
// bucket index for `key` (existing or first empty). cap is a power of two.
static u64 l1_probe(const u8 *map, u64 l1_off, u64 cap, u64 key) {
    u64 mask = cap - 1;
    u64 i = mix64(key) & mask;
    for (;;) {
        const u8 *e = map + l1_off + i*L1_ENTRY;
        u64 bref = rd_u64m(e + 8);
        if (bref == 0) return i;                 // empty bucket
        if (rd_u64m(e) == key) return i;         // found
        i = (i + 1) & mask;
    }
}

// Rehash L1 into a new, larger table appended at EOF. Caller holds excl lock.
static int l1_grow(vc_writer *w) {
    u64 old_off = rd_u64m(w->map + FH_L1OFF);
    u64 old_cap = rd_u64m(w->map + FH_L1CAP);
    u64 new_cap = old_cap * 2;
    u64 new_off = writer_alloc(w, new_cap * L1_ENTRY, 1);
    if (writer_ensure(w, new_off + new_cap*L1_ENTRY) != 0) return -1;
    memset(w->map + new_off, 0, new_cap * L1_ENTRY);
    // reinsert occupied buckets
    for (u64 i = 0; i < old_cap; ++i) {
        const u8 *e = w->map + old_off + i*L1_ENTRY;
        u64 bref = rd_u64m(e + 8);
        if (bref == 0) continue;
        u64 key = rd_u64m(e);
        u64 j = l1_probe(w->map, new_off, new_cap, key);
        u8 *ne = w->map + new_off + j*L1_ENTRY;
        wr_u64m(ne, key); wr_u64m(ne + 8, bref);
    }
    wr_u64m(w->map + FH_L1OFF, new_off);
    wr_u64m(w->map + FH_L1CAP, new_cap);
    // old table becomes dead space (no compaction; cache is rebuildable).
    return 0;
}

// Atomic accessors over an 8-byte field in the (naturally-aligned) mmap.
static inline _Atomic u64 *atomptr(u8 *p) { return (_Atomic u64 *)p; }

// LOCK-FREE find-or-create of a region's L2 block. Returns the block file offset,
// VC_ZERO_REGION if known-zero, or 0 (only when create=0 and absent). The L1
// table is preallocated large enough to never rehash, so there is NO lock: reads
// are atomic acquire-loads; a new region is claimed by CAS-publishing its block
// offset into an empty bucket (losers free their block and use the winner's).
//
// Bucket layout per entry: [u64 key][u64 block_ref]. We use block_ref as the
// single commit word: a reader that sees block_ref!=0 is guaranteed (via the CAS
// release) to also see the matching key. We therefore write `key` BEFORE the CAS
// that publishes block_ref.
static u64 region_block_lockfree(vc_writer *w, u64 key, int create) {
    u64 cap = rd_u64m(w->map + FH_L1CAP);
    u64 off = rd_u64m(w->map + FH_L1OFF);
    u64 mask = cap - 1;
    u64 i = mix64(key) & mask;
    for (;;) {
        u8 *e = w->map + off + i*L1_ENTRY;
        u64 bref = atomic_load_explicit(atomptr(e+8), memory_order_acquire);
        if (bref == VC_CLAIMED_REGION) {
            // Another thread won this bucket and is allocating the block right now.
            // It will be our key (key is written before the claim). Spin until the
            // real offset is published, then return it.
            if (rd_u64m(e) == key) {
                do { bref = atomic_load_explicit(atomptr(e+8), memory_order_acquire); }
                while (bref == VC_CLAIMED_REGION);
                return bref;
            }
            i = (i + 1) & mask; continue;           // claimed by a DIFFERENT key
        }
        if (bref != 0) {
            // occupied: matches us?
            if (rd_u64m(e) == key) return bref;     // hit (block or ZERO_REGION)
            i = (i + 1) & mask; continue;           // collision -> next bucket
        }
        // empty bucket
        if (!create) return 0;
        // Two-phase claim — allocate ONLY after winning, so a lost race wastes NO
        // disk block (the previous design pre-allocated a candidate block before the
        // CAS and orphaned it on a loss, bloating the file nondeterministically).
        // Phase 1: write key, then CAS-publish the transient CLAIMED sentinel.
        wr_u64m(e, key);                            // key visible before publish
        u64 expect = 0;
        if (!atomic_compare_exchange_strong_explicit(atomptr(e+8), &expect,
                VC_CLAIMED_REGION, memory_order_release, memory_order_acquire)) {
            // Lost: someone else published here first. If our key, resolve it
            // (spinning out any CLAIMED sentinel); else probe onward.
            if (rd_u64m(e) == key) {
                while (expect == VC_CLAIMED_REGION)
                    expect = atomic_load_explicit(atomptr(e+8), memory_order_acquire);
                return expect;
            }
            i = (i + 1) & mask; continue;
        }
        // Phase 2: we own the bucket. Allocate the block (advances EOF exactly once
        // for this region, for the lifetime of the archive), zero it, publish.
        u64 nb = writer_alloc(w, L2_BLOCK_BYTES, 0);
        if (nb == 0 || writer_ensure(w, nb + L2_BLOCK_BYTES) != 0) return 0;
        memset(w->map + nb, 0, L2_BLOCK_BYTES);
        // writer_ensure may have remapped (non-fixed_map mode) -> recompute e from
        // the current base before publishing through it.
        e = w->map + off + i*L1_ENTRY;
        atomic_store_explicit(atomptr(e+8), nb, memory_order_release);
        atomic_fetch_add_explicit((_Atomic u64*)(w->map+FH_L1CNT), 1, memory_order_relaxed);
        return nb;
    }
}

// Legacy locked find-or-create (superseded by region_block_lockfree; retained
// for reference and potential single-threaded setup paths).
__attribute__((unused))
static u64 region_block(vc_writer *w, u64 key, int create, int *created) {
    if (created) *created = 0;
    u64 cap = rd_u64m(w->map + FH_L1CAP);
    u64 off = rd_u64m(w->map + FH_L1OFF);
    u64 i = l1_probe(w->map, off, cap, key);
    u64 bref = rd_u64m(w->map + off + i*L1_ENTRY + 8);
    if (bref != 0) return bref;          // existing (block or ZERO_REGION)
    if (!create) return 0;
    // grow if this insert would exceed load factor (may remap + move w->map).
    u64 cnt = rd_u64m(w->map + FH_L1CNT);
    if ((cnt + 1) * L1_MAX_LOAD_D >= cap * L1_MAX_LOAD_N) {
        if (l1_grow(w) != 0) return 0;
    }
    // Allocate the L2 block (this MAY remap and move w->map), THEN recompute the
    // bucket from the current map/offset/cap before writing — never hold a raw
    // pointer across an allocation.
    u64 blk = writer_alloc(w, L2_BLOCK_BYTES, 1);
    if (writer_ensure(w, blk + L2_BLOCK_BYTES) != 0) return 0;
    memset(w->map + blk, 0, L2_BLOCK_BYTES);   // all slots AF_ABSENT
    cap = rd_u64m(w->map + FH_L1CAP);
    off = rd_u64m(w->map + FH_L1OFF);
    i = l1_probe(w->map, off, cap, key);
    u8 *e = w->map + off + i*L1_ENTRY;
    wr_u64m(e, key); wr_u64m(e + 8, blk);
    wr_u64m(w->map + FH_L1CNT, cnt + 1);
    if (created) *created = 1;
    return blk;
}

static inline u64 slot_index(u32 az, u32 ay, u32 ax) {
    u32 lz = az & (R_ATOMS-1), ly = ay & (R_ATOMS-1), lx = ax & (R_ATOMS-1);
    return ((u64)lz * R_ATOMS + ly) * R_ATOMS + lx;
}


vc_writer *vc_create(const char *path, vc_dims lod0_dims, float target_ratio) {
    build_tables();
    if (!dims_valid(lod0_dims) || target_ratio < 1.0f) return NULL;
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NULL;

    int nlod = 1;
    for (int l = 1; l < VC_NLOD; ++l) {
        vc_dims d = lod_dims_of(lod0_dims, l);
        nlod = l + 1;
        if (d.nx <= 1 && d.ny <= 1 && d.nz <= 1) break;
    }

    // Preallocate the L1 hash table large enough that it NEVER rehashes — this is
    // what makes region lookup/creation lock-free (a moving table would need a
    // lock). Max regions = sum over LODs of the region grid; size to ~2x that,
    // rounded up to a power of two (load factor < 0.5).
    u64 max_regions = 0;
    for (int l = 0; l < nlod; ++l) {
        vc_dims d = lod_dims_of(lod0_dims, l);
        u64 rx = ((d.nx + A - 1)/A + R_ATOMS - 1)/R_ATOMS;
        u64 ry = ((d.ny + A - 1)/A + R_ATOMS - 1)/R_ATOMS;
        u64 rz = ((d.nz + A - 1)/A + R_ATOMS - 1)/R_ATOMS;
        max_regions += rx*ry*rz;
    }
    u64 l1_cap = L1_INIT_CAP;
    while (l1_cap < max_regions * 2) l1_cap <<= 1;   // power-of-two, <50% load

    u64 l1_off = FILE_HDR;
    u64 init_len = l1_off + l1_cap * L1_ENTRY;
    if (ftruncate(fd, (off_t)init_len) != 0) { close(fd); return NULL; }

    // Prefer a fixed huge virtual reservation (base never moves -> lock-free
    // reads, no remap). MAP_NORESERVE keeps unwritten pages free. Fall back to a
    // plain right-sized map (remap-on-grow) if the big reservation fails (macOS
    // or constrained address space).
    int fixed = 0; u64 map_len = init_len;
    u8 *map = MAP_FAILED;
#if defined(__linux__)
    map = (u8 *)mmap(NULL, VC_RESERVE, PROT_READ|PROT_WRITE,
                     MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (map != MAP_FAILED) { fixed = 1; map_len = VC_RESERVE; }
#endif
    if (map == MAP_FAILED) {
        map = (u8 *)mmap(NULL, init_len, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { close(fd); return NULL; }
        fixed = 0; map_len = init_len;
    }
    memset(map, 0, init_len);

    wr_u32m(map+FH_MAGIC, VC_MAGIC);
    wr_u32m(map+FH_VER, VC_VERSION);
    wr_u32m(map+FH_ATOM, VC_ATOM);
    wr_u32m(map+FH_LAYOUT, VC_LAYOUT_SPARSE2);
    wr_u32m(map+FH_RGN, R_ATOMS);
    wr_u32m(map+FH_NLOD, (u32)nlod);
    wr_u32m(map+FH_NX, lod0_dims.nx);
    wr_u32m(map+FH_NY, lod0_dims.ny);
    wr_u32m(map+FH_NZ, lod0_dims.nz);
    wr_f32m(map+FH_TRATIO, target_ratio);
    wr_u64m(map+FH_L1CAP, l1_cap);
    wr_u64m(map+FH_L1CNT, 0);
    wr_u64m(map+FH_L1OFF, l1_off);
    wr_u64m(map+FH_CURSOR, init_len);

    vc_writer *w = (vc_writer *)vc_xcalloc(1, sizeof(*w));
    w->fd = fd; w->map = map; w->map_len = map_len; w->file_len = init_len;
    w->fixed_map = fixed;
    pthread_rwlock_init(&w->lock, NULL);
    pthread_mutex_init(&w->grow_mu, NULL);
    atomic_store(&w->cursor, init_len);
    w->dims0 = lod0_dims; w->nlod = nlod; w->target_ratio = target_ratio;
    return w;
}

vc_status vc_set_base_q(vc_writer *w, int lod, float q) {
    if (!w || lod < 0 || lod >= w->nlod) return VC_ERR_RANGE;
    if (q < 0.05f) q = 0.05f;
    if (q > 4096.0f) q = 4096.0f;
    pthread_rwlock_wrlock(&w->lock);
    wr_f32m(w->map + FH_BASEQ + lod*4, q);
    atomic_store_explicit(&w->base_q[lod], q, memory_order_release);
    pthread_rwlock_unlock(&w->lock);
    return VC_OK;
}


static vc_status append_one(vc_writer *w, int lod, u32 az, u32 ay, u32 ax,
                            const u8 vox[VC_ATOM3]) {
    u32 rz = az / R_ATOMS, ry = ay / R_ATOMS, rx = ax / R_ATOMS;
    u64 key = region_key(lod, rz, ry, rx);

    // Detect all-zero (lock-free, pure). An all-zero atom stores as AF_ZERO.
    int is_zero = 1;
    for (u32 i = 0; i < A3; ++i) if (vox[i]) { is_zero = 0; break; }

    // ENCODE OUTSIDE THE LOCK so appends run in parallel (the encode is the
    // expensive part and is a pure function of vox + base_q). base_q[lod] is
    // frozen from the first non-zero atom of the lod: if not yet frozen, take the
    // exclusive lock briefly to do the warmup probe + freeze, then release and
    // encode lock-free. Once frozen, all threads read it lock-free.
    bbuf pay = {0}; u8 dcv = 0; u32 len = 0;
    if (!is_zero) {
        // q must have been set explicitly via vc_set_base_q before appending.
        float q = atomic_load_explicit(&w->base_q[lod], memory_order_acquire);
        if (q <= 0.0f) return VC_ERR_RANGE;   // no q set for this LOD
        int z2 = 0;
        len = encode_atom(vox, q, &pay, &z2, &dcv);
    }

    // Ensure the region's L2 block exists. region_block mutates the L1 hash
    // table, so it runs under the exclusive lock (rare: only on first touch of a
    // region / rehash). It internally reserves+zeros the block via writer_alloc
    // (which grows the file under grow_mu, no remap on the fixed map).
    // LOCK-FREE region lookup/creation (the L1 table never rehashes — it is
    // preallocated from the volume dims). No lock on the common path.
    u64 blk = region_block_lockfree(w, key, 1);
    if (blk == VC_ZERO_REGION) {
        // Rare: region was declared all-zero, now an atom arrives -> promote to a
        // real block. This mutates an existing bucket, so take the lock.
        pthread_rwlock_wrlock(&w->lock);
        u64 cap = rd_u64m(w->map + FH_L1CAP), off = rd_u64m(w->map + FH_L1OFF);
        u64 i = l1_probe(w->map, off, cap, key);
        u8 *e = w->map + off + i*L1_ENTRY;
        u64 cur = rd_u64m(e+8);
        if (cur == VC_ZERO_REGION) {
            u64 nb = writer_alloc(w, L2_BLOCK_BYTES, 1);
            if (nb == 0 || writer_ensure(w, nb + L2_BLOCK_BYTES) != 0) { pthread_rwlock_unlock(&w->lock); free(pay.p); return VC_ERR_OOM; }
            memset(w->map + nb, 0, L2_BLOCK_BYTES);
            wr_u64m(e + 8, nb);
            blk = nb;
        } else blk = cur;   // another thread promoted it first
        pthread_rwlock_unlock(&w->lock);
    }
    if (blk == 0) { free(pay.p); return VC_ERR_OOM; }

    u64 slot_off = blk + slot_index(az,ay,ax)*ATOM_SLOT;

    if (is_zero) {
        // known-zero atom: no payload. publish flags as the commit point.
        u8 *slot = w->map + slot_off;
        wr_u64m(slot, ABSENT); wr_u32m(slot+8, 0); slot[13] = 0;
        atomic_thread_fence(memory_order_release);
        slot[12] = AF_ZERO;
        free(pay.p);
        return VC_OK;
    }

    // Reserve a unique EOF range (atomic cursor) and copy the payload there with
    // NO lock — on the fixed map the base is stable and the range is exclusively
    // ours. Then publish into our slot (a distinct slot per atom, so concurrent
    // appends to the same block don't conflict). `offset` is the release-
    // published commit word; readers acquire-load it.
    u64 poff = writer_alloc(w, len, 0);
    if (poff == 0 || writer_ensure(w, poff + len) != 0) { free(pay.p); return VC_ERR_OOM; }
    if (!w->fixed_map) pthread_rwlock_rdlock(&w->lock);   // remap mode: pin base
    memcpy(w->map + poff, pay.p, len);
    u8 *slot = w->map + slot_off;
    wr_u32m(slot+8, len); slot[12] = AF_PRESENT; slot[13] = dcv;
    atomic_thread_fence(memory_order_release);
    wr_u64m(slot, poff);   // RELEASE-published commit word
    if (!w->fixed_map) pthread_rwlock_unlock(&w->lock);
    free(pay.p);
    return VC_OK;
}

vc_status vc_append_atom(vc_writer *w, int lod, u32 az, u32 ay, u32 ax,
                         const u8 vox[VC_ATOM3]) {
    if (!w || lod < 0 || lod >= w->nlod) return VC_ERR_RANGE;
    vc_dims d = lod_dims_of(w->dims0, lod);
    u32 acx=(d.nx+A-1)/A, acy=(d.ny+A-1)/A, acz=(d.nz+A-1)/A;
    if (ax >= acx || ay >= acy || az >= acz) return VC_ERR_RANGE;
    return append_one(w, lod, az, ay, ax, vox);
}

vc_status vc_append_box(vc_writer *w, int lod, vc_box b, const u8 *voxels) {
    if (!w || lod < 0 || lod >= w->nlod || !voxels) return VC_ERR_RANGE;
    if (b.x0 % A || b.y0 % A || b.z0 % A) return VC_ERR_RANGE;
    if ((b.x1-b.x0) % A || (b.y1-b.y0) % A || (b.z1-b.z0) % A) return VC_ERR_RANGE;
    if (b.x1<=b.x0 || b.y1<=b.y0 || b.z1<=b.z0) return VC_ERR_RANGE;
    u32 bx = b.x1-b.x0, by = b.y1-b.y0;
    u8 atom[A3];
    for (u32 vz = b.z0; vz < b.z1; vz += A)
    for (u32 vy = b.y0; vy < b.y1; vy += A)
    for (u32 vx = b.x0; vx < b.x1; vx += A) {
        // gather one 32^3 atom out of the box buffer
        for (u32 z=0; z<A; ++z) for (u32 y=0; y<A; ++y) for (u32 x=0; x<A; ++x) {
            u64 si = ((u64)(vz - b.z0 + z)*by + (vy - b.y0 + y))*bx + (vx - b.x0 + x);
            atom[(z*A+y)*A+x] = voxels[si];
        }
        vc_status s = append_one(w, lod, vz/A, vy/A, vx/A, atom);
        if (s != VC_OK) return s;
    }
    return VC_OK;
}

vc_status vc_mark_zero_atom(vc_writer *w, int lod, u32 az, u32 ay, u32 ax) {
    if (!w || lod < 0 || lod >= w->nlod) return VC_ERR_RANGE;
    u64 key = region_key(lod, az/R_ATOMS, ay/R_ATOMS, ax/R_ATOMS);
    // Lock-free find-or-create, then write the explicit AF_ZERO slot. Creating
    // the block guarantees this atom is recorded as KNOWN_ZERO (never left ABSENT)
    // regardless of the order present/zero atoms arrive in the region.
    u64 blk = region_block_lockfree(w, key, 1 /*create*/);
    if (blk == 0) return VC_ERR_OOM;
    if (blk == VC_ZERO_REGION) return VC_OK;   // whole region already known-zero
    u8 *slot = w->map + blk + slot_index(az,ay,ax)*ATOM_SLOT;
    wr_u64m(slot, ABSENT); wr_u32m(slot+8, 0); slot[13]=0;
    atomic_thread_fence(memory_order_release);
    slot[12] = AF_ZERO;
    return VC_OK;
}

vc_status vc_mark_zero_region(vc_writer *w, int lod, u32 rz, u32 ry, u32 rx) {
    if (!w || lod < 0 || lod >= w->nlod) return VC_ERR_RANGE;
    u64 key = region_key(lod, rz, ry, rx);
    // Lock-free: CAS the ZERO_REGION sentinel into an empty bucket (preallocated
    // L1, never rehashes). If the region already has a block, leave it (write-
    // once — a populated region is not downgraded).
    u64 cap = rd_u64m(w->map + FH_L1CAP), off = rd_u64m(w->map + FH_L1OFF);
    u64 mask = cap - 1, i = mix64(key) & mask;
    for (;;) {
        u8 *e = w->map + off + i*L1_ENTRY;
        u64 bref = atomic_load_explicit(atomptr(e+8), memory_order_acquire);
        if (bref != 0) { if (rd_u64m(e) == key) return VC_OK; i = (i+1)&mask; continue; }
        wr_u64m(e, key);
        u64 expect = 0;
        if (atomic_compare_exchange_strong_explicit(atomptr(e+8), &expect, VC_ZERO_REGION,
                memory_order_release, memory_order_acquire)) {
            atomic_fetch_add_explicit((_Atomic u64*)(w->map+FH_L1CNT), 1, memory_order_relaxed);
            return VC_OK;
        }
        if (rd_u64m(e) == key) return VC_OK;   // someone else claimed it for us
        i = (i+1)&mask;
    }
}

void vc_writer_close(vc_writer *w) {
    if (!w) return;
    // trim the file to the exact used length, persist final cursor.
    u64 used = atomic_load(&w->cursor);
    wr_u64m(w->map + FH_CURSOR, used);
    // sync only the backed prefix (map_len may be a 10TB reservation).
    msync(w->map, w->file_len, MS_SYNC);
    munmap(w->map, w->map_len);
    if (used && used < w->file_len) { if (ftruncate(w->fd, (off_t)used) != 0) {/*ignore*/} }
    close(w->fd);
    pthread_rwlock_destroy(&w->lock);
    pthread_mutex_destroy(&w->grow_mu);
    free(w);
}

// ===========================================================================
// Reader
// ===========================================================================
struct vc_archive {
    const u8 *buf; size_t len;
    vc_dims dims0; int nlod;
    u64 l1_off, l1_cap;
    float base_q[VC_NLOD];
};

vc_archive *vc_open(const u8 *archive, size_t len) {
    build_tables();
    if (!archive || len < FILE_HDR) return NULL;
    if (rd_u32m(archive+FH_MAGIC) != VC_MAGIC) return NULL;
    if (rd_u32m(archive+FH_VER) != VC_VERSION) return NULL;
    if (rd_u32m(archive+FH_ATOM) != VC_ATOM) return NULL;
    if (rd_u32m(archive+FH_LAYOUT) != VC_LAYOUT_SPARSE2) return NULL;
    if (rd_u32m(archive+FH_RGN) != R_ATOMS) return NULL;
    u32 nlod = rd_u32m(archive+FH_NLOD);
    if (nlod == 0 || nlod > VC_NLOD) return NULL;
    u64 l1_off = rd_u64m(archive+FH_L1OFF);
    u64 l1_cap = rd_u64m(archive+FH_L1CAP);
    if (l1_cap == 0 || (l1_cap & (l1_cap-1))) return NULL;   // power of two
    if (l1_off > len || l1_cap > (len - l1_off) / L1_ENTRY) return NULL;
    vc_archive *a = (vc_archive *)vc_xcalloc(1, sizeof(*a));
    a->buf = archive; a->len = len; a->nlod = (int)nlod;
    a->dims0.nx = rd_u32m(archive+FH_NX);
    a->dims0.ny = rd_u32m(archive+FH_NY);
    a->dims0.nz = rd_u32m(archive+FH_NZ);
    a->l1_off = l1_off; a->l1_cap = l1_cap;
    for (int l=0;l<VC_NLOD;++l) a->base_q[l] = rd_f32m(archive+FH_BASEQ+l*4);
    return a;
}
void vc_close(vc_archive *a) { free(a); }

vc_status vc_lod_dims(const vc_archive *a, int lod, vc_dims *out) {
    if (!a || lod < 0 || lod >= a->nlod || !out) return VC_ERR_RANGE;
    *out = lod_dims_of(a->dims0, lod);
    return VC_OK;
}

// Look up the L2 block ref for a region (read-only). 0 = absent, else block_ref
// (possibly VC_ZERO_REGION).
static u64 reader_block(const vc_archive *a, u64 key) {
    u64 mask = a->l1_cap - 1;
    u64 i = mix64(key) & mask;
    for (;;) {
        const u8 *e = a->buf + a->l1_off + i*L1_ENTRY;
        u64 bref = rd_u64m(e + 8);
        if (bref == 0) return 0;
        if (rd_u64m(e) == key) return bref;
        i = (i + 1) & mask;
    }
}

// Resolve an atom's slot: returns flags, and (offset,len,dc) for PRESENT.
static u8 resolve_atom(const vc_archive *a, int lod, u32 az, u32 ay, u32 ax,
                       u64 *off, u32 *len, u8 *dc) {
    u64 key = region_key(lod, az/R_ATOMS, ay/R_ATOMS, ax/R_ATOMS);
    u64 blk = reader_block(a, key);
    if (blk == 0) return AF_ABSENT;
    if (blk == VC_ZERO_REGION) return AF_ZERO;
    if (blk + L2_BLOCK_BYTES > a->len) return AF_ABSENT; // corrupt guard
    const u8 *slot = a->buf + blk + slot_index(az,ay,ax)*ATOM_SLOT;
    // ACQUIRE-load the commit word (offset) before trusting the other fields.
    u64 o = rd_u64m(slot);
    atomic_thread_fence(memory_order_acquire);
    u8 flags = slot[12];
    if (flags == AF_PRESENT) { *off = o; *len = rd_u32m(slot+8); *dc = slot[13]; }
    return flags;
}

vc_status vc_decode_atom(vc_archive *a, int lod, int ax, int ay, int az,
                         u8 out[VC_ATOM3]) {
    if (!a || lod < 0 || lod >= a->nlod || !out) return VC_ERR_RANGE;
    vc_dims d = lod_dims_of(a->dims0, lod);
    u32 acx=(d.nx+A-1)/A, acy=(d.ny+A-1)/A, acz=(d.nz+A-1)/A;
    if ((u32)ax>=acx || (u32)ay>=acy || (u32)az>=acz) return VC_ERR_RANGE;
    u64 off=0; u32 len=0; u8 dc=0;
    u8 flags = resolve_atom(a, lod, az, ay, ax, &off, &len, &dc);
    if (flags == AF_PRESENT) {
        if (off == ABSENT || len < 1 || off + len > a->len) { memset(out,0,A3); return VC_ERR_FORMAT; }
        decode_atom_payload(a->buf + off, len, a->base_q[lod], out);
        return VC_OK;
    }
    memset(out, 0, A3);   // ABSENT or ZERO -> zeros
    return VC_OK;
}

vc_cover vc_atom_coverage(const vc_archive *a, int lod, u32 az, u32 ay, u32 ax) {
    if (!a || lod < 0 || lod >= a->nlod) return VC_ABSENT;
    u64 off; u32 len; u8 dc;
    u8 flags = resolve_atom(a, lod, az, ay, ax, &off, &len, &dc);
    if (flags == AF_PRESENT) return VC_PRESENT;
    if (flags == AF_ZERO)    return VC_KNOWN_ZERO;
    return VC_ABSENT;
}

// Build a transient reader view over the writer's live map. Caller must hold the
// writer's lock (shared is enough) so the map can't move underfoot.
static vc_archive writer_view(vc_writer *w) {
    vc_archive a;
    a.buf = w->map; a.len = w->map_len; a.dims0 = w->dims0; a.nlod = w->nlod;
    a.l1_off = rd_u64m(w->map + FH_L1OFF);
    a.l1_cap = rd_u64m(w->map + FH_L1CAP);
    for (int l=0;l<VC_NLOD;++l) a.base_q[l] = atomic_load_explicit(&w->base_q[l], memory_order_acquire);
    return a;
}

// Decode an already-written atom back from the writer (used to cascade the LOD
// pyramid: read a lower LOD, downscale, write the next). Reads under the shared
// lock so concurrent appends (which may remap) are safe.
vc_status vc_writer_decode_atom(vc_writer *w, int lod, u32 az, u32 ay, u32 ax,
                                u8 out[VC_ATOM3]) {
    if (!w || lod < 0 || lod >= w->nlod || !out) return VC_ERR_RANGE;
    pthread_rwlock_rdlock(&w->lock);
    vc_archive a = writer_view(w);
    vc_status s = vc_decode_atom(&a, lod, (int)ax, (int)ay, (int)az, out);
    pthread_rwlock_unlock(&w->lock);
    return s;
}

// Coverage of an already-written atom in the writer (PRESENT/KNOWN_ZERO/ABSENT).
vc_cover vc_writer_coverage(vc_writer *w, int lod, u32 az, u32 ay, u32 ax) {
    if (!w || lod < 0 || lod >= w->nlod) return VC_ABSENT;
    pthread_rwlock_rdlock(&w->lock);
    vc_archive a = writer_view(w);
    vc_cover c = vc_atom_coverage(&a, lod, az, ay, ax);
    pthread_rwlock_unlock(&w->lock);
    return c;
}

// Batched decode of a 2x2x2 block of source atoms with the base coords
// (az0,ay0,ax0) (must be even). Takes the shared lock ONCE and decodes all 8
// (vs 8-16 lock ops), which is what makes the cascade scale. `out8` is 8 atoms
// in (dz,dy,dx) order; absent/zero atoms are zeroed. *any set if any present.
vc_status vc_writer_decode_2x2x2(vc_writer *w, int lod, u32 az0, u32 ay0, u32 ax0,
                                 u8 out8[8][VC_ATOM3], int *any) {
    if (!w || lod < 0 || lod >= w->nlod || !out8) return VC_ERR_RANGE;
    int a_any = 0;
    pthread_rwlock_rdlock(&w->lock);
    vc_archive a = writer_view(w);
    vc_dims d = lod_dims_of(a.dims0, lod);
    u32 acx=(d.nx+A-1)/A, acy=(d.ny+A-1)/A, acz=(d.nz+A-1)/A;
    for (int dz=0; dz<2; ++dz) for (int dy=0; dy<2; ++dy) for (int dx=0; dx<2; ++dx) {
        int idx = (dz*2+dy)*2+dx;
        u32 az=az0+dz, ay=ay0+dy, ax=ax0+dx;
        u8 *o = out8[idx];
        if (az>=acz||ay>=acy||ax>=acx) { memset(o,0,A3); continue; }
        u64 off=0; u32 len=0; u8 dc=0;
        u8 fl = resolve_atom(&a, lod, az, ay, ax, &off, &len, &dc);
        if (fl == AF_PRESENT && off!=ABSENT && len>=1 && off+len<=a.len) {
            decode_atom_payload(a.buf+off, len, a.base_q[lod], o); a_any=1;
        } else memset(o,0,A3);  // ABSENT/ZERO -> zeros
    }
    pthread_rwlock_unlock(&w->lock);
    if (any) *any = a_any;
    return VC_OK;
}

vc_status vc_decode_region(vc_archive *a, int lod, vc_box box, u8 *out) {
    if (!a || lod < 0 || lod >= a->nlod || !out) return VC_ERR_RANGE;
    vc_dims d = lod_dims_of(a->dims0, lod);
    if (box.x1 > d.nx || box.y1 > d.ny || box.z1 > d.nz) return VC_ERR_RANGE;
    if (box.x0 >= box.x1 || box.y0 >= box.y1 || box.z0 >= box.z1) return VC_ERR_RANGE;
    u32 ox = box.x1-box.x0, oy = box.y1-box.y0;
    u8 atom[A3];
    u32 ax0=box.x0/A, ax1=(box.x1-1)/A, ay0=box.y0/A, ay1=(box.y1-1)/A, az0=box.z0/A, az1=(box.z1-1)/A;
    for (u32 az=az0; az<=az1; ++az)
    for (u32 ay=ay0; ay<=ay1; ++ay)
    for (u32 ax=ax0; ax<=ax1; ++ax) {
        vc_status s = vc_decode_atom(a, lod, (int)ax,(int)ay,(int)az, atom);
        if (s != VC_OK) return s;
        for (u32 z=0; z<A; ++z){ u32 vz=az*A+z; if(vz<box.z0||vz>=box.z1) continue;
        for (u32 y=0; y<A; ++y){ u32 vy=ay*A+y; if(vy<box.y0||vy>=box.y1) continue;
        for (u32 x=0; x<A; ++x){ u32 vx=ax*A+x; if(vx<box.x0||vx>=box.x1) continue;
            out[((size_t)(vz-box.z0)*oy + (vy-box.y0))*ox + (vx-box.x0)] = atom[(z*A+y)*A+x];
        }}}
    }
    return VC_OK;
}

// ===========================================================================
// Test-only helpers (not part of the public API; used by tests to form LODs).
// ===========================================================================
#ifdef VC_TESTHOOKS
u8 *vc_test_downsample2x(const u8 *vol, vc_dims in, vc_dims *out);
u8 *vc_test_downsample2x(const u8 *vol, vc_dims in, vc_dims *out) {
    u32 ox=(in.nx+1)/2, oy=(in.ny+1)/2, oz=(in.nz+1)/2;
    if(!ox)ox=1;
    if(!oy)oy=1;
    if(!oz)oz=1;
    u8 *o=(u8*)vc_xmalloc((size_t)ox*oy*oz);
    for(u32 z=0;z<oz;++z)for(u32 y=0;y<oy;++y)for(u32 x=0;x<ox;++x){
        u32 x0=x*2,y0=y*2,z0=z*2,acc=0,n=0;
        for(u32 dz=0;dz<2;++dz){u32 zz=z0+dz;if(zz>=in.nz)continue;
        for(u32 dy=0;dy<2;++dy){u32 yy=y0+dy;if(yy>=in.ny)continue;
        for(u32 dx=0;dx<2;++dx){u32 xx=x0+dx;if(xx>=in.nx)continue;
            acc+=vol[((size_t)zz*in.ny+yy)*in.nx+xx];n++;}}}
        o[((size_t)z*oy+y)*ox+x]=(u8)((acc+n/2)/(n?n:1));
    }
    out->nx=ox;out->ny=oy;out->nz=oz; return o;
}

// Encode one 32^3 atom at an explicit q, return payload bytes, and (optionally)
// reconstruct it into `recon`. For calibration sweeps / floor analysis.
u32 vc_test_atom_rt(const u8 vox[VC_ATOM3], float q, u8 *recon);
u32 vc_test_atom_rt(const u8 vox[VC_ATOM3], float q, u8 *recon) {
    build_tables();
    bbuf pay = {0}; int z=0; u8 dc=0;
    u32 len = encode_atom(vox, q, &pay, &z, &dc);
    if (z) { if (recon) memset(recon, 0, A3); free(pay.p); return 0; }
    if (recon) decode_atom_payload(pay.p, len, q, recon);
    free(pay.p);
    return len;
}
#endif
