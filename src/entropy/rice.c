// Golomb-Rice entropy coder over signed quantized levels (PLAN §3 Phase-0
// entropy). Signed levels are zigzag-mapped to unsigned, then Rice-coded with a
// block-adaptive parameter k (one k per RICE_BLOCK values, chosen to minimize
// that block's coded length). Block-adaptive k tracks the heavy spatial variance
// in DCT coefficients (DC-heavy low-freq blocks vs near-zero HF runs) far better
// than a single global k, at negligible overhead (4 bits per block).
//
// Stream layout (all via the LE bit IO):
//   per block of RICE_BLOCK values: [4-bit k][RICE_BLOCK Rice codes]
// The final partial block is shorter (n known to the decoder).
#include "../core/bitio.h"
#include "../../include/vc/types.h"

#define RICE_BLOCK 64u
#define KMAX 15u   // 4-bit k field

// Zigzag map signed->unsigned: 0,-1,1,-2,2 -> 0,1,2,3,4.
static inline u32 zigzag(i16 v) {
    i32 x = (i32)v;
    return (u32)((x << 1) ^ (x >> 31));
}
static inline i16 unzigzag(u32 u) {
    i32 x = (i32)((u >> 1) ^ (~(u & 1u) + 1u));
    return (i16)x;
}

// Cost in bits of Rice-coding value u with parameter k: (u>>k) + 1 + k.
static inline u32 rice_cost(u32 u, u32 k) { return (u >> k) + 1u + k; }

// Choose the best k for a block of `cnt` zigzagged values by direct cost eval.
static inline u32 best_k(const u32 *restrict zz, u32 cnt) {
    // Seed near log2(mean) then scan a small window — but a full 0..KMAX scan is
    // only 16*cnt adds and stays autovectorizable; keep it simple and exact.
    u32 bestk = 0; u64 bestbits = (u64)-1;
    for (u32 k = 0; k <= KMAX; ++k) {
        u64 bits = 0;
        for (u32 i = 0; i < cnt; ++i) bits += rice_cost(zz[i], k);
        if (bits < bestbits) { bestbits = bits; bestk = k; }
    }
    return bestk;
}

size_t vc_rice_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n) {
    vc_bitwriter w; vc_bw_init(&w, out, cap);
    u32 zz[RICE_BLOCK];
    for (size_t base = 0; base < n; base += RICE_BLOCK) {
        u32 cnt = (n - base) < RICE_BLOCK ? (u32)(n - base) : RICE_BLOCK;
        for (u32 i = 0; i < cnt; ++i) zz[i] = zigzag(q[base + i]);
        u32 k = best_k(zz, cnt);
        vc_bw_put(&w, k, 4);
        for (u32 i = 0; i < cnt; ++i) {
            u32 u = zz[i];
            vc_bw_put_unary(&w, u >> k);   // quotient (unary)
            if (k) vc_bw_put(&w, u & ((1u << k) - 1u), k); // remainder
        }
        if (w.overflow) return cap + 1; // signal overflow
    }
    return vc_bw_finish(&w);
}

void vc_rice_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len) {
    vc_bitreader r; vc_br_init(&r, in, len);
    for (size_t base = 0; base < n; base += RICE_BLOCK) {
        u32 cnt = (n - base) < RICE_BLOCK ? (u32)(n - base) : RICE_BLOCK;
        u32 k = vc_br_get(&r, 4);
        for (u32 i = 0; i < cnt; ++i) {
            u32 hi = vc_br_get_unary(&r);
            u32 lo = k ? vc_br_get(&r, k) : 0u;
            u32 u = (hi << k) | lo;
            q[base + i] = unzigzag(u);
        }
    }
}
