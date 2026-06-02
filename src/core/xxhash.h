// XXH64 (Yann Collet) single-shot — fast non-cryptographic integrity hash for
// per-chunk payloads and the archive directory/index (PLAN §3). Pure C23.
#ifndef VC_XXHASH_H
#define VC_XXHASH_H

#include "../../include/vc/types.h"
#include <string.h>

static inline u64 vc_rotl64(u64 v, int r) { return (v << r) | (v >> (64 - r)); }

static inline u64 vc_xxh64(const void *data, size_t len, u64 seed) {
    static const u64 P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full,
                     P3 = 0x165667B19E3779F9ull, P4 = 0x85EBCA77C2B2AE63ull,
                     P5 = 0x27D4EB2F165667C5ull;
    const u8 *p = (const u8 *)data;
    const u8 *end = p + len;
    u64 h;
    if (len >= 32) {
        u64 v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1;
        const u8 *limit = end - 32;
        do {
            u64 k;
            memcpy(&k, p, 8); p += 8; v1 += k * P2; v1 = vc_rotl64(v1, 31); v1 *= P1;
            memcpy(&k, p, 8); p += 8; v2 += k * P2; v2 = vc_rotl64(v2, 31); v2 *= P1;
            memcpy(&k, p, 8); p += 8; v3 += k * P2; v3 = vc_rotl64(v3, 31); v3 *= P1;
            memcpy(&k, p, 8); p += 8; v4 += k * P2; v4 = vc_rotl64(v4, 31); v4 *= P1;
        } while (p <= limit);
        h = vc_rotl64(v1, 1) + vc_rotl64(v2, 7) + vc_rotl64(v3, 12) + vc_rotl64(v4, 18);
        #define VC_XXH_MERGE(acc, v) do { u64 vv = (v); vv *= P2; vv = vc_rotl64(vv,31); vv *= P1; \
            acc ^= vv; acc = acc * P1 + P4; } while (0)
        VC_XXH_MERGE(h, v1); VC_XXH_MERGE(h, v2); VC_XXH_MERGE(h, v3); VC_XXH_MERGE(h, v4);
        #undef VC_XXH_MERGE
    } else {
        h = seed + P5;
    }
    h += (u64)len;
    while (p + 8 <= end) {
        u64 k; memcpy(&k, p, 8); p += 8;
        k *= P2; k = vc_rotl64(k, 31); k *= P1;
        h ^= k; h = vc_rotl64(h, 27) * P1 + P4;
    }
    if (p + 4 <= end) {
        u32 k; memcpy(&k, p, 4); p += 4;
        h ^= (u64)k * P1; h = vc_rotl64(h, 23) * P2 + P3;
    }
    while (p < end) { h ^= (u64)(*p++) * P5; h = vc_rotl64(h, 11) * P1; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

#endif // VC_XXHASH_H
