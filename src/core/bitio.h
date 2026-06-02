// Little-endian bit/byte IO (PLAN §3 core). Writer appends bits LSB-first into
// bytes; reader pulls them back in the same order. Used by the Rice coder.
#ifndef VC_BITIO_H
#define VC_BITIO_H

#include "../../include/vc/types.h"
#include <string.h>

// --- Bit writer ------------------------------------------------------------
// Appends into a caller-provided buffer of `cap` bytes. Sets `overflow` if it
// would exceed cap (caller must size generously; the codec falls back on overflow).
typedef struct {
    u8 *buf;
    size_t cap;
    size_t byte;     // next byte index
    u32    acc;      // bit accumulator (LSB-first)
    u32    nbits;    // valid bits in acc
    int    overflow;
} vc_bitwriter;

static inline void vc_bw_init(vc_bitwriter *w, u8 *buf, size_t cap) {
    w->buf = buf; w->cap = cap; w->byte = 0; w->acc = 0; w->nbits = 0; w->overflow = 0;
}

// Write the low `n` bits of `v` (n in 0..24 to keep acc within 32 bits safely).
static inline void vc_bw_put(vc_bitwriter *w, u32 v, u32 n) {
    if (n == 0) return;
    w->acc |= (v & ((n >= 32) ? 0xffffffffu : ((1u << n) - 1u))) << w->nbits;
    w->nbits += n;
    while (w->nbits >= 8) {
        if (w->byte >= w->cap) { w->overflow = 1; w->nbits = 0; w->acc = 0; return; }
        w->buf[w->byte++] = (u8)(w->acc & 0xff);
        w->acc >>= 8;
        w->nbits -= 8;
    }
}

// Write `n` unary "1" bits followed by a terminating "0" (used for Rice quotient).
static inline void vc_bw_put_unary(vc_bitwriter *w, u32 n) {
    while (n >= 24) { vc_bw_put(w, 0xffffffu, 24); n -= 24; }
    if (n) vc_bw_put(w, (1u << n) - 1u, n);
    vc_bw_put(w, 0u, 1); // terminator
}

// Flush partial byte; returns total bytes written.
static inline size_t vc_bw_finish(vc_bitwriter *w) {
    if (w->nbits > 0) {
        if (w->byte >= w->cap) { w->overflow = 1; return w->byte; }
        w->buf[w->byte++] = (u8)(w->acc & 0xff);
        w->acc = 0; w->nbits = 0;
    }
    return w->byte;
}

// --- Bit reader ------------------------------------------------------------
typedef struct {
    const u8 *buf;
    size_t len;
    size_t byte;
    u32    acc;
    u32    nbits;
} vc_bitreader;

static inline void vc_br_init(vc_bitreader *r, const u8 *buf, size_t len) {
    r->buf = buf; r->len = len; r->byte = 0; r->acc = 0; r->nbits = 0;
}

static inline void vc_br_refill(vc_bitreader *r) {
    while (r->nbits <= 24 && r->byte < r->len) {
        r->acc |= (u32)r->buf[r->byte++] << r->nbits;
        r->nbits += 8;
    }
}

// Read `n` bits (n in 0..24).
static inline u32 vc_br_get(vc_bitreader *r, u32 n) {
    if (n == 0) return 0;
    if (r->nbits < n) vc_br_refill(r);
    u32 v = r->acc & ((n >= 32) ? 0xffffffffu : ((1u << n) - 1u));
    r->acc >>= n;
    r->nbits -= n;
    return v;
}

// Read a unary code: count "1" bits up to the terminating "0".
static inline u32 vc_br_get_unary(vc_bitreader *r) {
    u32 q = 0;
    for (;;) {
        if (r->nbits == 0) vc_br_refill(r);
        if (r->nbits == 0) return q; // ran out (corrupt / EOF guard)
        if (r->acc & 1u) { q++; r->acc >>= 1; r->nbits--; }
        else { r->acc >>= 1; r->nbits--; return q; }
    }
}

#endif // VC_BITIO_H
