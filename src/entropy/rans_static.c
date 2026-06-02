// Static rANS entropy coder (pure C23 port of the c4d reciprocal-multiply
// technique, see /home/forrest/c3d2/c4d/include/c4d/rans.hpp). Same
// VC_ENTROPY_ENC/DEC signature as rice.c; round-trips exactly.
//
// Pipeline (the JPEG-XL "symbol + bypass" split):
//   * Each signed level is zigzag-folded to a nonnegative u.
//   * Token alphabet of 256: tokens 0..254 code u directly; token 255 is an
//     ESCAPE — the true u is emitted into a separate raw bypass bitstream as a
//     LEB128-style varint (rANS gives uniform/rare mantissa data nothing, and
//     bit-packing is faster). This bounds the table at 256 entries and still
//     covers the full i16 range, while keeping long zero-runs (token 0) and the
//     small-coefficient mass exactly where the static histogram pays off.
//   * A per-chunk normalized 12-bit frequency table (sum == 4096) is built from
//     the token histogram and serialized; the decoder rebuilds cum/slot2sym.
//   * 32-bit rANS state, 16-bit renorm. The per-token integer divide is replaced
//     by a precomputed round-up reciprocal multiply via unsigned __int128.
//
// Stream layout (LE):
//   u32 token_byte_len           bytes of the rANS token stream
//   u8  freq[NSYM*2]             normalized 12-bit freqs (little-endian u16 each)
//   u8  tokens[token_byte_len]   rANS bytes (decoded front-to-back)
//   u8  bypass[...]              escape varints, LSB-first bit-packed (rest)
//
// rANS is inherently serial (single scalar state), so this stage is not claimed
// to vectorize; the histogram build IS a straight-line counting loop.
#include "../core/bitio.h"
#include "../../include/vc/types.h"
#include <string.h>

#define NSYM        256u
#define ESC         255u
#define PROB_BITS   12u
#define PROB_SCALE  (1u << PROB_BITS)   // 4096
#define RANS_L      (1u << 16)          // renorm lower bound
#define RCP_SH      48u

static inline u32 zz(i16 v) { i32 x = (i32)v; return (u32)((x << 1) ^ (x >> 31)); }
static inline i16 unzz(u32 u) { i32 x = (i32)((u >> 1) ^ (~(u & 1u) + 1u)); return (i16)x; }

// --- frequency table -------------------------------------------------------
typedef struct {
    u16 freq[NSYM];
    u16 cum[NSYM + 1];
    u16 slot2sym[PROB_SCALE];
    u64 rcp[NSYM];                       // round-up reciprocals (0 if freq==0)
} freq_table;

static void ft_finish(freq_table *t) {
    t->cum[0] = 0;
    for (u32 s = 0; s < NSYM; ++s) t->cum[s + 1] = (u16)(t->cum[s] + t->freq[s]);
    for (u32 s = 0; s < NSYM; ++s)
        for (u32 i = t->cum[s]; i < t->cum[s + 1]; ++i) t->slot2sym[i] = (u16)s;
    for (u32 s = 0; s < NSYM; ++s)
        t->rcp[s] = t->freq[s] ? ((u64)1 << RCP_SH) / t->freq[s] + 1 : 0;
}

// floor(x / freq[s]) without an integer divide.
static inline u32 ft_divf(const freq_table *t, u32 x, u32 s) {
    return (u32)(((unsigned __int128)x * t->rcp[s]) >> RCP_SH);
}

static void ft_build(freq_table *t, const u32 *counts) {
    u64 total = 0;
    for (u32 s = 0; s < NSYM; ++s) total += counts[s];
    memset(t->freq, 0, sizeof(t->freq));
    if (total == 0) {
        for (u32 s = 0; s < NSYM; ++s) t->freq[s] = 1; // degenerate
    } else {
        for (u32 s = 0; s < NSYM; ++s) {
            if (!counts[s]) continue;
            u32 f = (u32)(((u64)counts[s] * PROB_SCALE) / total);
            t->freq[s] = (u16)(f == 0 ? 1u : f);
        }
    }
    // fixup so freqs sum to PROB_SCALE (adjust the largest)
    u32 sum = 0; for (u32 s = 0; s < NSYM; ++s) sum += t->freq[s];
    if (sum != PROB_SCALE) {
        u32 best = 0; for (u32 s = 1; s < NSYM; ++s) if (t->freq[s] > t->freq[best]) best = s;
        t->freq[best] = (u16)((i32)t->freq[best] + ((i32)PROB_SCALE - (i32)sum));
    }
    ft_finish(t);
}

// --- bypass varint (LSB-first 7-bit groups, continuation bit high) ----------
static inline void bypass_put_varint(vc_bitwriter *w, u32 u) {
    do {
        u32 b = u & 0x7fu; u >>= 7;
        vc_bw_put(w, b | (u ? 0x80u : 0u), 8);
    } while (u);
}
static inline u32 bypass_get_varint(vc_bitreader *r) {
    u32 u = 0, sh = 0, b;
    do { b = vc_br_get(r, 8); u |= (b & 0x7fu) << sh; sh += 7; } while (b & 0x80u);
    return u;
}

size_t vc_rans_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n) {
    // 1) build token array + histogram.
    u32 counts[NSYM]; memset(counts, 0, sizeof(counts));
    // tokens stored transiently in a scratch within `out`'s tail is awkward;
    // instead recompute tokens during the reverse encode pass from q[] directly.
    for (size_t i = 0; i < n; ++i) {
        u32 u = zz(q[i]);
        counts[u < ESC ? u : ESC]++;
    }
    freq_table t; ft_build(&t, counts);

    // 2) header: token_byte_len placeholder (u32) + serialized freqs.
    if (cap < 4 + NSYM * 2) return cap + 1;
    u8 *freq_bytes = out + 4;
    for (u32 s = 0; s < NSYM; ++s) {
        freq_bytes[2 * s]     = (u8)(t.freq[s] & 0xff);
        freq_bytes[2 * s + 1] = (u8)(t.freq[s] >> 8);
    }
    size_t tok_off = 4 + NSYM * 2;

    // 3) rANS encode tokens in REVERSE so decode reads forward. Emit 16-bit
    //    renorm words into a temporary stack growing downward in the output
    //    tail; we cannot know the length up front, so write words forward into
    //    a scratch region and reverse-copy at the end.
    //    Scratch words live just after tok_off; final tokens overwrite them.
    u16 *words = (u16 *)(out + tok_off);
    size_t nwords = 0;
    size_t words_cap = (cap > tok_off) ? (cap - tok_off) / 2 : 0;
    u32 state = RANS_L;
    for (size_t ii = n; ii-- > 0; ) {
        u32 u = zz(q[ii]);
        u32 sym = u < ESC ? u : ESC;
        u32 f = t.freq[sym], c = t.cum[sym];
        u64 x_max = (u64)((RANS_L >> PROB_BITS) << 16) * f;
        while (state >= x_max) {
            if (nwords >= words_cap) return cap + 1;
            words[nwords++] = (u16)(state & 0xffff);
            state >>= 16;
        }
        u32 qd = ft_divf(&t, state, sym);
        state = (qd << PROB_BITS) + (state - qd * f) + c;
    }
    // flush state (two words)
    if (nwords + 2 > words_cap) return cap + 1;
    words[nwords++] = (u16)(state & 0xffff);
    words[nwords++] = (u16)((state >> 16) & 0xffff);

    // reverse the words into the final token byte stream (decode reads forward).
    // Build into a small heap-free approach: reverse in place using two ends.
    for (size_t a = 0, b = nwords - 1; a < b; ++a, --b) {
        u16 tmp = words[a]; words[a] = words[b]; words[b] = tmp;
    }
    size_t tok_bytes = nwords * 2;
    // words already occupy out+tok_off as u16; reinterpret as the byte stream
    // (little-endian u16 == byte pair, matches decoder read_word).
    out[0] = (u8)(tok_bytes & 0xff);
    out[1] = (u8)((tok_bytes >> 8) & 0xff);
    out[2] = (u8)((tok_bytes >> 16) & 0xff);
    out[3] = (u8)((tok_bytes >> 24) & 0xff);

    // 4) bypass varints for escapes, packed after the tokens.
    size_t bypass_off = tok_off + tok_bytes;
    if (bypass_off > cap) return cap + 1;
    vc_bitwriter bw; vc_bw_init(&bw, out + bypass_off, cap - bypass_off);
    for (size_t i = 0; i < n; ++i) {
        u32 u = zz(q[i]);
        if (u >= ESC) bypass_put_varint(&bw, u);
    }
    if (bw.overflow) return cap + 1;
    size_t bypass_len = vc_bw_finish(&bw);
    return bypass_off + bypass_len;
}

void vc_rans_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len) {
    size_t tok_bytes = (size_t)in[0] | ((size_t)in[1] << 8)
                     | ((size_t)in[2] << 16) | ((size_t)in[3] << 24);
    const u8 *freq_bytes = in + 4;
    freq_table t;
    for (u32 s = 0; s < NSYM; ++s)
        t.freq[s] = (u16)(freq_bytes[2 * s] | ((u32)freq_bytes[2 * s + 1] << 8));
    ft_finish(&t);

    size_t tok_off = 4 + NSYM * 2;
    const u8 *toks = in + tok_off;
    size_t bypass_off = tok_off + tok_bytes;

    vc_bitreader br; vc_br_init(&br, in + bypass_off, len - bypass_off);

    // rANS decode forward.
    size_t pos = 0;
    u32 hi = (pos + 1 < tok_bytes) ? (toks[pos] | ((u32)toks[pos + 1] << 8)) : 0; pos += 2;
    u32 lo = (pos + 1 < tok_bytes) ? (toks[pos] | ((u32)toks[pos + 1] << 8)) : 0; pos += 2;
    u32 state = (hi << 16) | lo;

    for (size_t i = 0; i < n; ++i) {
        u32 slot = state & (PROB_SCALE - 1);
        u32 sym = t.slot2sym[slot];
        state = t.freq[sym] * (state >> PROB_BITS) + slot - t.cum[sym];
        while (state < RANS_L) {
            u32 w = (pos + 1 < tok_bytes) ? (toks[pos] | ((u32)toks[pos + 1] << 8)) : 0;
            pos += 2;
            state = (state << 16) | w;
        }
        u32 u = (sym == ESC) ? bypass_get_varint(&br) : sym;
        q[i] = unzz(u);
    }
}
