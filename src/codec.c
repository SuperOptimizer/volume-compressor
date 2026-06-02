// The ONE Phase-0 codec pipeline (PLAN §3 item 3), wired through the
// compile-time block contract in config.h:
//   integer-DCT 8^3  +  dead-zone scalar quant  +  Golomb-Rice  +
//   independent chunks  +  fixed-q rate control  +  no healing.
//
// Independent-chunk model (VC_CHUNKMODEL = independent): each chunk is encoded
// and decoded entirely on its own — there is no shared entropy state or
// cross-chunk prediction, so the loop below carries no inter-chunk dependency.
// No-healing (VC_HEALING = none): decode writes reconstructed chunks straight
// back with no deblock/seam pass.
//
// Per-chunk payload byte layout (LE):
//   u8  tag        (0 = normal, 1 = uniform constant chunk)
//   -- uniform: --
//   u8  value
//   -- normal:  --
//   i16 dc         (per-chunk DC bias removed before the transform)
//   f32 step       (dead-zone step used; decoder dequantizes with it)
//   u32 rice_len   (entropy byte count)
//   u8  rice[rice_len]
//
// Hot loops (DCT, quant, dequant, Rice) are straight-line over contiguous
// arrays so the compiler autovectorizes them; chunk-sized scratch is heap/arena
// (never tile-sized stack), per PLAN §7.
#include "../include/vc/vc.h"
#include "config.h"
#include "blocks.h"
#include "core/chunk.h"
#include "core/archive.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

u32 vc_chunk_side(void) { return VC_CS; }

// Per-thread reusable scratch (PLAN §7: chunk-sized arena, never stack).
typedef struct {
    u8  *vox;     // VC_CVOX
    i16 *coef;    // VC_CVOX
    i16 *qlev;    // VC_CVOX
    u8  *payload; // worst-case payload bytes
} vc_scratch;

static void vc_scratch_alloc(vc_scratch *s) {
    s->vox     = (u8  *)malloc(VC_CVOX);
    s->coef    = (i16 *)malloc(VC_CVOX * sizeof(i16));
    s->qlev    = (i16 *)malloc(VC_CVOX * sizeof(i16));
    // Worst case Rice ~ a few bits/coef; size generously: 4 bytes/coef + header.
    s->payload = (u8  *)malloc(VC_CVOX * 4 + 64);
}
static void vc_scratch_free(vc_scratch *s) {
    free(s->vox); free(s->coef); free(s->qlev); free(s->payload);
}

// Quantization + frequency scan is the VC_QUANT block (quant/subband.c): a
// per-subband HF-weighted dead-zone quantizer that also reorders coefficients
// low->high frequency so HF zero-runs are long and contiguous for the entropy
// coder. Phase-0's inline flat dead-zone is now VC_QWEIGHT_FLAT in that block.

// Encode one chunk into scratch->payload; returns payload length (0 => ABSENT).
static size_t encode_chunk(vc_scratch *s, f32 step) {
    // Uniform fast-path: a perfectly constant chunk codes as just its value.
    u8 v0 = s->vox[0];
    int uniform = 1;
    for (size_t i = 1; i < VC_CVOX; ++i) if (s->vox[i] != v0) { uniform = 0; break; }
    if (uniform) {
        // ABSENT if all-zero (no payload); else 2-byte uniform payload.
        if (v0 == 0) return 0;
        s->payload[0] = 1; s->payload[1] = v0;
        return 2;
    }

    // Per-chunk DC = rounded mean.
    u64 sum = 0;
    for (size_t i = 0; i < VC_CVOX; ++i) sum += s->vox[i];
    i32 dc = (i32)((sum + VC_CVOX / 2) / VC_CVOX);

    VC_TRANSFORM_FWD(s->coef, s->vox, dc);
    VC_QUANT_FWD(s->qlev, s->coef, step);

    u8 *p = s->payload;
    size_t pos = 0;
    p[pos++] = 0;                                   // tag: normal
    i16 dc16 = (i16)dc; memcpy(p + pos, &dc16, 2); pos += 2;
    memcpy(p + pos, &step, 4); pos += 4;
    size_t rlen_pos = pos; pos += 4;                // rice_len placeholder
    size_t cap = VC_CVOX * 4 + 64 - pos;
    size_t rlen = VC_ENTROPY_ENC(p + pos, cap, s->qlev, VC_CVOX);
    u32 rlen32 = (u32)rlen;
    memcpy(p + rlen_pos, &rlen32, 4);
    pos += rlen;
    return pos;
}

// Decode one chunk payload into scratch->vox (VC_CHUNK_SIDE^3).
static void decode_chunk(vc_scratch *s, const u8 *p, size_t plen) {
    if (plen == 0) { memset(s->vox, 0, VC_CVOX); return; }  // ABSENT => all zero
    if (p[0] == 1) { memset(s->vox, p[1], VC_CVOX); return; } // uniform
    size_t pos = 1;
    i16 dc16; memcpy(&dc16, p + pos, 2); pos += 2;
    f32 step;  memcpy(&step, p + pos, 4); pos += 4;
    u32 rlen;  memcpy(&rlen, p + pos, 4); pos += 4;
    VC_ENTROPY_DEC(s->qlev, VC_CVOX, p + pos, rlen);
    VC_QUANT_INV(s->coef, s->qlev, step);
    VC_TRANSFORM_INV(s->vox, s->coef, (i32)dc16);
    (void)plen;
}

int vc_encode_volume(const u8 *vol, u32 dz, u32 dy, u32 dx, f32 q,
                     u8 **out, size_t *out_len) {
    const f32 step = VC_RATECTRL_STEP(q);
    const u32 ncz = vc_nchunks(dz), ncy = vc_nchunks(dy), ncx = vc_nchunks(dx);

    vc_scratch s; vc_scratch_alloc(&s);
    if (!s.vox || !s.coef || !s.qlev || !s.payload) { vc_scratch_free(&s); return 1; }

    vc_writer w; vc_writer_init(&w, "0/");
    // Independent chunks in row-major chunk-grid order (positional index).
    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx) {
        vc_chunk_gather(s.vox, vol, dz, dy, dx, cz, cy, cx);
        size_t plen = encode_chunk(&s, step);
        vc_writer_add_chunk(&w, plen ? s.payload : NULL, plen, q);
    }

    vc_config_hdr cfg = { VC_CS, VC_CODEC_TAG_TRANSFORM, VC_CODEC_TAG_ENTROPY, 0 };
    vc_coord3 shape = { dz, dy, dx };
    int rc = vc_writer_finish(&w, shape, &cfg, out, out_len);
    vc_scratch_free(&s);
    return rc;
}

int vc_decode_volume(const u8 *archive, size_t len,
                     u8 **vol, u32 *dz, u32 *dy, u32 *dx) {
    vc_reader r;
    int rc = vc_reader_open(&r, archive, len);
    if (rc) return rc;
    if (r.cfg.chunk_side != VC_CS) return 100;             // pipeline mismatch
    if (r.cfg.transform_tag != VC_CODEC_TAG_TRANSFORM) return 101;
    if (r.cfg.entropy_tag != VC_CODEC_TAG_ENTROPY) return 102;

    const u32 D = r.shape.z, H = r.shape.y, W = r.shape.x;
    const u32 ncz = vc_nchunks(D), ncy = vc_nchunks(H), ncx = vc_nchunks(W);
    if ((u64)ncz * ncy * ncx != r.nchunks) return 103;

    u8 *out = (u8 *)calloc((size_t)D * H * W, 1);
    if (!out) return 104;

    vc_scratch s; vc_scratch_alloc(&s);
    u32 ci = 0;
    for (u32 cz = 0; cz < ncz; ++cz)
    for (u32 cy = 0; cy < ncy; ++cy)
    for (u32 cx = 0; cx < ncx; ++cx, ++ci) {
        size_t plen; int absent;
        const u8 *p = vc_reader_chunk(&r, ci, &plen, &absent);
        decode_chunk(&s, p, plen);
        vc_chunk_scatter(out, s.vox, D, H, W, cz, cy, cx);
    }
    vc_scratch_free(&s);
    *vol = out; *dz = D; *dy = H; *dx = W;
    return 0;
}
