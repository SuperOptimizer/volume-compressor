// ============================================================================
// v2codec.h — FROZEN v2 block codec (2026-06-09). See memory/codec-freeze.md.
//
// Compresses a 16^3 u8 voxel block: integer separable DCT-16 + dead-zone quant +
// CABAC range coder. Air voxels (value 0, the SAM2 mask) are handled mask-aware:
// the air boundary is coded ONCE per 256^3 chunk as a coherent surface (the
// "chunk mask"), and within a block air voxels are harmonically air-filled before
// the DCT and force-zeroed on decode.
//
// CLEAN SEAM: this codec knows nothing about the archive container, the zarr
// volume, or LODs. It operates on gathered 16^3 blocks and a prepared 256^3 chunk
// air mask handed in by the caller (v3archive). Dependency direction is one-way:
// v3archive depends on v2codec, never the reverse.
//
// The ONLY runtime parameter is `base_q` (quality dial; higher = more compression,
// lower fidelity). Everything else (dead-zone 0.80, HF power-law 0.65, harmonic
// fill 8 sweeps, full-res flat chunk-mask coder) is FROZEN — every other lever was
// tested and is either off-by-default-loses or baked-in-wins. Do not add knobs.
// ============================================================================
#ifndef V2CODEC_H
#define V2CODEC_H
#include <stdint.h>
#include <stddef.h>

typedef uint8_t  v2_u8;
typedef int32_t  v2_i32;

#define V2_BLK    16     // DCT block edge (DCT-16 everywhere)
#define V2_CHUNK  256    // chunk-mask edge (16 blocks); the air surface is coded per 256^3 chunk

// ---- frozen quant constants (the proven winners; see codec-freeze.md) ----
#define V2_DZ_FRAC   0.80f   // dead-zone width fraction
#define V2_HF_EXP    0.65f   // HF quant power-law exponent: step = base_q*(1+freq)^HF_EXP
#define V2_FILL_SWEEPS 8     // harmonic (Jacobi) air-fill sweeps before the DCT

// quality dial: the one runtime parameter. set once before encoding; decode reads
// the SAME value (it is NOT stored per-block — the archive must encode/decode with
// the matching base_q, exactly as a JPEG quality setting works).
void  v2_set_quality(float base_q);
float v2_get_quality(void);

// ---- block codec (operates on gathered 16^3 blocks) ----
// A growable output byte buffer the codec appends the block payload to.
typedef struct { v2_u8 *p; size_t len, cap; } v2_buf;
void   v2_buf_put(v2_buf *b, const void *s, size_t n);

// Encode one 16^3 block. `vox` = the 16^3 source voxels (row-major z,y,x; air=0).
// `rair` = the 16^3 reconstructed air mask for this block (air=1), sliced by the
// caller from the prepared chunk mask. Appends the block payload to `out` and
// returns its length via *len_out. Returns 1 if the block was coded (nonzero),
// 0 if all-zero (no payload — caller marks it absent in the block directory).
int    v2_enc_block(const v2_u8 *vox, const v2_u8 *rair, v2_buf *out, uint32_t *len_out);

// Decode one 16^3 block payload (as written by v2_enc_block) into `dst` (16^3).
// `rair` = the same 16^3 reconstructed air mask (air -> output 0).
void   v2_dec_block(const v2_u8 *payload, const v2_u8 *rair, v2_u8 *dst);

// ---- chunk-mask surface coder (the 256^3 air boundary, coded once per chunk) ----
// Encode a 256^3 air mask (air=1) into `buf` (cap bytes); returns encoded length.
uint32_t v2_enc_chunkmask(const v2_u8 *mask256, v2_u8 *buf, size_t cap);
// Decode a chunk-mask stream back into a 256^3 buffer.
void     v2_dec_chunkmask(const v2_u8 *buf, size_t len, v2_u8 *mask256);

// one-time init (builds the DCT tables). Call before any encode/decode.
void v2_codec_init(void);

#endif
