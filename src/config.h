// COMPILE-TIME CODEC CONFIGURATION (PLAN §3 "block contract").
//
// Each generic block name below is #defined to exactly ONE concrete impl. The
// preprocessor resolves all dispatch, so the emitted codec is monomorphic, fully
// inlinable, and compiler-autovectorizable: NO function pointers, NO registry,
// NO switch in any hot loop. Sweeping the experiment matrix = compiling many
// configs (each a different set of these #defines) and racing the binaries.
//
// Later agents ADD blocks by writing a new .c implementing the SAME generic
// signature (see the contract comments at each block), then selecting it here.
// The signatures below are the stable contract — do not change them lightly.
#ifndef VC_CONFIG_H
#define VC_CONFIG_H

// --- Chunk geometry --------------------------------------------------------
// Chunk side is a COMPILE-TIME CONSTANT. Default 64; must also build at
// 32/128/256. Power of two; the codec atom and the random-access unit.
#ifndef VC_CHUNK_SIDE
#define VC_CHUNK_SIDE 64
#endif

// --- Transform (transform/*.c) ---------------------------------------------
// Contract:
//   void VC_TRANSFORM_FWD(i16 *restrict coef, const u8 *restrict vox, i32 dc);
//   void VC_TRANSFORM_INV(u8 *restrict vox,  const i16 *restrict coef, i32 dc);
// Operates on one VC_CHUNK_SIDE^3 cube. `dc` is the per-chunk DC bias removed
// before / added after the transform. coef is a chunk-sized i16 SoA buffer in
// the transform's own coefficient layout (the inverse must match the forward).
#define VC_TRANSFORM_FWD  vc_dct_int8_fwd
#define VC_TRANSFORM_INV  vc_dct_int8_inv

// --- Quantizer (quant/*.c, but Phase 0 inlines dead-zone in the pipeline) ---
// Dead-zone scalar quantizer; see transform/pipeline usage.

// --- Entropy coder (entropy/*.c) -------------------------------------------
// Contract:
//   size_t VC_ENTROPY_ENC(u8 *restrict out, size_t cap,
//                         const i16 *restrict q, size_t n);
//   void   VC_ENTROPY_DEC(i16 *restrict q, size_t n,
//                         const u8 *restrict in, size_t len);
// Encodes/decodes a flat array of n signed quantized levels. Returns bytes
// written (encode) / consumes `len` bytes (decode). Caller sizes `cap`.
#define VC_ENTROPY_ENC  vc_rice_encode
#define VC_ENTROPY_DEC  vc_rice_decode

// --- Chunk dependency model (chunkmodel/*.c) -------------------------------
// Phase 0: independent chunks. The model wraps per-chunk encode/decode; for
// "independent" it is a thin pass-through (no shared state between chunks).
#define VC_CHUNKMODEL  vc_cm_independent

// --- Rate control (ratectrl/*.c) -------------------------------------------
// Contract:
//   f32 VC_RATECTRL_STEP(f32 q);   // base dead-zone step from the q knob
// Phase 0: fixed-q (step is a deterministic function of q only).
#define VC_RATECTRL_STEP  vc_rc_fixed_step

// --- Boundary healing (healing/*.c) ----------------------------------------
// Phase 0: none.
#define VC_HEALING  vc_heal_none

// --- Codec tag recorded in the archive header ------------------------------
// Identifies the active pipeline so a decoder validates it matches the build.
#define VC_CODEC_TAG_TRANSFORM 1   /* int-DCT 8^3 */
#define VC_CODEC_TAG_ENTROPY   1   /* Golomb-Rice */

#endif // VC_CONFIG_H
