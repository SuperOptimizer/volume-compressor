// Declarations for every concrete block implementation. config.h #defines the
// generic block names to one of these. Keeping all declarations here lets the
// pipeline include a single header and the compiler inline the selected impl.
#ifndef VC_BLOCKS_H
#define VC_BLOCKS_H

#include "../include/vc/types.h"

// --- Transform (transform/*.c) ---------------------------------------------
void vc_dct_int8_fwd(i16 *restrict coef, const u8 *restrict vox, i32 dc);
void vc_dct_int8_inv(u8 *restrict vox, const i16 *restrict coef, i32 dc);

// --- Entropy (entropy/*.c) -------------------------------------------------
size_t vc_rice_encode(u8 *restrict out, size_t cap, const i16 *restrict q, size_t n);
void   vc_rice_decode(i16 *restrict q, size_t n, const u8 *restrict in, size_t len);

// --- Rate control (ratectrl/*.c) -------------------------------------------
f32 vc_rc_fixed_step(f32 q);

#endif // VC_BLOCKS_H
