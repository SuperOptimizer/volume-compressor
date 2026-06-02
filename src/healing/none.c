// VC_HEALING = none. Identity passthrough (Phase-0 default, PLAN §2). The
// reconstructed volume is returned untouched.
#include "healing.h"

void vc_heal_none(u8 *restrict vol, u32 dz, u32 dy, u32 dx, f32 step, f32 strength) {
    (void)vol; (void)dz; (void)dy; (void)dx; (void)step; (void)strength;
}
