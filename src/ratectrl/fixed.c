// Fixed-q rate control (PLAN §3 Phase-0). The dead-zone base step is a
// deterministic function of the quality knob q only — no probing, no feedback.
// Larger q => coarser step => higher ratio, lower quality. The mapping is
// intentionally simple (step == q) so q reads directly as "DCT-coefficient
// quantization step in coefficient units"; the harness sweeps q to trace the
// rate-distortion curve.
#include "../../include/vc/types.h"

f32 vc_rc_fixed_step(f32 q) {
    return q > 0.f ? q : 1.f;
}
