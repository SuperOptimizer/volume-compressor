// libFuzzer target: feed arbitrary bytes to the decoder. Must NEVER crash/OOB —
// only return errors. Catches malformed-archive handling bugs before release.
#include "../src/vc/vc.h"
#include <stdint.h>
#include <stdlib.h>
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t len) {
    vc_archive *a = vc_open(data, len);
    if (!a) return 0;
    vc_dims d;
    for (int lod = 0; lod < VC_NLOD; ++lod) {
        if (vc_lod_dims(a, lod, &d) != VC_OK) continue;
        // cap allocation so the fuzzer doesn't OOM on huge claimed dims
        size_t n = (size_t)d.nx*d.ny*d.nz;
        if (n == 0 || n > (64u<<20)) continue;
        unsigned char *out = malloc(n); vc_dims od;
        if (out) { vc_decode_lod(a, lod, out, &od); free(out); }
        unsigned char atom[VC_ATOM3];
        vc_decode_atom(a, lod, 0, 0, 0, atom);
    }
    vc_close(a);
    return 0;
}
