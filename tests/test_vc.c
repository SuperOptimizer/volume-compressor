// Round-trip + random-access tests for the frozen vc codec.
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double psnr(const unsigned char *a, const unsigned char *b, size_t n) {
    double se = 0;
    for (size_t i = 0; i < n; ++i) { double d = (double)a[i] - b[i]; se += d*d; }
    double mse = se / n;
    if (mse <= 0) return 99.0;
    return 10.0 * log10(255.0*255.0 / mse);
}

int main(int argc, char **argv) {
    vc_dims d = { 64, 48, 32 };
    size_t n = (size_t)d.nx * d.ny * d.nz;
    unsigned char *vol = malloc(n);
    // synthetic structured volume + a uniform region
    for (unsigned z = 0; z < d.nz; ++z)
    for (unsigned y = 0; y < d.ny; ++y)
    for (unsigned x = 0; x < d.nx; ++x) {
        size_t i = ((size_t)z*d.ny + y)*d.nx + x;
        if (x < 16 && y < 16 && z < 16) vol[i] = 7; // uniform atom
        else vol[i] = (unsigned char)(128 + 60*sin(x*0.3) + 40*cos(y*0.25) + 20*sin(z*0.4));
    }

    unsigned char *arc; size_t alen;
    float target = (argc > 1) ? atof(argv[1]) : 20.0f;
    vc_status s = vc_encode(vol, d, target, &arc, &alen);
    if (s != VC_OK) { printf("encode fail %d\n", s); return 1; }
    printf("encoded: raw=%zu arc=%zu ratio=%.2f\n", n, alen, (double)n/alen);

    vc_archive *a = vc_open(arc, alen);
    if (!a) { printf("open fail\n"); return 1; }

    vc_dims od;
    unsigned char *out = malloc(n);
    s = vc_decode_lod(a, 0, out, &od);
    if (s != VC_OK) { printf("decode_lod fail %d\n", s); return 1; }
    if (od.nx != d.nx || od.ny != d.ny || od.nz != d.nz) { printf("dims mismatch\n"); return 1; }
    printf("LOD0 PSNR=%.2f dB\n", psnr(vol, out, n));

    // random-access: decode one atom standalone, compare to full-decode region
    unsigned char atom[16*16*16];
    s = vc_decode_atom(a, 0, 1, 1, 1, atom);
    if (s != VC_OK) { printf("decode_atom fail %d\n", s); return 1; }
    // compare atom to the same atom extracted from out
    int mismatch = 0;
    for (unsigned z = 0; z < 16; ++z)
    for (unsigned y = 0; y < 16; ++y)
    for (unsigned x = 0; x < 16; ++x) {
        unsigned vx = 16+x, vy = 16+y, vz = 16+z;
        if (vx>=d.nx||vy>=d.ny||vz>=d.nz) continue;
        unsigned char fromlod = out[((size_t)vz*d.ny+vy)*d.nx+vx];
        unsigned char fromatom = atom[(z*16+y)*16+x];
        if (fromlod != fromatom) mismatch++;
    }
    printf("random-access atom match: %s (%d mismatches)\n", mismatch==0?"OK":"FAIL", mismatch);

    // uniform atom (0,0,0) should be exactly 7
    unsigned char uatom[16*16*16];
    vc_decode_atom(a, 0, 0, 0, 0, uatom);
    int upass = 1; for (int i=0;i<4096;i++) if (uatom[i]!=7) upass=0;
    printf("uniform atom: %s\n", upass?"OK":"FAIL");

    // LOD count
    printf("nmembers (LODs): ");
    int nl=0; vc_dims t; for (int l=0;l<8;l++){ if(vc_lod_dims(a,l,&t)==VC_OK){nl++; } }
    printf("%d\n", nl);

    int ok = (mismatch==0) && upass && (psnr(vol,out,n) > 25.0);
    printf("RESULT: %s\n", ok?"PASS":"FAIL");
    vc_close(a); free(arc); free(vol); free(out);
    return ok?0:1;
}
