// Comprehensive round-trip + random-access + format test for the frozen vc 1.0
// codec. Atom-size-agnostic (uses VC_ATOM, never hardcodes 16/32). Exercises:
//   1. encode/decode round-trip + quality sanity across ratio targets
//   2. random-access vc_decode_atom EXACTLY equals full vc_decode_lod (every atom)
//   3. all LOD members decode at their own dims
//   4. uniform-atom + absent-region fast paths
//   5. arbitrary sub-box vc_decode_region matches the full decode
//   6. edge/odd dimensions (not multiples of the atom size)
//   7. self-describing-format validation (reject mismatched magic/version/atom)
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); return 1; } } while (0)

static double psnr(const unsigned char *a, const unsigned char *b, size_t n) {
    double se = 0; for (size_t i = 0; i < n; ++i) { double d = (double)a[i]-b[i]; se += d*d; }
    double mse = se/n; return mse <= 0 ? 99.0 : 10.0*log10(255.0*255.0/mse);
}

// structured volume + a guaranteed-zero (absent) region + a guaranteed-uniform region
static void fill(unsigned char *v, vc_dims d) {
    for (unsigned z = 0; z < d.nz; ++z)
    for (unsigned y = 0; y < d.ny; ++y)
    for (unsigned x = 0; x < d.nx; ++x) {
        size_t i = ((size_t)z*d.ny + y)*d.nx + x;
        if (z < VC_ATOM && y < VC_ATOM && x < VC_ATOM) v[i] = 0;               // absent/zero
        else if (z >= d.nz-VC_ATOM && y < VC_ATOM && x < VC_ATOM) v[i] = 200;  // uniform
        else v[i] = (unsigned char)(128 + 60*sin(x*0.21) + 40*cos(y*0.17) + 20*sin(z*0.29));
    }
}

static int test_dims(vc_dims d, float target) {
    size_t n = (size_t)d.nx*d.ny*d.nz;
    unsigned char *vol = malloc(n), *full = malloc(n);
    fill(vol, d);

    unsigned char *arc; size_t alen;
    CHECK(vc_encode(vol, d, target, &arc, &alen) == VC_OK, "encode %ux%ux%u", d.nx,d.ny,d.nz);
    vc_archive *a = vc_open(arc, alen);
    CHECK(a != NULL, "vc_open %ux%ux%u", d.nx,d.ny,d.nz);

    vc_dims od;
    CHECK(vc_decode_lod(a, 0, full, &od) == VC_OK, "decode_lod0");
    CHECK(od.nx==d.nx && od.ny==d.ny && od.nz==d.nz, "lod0 dims");
    double p = psnr(vol, full, n);
    CHECK(p > 25.0, "PSNR %.1f too low (%ux%ux%u @%gx)", p, d.nx,d.ny,d.nz, target);

    // EVERY atom: random-access decode must EXACTLY equal the full decode
    vc_dims m0; vc_lod_dims(a, 0, &m0);
    unsigned acx=(m0.nx+VC_ATOM-1)/VC_ATOM, acy=(m0.ny+VC_ATOM-1)/VC_ATOM, acz=(m0.nz+VC_ATOM-1)/VC_ATOM;
    unsigned char *atom = malloc(VC_ATOM3);
    long mism = 0;
    for (unsigned az=0; az<acz; ++az) for (unsigned ay=0; ay<acy; ++ay) for (unsigned ax=0; ax<acx; ++ax) {
        CHECK(vc_decode_atom(a, 0, ax, ay, az, atom) == VC_OK, "decode_atom %u,%u,%u", ax,ay,az);
        for (unsigned z=0; z<VC_ATOM; ++z) { unsigned vz=az*VC_ATOM+z; if (vz>=d.nz) break;
        for (unsigned y=0; y<VC_ATOM; ++y) { unsigned vy=ay*VC_ATOM+y; if (vy>=d.ny) break;
        for (unsigned x=0; x<VC_ATOM; ++x) { unsigned vx=ax*VC_ATOM+x; if (vx>=d.nx) break;
            if (atom[(z*VC_ATOM+y)*VC_ATOM+x] != full[((size_t)vz*d.ny+vy)*d.nx+vx]) mism++;
        }}}
    }
    CHECK(mism == 0, "%ld atom-vs-full mismatches (%ux%ux%u)", mism, d.nx,d.ny,d.nz);

    // arbitrary sub-box region decode matches the full decode
    if (d.nx > VC_ATOM && d.ny > VC_ATOM && d.nz > VC_ATOM) {
        vc_box box = { 3, 5, 7, d.nx-2, d.ny-3, VC_ATOM+5 };
        unsigned ox=box.x1-box.x0, oy=box.y1-box.y0, oz=box.z1-box.z0;
        unsigned char *reg = malloc((size_t)ox*oy*oz);
        CHECK(vc_decode_region(a, 0, box, reg) == VC_OK, "decode_region");
        long rmism=0;
        for (unsigned z=0; z<oz; ++z) for (unsigned y=0; y<oy; ++y) for (unsigned x=0; x<ox; ++x)
            if (reg[((size_t)z*oy+y)*ox+x] != full[((size_t)(z+box.z0)*d.ny+(y+box.y0))*d.nx+(x+box.x0)]) rmism++;
        CHECK(rmism == 0, "%ld region-vs-full mismatches", rmism);
        free(reg);
    }

    // all LODs decode at their own dims
    for (int lod=0; lod<VC_NLOD; ++lod) {
        vc_dims ld; if (vc_lod_dims(a, lod, &ld) != VC_OK) break;
        size_t ln=(size_t)ld.nx*ld.ny*ld.nz; if (ln==0) break;
        unsigned char *lo = malloc(ln); vc_dims od2;
        CHECK(vc_decode_lod(a, lod, lo, &od2) == VC_OK, "decode_lod %d", lod);
        free(lo);
    }

    vc_close(a); free(arc); free(atom); free(vol); free(full);
    return 0;
}

int main(void) {
    printf("VC_ATOM=%u  testing format + round-trip + random-access...\n", VC_ATOM);

    vc_dims cases[] = {
        {128,128,128}, {96,96,96}, {130,127,65}, {VC_ATOM,VC_ATOM,VC_ATOM},
        {VC_ATOM*2+1, VC_ATOM*3-1, VC_ATOM+7}, {200,40,40}, {64,256,48},
    };
    float ratios[] = { 10.0f, 50.0f, 100.0f };
    for (size_t c=0; c<sizeof(cases)/sizeof(cases[0]); ++c)
        for (size_t r=0; r<sizeof(ratios)/sizeof(ratios[0]); ++r)
            if (test_dims(cases[c], ratios[r])) return 1;

    // self-describing-format rejection
    { vc_dims d={96,96,96}; size_t n=96*96*96; unsigned char*v=malloc(n); fill(v,d);
      unsigned char*arc; size_t al; vc_encode(v,d,20.0f,&arc,&al);
      CHECK(vc_open(arc,al)!=NULL, "valid archive should open");
      unsigned char save=arc[8]; arc[8]=(unsigned char)(VC_ATOM==16?32:16);
      CHECK(vc_open(arc,al)==NULL, "atom-mismatch archive must be rejected");
      arc[8]=save; arc[4]^=0xFF;
      CHECK(vc_open(arc,al)==NULL, "version-mismatch archive must be rejected");
      free(arc); free(v); }

    printf("ALL TESTS PASSED\n");
    return 0;
}
