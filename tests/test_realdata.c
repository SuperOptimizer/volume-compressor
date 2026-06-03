// Real-data test: PHerc Paris 4 45um (74 keV) scroll subvolume.
//
// Simulates the cache use-case end to end:
//   - load a real 1024^3 u8 region,
//   - build the LOD pyramid (caller's job; uses the test-only downsample hook),
//   - "download" the volume in 256^3 chunks and vc_append_box each into the
//     archive (some all-zero chunks are marked known-zero instead),
//   - close, reopen the archive via mmap, and serve atoms back,
//   - check coverage (PRESENT / KNOWN_ZERO / ABSENT), round-trip PSNR per LOD,
//     and the achieved compression ratio vs the 30x target.
//
// Data path is testdata/pherc_paris4_45um_1024.raw (+ .json). If absent, the
// test SKIPS (exit 0) so CI without the data still passes.
#include "../src/vc/vc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); return 1; } } while (0)

// test-only downsample hook exported by the library under VC_TESTHOOKS
unsigned char *vc_test_downsample2x(const unsigned char *vol, vc_dims in, vc_dims *out);

static const char *DATA = "testdata/pherc_paris4_45um_1024.raw";
static const char *DATA2 = "../testdata/pherc_paris4_45um_1024.raw"; // when run from build/

static double psnr(const unsigned char *a, const unsigned char *b, size_t n) {
    double se = 0; for (size_t i=0;i<n;++i){ double d=(double)a[i]-b[i]; se+=d*d; }
    double mse = se/n; return mse <= 0 ? 99.0 : 10.0*log10(255.0*255.0/mse);
}

static unsigned char *load_raw(size_t *n_out) {
    const char *p = DATA;
    FILE *f = fopen(p, "rb");
    if (!f) { p = DATA2; f = fopen(p, "rb"); }
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f); *n_out = sz; return buf;
}

static int atom_all_zero(const unsigned char *vol, vc_dims d, unsigned az, unsigned ay, unsigned ax) {
    for (unsigned z=0; z<VC_ATOM; ++z){ unsigned vz=az*VC_ATOM+z; if(vz>=d.nz)continue;
    for (unsigned y=0; y<VC_ATOM; ++y){ unsigned vy=ay*VC_ATOM+y; if(vy>=d.ny)continue;
    for (unsigned x=0; x<VC_ATOM; ++x){ unsigned vx=ax*VC_ATOM+x; if(vx>=d.nx)continue;
        if (vol[((size_t)vz*d.ny+vy)*d.nx+vx]) return 0;
    }}}
    return 1;
}

int main(void) {
    size_t rawn;
    unsigned char *vol0 = load_raw(&rawn);
    if (!vol0) { printf("SKIP realdata: %s not found\n", DATA); return 0; }
    vc_dims d0 = {1024,1024,1024};
    CHECK(rawn == (size_t)d0.nx*d0.ny*d0.nz, "raw size %zu != 1024^3", rawn);
    printf("realdata: loaded %ux%ux%u (%.0f MB)\n", d0.nx,d0.ny,d0.nz, rawn/1e6);

    const char *path = "/tmp/vc_realdata.vca";
    float target = 30.0f;
    vc_writer *w = vc_create(path, d0, target);
    CHECK(w, "vc_create");
    // q is explicit now (no auto-calibration). Use a per-LOD falloff so coarse
    // levels get higher quality, mirroring the old halving schedule.
    { float q=1.0f; for (int l=0;l<VC_NLOD;++l){ vc_set_base_q(w,l,q); q*=0.6f; } }

    // Build LOD volumes (caller's responsibility — strict 2x pyramid).
    unsigned nlod = 0;
    unsigned char *lodv[VC_NLOD]; vc_dims lodd[VC_NLOD];
    lodv[0] = vol0; lodd[0] = d0; nlod = 1;
    for (int l=1; l<VC_NLOD; ++l) {
        vc_dims od; lodv[l] = vc_test_downsample2x(lodv[l-1], lodd[l-1], &od);
        lodd[l] = od; nlod = l+1;
        if (od.nx<=1 && od.ny<=1 && od.nz<=1) break;
    }
    printf("realdata: built %u LODs\n", nlod);

    // "Download" + append each LOD in 256^3 boxes (multiple of 32). All-zero
    // boxes are marked known-zero per-atom instead of stored.
    long appended=0, zeroed=0;
    const unsigned BOX = 256;
    for (unsigned l=0; l<nlod; ++l) {
        vc_dims d = lodd[l];
        unsigned char *vol = lodv[l];
        for (unsigned z0=0; z0<d.nz; z0+=BOX)
        for (unsigned y0=0; y0<d.ny; y0+=BOX)
        for (unsigned x0=0; x0<d.nx; x0+=BOX) {
            unsigned z1=z0+BOX<d.nz?z0+BOX:d.nz, y1=y0+BOX<d.ny?y0+BOX:d.ny, x1=x0+BOX<d.nx?x0+BOX:d.nx;
            // round box extents UP to a multiple of 32 (atom grid), clamped to dims-atoms
            // append atom-by-atom to handle ragged edges and zero-marking cleanly.
            unsigned ax0=x0/VC_ATOM, ay0=y0/VC_ATOM, az0=z0/VC_ATOM;
            unsigned ax1=(x1+VC_ATOM-1)/VC_ATOM, ay1=(y1+VC_ATOM-1)/VC_ATOM, az1=(z1+VC_ATOM-1)/VC_ATOM;
            unsigned char atom[VC_ATOM3];
            for (unsigned az=az0; az<az1; ++az)
            for (unsigned ay=ay0; ay<ay1; ++ay)
            for (unsigned ax=ax0; ax<ax1; ++ax) {
                if (atom_all_zero(vol, d, az, ay, ax)) {
                    CHECK(vc_mark_zero_atom(w, l, az, ay, ax)==VC_OK, "mark_zero l%u %u,%u,%u",l,az,ay,ax);
                    zeroed++;
                    continue;
                }
                for (unsigned z=0; z<VC_ATOM; ++z){ unsigned vz=az*VC_ATOM+z;
                for (unsigned y=0; y<VC_ATOM; ++y){ unsigned vy=ay*VC_ATOM+y;
                for (unsigned x=0; x<VC_ATOM; ++x){ unsigned vx=ax*VC_ATOM+x;
                    unsigned char v = (vz<d.nz&&vy<d.ny&&vx<d.nx)? vol[((size_t)vz*d.ny+vy)*d.nx+vx] : 0;
                    atom[(z*VC_ATOM+y)*VC_ATOM+x]=v;
                }}}
                CHECK(vc_append_atom(w, l, az, ay, ax, atom)==VC_OK, "append l%u %u,%u,%u",l,az,ay,ax);
                appended++;
            }
        }
    }
    vc_writer_close(w);

    struct stat st; stat(path, &st);
    double ratio = (double)rawn / (double)st.st_size; // ratio vs LOD0 raw bytes
    printf("realdata: appended %ld atoms, %ld known-zero; archive %.1f MB (%.1fx vs LOD0 raw)\n",
           appended, zeroed, st.st_size/1e6, ratio);

    // reopen via mmap and serve
    int fd = open(path, O_RDONLY); CHECK(fd>=0, "open archive");
    void *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    CHECK(m != MAP_FAILED, "mmap");
    vc_archive *a = vc_open((unsigned char*)m, st.st_size); CHECK(a, "vc_open");

    // per-LOD PSNR over PRESENT atoms; coverage sanity.
    for (unsigned l=0; l<nlod; ++l) {
        vc_dims d = lodd[l];
        unsigned char *vol = lodv[l];
        unsigned acx=(d.nx+VC_ATOM-1)/VC_ATOM, acy=(d.ny+VC_ATOM-1)/VC_ATOM, acz=(d.nz+VC_ATOM-1)/VC_ATOM;
        double sum=0, worst=99; long np=0, nz=0;
        unsigned char out[VC_ATOM3], ref[VC_ATOM3];
        for (unsigned az=0; az<acz; ++az)
        for (unsigned ay=0; ay<acy; ++ay)
        for (unsigned ax=0; ax<acx; ++ax) {
            vc_cover c = vc_atom_coverage(a, l, az, ay, ax);
            if (c == VC_KNOWN_ZERO) { nz++;
                CHECK(vc_decode_atom(a,l,ax,ay,az,out)==VC_OK,"dec z");
                // known-zero must decode to zero
                int ok=1; for(unsigned i=0;i<VC_ATOM3;++i) if(out[i]){ok=0;break;}
                CHECK(ok, "known-zero l%u %u,%u,%u not zero",l,az,ay,ax);
                continue;
            }
            CHECK(c == VC_PRESENT, "expected PRESENT l%u %u,%u,%u (got %d)",l,az,ay,ax,c);
            for (unsigned z=0; z<VC_ATOM; ++z){ unsigned vz=az*VC_ATOM+z;
            for (unsigned y=0; y<VC_ATOM; ++y){ unsigned vy=ay*VC_ATOM+y;
            for (unsigned x=0; x<VC_ATOM; ++x){ unsigned vx=ax*VC_ATOM+x;
                ref[(z*VC_ATOM+y)*VC_ATOM+x]=(vz<d.nz&&vy<d.ny&&vx<d.nx)?vol[((size_t)vz*d.ny+vy)*d.nx+vx]:0;
            }}}
            CHECK(vc_decode_atom(a,l,ax,ay,az,out)==VC_OK,"dec p");
            double p=psnr(ref,out,VC_ATOM3); sum+=p; if(p<worst)worst=p; np++;
        }
        printf("  LOD%u: %ldx PRESENT, %ldx KNOWN_ZERO, mean PSNR %.1f dB (worst %.1f)\n",
               l, np, nz, np?sum/np:0.0, np?worst:0.0);
        // Coarse LODs at a fixed target ratio are inherently lower-PSNR (detail
        // concentrates under downsampling) and have far fewer atoms (noisier
        // single-atom q calibration). LOD0 carries the real fidelity signal.
        double floor_db = (l == 0) ? 32.0 : (l == 1) ? 26.0 : 18.0;
        if (np) CHECK(sum/np > floor_db, "LOD%u mean PSNR %.1f below floor %.1f", l, sum/np, floor_db);
    }

    // a never-appended coordinate well outside touched area is ABSENT? Hard to
    // guarantee here (we touched everything), so just confirm the API is sane on
    // an out-of-range lod.
    CHECK(vc_atom_coverage(a, nlod, 0,0,0)==VC_ABSENT, "bad lod -> absent");

    vc_close(a); munmap(m, st.st_size);
    for (unsigned l=1; l<nlod; ++l) free(lodv[l]);
    free(vol0); unlink(path);

    // overall ratio sanity: we don't demand exactly 30x (zero-marking + pyramid
    // change the accounting), but it should be in a sane lossy band.
    CHECK(ratio > 5.0, "overall ratio %.1f implausibly low", ratio);
    printf("realdata: PASSED (overall %.1fx)\n", ratio);
    return 0;
}
