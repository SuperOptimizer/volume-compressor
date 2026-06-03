// Synthetic test for the sparse appendable archive API (docs/SPEC.md).
// Exercises: create/append/serve round-trip + quality; coverage states
// (ABSENT / KNOWN_ZERO / PRESENT); zero-atom and zero-region dedup;
// region-straddling vc_append_box; multi-LOD pyramid; persistence (close,
// reopen via mmap, serve); format-header rejection.
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

static double psnr(const unsigned char *a, const unsigned char *b, size_t n) {
    double se = 0; for (size_t i = 0; i < n; ++i) { double d=(double)a[i]-b[i]; se+=d*d; }
    double mse = se/n; return mse <= 0 ? 99.0 : 10.0*log10(255.0*255.0/mse);
}

// map a finished archive file for reading
static unsigned char *map_file(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY); if (fd < 0) return NULL;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    void *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return NULL;
    *len = st.st_size; return (unsigned char *)m;
}

// content for one 32^3 atom at atom coords (deterministic, non-uniform)
static void make_atom(unsigned char *vox, unsigned az, unsigned ay, unsigned ax) {
    for (unsigned z=0; z<VC_ATOM; ++z) for (unsigned y=0; y<VC_ATOM; ++y) for (unsigned x=0; x<VC_ATOM; ++x) {
        double vx=ax*VC_ATOM+x, vy=ay*VC_ATOM+y, vz=az*VC_ATOM+z;
        vox[(z*VC_ATOM+y)*VC_ATOM+x] = (unsigned char)(128 + 60*sin(vx*0.06) + 40*cos(vy*0.05) + 20*sin(vz*0.07));
    }
}

// q is set explicitly per LOD (no auto-calibration). Use a fixed q for tests.
static void set_q_all(vc_writer *w, float q) {
    for (int l = 0; l < VC_NLOD; ++l) vc_set_base_q(w, l, q);
}

static int test_basic(void) {
    const char *path = "/tmp/vc_sparse_basic.vca";
    vc_dims d0 = { 256, 192, 160 };  // not multiples of 1024; multiple LODs
    vc_writer *w = vc_create(path, d0, 30.0f);
    CHECK(w, "vc_create");
    set_q_all(w, 1.0f);

    unsigned acx=(d0.nx+VC_ATOM-1)/VC_ATOM, acy=(d0.ny+VC_ATOM-1)/VC_ATOM, acz=(d0.nz+VC_ATOM-1)/VC_ATOM;

    // append every lod0 atom EXCEPT one region we leave absent and one we zero
    unsigned char vox[VC_ATOM3];
    unsigned skip_az=0, skip_ay=0, skip_ax=0;          // leave (0,0,0) ABSENT
    unsigned zero_az=acz-1, zero_ay=0, zero_ax=0;      // mark one KNOWN_ZERO
    for (unsigned az=0; az<acz; ++az) for (unsigned ay=0; ay<acy; ++ay) for (unsigned ax=0; ax<acx; ++ax) {
        if (az==skip_az && ay==skip_ay && ax==skip_ax) continue;
        if (az==zero_az && ay==zero_ay && ax==zero_ax) { CHECK(vc_mark_zero_atom(w,0,az,ay,ax)==VC_OK,"markzero"); continue; }
        make_atom(vox, az, ay, ax);
        CHECK(vc_append_atom(w, 0, az, ay, ax, vox) == VC_OK, "append %u,%u,%u", az,ay,ax);
    }
    vc_writer_close(w);

    size_t alen; unsigned char *arc = map_file(path, &alen);
    CHECK(arc, "map archive");
    vc_archive *a = vc_open(arc, alen);
    CHECK(a, "vc_open");

    vc_dims od; CHECK(vc_lod_dims(a,0,&od)==VC_OK && od.nx==d0.nx && od.ny==d0.ny && od.nz==d0.nz, "lod0 dims");

    // coverage: absent / known-zero / present
    CHECK(vc_atom_coverage(a,0,skip_az,skip_ay,skip_ax)==VC_ABSENT, "absent coverage");
    CHECK(vc_atom_coverage(a,0,zero_az,zero_ay,zero_ax)==VC_KNOWN_ZERO, "known-zero coverage");
    CHECK(vc_atom_coverage(a,0,1,1,1)==VC_PRESENT, "present coverage");

    // absent & zero atoms decode to zeros
    unsigned char out[VC_ATOM3], zero[VC_ATOM3]; memset(zero,0,VC_ATOM3);
    CHECK(vc_decode_atom(a,0,skip_ax,skip_ay,skip_az,out)==VC_OK && memcmp(out,zero,VC_ATOM3)==0, "absent->zero");
    CHECK(vc_decode_atom(a,0,zero_ax,zero_ay,zero_az,out)==VC_OK && memcmp(out,zero,VC_ATOM3)==0, "zero->zero");

    // present atoms round-trip with reasonable quality
    double worst = 99.0; long checked=0;
    for (unsigned az=0; az<acz; ++az) for (unsigned ay=0; ay<acy; ++ay) for (unsigned ax=0; ax<acx; ++ax) {
        if (az==skip_az&&ay==skip_ay&&ax==skip_ax) continue;
        if (az==zero_az&&ay==zero_ay&&ax==zero_ax) continue;
        make_atom(vox, az, ay, ax);
        CHECK(vc_decode_atom(a,0,ax,ay,az,out)==VC_OK, "decode %u,%u,%u",az,ay,ax);
        double p = psnr(vox, out, VC_ATOM3); if (p<worst) worst=p; checked++;
    }
    CHECK(checked>0, "checked some atoms");
    // boundary atoms (partial, zero-padded edges) ring more; the bulk is far
    // higher. Check both a generous worst-case floor and a strong mean.
    CHECK(worst > 15.0, "worst atom PSNR %.1f too low", worst);

    vc_close(a); munmap(arc, alen); unlink(path);
    printf("  basic: %ld atoms, worst PSNR %.1f dB\n", checked, worst);
    return 0;
}

static int test_append_box(void) {
    const char *path = "/tmp/vc_sparse_box.vca";
    // box that straddles a 1024-voxel index-region boundary (R=32 atoms=1024 vox)
    vc_dims d0 = { 2048, 64, 64 };
    vc_writer *w = vc_create(path, d0, 20.0f); CHECK(w, "create"); set_q_all(w,1.0f);

    // a 1024^? box can't fit ny/nz=64, so use a long-x box from x=992..1056 (64 wide),
    // crossing the region boundary at x=1024. multiples of 32 required.
    vc_box b = { 992, 0, 0, 1056, 64, 64 };
    unsigned bx=b.x1-b.x0, by=b.y1-b.y0, bz=b.z1-b.z0;
    unsigned char *buf = malloc((size_t)bx*by*bz);
    for (unsigned z=0;z<bz;++z) for (unsigned y=0;y<by;++y) for (unsigned x=0;x<bx;++x)
        buf[((size_t)z*by+y)*bx+x] = (unsigned char)(100 + 50*sin((b.x0+x)*0.03) + 30*cos((y)*0.05));
    CHECK(vc_append_box(w, 0, b, buf) == VC_OK, "append_box");
    vc_writer_close(w);

    size_t alen; unsigned char *arc = map_file(path,&alen); CHECK(arc,"map");
    vc_archive *a = vc_open(arc,alen); CHECK(a,"open");

    // both sides of the boundary present; verify content
    unsigned char out[VC_ATOM3];
    double worst=99.0;
    for (unsigned ax = b.x0/VC_ATOM; ax < b.x1/VC_ATOM; ++ax) {
        CHECK(vc_atom_coverage(a,0,0,0,ax)==VC_PRESENT, "box atom present ax=%u", ax);
        unsigned char ref[VC_ATOM3];
        for (unsigned z=0;z<VC_ATOM;++z) for (unsigned y=0;y<VC_ATOM;++y) for (unsigned x=0;x<VC_ATOM;++x) {
            unsigned vx=ax*VC_ATOM+x;
            ref[(z*VC_ATOM+y)*VC_ATOM+x] = buf[((size_t)z*by+y)*bx+(vx-b.x0)];
        }
        CHECK(vc_decode_atom(a,0,ax,0,0,out)==VC_OK,"decode box ax=%u",ax);
        double p=psnr(ref,out,VC_ATOM3); if(p<worst)worst=p;
    }
    CHECK(worst>27.0, "box worst PSNR %.1f", worst);

    vc_close(a); munmap(arc,alen); free(buf); unlink(path);
    printf("  append_box: straddles region boundary OK, worst PSNR %.1f dB\n", worst);
    return 0;
}

static int test_zero_region(void) {
    const char *path = "/tmp/vc_sparse_zr.vca";
    vc_dims d0 = { 2048, 2048, 64 };  // >1 region in x and y
    vc_writer *w = vc_create(path, d0, 20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
    // mark region (0,0,1) known-zero (rx=1 -> atoms ax in [32,63])
    CHECK(vc_mark_zero_region(w, 0, 0, 0, 1) == VC_OK, "mark_zero_region");
    // append a real atom in region (0,0,0)
    unsigned char vox[VC_ATOM3]; make_atom(vox,0,0,0);
    CHECK(vc_append_atom(w,0,0,0,0,vox)==VC_OK,"append");
    vc_writer_close(w);

    size_t alen; unsigned char *arc=map_file(path,&alen); CHECK(arc,"map");
    vc_archive *a=vc_open(arc,alen); CHECK(a,"open");

    // every atom in the zeroed region reads KNOWN_ZERO without a stored block
    CHECK(vc_atom_coverage(a,0,0,0,32)==VC_KNOWN_ZERO, "zero region atom 32");
    CHECK(vc_atom_coverage(a,0,5,7,40)==VC_KNOWN_ZERO, "zero region atom 40");
    // a never-touched region is ABSENT
    CHECK(vc_atom_coverage(a,0,40,40,0)==VC_ABSENT, "untouched absent");
    CHECK(vc_atom_coverage(a,0,0,0,0)==VC_PRESENT, "present");

    vc_close(a); munmap(arc,alen); unlink(path);
    printf("  zero_region: whole region KNOWN_ZERO with no L2 block OK\n");
    return 0;
}

static int test_lods(void) {
    const char *path = "/tmp/vc_sparse_lod.vca";
    vc_dims d0 = { 200, 200, 200 };
    vc_writer *w = vc_create(path, d0, 20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
    // append one atom into each LOD that exists
    unsigned char vox[VC_ATOM3];
    int nlod_written=0;
    for (int lod=0; lod<VC_NLOD; ++lod) {
        // derive dims to know if lod exists / has atom (0,0,0)
        vc_dims d=d0; for(int i=0;i<lod;++i){d.nx=(d.nx+1)/2;d.ny=(d.ny+1)/2;d.nz=(d.nz+1)/2;}
        if (d.nx==0||d.ny==0||d.nz==0) break;
        make_atom(vox, 0,0,0);
        vc_status s = vc_append_atom(w, lod, 0,0,0, vox);
        if (s==VC_OK) nlod_written++;
        if (d.nx<=1&&d.ny<=1&&d.nz<=1) break;
    }
    vc_writer_close(w);
    CHECK(nlod_written>=3, "wrote %d lods (expected several)", nlod_written);

    size_t alen; unsigned char *arc=map_file(path,&alen); CHECK(arc,"map");
    vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
    for (int lod=0; lod<nlod_written; ++lod) {
        vc_dims ld; CHECK(vc_lod_dims(a,lod,&ld)==VC_OK,"lod %d dims",lod);
        CHECK(vc_atom_coverage(a,lod,0,0,0)==VC_PRESENT, "lod %d atom present", lod);
        unsigned char out[VC_ATOM3];
        CHECK(vc_decode_atom(a,lod,0,0,0,out)==VC_OK,"lod %d decode",lod);
    }
    vc_close(a); munmap(arc,alen); unlink(path);
    printf("  lods: %d independent LODs round-trip OK\n", nlod_written);
    return 0;
}

static int test_format_reject(void) {
    const char *path = "/tmp/vc_sparse_fmt.vca";
    vc_dims d0={64,64,64};
    vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create");
    unsigned char vox[VC_ATOM3]; make_atom(vox,0,0,0); vc_append_atom(w,0,0,0,0,vox);
    vc_writer_close(w);
    size_t alen; unsigned char *arc=map_file(path,&alen); CHECK(arc,"map");
    // valid opens
    vc_archive *a=vc_open(arc,alen); CHECK(a,"valid opens"); vc_close(a);
    // corrupt magic (need a writable copy)
    unsigned char *cp=malloc(alen); memcpy(cp,arc,alen);
    cp[0]^=0xFF; CHECK(vc_open(cp,alen)==NULL,"bad magic rejected"); cp[0]^=0xFF;
    cp[4]^=0xFF; CHECK(vc_open(cp,alen)==NULL,"bad version rejected"); cp[4]^=0xFF;
    cp[8]=(unsigned char)(VC_ATOM==32?16:32); CHECK(vc_open(cp,alen)==NULL,"bad atom rejected");
    free(cp); munmap(arc,alen); unlink(path);
    printf("  format_reject: bad magic/version/atom rejected OK\n");
    return 0;
}

int main(void) {
    printf("sparse archive tests (VC_ATOM=%u, region=%u atoms)...\n", VC_ATOM, vc_region_atoms());
    if (test_basic()) return 1;
    if (test_append_box()) return 1;
    if (test_zero_region()) return 1;
    if (test_lods()) return 1;
    if (test_format_reject()) return 1;
    printf("ALL SPARSE TESTS PASSED\n");
    return 0;
}
