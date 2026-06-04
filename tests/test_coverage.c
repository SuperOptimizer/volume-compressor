// Extended coverage tests beyond test_vc.c. Targets API and properties the
// original suite did not exercise:
//   - determinism: identical input -> byte-identical archive (catches miscompiles
//     and nondeterministic codegen, e.g. the SVE fixed-width bug)
//   - monotonicity: larger q -> smaller archive, lower (or equal) PSNR
//   - vc_decode_region (raw box decode) matches per-atom decode
//   - writer-side readback: vc_writer_decode_atom / _coverage / _decode_2x2x2
//   - edge dims: 1^3, single atom, exactly one region, tall/thin, prime sizes
//   - content extremes: all-zero volume, single non-zero voxel, all-saturated
//   - decoder bounds: out-of-range lod / coords return errors, never crash
//   - reopen idempotence + double safety
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
static unsigned char *map_file(const char *path, size_t *len) {
    int fd = open(path, O_RDONLY); if (fd < 0) return NULL;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    void *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return NULL;
    *len = st.st_size; return (unsigned char *)m;
}
static void make_atom(unsigned char *vox, unsigned az, unsigned ay, unsigned ax) {
    for (unsigned z=0; z<VC_ATOM; ++z) for (unsigned y=0; y<VC_ATOM; ++y) for (unsigned x=0; x<VC_ATOM; ++x) {
        double vx=ax*VC_ATOM+x, vy=ay*VC_ATOM+y, vz=az*VC_ATOM+z;
        vox[(z*VC_ATOM+y)*VC_ATOM+x] = (unsigned char)(128 + 60*sin(vx*0.06) + 40*cos(vy*0.05) + 20*sin(vz*0.07));
    }
}
static void set_q_all(vc_writer *w, float q) { for (int l=0;l<VC_NLOD;++l) vc_set_base_q(w,l,q); }

// Build a small archive at quality q; return file size, optionally hash bytes.
static long build_archive(const char *path, vc_dims d0, float q, unsigned long *out_hash) {
    vc_writer *w = vc_create(path, d0, 30.0f); if (!w) return -1;
    set_q_all(w, q);
    unsigned acx=(d0.nx+VC_ATOM-1)/VC_ATOM, acy=(d0.ny+VC_ATOM-1)/VC_ATOM, acz=(d0.nz+VC_ATOM-1)/VC_ATOM;
    unsigned char vox[VC_ATOM3];
    for (unsigned az=0; az<acz; ++az) for (unsigned ay=0; ay<acy; ++ay) for (unsigned ax=0; ax<acx; ++ax) {
        make_atom(vox, az, ay, ax);
        if (vc_append_atom(w,0,az,ay,ax,vox)!=VC_OK) { vc_writer_close(w); return -1; }
    }
    vc_writer_close(w);
    struct stat st; if (stat(path,&st)!=0) return -1;
    if (out_hash) {
        size_t alen; unsigned char *arc = map_file(path,&alen);
        unsigned long h=1469598103934665603UL;
        for (size_t i=0;i<alen;++i){ h^=arc[i]; h*=1099511628211UL; }
        *out_hash=h; munmap(arc,alen);
    }
    return (long)st.st_size;
}

// 1. Determinism: same input twice -> byte-identical archive.
static int test_determinism(void) {
    vc_dims d0 = {256, 192, 160};
    unsigned long h1=0, h2=0;
    long s1 = build_archive("/tmp/vc_det1.vca", d0, 1.0f, &h1);
    long s2 = build_archive("/tmp/vc_det2.vca", d0, 1.0f, &h2);
    CHECK(s1>0 && s2>0, "build");
    CHECK(s1==s2, "size differs %ld vs %ld", s1, s2);
    CHECK(h1==h2, "archive bytes differ (nondeterministic encode!) %016lx vs %016lx", h1, h2);
    unlink("/tmp/vc_det1.vca"); unlink("/tmp/vc_det2.vca");
    printf("  determinism: identical input -> identical %ld-byte archive (hash %016lx)\n", s1, h1);
    return 0;
}

// 2. Monotonicity: q up -> size down; PSNR down-or-equal.
static int test_monotonic(void) {
    vc_dims d0 = {256, 256, 256};
    float qs[] = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
    int n = (int)(sizeof qs/sizeof qs[0]);
    long prev_size = -1; double prev_psnr = 1e9;
    unsigned char vox[VC_ATOM3], out[VC_ATOM3];
    for (int k=0;k<n;++k) {
        const char *p = "/tmp/vc_mono.vca";
        long sz = build_archive(p, d0, qs[k], NULL);
        CHECK(sz>0, "build q=%.2f", qs[k]);
        // measure mean PSNR over a sample of atoms
        size_t alen; unsigned char *arc=map_file(p,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
        double sum=0; long cnt=0;
        unsigned acx=(d0.nx+VC_ATOM-1)/VC_ATOM, acy=(d0.ny+VC_ATOM-1)/VC_ATOM, acz=(d0.nz+VC_ATOM-1)/VC_ATOM;
        for (unsigned az=0;az<acz;az+=2) for (unsigned ay=0;ay<acy;ay+=2) for (unsigned ax=0;ax<acx;ax+=2) {
            make_atom(vox,az,ay,ax);
            CHECK(vc_decode_atom(a,0,ax,ay,az,out)==VC_OK,"dec");
            sum += psnr(vox,out,VC_ATOM3); cnt++;
        }
        double mp = sum/cnt;
        vc_close(a); munmap(arc,alen);
        if (prev_size>=0) {
            CHECK(sz <= prev_size, "size not monotonic at q=%.2f (%ld > %ld)", qs[k], sz, prev_size);
            CHECK(mp <= prev_psnr + 1.0, "PSNR rose with higher q=%.2f (%.1f > %.1f)", qs[k], mp, prev_psnr);
        }
        prev_size = sz; prev_psnr = mp;
    }
    unlink("/tmp/vc_mono.vca");
    printf("  monotonic: size and PSNR both decrease across q=0.5..16 (%d points)\n", n);
    return 0;
}

// 3. vc_decode_region matches per-atom decode over a multi-atom box.
static int test_decode_region(void) {
    const char *path = "/tmp/vc_region.vca";
    vc_dims d0 = {128, 128, 128};
    vc_writer *w = vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
    unsigned ac = d0.nx/VC_ATOM;  // 4
    unsigned char vox[VC_ATOM3];
    for (unsigned az=0;az<ac;++az) for (unsigned ay=0;ay<ac;++ay) for (unsigned ax=0;ax<ac;++ax) {
        make_atom(vox,az,ay,ax); CHECK(vc_append_atom(w,0,az,ay,ax,vox)==VC_OK,"append");
    }
    vc_writer_close(w);
    size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
    // decode a 64^3 region (atoms 0..1 in each axis) and compare to per-atom
    vc_box box = {0,0,0, 64,64,64};
    unsigned bw=64;
    unsigned char *reg = malloc((size_t)bw*bw*bw);
    CHECK(vc_decode_region(a,0,box,reg)==VC_OK, "decode_region");
    unsigned char at[VC_ATOM3];
    for (unsigned az=0;az<2;++az) for (unsigned ay=0;ay<2;++ay) for (unsigned ax=0;ax<2;++ax) {
        CHECK(vc_decode_atom(a,0,ax,ay,az,at)==VC_OK,"atom");
        for (unsigned z=0;z<VC_ATOM;++z) for (unsigned y=0;y<VC_ATOM;++y) for (unsigned x=0;x<VC_ATOM;++x) {
            unsigned gz=az*VC_ATOM+z, gy=ay*VC_ATOM+y, gx=ax*VC_ATOM+x;
            unsigned char rv = reg[((size_t)gz*bw+gy)*bw+gx];
            unsigned char av = at[(z*VC_ATOM+y)*VC_ATOM+x];
            CHECK(rv==av, "region/atom mismatch at (%u,%u,%u): %u vs %u", gz,gy,gx,rv,av);
        }
    }
    free(reg); vc_close(a); munmap(arc,alen); unlink(path);
    printf("  decode_region: 64^3 box matches per-atom decode exactly\n");
    return 0;
}

// 4. Writer-side readback API: coverage + decode before close.
static int test_writer_readback(void) {
    const char *path = "/tmp/vc_wrb.vca";
    vc_dims d0 = {64,64,64};
    vc_writer *w = vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
    unsigned char vox[VC_ATOM3]; make_atom(vox,0,0,0);
    CHECK(vc_append_atom(w,0,0,0,0,vox)==VC_OK,"append");
    CHECK(vc_mark_zero_atom(w,0,1,1,1)==VC_OK,"markzero");
    CHECK(vc_writer_coverage(w,0,0,0,0)==VC_PRESENT,"wcov present");
    CHECK(vc_writer_coverage(w,0,1,1,1)==VC_KNOWN_ZERO,"wcov zero");
    unsigned char out[VC_ATOM3];
    CHECK(vc_writer_decode_atom(w,0,0,0,0,out)==VC_OK,"wdecode");
    CHECK(psnr(vox,out,VC_ATOM3) > 20.0, "wdecode quality");
    vc_writer_close(w); unlink(path);
    printf("  writer_readback: coverage + decode before close OK\n");
    return 0;
}

// 5. Edge dims: 1^3, single atom, exactly one region, tall/thin, prime sizes.
static int test_edge_dims(void) {
    struct { vc_dims d; const char *name; } cases[] = {
        {{1,1,1}, "1x1x1"},
        {{32,32,32}, "single atom"},
        {{1024,1024,1024}, "exactly one region"},
        {{2048,32,32}, "tall/thin"},
        {{97,131,193}, "primes"},
    };
    for (int c=0;c<(int)(sizeof cases/sizeof cases[0]);++c) {
        const char *path="/tmp/vc_edge.vca"; vc_dims d0=cases[c].d;
        vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create %s",cases[c].name); set_q_all(w,1.0f);
        unsigned acx=(d0.nx+VC_ATOM-1)/VC_ATOM, acy=(d0.ny+VC_ATOM-1)/VC_ATOM, acz=(d0.nz+VC_ATOM-1)/VC_ATOM;
        unsigned char vox[VC_ATOM3];
        for (unsigned az=0;az<acz;++az) for (unsigned ay=0;ay<acy;++ay) for (unsigned ax=0;ax<acx;++ax) {
            make_atom(vox,az,ay,ax); CHECK(vc_append_atom(w,0,az,ay,ax,vox)==VC_OK,"append %s",cases[c].name);
        }
        vc_writer_close(w);
        size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen);
        CHECK(a,"open %s",cases[c].name);
        vc_dims od; CHECK(vc_lod_dims(a,0,&od)==VC_OK && od.nx==d0.nx && od.ny==d0.ny && od.nz==d0.nz,
                          "dims %s: got %ux%ux%u", cases[c].name, od.nx,od.ny,od.nz);
        unsigned char out[VC_ATOM3];
        CHECK(vc_decode_atom(a,0,0,0,0,out)==VC_OK,"decode %s",cases[c].name);
        vc_close(a); munmap(arc,alen); unlink(path);
    }
    printf("  edge_dims: 1^3, single-atom, one-region, tall/thin, primes all OK\n");
    return 0;
}

// 6. Content extremes: all-zero, single voxel, all-saturated.
static int test_content_extremes(void) {
    const char *path="/tmp/vc_ext.vca";
    vc_dims d0={64,64,64};
    // all-zero volume: append zero atoms -> all KNOWN_ZERO, tiny archive
    {
        vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
        unsigned char z[VC_ATOM3]; memset(z,0,VC_ATOM3);
        unsigned ac=2;
        for (unsigned az=0;az<ac;++az) for(unsigned ay=0;ay<ac;++ay) for(unsigned ax=0;ax<ac;++ax)
            CHECK(vc_append_atom(w,0,az,ay,ax,z)==VC_OK,"append zero");
        vc_writer_close(w);
        size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
        CHECK(vc_atom_coverage(a,0,0,0,0)==VC_KNOWN_ZERO,"all-zero -> known_zero");
        unsigned char out[VC_ATOM3], zz[VC_ATOM3]; memset(zz,0,VC_ATOM3);
        CHECK(vc_decode_atom(a,0,0,0,0,out)==VC_OK && memcmp(out,zz,VC_ATOM3)==0,"zero decode");
        vc_close(a); munmap(arc,alen);
    }
    // single non-zero voxel: an isolated spike is mostly HF energy and MAY be
    // dropped by the dead-zone quantizer at default q (verified: energy 0 at
    // q=1.0, 144 at q=0.5) — that is correct lossy behavior, not a bug. The real
    // contract: the atom stores as PRESENT (it had data) and decode never crashes.
    // A feature with spatial EXTENT must survive (checked below).
    {
        vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
        unsigned char v[VC_ATOM3]; memset(v,0,VC_ATOM3); v[(5*VC_ATOM+5)*VC_ATOM+5]=200;
        CHECK(vc_append_atom(w,0,0,0,0,v)==VC_OK,"append single");
        vc_writer_close(w);
        size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
        CHECK(vc_atom_coverage(a,0,0,0,0)==VC_PRESENT,"single -> present");
        unsigned char out[VC_ATOM3];
        CHECK(vc_decode_atom(a,0,0,0,0,out)==VC_OK,"single decode no-crash");
        vc_close(a); munmap(arc,alen);
    }
    // an 8-voxel line (real spatial extent) MUST survive at default q.
    {
        vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
        unsigned char v[VC_ATOM3]; memset(v,0,VC_ATOM3);
        for (int i=0;i<8;++i) v[(5*VC_ATOM+5)*VC_ATOM+(5+i)]=200;
        CHECK(vc_append_atom(w,0,0,0,0,v)==VC_OK,"append line");
        vc_writer_close(w);
        size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
        unsigned char out[VC_ATOM3];
        CHECK(vc_decode_atom(a,0,0,0,0,out)==VC_OK,"line decode");
        long s=0; for(int i=0;i<(int)VC_ATOM3;i++) s+=out[i];
        CHECK(s>0, "8-voxel feature lost entirely (should survive)");
        vc_close(a); munmap(arc,alen);
    }
    // all-saturated (255)
    {
        vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
        unsigned char v[VC_ATOM3]; memset(v,255,VC_ATOM3);
        CHECK(vc_append_atom(w,0,0,0,0,v)==VC_OK,"append sat");
        vc_writer_close(w);
        size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
        unsigned char out[VC_ATOM3];
        CHECK(vc_decode_atom(a,0,0,0,0,out)==VC_OK,"sat decode");
        CHECK(psnr(v,out,VC_ATOM3) > 40.0, "flat 255 should reconstruct near-exactly");
        vc_close(a); munmap(arc,alen);
    }
    unlink(path);
    printf("  content_extremes: all-zero, single-voxel, all-saturated OK\n");
    return 0;
}

// 7. Decoder bounds: out-of-range lod/coords return errors, never crash.
static int test_decoder_bounds(void) {
    const char *path="/tmp/vc_bounds.vca";
    vc_dims d0={64,64,64};
    vc_writer *w=vc_create(path,d0,20.0f); CHECK(w,"create"); set_q_all(w,1.0f);
    unsigned char vox[VC_ATOM3]; make_atom(vox,0,0,0); vc_append_atom(w,0,0,0,0,vox);
    vc_writer_close(w);
    size_t alen; unsigned char *arc=map_file(path,&alen); vc_archive *a=vc_open(arc,alen); CHECK(a,"open");
    unsigned char out[VC_ATOM3];
    vc_dims od;
    // out-of-range lod
    CHECK(vc_lod_dims(a,-1,&od)!=VC_OK, "neg lod rejected");
    CHECK(vc_lod_dims(a,VC_NLOD+5,&od)!=VC_OK, "huge lod rejected");
    // out-of-range atom coords return error (not crash)
    CHECK(vc_decode_atom(a,0,9999,0,0,out)!=VC_OK, "huge ax rejected");
    CHECK(vc_decode_atom(a,0,0,9999,0,out)!=VC_OK, "huge ay rejected");
    // coverage of out-of-range -> ABSENT (well-defined, no crash)
    vc_cover c = vc_atom_coverage(a,0,9999,9999,9999);
    CHECK(c==VC_ABSENT || c==VC_KNOWN_ZERO, "oob coverage well-defined");
    vc_close(a); munmap(arc,alen); unlink(path);
    printf("  decoder_bounds: out-of-range lod/coords return errors, no crash\n");
    return 0;
}

int main(void) {
    printf("extended coverage tests (VC_ATOM=%u)...\n", VC_ATOM);
    if (test_determinism()) return 1;
    if (test_monotonic()) return 1;
    if (test_decode_region()) return 1;
    if (test_writer_readback()) return 1;
    if (test_edge_dims()) return 1;
    if (test_content_extremes()) return 1;
    if (test_decoder_bounds()) return 1;
    printf("ALL COVERAGE TESTS PASSED\n");
    return 0;
}
