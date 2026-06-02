// Block-grid (chunk-model) self-test: round-trip + RANDOM-ACCESS correctness.
// The key invariant (spec §5): encode -> decode ONE random 16^3 atom matches the
// full decode of that atom, for every {stencil x traversal x edge x chunk-size x
// entropy} combination exercised here.
#include "../src/chunkmodel/blockgrid.h"
#include "../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int fail = 0;
#define CHECK(c,m) do{ if(!(c)){printf("FAIL: %s\n",m); fail=1;} else printf("ok: %s\n",m);}while(0)

static u8 *gen(u32 d, u32 h, u32 w, int kind) {
    u8 *v = malloc((size_t)d*h*w);
    for (u32 z=0; z<d; ++z) for (u32 y=0; y<h; ++y) for (u32 x=0; x<w; ++x) {
        int val;
        if (kind==0) val = (int)(128 + 60*sin(x*0.08)*cos(y*0.06+z*0.05));
        else if (kind==1) val = (int)(100 + 50*sin((x+y+z)*0.04) + 30*((((x/13)+(y/11)+(z/9))&1)?1:0));
        else val = (int)((x*7+y*13+z*3)&0xff);
        v[((size_t)z*h+y)*w+x] = (u8)(val<0?0:(val>255?255:val));
    }
    return v;
}

// Extract the reference 16^3 atom (az,ay,ax) directly from a full-decoded volume.
static void ref_atom(const u8 *vol, u32 d, u32 h, u32 w, u32 az, u32 ay, u32 ax, u8 *out) {
    u32 oz=az*16, oy=ay*16, ox=ax*16;
    for (u32 z=0; z<16; ++z) for (u32 y=0; y<16; ++y) for (u32 x=0; x<16; ++x) {
        u32 vz=oz+z, vy=oy+y, vx=ox+x;
        out[(z*16+y)*16+x] = (vz<d && vy<h && vx<w) ? vol[((size_t)vz*h+vy)*w+vx] : 0;
    }
}

static int one(u32 d, u32 h, u32 w, int kind, vc_bg_cfg cfg, const char *name) {
    u8 *vol = gen(d,h,w,kind);
    vc_bg_archive *a = NULL; vc_bg_stats st;
    if (vc_bg_encode(vol, d,h,w, &cfg, &a, &st)) { printf("FAIL encode %s\n",name); free(vol); return 1; }
    u8 *rec = calloc((size_t)d*h*w,1);
    vc_bg_decode(a, rec);
    vc_metrics m; vc_compute_metrics(vol, rec, d,h,w, &m);
    double ratio = (double)((size_t)d*h*w)/st.total_bytes;

    // Random-access: decode several atoms, compare to full-decode atoms.
    u32 Az=vc_bg_natoms(d), Ay=vc_bg_natoms(h), Ax=vc_bg_natoms(w);
    int ra_ok = 1; u32 maxtouch = 0;
    u8 a1[4096], a2[4096];
    unsigned seed = 12345u + kind;
    for (int t=0; t<24; ++t) {
        seed = seed*1103515245u+12345u;
        u32 az=(seed>>16)%Az; seed=seed*1103515245u+12345u;
        u32 ay=(seed>>16)%Ay; seed=seed*1103515245u+12345u;
        u32 ax=(seed>>16)%Ax;
        u32 touched=0;
        vc_bg_decode_atom(a, az,ay,ax, a1, &touched);
        if (touched>maxtouch) maxtouch=touched;
        ref_atom(rec, d,h,w, az,ay,ax, a2);
        // Compare ONLY in-bounds voxels: random access reconstructs the full 16^3
        // atom incl. padding cols/rows past the volume edge, which the full decode
        // discards (scatter clips). Padding is undefined; the in-bounds region is
        // the random-access correctness contract.
        u32 oz=az*16, oy=ay*16, ox=ax*16; int diff=0;
        for (u32 z=0; z<16; ++z) for (u32 y=0; y<16; ++y) for (u32 x=0; x<16; ++x) {
            if (oz+z>=d || oy+y>=h || ox+x>=w) continue;
            if (a1[(z*16+y)*16+x] != a2[(z*16+y)*16+x]) diff++;
        }
        if (diff!=0) {
            printf("   RA mismatch %s atom(%u,%u,%u): %d in-bounds voxels differ\n",name,az,ay,ax,diff);
            ra_ok=0; break;
        }
    }
    printf("  %-30s PSNR=%.2f ratio=%.1f RA_match=%d maxtouch=%u\n",
           name, m.psnr, ratio, ra_ok, maxtouch);
    // Curve-group + DC-sub-volume + no-stencil paths MUST keep touched==1.
    int touch_ok = 1;
    if (cfg.group_mode == VC_GROUP_CURVE || cfg.dc_subvol
        || cfg.stencil == VC_STENCIL_NONE)
        touch_ok = (maxtouch == 1);
    vc_bg_free(a); free(vol); free(rec);
    return (m.psnr > 24.0 && ra_ok && touch_ok) ? 0 : 1;
}

int main(void) {
    vc_bg_cfg base = { VC_STENCIL_NONE, VC_TRAV_RASTER, VC_EDGE_SELF,
                       VC_ENT_RANS_SHARED, 4, 1, 8.0f, 0 };

    // M1 baselines across entropy modes
    vc_bg_cfg c;
    c=base; c.entropy=VC_ENT_RANS_SHARED; CHECK(one(128,128,128,0,c,"M1 rans-shared none/raster")==0,"M1 rans");
    c=base; c.entropy=VC_ENT_RANS_INDEP;  CHECK(one(128,128,128,0,c,"M0 rans-indep none/raster")==0,"M0 rans");
    c=base; c.entropy=VC_ENT_RLGR;        CHECK(one(128,128,128,0,c,"RLGR none/raster")==0,"RLGR");

    // Stencils
    c=base; c.stencil=VC_STENCIL_6;  CHECK(one(128,128,128,1,c,"6-conn raster self")==0,"stencil-6");
    c=base; c.stencil=VC_STENCIL_18; CHECK(one(128,128,128,1,c,"18-conn raster self")==0,"stencil-18");
    c=base; c.stencil=VC_STENCIL_26; CHECK(one(128,128,128,1,c,"26-conn raster self")==0,"stencil-26");

    // Traversals (with prediction)
    c=base; c.stencil=VC_STENCIL_6; c.traversal=VC_TRAV_MORTON;  CHECK(one(128,128,128,1,c,"6-conn morton self")==0,"morton");
    c=base; c.stencil=VC_STENCIL_6; c.traversal=VC_TRAV_HILBERT; CHECK(one(128,128,128,1,c,"6-conn hilbert self")==0,"hilbert");

    // Edge policies (cross-chunk prediction; chunk_atoms small so many faces)
    c=base; c.stencil=VC_STENCIL_6; c.chunk_atoms=2; c.edge=VC_EDGE_SELF;  CHECK(one(96,96,96,1,c,"6-conn ca2 self")==0,"edge-self");
    c=base; c.stencil=VC_STENCIL_6; c.chunk_atoms=2; c.edge=VC_EDGE_HALO;  CHECK(one(96,96,96,1,c,"6-conn ca2 halo")==0,"edge-halo");
    c=base; c.stencil=VC_STENCIL_6; c.chunk_atoms=2; c.edge=VC_EDGE_FETCH; CHECK(one(96,96,96,1,c,"6-conn ca2 fetch")==0,"edge-fetch");

    // Chunk sizes
    c=base; c.stencil=VC_STENCIL_6; c.chunk_atoms=8;  CHECK(one(128,128,128,1,c,"6-conn ca8")==0,"ca8");
    c=base; c.stencil=VC_STENCIL_6; c.chunk_atoms=16; CHECK(one(256,256,256,1,c,"6-conn ca16")==0,"ca16");

    // DC sub-volume (global predicted DC frame) — keeps random access O(1).
    c=base; c.dc_subvol=1; c.stencil=VC_STENCIL_NONE; CHECK(one(128,128,128,0,c,"DCsv none rans")==0,"dc-subvol rans");
    c=base; c.dc_subvol=1; c.entropy=VC_ENT_RLGR;     CHECK(one(128,128,128,1,c,"DCsv none rlgr")==0,"dc-subvol rlgr");
    c=base; c.dc_subvol=1; CHECK(one(100,70,53,1,c,"DCsv unaligned")==0,"dc-subvol unaligned");

    // Unaligned shape (edge padding path)
    c=base; c.stencil=VC_STENCIL_6; CHECK(one(100,70,53,1,c,"6-conn unaligned")==0,"unaligned");

    // --- CURVE-GROUP mode (curve-groups experiment): round-trip + RA, touched=1.
    // Sharing scope = N consecutive atoms along a global curve, no chunk boxes.
    vc_bg_cfg cg = { VC_STENCIL_NONE, VC_TRAV_MORTON, VC_EDGE_SELF,
                     VC_ENT_RANS_SHARED, 8, 1, 8.0f, 0, VC_GROUP_CURVE, 64 };
    c=cg; c.traversal=VC_TRAV_MORTON;  c.entropy=VC_ENT_RANS_SHARED; c.group_n=64;
        CHECK(one(128,128,128,0,c,"curve Morton N=64 rans")==0,"curve-morton-64-rans");
    c=cg; c.traversal=VC_TRAV_HILBERT; c.entropy=VC_ENT_RANS_SHARED; c.group_n=256;
        CHECK(one(128,128,128,1,c,"curve Hilbert N=256 rans")==0,"curve-hilbert-256-rans");
    c=cg; c.traversal=VC_TRAV_HILBERT; c.entropy=VC_ENT_RLGR;        c.group_n=512;
        CHECK(one(128,128,128,1,c,"curve Hilbert N=512 rlgr")==0,"curve-hilbert-512-rlgr");
    c=cg; c.traversal=VC_TRAV_MORTON;  c.entropy=VC_ENT_RLGR;        c.group_n=256;
        CHECK(one(100,70,53,2,c,"curve Morton N=256 rlgr unaligned")==0,"curve-morton-256-unaligned");

    // --- THREE-LEVEL HIERARCHY variants (E1 table-delta, E2 base, I1 drift, B1
    // DC-curve-pred, I2 nested). Each must round-trip + keep random access touched=1.
    // E1: group-to-group table DELTA (rANS only; affects bytes, must round-trip).
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=64;
        c.table_coding=VC_TABLE_DELTA;
        CHECK(one(128,128,128,0,c,"curve Hilbert N=64 rans E1-tabledelta")==0,"E1-table-delta");
    // E2: coarse super-group base table + per-group delta.
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=64;
        c.table_coding=VC_TABLE_BASE;
        CHECK(one(128,128,128,1,c,"curve Hilbert N=64 rans E2-base")==0,"E2-base");
    // I1: drift-adaptive group boundaries (variable size).
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=256;
        c.boundary=VC_BOUND_DRIFT; c.drift_thresh=0.15f;
        CHECK(one(128,128,128,1,c,"curve Hilbert drift rans")==0,"I1-drift-rans");
    c=cg; c.entropy=VC_ENT_RLGR; c.traversal=VC_TRAV_MORTON; c.group_n=256;
        c.boundary=VC_BOUND_DRIFT; c.drift_thresh=0.10f;
        CHECK(one(100,70,53,2,c,"curve Morton drift rlgr unaligned")==0,"I1-drift-rlgr-unaligned");
    // I1+E1 combined (the spec's likely-winner shape).
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=128;
        c.boundary=VC_BOUND_DRIFT; c.drift_thresh=0.15f; c.table_coding=VC_TABLE_DELTA;
        CHECK(one(128,128,128,0,c,"curve Hilbert drift+E1 rans")==0,"I1+E1");
    // B1: DC-only curve-predecessor prediction (touched must stay 1).
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=256;
        c.dc_pred_curve=1;
        CHECK(one(128,128,128,0,c,"curve Hilbert N=256 B1-dcpred")==0,"B1-dc-pred");
    c=cg; c.entropy=VC_ENT_RLGR; c.traversal=VC_TRAV_MORTON; c.group_n=64;
        c.dc_pred_curve=1;
        CHECK(one(100,70,53,1,c,"curve Morton B1-dcpred unaligned")==0,"B1-dc-pred-unaligned");
    // I2: nested sub-groups (index overhead; must still round-trip).
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=256;
        c.nested_sub=16;
        CHECK(one(128,128,128,1,c,"curve Hilbert N=256 I2-nested")==0,"I2-nested");
    // Full stack: drift + E1 + B1 + Hilbert.
    c=cg; c.entropy=VC_ENT_RANS_SHARED; c.traversal=VC_TRAV_HILBERT; c.group_n=128;
        c.boundary=VC_BOUND_DRIFT; c.drift_thresh=0.15f; c.table_coding=VC_TABLE_DELTA;
        c.dc_pred_curve=1;
        CHECK(one(128,128,128,0,c,"curve full-stack rans")==0,"full-stack");

    // --- EG2024 experiment: per-band shared tables, sparse prepass, skip-meta.
    // All on the box rANS-shared path (band split is a no-op under RLGR). Each
    // must round-trip + keep random-access touched==1 (stencil NONE).
    // (1) DC|AC split (2 tables)
    c=base; c.entropy=VC_ENT_RANS_SHARED; c.chunk_atoms=8; c.band_split=VC_BAND_DC_AC;
        CHECK(one(128,128,128,0,c,"EG band DC|AC rans")==0,"eg-band-2");
    // (1) DC|lowAC|hiAC split (3 tables)
    c=base; c.entropy=VC_ENT_RANS_SHARED; c.chunk_atoms=8; c.band_split=VC_BAND_DC_LO_HI;
        CHECK(one(128,128,128,1,c,"EG band DC|lo|hi rans")==0,"eg-band-3");
    c=base; c.entropy=VC_ENT_RANS_SHARED; c.band_split=VC_BAND_DC_LO_HI;
        CHECK(one(100,70,53,1,c,"EG band DC|lo|hi unaligned")==0,"eg-band-3-unaligned");
    // (3) sparse prepass (stride 8) — table from a sample, payload codes all atoms
    c=base; c.entropy=VC_ENT_RANS_SHARED; c.chunk_atoms=8; c.sparse_prepass=8;
        CHECK(one(128,128,128,0,c,"EG sparse-prepass/8 rans")==0,"eg-sparse");
    // (3)+(1) sparse prepass + band split together
    c=base; c.entropy=VC_ENT_RANS_SHARED; c.band_split=VC_BAND_DC_LO_HI; c.sparse_prepass=16;
        CHECK(one(128,128,128,1,c,"EG band3 + sparse/16")==0,"eg-band3-sparse");
    // (4) skip-meta on uniform (constant) volume -> uniform chunks, RA constant.
    {
        u32 d=128,h=128,w=128; u8 *uv=malloc((size_t)d*h*w); memset(uv,77,(size_t)d*h*w);
        vc_bg_cfg cu=base; cu.entropy=VC_ENT_RANS_SHARED; cu.chunk_atoms=8; cu.skip_meta=1;
        vc_bg_archive *ua=NULL; vc_bg_stats us;
        int ok = (vc_bg_encode(uv,d,h,w,&cu,&ua,&us)==0);
        u8 *ur=calloc((size_t)d*h*w,1); if(ok) vc_bg_decode(ua,ur);
        int same=1; for(size_t i=0;i<(size_t)d*h*w;++i) if(ur[i]!=77){same=0;break;}
        u32 tch=0; u8 at1[4096]; if(ok) vc_bg_decode_atom(ua,1,1,1,at1,&tch);
        int constok=1; for(int i=0;i<4096;++i) if(at1[i]!=77){constok=0;break;}
        printf("  EG skip-meta uniform: uniform_chunks=%u skip_meta_bytes=%zu RA_match=%d touched=%u\n",
               us.n_uniform_chunks, us.skip_meta_bytes, (same&&constok), tch);
        CHECK(ok && same && constok && us.n_uniform_chunks>0 && tch==1,"eg-skip-meta-uniform");
        vc_bg_free(ua); free(uv); free(ur);
    }

    // --- RDOQ (EXPERIMENT #19): encode-only, decode/bitstream layout unchanged,
    // so the round-trip + RA + touched==1 contract must still hold. one() already
    // checks decode == re-decode and touched for NONE-stencil paths.
    c=base; c.entropy=VC_ENT_RLGR;         c.chunk_atoms=8; c.rdoq=1; c.rdoq_lambda=0.06f;
        CHECK(one(128,128,128,0,c,"RDOQ RLGR l=0.06 box")==0,"rdoq-rlgr");
    c=base; c.entropy=VC_ENT_RANS_SHARED;  c.chunk_atoms=8; c.rdoq=1; c.rdoq_lambda=0.15f;
        CHECK(one(128,128,128,0,c,"RDOQ rans l=0.15 box")==0,"rdoq-rans");
    c=base; c.entropy=VC_ENT_RLGR;         c.dc_subvol=1;   c.rdoq=1; c.rdoq_lambda=0.10f;
        CHECK(one(128,128,128,0,c,"RDOQ RLGR DCsv")==0,"rdoq-dcsv");
    c=base; c.entropy=VC_ENT_RLGR;         c.chunk_atoms=8; c.rdoq=1; c.rdoq_lambda=0.10f;
        CHECK(one(100,70,53,0,c,"RDOQ RLGR unaligned")==0,"rdoq-unaligned");

    printf(fail ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return fail;
}
