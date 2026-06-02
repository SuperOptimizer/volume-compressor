#!/usr/bin/env python3
"""compare_c3d_c4d — Phase 0.5 validation backbone (PLAN §0.5).

Runs the reference codecs c3d and c4d AND our configured codec on the SAME
PHerc Paris 4 inputs the rest of the harness uses, and emits a side-by-side
ratio / PSNR / MS-SSIM / GMSD / enc-MB/s / dec-MB/s table.

Strategy
--------
* OURS + c3d : a single linked C binary (compare_c3d_c4d.c) that runs our codec
  (one build per entropy x qweight config) and, when linked with libc3d, the
  c3d reference, on the same raw volume across a q sweep. We build the 9 our-
  codec configs (rice/rlgr/rans x flat/HF/adaptive) once each.
* c4d        : driven via its clean CLI (encode/decode), metrics computed by a
  tiny scorer that reuses our metrics (vc_score helper binary).

If a reference fails to build, we note it and continue with whatever built
(PLAN mandate). All builds are out-of-tree under harness/refbuild/.

Usage:
  compare_c3d_c4d.py [--inputs hires|coarse|both] [--q ...]
"""
import os, sys, subprocess, struct, shutil, argparse, tempfile

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C3D    = os.path.join(os.path.dirname(ROOT), "c3d")
C4D    = os.path.join(os.path.dirname(ROOT), "c4d")
REFB   = os.path.join(ROOT, "harness", "refbuild")
DATA   = os.path.join(ROOT, "data")

ENTROPIES = ["rice", "rlgr", "rans"]
QWEIGHTS  = {"flat": "0", "hf": "1", "adapt": "2"}

def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)

def log(*a): print(*a, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------
def build_c3d():
    bd = os.path.join(C3D, "build")
    lib = os.path.join(bd, "libc3d.a")
    if os.path.exists(lib):
        return lib
    r = run(["cmake", "-S", C3D, "-B", bd, "-DCMAKE_BUILD_TYPE=Release"])
    r = run(["cmake", "--build", bd, "--target", "c3d"])
    return lib if os.path.exists(lib) else None

def build_c4d():
    bd = os.path.join(C4D, "build")
    cli = os.path.join(bd, "c4d")
    if os.path.exists(cli):
        return cli
    args = ["cmake", "-S", C4D, "-B", bd, "-DCMAKE_BUILD_TYPE=Release"]
    if shutil.which("ninja"):
        args += ["-G", "Ninja"]
    run(args)
    run(["cmake", "--build", bd, "--target", "c4d_cli"])
    return cli if os.path.exists(cli) else None

def build_our_config(entropy, qwname, qwval, c3d_lib):
    """Compile compare_c3d_c4d.c against our codec sources for one config."""
    tag = f"{entropy}-{qwname}"
    outdir = os.path.join(REFB, tag)
    os.makedirs(outdir, exist_ok=True)
    binp = os.path.join(outdir, "cmp")
    srcs = [
        "src/codec.c", "src/transform/dct_int.c",
        "src/entropy/rice.c", "src/entropy/rlgr.c", "src/entropy/rans_static.c",
        "src/quant/subband.c", "src/ratectrl/fixed.c", "src/metrics/metrics.c",
        "harness/compare_c3d_c4d.c",
    ]
    srcs = [os.path.join(ROOT, s) for s in srcs]
    ent_map = {"rice": ("vc_rice_encode", "vc_rice_decode", "1"),
               "rlgr": ("vc_rlgr_encode", "vc_rlgr_decode", "2"),
               "rans": ("vc_rans_encode", "vc_rans_decode", "3")}
    enc, dec, etag = ent_map[entropy]
    cmd = ["cc", "-O3", "-ffp-contract=fast", "-ffast-math", "-std=c23",
           "-I", os.path.join(ROOT, "include"),
           f"-DVC_CHUNK_SIDE=64", f"-DVC_QWEIGHT={qwval}",
           f"-DVC_ENTROPY_ENC={enc}", f"-DVC_ENTROPY_DEC={dec}",
           f"-DVC_CFG_ENTROPY_TAG={etag}",
           f'-DVC_TAG_STR="{tag}"']
    if c3d_lib:
        cmd += ["-DHAVE_C3D", "-I", C3D]
    cmd += srcs
    if c3d_lib:
        cmd += [c3d_lib, "-fopenmp"]   # c3d's codec uses OpenMP internally
    cmd += ["-lm", "-o", binp]
    r = run(cmd)
    if r.returncode != 0:
        log(f"[build fail {tag}]\n{r.stderr[-2000:]}")
        return None
    return binp

# ---------------------------------------------------------------------------
# Inputs: assemble real PHerc Paris 4 volumes from the cached chunks.
# ---------------------------------------------------------------------------
def assemble_hires_256():
    """8 hires 128^3 chunks -> one 256^3 cube (2x2x2), saved to a temp raw."""
    d = os.path.join(DATA, "cache_hires")
    files = sorted(f for f in os.listdir(d) if f.endswith(".u8"))
    if len(files) < 8:
        return None
    C = 128
    vol = bytearray(256 * 256 * 256)
    order = files[:8]
    for idx, fn in enumerate(order):
        cz, cy, cx = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        with open(os.path.join(d, fn), "rb") as fh:
            chunk = fh.read()
        for z in range(C):
            for y in range(C):
                src = ((z * C) + y) * C
                vz, vy, vx = cz*C+z, cy*C+y, cx*C
                dst = ((vz*256)+vy)*256 + vx
                vol[dst:dst+C] = chunk[src:src+C]
    p = os.path.join(REFB, "hires_256.u8")
    with open(p, "wb") as fh: fh.write(vol)
    return (p, 256, 256, 256, "hires-256")

def assemble_coarse_256():
    """Crop the coarse 384^3 volume to a 256^3 sub-region."""
    src = os.path.join(DATA, "coarse", "coarse_z15_y8_x8_3x3x3.u8")
    if not os.path.exists(src):
        return None
    S = 384
    with open(src, "rb") as fh: raw = fh.read()
    if len(raw) < S*S*S:
        return None
    C = 256
    vol = bytearray(C*C*C)
    for z in range(C):
        for y in range(C):
            s = ((z*S)+y)*S
            d = ((z*C)+y)*C
            vol[d:d+C] = raw[s:s+C]
    p = os.path.join(REFB, "coarse_256.u8")
    with open(p, "wb") as fh: fh.write(vol)
    return (p, 256, 256, 256, "coarse-256")

# ---------------------------------------------------------------------------
# c4d via CLI
# ---------------------------------------------------------------------------
def build_scorer():
    """Tiny binary: score <ref.u8> <rec.u8> <d> <h> <w> -> psnr ms_ssim gmsd."""
    binp = os.path.join(REFB, "score")
    src = os.path.join(REFB, "score.c")
    with open(src, "w") as f:
        f.write('''#include "../../include/vc/vc.h"
#include "../../src/metrics/metrics.h"
#include <stdio.h>
#include <stdlib.h>
static unsigned char* rd(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f)return 0;
fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);unsigned char*b=malloc(s);
fread(b,1,s,f);fclose(f);*n=s;return b;}
int main(int c,char**v){if(c<6)return 2;size_t a,b;
unsigned char*r=rd(v[1],&a),*q=rd(v[2],&b);
unsigned d=atoi(v[3]),h=atoi(v[4]),w=atoi(v[5]);
vc_metrics m;vc_compute_metrics(r,q,d,h,w,&m);
double g=vc_gmsd(r,q,d,h,w);
printf("%.2f %.4f %.4f\\n",m.psnr,m.ms_ssim,g);return 0;}''')
    r = run(["cc","-O2","-std=c23","-I",os.path.join(ROOT,"include"),
             src, os.path.join(ROOT,"src/metrics/metrics.c"),"-lm","-o",binp])
    if r.returncode != 0:
        log("[scorer build fail]\n"+r.stderr); return None
    return binp

def run_c4d(cli, scorer, raw, d, h, w, label, qs, rows):
    raw_sz = os.path.getsize(raw)
    for q in qs:
        arc = os.path.join(REFB, "c4d.tmp.c4d")
        out = os.path.join(REFB, "c4d.tmp.u8")
        import time
        t0 = time.time()
        e = run([cli, "encode", raw, arc, "--shape", f"{d},{h},{w}", "--q", str(q)])
        t1 = time.time()
        if e.returncode != 0 or not os.path.exists(arc):
            log(f"[c4d encode q={q} fail] {e.stderr[:200]}"); continue
        dd = run([cli, "decode", arc, out]); t2 = time.time()
        if dd.returncode != 0 or not os.path.exists(out):
            log(f"[c4d decode q={q} fail] {dd.stderr[:200]}"); continue
        alen = os.path.getsize(arc)
        sc = run([scorer, raw, out, str(d), str(h), str(w)])
        if sc.returncode != 0: log("[score fail]"); continue
        psnr, ssim, gmsd = sc.stdout.split()
        ratio = raw_sz / alen
        enc_mbs = raw_sz/1e6/max(t1-t0,1e-9); dec_mbs = raw_sz/1e6/max(t2-t1,1e-9)
        rows.append(("c4d", label, float(q), ratio, float(psnr), float(ssim),
                     float(gmsd), enc_mbs, dec_mbs, "ref-c4d"))

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="both", choices=["hires","coarse","both"])
    ap.add_argument("--q", nargs="*", type=float,
                    default=[2,4,8,16,32,64,128])
    ap.add_argument("--configs", default="all",
                    help="comma list of entropy-qweight tags, or 'all' or 'lead'")
    args = ap.parse_args()
    os.makedirs(REFB, exist_ok=True)

    log("== building references ==")
    c3d_lib = build_c3d()
    log(f"c3d: {'OK '+c3d_lib if c3d_lib else 'FAILED (continuing without)'}")
    c4d_cli = build_c4d()
    log(f"c4d: {'OK '+c4d_cli if c4d_cli else 'FAILED (continuing without)'}")
    scorer = build_scorer()

    # select our configs
    if args.configs == "all":
        combos = [(e,wn,wv) for e in ENTROPIES for wn,wv in QWEIGHTS.items()]
    elif args.configs == "lead":
        combos = [("rlgr","hf",QWEIGHTS["hf"]), ("rans","hf",QWEIGHTS["hf"]),
                  ("rice","flat",QWEIGHTS["flat"])]
    else:
        combos = []
        for tag in args.configs.split(","):
            e, wn = tag.split("-"); combos.append((e, wn, QWEIGHTS[wn]))

    log("== building our %d configs ==" % len(combos))
    ourbins = {}
    for e,wn,wv in combos:
        b = build_our_config(e, wn, wv, c3d_lib)
        if b: ourbins[f"{e}-{wn}"] = b

    inputs = []
    if args.inputs in ("hires","both"):
        x = assemble_hires_256();  inputs.append(x) if x else log("[no hires data]")
    if args.inputs in ("coarse","both"):
        x = assemble_coarse_256(); inputs.append(x) if x else log("[no coarse data]")

    rows = []
    qstr = [str(q) for q in args.q]
    for raw,d,h,w,label in inputs:
        log(f"== input {label} ({d}x{h}x{w}) ==")
        # our configs + c3d (c3d runs inside the FIRST our-binary that has it
        # linked; to avoid duplicate c3d rows, run c3d only once via any binary)
        c3d_done = False
        for tag, binp in ourbins.items():
            r = run([binp, raw, str(d), str(h), str(w), label] + qstr)
            if r.returncode != 0:
                log(f"[run {tag} fail] {r.stderr[:300]}"); continue
            for line in r.stdout.strip().splitlines():
                f = line.split("\t")
                codec = f[0]
                if codec == "c3d":
                    if c3d_done: continue
                    rows.append((codec,f[1],float(f[2]),float(f[3]),float(f[4]),
                                 float(f[5]),float(f[6]),float(f[7]),float(f[8]),f[9]))
                else:
                    rows.append((codec,f[1],float(f[2]),float(f[3]),float(f[4]),
                                 float(f[5]),float(f[6]),float(f[7]),float(f[8]),tag))
            c3d_done = True
        # c4d
        if c4d_cli and scorer:
            run_c4d(c4d_cli, scorer, raw, d, h, w, label, args.q, rows)

    # ---- print the side-by-side table ----
    print("\n# codec/config\tinput\tq\tratio\tPSNR\tMS-SSIM\tGMSD\tenc MB/s\tdec MB/s")
    rows.sort(key=lambda r:(r[1], -r[3]))
    for r in rows:
        print(f"{r[9]:<16}\t{r[1]:<11}\t{r[2]:>6.4g}\t{r[3]:>7.1f}\t{r[4]:>6.2f}"
              f"\t{r[5]:>6.4f}\t{r[6]:>6.4f}\t{r[7]:>7.0f}\t{r[8]:>7.0f}")

if __name__ == "__main__":
    main()
