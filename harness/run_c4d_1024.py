#!/usr/bin/env python3
# Drive the c4d arm of the 1024^3 benchmark: for each target ratio, bisect c4d's
# q to hit that ratio (+-5%), then report ratio/PSNR/MS-SSIM/GMSD/enc/dec/ttf.
import subprocess, sys, os
BD="/home/forrest/c3d2/vc-wt-build"
RAW="/home/forrest/c3d2/volume-compressor/data/region_1024.u8"; N=1024
BENCH=BD+"/bench_c4d_1024"; SCORE=BD+"/score"; REC="/tmp/c4d_rec_1024.u8"

def run_q(q):
    out=subprocess.run([BENCH,RAW,str(N),REC,str(q)],capture_output=True,text=True)
    if out.returncode!=0: sys.stderr.write(out.stderr); raise SystemExit("c4d bench fail q=%s"%q)
    ratio,enc,dec,ttf=map(float,out.stdout.split())
    return ratio,enc,dec,ttf

def score():
    out=subprocess.run([SCORE,RAW,REC,str(N)],capture_output=True,text=True)
    psnr,mssim,gmsd=map(float,out.stdout.split())
    return psnr,mssim,gmsd

# seed q-guesses from the 256-sub mapping: 10x~6, 20x~11, 50x~26, 100x~81
SEED={10:6.0,20:11.0,50:26.0,100:81.0}
for tgt in (10,20,50,100):
    # bisect q in log-ish space to hit tgt within 4%
    q=SEED[tgt]; lo,hi=1.0,400.0
    best=None
    for it in range(9):
        ratio,enc,dec,ttf=run_q(q)
        if best is None or abs(ratio-tgt)<abs(best[1]-tgt):
            best=(q,ratio,enc,dec,ttf)
        if abs(ratio-tgt)/tgt<=0.04: break
        if ratio<tgt: lo=q; q=q*1.4 if hi==400.0 else (q+hi)/2
        else: hi=q; q=(lo+q)/2
        sys.stderr.write("  tgt%d it%d q=%.2f ratio=%.2f\n"%(tgt,it,best[0],best[1]))
    q,ratio,enc,dec,ttf=best
    # re-run at best q to leave the matching reconstruction on disk for scoring
    ratio,enc,dec,ttf=run_q(q)
    psnr,mssim,gmsd=score()
    # codec target ratio lod0 psnr mssim gmsd enc_e2e enc_sp dec ttf_us
    print("c4d\t%d\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.2f\t%.1f\t%.2f\t%.2f"%(
        tgt,ratio,ratio,psnr,mssim,gmsd,enc,enc,dec,ttf))
    sys.stdout.flush()
