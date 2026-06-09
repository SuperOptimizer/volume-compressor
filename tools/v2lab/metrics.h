// Wide metric basket for the v2 bake-off (u8 reference vs reconstruction).
// Per the goal: PSNR, SSIM, MAE, RMSE, max error, p90/p95/p99 absolute error.
// Operates on 3D u8 cubes of identical dims. Pure C, libc+libm only.
#ifndef V2_METRICS_H
#define V2_METRICS_H
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    double psnr, ssim, mae, rmse, max_err, p90, p95, p99;
    double bpp;        // bits per (logical) voxel — filled by caller
    double ratio;      // logical_bytes / compressed_bytes — filled by caller
} basket_t;

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

// ---- STREAMING metric accumulator (no full rec/mask/errs buffers) ----
// Error percentiles via a 256-bin histogram (u8 abs error is 0..255 — exact, not approximate).
// SSIM accumulated per 8^3 block as blocks are decoded. Holds O(1) memory, not O(volume).
typedef struct {
    long ehist[256];        // abs-error histogram over scored (material) voxels
    double se, sa, mx; long m;   // sum sq err, sum abs err, max err, count
    double ssim_acc; long ssim_nw;  // SSIM block accumulator
} metric_acc;
static void macc_init(metric_acc *a){ memset(a,0,sizeof *a); }
// add one voxel's error (ref,rec) if scored (caller already applied mask)
static inline void macc_add_err(metric_acc *a, int ref, int rec){
    int e = ref>rec ? ref-rec : rec-ref; a->ehist[e]++; a->se += (double)e*e; a->sa += e;
    if(e>a->mx)a->mx=e; a->m++;
}
// add one fully-decoded 8^3 SSIM window (ref/rec are 512-elem block samples, mean/var streamed)
static inline void macc_add_ssim_window(metric_acc *a, const uint8_t *ref, const uint8_t *rec){
    const double C1=(0.01*255)*(0.01*255), C2=(0.03*255)*(0.03*255); const int W=8; const double n=W*W*W;
    double ma=0,mb=0; for(int i=0;i<W*W*W;++i){ma+=ref[i];mb+=rec[i];} ma/=n; mb/=n;
    double va=0,vb=0,cov=0; for(int i=0;i<W*W*W;++i){double da=ref[i]-ma,db=rec[i]-mb; va+=da*da;vb+=db*db;cov+=da*db;}
    va/=n-1;vb/=n-1;cov/=n-1;
    a->ssim_acc += ((2*ma*mb+C1)*(2*cov+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2)); a->ssim_nw++;
}
static basket_t macc_finish(const metric_acc *a){
    basket_t b; memset(&b,0,sizeof b); long m=a->m?a->m:1;
    double mse=a->se/m; b.psnr = mse<=0?99.0:10.0*log10(255.0*255.0/mse);
    b.mae=a->sa/m; b.rmse=sqrt(mse); b.max_err=a->mx;
    long c90=(long)(0.90*(m-1)),c95=(long)(0.95*(m-1)),c99=(long)(0.99*(m-1));
    long cum=0; b.p90=b.p95=b.p99=0;
    for(int e=0;e<256;++e){ long prev=cum; cum+=a->ehist[e];
        if(prev<=c90&&c90<cum)b.p90=e; if(prev<=c95&&c95<cum)b.p95=e; if(prev<=c99&&c99<cum)b.p99=e; }
    b.ssim = a->ssim_nw ? a->ssim_acc/a->ssim_nw : 1.0;
    return b;
}

// 8^3-block mean SSIM over the cube (non-overlapping), global C1/C2.
static double ssim3d(const uint8_t *ref, const uint8_t *rec, int nz, int ny, int nx) {
    const int W = 8;
    const double C1 = (0.01*255)*(0.01*255), C2 = (0.03*255)*(0.03*255);
    double acc = 0; long nw = 0;
    for (int z = 0; z + W <= nz; z += W)
    for (int y = 0; y + W <= ny; y += W)
    for (int x = 0; x + W <= nx; x += W) {
        double ma=0, mb=0;
        for (int dz=0; dz<W; ++dz) for (int dy=0; dy<W; ++dy) for (int dx=0; dx<W; ++dx) {
            size_t i = ((size_t)(z+dz)*ny + (y+dy))*nx + (x+dx);
            ma += ref[i]; mb += rec[i];
        }
        double n = (double)(W*W*W); ma/=n; mb/=n;
        double va=0, vb=0, cov=0;
        for (int dz=0; dz<W; ++dz) for (int dy=0; dy<W; ++dy) for (int dx=0; dx<W; ++dx) {
            size_t i = ((size_t)(z+dz)*ny + (y+dy))*nx + (x+dx);
            double da = ref[i]-ma, db = rec[i]-mb;
            va += da*da; vb += db*db; cov += da*db;
        }
        va/=n-1; vb/=n-1; cov/=n-1;
        double s = ((2*ma*mb+C1)*(2*cov+C2)) / ((ma*ma+mb*mb+C1)*(va+vb+C2));
        acc += s; nw++;
    }
    return nw ? acc/nw : 1.0;
}

// Compute the basket. If mask!=NULL, error stats are restricted to voxels where
// mask[i]!=0 (score only the region we care about — true signal, not air).
static basket_t metrics_eval(const uint8_t *ref, const uint8_t *rec,
                             int nz, int ny, int nx, const uint8_t *mask) {
    basket_t b; memset(&b, 0, sizeof b);
    size_t n = (size_t)nz*ny*nx;
    double *errs = (double *)malloc(n * sizeof(double));
    size_t m = 0; double se = 0, sa = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (mask && !mask[i]) continue;
        double e = fabs((double)ref[i] - (double)rec[i]);
        errs[m++] = e; se += e*e; sa += e; if (e > mx) mx = e;
    }
    if (m == 0) { errs[m++] = 0; }
    double mse = se / m;
    b.psnr = mse <= 0 ? 99.0 : 10.0*log10(255.0*255.0/mse);
    b.mae = sa / m; b.rmse = sqrt(mse); b.max_err = mx;
    qsort(errs, m, sizeof(double), cmp_dbl);
    b.p90 = errs[(size_t)(0.90*(m-1))];
    b.p95 = errs[(size_t)(0.95*(m-1))];
    b.p99 = errs[(size_t)(0.99*(m-1))];
    free(errs);
    b.ssim = ssim3d(ref, rec, nz, ny, nx);
    return b;
}

static void basket_print(const char *name, const basket_t *b) {
    printf("%-18s ratio %6.2fx  bpp %5.3f  PSNR %5.2f  SSIM %.4f  MAE %5.2f  max %3.0f  p90 %4.1f  p95 %4.1f  p99 %4.1f\n",
           name, b->ratio, b->bpp, b->psnr, b->ssim, b->mae, b->max_err, b->p90, b->p95, b->p99);
}
#endif
