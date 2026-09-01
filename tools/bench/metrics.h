/* Quality metrics for u8 volumes. Precise (no fast-math in this TU). */
#ifndef BAKE_METRICS_H
#define BAKE_METRICS_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* Peak signal-to-noise ratio in dB; returns INFINITY for identical input. */
double metric_psnr_u8(const uint8_t *a, const uint8_t *b, size_t n);

/* Sum of squared error (for pooled aggregate PSNR). */
double metric_sse_u8(const uint8_t *a, const uint8_t *b, size_t n);

/* Mean 3D SSIM over a dim^3 volume (gaussian window 11, sigma 1.5, K1=.01 K2=.03). */
double metric_ssim3d_u8(const uint8_t *a, const uint8_t *b, uint32_t dim);

/* Max absolute voxel error. */
uint32_t metric_maxerr_u8(const uint8_t *a, const uint8_t *b, size_t n);

/* |err| histogram (256 bins) accumulated into hist; caller zeroes it. */
void metric_errhist_u8(const uint8_t *a, const uint8_t *b, size_t n, uint64_t hist[256]);
/* From a histogram: mean abs error and the smallest e with cdf(e) >= pct. */
double errhist_mae(const uint64_t hist[256]);
uint32_t errhist_percentile(const uint64_t hist[256], double pct);

/* Transform-boundary artifact metric. For each block face, compare the
 * reconstructed cross-face gradient with the source gradient; mid-block
 * planes provide the like-for-like interior baseline. Calls accumulate. */
typedef struct metric_blocking {
  long double boundary_abs, boundary_sse;
  long double interior_abs, interior_sse;
  uint64_t boundary_n, interior_n;
} metric_blocking;

void metric_blocking_u8(const uint8_t *src, const uint8_t *dec, uint32_t dim, uint32_t block_dim,
                        metric_blocking *out);
double metric_blocking_boundary_rmse(const metric_blocking *m);
double metric_blocking_interior_rmse(const metric_blocking *m);
double metric_blocking_amplification(const metric_blocking *m);

#endif
