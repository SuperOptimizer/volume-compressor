#include "metrics.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

double metric_sse_u8(const uint8_t *a, const uint8_t *b, size_t n) {
  double se = 0.0;
  for (size_t i = 0; i < n; i++) {
    double d = (double)a[i] - (double)b[i];
    se += d * d;
  }
  return se;
}

double metric_psnr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
  double se = 0.0;
  for (size_t i = 0; i < n; i++) {
    double d = (double)a[i] - (double)b[i];
    se += d * d;
  }
  if (se == 0.0) return HUGE_VAL;
  double mse = se / (double)n;
  return 10.0 * log10(255.0 * 255.0 / mse);
}

uint32_t metric_maxerr_u8(const uint8_t *a, const uint8_t *b, size_t n) {
  uint32_t m = 0;
  for (size_t i = 0; i < n; i++) {
    uint32_t d = (uint32_t)abs((int)a[i] - (int)b[i]);
    if (d > m) m = d;
  }
  return m;
}

/* ---- 3D SSIM ---- */

enum { SSIM_RADIUS = 5 }; /* 11-tap gaussian, sigma 1.5 */

static void make_kernel(double *k) {
  const double sigma = 1.5;
  double sum = 0.0;
  for (int i = -SSIM_RADIUS; i <= SSIM_RADIUS; i++) {
    double v = exp(-(double)(i * i) / (2.0 * sigma * sigma));
    k[i + SSIM_RADIUS] = v;
    sum += v;
  }
  for (int i = 0; i < 2 * SSIM_RADIUS + 1; i++) k[i] /= sum;
}

/* Separable 1D gaussian along `axis` (0=x contiguous, 1=y, 2=z), clamped edges. */
static void blur_axis(const float *src, float *dst, uint32_t dim, int axis, const double *k) {
  const int n = (int)dim;
  const size_t sx = 1, sy = dim, sz = (size_t)dim * dim;
  size_t stride = axis == 0 ? sx : (axis == 1 ? sy : sz);
  /* iterate over all lines along axis */
  for (int c = 0; c < n; c++) {
    for (int r = 0; r < n; r++) {
      size_t base;
      if (axis == 0) base = (size_t)c * sz + (size_t)r * sy;
      else if (axis == 1) base = (size_t)c * sz + (size_t)r * sx;
      else base = (size_t)c * sy + (size_t)r * sx;
      for (int i = 0; i < n; i++) {
        double acc = 0.0;
        for (int t = -SSIM_RADIUS; t <= SSIM_RADIUS; t++) {
          int j = i + t;
          if (j < 0) j = 0;
          if (j >= n) j = n - 1;
          acc += k[t + SSIM_RADIUS] * (double)src[base + (size_t)j * stride];
        }
        dst[base + (size_t)i * stride] = (float)acc;
      }
    }
  }
}

static void blur3(const float *src, float *dst, float *tmp, uint32_t dim, const double *k) {
  blur_axis(src, dst, dim, 0, k);
  blur_axis(dst, tmp, dim, 1, k);
  blur_axis(tmp, dst, dim, 2, k);
}

static double ssim3d_box(const uint8_t *a, const uint8_t *b, uint32_t dim, const uint32_t lo[3],
                            const uint32_t hi[3]) {
  const size_t n = (size_t)dim * dim * dim;
  const double c1 = (0.01 * 255.0) * (0.01 * 255.0);
  const double c2 = (0.03 * 255.0) * (0.03 * 255.0);
  double kern[2 * SSIM_RADIUS + 1];
  make_kernel(kern);

  /* fields: x, y, x^2, y^2, xy -> blurred mu_x, mu_y, m_xx, m_yy, m_xy */
  float *bufs = malloc(sizeof(float) * n * 7);
  if (!bufs) return -1.0;
  float *fx = bufs, *fy = bufs + n, *work = bufs + 2 * n, *tmp = bufs + 3 * n;
  float *mu_x = bufs + 4 * n, *mu_y = bufs + 5 * n, *blur = bufs + 6 * n;

  for (size_t i = 0; i < n; i++) fx[i] = (float)a[i];
  for (size_t i = 0; i < n; i++) fy[i] = (float)b[i];
  blur3(fx, mu_x, tmp, dim, kern);
  blur3(fy, mu_y, tmp, dim, kern);

  double ssim_sum = 0.0;

  /* sigma_x^2 pass reuses work/blur; accumulate per-voxel ssim in three sweeps:
     we need m_xx, m_yy, m_xy blurred; do them one at a time to cap memory. */
  float *m_xx = malloc(sizeof(float) * n);
  float *m_yy = malloc(sizeof(float) * n);
  float *m_xy = malloc(sizeof(float) * n);
  if (!m_xx || !m_yy || !m_xy) {
    free(bufs);
    free(m_xx);
    free(m_yy);
    free(m_xy);
    return -1.0;
  }
  for (size_t i = 0; i < n; i++) work[i] = fx[i] * fx[i];
  blur3(work, blur, tmp, dim, kern);
  memcpy(m_xx, blur, sizeof(float) * n);
  for (size_t i = 0; i < n; i++) work[i] = fy[i] * fy[i];
  blur3(work, blur, tmp, dim, kern);
  memcpy(m_yy, blur, sizeof(float) * n);
  for (size_t i = 0; i < n; i++) work[i] = fx[i] * fy[i];
  blur3(work, blur, tmp, dim, kern);
  memcpy(m_xy, blur, sizeof(float) * n);

  size_t count = 0;
  for (uint32_t z = lo[2]; z < hi[2]; z++)
    for (uint32_t y = lo[1]; y < hi[1]; y++)
      for (uint32_t x = lo[0]; x < hi[0]; x++) {
        size_t i = ((size_t)z * dim + y) * dim + x;
        double mx = (double)mu_x[i], my = (double)mu_y[i];
        double vx = (double)m_xx[i] - mx * mx;
        double vy = (double)m_yy[i] - my * my;
        double cov = (double)m_xy[i] - mx * my;
        double num = (2.0 * mx * my + c1) * (2.0 * cov + c2);
        double den = (mx * mx + my * my + c1) * (vx + vy + c2);
        ssim_sum += num / den;
        count++;
      }
  free(m_xx);
  free(m_yy);
  free(m_xy);
  free(bufs);
  return count ? ssim_sum / (double)count : -1.0;
}

double metric_ssim3d_u8(const uint8_t *a, const uint8_t *b, uint32_t dim) {
  const uint32_t lo[3] = {0, 0, 0}, hi[3] = {dim, dim, dim};
  return ssim3d_box(a, b, dim, lo, hi);
}

void metric_errhist_u8(const uint8_t *a, const uint8_t *b, size_t n, uint64_t hist[256]) {
  for (size_t i = 0; i < n; i++) hist[abs((int)a[i] - (int)b[i])]++;
}

double errhist_mae(const uint64_t hist[256]) {
  uint64_t total = 0, sum = 0;
  for (int e = 0; e < 256; e++) {
    total += hist[e];
    sum += hist[e] * (uint64_t)e;
  }
  return total ? (double)sum / (double)total : 0.0;
}

uint32_t errhist_percentile(const uint64_t hist[256], double pct) {
  uint64_t total = 0;
  for (int e = 0; e < 256; e++) total += hist[e];
  if (total == 0) return 0;
  uint64_t need = pct <= 0.0 ? 1 : (pct >= 1.0 ? total : (uint64_t)ceil((double)total * pct));
  uint64_t acc = 0;
  for (int e = 0; e < 256; e++) {
    acc += hist[e];
    if (acc >= need) return (uint32_t)e;
  }
  return 255;
}

static void blocking_plane(const uint8_t *src, const uint8_t *dec, uint32_t dim, int axis,
                           uint32_t plane, bool boundary, metric_blocking *out) {
  for (uint32_t a = 0; a < dim; a++)
    for (uint32_t b = 0; b < dim; b++) {
      uint32_t x0, y0, z0, x1, y1, z1;
      if (axis == 0) {
        x0 = plane - 1;
        x1 = plane;
        y0 = y1 = b;
        z0 = z1 = a;
      } else if (axis == 1) {
        y0 = plane - 1;
        y1 = plane;
        x0 = x1 = b;
        z0 = z1 = a;
      } else {
        z0 = plane - 1;
        z1 = plane;
        x0 = x1 = b;
        y0 = y1 = a;
      }
      size_t p = ((size_t)z0 * dim + y0) * dim + x0;
      size_t q = ((size_t)z1 * dim + y1) * dim + x1;
      int e = ((int)dec[q] - (int)dec[p]) - ((int)src[q] - (int)src[p]);
      long double v = (long double)e;
      if (boundary) {
        out->boundary_abs += e < 0 ? -v : v;
        out->boundary_sse += v * v;
        out->boundary_n++;
      } else {
        out->interior_abs += e < 0 ? -v : v;
        out->interior_sse += v * v;
        out->interior_n++;
      }
    }
}

void metric_blocking_u8(const uint8_t *src, const uint8_t *dec, uint32_t dim, uint32_t block_dim,
                        metric_blocking *out) {
  if (!src || !dec || !out || block_dim < 2 || dim <= block_dim) return;
  for (int axis = 0; axis < 3; axis++) {
    for (uint32_t p = block_dim; p < dim; p += block_dim)
      blocking_plane(src, dec, dim, axis, p, true, out);
    for (uint32_t p = block_dim / 2; p < dim; p += block_dim)
      blocking_plane(src, dec, dim, axis, p, false, out);
  }
}

double metric_blocking_boundary_rmse(const metric_blocking *m) {
  return m->boundary_n ? sqrt((double)(m->boundary_sse / m->boundary_n)) : 0.0;
}

double metric_blocking_interior_rmse(const metric_blocking *m) {
  return m->interior_n ? sqrt((double)(m->interior_sse / m->interior_n)) : 0.0;
}

double metric_blocking_amplification(const metric_blocking *m) {
  double boundary = metric_blocking_boundary_rmse(m);
  double interior = metric_blocking_interior_rmse(m);
  if (interior == 0.0) return boundary == 0.0 ? 0.0 : HUGE_VAL;
  return boundary / interior;
}
