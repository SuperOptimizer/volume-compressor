/* volcomp-bench: encode/decode a corpus of 128^3 u8 chunks and print a table.
 *   volcomp-bench --corpus=DIR [--q=Q[,Q...]] [--reps=N] [--no-ssim] [--codec=volcomp|zstd|all]
 * Columns: q ratio psnr ssim mae p90 p95 p99 max enc_MB/s dec_MB/s enc_ms_p50 dec_ms_p50/p99
 * plus block-seam (16-plane) and chunk-seam (128-plane) gradient metrics on any
 * "octet" chunks (a contiguous 2x2x2 group tagged class=octet in the sidecars).
 * Single-threaded; timings are medians of `reps` per chunk. */
#include "../../volcomp.h"
#include "corpus.h"
#include "metrics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef VOLCOMP_HAVE_ZSTD
#include <zstd.h>
#endif

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double pct(double *v, size_t n, double p) {
  qsort(v, n, sizeof *v, cmpd);
  size_t k = (size_t)((double)n * p);
  return v[k >= n ? n - 1 : k];
}

typedef struct agg {
  double raw, comp, se, ssim;
  uint64_t hist[256];
  uint32_t maxerr;
  size_t n;
  double *enc_ms, *dec_ms;
  size_t nt;
  metric_blocking blk;
} agg;

/* seam metric on an assembled 2x2x2 octet: planes at multiples of 128 vs
 * multiples of 16 (excluding 128) vs mid-block; flat-conditioned mean |step| too */
static void seam_metrics(const uint8_t *src, const uint8_t *dec, uint32_t dim, double out[6]) {
  double s128 = 0, s16 = 0, sint = 0, f128 = 0, f16 = 0;
  uint64_t n128 = 0, n16 = 0, nint = 0, m128 = 0, m16 = 0;
  size_t st[3] = {(size_t)dim * dim, dim, 1};
  for (int ax = 0; ax < 3; ax++)
    for (uint32_t p = 8; p < dim; p += 8) { /* planes p-1 | p */
      bool is128 = p % 128 == 0, is16 = p % 16 == 0;
      if (p == 0) continue;
      for (uint32_t a = 0; a < dim; a++)
        for (uint32_t b = 0; b < dim; b++) {
          size_t i;
          if (ax == 0) i = (size_t)p * st[0] + a * st[1] + b;
          else if (ax == 1) i = a * st[0] + (size_t)p * st[1] + b;
          else i = a * st[0] + b * st[1] + p;
          size_t j = i - st[ax];
          double e = ((double)dec[i] - dec[j]) - ((double)src[i] - src[j]);
          bool flat = abs((int)src[i] - (int)src[j]) <= 3;
          double step = fabs((double)dec[i] - dec[j]);
          if (is128) {
            s128 += e * e;
            n128++;
            if (flat) f128 += step, m128++;
          } else if (is16) {
            s16 += e * e;
            n16++;
            if (flat) f16 += step, m16++;
          } else {
            sint += e * e;
            nint++;
          }
        }
    }
  out[0] = sqrt(s128 / (double)n128);
  out[1] = sqrt(s16 / (double)n16);
  out[2] = sqrt(sint / (double)nint);
  out[3] = f128 / (double)(m128 ? m128 : 1);
  out[4] = f16 / (double)(m16 ? m16 : 1);
  out[5] = out[1] / (out[2] > 0 ? out[2] : 1);
}

int main(int argc, char **argv) {
  const char *dir = "corpus/heldout";
  const char *qs = "2,4,8,16,32";
  const char *codec = "volcomp";
  int reps = 3;
  bool do_ssim = true, do_deblock = false;
  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "--corpus=", 9)) dir = argv[i] + 9;
    else if (!strncmp(argv[i], "--q=", 4)) qs = argv[i] + 4;
    else if (!strncmp(argv[i], "--reps=", 7)) reps = atoi(argv[i] + 7);
    else if (!strcmp(argv[i], "--no-ssim")) do_ssim = false;
    else if (!strcmp(argv[i], "--deblock")) do_deblock = true;
    else if (!strncmp(argv[i], "--codec=", 8)) codec = argv[i] + 8;
    else {
      fprintf(stderr, "usage: %s --corpus=DIR [--q=2,4,8] [--reps=N] [--no-ssim] [--codec=volcomp|zstd]\n",
              argv[0]);
      return 1;
    }
  }
  corpus c;
  if (corpus_load(dir, &c) != 0 || c.count == 0) {
    fprintf(stderr, "no chunks in %s\n", dir);
    return 1;
  }
  size_t nvox = VOLCOMP_CHUNK_VOXELS;
  uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND), *dec = malloc(nvox);
  bool zstd = !strcmp(codec, "zstd");
  printf("corpus %s: %zu chunks, %d reps, codec %s\n", dir, (size_t)c.count, reps, codec);
  printf("%6s %8s %7s %7s %6s %4s %4s %4s %4s %8s %8s %8s %8s %8s\n", "q", "ratio", "psnr", "ssim",
         "mae", "p90", "p95", "p99", "max", "enc_MBps", "dec_MBps", "enc_p50", "dec_p50", "dec_p99");
  char qbuf[256];
  strncpy(qbuf, qs, sizeof qbuf - 1);
  qbuf[sizeof qbuf - 1] = 0;
  for (char *tok = strtok(qbuf, ","); tok; tok = strtok(NULL, ",")) {
    float q = (float)atof(tok);
    agg A = {0};
    A.enc_ms = malloc(sizeof(double) * c.count * (size_t)reps);
    A.dec_ms = malloc(sizeof(double) * c.count * (size_t)reps);
    for (uint32_t i = 0; i < c.count; i++) {
      const uint8_t *src = c.bricks[i].voxels;
      if (c.bricks[i].dim != 128) continue;
      size_t n = 0;
      for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        if (zstd) {
#ifdef VOLCOMP_HAVE_ZSTD
          n = ZSTD_compress(enc, VOLCOMP_ENCODE_BOUND, src, nvox, 3);
#endif
        } else if (volcomp_encode(src, q, enc, VOLCOMP_ENCODE_BOUND, &n) != VOLCOMP_OK) {
          fprintf(stderr, "encode failed on %s\n", c.bricks[i].id);
          return 2;
        }
        A.enc_ms[A.nt] = now_ms() - t0;
        t0 = now_ms();
        if (zstd) {
#ifdef VOLCOMP_HAVE_ZSTD
          ZSTD_decompress(dec, nvox, enc, n);
#endif
        } else if (volcomp_decode(enc, n, dec, nvox) != VOLCOMP_OK) {
          fprintf(stderr, "decode failed on %s\n", c.bricks[i].id);
          return 2;
        }
        A.dec_ms[A.nt++] = now_ms() - t0;
      }
      if (do_deblock) volcomp_deblock(dec, 128, 128, 128, q);
      A.raw += (double)nvox;
      A.comp += (double)n;
      A.se += metric_sse_u8(src, dec, nvox);
      if (do_ssim) A.ssim += metric_ssim3d_u8(src, dec, 128);
      metric_errhist_u8(src, dec, nvox, A.hist);
      uint32_t mx = metric_maxerr_u8(src, dec, nvox);
      if (mx > A.maxerr) A.maxerr = mx;
      metric_blocking_u8(src, dec, 128, 16, &A.blk);
      A.n++;
    }
    double mse = A.se / A.raw;
    double psnr = mse > 0 ? 10 * log10(255.0 * 255.0 / mse) : 99.0;
    double enc_med = pct(A.enc_ms, A.nt, 0.5), dec_med = pct(A.dec_ms, A.nt, 0.5);
    double dec_p99 = pct(A.dec_ms, A.nt, 0.99);
    double enc_tot = 0, dec_tot = 0;
    for (size_t k = 0; k < A.nt; k++) enc_tot += A.enc_ms[k], dec_tot += A.dec_ms[k];
    printf("%6.2f %8.2f %7.2f %7.4f %6.3f %4u %4u %4u %4u %8.0f %8.0f %8.2f %8.2f %8.2f\n", (double)q,
           A.raw / A.comp, psnr, A.ssim / (double)A.n, errhist_mae(A.hist),
           errhist_percentile(A.hist, 0.90), errhist_percentile(A.hist, 0.95),
           errhist_percentile(A.hist, 0.99), A.maxerr, (double)nvox * (double)A.nt / (enc_tot * 1e3),
           (double)nvox * (double)A.nt / (dec_tot * 1e3), enc_med, dec_med, dec_p99);
    printf("       blocking: boundary_rmse %.3f interior_rmse %.3f amplification %.3f\n",
           metric_blocking_boundary_rmse(&A.blk), metric_blocking_interior_rmse(&A.blk),
           metric_blocking_amplification(&A.blk));
    /* octet seam evaluation */
    {
      const corpus_brick *oct[8] = {0};
      int no = 0;
      for (uint32_t i = 0; i < c.count && no < 8; i++)
        if (!strcmp(c.bricks[i].cls, "octet")) oct[no++] = &c.bricks[i];
      if (no == 8 && !zstd) {
        /* ids sorted by z,y,x already (corpus is sorted by id) -> order z0y0x0.. */
        uint8_t *S = malloc(256u * 256u * 256u), *D = malloc(256u * 256u * 256u);
        for (int k = 0; k < 8; k++) {
          uint32_t oz = (uint32_t)(k >> 2) & 1, oy = (uint32_t)(k >> 1) & 1, ox = (uint32_t)k & 1;
          size_t n = 0;
          volcomp_encode(oct[k]->voxels, q, enc, VOLCOMP_ENCODE_BOUND, &n);
          volcomp_decode(enc, n, dec, nvox);
          if (do_deblock) volcomp_deblock(dec, 128, 128, 128, q); /* per chunk: chunk faces stay unfiltered, as a naive caller would see them */
          for (uint32_t z = 0; z < 128; z++)
            for (uint32_t y = 0; y < 128; y++) {
              size_t o = ((size_t)(oz * 128 + z) * 256 + oy * 128 + y) * 256 + ox * 128;
              memcpy(S + o, oct[k]->voxels + ((size_t)z * 128 + y) * 128, 128);
              memcpy(D + o, dec + ((size_t)z * 128 + y) * 128, 128);
            }
        }
        double m[6];
        seam_metrics(S, D, 256, m);
        printf("       seam: rmse128 %.3f rmse16 %.3f interior %.3f | flatstep128 %.3f flatstep16 %.3f"
               " | ratio128/16 %.3f amp16 %.3f\n",
               m[0], m[1], m[2], m[3], m[4], m[0] / (m[1] > 0 ? m[1] : 1), m[5]);
        free(S);
        free(D);
      }
    }
    free(A.enc_ms);
    free(A.dec_ms);
  }
  free(enc);
  free(dec);
  corpus_free(&c);
  return 0;
}
