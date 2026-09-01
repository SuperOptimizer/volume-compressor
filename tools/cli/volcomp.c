/* volcomp CLI
 *   volcomp encode  in.u8 out.volc --q=Q
 *   volcomp decode  in.volc out.u8
 *   volcomp verify  in.volc ref.u8            (decode + error metrics)
 *   volcomp shard-pack DIR out.shard --q=Q    (DIR/z_y_x.u8, z,y,x in 0..7 -> zarr v3 shard)
 * Input chunks are raw 128^3 u8 files (2097152 bytes), z-major. */
#include "../../volcomp.h"
#include "metrics.h"
#include "shard_pack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_file(const char *path, size_t *n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz < 0) {
    fclose(f);
    return NULL;
  }
  uint8_t *b = malloc((size_t)sz + 1);
  if (b && fread(b, 1, (size_t)sz, f) != (size_t)sz) {
    free(b);
    b = NULL;
  }
  fclose(f);
  *n = (size_t)sz;
  return b;
}
static int write_file(const char *path, const void *buf, size_t n) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  int ok = fwrite(buf, 1, n, f) == n;
  ok &= fclose(f) == 0;
  return ok ? 0 : -1;
}
static float parse_q(int argc, char **argv) {
  for (int i = 1; i < argc; i++)
    if (!strncmp(argv[i], "--q=", 4)) return (float)atof(argv[i] + 4);
  return 0;
}
static int usage(void) {
  fprintf(stderr, "usage:\n  volcomp encode in.u8 out.volc --q=Q\n  volcomp decode in.volc out.u8\n"
                  "  volcomp verify in.volc ref.u8\n  volcomp shard-pack DIR out.shard --q=Q\n");
  return 1;
}

int main(int argc, char **argv) {
  if (argc < 4) return usage();
  const char *cmd = argv[1];
  if (!strcmp(cmd, "encode")) {
    float q = parse_q(argc, argv);
    if (q <= 0) return usage();
    size_t n;
    uint8_t *src = read_file(argv[2], &n);
    if (!src || n != VOLCOMP_CHUNK_VOXELS) {
      fprintf(stderr, "%s: need exactly %u bytes\n", argv[2], VOLCOMP_CHUNK_VOXELS);
      return 2;
    }
    uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
    size_t en;
    volcomp_status st = volcomp_encode(src, q, enc, VOLCOMP_ENCODE_BOUND, &en);
    if (st) {
      fprintf(stderr, "encode: %s\n", volcomp_status_string(st));
      return 3;
    }
    if (write_file(argv[3], enc, en)) return 2;
    printf("%zu bytes (%.2fx)\n", en, (double)n / (double)en);
    return 0;
  }
  if (!strcmp(cmd, "decode") || !strcmp(cmd, "verify")) {
    size_t n;
    uint8_t *enc = read_file(argv[2], &n);
    if (!enc) return 2;
    uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
    volcomp_status st = volcomp_decode(enc, n, dec, VOLCOMP_CHUNK_VOXELS);
    if (st) {
      fprintf(stderr, "decode: %s\n", volcomp_status_string(st));
      return 3;
    }
    if (!strcmp(cmd, "decode")) return write_file(argv[3], dec, VOLCOMP_CHUNK_VOXELS) ? 2 : 0;
    size_t rn;
    uint8_t *ref = read_file(argv[3], &rn);
    if (!ref || rn != VOLCOMP_CHUNK_VOXELS) return 2;
    uint64_t h[256] = {0};
    metric_errhist_u8(ref, dec, rn, h);
    metric_blocking mb = {0};
    metric_blocking_u8(ref, dec, 128, 16, &mb);
    printf("bytes %zu ratio %.2f psnr %.2f ssim %.4f mae %.3f p90 %u p95 %u p99 %u max %u blocking_amp %.3f\n",
           n, (double)rn / (double)n, metric_psnr_u8(ref, dec, rn), metric_ssim3d_u8(ref, dec, 128),
           errhist_mae(h), errhist_percentile(h, 0.90), errhist_percentile(h, 0.95),
           errhist_percentile(h, 0.99), metric_maxerr_u8(ref, dec, rn),
           metric_blocking_amplification(&mb));
    return 0;
  }
  if (!strcmp(cmd, "shard-pack")) {
    float q = parse_q(argc, argv);
    if (q <= 0) return usage();
    unsigned present = 0;
    uint64_t bytes = 0;
    int rc = shard_pack(argv[2], argv[3], q, &present, &bytes);
    if (rc) {
      fprintf(stderr, "shard-pack failed (%d)\n", rc);
      return rc;
    }
    printf("%u chunks present, %llu payload bytes\n", present, (unsigned long long)bytes);
    return 0;
  }
  return usage();
}
