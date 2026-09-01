/* volcomp CLI
 *   volcomp encode  in.u8 out.volc --q=Q
 *   volcomp decode  in.volc out.u8
 *   volcomp verify  in.volc ref.u8            (decode + error metrics)
 *   volcomp shard-pack DIR out.shard --q=Q    (DIR/z_y_x.u8, z,y,x in 0..7 -> zarr v3 shard)
 *   volcomp shard-verify in.shard DIR [--samples=N]
 *       index CRC + every chunk decodes; N (default 8, 0 = all) present chunks are
 *       compared with DIR/z_y_x.u8; a nonzero source chunk missing from the shard fails
 * Input chunks are raw 128^3 u8 files (2097152 bytes), z-major. */
#include "../../volcomp.h"
#include "metrics.h"
#include "shard_pack.h"

#include <math.h>
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
static long parse_opt(int argc, char **argv, const char *name, long dflt) {
  size_t ln = strlen(name);
  for (int i = 1; i < argc; i++)
    if (!strncmp(argv[i], name, ln)) return atol(argv[i] + ln);
  return dflt;
}
/* PSNR of dec vs ref over one chunk (999 when identical) */
static double chunk_psnr(const uint8_t *ref, const uint8_t *dec, unsigned *max_err) {
  double se = 0;
  unsigned mx = 0;
  for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS; i++) {
    int d = (int)ref[i] - (int)dec[i];
    unsigned a = (unsigned)(d < 0 ? -d : d);
    se += (double)d * d;
    if (a > mx) mx = a;
  }
  *max_err = mx;
  if (se == 0) return 999.0;
  return 10.0 * log10(65025.0 * (double)VOLCOMP_CHUNK_VOXELS / se);
}
static int shard_verify(const char *shard_path, const char *dir, long samples) {
  size_t sn;
  uint8_t *shard = read_file(shard_path, &sn);
  if (!shard) {
    fprintf(stderr, "cannot read %s\n", shard_path);
    return 2;
  }
  uint64_t off[512], nb[512];
  if (!shard_index_parse(shard, sn, off, nb)) {
    fprintf(stderr, "bad shard index (size or crc32c)\n");
    return 3;
  }
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  unsigned present = 0, compared = 0, max_err = 0;
  double psnr_min = 999.0, psnr_sum = 0;
  uint64_t payload = sn - SHARD_INDEX_BYTES;
  for (uint32_t i = 0; i < SHARD_CHUNKS; i++)
    if (off[i] != ~0ull) present++;
  long stride = (samples <= 0 || (long)present <= samples) ? 1 : (long)present / samples;
  long seen = 0;
  for (uint32_t i = 0; i < SHARD_CHUNKS; i++) {
    uint32_t z = i >> 6, y = (i >> 3) & 7, x = i & 7;
    char path[4096];
    snprintf(path, sizeof path, "%s/%u_%u_%u.u8", dir, z, y, x);
    if (off[i] == ~0ull) { /* missing: the source must be absent or all zero */
      size_t rn;
      uint8_t *ref = read_file(path, &rn);
      if (ref) {
        bool zero = rn == VOLCOMP_CHUNK_VOXELS;
        for (size_t k = 0; k < rn && zero; k++) zero = ref[k] == 0;
        free(ref);
        if (!zero) {
          fprintf(stderr, "chunk %u_%u_%u is nonzero in the source but missing from the shard\n", z, y, x);
          return 4;
        }
      }
      continue;
    }
    if (off[i] + nb[i] > payload) {
      fprintf(stderr, "chunk %u index entry out of range\n", i);
      return 3;
    }
    volcomp_status st = volcomp_decode(shard + off[i], (size_t)nb[i], dec, VOLCOMP_CHUNK_VOXELS);
    if (st) {
      fprintf(stderr, "chunk %u_%u_%u: %s\n", z, y, x, volcomp_status_string(st));
      return 5;
    }
    if (seen++ % stride) continue;
    size_t rn;
    uint8_t *ref = read_file(path, &rn);
    if (!ref || rn != VOLCOMP_CHUNK_VOXELS) {
      fprintf(stderr, "chunk %u_%u_%u present in the shard but source %s unreadable\n", z, y, x, path);
      free(ref);
      return 4;
    }
    unsigned me;
    double p = chunk_psnr(ref, dec, &me);
    free(ref);
    if (me > max_err) max_err = me;
    if (p < psnr_min) psnr_min = p;
    psnr_sum += p;
    compared++;
  }
  printf("ok present=%u payload=%llu compared=%u psnr_min=%.2f psnr_mean=%.2f max_err=%u\n", present,
         (unsigned long long)payload, compared, psnr_min, compared ? psnr_sum / compared : 0.0, max_err);
  free(dec);
  free(shard);
  return 0;
}
static int usage(void) {
  fprintf(stderr, "usage:\n  volcomp encode in.u8 out.volc --q=Q\n  volcomp decode in.volc out.u8\n"
                  "  volcomp verify in.volc ref.u8\n  volcomp shard-pack DIR out.shard --q=Q\n"
                  "  volcomp shard-verify in.shard DIR [--samples=N]\n");
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
    printf("ok present=%u payload=%llu\n", present, (unsigned long long)bytes);
    return 0;
  }
  if (!strcmp(cmd, "shard-verify")) return shard_verify(argv[2], argv[3], parse_opt(argc, argv, "--samples=", 8));
  return usage();
}
