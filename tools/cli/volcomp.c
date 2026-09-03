/* volcomp CLI
 *   volcomp encode  in.u8 out.volc --q=Q
 *   volcomp decode  in.volc out.u8
 *   volcomp verify  in.volc ref.u8            (decode + error metrics)
 *   volcomp shard-pack DIR out.shard --q=Q    (DIR/z_y_x.u8, z,y,x in 0..7 -> zarr v3 shard)
 *   volcomp shard-verify in.shard DIR [--samples=N]
 *       index CRC + every chunk decodes; N (default 8, 0 = all) present chunks are
 *       compared with DIR/z_y_x.u8; a nonzero source chunk missing from the shard fails
 *   volcomp occupancy in.u8|DIR out.bin [--shape=Z,Y,X] [--factor=F] [--dilate=D] [--grid=GZ,GY,GX] [--shards] [--chunks]
 *       occupancy of a raw volume (or, --chunks, a directory of 128^3 chunks cz_cy_cx.u8) on an F^3-cell grid (1 where any voxel within the cell,
 *       dilated by D voxels, is nonzero); --shards emits 64-byte per-shard chunk bitmasks.
 *       Used on a downsampled level to know which chunks of finer levels hold data.
 *   volcomp label-encode DIR out.voll --q=Q   (DIR/<cls>.u8 class probability planes -> label chunk)
 *   volcomp label-decode in.voll DIR          (writes DIR/<cls>.u8 for every stored class)
 *   volcomp label-verify in.voll DIR          (decode + error stats per class)
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
static bool parse_dims(int argc, char **argv, const char *name, unsigned d[3], unsigned def) {
  d[0] = d[1] = d[2] = def;
  for (int i = 4; i < argc; i++)
    if (!strncmp(argv[i], name, strlen(name)))
      return sscanf(argv[i] + strlen(name), "%u,%u,%u", &d[0], &d[1], &d[2]) == 3;
  return true;
}
static bool has_flag(int argc, char **argv, const char *name) {
  for (int i = 4; i < argc; i++)
    if (!strcmp(argv[i], name)) return true;
  return false;
}
typedef struct {
  unsigned g[3], F, D;
  uint8_t *cell;
} occ_grid;
/* mark every cell whose footprint, dilated by D voxels, contains a nonzero voxel of this row */
static void occ_row(occ_grid *o, unsigned z, unsigned y, const uint8_t *row, unsigned x_begin, unsigned len) {
  const unsigned F = o->F, D = o->D;
  unsigned z0 = (z < D ? 0 : z - D) / F, z1 = (z + D) / F, y0 = (y < D ? 0 : y - D) / F, y1 = (y + D) / F;
  if (z0 >= o->g[0]) return;
  if (y0 >= o->g[1]) return;
  if (z1 >= o->g[0]) z1 = o->g[0] - 1;
  if (y1 >= o->g[1]) y1 = o->g[1] - 1;
  for (unsigned i = 0; i < len; i++) {
    if (!row[i]) continue;
    unsigned x = x_begin + i, x0 = (x < D ? 0 : x - D) / F, x1 = (x + D) / F;
    if (x0 >= o->g[2]) return;
    if (x1 >= o->g[2]) x1 = o->g[2] - 1;
    for (unsigned cz = z0; cz <= z1; cz++)
      for (unsigned cy = y0; cy <= y1; cy++)
        for (unsigned cx = x0; cx <= x1; cx++) o->cell[((size_t)cz * o->g[1] + cy) * o->g[2] + cx] = 1;
    unsigned next = (x1 + 1) * F - D;  /* first x whose dilated range reaches a new cell */
    if (next > x + 1) i = next - x_begin - 1;
  }
}
/* occupancy of a Z*Y*X u8 volume on a grid of F^3-voxel cells (cell = one chunk of a finer
 * level): a cell is set when any voxel within its footprint dilated by D voxels is nonzero.
 * Input: a raw file, or with --chunks a directory of 128^3 chunk files cz_cy_cx.u8 (missing =
 * zero). Plain output: gz*gy*gx bytes (grid from --grid or ceil(shape/F)). --shards: one
 * 64-byte bitmask per 8^3 shard of cells, bit (cz*8+cy)*8+cx, shards in (sz,sy,sx) order. */
static int occupancy(int argc, char **argv) {
  const char *in_path = argv[2], *out_path = argv[3];
  long f = parse_opt(argc, argv, "--factor=", 4), d = parse_opt(argc, argv, "--dilate=", 0);
  unsigned sh[3];
  occ_grid o = {.F = (unsigned)f, .D = (unsigned)d};
  if (!parse_dims(argc, argv, "--shape=", sh, 128) || f < 1 || d < 0 || d >= f) {
    fprintf(stderr, "bad --shape / --factor / --dilate (need 0 <= dilate < factor)\n");
    return 1;
  }
  if (!parse_dims(argc, argv, "--grid=", o.g, 0)) return 1;
  for (int i = 0; i < 3; i++)
    if (!o.g[i]) o.g[i] = (unsigned)((sh[i] + f - 1) / f);
  const size_t ncell = (size_t)o.g[0] * o.g[1] * o.g[2];
  o.cell = calloc(1, ncell);
  if (!has_flag(argc, argv, "--chunks")) {
    size_t n;
    uint8_t *v = read_file(in_path, &n);
    if (!v || n != (size_t)sh[0] * sh[1] * sh[2]) {
      fprintf(stderr, "%s: need exactly %zu bytes for shape %u,%u,%u\n", in_path, (size_t)sh[0] * sh[1] * sh[2], sh[0],
              sh[1], sh[2]);
      free(v);
      return 2;
    }
    for (unsigned z = 0; z < sh[0]; z++)
      for (unsigned y = 0; y < sh[1]; y++) occ_row(&o, z, y, v + ((size_t)z * sh[1] + y) * sh[2], 0, sh[2]);
    free(v);
  } else {
    const unsigned C = VOLCOMP_CHUNK_DIM;
    unsigned cg[3] = {(sh[0] + C - 1) / C, (sh[1] + C - 1) / C, (sh[2] + C - 1) / C};
    size_t nread = 0;
    for (unsigned cz = 0; cz < cg[0]; cz++)
      for (unsigned cy = 0; cy < cg[1]; cy++)
        for (unsigned cx = 0; cx < cg[2]; cx++) {
          char path[4096];
          snprintf(path, sizeof path, "%s/%u_%u_%u.u8", in_path, cz, cy, cx);
          size_t n;
          uint8_t *v = read_file(path, &n);
          if (!v) continue;
          if (n != VOLCOMP_CHUNK_VOXELS) {
            fprintf(stderr, "%s: %zu bytes, expected %u\n", path, n, VOLCOMP_CHUNK_VOXELS);
            free(v);
            return 2;
          }
          nread++;
          unsigned nz = sh[0] - cz * C < C ? sh[0] - cz * C : C, ny = sh[1] - cy * C < C ? sh[1] - cy * C : C,
                   nx = sh[2] - cx * C < C ? sh[2] - cx * C : C;
          for (unsigned z = 0; z < nz; z++)
            for (unsigned y = 0; y < ny; y++)
              occ_row(&o, cz * C + z, cy * C + y, v + ((size_t)z * C + y) * C, cx * C, nx);
          free(v);
        }
    fprintf(stderr, "read %zu of %u chunks\n", nread, cg[0] * cg[1] * cg[2]);
  }
  int rc;
  if (!has_flag(argc, argv, "--shards")) {
    rc = write_file(out_path, o.cell, ncell);
  } else {
    size_t s[3] = {(o.g[0] + 7) / 8, (o.g[1] + 7) / 8, (o.g[2] + 7) / 8};
    uint8_t *m = calloc(1, s[0] * s[1] * s[2] * 64);
    size_t set = 0;
    for (size_t cz = 0; cz < o.g[0]; cz++)
      for (size_t cy = 0; cy < o.g[1]; cy++)
        for (size_t cx = 0; cx < o.g[2]; cx++) {
          if (!o.cell[(cz * o.g[1] + cy) * o.g[2] + cx]) continue;
          size_t sidx = ((cz / 8) * s[1] + cy / 8) * s[2] + cx / 8, bit = ((cz % 8) * 8 + cy % 8) * 8 + cx % 8;
          m[sidx * 64 + bit / 8] |= (uint8_t)(1u << (bit % 8));
          set++;
        }
    rc = write_file(out_path, m, s[0] * s[1] * s[2] * 64);
    printf("grid=%u,%u,%u shards=%zu,%zu,%zu occupied=%zu of %zu\n", o.g[0], o.g[1], o.g[2], s[0], s[1], s[2], set, ncell);
    free(m);
  }
  free(o.cell);
  return rc ? 2 : 0;
}
static int usage(void) {
  fprintf(stderr, "usage:\n  volcomp encode in.u8 out.volc --q=Q\n  volcomp decode in.volc out.u8\n"
                  "  volcomp verify in.volc ref.u8\n  volcomp shard-pack DIR out.shard --q=Q\n"
                  "  volcomp shard-verify in.shard DIR [--samples=N]\n"
                  "  volcomp occupancy in.u8|DIR out.bin [--shape=Z,Y,X] [--factor=F] [--dilate=D] [--grid=GZ,GY,GX] [--shards] [--chunks]\n"
                  "  volcomp label-encode DIR out.voll --q=Q\n  volcomp label-decode in.voll DIR\n"
                  "  volcomp label-verify in.voll DIR\n");
  return 1;
}
#include "label_cli.h"

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
  if (!strcmp(cmd, "occupancy")) return occupancy(argc, argv);
  if (!strcmp(cmd, "label-encode")) return label_encode(argc, argv);
  if (!strcmp(cmd, "label-decode")) return label_decode(argc, argv);
  if (!strcmp(cmd, "label-verify")) return label_verify(argc, argv);
  return usage();
}
