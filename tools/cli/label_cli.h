/* label-encode / label-decode / label-verify subcommands (included by volcomp.c).
 * A label chunk on disk is a directory of class planes named <cls>.u8 (cls in
 * 0..255, 2097152 bytes each); absent files are absent classes. */
#ifndef VOLCOMP_LABEL_CLI_H
#define VOLCOMP_LABEL_CLI_H

#include "../../volcomp_label.h"
#include "metrics.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct label_dir {
  uint8_t *plane[256]; /* NULL = absent */
} label_dir;

/* parse "<n>.u8" with n in 0..255 */
static int label_cls_of(const char *name) {
  size_t l = strlen(name);
  if (l < 4 || l > 6 || strcmp(name + l - 3, ".u8")) return -1;
  int v = 0;
  for (size_t i = 0; i + 3 < l; i++) {
    if (name[i] < '0' || name[i] > '9') return -1;
    v = v * 10 + (name[i] - '0');
  }
  return v > 255 ? -1 : v;
}

static int label_dir_read(const char *dir, label_dir *d) {
  memset(d, 0, sizeof *d);
  DIR *dp = opendir(dir);
  if (!dp) {
    fprintf(stderr, "%s: cannot open directory\n", dir);
    return -1;
  }
  struct dirent *e;
  int n = 0;
  while ((e = readdir(dp))) {
    int c = label_cls_of(e->d_name);
    if (c < 0) continue;
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
    size_t sz;
    uint8_t *b = read_file(path, &sz);
    if (!b || sz != VOLCOMP_CHUNK_VOXELS) {
      fprintf(stderr, "%s: need exactly %u bytes\n", path, VOLCOMP_CHUNK_VOXELS);
      free(b);
      closedir(dp);
      return -1;
    }
    d->plane[c] = b;
    n++;
  }
  closedir(dp);
  return n;
}
static void label_dir_free(label_dir *d) {
  for (int c = 0; c < 256; c++) free(d->plane[c]);
}

static int label_encode(int argc, char **argv) {
  long t = parse_opt(argc, argv, "--t=", 0);
  float q = parse_q(argc, argv);
  if (q <= 0) q = 4.0f;
  if (t < 0 || t > (long)VOLCOMP_LABEL_MAX_TOLERANCE) return usage();
  label_dir d;
  if (label_dir_read(argv[2], &d) < 0) return 2;
  volcomp_label_plane pl[256];
  uint32_t np = 0;
  for (int c = 0; c < 256; c++)
    if (d.plane[c]) pl[np++] = (volcomp_label_plane){(uint8_t)c, d.plane[c]};
  size_t cap = VOLCOMP_LABEL_ENCODE_BOUND(np), n;
  uint8_t *enc = malloc(cap);
  volcomp_status st = volcomp_label_encode(pl, np, (uint32_t)t, q, enc, cap, &n);
  if (st) {
    fprintf(stderr, "label-encode: %s\n", volcomp_status_string(st));
    return 3;
  }
  if (write_file(argv[3], enc, n)) return 2;
  uint8_t cls[255];
  uint32_t nc;
  volcomp_label_classes(enc, n, cls, &nc);
  printf("%zu bytes, %u of %u planes stored (%.1fx vs stored planes raw)\n", n, nc, np,
         (double)nc * VOLCOMP_CHUNK_VOXELS / (double)(n ? n : 1));
  const uint8_t *dir = enc + VOLCOMP_LABEL_HDR_BYTES;
  static const char *modes[] = {"const", "palette", "image", "raw"};
  for (uint32_t i = 0; i < nc; i++) {
    const uint8_t *e = dir + i * VOLCOMP_LABEL_DIR_ENTRY;
    printf("  class %3u: %-7s %u bytes\n", e[0], modes[e[1] & 3], vf_rd_u16(e + 2) + vf_rd_u32(e + 4));
  }
  free(enc);
  label_dir_free(&d);
  return 0;
}

static int label_decode(int argc, char **argv) {
  (void)argc;
  size_t n;
  uint8_t *enc = read_file(argv[2], &n);
  if (!enc) return 2;
  uint8_t cls[255];
  uint32_t nc;
  volcomp_status st = volcomp_label_classes(enc, n, cls, &nc);
  if (st) {
    fprintf(stderr, "label-decode: %s\n", volcomp_status_string(st));
    return 3;
  }
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  for (uint32_t i = 0; i < nc; i++) {
    st = volcomp_label_decode(enc, n, cls[i], dec, VOLCOMP_CHUNK_VOXELS);
    if (st) {
      fprintf(stderr, "label-decode class %u: %s\n", cls[i], volcomp_status_string(st));
      return 3;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/%u.u8", argv[3], cls[i]);
    if (write_file(path, dec, VOLCOMP_CHUNK_VOXELS)) {
      fprintf(stderr, "%s: write failed\n", path);
      return 2;
    }
  }
  printf("%u planes written\n", nc);
  free(dec);
  free(enc);
  return 0;
}

/* Verification of the tolerance contract: every changed voxel must be within t
 * (Chebyshev) of a source voxel holding its new value. */
static bool label_within_tolerance(const uint8_t *src, const uint8_t *dec, uint32_t t, uint32_t *changed,
                                   uint32_t *violations) {
  *changed = 0;
  *violations = 0;
  for (uint32_t z = 0; z < 128; z++)
    for (uint32_t y = 0; y < 128; y++)
      for (uint32_t x = 0; x < 128; x++) {
        size_t i = ((size_t)z * 128 + y) * 128 + x;
        if (src[i] == dec[i]) continue;
        (*changed)++;
        bool ok = false;
        for (int dz = -(int)t; dz <= (int)t && !ok; dz++)
          for (int dy = -(int)t; dy <= (int)t && !ok; dy++)
            for (int dx = -(int)t; dx <= (int)t && !ok; dx++) {
              int zz = (int)z + dz, yy = (int)y + dy, xx = (int)x + dx;
              if (zz < 0 || yy < 0 || xx < 0 || zz > 127 || yy > 127 || xx > 127) continue;
              ok = src[((size_t)zz * 128 + (size_t)yy) * 128 + (size_t)xx] == dec[i];
            }
        if (!ok) (*violations)++;
      }
  return *violations == 0;
}

static int label_verify(int argc, char **argv) {
  (void)argc;
  size_t n;
  uint8_t *enc = read_file(argv[2], &n);
  if (!enc) return 2;
  uint8_t cls[255];
  uint32_t nc, tol;
  float q;
  volcomp_status st = volcomp_label_classes(enc, n, cls, &nc);
  if (st == VOLCOMP_OK) st = volcomp_label_params(enc, n, &tol, &q);
  if (st) {
    fprintf(stderr, "label-verify: %s\n", volcomp_status_string(st));
    return 3;
  }
  label_dir d;
  if (label_dir_read(argv[3], &d) < 0) return 2;
  uint8_t *dec = malloc(VOLCOMP_CHUNK_VOXELS);
  int rc = 0;
  bool stored[256] = {0};
  const uint8_t *dir = enc + VOLCOMP_LABEL_HDR_BYTES;
  static const char *modes[] = {"const", "palette", "image", "raw"};
  printf("bytes %zu tolerance %u q %.2f planes %u\n", n, tol, (double)q, nc);
  for (uint32_t i = 0; i < nc; i++) {
    uint32_t c = cls[i];
    stored[c] = true;
    const uint8_t *e = dir + i * VOLCOMP_LABEL_DIR_ENTRY;
    uint32_t mode = e[1], bytes = vf_rd_u16(e + 2) + vf_rd_u32(e + 4);
    st = volcomp_label_decode(enc, n, (uint8_t)c, dec, VOLCOMP_CHUNK_VOXELS);
    if (st) {
      printf("class %3u: decode failed: %s\n", c, volcomp_status_string(st));
      rc = 1;
      continue;
    }
    if (!d.plane[c]) {
      printf("class %3u: stored but no source plane\n", c);
      rc = 1;
      continue;
    }
    const uint8_t *src = d.plane[c];
    if (mode == VL_MODE_IMAGE) {
      uint64_t h[256] = {0};
      metric_errhist_u8(src, dec, VOLCOMP_CHUNK_VOXELS, h);
      printf("class %3u: %-7s %8u bytes psnr %.2f mae %.3f p99 %u max %u\n", c, modes[mode & 3], bytes,
             metric_psnr_u8(src, dec, VOLCOMP_CHUNK_VOXELS), errhist_mae(h), errhist_percentile(h, 0.99),
             metric_maxerr_u8(src, dec, VOLCOMP_CHUNK_VOXELS));
      continue;
    }
    uint32_t changed, viol;
    bool ok = label_within_tolerance(src, dec, tol, &changed, &viol);
    printf("class %3u: %-7s %8u bytes changed %u (%.4f%%) tolerance %s\n", c, modes[mode & 3], bytes, changed,
           100.0 * changed / VOLCOMP_CHUNK_VOXELS, ok ? "ok" : "VIOLATED");
    if (!ok) rc = 1;
  }
  for (int c = 0; c < 256; c++) {
    if (!d.plane[c] || stored[c]) continue;
    bool nz = false;
    for (size_t i = 0; i < VOLCOMP_CHUNK_VOXELS && !nz; i++) nz = d.plane[c][i] != 0;
    if (nz) {
      printf("class %3d: nonzero source plane missing from stream\n", c);
      rc = 1;
    }
  }
  printf(rc ? "FAIL\n" : "ok\n");
  free(dec);
  free(enc);
  label_dir_free(&d);
  return rc;
}

#endif
