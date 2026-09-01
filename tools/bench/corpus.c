#include "corpus.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tiny extractor for flat string/number fields in our own controlled sidecars.
 * Not a general JSON parser; sidecars are machine-written by fetch_corpus.py. */
static int json_str_field(const char *json, const char *key, char *out, size_t out_len) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\"", key);
  const char *p = strstr(json, pat);
  if (!p) return -1;
  p = strchr(p + strlen(pat), ':');
  if (!p) return -1;
  p = strchr(p, '"');
  if (!p) return -1;
  p++;
  const char *e = strchr(p, '"');
  if (!e || (size_t)(e - p) >= out_len) return -1;
  memcpy(out, p, (size_t)(e - p));
  out[e - p] = '\0';
  return 0;
}

static uint8_t *read_file(const char *path, size_t *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return nullptr;
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (n <= 0) {
    fclose(f);
    return nullptr;
  }
  uint8_t *buf = malloc((size_t)n + 1);
  if (!buf) {
    fclose(f);
    return nullptr;
  }
  size_t rd = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (rd != (size_t)n) {
    free(buf);
    return nullptr;
  }
  buf[n] = 0;
  *n_out = (size_t)n;
  return buf;
}

static int brick_id_cmp(const void *va, const void *vb) {
  const corpus_brick *a = va, *b = vb;
  return strcmp(a->id, b->id);
}

int corpus_load(const char *dir, corpus *c) {
  memset(c, 0, sizeof *c);
  DIR *d = opendir(dir);
  if (!d) {
    fprintf(stderr, "corpus: cannot open %s\n", dir);
    return -1;
  }
  struct dirent *de;
  while ((de = readdir(d)) != nullptr && c->count < CORPUS_MAX_BRICKS) {
    size_t len = strlen(de->d_name);
    if (len < 4 || strcmp(de->d_name + len - 3, ".u8") != 0) continue;

    corpus_brick *b = &c->bricks[c->count];
    snprintf(b->id, sizeof b->id, "%.*s", (int)(len - 3), de->d_name);

    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
    size_t n = 0;
    b->voxels = read_file(path, &n);
    if (!b->voxels) {
      fprintf(stderr, "corpus: failed to read %s\n", path);
      continue;
    }
    double edge = cbrt((double)n);
    b->dim = (uint32_t)llround(edge);
    if ((size_t)b->dim * b->dim * b->dim != n) {
      fprintf(stderr, "corpus: %s size %zu is not a cube, skipping\n", path, n);
      free(b->voxels);
      b->voxels = nullptr;
      continue;
    }

    snprintf(path, sizeof path, "%s/%s.json", dir, b->id);
    size_t jn = 0;
    uint8_t *json = read_file(path, &jn);
    strcpy(b->cls, "unknown");
    if (json) {
      (void)json_str_field((const char *)json, "class", b->cls, sizeof b->cls);
      free(json);
    }
    c->count++;
  }
  closedir(d);
  if (c->count == 0) {
    fprintf(stderr, "corpus: no bricks found in %s\n", dir);
    return -1;
  }
  qsort(c->bricks, c->count, sizeof c->bricks[0], brick_id_cmp);
  return 0;
}

void corpus_free(corpus *c) {
  for (size_t i = 0; i < c->count; i++) free(c->bricks[i].voxels);
  c->count = 0;
}
