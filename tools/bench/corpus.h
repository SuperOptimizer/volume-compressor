/* Corpus loader: flat raw .u8 bricks + .json sidecars (see tools/fetch_corpus.py). */
#ifndef BAKE_CORPUS_H
#define BAKE_CORPUS_H

#include <stddef.h>
#include <stdint.h>

#define CORPUS_MAX_BRICKS 1024
#define CORPUS_CLASS_LEN 32
#define CORPUS_ID_LEN 128

typedef struct corpus_brick {
  char id[CORPUS_ID_LEN];
  char cls[CORPUS_CLASS_LEN]; /* dense_interior / smooth_papyrus / surface_boundary / air */
  uint32_t dim;               /* cubic edge length, voxels (128 expected) */
  uint8_t *voxels;            /* dim^3 bytes, malloc'd */
} corpus_brick;

typedef struct corpus {
  corpus_brick bricks[CORPUS_MAX_BRICKS];
  size_t count;
} corpus;

/* Loads every "<id>.u8" with sidecar "<id>.json" under dir. Returns 0 on success. */
int corpus_load(const char *dir, corpus *c);
void corpus_free(corpus *c);

#endif
