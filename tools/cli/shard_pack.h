/* zarr v3 sharding_indexed writer for volcomp chunks (spec/format.md §7).
 * Shared by the CLI and its test. Input: DIR/z_y_x.u8 (z,y,x in 0..7), each a
 * raw 128^3 chunk; missing or all-zero chunks become "missing" index entries. */
#ifndef VOLCOMP_SHARD_PACK_H
#define VOLCOMP_SHARD_PACK_H

#include "../../volcomp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHARD_CHUNKS 512u
#define SHARD_INDEX_BYTES (SHARD_CHUNKS * 16u + 4u)

static inline void shard_wr_u64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static inline uint64_t shard_rd_u64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) v = v << 8 | p[i];
  return v;
}
/* CRC-32C (Castagnoli), table-driven */
static inline uint32_t shard_crc32c(const uint8_t *p, size_t n) {
  static uint32_t tab[256];
  if (!tab[1])
    for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int k = 0; k < 8; k++) c = (c & 1) ? 0x82F63B78u ^ (c >> 1) : c >> 1;
      tab[i] = c;
    }
  uint32_t c = ~0u;
  for (size_t i = 0; i < n; i++) c = tab[(c ^ p[i]) & 0xff] ^ (c >> 8);
  return ~c;
}
static inline uint8_t *shard_read_file(const char *path, size_t *n) {
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

/* Returns 0 on success; 2 = I/O error, 3 = encode error. */
static int shard_pack(const char *dir, const char *out_path, float q, unsigned *present,
                      uint64_t *payload_bytes) {
  FILE *out = fopen(out_path, "wb");
  if (!out) return 2;
  uint8_t *idx = calloc(1, SHARD_INDEX_BYTES);
  uint8_t *enc = malloc(VOLCOMP_ENCODE_BOUND);
  uint64_t off = 0;
  unsigned np = 0;
  int rc = 0;
  for (uint32_t i = 0; i < SHARD_CHUNKS && rc == 0; i++) {
    uint32_t z = i >> 6, y = (i >> 3) & 7, x = i & 7;
    char path[4096];
    snprintf(path, sizeof path, "%s/%u_%u_%u.u8", dir, z, y, x);
    size_t n = 0;
    uint8_t *src = shard_read_file(path, &n);
    bool zero = true;
    if (src && n == VOLCOMP_CHUNK_VOXELS)
      for (size_t k = 0; k < n && zero; k++) zero = src[k] == 0;
    if (!src || n != VOLCOMP_CHUNK_VOXELS || zero) {
      shard_wr_u64(idx + i * 16, ~0ull);
      shard_wr_u64(idx + i * 16 + 8, ~0ull);
      free(src);
      continue;
    }
    size_t en;
    volcomp_status st = volcomp_encode(src, q, enc, VOLCOMP_ENCODE_BOUND, &en);
    free(src);
    if (st) {
      rc = 3;
      break;
    }
    if (fwrite(enc, 1, en, out) != en) {
      rc = 2;
      break;
    }
    shard_wr_u64(idx + i * 16, off);
    shard_wr_u64(idx + i * 16 + 8, en);
    off += en;
    np++;
  }
  if (rc == 0) {
    uint32_t crc = shard_crc32c(idx, SHARD_CHUNKS * 16u);
    for (int i = 0; i < 4; i++) idx[SHARD_CHUNKS * 16u + (uint32_t)i] = (uint8_t)(crc >> (8 * i));
    if (fwrite(idx, 1, SHARD_INDEX_BYTES, out) != SHARD_INDEX_BYTES) rc = 2;
  }
  if (fclose(out) && rc == 0) rc = 2;
  free(idx);
  free(enc);
  if (present) *present = np;
  if (payload_bytes) *payload_bytes = off;
  return rc;
}

/* Parse the trailing index of a shard image; returns false if the CRC does not
 * match. offsets/nbytes are 0xFF..FF for missing chunks. */
static inline bool shard_index_parse(const uint8_t *shard, size_t n, uint64_t offsets[512],
                                     uint64_t nbytes[512]) {
  if (n < SHARD_INDEX_BYTES) return false;
  const uint8_t *idx = shard + n - SHARD_INDEX_BYTES;
  uint32_t crc = shard_crc32c(idx, SHARD_CHUNKS * 16u), stored = 0;
  for (int i = 3; i >= 0; i--) stored = stored << 8 | idx[SHARD_CHUNKS * 16u + (uint32_t)i];
  if (crc != stored) return false;
  for (uint32_t i = 0; i < SHARD_CHUNKS; i++) {
    offsets[i] = shard_rd_u64(idx + i * 16);
    nbytes[i] = shard_rd_u64(idx + i * 16 + 8);
  }
  return true;
}

#endif
