/* volcomp_label.h — single-header compressor for multi-class probability volumes.
 *
 * A LABEL CHUNK covers one 128^3 region and holds up to 255 class PLANES; each
 * plane is a u8 probability map (0..255) for one class, as produced by a model.
 * Only classes present (any nonzero voxel) are stored; every other class
 * decodes as zeros. A class id is an explicit tag on its plane, so a decoded
 * chunk can never carry a class that was absent from the source and two
 * classes can never be swapped. Each stored plane is a volcomp.h stream at its
 * own q (a plane that would not compress below 2 MiB is stored raw), so the
 * value error statistics of a plane are exactly those of volcomp at that q.
 *
 * q is per plane: the header carries a default and each directory entry carries
 * the q that plane was actually encoded with, so one chunk can hold, say, a
 * lossless class map (q = VOLCOMP_Q_LOSSLESS) next to lossy probability planes
 * (q = 8). volcomp_label_encode() puts every plane at one q;
 * volcomp_label_encode_q() takes per-plane overrides.
 *
 * Thresholded / downscaled representations are derived by the consumer from
 * the decoded probabilities; this format stores probabilities only.
 *
 * Buffers are z-major like volcomp.h. Encoding performs one malloc; decoding
 * allocates nothing. This header includes volcomp.h. Format version 2
 * (spec/format.md §9); version 1 (one q for the whole chunk) is not read. */
#ifndef VOLCOMP_LABEL_H
#define VOLCOMP_LABEL_H

#include "volcomp.h"

#define VOLCOMP_LABEL_MAX_PLANES 255u
#define VOLCOMP_LABEL_HDR_BYTES 12u
#define VOLCOMP_LABEL_DIR_ENTRY 8u
/* Per-plane worst case: raw fallback (2 MiB). */
#define VOLCOMP_LABEL_PLANE_BOUND ((size_t)VOLCOMP_CHUNK_VOXELS)
#define VOLCOMP_LABEL_ENCODE_BOUND(nplanes) \
  ((size_t)VOLCOMP_LABEL_HDR_BYTES + (size_t)(nplanes) * (VOLCOMP_LABEL_DIR_ENTRY + VOLCOMP_LABEL_PLANE_BOUND))

typedef struct volcomp_label_plane {
  uint8_t cls;          /* class id; strictly ascending across the array          */
  const uint8_t *plane; /* 128^3 u8 z-major, or NULL for an all-zero (absent) class */
} volcomp_label_plane;

/* Encode nplanes class planes, every one at quantiser step q
 * (VOLCOMP_Q_LOSSLESS or 1..255). All-zero planes are dropped (they decode as
 * zeros). */
static inline volcomp_status volcomp_label_encode(const volcomp_label_plane *planes, uint32_t nplanes, float q,
                                                  void *restrict dst, size_t dst_cap, size_t *out_n);
/* As volcomp_label_encode, with per-plane quantiser steps: q_plane, if not NULL,
 * holds nplanes entries and a negative entry means "use the default q". The
 * default is recorded in the header; each plane's own q is recorded in its
 * directory entry. */
static inline volcomp_status volcomp_label_encode_q(const volcomp_label_plane *planes, uint32_t nplanes,
                                                    float q_default, const float *q_plane,
                                                    void *restrict dst, size_t dst_cap, size_t *out_n);
/* List the classes stored in a chunk: cls_out (>= 255 entries) receives them in
 * ascending order, *count their number. */
static inline volcomp_status volcomp_label_classes(const void *restrict enc, size_t enc_n,
                                                   uint8_t *cls_out, uint32_t *count);
/* Decode one class plane into dst (dst_cap >= VOLCOMP_CHUNK_VOXELS). A class
 * that is not stored decodes as all zeros (VOLCOMP_OK). Contents are
 * unspecified on failure. */
static inline volcomp_status volcomp_label_decode(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                  uint8_t *restrict dst, size_t dst_cap);
/* Decode block (bz,by,bx) of one class plane, as volcomp_decode_block. */
static inline volcomp_status volcomp_label_decode_block(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                        uint32_t bz, uint32_t by, uint32_t bx,
                                                        uint8_t *restrict dst_block, size_t dst_cap);
/* The default q recorded in the header. */
static inline volcomp_status volcomp_label_q(const void *restrict enc, size_t enc_n, float *q);
/* The q one stored class was encoded with (0 = lossless). A class that is not
 * stored decodes as zeros and reports q = VOLCOMP_Q_LOSSLESS. */
static inline volcomp_status volcomp_label_plane_q(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                   float *q);

/* ======================================================================== */
/*                          implementation                                  */
/* ======================================================================== */

#define VL_VERSION 2u
#define VL_MODE_IMAGE 0u
#define VL_MODE_RAW 1u

typedef struct vl_entry {
  uint8_t cls, mode;
  uint32_t q_raw, n;
  size_t off;
} vl_entry;

typedef struct vl_parsed {
  uint32_t nplanes, q_raw;
  vl_entry e[VOLCOMP_LABEL_MAX_PLANES];
} vl_parsed;

/* stream: "VOLL" u8 version u8 nplanes u16 q_raw(default) u32 reserved=0;
 * directory nplanes x { u8 cls, u8 mode, u16 q_raw, u32 n };
 * then the planes' bytes in directory order. Exact accounting. A q_raw of 0
 * (header or entry) is VOLCOMP_Q_LOSSLESS; otherwise it is 256..65280. */
static inline bool vl_q_raw_ok(uint32_t q) { return q == 0u || (q >= VF_Q_RAW_MIN && q <= VF_Q_RAW_MAX); }
static inline float vl_q_of(uint32_t q_raw) { return (float)q_raw * (1.0f / 256.0f); }
static inline bool vl_q_ok(float q) {
  return q == VOLCOMP_Q_LOSSLESS || (q >= VOLCOMP_Q_MIN && q <= VOLCOMP_Q_MAX);
}
/* q -> the 1/256-precision code the stream stores (0 stays 0 = lossless) */
static inline uint32_t vl_q_raw(float q) {
  if (q == VOLCOMP_Q_LOSSLESS) return 0u;
  uint32_t r = (uint32_t)lrintf(q * 256.0f);
  return r < VF_Q_RAW_MIN ? VF_Q_RAW_MIN : (r > VF_Q_RAW_MAX ? VF_Q_RAW_MAX : r);
}
static volcomp_status vl_parse(const uint8_t *s, size_t n, vl_parsed *p) {
  if (n < VOLCOMP_LABEL_HDR_BYTES || memcmp(s, "VOLL", 4) != 0) return VOLCOMP_ERR_CORRUPT;
  if (s[4] != VL_VERSION) return VOLCOMP_ERR_VERSION;
  if (vf_rd_u32(s + 8) != 0) return VOLCOMP_ERR_CORRUPT;
  p->nplanes = s[5];
  p->q_raw = vf_rd_u16(s + 6);
  if (!vl_q_raw_ok(p->q_raw)) return VOLCOMP_ERR_CORRUPT;
  size_t off = VOLCOMP_LABEL_HDR_BYTES + (size_t)p->nplanes * VOLCOMP_LABEL_DIR_ENTRY;
  if (n < off) return VOLCOMP_ERR_CORRUPT;
  for (uint32_t i = 0; i < p->nplanes; i++) {
    const uint8_t *d = s + VOLCOMP_LABEL_HDR_BYTES + (size_t)i * VOLCOMP_LABEL_DIR_ENTRY;
    vl_entry *e = &p->e[i];
    e->cls = d[0];
    e->mode = d[1];
    e->q_raw = vf_rd_u16(d + 2);
    e->n = vf_rd_u32(d + 4);
    e->off = off;
    if (!vl_q_raw_ok(e->q_raw)) return VOLCOMP_ERR_CORRUPT;
    if (i > 0 && e->cls <= p->e[i - 1].cls) return VOLCOMP_ERR_CORRUPT;
    if (e->mode == VL_MODE_RAW) {
      if (e->n != VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_CORRUPT;
    } else if (e->mode == VL_MODE_IMAGE) {
      if (e->n >= VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_CORRUPT;
    } else {
      return VOLCOMP_ERR_CORRUPT;
    }
    if (n - off < e->n) return VOLCOMP_ERR_CORRUPT;
    off += e->n;
  }
  if (off != n) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

static inline const vl_entry *vl_find(const vl_parsed *p, uint8_t cls) {
  for (uint32_t i = 0; i < p->nplanes; i++)
    if (p->e[i].cls == cls) return &p->e[i];
  return NULL;
}

static inline volcomp_status volcomp_label_q(const void *restrict enc, size_t enc_n, float *q) {
  if (!enc || !q) return VOLCOMP_ERR_ARG;
  vl_parsed p;
  volcomp_status st = vl_parse((const uint8_t *)enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  *q = vl_q_of(p.q_raw);
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_label_plane_q(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                   float *q) {
  if (!enc || !q) return VOLCOMP_ERR_ARG;
  vl_parsed p;
  volcomp_status st = vl_parse((const uint8_t *)enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  const vl_entry *e = vl_find(&p, cls);
  *q = e ? vl_q_of(e->q_raw) : VOLCOMP_Q_LOSSLESS;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_label_classes(const void *restrict enc, size_t enc_n,
                                                   uint8_t *cls_out, uint32_t *count) {
  if (!enc || !cls_out || !count) return VOLCOMP_ERR_ARG;
  vl_parsed p;
  volcomp_status st = vl_parse((const uint8_t *)enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  for (uint32_t i = 0; i < p.nplanes; i++) cls_out[i] = p.e[i].cls;
  *count = p.nplanes;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_label_decode(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                  uint8_t *restrict dst, size_t dst_cap) {
  if (!enc || !dst) return VOLCOMP_ERR_ARG;
  if (dst_cap < VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_SHORT_BUF;
  const uint8_t *s = (const uint8_t *)enc;
  vl_parsed p;
  volcomp_status st = vl_parse(s, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  const vl_entry *e = vl_find(&p, cls);
  if (!e) {
    memset(dst, 0, VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  }
  if (e->mode == VL_MODE_RAW) {
    memcpy(dst, s + e->off, VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  }
  return volcomp_decode(s + e->off, e->n, dst, dst_cap);
}

static inline volcomp_status volcomp_label_decode_block(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                        uint32_t bz, uint32_t by, uint32_t bx,
                                                        uint8_t *restrict dst_block, size_t dst_cap) {
  if (!enc || !dst_block) return VOLCOMP_ERR_ARG;
  if (bz >= VF_BLOCKS_AXIS || by >= VF_BLOCKS_AXIS || bx >= VF_BLOCKS_AXIS) return VOLCOMP_ERR_ARG;
  if (dst_cap < VOLCOMP_BLOCK_VOXELS) return VOLCOMP_ERR_SHORT_BUF;
  const uint8_t *s = (const uint8_t *)enc;
  vl_parsed p;
  volcomp_status st = vl_parse(s, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  const vl_entry *e = vl_find(&p, cls);
  if (!e) {
    memset(dst_block, 0, VOLCOMP_BLOCK_VOXELS);
    return VOLCOMP_OK;
  }
  if (e->mode == VL_MODE_RAW) {
    const uint8_t *src = s + e->off;
    for (uint32_t z = 0; z < VOLCOMP_BLOCK_DIM; z++)
      for (uint32_t y = 0; y < VOLCOMP_BLOCK_DIM; y++)
        memcpy(dst_block + (z * VOLCOMP_BLOCK_DIM + y) * VOLCOMP_BLOCK_DIM,
               src + (((size_t)bz * VOLCOMP_BLOCK_DIM + z) * VOLCOMP_CHUNK_DIM + by * VOLCOMP_BLOCK_DIM + y) *
                         VOLCOMP_CHUNK_DIM +
                   bx * VOLCOMP_BLOCK_DIM,
               VOLCOMP_BLOCK_DIM);
    return VOLCOMP_OK;
  }
  return volcomp_decode_block(s + e->off, e->n, bz, by, bx, dst_block, dst_cap);
}

static inline volcomp_status volcomp_label_encode_q(const volcomp_label_plane *planes, uint32_t nplanes,
                                                    float q_default, const float *q_plane,
                                                    void *restrict dst, size_t dst_cap, size_t *out_n) {
  if (!dst || !out_n || (nplanes && !planes)) return VOLCOMP_ERR_ARG;
  if (nplanes > VOLCOMP_LABEL_MAX_PLANES) return VOLCOMP_ERR_ARG;
  if (!vl_q_ok(q_default)) return VOLCOMP_ERR_ARG;
  for (uint32_t i = 0; i < nplanes; i++) {
    if (i > 0 && planes[i].cls <= planes[i - 1].cls) return VOLCOMP_ERR_ARG;
    if (q_plane && q_plane[i] >= 0.0f && !vl_q_ok(q_plane[i])) return VOLCOMP_ERR_ARG;
  }
  const size_t N = VOLCOMP_CHUNK_VOXELS;

  uint32_t present[VOLCOMP_LABEL_MAX_PLANES], np = 0;
  for (uint32_t i = 0; i < nplanes; i++) {
    const uint8_t *pl = planes[i].plane;
    if (!pl) continue;
    bool nz = false;
    for (size_t j = 0; j < N && !nz; j++) nz = pl[j] != 0;
    if (nz) present[np++] = i;
  }
  size_t dir_end = VOLCOMP_LABEL_HDR_BYTES + (size_t)np * VOLCOMP_LABEL_DIR_ENTRY;
  if (dst_cap < dir_end) return VOLCOMP_ERR_SHORT_BUF;

  uint8_t *s = (uint8_t *)dst;
  memcpy(s, "VOLL", 4);
  s[4] = VL_VERSION;
  s[5] = (uint8_t)np;
  vf_wr_u16(s + 6, vl_q_raw(q_default));
  vf_wr_u32(s + 8, 0);
  if (np == 0) {
    *out_n = dir_end;
    return VOLCOMP_OK;
  }
  /* volcomp_encode needs its full bound as capacity; encode into scratch and copy */
  uint8_t *scratch = (uint8_t *)VOLCOMP_MALLOC(VOLCOMP_ENCODE_BOUND);
  if (!scratch) return VOLCOMP_ERR_NOMEM;
  size_t off = dir_end;
  volcomp_status st = VOLCOMP_OK;
  for (uint32_t k = 0; k < np; k++) {
    uint32_t idx = present[k];
    const volcomp_label_plane *p = &planes[idx];
    float pq = (q_plane && q_plane[idx] >= 0.0f) ? q_plane[idx] : q_default;
    uint32_t q_raw = vl_q_raw(pq);
    size_t n;
    st = volcomp_encode(p->plane, vl_q_of(q_raw), scratch, VOLCOMP_ENCODE_BOUND, &n);
    if (st != VOLCOMP_OK) break;
    uint32_t mode = VL_MODE_IMAGE;
    const uint8_t *src = scratch;
    if (n >= N) { /* incompressible at this q: the plane itself is smaller */
      mode = VL_MODE_RAW;
      src = p->plane;
      n = N;
    }
    if (dst_cap - off < n) {
      st = VOLCOMP_ERR_SHORT_BUF;
      break;
    }
    memcpy(s + off, src, n);
    uint8_t *d = s + VOLCOMP_LABEL_HDR_BYTES + (size_t)k * VOLCOMP_LABEL_DIR_ENTRY;
    d[0] = p->cls;
    d[1] = (uint8_t)mode;
    vf_wr_u16(d + 2, q_raw);
    vf_wr_u32(d + 4, (uint32_t)n);
    off += n;
  }
  VOLCOMP_FREE(scratch);
  if (st != VOLCOMP_OK) return st;
  *out_n = off;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_label_encode(const volcomp_label_plane *planes, uint32_t nplanes, float q,
                                                  void *restrict dst, size_t dst_cap, size_t *out_n) {
  return volcomp_label_encode_q(planes, nplanes, q, NULL, dst, dst_cap, out_n);
}

#endif /* VOLCOMP_LABEL_H */
