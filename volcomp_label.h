/* volcomp_label.h — single-header compressor for multi-class u8 label volumes.
 *
 * A LABEL CHUNK covers one 128^3 region and holds up to 255 class PLANES; each
 * plane is a u8 value map (a mask, an id map, or a probability map) for one
 * class. Only classes present (any nonzero voxel) are stored; every other
 * class decodes as zeros. A class id is an explicit tag on its plane, so a
 * decoded chunk can never carry a class that was absent from the source and
 * two classes can never be swapped: the lossy freedom is purely spatial.
 *
 * Per plane the encoder picks one of four modes (spec/format.md §9):
 *   const    all voxels share one value                       (0 payload bytes)
 *   palette  <= 16 distinct values: one context-coded mask per value; values
 *            are reproduced exactly, so an id map decodes to the same ids
 *   image    > 16 distinct values (a probability map): volcomp.h stream at q
 *   raw      fallback when a coded plane would not be smaller than 2 MiB
 *
 * tolerance t (0..7 voxels): before coding, a palette plane is replaced by its
 * plurality vote over the (2t+1)^3 window (ties keep the source value). A voxel
 * farther than t from a source boundary has a uniform window and is therefore
 * unchanged; every change lies within t voxels of a boundary, in the direction
 * of a value that was present there. t = 0 is lossless. Const and image planes
 * ignore t (image planes are shaped by q instead).
 *
 * Buffers are z-major like volcomp.h. Encoding performs one malloc; decoding
 * allocates nothing. Include after nothing in particular: this header includes
 * volcomp.h itself. Format version 1. */
#ifndef VOLCOMP_LABEL_H
#define VOLCOMP_LABEL_H

#include "volcomp.h"

#define VOLCOMP_LABEL_MAX_PLANES 255u
#define VOLCOMP_LABEL_MAX_TOLERANCE 7u
#define VOLCOMP_LABEL_PALETTE_MAX 16u
#define VOLCOMP_LABEL_HDR_BYTES 12u
#define VOLCOMP_LABEL_DIR_ENTRY 8u
/* Per-plane worst case: directory entry + header + raw fallback (2 MiB). */
#define VOLCOMP_LABEL_PLANE_BOUND ((size_t)VOLCOMP_CHUNK_VOXELS + 128u)
#define VOLCOMP_LABEL_ENCODE_BOUND(nplanes) \
  ((size_t)VOLCOMP_LABEL_HDR_BYTES + (size_t)(nplanes) * (VOLCOMP_LABEL_DIR_ENTRY + VOLCOMP_LABEL_PLANE_BOUND))

typedef struct volcomp_label_plane {
  uint8_t cls;          /* class id; strictly ascending across the array          */
  const uint8_t *plane; /* 128^3 u8 z-major, or NULL for an all-zero (absent) class */
} volcomp_label_plane;

/* Encode nplanes class planes. All-zero planes are dropped (they decode as
 * zeros). q (1..255) is used only for image-mode planes. */
static inline volcomp_status volcomp_label_encode(const volcomp_label_plane *planes, uint32_t nplanes,
                                                  uint32_t tolerance, float q, void *restrict dst,
                                                  size_t dst_cap, size_t *out_n);
/* List the classes stored in a chunk: cls_out (>= 255 entries) receives them in
 * ascending order, *count their number. */
static inline volcomp_status volcomp_label_classes(const void *restrict enc, size_t enc_n,
                                                   uint8_t *cls_out, uint32_t *count);
/* Decode one class plane into dst (dst_cap >= VOLCOMP_CHUNK_VOXELS). A class
 * that is not stored decodes as all zeros (VOLCOMP_OK). Contents are
 * unspecified on failure. */
static inline volcomp_status volcomp_label_decode(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                  uint8_t *restrict dst, size_t dst_cap);
/* Tolerance and q recorded in the header. */
static inline volcomp_status volcomp_label_params(const void *restrict enc, size_t enc_n,
                                                  uint32_t *tolerance, float *q);

/* ======================================================================== */
/*                          implementation                                  */
/* ======================================================================== */

#define VL_VERSION 1u
#define VL_MODE_CONST 0u
#define VL_MODE_PALETTE 1u
#define VL_MODE_IMAGE 2u
#define VL_MODE_RAW 3u
#define VL_DIM 128u
#define VL_NCTX 1024u
#define VL_PROB_BITS 11u
#define VL_PROB_INIT (1u << (VL_PROB_BITS - 1))
#define VL_PROB_SHIFT 5u
#define VL_RC_TOP (1u << 24)
#define VL_PALETTE_HDR_MAX (2u + VOLCOMP_LABEL_PALETTE_MAX + 4u * (VOLCOMP_LABEL_PALETTE_MAX - 1u))

/* ---- adaptive binary range coder (LZMA-style) ---- */
typedef struct vl_rce {
  uint64_t low;
  uint32_t range;
  uint8_t cache;
  uint64_t cache_size;
  uint8_t *out;
  size_t cap, n;
  bool overflow;
} vl_rce;

static inline void vl_rce_init(vl_rce *e, uint8_t *out, size_t cap) {
  e->low = 0;
  e->range = 0xFFFFFFFFu;
  e->cache = 0;
  e->cache_size = 1;
  e->out = out;
  e->cap = cap;
  e->n = 0;
  e->overflow = false;
}
static inline void vl_rce_put(vl_rce *e, uint8_t b) {
  if (e->n < e->cap) e->out[e->n] = b;
  else e->overflow = true;
  e->n++;
}
static inline void vl_rce_shift(vl_rce *e) {
  if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
    uint8_t carry = (uint8_t)(e->low >> 32);
    vl_rce_put(e, (uint8_t)(e->cache + carry));
    for (; e->cache_size > 1; e->cache_size--) vl_rce_put(e, (uint8_t)(0xFFu + carry));
    e->cache = (uint8_t)(e->low >> 24);
    e->cache_size = 0;
  }
  e->cache_size++;
  e->low = (e->low & 0x00FFFFFFu) << 8;
}
static inline void vl_rce_bit(vl_rce *e, uint16_t *p, uint32_t bit) {
  uint32_t bound = (e->range >> VL_PROB_BITS) * *p;
  if (bit == 0) {
    e->range = bound;
    *p = (uint16_t)(*p + (((1u << VL_PROB_BITS) - *p) >> VL_PROB_SHIFT));
  } else {
    e->low += bound;
    e->range -= bound;
    *p = (uint16_t)(*p - (*p >> VL_PROB_SHIFT));
  }
  while (e->range < VL_RC_TOP) {
    e->range <<= 8;
    vl_rce_shift(e);
  }
}
static inline void vl_rce_flush(vl_rce *e) {
  for (int i = 0; i < 5; i++) vl_rce_shift(e);
}

typedef struct vl_rcd {
  uint32_t code, range;
  const uint8_t *in;
  size_t n, pos;
  bool bad;
} vl_rcd;

static inline uint8_t vl_rcd_byte(vl_rcd *d) {
  if (d->pos < d->n) return d->in[d->pos++];
  d->bad = true;
  return 0;
}
static inline bool vl_rcd_init(vl_rcd *d, const uint8_t *in, size_t n) {
  d->in = in;
  d->n = n;
  d->pos = 0;
  d->bad = false;
  d->range = 0xFFFFFFFFu;
  if (n < 5 || in[0] != 0) return false;
  d->pos = 1;
  d->code = 0;
  for (int i = 0; i < 4; i++) d->code = d->code << 8 | vl_rcd_byte(d);
  return true;
}
static inline uint32_t vl_rcd_bit(vl_rcd *d, uint16_t *p) {
  uint32_t bound = (d->range >> VL_PROB_BITS) * *p, bit;
  if (d->code < bound) {
    d->range = bound;
    *p = (uint16_t)(*p + (((1u << VL_PROB_BITS) - *p) >> VL_PROB_SHIFT));
    bit = 0;
  } else {
    d->code -= bound;
    d->range -= bound;
    *p = (uint16_t)(*p - (*p >> VL_PROB_SHIFT));
    bit = 1;
  }
  while (d->range < VL_RC_TOP) {
    d->range <<= 8;
    d->code = d->code << 8 | vl_rcd_byte(d);
  }
  return bit;
}
/* The encoder emits exactly 5 + (number of renormalisations) bytes and the
 * decoder consumes the same count, so a valid stream is consumed exactly. */
static inline bool vl_rcd_finished(const vl_rcd *d) { return !d->bad && d->pos == d->n; }

/* ---- mask context: 10 causal neighbours of (z,y,x), outside the chunk = 0 ----
 * bit(z,y,x) = plane[z,y,x] == v; the plane being coded holds the value v at
 * mask voxels, so the mask never needs its own buffer (decoder: undecoded
 * voxels are never read, all neighbours are causal in z-major order). */
static inline uint32_t vl_ctx(const uint8_t *restrict pl, uint8_t v, uint32_t z, uint32_t y, uint32_t x) {
  size_t i = ((size_t)z * VL_DIM + y) * VL_DIM + x;
  uint32_t c = 0;
#define VL_B(cond, idx, sh) c |= ((cond) && pl[idx] == v ? 1u : 0u) << (sh)
  VL_B(x > 0, i - 1, 0);
  VL_B(x > 1, i - 2, 1);
  VL_B(y > 0, i - VL_DIM, 2);
  VL_B(y > 0 && x > 0, i - VL_DIM - 1, 3);
  VL_B(y > 0 && x + 1 < VL_DIM, i - VL_DIM + 1, 4);
  size_t j = i - VL_DIM * VL_DIM; /* only dereferenced when z > 0 */
  VL_B(z > 0, j, 5);
  VL_B(z > 0 && x > 0, j - 1, 6);
  VL_B(z > 0 && x + 1 < VL_DIM, j + 1, 7);
  VL_B(z > 0 && y > 0, j - VL_DIM, 8);
  VL_B(z > 0 && y + 1 < VL_DIM, j + VL_DIM, 9);
#undef VL_B
  return c;
}

/* Code the mask "plane == v" over voxels whose value has rank >= k under
 * rank[] (voxels of earlier palette entries are skipped). */
static size_t vl_mask_encode(const uint8_t *restrict pl, uint8_t v, const uint8_t rank[256], uint32_t k,
                             uint8_t *out, size_t cap, bool *overflow) {
  uint16_t prob[VL_NCTX];
  for (uint32_t i = 0; i < VL_NCTX; i++) prob[i] = VL_PROB_INIT;
  vl_rce e;
  vl_rce_init(&e, out, cap);
  size_t i = 0;
  for (uint32_t z = 0; z < VL_DIM; z++)
    for (uint32_t y = 0; y < VL_DIM; y++)
      for (uint32_t x = 0; x < VL_DIM; x++, i++) {
        if (rank[pl[i]] < k) continue;
        vl_rce_bit(&e, &prob[vl_ctx(pl, v, z, y, x)], pl[i] == v);
      }
  vl_rce_flush(&e);
  *overflow = e.overflow;
  return e.n;
}
/* Inverse: voxels currently holding `fill` are undecided; set them to v where
 * the mask says so. */
static bool vl_mask_decode(uint8_t *restrict pl, uint8_t v, uint8_t fill, const uint8_t *in, size_t n) {
  uint16_t prob[VL_NCTX];
  for (uint32_t i = 0; i < VL_NCTX; i++) prob[i] = VL_PROB_INIT;
  vl_rcd d;
  if (!vl_rcd_init(&d, in, n)) return false;
  size_t i = 0;
  for (uint32_t z = 0; z < VL_DIM; z++)
    for (uint32_t y = 0; y < VL_DIM; y++)
      for (uint32_t x = 0; x < VL_DIM; x++, i++) {
        if (pl[i] != fill) continue;
        if (vl_rcd_bit(&d, &prob[vl_ctx(pl, v, z, y, x)])) pl[i] = v;
      }
  return vl_rcd_finished(&d);
}

/* ---- plurality filter over a (2t+1)^3 window, in-bounds voxels only ----
 * scratch: cnt (u16 x N) running box sums, best (u16 x N), own (u16 x N). */
static void vl_box_axis(uint16_t *restrict a, size_t stride, size_t len, size_t nlines, size_t line_stride,
                        uint32_t t, uint16_t *restrict tmp) {
  for (size_t l = 0; l < nlines; l++) {
    uint16_t *line = a + l * line_stride;
    /* prefix sums with a 0 sentinel */
    tmp[0] = 0;
    for (size_t i = 0; i < len; i++) tmp[i + 1] = (uint16_t)(tmp[i] + line[i * stride]);
    for (size_t i = 0; i < len; i++) {
      size_t lo = i > t ? i - t : 0, hi = i + t + 1 < len ? i + t + 1 : len;
      line[i * stride] = (uint16_t)(tmp[hi] - tmp[lo]);
    }
  }
}
/* 3-D box count of (src == v) into cnt. Lines along x are contiguous; y and z
 * lines are strided. tmp needs len+1 entries. */
static void vl_box3(const uint8_t *restrict src, uint8_t v, uint32_t t, uint16_t *restrict cnt,
                    uint16_t *restrict tmp) {
  const size_t N = VOLCOMP_CHUNK_VOXELS, D = VL_DIM;
  for (size_t i = 0; i < N; i++) cnt[i] = src[i] == v;
  /* x: lines are contiguous runs of D */
  vl_box_axis(cnt, 1, D, D * D, D, t, tmp);
  /* y: for each z, D lines of stride D starting at z*D*D + x */
  for (size_t z = 0; z < D; z++) vl_box_axis(cnt + z * D * D, D, D, D, 1, t, tmp);
  /* z: lines of stride D*D starting at y*D + x */
  vl_box_axis(cnt, D * D, D, D * D, 1, t, tmp);
}
/* dst = plurality vote of src over the window; ties keep src. vals[nv] are the
 * distinct values of src. Guarantee: a voxel whose window is uniform is unchanged. */
static void vl_plurality(const uint8_t *restrict src, uint8_t *restrict dst, const uint8_t *vals, uint32_t nv,
                         uint32_t t, uint16_t *restrict cnt, uint16_t *restrict best, uint16_t *restrict own,
                         uint16_t *restrict tmp) {
  const size_t N = VOLCOMP_CHUNK_VOXELS;
  memset(best, 0, N * sizeof *best);
  for (uint32_t k = 0; k < nv; k++) {
    uint8_t v = vals[k];
    vl_box3(src, v, t, cnt, tmp);
    for (size_t i = 0; i < N; i++) {
      if (src[i] == v) own[i] = cnt[i];
      if (cnt[i] > best[i]) {
        best[i] = cnt[i];
        dst[i] = v;
      }
    }
  }
  for (size_t i = 0; i < N; i++)
    if (own[i] == best[i]) dst[i] = src[i];
}

/* ---- stream helpers ---- */
static inline void vl_wr_hdr(uint8_t *h, uint32_t nplanes, uint32_t tol, uint32_t q_raw) {
  memcpy(h, "VOLL", 4);
  h[4] = VL_VERSION;
  h[5] = (uint8_t)nplanes;
  h[6] = (uint8_t)tol;
  h[7] = 0;
  vf_wr_u16(h + 8, q_raw);
  h[10] = 0;
  h[11] = 0;
}

typedef struct vl_entry {
  uint8_t cls, mode;
  uint32_t hdr_n, payload_n;
  size_t off; /* of the header bytes within the stream */
} vl_entry;

typedef struct vl_parsed {
  uint32_t nplanes, tol, q_raw;
  vl_entry e[VOLCOMP_LABEL_MAX_PLANES];
} vl_parsed;

static volcomp_status vl_parse(const uint8_t *s, size_t n, vl_parsed *p) {
  if (n < VOLCOMP_LABEL_HDR_BYTES || memcmp(s, "VOLL", 4) != 0) return VOLCOMP_ERR_CORRUPT;
  if (s[4] != VL_VERSION) return VOLCOMP_ERR_VERSION;
  if (s[7] != 0 || s[10] != 0 || s[11] != 0) return VOLCOMP_ERR_CORRUPT;
  p->nplanes = s[5];
  p->tol = s[6];
  p->q_raw = vf_rd_u16(s + 8);
  if (p->tol > VOLCOMP_LABEL_MAX_TOLERANCE) return VOLCOMP_ERR_CORRUPT;
  if (p->q_raw != 0 && (p->q_raw < VF_Q_RAW_MIN || p->q_raw > VF_Q_RAW_MAX)) return VOLCOMP_ERR_CORRUPT;
  size_t off = VOLCOMP_LABEL_HDR_BYTES + (size_t)p->nplanes * VOLCOMP_LABEL_DIR_ENTRY;
  if (n < off) return VOLCOMP_ERR_CORRUPT;
  for (uint32_t i = 0; i < p->nplanes; i++) {
    const uint8_t *d = s + VOLCOMP_LABEL_HDR_BYTES + (size_t)i * VOLCOMP_LABEL_DIR_ENTRY;
    vl_entry *e = &p->e[i];
    e->cls = d[0];
    e->mode = d[1];
    e->hdr_n = vf_rd_u16(d + 2);
    e->payload_n = vf_rd_u32(d + 4);
    e->off = off;
    if (i > 0 && e->cls <= p->e[i - 1].cls) return VOLCOMP_ERR_CORRUPT;
    switch (e->mode) {
    case VL_MODE_CONST:
      if (e->hdr_n != 1 || e->payload_n != 0) return VOLCOMP_ERR_CORRUPT;
      break;
    case VL_MODE_PALETTE:
      if (e->hdr_n < 2 || e->hdr_n > VL_PALETTE_HDR_MAX) return VOLCOMP_ERR_CORRUPT;
      break;
    case VL_MODE_IMAGE:
      if (e->hdr_n != 0 || p->q_raw == 0) return VOLCOMP_ERR_CORRUPT;
      break;
    case VL_MODE_RAW:
      if (e->hdr_n != 0 || e->payload_n != VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_CORRUPT;
      break;
    default:
      return VOLCOMP_ERR_CORRUPT;
    }
    if (n - off < (size_t)e->hdr_n + e->payload_n) return VOLCOMP_ERR_CORRUPT;
    off += (size_t)e->hdr_n + e->payload_n;
  }
  if (off != n) return VOLCOMP_ERR_CORRUPT;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_label_params(const void *restrict enc, size_t enc_n,
                                                  uint32_t *tolerance, float *q) {
  if (!enc) return VOLCOMP_ERR_ARG;
  vl_parsed p;
  volcomp_status st = vl_parse((const uint8_t *)enc, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  if (tolerance) *tolerance = p.tol;
  if (q) *q = (float)p.q_raw / 256.0f;
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

static volcomp_status vl_decode_palette(const uint8_t *hdr, uint32_t hdr_n, const uint8_t *pay, size_t pay_n,
                                        uint8_t *restrict dst) {
  uint32_t nv = hdr[0];
  if (nv < 2 || nv > VOLCOMP_LABEL_PALETTE_MAX) return VOLCOMP_ERR_CORRUPT;
  if (hdr_n != 1 + nv + 4 * (nv - 1)) return VOLCOMP_ERR_CORRUPT;
  const uint8_t *vals = hdr + 1;
  for (uint32_t i = 0; i < nv; i++)
    for (uint32_t j = 0; j < i; j++)
      if (vals[i] == vals[j]) return VOLCOMP_ERR_CORRUPT;
  uint8_t fill = vals[nv - 1];
  memset(dst, fill, VOLCOMP_CHUNK_VOXELS);
  size_t off = 0;
  for (uint32_t k = 0; k + 1 < nv; k++) {
    uint32_t len = vf_rd_u32(hdr + 1 + nv + 4 * k);
    if (pay_n - off < len) return VOLCOMP_ERR_CORRUPT;
    if (!vl_mask_decode(dst, vals[k], fill, pay + off, len)) return VOLCOMP_ERR_CORRUPT;
    off += len;
  }
  return off == pay_n ? VOLCOMP_OK : VOLCOMP_ERR_CORRUPT;
}

static inline volcomp_status volcomp_label_decode(const void *restrict enc, size_t enc_n, uint8_t cls,
                                                  uint8_t *restrict dst, size_t dst_cap) {
  if (!enc || !dst) return VOLCOMP_ERR_ARG;
  if (dst_cap < VOLCOMP_CHUNK_VOXELS) return VOLCOMP_ERR_SHORT_BUF;
  const uint8_t *s = (const uint8_t *)enc;
  vl_parsed p;
  volcomp_status st = vl_parse(s, enc_n, &p);
  if (st != VOLCOMP_OK) return st;
  const vl_entry *e = NULL;
  for (uint32_t i = 0; i < p.nplanes; i++)
    if (p.e[i].cls == cls) e = &p.e[i];
  if (!e) {
    memset(dst, 0, VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  }
  const uint8_t *hdr = s + e->off, *pay = hdr + e->hdr_n;
  switch (e->mode) {
  case VL_MODE_CONST:
    memset(dst, hdr[0], VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  case VL_MODE_PALETTE:
    return vl_decode_palette(hdr, e->hdr_n, pay, e->payload_n, dst);
  case VL_MODE_IMAGE:
    return volcomp_decode(pay, e->payload_n, dst, dst_cap);
  case VL_MODE_RAW:
    memcpy(dst, pay, VOLCOMP_CHUNK_VOXELS);
    return VOLCOMP_OK;
  default:
    return VOLCOMP_ERR_CORRUPT;
  }
}

/* ---- encoder ---- */
typedef struct vl_scratch {
  uint8_t *filt;                    /* filtered plane                */
  uint16_t *cnt, *best, *own, *tmp; /* plurality filter              */
  uint8_t *img;                     /* volcomp scratch (image mode)  */
} vl_scratch;

static volcomp_status vl_encode_plane(const uint8_t *restrict src, uint32_t tol, float q, vl_scratch *sc,
                                      uint8_t *out, size_t cap, size_t *entry_hdr_n, size_t *entry_pay_n,
                                      uint32_t *mode) {
  const size_t N = VOLCOMP_CHUNK_VOXELS;
  /* histogram -> distinct values, most frequent first */
  uint32_t hist[256] = {0};
  for (size_t i = 0; i < N; i++) hist[src[i]]++;
  uint8_t vals[256];
  uint32_t nv = 0;
  for (uint32_t v = 0; v < 256; v++)
    if (hist[v]) vals[nv++] = (uint8_t)v;
  if (nv == 1) {
    if (cap < 1) return VOLCOMP_ERR_SHORT_BUF;
    out[0] = vals[0];
    *entry_hdr_n = 1;
    *entry_pay_n = 0;
    *mode = VL_MODE_CONST;
    return VOLCOMP_OK;
  }
  if (nv <= VOLCOMP_LABEL_PALETTE_MAX) {
    /* sort by count descending (stable on value) */
    for (uint32_t i = 1; i < nv; i++)
      for (uint32_t j = i; j > 0 && hist[vals[j]] > hist[vals[j - 1]]; j--) {
        uint8_t t = vals[j];
        vals[j] = vals[j - 1];
        vals[j - 1] = t;
      }
    const uint8_t *pl = src;
    if (tol > 0) {
      vl_plurality(src, sc->filt, vals, nv, tol, sc->cnt, sc->best, sc->own, sc->tmp);
      pl = sc->filt;
      /* the filter may remove a value entirely; re-derive the palette */
      uint32_t h2[256] = {0};
      for (size_t i = 0; i < N; i++) h2[pl[i]]++;
      uint32_t nv2 = 0;
      for (uint32_t k = 0; k < nv; k++)
        if (h2[vals[k]]) vals[nv2++] = vals[k];
      nv = nv2;
      if (nv == 1) {
        if (cap < 1) return VOLCOMP_ERR_SHORT_BUF;
        out[0] = vals[0];
        *entry_hdr_n = 1;
        *entry_pay_n = 0;
        *mode = VL_MODE_CONST;
        return VOLCOMP_OK;
      }
    }
    uint8_t rank[256];
    memset(rank, 0xFF, sizeof rank);
    for (uint32_t k = 0; k < nv; k++) rank[vals[k]] = (uint8_t)k;
    size_t hdr_n = 1 + nv + 4 * (nv - 1);
    if (cap < hdr_n) return VOLCOMP_ERR_SHORT_BUF;
    out[0] = (uint8_t)nv;
    memcpy(out + 1, vals, nv);
    size_t off = hdr_n;
    bool overflow = false;
    for (uint32_t k = 0; k + 1 < nv && !overflow; k++) {
      size_t len = vl_mask_encode(pl, vals[k], rank, k, out + off, cap > off ? cap - off : 0, &overflow);
      if (len > UINT32_MAX) overflow = true;
      vf_wr_u32(out + 1 + nv + 4 * k, (uint32_t)len);
      off += len;
      if (off >= N) overflow = true;
    }
    if (!overflow) {
      *entry_hdr_n = hdr_n;
      *entry_pay_n = off - hdr_n;
      *mode = VL_MODE_PALETTE;
      return VOLCOMP_OK;
    }
    /* coded plane not smaller than raw (or buffer too small for it): fall
     * through to raw, which stores the filtered plane */
    if (cap < N) return VOLCOMP_ERR_SHORT_BUF;
    memcpy(out, pl, N);
    *entry_hdr_n = 0;
    *entry_pay_n = N;
    *mode = VL_MODE_RAW;
    return VOLCOMP_OK;
  }
  /* image mode */
  if (!sc->img) {
    sc->img = (uint8_t *)VOLCOMP_MALLOC(VOLCOMP_ENCODE_BOUND);
    if (!sc->img) return VOLCOMP_ERR_NOMEM;
  }
  size_t n;
  volcomp_status st = volcomp_encode(src, q, sc->img, VOLCOMP_ENCODE_BOUND, &n);
  if (st != VOLCOMP_OK) return st;
  if (n < N) {
    if (cap < n) return VOLCOMP_ERR_SHORT_BUF;
    memcpy(out, sc->img, n);
    *entry_hdr_n = 0;
    *entry_pay_n = n;
    *mode = VL_MODE_IMAGE;
    return VOLCOMP_OK;
  }
  if (cap < N) return VOLCOMP_ERR_SHORT_BUF;
  memcpy(out, src, N);
  *entry_hdr_n = 0;
  *entry_pay_n = N;
  *mode = VL_MODE_RAW;
  return VOLCOMP_OK;
}

static inline volcomp_status volcomp_label_encode(const volcomp_label_plane *planes, uint32_t nplanes,
                                                  uint32_t tolerance, float q, void *restrict dst,
                                                  size_t dst_cap, size_t *out_n) {
  if (!dst || !out_n || (nplanes && !planes)) return VOLCOMP_ERR_ARG;
  if (nplanes > VOLCOMP_LABEL_MAX_PLANES || tolerance > VOLCOMP_LABEL_MAX_TOLERANCE) return VOLCOMP_ERR_ARG;
  if (!(q >= VOLCOMP_Q_MIN && q <= VOLCOMP_Q_MAX)) return VOLCOMP_ERR_ARG;
  for (uint32_t i = 0; i < nplanes; i++)
    if (i > 0 && planes[i].cls <= planes[i - 1].cls) return VOLCOMP_ERR_ARG;
  const size_t N = VOLCOMP_CHUNK_VOXELS;
  uint32_t q_raw = (uint32_t)lrintf(q * 256.0f);
  if (q_raw < VF_Q_RAW_MIN) q_raw = VF_Q_RAW_MIN;
  if (q_raw > VF_Q_RAW_MAX) q_raw = VF_Q_RAW_MAX;

  /* present planes */
  uint32_t present[VOLCOMP_LABEL_MAX_PLANES], np = 0;
  for (uint32_t i = 0; i < nplanes; i++) {
    const uint8_t *pl = planes[i].plane;
    if (!pl) continue;
    bool nz = false;
    for (size_t j = 0; j < N && !nz; j += 64) {
      size_t lim = j + 64 < N ? j + 64 : N;
      for (size_t k = j; k < lim; k++)
        if (pl[k]) {
          nz = true;
          break;
        }
    }
    if (nz) present[np++] = i;
  }
  size_t dir_end = VOLCOMP_LABEL_HDR_BYTES + (size_t)np * VOLCOMP_LABEL_DIR_ENTRY;
  if (dst_cap < dir_end) return VOLCOMP_ERR_SHORT_BUF;

  /* scratch: filt (N) + 4 x u16 volumes (cnt, best, own) + tmp (D+1) */
  vl_scratch sc = {0};
  uint8_t *mem = NULL;
  if (tolerance > 0) {
    size_t bytes = N + 3 * N * sizeof(uint16_t) + (VL_DIM + 1) * sizeof(uint16_t);
    mem = (uint8_t *)VOLCOMP_MALLOC(bytes);
    if (!mem) return VOLCOMP_ERR_NOMEM;
    sc.filt = mem;
    sc.cnt = (uint16_t *)(void *)(mem + N);
    sc.best = sc.cnt + N;
    sc.own = sc.best + N;
    sc.tmp = sc.own + N;
  }
  uint8_t *s = (uint8_t *)dst;
  vl_wr_hdr(s, np, tolerance, q_raw);
  size_t off = dir_end;
  volcomp_status st = VOLCOMP_OK;
  for (uint32_t k = 0; k < np && st == VOLCOMP_OK; k++) {
    const volcomp_label_plane *p = &planes[present[k]];
    size_t hn = 0, pn = 0;
    uint32_t mode = 0;
    st = vl_encode_plane(p->plane, tolerance, q, &sc, s + off, dst_cap - off, &hn, &pn, &mode);
    if (st != VOLCOMP_OK) break;
    uint8_t *d = s + VOLCOMP_LABEL_HDR_BYTES + (size_t)k * VOLCOMP_LABEL_DIR_ENTRY;
    d[0] = p->cls;
    d[1] = (uint8_t)mode;
    vf_wr_u16(d + 2, (uint32_t)hn);
    vf_wr_u32(d + 4, (uint32_t)pn);
    off += hn + pn;
  }
  if (mem) VOLCOMP_FREE(mem);
  if (sc.img) VOLCOMP_FREE(sc.img);
  if (st != VOLCOMP_OK) return st;
  *out_n = off;
  return VOLCOMP_OK;
}

#endif /* VOLCOMP_LABEL_H */
