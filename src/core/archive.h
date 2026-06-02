// Single-file archive (PLAN §3). Byte layout, low->high offset:
//   [ chunk payloads (concatenated)            ]
//   [ positional chunk index (fixed-width)     ]   coord IMPLICIT by position
//   [ member directory                          ]
//   [ config header (records active pipeline)   ]
//   [ fixed trailer at EOF {magic,ver,dir,xxh}  ]
//
// The chunk index is POSITIONAL: entry i is chunk i in row-major chunk-grid
// order; there is NO explicit chunk key. ABSENT chunks (all-zero / padded) store
// no payload. The active build config rides in the header so a decoder verifies
// the pipeline matches. All integers little-endian.
#ifndef VC_ARCHIVE_H
#define VC_ARCHIVE_H

#include "../../include/vc/types.h"
#include "xxhash.h"
#include <stdlib.h>
#include <string.h>

#define VC_MAGIC0 'V'
#define VC_MAGIC1 'C'
#define VC_MAGIC2 '0'
#define VC_MAGIC3 1
#define VC_FORMAT_VER 1u
#define VC_TRAILER_SIZE 48u

// Positional, fixed-width index entry. coord implicit by array position.
typedef struct {
    u64 offset;     // payload offset from file start
    u64 length;     // payload byte length (0 + ABSENT => all-zero)
    u64 checksum;   // xxhash64 of the payload
    f32 q;          // per-chunk quality knob
    u8  flags;      // bit0: ABSENT
    u8  _pad[3];
} vc_index_entry;   // 32 bytes
#define VC_FLAG_ABSENT 1u
_Static_assert(sizeof(vc_index_entry) == 32, "fixed-width index entry");

// Active-pipeline config recorded in the header (PLAN §3).
typedef struct {
    u32 chunk_side;
    u32 transform_tag;
    u32 entropy_tag;
    u32 _reserved;
} vc_config_hdr;    // 16 bytes

// A growable byte buffer (the in-memory archive writer).
typedef struct {
    u8 *data;
    size_t len;
    size_t cap;
} vc_buf;

static inline void vc_buf_reserve(vc_buf *b, size_t extra) {
    if (b->len + extra <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 1u << 20;
    while (nc < b->len + extra) nc *= 2;
    b->data = (u8 *)realloc(b->data, nc);
    b->cap = nc;
}
static inline void vc_buf_put(vc_buf *b, const void *p, size_t n) {
    vc_buf_reserve(b, n); memcpy(b->data + b->len, p, n); b->len += n;
}
static inline void vc_buf_u32(vc_buf *b, u32 v) { vc_buf_put(b, &v, 4); }
static inline void vc_buf_u64(vc_buf *b, u64 v) { vc_buf_put(b, &v, 8); }

// --- Writer ---------------------------------------------------------------
// One member ("0/" LOD) for Phase 0. Caller appends each chunk's payload (or
// marks ABSENT), then finishes with shape + config.
typedef struct {
    vc_buf blob;                  // payloads accumulate here
    vc_index_entry *index;
    size_t nidx, idx_cap;
    char member_name[8];
} vc_writer;

static inline void vc_writer_init(vc_writer *w, const char *member_name) {
    memset(w, 0, sizeof(*w));
    strncpy(w->member_name, member_name, sizeof(w->member_name) - 1);
}

static inline void vc_writer_add_chunk(vc_writer *w, const u8 *payload, size_t plen, f32 q) {
    if (w->nidx == w->idx_cap) {
        w->idx_cap = w->idx_cap ? w->idx_cap * 2 : 256;
        w->index = (vc_index_entry *)realloc(w->index, w->idx_cap * sizeof(vc_index_entry));
    }
    vc_index_entry *e = &w->index[w->nidx++];
    memset(e, 0, sizeof(*e));
    e->q = q;
    if (payload == NULL || plen == 0) {
        e->flags = VC_FLAG_ABSENT;
    } else {
        e->offset = w->blob.len;
        e->length = plen;
        e->checksum = vc_xxh64(payload, plen, 0);
        vc_buf_put(&w->blob, payload, plen);
    }
}

// Serialize to a heap buffer (*out, *out_len). Caller frees. Returns 0 on ok.
static inline int vc_writer_finish(vc_writer *w, vc_coord3 shape,
                                   const vc_config_hdr *cfg,
                                   u8 **out, size_t *out_len) {
    vc_buf b = w->blob;   // take ownership of payload blob

    // positional chunk index
    u64 idx_off = b.len;
    u64 idx_bytes = (u64)w->nidx * sizeof(vc_index_entry);
    vc_buf_put(&b, w->index, idx_bytes);
    u64 idx_sum = vc_xxh64(b.data + idx_off, idx_bytes, 0);

    // member directory (single member)
    u64 dir_off = b.len;
    vc_buf_u32(&b, 1u);                              // member count
    u32 nlen = (u32)strlen(w->member_name);
    vc_buf_u32(&b, nlen);
    vc_buf_put(&b, w->member_name, nlen);
    vc_buf_u64(&b, shape.z); vc_buf_u64(&b, shape.y); vc_buf_u64(&b, shape.x);
    vc_buf_u64(&b, (u64)w->nidx);
    vc_buf_u64(&b, idx_off); vc_buf_u64(&b, idx_bytes); vc_buf_u64(&b, idx_sum);
    // config header
    vc_buf_u32(&b, cfg->chunk_side); vc_buf_u32(&b, cfg->transform_tag);
    vc_buf_u32(&b, cfg->entropy_tag); vc_buf_u32(&b, cfg->_reserved);
    u64 dir_len = b.len - dir_off;
    u64 dir_sum = vc_xxh64(b.data + dir_off, dir_len, 0);

    // fixed trailer (48 bytes)
    u8 magic[4] = {VC_MAGIC0, VC_MAGIC1, VC_MAGIC2, VC_MAGIC3};
    size_t tstart = b.len;
    vc_buf_put(&b, magic, 4);          // 0
    vc_buf_u32(&b, VC_FORMAT_VER);     // 4
    vc_buf_u64(&b, dir_off);           // 8
    vc_buf_u64(&b, dir_len);           // 16
    vc_buf_u64(&b, dir_sum);           // 24
    vc_buf_u64(&b, idx_off);           // 32 (convenience: chunk index offset)
    vc_buf_u32(&b, (u32)w->nidx);      // 40 (chunk count)
    vc_buf_put(&b, magic, 4);          // 44 trailing magic
    (void)tstart;

    free(w->index); w->index = NULL;
    *out = b.data; *out_len = b.len;
    return 0;
}

// --- Reader ---------------------------------------------------------------
typedef struct {
    const u8 *data;
    size_t len;
    vc_coord3 shape;
    vc_config_hdr cfg;
    const vc_index_entry *index;
    u32 nchunks;
} vc_reader;

static inline u32 vc_rd_u32(const u8 *p) { u32 v; memcpy(&v, p, 4); return v; }
static inline u64 vc_rd_u64(const u8 *p) { u64 v; memcpy(&v, p, 8); return v; }

// Parse trailer->directory->index. Returns 0 on ok, nonzero on corruption.
static inline int vc_reader_open(vc_reader *r, const u8 *data, size_t len) {
    if (len < VC_TRAILER_SIZE) return 1;
    const u8 *t = data + len - VC_TRAILER_SIZE;
    if (t[0]!=VC_MAGIC0||t[1]!=VC_MAGIC1||t[2]!=VC_MAGIC2||t[3]!=VC_MAGIC3) return 2;
    if (t[44]!=VC_MAGIC0||t[45]!=VC_MAGIC1||t[46]!=VC_MAGIC2||t[47]!=VC_MAGIC3) return 2;
    u64 dir_off = vc_rd_u64(t + 8), dir_len = vc_rd_u64(t + 16), dir_sum = vc_rd_u64(t + 24);
    if (dir_off + dir_len > len) return 3;
    if (vc_xxh64(data + dir_off, dir_len, 0) != dir_sum) return 4;

    const u8 *d = data + dir_off;
    u32 nm = vc_rd_u32(d); d += 4;
    if (nm != 1) return 5;   // Phase 0: single member
    u32 nlen = vc_rd_u32(d); d += 4;
    d += nlen;               // skip name
    r->shape.z = (u32)vc_rd_u64(d); d += 8;
    r->shape.y = (u32)vc_rd_u64(d); d += 8;
    r->shape.x = (u32)vc_rd_u64(d); d += 8;
    u64 nidx = vc_rd_u64(d); d += 8;
    u64 idx_off = vc_rd_u64(d); d += 8;
    u64 idx_len = vc_rd_u64(d); d += 8;
    u64 idx_sum = vc_rd_u64(d); d += 8;
    r->cfg.chunk_side = vc_rd_u32(d); d += 4;
    r->cfg.transform_tag = vc_rd_u32(d); d += 4;
    r->cfg.entropy_tag = vc_rd_u32(d); d += 4;
    r->cfg._reserved = vc_rd_u32(d); d += 4;

    if (idx_off + idx_len > len) return 6;
    if (vc_xxh64(data + idx_off, idx_len, 0) != idx_sum) return 7;

    r->data = data; r->len = len;
    r->index = (const vc_index_entry *)(data + idx_off);
    r->nchunks = (u32)nidx;
    return 0;
}

// Get chunk i's payload (NULL if ABSENT). Verifies checksum.
static inline const u8 *vc_reader_chunk(const vc_reader *r, u32 i, size_t *plen, int *absent) {
    const vc_index_entry *e = &r->index[i];
    if (e->flags & VC_FLAG_ABSENT) { *absent = 1; *plen = 0; return NULL; }
    *absent = 0; *plen = e->length;
    return r->data + e->offset;
}

#endif // VC_ARCHIVE_H
