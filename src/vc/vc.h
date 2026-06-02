// volume-compressor — frozen single-pipeline lossy 3D u8 codec.
// Pure C23, CPU-only, single .h + .c drop-in unit. See plan.txt.
#ifndef VC_H
#define VC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VC_OK = 0,
    VC_ERR_FORMAT = 1,
    VC_ERR_RANGE = 2,
    VC_ERR_OOM = 3,
} vc_status;

typedef struct {
    uint32_t nx, ny, nz; // logical extent (voxels), x fastest, z slowest
} vc_dims;

typedef struct {
    uint32_t x0, y0, z0; // inclusive lower corner
    uint32_t x1, y1, z1; // exclusive upper corner
} vc_box;

#define VC_ATOM   16u            // atom edge (voxels)
#define VC_ATOM3  (16u*16u*16u)  // 4096
#define VC_NLOD   8              // fixed LOD count

typedef struct vc_archive vc_archive;

// Encode one u8 volume (all 8 LODs) to a single in-memory archive at a target
// compression ratio (10..100x). Caller frees *out_archive with free().
vc_status vc_encode(const uint8_t *vol, vc_dims dims, float target_ratio,
                    uint8_t **out_archive, size_t *out_len);

// Open an archive for random access. Borrows the buffer (must outlive handle).
vc_archive *vc_open(const uint8_t *archive, size_t len);
void        vc_close(vc_archive *a);

// Query a LOD member's logical dims.
vc_status vc_lod_dims(const vc_archive *a, int lod, vc_dims *out);

// Decode a whole LOD volume into out_vol (caller-sized: nx*ny*nz of that LOD).
vc_status vc_decode_lod(vc_archive *a, int lod, uint8_t *out_vol, vc_dims *out_dims);

// Decode ONE 16^3 atom (random access; touched==1). ax,ay,az are atom indices.
vc_status vc_decode_atom(vc_archive *a, int lod, int ax, int ay, int az,
                         uint8_t out[VC_ATOM3]);

// Decode an arbitrary sub-box of a LOD (gathers overlapping atoms, deblocks).
vc_status vc_decode_region(vc_archive *a, int lod, vc_box box, uint8_t *out);

// Override the fail-fast panic hook (default aborts).
void vc_set_panic_hook(void (*hook)(const char *msg));

#ifdef __cplusplus
}
#endif
#endif // VC_H
