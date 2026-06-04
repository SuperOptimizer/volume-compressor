// volume-compressor — lossy 3D u8 codec with a sparse, appendable archive.
// Pure C23, CPU-only, single .h + .c drop-in unit. See docs/SPEC.md for the
// archive format, index, coverage model, and concurrency; plan.txt §2 for the
// per-atom codec pipeline.
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
    VC_ERR_IO = 4,
} vc_status;

typedef struct {
    uint32_t nx, ny, nz; // logical extent (voxels), x fastest, z slowest
} vc_dims;

typedef struct {
    uint32_t x0, y0, z0; // inclusive lower corner (voxels)
    uint32_t x1, y1, z1; // exclusive upper corner (voxels)
} vc_box;

#define VC_ATOM    32u            // atom edge (voxels)
#define VC_ATOM3   (32u*32u*32u)  // 32768
#define VC_NLOD    8              // max LOD count
#define VC_MAX_DIM (1u<<20)       // hard max voxels per axis (2^20 -> 2^15 atoms)

// Coverage of an atom in the cache: both ABSENT and KNOWN_ZERO decode to zeros,
// but ABSENT means "never fetched — must download to find out" whereas
// KNOWN_ZERO means "fetched, confirmed empty — do not re-fetch". See SPEC §5.
typedef enum { VC_ABSENT = 0, VC_KNOWN_ZERO = 1, VC_PRESENT = 2 } vc_cover;

// ===========================================================================
// Writer — builds/extends a sparse appendable archive on disk.
// LODs are independent volumes the caller supplies; they MUST form a strict 2x
// pyramid from lod0_dims (dim_{k+1} = ceil(dim_k/2)). The archive never
// downsamples. Thread-safe: many readers + writers may share one vc_writer.
// ===========================================================================
typedef struct vc_writer vc_writer;

// Create (or truncate) an archive at `path` for a volume whose LOD0 is
// lod0_dims, targeting `target_ratio` (10..100x). Returns NULL on failure.
// If base_q is left unset for a LOD, the writer auto-calibrates it from the
// first appended atom of that LOD (cheap, but a single-atom estimate is noisy).
vc_writer *vc_create(const char *path, vc_dims lod0_dims, float target_ratio);

// Set the fixed base_q for `lod`. There is NO auto-calibration: the caller must
// call this for every LOD before appending atoms to it. The quantizer step q
// directly controls quality/ratio (higher q = more compression, lower quality).
// q is clamped to [0.05, 4096]. The caller chooses q (and re-runs at a different
// q if the achieved ratio isn't what they want).
vc_status vc_set_base_q(vc_writer *w, int lod, float q);

// Append one 32^3 atom at atom coords (az,ay,ax) in the given lod.
vc_status vc_append_atom(vc_writer *w, int lod, uint32_t az, uint32_t ay, uint32_t ax,
                         const uint8_t vox[VC_ATOM3]);

// Append an arbitrary voxel box. Its origin and size MUST be multiples of 32
// (so it splits cleanly into 32^3 atoms). The box may be any size and at any
// 32-aligned offset, and may straddle internal index regions. `voxels` is the
// box contents in raster (z,y,x) order, x fastest, sized (z1-z0)*(y1-y0)*(x1-x0).
vc_status vc_append_box(vc_writer *w, int lod, vc_box voxel_box, const uint8_t *voxels);

// Mark a single atom / a whole index-region as known-zero (coverage becomes
// KNOWN_ZERO) without storing any payload. Region coords are in index-region
// units (atom coord >> log2(R); R from vc_region_atoms()).
vc_status vc_mark_zero_atom(vc_writer *w, int lod, uint32_t az, uint32_t ay, uint32_t ax);
vc_status vc_mark_zero_region(vc_writer *w, int lod, uint32_t rz, uint32_t ry, uint32_t rx);

// Read an already-written atom back from the writer (PRESENT/KNOWN_ZERO/ABSENT
// resolved as in the reader; ABSENT/ZERO decode to zeros). Thread-safe. Used to
// cascade the LOD pyramid: decode a lower LOD, downscale, append the next.
vc_status vc_writer_decode_atom(vc_writer *w, int lod, uint32_t az, uint32_t ay, uint32_t ax,
                                uint8_t out[VC_ATOM3]);
vc_cover  vc_writer_coverage(vc_writer *w, int lod, uint32_t az, uint32_t ay, uint32_t ax);

// Decode a 2x2x2 block of source atoms in one locked pass (cascade fast path).
// out8 holds 8 atoms in (dz,dy,dx) order; absent/zero are zeroed. *any set if
// any of the 8 were present.
vc_status vc_writer_decode_2x2x2(vc_writer *w, int lod, uint32_t az0, uint32_t ay0, uint32_t ax0,
                                 uint8_t out8[8][VC_ATOM3], int *any);

// Flush and close the writer (does not unlink the file).
void vc_writer_close(vc_writer *w);

// Index-region edge in atoms (R). Compile-time constant exposed for callers
// that want to address regions for vc_mark_zero_region.
uint32_t vc_region_atoms(void);

// ===========================================================================
// Reader — random-access decode from an mmap'd archive.
// ===========================================================================
typedef struct vc_archive vc_archive;

// Open an archive from an in-memory buffer (typically an mmap of the file).
// Borrows the buffer — it must outlive the handle.
vc_archive *vc_open(const uint8_t *archive, size_t len);

// Byte-source callback for streaming open: fill dst with exactly `len` bytes
// starting at `off` in the archive. Return VC_OK on success. The implementation
// typically range-fetches + caches (e.g. from S3) behind this. Called on demand
// for the header, L1 index probes, L2 slots, and atom payloads.
typedef vc_status (*vc_read_fn)(void *ud, uint64_t off, uint32_t len, uint8_t *dst);

// Open an archive WITHOUT holding it whole in memory: libvc walks its index and
// decodes by pulling byte ranges through `read` on demand. `total_len` is the
// true archive size. Decode results are identical to vc_open on the same bytes;
// only the byte source differs. The callback must outlive the handle.
vc_archive *vc_open_streaming(vc_read_fn read, void *ud, uint64_t total_len);

void        vc_close(vc_archive *a);

// LOD0 dims, and derived dims of any LOD (strict 2x pyramid).
vc_status vc_lod_dims(const vc_archive *a, int lod, vc_dims *out);

// Decode one 32^3 atom. ABSENT / KNOWN_ZERO atoms decode to all-zero.
vc_status vc_decode_atom(vc_archive *a, int lod, int ax, int ay, int az,
                         uint8_t out[VC_ATOM3]);

// Decode an arbitrary voxel sub-box of a LOD (gathers overlapping atoms).
vc_status vc_decode_region(vc_archive *a, int lod, vc_box box, uint8_t *out);

// Coverage of an atom without decoding it (ABSENT vs KNOWN_ZERO vs PRESENT).
vc_cover vc_atom_coverage(const vc_archive *a, int lod,
                          uint32_t az, uint32_t ay, uint32_t ax);

// Resolve an atom to its compressed payload byte range [*off,*off+*len) WITHOUT
// decoding (also returns coverage). Lets a streaming reader fetch exact bytes or a
// repacker copy payloads verbatim. *off/*len are 0 for ABSENT/KNOWN_ZERO.
vc_cover vc_atom_payload_range(const vc_archive *a, int lod,
                               uint32_t az, uint32_t ay, uint32_t ax,
                               uint64_t *off, uint32_t *len);

// Override the fail-fast panic hook (default aborts).
void vc_set_panic_hook(void (*hook)(const char *msg));

#ifdef __cplusplus
}
#endif
#endif // VC_H
