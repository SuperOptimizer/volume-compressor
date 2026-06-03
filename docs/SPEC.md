# volume-compressor archive format & API (v1.0.1)

Authoritative spec for the on-disk archive, its index, the coverage model, and
the read/write API. The per-atom codec pipeline (DCT-32 → dead-zone quant →
context range coder) is unchanged from v1.0; see `plan.txt §2` for that. This
document covers everything *around* the atom: how atoms are addressed, stored,
indexed, appended, and served.

The archive is an on-disk cache that fills incrementally. A consumer downloads
source data, splits it into 32³ atoms, and appends them; readers serve atoms
back. The whole file is mmap'd and mutated in place; payloads only ever append
at EOF and are never moved or rewritten. There is exactly ONE index — no footer,
no second copy.

--------------------------------------------------------------------------------
## 1. Units

- **ATOM** — 32³ voxels (u8). The fundamental and only sub-archive unit:
  transform, quant, entropy, index, and random-access decode unit. Every atom is
  independently decodable; decoding one atom touches only that atom.
- **INDEX REGION** — a cube of R×R×R atoms; the level-2 dense-block grouping.
  R = 32 atoms = 1024 voxels per axis (stored in the header, tunable). INTERNAL
  to the index; callers never align to it.
- **LOD** — a resolution level, 0..7. Independent volumes supplied by the caller
  (the archive does NOT downsample). Strict 2× pyramid contract (§2).
- **ARCHIVE** — one file: header, a two-level sparse index, and an append log of
  L2 blocks + atom payloads.

Ordering everywhere: Z, Y, X (Z outermost), little-endian. Atom and voxel grids
are raster (z,y,x) — not Morton/Hilbert.

--------------------------------------------------------------------------------
## 2. LODs — strict 2× pyramid (caller contract)

LODs are independent volumes the caller supplies, but they MUST form a strict
2× pyramid: `dim_{k+1} = ceil(dim_k / 2)` per axis (e.g. 1000³→500³→250³→…).
The archive stores only LOD0 dims and derives every LOD's atom grid from them;
it never downsamples and never stores per-LOD dims. The downscaling METHOD (box,
gaussian, …) is the caller's choice. Up to 8 LODs; `lod >= 8` is rejected.

--------------------------------------------------------------------------------
## 3. Addressing

- Atom coord per axis = voxel coord >> 5.
- Hard max: 2²⁰ voxels/axis → 2¹⁵ atoms/axis. `vc_create`/append reject any
  dimension or coordinate beyond this.
- Packed atom key (u64, 48 bits used):
  `key = (lod:3)<<45 | (az:15)<<30 | (ay:15)<<15 | (ax:15)`
- Region coord = atom coord >> log2(R); local coord = atom coord & (R-1).
  Region key packs the same way on region coords.

--------------------------------------------------------------------------------
## 4. Two-level sparse index: hash map → dense block

Occupancy is sparse over the address space but dense and clustered where present
(a scroll is one connected blob inside a large zero-padded box). The index is
therefore a sparse map of regions, each pointing to a dense block of atom slots.

### Level 1 — region hash table (open-addressed, linear probe)
Keyed on region. One entry per touched region (few). Mutated in place. Grows by
rehashing into a fresh table at EOF when load exceeds ~70% (rare; L1 is small).

    L1 entry (16B): [u64 region_key][u64 block_ref]
      block_ref == 0            → region never inserted   (ABSENT: unknown)
      block_ref == ZERO_REGION  → known all-zero region   (no L2 block)
      else                      → byte offset of the dense L2 block

### Level 2 — dense block per populated region
`R³` fixed slots, direct-indexed by local coord — no hashing, no probing.
Allocated (zeroed) at EOF the first time any atom in the region is touched.

    slot index = (lz*R + ly)*R + lx
    slot (16B): [u64 offset][u32 len][u8 flags][u8 uval][u8 dc][u8 pad]
      flags == AF_ABSENT (0)  → unwritten           (ABSENT: unknown)
      flags &  AF_ZERO        → known-zero atom, no payload
      flags &  AF_UNIFORM     → uniform `uval`, no payload
      else                    → real payload at `offset`, `len` bytes

`offset` is the atomic commit word (§7).

--------------------------------------------------------------------------------
## 5. Coverage: known-zero vs not-yet-downloaded

The cache must distinguish "downloaded, confirmed zero" from "never fetched."
BOTH decode to zeros, but they drive opposite actions:

    PRESENT      → has content        → serve from archive
    KNOWN_ZERO   → confirmed empty    → serve zeros, DO NOT re-fetch
    ABSENT       → never fetched      → MUST download from source to find out

Conflating KNOWN_ZERO and ABSENT breaks the cache: it would either re-download
confirmed-empty space forever, or never fetch genuinely-missing data.

Resolved top-down across both levels (partial coverage within a region is a
first-class state — a block can exist while some of its slots stay ABSENT):

    1. L1 miss / block_ref == 0       → ABSENT       (region unknown)
    2. block_ref == ZERO_REGION       → KNOWN_ZERO   (all atoms in region known 0)
    3. block present, slot AF_ABSENT  → ABSENT       (atom unknown)
    4. block present, slot AF_ZERO    → KNOWN_ZERO   (atom known 0)
    5. block present, real/uniform    → PRESENT

Exposed without decoding via `vc_atom_coverage` (§9).

--------------------------------------------------------------------------------
## 6. On-disk layout

    [file header]   magic, version, atom=32, layout=SPARSE2, R, lod0 dims
    [index header]  base_q[8], L1 capacity, L1 count, L1 table offset, write cursor
    [L1 hash table] capacity × 16B                  ← mutated in place
    [L2 blocks + atom payloads, interleaved]        ← append at EOF, never rewritten
    [trailer]       points at the index header; magic

All multi-byte fields little-endian. The header validates magic/version/atom/
layout/R on open and rejects a mismatch (the codec misdecodes silently across a
different atom size, so this guard is mandatory).

--------------------------------------------------------------------------------
## 7. Concurrency (multi-thread, single-process)

In-process only; multi-process is NOT supported. All access goes through the
API — consumers poking the mmap directly is unsupported and discouraged. One
RW-lock per open archive plus atomic publish:

- **READ** — shared lock; readers run concurrently.
- **APPEND into an existing block** — shared lock; reserve an EOF payload range
  by atomic `fetch_add` on the write cursor (writers fill disjoint ranges in
  parallel), write all slot fields, then RELEASE-store the slot's 8-byte
  `offset` LAST as the commit word. Readers ACQUIRE-load `offset` and see either
  ABSENT or a fully-written slot — never a torn one.
- **STRUCTURAL ops** (allocate an L2 block, rehash L1, grow/remap the file) —
  EXCLUSIVE lock. Rare relative to atom appends. File growth + remap is hidden
  behind this lock inside the API.

Single API surface (no separate single-threaded variant — the shared-lock read
path is effectively free).

--------------------------------------------------------------------------------
## 8. Lifecycle: write-once, append-only

Source data is immutable → atoms are WRITE-ONCE: a slot only transitions
ABSENT → present (or ABSENT → KNOWN_ZERO). Never overwritten, never downgraded.
Region blocks live forever at fixed R³ size. NO free-list, NO generation
counter, NO compaction. The archive is unbounded; to reset, delete the file and
rebuild. (Eviction is possible future work, explicitly out of scope here — the
format reserves nothing for it.)

--------------------------------------------------------------------------------
## 9. Rate control / base_q

One `base_q` per LOD, stored in `base_q[8]` in the index header. It must be set
before any atom of the LOD is stored, since the reader decodes every atom of a
LOD with that single base_q. Two ways:

  - Explicit (recommended for export): `vc_set_base_q(w, lod, q)` before the
    first append. The producer calibrates offline — sample many present atoms,
    bracket the target ratio on a coarse q ladder, then log-interpolate
    (~3 measurements; log(ratio) ≈ linear in log(q)). Seeding each LOD's bracket
    from the previous LOD's q converges later LODs in 1–2 tries.
  - Auto: if left unset, the writer freezes q from the FIRST non-zero atom of the
    LOD. Cheap but a single-atom estimate is noisy — prefer explicit for quality.

CRITICAL — the ratio is measured over PRESENT (non-zero) atoms ONLY. Zero/mask
atoms (usually the majority — a scroll is a blob in a large zero-padded box)
never enter calibration, so the mask does NOT inflate the achieved ratio. "10x"
means 10x on real material. (A "vs full raw" number IS mask-inflated and is
informational only.)

q floor is 0.05, not 0.5. MEASURED on 45µm scroll data: present-atom ratio is
~5x at q=0.5 and ~10x at q≈1.0, with valid output down to q≈0.05 (~1.6x). A 0.5
floor wrongly clamped LOD0 to 5x and collapsed adjacent LODs to identical q.

Per-LOD target ratio: LOD0 dominates the byte budget by far (in a real scroll,
level 0 ≈ 83% of all bytes; each coarser level is ≥4× smaller). So the overall
archive ratio is governed almost entirely by LOD0, and compressing coarse LODs
hard saves a negligible number of bytes while visibly hurting the zoomed-out
views users see first. The writer therefore targets a LOWER ratio (higher
quality) at coarser levels — halving the target ratio per level, floored at 1.0
(near-lossless). The headline ratio is effectively LOD0's; coarse levels get
their quality back for free.

--------------------------------------------------------------------------------
## 10. API

### Writer

    vc_writer *vc_create(const char *path, vc_dims lod0_dims, float target_ratio);

    // One 32³ atom at atom coords.
    vc_status vc_append_atom(vc_writer*, int lod, uint32_t az, uint32_t ay, uint32_t ax,
                             const uint8_t vox[VC_ATOM3]);

    // Arbitrary caller "append box": origin and size MUST be multiples of 32
    // voxels (splits cleanly into 32³ atoms). Any size (e.g. a downloaded 128³ or
    // 256³ chunk) at any 32-aligned offset; MAY straddle index-region boundaries
    // — atoms are scattered into whatever regions they fall in (find-or-create per
    // region). Callers are fully decoupled from the 1024³ region granularity.
    vc_status vc_append_box(vc_writer*, int lod, vc_box voxel_box, const uint8_t *voxels);

    // Mark known-zero without storing payload (coverage = KNOWN_ZERO).
    vc_status vc_mark_zero_atom(vc_writer*, int lod, uint32_t az, uint32_t ay, uint32_t ax);
    vc_status vc_mark_zero_region(vc_writer*, int lod, uint32_t rz, uint32_t ry, uint32_t rx);

    void vc_writer_close(vc_writer*);

### Reader

    vc_archive *vc_open(const uint8_t *buf, size_t len);   // borrows the mmap
    void        vc_close(vc_archive*);

    vc_status vc_lod_dims(const vc_archive*, int lod, vc_dims *out);

    // Decode one 32³ atom. ABSENT/KNOWN_ZERO → zeros.
    vc_status vc_decode_atom(vc_archive*, int lod, int ax, int ay, int az,
                             uint8_t out[VC_ATOM3]);

    typedef enum { VC_ABSENT, VC_KNOWN_ZERO, VC_PRESENT } vc_cover;
    vc_cover  vc_atom_coverage(const vc_archive*, int lod,
                               uint32_t az, uint32_t ay, uint32_t ax);

A reader opened on a still-growing archive sees atoms published so far; the
acquire/release discipline (§7) guarantees consistency.

--------------------------------------------------------------------------------
## 11. Out of scope (do not add)

Internal downsampling/pyramiding (caller supplies LODs — `downsample2x` is kept
TEST-ONLY); the old build-once whole-volume `vc_encode` (removed); chunks (one
grouping level only — the index region — and it is internal); multi-process
access; atom overwrite; eviction/compaction; cross-endian archives; lossless
mode. See also `docs/DO_NOT_IMPLEMENT.md`.
