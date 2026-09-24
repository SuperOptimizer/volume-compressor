# volcomp stream format, version 1 (revision 4)

Normative. A conforming decoder accepts exactly the streams described here and
rejects everything else with `VOLCOMP_ERR_CORRUPT` (or `VOLCOMP_ERR_VERSION`
for an unknown version byte). All multi-byte integers are little-endian unless
stated otherwise.

## 1. Terms and geometry

- **voxel**: one `uint8` sample.
- **block**: 16×16×16 voxels, the transform and random-access unit. Block
  voxel index `v = z*256 + y*16 + x`.
- **chunk**: 128×128×128 voxels = 8×8×8 blocks, the independently decodable
  unit (one zarr chunk). Chunk voxel index `(z*128 + y)*128 + x`. Block index
  `bi = bz*64 + by*8 + bx`.
- **substream**: 16 consecutive blocks in block-index order (`bi/16`), each with
  its own rANS flush and bypass bytes. There are exactly 32 substreams.
- **shard**: 1024³ voxels = 8×8×8 chunks stored as one zarr v3
  `sharding_indexed` object (§7).

## 2. Chunk stream layout

```
offset  size   field
0       4      magic  "VOLC"
4       1      version = 1
5       1      mode   0 = lossy DCT (this section), 1..3 = lossless (§10),
                      4 = binary mask (§11), 5 = lossless binary mask (§12),
                      6 = surface chunk (§13)
6       2      q_raw  u16, quantiser step q = q_raw / 256, 256 <= q_raw <= 65280
8       T      frequency tables (§4.3), 10 models
8+T     256    directory: 32 × { u32 tok_n, u32 bypass_n }
8+T+256 P      payload: for s = 0..31: tok_n[s] bytes of tANS token bits, then bypass_n[s] bypass bytes
```

Exact accounting is normative: `8 + T + 256 + Σ(tok_n[s] + bypass_n[s])` must
equal the stream length. Per substream: `tok_n >= 3`, `tok_n <= 262152`,
`bypass_n <= 196616`.

Byte 5 was `reserved = 0` in volcomp 1.0 and every 1.0 decoder rejects a
nonzero value, so the lossless modes of §10 cannot be misread by an old
decoder; a mode-0 stream is byte for byte what volcomp 1.0 produced.

**Revisions.** The version byte stays 1 across revisions of this document,
because every stream a revision defines is a stream every earlier decoder
already rejects: the mode byte is the compatibility mechanism. Revision 1
(volcomp 1.0.x) defines modes 0..3 and rejects 4..255 with
`VOLCOMP_ERR_CORRUPT`; revision 2 (volcomp 1.1.0) adds mode 4, the binary mask
chunk of §11, and changes nothing else; revision 3 (volcomp 1.2.0) adds mode 5,
the spatially lossless binary mask chunk of §12, and changes nothing else;
revision 4 (volcomp 1.3.0, `VOLCOMP_FORMAT_REVISION`) adds mode 6, the surface
chunk of §13, and changes nothing else. A writer must not emit a mode-4 chunk to a
consumer that only reads revision 1, a mode-5 chunk to one that only reads
revision 2, nor a mode-6 chunk to one that only reads revision 3; that consumer
will refuse it cleanly rather than misread it.

## 3. Encoding pipeline (informative summary; §4–§6 are normative)

For each block: gather voxels as `float v - 128`; 3-D orthonormal DCT-II
(separable 16-point, even/odd factorisation, tables in `volcomp.h`); dead-zone
quantisation with the radial step law; tokenise in the fixed 16³ zigzag scan
order (`VF_SCAN16`); entropy-code tokens with a 2-lane tANS over 10 context
models whose frequencies are transmitted per chunk; raw bits go to a
byte-aligned LSB-first bypass stream.

## 4. Entropy coding

### 4.1 Tokens (HybridUint)

A non-negative integer `u` is coded as a token `t` (alphabet 0..31, 31 = EOB)
plus bypass bits: `u < 4 → t = u`, no bits; otherwise `k = floor(log2 u)`,
`t = 2 + k`, and the low `k` bits of `u` are written to the bypass stream
LSB-first. Token class caps are normative (they follow from `q >= 1`):
DC token ≤ 20, run token ≤ 13, level token ≤ 14. A decoder must check the cap
before reading bypass bits.

### 4.2 Block token sequence

For each block, in order:

1. **DC**: `zigzag(dc - prev_dc)` with model 9, where `prev_dc` is the DC level
   of the previous block in the same substream (0 at the start of each
   substream). `|dc| <= 65536` is normative; accumulate in 64-bit.
2. For each nonzero AC coefficient in scan order, at scan position `p`
   (1..4095), with `run` = number of zeros since the previous coded position
   (`prev + 1 + run = p`, `prev` starting at 0):
   - `run` with model `band(p - run)`;
   - `|level| - 1` with model `3 + 2*band(p) + (run == 0)`; `|level| <= 5220`;
   - one bypass bit: sign (1 = negative).
3. **EOB** token 31 with model `band(last + 1)`, `last` = scan position of the
   final nonzero (or `band(1)` if none).

`band(p) = 0` for `p < 128`, `1` for `p < 1024`, else `2`.

### 4.3 tANS and frequency tables

Tokens are coded with a table-based ANS (tANS / FSE, Collet) with
`TABLE_LOG = 10` (1024 states) per context model, two interleaved lanes:
token `i` of a substream belongs to lane `i & 1`. Each model's table is built
from its transmitted frequencies exactly as zstd's `FSE_buildDTable`: symbols
are spread over the 1024 slots with step `(1024>>1) + (1024>>3) + 3` in
symbol order, then slot `u` gets `nbBits = 10 - floor(log2(next[s]))` and
`newState = (next[s] << nbBits) - 1024` where `next[s]` counts from `freq[s]`
upward. Decoding a token: `sym = table[state].sym`, then
`state = table[state].newState + read(nbBits)`.

The token bit stream of a substream is one LSB-first bit sequence containing,
in decode order: the initial state of lane 0 (10 bits), the initial state of
lane 1 (10 bits), then for each token in order the `nbBits` bits of its
transition. It is flushed to a byte boundary with zero padding. A decoder must
verify that exactly `tok_n` bytes are consumed, that the padding bits are
zero, and that both lanes end at state 0 (the encoder starts both lanes at
state 0 and encodes backwards; see `vf_tans_encode2` in `volcomp.h`).

Tables: for each model `m = 0..9`: a `u32` presence bitmap (bit `t` set iff
`freq[t] > 0`), then for each set bit in ascending order the frequency as a
canonical LEB128 (1 ≤ freq ≤ 1024). Each model's frequencies must sum to
exactly 1024; the decoder must not renormalise. Non-canonical LEB128 (a
trailing zero continuation byte, > 5 bytes, or > 32 bits) is rejected.

### 4.4 Bypass stream

LSB-first bit packing; each substream's bypass bytes are flushed to a byte
boundary with zero padding. A decoder must verify that exactly `bypass_n`
bytes are consumed and that any remaining padding bits are zero.

## 5. Quantiser and reconstruction

Coefficient natural index `c = z*256 + y*16 + x`, radius `r = x + y + z`.

```
step(0)      = q * 0.125                        (DC)
step(r >= 1) = q * (1 + r)^0.65                 (AC)
encoder AC:  level = sign(X) * floor(|X| / step + 0.2)
encoder DC:  level = sign(X) * floor(|X| / step + 0.5)
decoder DC:  X' = level * step
decoder AC:  X' = sign(level) * (|level| + d) * step,  d = 0.15 if |level| == 1 else 0.30
             (0 if level == 0)
```

Reconstruction order per block, normative: dequantise → inverse DCT (a block
with no nonzero AC coefficient reconstructs to the constant `dc * 0.015625`
without a transform) → add 128, round half up, clamp to [0, 255]. (The
reference implementation runs unnormalised DCT kernels and folds the
orthonormal scale — 1/4 per zero index, √2/4 otherwise, per axis — into the
quantiser tables; the result is the same transform.)

Arithmetic is IEEE single precision; two conforming decoders built with
different compilers or vector widths (the reference has AVX2 and plain C
kernel sets, selected at runtime) may differ by at most ±1 in any voxel.
The same build must be deterministic.

## 6. No post-filter

Version 1 has no in-format deblocking or seam filter: the reconstruction of a
block is exactly dequantisation → inverse DCT → rounding. (`volcomp_deblock`
is an optional post-process a caller may run over an assembled decoded
volume; it is not part of the format.) Consequently every block
and every chunk is reconstructed independently, all 16-planes (including
chunk faces) are statistically identical, and `volcomp_decode_block` is
bit-identical to the corresponding region of `volcomp_decode`. Users who
need smoother output choose a smaller `q`. (The measured evaluation of
block-local filter candidates is recorded in `docs/measured.md`; none
improved on no filter.)

## 7. Shard layout (zarr v3 `sharding_indexed`)

A shard file holds up to 512 chunk streams back to back, in block-major chunk
order `ci = cz*64 + cy*8 + cx`, followed by an index of 512 entries
`{ u64 offset, u64 nbytes }` (offsets relative to the shard start) and a
`u32` CRC-32C of the 8192 index bytes. A missing chunk — including any chunk
whose source voxels are all zero — has `offset = nbytes = 0xFFFFFFFFFFFFFFFF`
and decodes as all zeros (`fill_value = 0`). This is byte-compatible with
zarr v3 `sharding_indexed` with `index_location = "end"`,
`index_codecs = [bytes(little), crc32c]`, chunk shape 128³, shard shape 1024³,
inner codec `"volcomp"` with configuration `{"q": q}`. Edge chunks are zero
padded to 128³ before encoding.

## 8. Bounds

Tokens per block ≤ 8192; bypass bits per block ≤ 18 + 4095·(11 + 12 + 1)
⇒ ≤ 12288 bytes; token-stream bytes per substream ≤ 2·tokens + 8 (tANS emits
at most 10 bits per token; the bound is kept loose). Therefore the
encoded size of any chunk is at most
`8 + 680 + 256 + 32·(2·16·8192 + 8) + 512·12288 = 14 681 264` bytes
(`VOLCOMP_ENCODE_BOUND`).

## 9. Label chunks (`volcomp_label.h`)

A label chunk stores up to 255 class PLANES (u8 probability maps) for one
128³ region. Only planes with a nonzero voxel are stored; a class that is not
stored decodes as all zeros. Class identity is an explicit tag, never coded
lossily.

```
offset  size   field
0       4      magic "VOLL"
4       1      version = 2
5       1      nplanes (0..255)
6       2      q_raw  u16, the chunk's DEFAULT q; 0 (lossless) or 256..65280
8       4      reserved = 0
12      8*n    directory: { u8 cls, u8 mode, u16 q_raw, u32 n } per plane,
               cls strictly ascending; the entry's q_raw is the q that plane
               was encoded with, 0 (lossless) or 256..65280
        var    plane bytes in directory order, n bytes each
```

Exact accounting is normative: header, directory and every plane's bytes
must sum to the stream length. Mode 0 (image): the plane is a §2 or §10
stream whose q equals the directory entry's `q_raw`, `n < 2 097 152`. Mode 1 (raw): the plane is
stored verbatim, `n = 2 097 152` (the encoder uses it only when the §2 stream
would not be smaller). Any other mode is rejected.

**Bound.** `VOLCOMP_LABEL_ENCODE_BOUND(n) = 12 + n·(8 + 2 097 152)`.

Version 1 (one `q_raw` for the whole chunk, `reserved = 0` where the entry's
`q_raw` now sits) is not read; `volcomp_label_decode` returns
`VOLCOMP_ERR_VERSION` for it.

## 10. Lossless chunks (`q = 0`)

`q = VOLCOMP_Q_LOSSLESS` (0) selects an exact codec. Header byte 5 says which
of three forms the chunk takes; all three set `q_raw = 0`.

```
mode 1  CODED   tables + directory + 32 substreams (below)
mode 2  RAW     8-byte header + 2 097 152 voxels, z-major
mode 3  CONST   8-byte header + 1 byte, the value of every voxel
```

Mode 1 keeps §1's geometry: 512 blocks of 16³ in z-major block order, 32
substreams of 16 consecutive blocks, so a single block decodes from one
substream and is byte-identical to the same region of the full decode.

```
offset  size   field
0       8      header, mode = 1, q_raw = 0
8       T      frequency tables (§4.3), 8 models
8+T     D      directory: 32 × { LEB128 tok_n, LEB128 bypass_n }
8+T+D   P      payload: for s = 0..31: tok_n[s] token bytes then bypass_n[s] bypass bytes
```

Exact accounting is normative, as in §2. Per substream `tok_n >= 3`,
`tok_n <= 131 112`, `bypass_n <= 73 736`.

**Models.** 0 block mode, 1..2 SPARSE runs and EOB, 3..4 SPARSE levels, 5..7
DENSE levels. Frequency tables are encoded exactly as in §4.3.

**Tokens.** Values use a HybridUint with 16 literals rather than §4.1's 4:
`u < 16` is token `u` with no bypass bits; otherwise `tok = 12 + msb(u)` and
the low `msb(u)` bits go to the bypass stream. A level token is at most 19
(`u <= 255`), a run token at most 23 (`u <= 4095`); `VF_TOK_EOB = 31`.

**Prediction.** For voxel `i = (z·16 + y)·16 + x` of a block, `pred` is the
voxel at `z-1` (same y, x), or at `y-1` when `z = 0`, or at `x-1` when
`z = y = 0`, or 128 at `i = 0`. The residual is `(v - pred) mod 256`
zigzagged into 0..255. Prediction never crosses a block boundary.

**Block sequence.** Each block starts with a mode token from model 0:

- `0 CONST` — 4096 equal voxels; 8 bypass bits hold the value. No further
  tokens; an all-flat chunk is mode 3 and costs 9 bytes in total.
- `1 SPARSE` — `(run, level)*` then `EOB`. `run` is the number of zero
  residuals skipped, coded with model `1 + (prev_run != 0)` (`prev_run = 1`
  at the start of the block); `level` is `u - 1` for the nonzero residual `u`
  that follows (`level <= 254`), coded with model `3 + (run != 0)`. `EOB` uses
  the run model. `pos + run <= 4095` is normative.
- `2 DENSE` — exactly 4096 level tokens, one residual each (`u <= 255`). The
  model is `5 + c` where `c = 0` if the residual one z-plane back is 0, `1` if
  it is 1 or 2, else `2` (`c = 0` in the first z-plane). The context looks a
  whole plane back so that it is never on the entropy decoder's critical path.
- `3 RAW` — 4096 bypass bytes, the block verbatim.

**Bound.** The encoder emits mode 2 whenever the coded form would reach
2 097 160 bytes, so a lossless stream is never larger than the raw chunk plus
the 8-byte header. `VOLCOMP_ENCODE_BOUND` is unchanged.

## 11. Mask chunks (mode 4)

A mask chunk is the storage form of a binary volume. It stores the 2×2×2
**majority** pool of a 128³ mask — a 64³ grid of bits — exactly, and decodes to
that grid **trilinearly interpolated** back to 128³, so the decoded chunk is a
continuous `u8` field with a two-voxel ramp around the boundary rather than a
0/255 mask. There is nothing to configure: no factor, no rule, no quantiser.
`q_raw` is 0, as for the lossless modes.

```
offset  size   field
0       8      header, mode = 4, q_raw = 0
8       1      flags: bits 0..1 = form, bits 2..7 = 0 (reserved, must be zero)
9       P      payload, per form
```

```
form 0  CODED  the range-coded grid (below); P >= 5
form 1  CONST  P = 1, one byte: 0 or 1, the value of every grid cell
form 2  RAW    P = 32 768, the grid packed LSB-first, bit i = cell i
```

Exact accounting is normative: `9 + P` must equal the stream length, and any
other form value is rejected. The encoder emits CONST for a uniform grid and
RAW whenever the coded form would not be smaller, so a mask chunk is never
larger than **32 777 bytes**.

### 11.1 The stored grid

Grid cell `(z, y, x)`, each in 0..63, has index `i = (z·64 + y)·64 + x`, and is
the majority of the eight source voxels `(2z+dz, 2y+dy, 2x+dx)`:

```
count = number of the 8 source voxels that are nonzero
cell  = count * 2 >= 8
```

### 11.2 Decoding (interpolation)

Stored cell `i` sits at full-resolution voxel `2i`, so output voxel `j` along
each axis samples the grid at `j/2`:

```
i0 = j >> 1
i1 = (j odd and i0 + 1 < 64) ? i0 + 1 : i0        (clamped at the top edge)
weights: (1, 0) for j even, (1/2, 1/2) for j odd
```

The output voxel is the trilinear combination of the eight grid cells
`(i0|i1, i0|i1, i0|i1)` with those per-axis weights. Every weight product is a
multiple of 1/8, so the value is `k/8` for an integer `k` in 0..8 and the
stored byte is normatively

```
k       0   1   2   3   4    5    6    7    8
value   0   32  64  96  128  159  191  223  255      (= round(255 k / 8))
```

A block centre — every voxel with all three coordinates even — is therefore
exactly 0 or 255, and equals its grid cell. The arithmetic is integer, so mask
chunks are byte-identical on every build.

### 11.3 The coded form

The grid's 262 144 bits are coded in index order with a binary range coder in
the LZMA arrangement: 32-bit `range` and `low`, renormalisation while
`range < 2^24`, 12-bit probabilities. Each bit `b` in context `c` with
probability `p[c]` (the probability of a **zero** bit, initialised to 2048):

```
bound = (range >> 12) * p[c]
b = 0:  range  = bound;          p[c] += (4096 - p[c]) >> s
b = 1:  low   += bound;
        range -= bound;          p[c] -= p[c] >> s
```

The adaptation shift `s` depends on how many bits that context has already
coded: `s = 1, 1, 2, 2` for the first four and 3 from then on (a context here
sees ~64 bits, so a fixed rate never converges).

The context is the 12 causal neighbours of the cell, bit `j` of the context
being neighbour `j` of this list, a neighbour outside the grid counting as 0:

```
j     0        1        2        3        4        5
d     (0,0,-1) (0,-1,0) (-1,0,0) (0,-1,-1)(-1,0,-1)(-1,-1,0)
j     6        7        8        9        10       11
d     (0,0,-2) (0,-2,0) (-2,0,0) (0,-1,1) (-1,0,1) (-1,1,0)      [(dz,dy,dx)]
```

— 4096 contexts, each with its own probability and count.

The encoder starts with `low = 0`, `range = 0xFFFFFFFF` and a one-byte cache of
0 (so the payload's first byte is always 0, and a decoder must check it), and
finishes with five shift-low steps. The payload is then exactly as long as the
decoder's reads: five bytes to prime it (`code` is the four after the leading
zero) and one per renormalisation. A decoder must therefore consume the payload
exactly — a read past its end, or a byte left over, is `VOLCOMP_ERR_CORRUPT`.

### 11.4 Notes for readers

`volcomp_decode` and `volcomp_decode_block` return the interpolated 128³ chunk,
so a mask chunk needs no special handling anywhere. `volcomp_is_lossless` is
false for one (the 2× pool is not the source); `volcomp_stream_q` reports 0,
since there is no quantiser. `volcomp_mask_info` identifies a mask chunk of
either mode and reports the edge of the grid it stores — 64 here, 128 for §12 —
and `volcomp_mask_decode_stored` returns that grid itself, which is what the
next coarser rung of a pyramid is made of.

Measured on a 256³ region of the PHercParis4 recto surface prediction (2.4 µm,
19.6 % foreground): **0.0057 bits per full-resolution voxel** (1 500 bytes per
128³ chunk), against 0.062 for packbits + zstd-19 of the same mask and 0.022 for
an ideal 12-neighbour model of the full-resolution mask. Thresholding the
decoded field at 128 changes 1.7 % of the voxels, each of them a mean of 1.06
and at most 2 voxels from the published boundary.


## 12. Lossless mask chunks (mode 5)

A lossless mask chunk stores a binary volume **spatially exactly**: the whole
128³ mask, every voxel of it, with no downscale and no interpolation.
Everything else is §11 — the same header, the same three forms, the same range
coder, the same 12-neighbour context model and the same adaptation schedule —
applied to a 128³ grid of bits instead of a 64³ one. There is nothing to
configure. `q_raw` is 0.

```
offset  size   field
0       8      header, mode = 5, q_raw = 0
8       1      flags: bits 0..1 = form, bits 2..7 = 0 (reserved, must be zero)
9       P      payload, per form
```

```
form 0  CODED  the range-coded grid (§12.2); P >= 5
form 1  CONST  P = 1, one byte: 0 or 1, the value of every cell
form 2  RAW    P = 262 144, the grid packed LSB-first, bit i = cell i
```

Exact accounting is normative: `9 + P` must equal the stream length, and any
other form value is rejected. The encoder emits CONST for a uniform mask and
RAW whenever the coded form would not be smaller, so a lossless mask chunk is
never larger than **262 153 bytes** (`VOLCOMP_MASK_LL_BOUND`) — the packed mask
plus the nine-byte header.

### 12.1 The stored grid

Grid cell `(z, y, x)`, each in 0..127, has index `i = (z·128 + y)·128 + x` and
is `1` iff source voxel `i` is nonzero. `volcomp_decode` returns cell `i` as
`255 · cell`, so the decoded chunk is 0/255 exactly as stored, and
`volcomp_decode_block` returns the corresponding region of it.

### 12.2 The coded form

The grid's 2 097 152 bits are coded in index order exactly as §11.3 codes its
262 144: the same binary range coder (32-bit `range` and `low`, renormalisation
while `range < 2^24`, 12-bit probabilities initialised to 2048), the same
adaptation shifts `1, 1, 2, 2` then 3 by context age, the same 12 causal
neighbours (a neighbour outside the **128³** grid counting as 0) forming the
same 4096 contexts, the same leading zero byte and the same five closing
shift-low steps. A decoder must consume the payload exactly; a read past its
end, or a byte left over, is `VOLCOMP_ERR_CORRUPT`.

### 12.3 Notes for readers

`volcomp_decode` and `volcomp_decode_block` return the 0/255 mask, so a lossless
mask chunk needs no special handling anywhere. `volcomp_is_lossless` is **false**
for one: the chunk decodes to its source's *support* exactly, but a source voxel
of 7 comes back as 255, so the stream is not a byte-exact copy of an arbitrary
u8 chunk. `volcomp_stream_q` reports 0. `volcomp_mask_info` reports a stored
grid edge of 128, which is how a reader tells mode 5 from mode 4, and
`volcomp_mask_decode_stored` returns that 128³ grid as 0/255.

Measured on the same 256³ region of the PHercParis4 recto surface prediction
(2.4 µm, 19.6 % foreground): **0.0255 bits per voxel** (6 682 bytes per 128³
chunk), against 0.0057 for mode 4's 2× pool of the same data, 0.062 for
packbits + zstd-19, and 0.022 for an *ideal* (two-pass, non-adaptive)
12-neighbour model of the same mask — the adaptive coder is within 16 % of what
this context set can do. Mode 5 is 4.5× the size of mode 4 and codes 8× as many
bits, so it runs at roughly a fifth of mode 4's throughput: **250–290 M voxels/s
encode and 230–270 M decode** per core on a Core Ultra 9 275HX, against
1 350–1 540 M / 760–1 140 M for mode 4 on the same box.

## 13. Surface chunks (mode 6)

A surface chunk stores a smooth probability field — a surface prediction,
`u8 = round(255 p)` — whose consumers threshold it. It is a DCT stream (the
**base**) plus a **refinement** that makes the threshold exact: for every voxel,

```
(source >= thr)  ==  (decoded >= thr)
```

so the thresholded mask, its skeleton and its distance field are exactly those
of the source, while the soft values carry the base's quantisation error. `thr`
is stored in the chunk (128, i.e. p >= 0.5, is the encoder's default).

```
offset  size    field
0       8       header: magic, version 1, mode = 6, q_raw = the base's step, 512 <= q_raw <= 65280
8       1       thr     1..255
9       1       law     the base's step law; 1 (flat, §13.1) is the only value defined
10      2       M       u16 refinement margin: 0, or odd and <= 509
12      4       base_n  u32 length of the base
16      base_n  base    frequency tables, directory and payload of a DCT stream
                        (the layout §2 gives from offset 8 on)
16+base_n  R    refinement payload (§13.3); R = 0 iff M = 0, else R >= 5
```

Exact accounting is normative: `16 + base_n + R` is the stream length. Any other
`law`, a `thr` of 0, an even or oversized `M`, a refinement payload with `M = 0`
or a missing one with `M > 0` is `VOLCOMP_ERR_CORRUPT`.

### 13.1 The base

The base is parsed as §2 parses the bytes after a mode-0 header, with the
exact-accounting rule applied to `base_n`, and decoded as §4 and §5 decode a
mode-0 stream at `q = q_raw / 256`, with two differences:

- **Step law (flat).** `step(0) = q · 0.125` as in §5, and `step(r) = q` for every
  `r >= 1` — no `(1 + r)^0.65` growth. A smooth probability field has no
  perceptual reason to quantise its high frequencies harder; at equal size the
  flat law gives a lower error near the threshold (measured,
  `docs/surface_mode.md`). With a flat AC step the largest level of a full-swing
  block is `8192 / q`, which the §4.2 cap `|level| <= 5220` bounds to `q >= 2`,
  hence `q_raw >= 512`.
- **Bit-exact reconstruction (§13.2).**

### 13.2 Bit-exact reconstruction

The refinement's contexts read the base's reconstruction, so every conforming
decoder must produce it to the last bit (the ±1 tolerance of §5 does not apply).
Reconstruction is §5 evaluated in IEEE-754 binary32 with round-to-nearest-even,
**no fused multiply-add and no reassociation**:

- dequantisation `X' = ((|level| + d) · step) · s` for AC and
  `X' = (|level| · step) · 0.015625` for DC, where `s` is the orthonormal scale
  by the number of zero frequency indices (3: 0.015625, 2: 0.0220970869,
  1: 0.03125, 0: 0.0441941738, as binary32 literals) and `step` is the binary32
  product `q · 1.0f` (AC) or `q · 0.125f` (DC);
- the inverse transform is the reference 16-point factorisation
  (`VF_DCT16_INV_BODY` in `volcomp.h`; its operation sequence and its decimal
  constants, read as binary32, are normative for this mode), applied along x (to
  the planes that hold a nonzero coefficient, and to their upper eight x lines
  only if one of those does), then y (the same planes), then z (every line);
  lines not transformed are zero;
- a block with no nonzero AC coefficient is the constant of §5, and the rounding
  (`+ 128.5`, clamp to [0, 255], truncation) is that of §5.

`volcomp.h` implements this with plain C and AVX2 kernels that run the same scalar
operation sequence (`vf_dct16_inv_strict_c` / `_avx2`); `tests/test_surfchunk.c`
checks that both decode the golden stream `tests/golden/surf_q16.volc` to the
same hash, and it has been checked under clang and GCC.

### 13.3 Refinement

Let `v` be the base's reconstruction, `T2 = 2·thr − 1` and, for a voxel,
`dd = |2v − T2|` (odd; the voxel's distance from the threshold in half steps).
The encoder sets `M` to the largest `dd` of a voxel on the wrong side
(`(source >= thr) != (v >= thr)`), or 0 if there is none. With `M > 0` the decoder
visits the voxels in index order and, for each voxel with `dd <= M` (a
**candidate**), decodes one bit `f` — 1 iff the voxel is on the wrong side — and,
if `f = 1`, sets it to `thr − 1` if `v >= thr`, else to `thr`. Non-candidates are
not coded and keep their base value.

The bit is coded with §11.3's binary range coder (same arithmetic, leading zero
byte, five closing shift-low steps, exact consumption of the payload) under a
context `c` with its own adaptive probability of a zero bit:

```
c = ((dcls · 64 + pat) · 4 + b) · 2 + s                    3072 contexts
s    = (v >= thr)                         this voxel's base side
dcls = 0 if dd <= 1, 1 if <= 3, 2 if <= 7, 3 if <= 15, 4 if <= 31, else 5
pat  = bit j set iff causal neighbour j is on the other side than s:
       j  0 (0,0,-1)  1 (0,-1,0)  2 (-1,0,0)  3 (0,-1,-1)  4 (-1,0,-1)  5 (-1,-1,0)   [(dz,dy,dx)]
b    = how many of (0,0,+1), (0,+1,0), (+1,0,0) are on the other side than s
```

A neighbour's side is read from the chunk as it stands when the voxel is visited:
the final side of an earlier voxel (after its own refinement), the base side of a
later one. A neighbour coordinate outside the chunk is replaced by the voxel's own
coordinate on that axis (so it counts as agreeing).

Probabilities are 12-bit, start at 2048 and adapt by `p += (4096 − p) >> k` on a
zero bit and `p −= p >> k` on a one, with `k = 1 + a / 2` for a context that has
coded `a` bits before (`a` saturating at 6: `k = 1, 1, 2, 2, 3, 3, 4, …`), and are
then clamped to [16, 4080].

### 13.4 Notes for readers

`volcomp_decode` returns the refined chunk; `volcomp_decode_block` returns its
block (it decodes the whole chunk, since the refinement is one adaptive pass).
`volcomp_stream_q` reports the base's `q` (a flat-law step, so a surface chunk at
`q` is finer than a mode-0 chunk at the same `q`: surface q 48 is about the size
of mode-0 q 16); `volcomp_is_lossless` is false; `volcomp_surface_info` returns
`thr` and `M`. Measurements: `docs/surface_mode.md`.
