# volcomp stream format, version 1

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
5       1      mode   0 = lossy DCT (this section), 1..3 = lossless (§10)
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
8       T      frequency tables (§4.3), 6 models
8+T     D      directory: 32 × { LEB128 tok_n, LEB128 bypass_n }
8+T+D   P      payload: for s = 0..31: tok_n[s] token bytes then bypass_n[s] bypass bytes
```

Exact accounting is normative, as in §2. Per substream `tok_n >= 3`,
`tok_n <= 131 112`, `bypass_n <= 73 736`.

**Models.** 0 block mode, 1 SPARSE runs and EOB, 2 SPARSE levels, 3..5 DENSE
levels by context. Frequency tables are encoded exactly as in §4.3.

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
- `1 SPARSE` — `(run, level)*` then `EOB`, all runs and `EOB` from model 1 and
  all levels from model 2. `run` is the number of zero residuals skipped;
  `level` is `u - 1` for the nonzero residual `u` that follows, so
  `level <= 254`. `pos + run <= 4095` is normative.
- `2 DENSE` — exactly 4096 level tokens, one residual each (`u <= 255`). The
  model is `3 + c` where `c = 0` if the residual one z-plane back is 0, `1` if
  it is 1 or 2, else `2` (`c = 0` in the first z-plane). The context looks a
  whole plane back so that it is never on the entropy decoder's critical path.
- `3 RAW` — 4096 bypass bytes, the block verbatim.

**Bound.** The encoder emits mode 2 whenever the coded form would reach
2 097 160 bytes, so a lossless stream is never larger than the raw chunk plus
the 8-byte header. `VOLCOMP_ENCODE_BOUND` is unchanged.
