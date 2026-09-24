# Surface mode (mode 6): storing surface predictions

Surface predictions are the probability volumes of the Vesuvius surface
teachers: the `surface_recto_3dunet` "recto" teacher (a 3-D U-Net at 2.4 µm,
softmax foreground probability) and the `surface_m7_nnunet` "m7" teacher (nnU-Net
run at 9.6 µm and upsampled 4×). Both are stored as `u8 = round(255 p)`. The
Vesuvius team publishes only the thresholded masks. This note measures what these
fields are and how volcomp spends bits on them, compares candidate designs on the
same data, and describes the mode that was built: **mode 6, a flat-law DCT base
plus a refinement that makes the 0.5 threshold exact**.

The format is in `spec/format.md` §13 and the API in `volcomp.h`
(`volcomp_surface_encode`, `volcomp_surface_info`). In Python it is
`VolcompCodec(mode="surface", q=…)`; on the command line it is
`volcomp encode --surface`. The study scripts are in `tools/surface_study/`.

## 1. Data

**Fresh float predictions (the ground truth for this study).** Both teachers
were run with their fp16 TensorRT engines on the two RTX 5060 Ti cards of
forlindesk2: recto on GPU 0, m7 on GPU 1, one card per teacher. The input was the
PHercParis4 2.4 µm masked CT
(`/vesuvius/usrm/volcomp/PHercParis4/20260411134726-2.400um-0.2m-78keV-masked.zarr`).
The probabilities were kept as floats, before any u8 or q=8 quantisation. The
regions below were picked for papyrus density (CT non-air fraction at level 4 is
1.0) and spread along z. Script: `tools/surface_study/gen_float.py`.

| region | origin z, y, x (level 0) | size | stored as | recto time | m7 time |
|---|---|---|---|---|---|
| evalbox | 34432, 15104, 18432 | 256×1024×1024 | float32 | 22 s | 3 s |
| bbox | 45056, 13824, 18432 | 256×1024×1024 | float32 | 22 s | 3 s |
| r15k | 15360, 21504, 11264 | 1024³ | float16 | 107 s | 10 s |
| r32k | 32256, 20992, 20992 | 1024³ | float16 | 106 s | 18 s |
| r47k | 47104, 10240, 11776 | 1024³ | float16 | 110 s | 15 s |
| r60k | 60416, 19456, 17920 | 1024³ | float16 | 176 s | 35 s |

The raw outputs take 21 GiB in `/vesuvius/usrm2/volcomp_surface/float/`: each
256×1024² region costs 1 GiB per teacher and each 1024³ region costs 2 GiB. For
evaluation, two 256³ cubes from every region give 12 cubes per teacher, about
201 M voxels each. Every cube is 8 whole 128³ chunks.

**What the fields are.** On the evalbox, with u8 = round(255 p):

| | recto | m7 |
|---|---|---|
| max p | 0.946 (never reaches 1) | 0.999 |
| voxels exactly 0 in u8 | 6.3 % | 64 % |
| soft voxels (1..254) | 94 % | 36 % |
| voxels >= 128 | 23 % | 17 % |
| voxels in u8 0..15 | 48 % (a noise floor everywhere) | 74 % |
| soft voxels' distance to the 0.5 surface: median / 90th pct | 5.7 / 18 voxels | 2.5 / 4.7 voxels |

- **Recto** is a fat, soft band. Its probability rises slowly across the sheet
  and sits on a low noise floor away from it.
- **m7** is thin and smooth: it is a trilinear 4× upsample, zero away from the
  sheets, with a sharp air-mask edge.

**The published predictions** are binary u8 masks (0/255, thresholded at
0.45). `surface-pack` already stores those with the mask modes (§11/§12), so no
new reads of dl.ash2txt.org were needed.

**The existing stores (volcomp q=8).** 48 random stored chunks per store, via
`tools/surface_study/measure.py`. The bits/voxel columns are over every voxel in
the store; the per-chunk sizes are of the volcomp stream, without the zstd
wrapper those stores carry.

| store | shape | bits/voxel | bytes per stored chunk | order-0 entropy of the decoded u8 | frac 0 | frac 255 | frac >= 128 | frac 1..254 |
|---|---|---|---|---|---|---|---|---|
| eval.zarr (recto) | 256×1024² | 0.0689 | 17 735 | 6.30 bits | 0.102 | 0 | 0.211 | 0.898 |
| a.zarr (recto) | 384×6144² | 0.0647 | 16 548 | 5.88 | 0.137 | 0 | 0.192 | 0.863 |
| b.zarr (recto) | 384×4096² | 0.0639 | 17 095 | 6.48 | 0.080 | 0 | 0.213 | 0.920 |
| p4val256_m7_tta0.zarr | 256×1024² | 0.0775 | 19 919 | 3.74 | 0.575 | 0.011 | 0.162 | 0.414 |
| a_m7.zarr | 384×6144² | 0.0624 | 17 647 | 4.01 | 0.525 | 0.008 | 0.144 | 0.467 |
| b_m7.zarr | 384×4096² | 0.0628 | 16 088 | 4.36 | 0.460 | 0.003 | 0.149 | 0.537 |

## 2. How volcomp spends bits today

The premise that "per-voxel 8-bit quantisation wastes bits on flat regions" does
not hold for volcomp. The q=8 mode is a 16³ 3-D DCT with a dead-zone quantiser
and tANS. It already costs **0.06-0.08 bits per voxel** on these stores, which
is 100-125× smaller than u8. Flat or empty 16³ blocks cost a DC token and an EOB.
The bits go to the blocks the sheets cross.

For scale, this is what the same probabilities cost in other forms, averaged over
the evalbox and r32k cubes (`rawcost.py`):

| form | recto bits/voxel | m7 bits/voxel |
|---|---|---|
| float32 + zstd-19 | 20.2 | 21.6 |
| float16 + zstd-19 | 12.9 | 11.2 |
| u8 + zstd-19 | 3.94 | 1.70 |
| u8 volcomp lossless (q=0) | 2.50 | 1.03 |
| **u8 volcomp q=8** | **0.065** | **0.052** |
| threshold mask, zstd-19 / volcomp mode 5 | 0.076 / 0.025 | 0.047 / 0.012 |

What q=8 does lose is the thing consumers use: the **threshold**. Over the 12
cubes of each teacher:

- Recto at q=8 puts **0.66 % of voxels** on the wrong side of 0.5, for a dice of
  0.981.
- m7 at q=8 puts 0.18 % of voxels on the wrong side, for a dice of 0.992.
- The mask's skeleton is fragile: the skeleton of the q=8 mask agrees with the
  source's (F1, 1.75-voxel tolerance) only **0.52 (recto) and 0.64 (m7)** of
  the time.

## 3. Candidates measured

All candidates were run on the same real chunks, with the prototypes in
`tools/surface_study/`. Rates of the context-coded prototypes are ideal adaptive
code lengths under the same probability state machine as volcomp's range coder,
which lands within 0.1 % of them. Quality is measured against the FLOAT source,
in u8 units (1/255):

| metric | definition |
|---|---|
| dice | agreement of the masks at 0.5 |
| band MAE | mean error on voxels within 3 voxels of the 0.5 surface |
| disp | first-order shift of the 0.5 iso-surface, `|error| / |∇p|`, on voxels next to it, in voxels |
| bdist | symmetric boundary distance of the masks |
| sdf MAE | error of the signed distance field within 16 voxels of the surface |
| skel F1 | agreement of the mask skeletons |

**Round 1 (evalbox cube).** Per-voxel coarse quantisation loses badly.

- The pointwise designs below all keep the threshold exact (a bin edge sits at 128).
- The index-coding contexts are a 3-D median/Lorenzo predictor, gradient
  activity, prediction class and 8³ zero-block flags.

| design | recto size vs q8 | m7 size vs q8 | band MAE recto / m7 |
|---|---|---|---|
| uniform 4-level steps, dead zone 2, index context-coded | 16.5× | 6.6× | 1.1 / 1.1 |
| uniform 16-level steps, dead zone 8, context-coded | 8.3× | 3.8× | 4.0 / 4.0 |
| uniform 32-level steps (9 levels), dead zone 16, context-coded | 4.9× | 2.3× | 7.9 / 7.8 |
| log-odds bins (19 levels) | 9.8× | 4.5× | 6.7 / 5.3 |
| two-step: 4 near 0.5, 16 elsewhere | 10.3× | 4.7× | 2.8 / 3.3 |
| the same codebooks coded with the existing lossless mode | 1.1-1.6× larger again | | |
| near-lossless DPCM, error <= δ, in-loop quantiser, δ = 2 / 4 / 8 | — | 6.1× / 5.0× / 3.7× | — / 1.2, 2.3, 4.2 |
| mask only: mode 4 (2× pool) / mode 5 (exact) | 0.06× / 0.27× | 0.05× / 0.18× | 66 / 87 (no soft values) |

A quantised index field is its own staircase. Coding where the staircase steps
costs far more than the DCT spends on a smooth field. DCT coding stays well
ahead at 0.05-0.08 bpv.

**Round 2: variations around the DCT.**

- **Companding before the DCT** (tanh, steeper near 0.5): worse at every rate
  (m7 q16: +29 % size, band MAE 7.7 against 4.1).
- **Dead-zone floor** (source < t becomes 0, decode < t/2 becomes 0; t = 4..24):
  always larger (+0.2 % to +11 %). The floor creates edges that the DCT pays for.
- **2× mean pool, DCT, trilinear back:**
  - On m7 it is 0.64× q8 at q2, but band MAE 5.2.
  - At equal size the direct DCT has about 3.
  - On recto it is worse still (dice 0.986 at q2), because recto is
    native-resolution.
- **Exact mask (mode 5) plus DCT, forcing the side:** exact threshold, but the
  mask costs 0.012-0.025 bpv, 1.5-2× the refinement below.
  - Residual coding against a mask-derived ramp (signed-distance ramp, clip 3) or
    a blurred mask is worse again: the ramp inherits the mask's staircase.
- **The DCT step law** was swept over exponents 0 (flat) to 1.3, where step(r) =
  q·(1+r)^e (tools/surface_study/run3.py).
  - For these fields **flat is best**: at equal band MAE it is 6-7 % smaller.
  - At equal size it has about **13 % lower 99th-percentile error** near the
    surface (m7: flat q48 against standard q16, 0.055 against 0.053 bpv, p99 12.0
    against 13.8).
  - 0.65 is right for CT texture, not for smooth probabilities.
- **DCT plus an exact-threshold refinement** (§4): the winner. See the final
  table.

## 4. The design: mode 6

```
surface chunk = header(q, thr) + DCT base (flat step law, bit-exact) + refinement
```

**Base.** The existing DCT body (same tables, tokens, tANS and substreams), with
every AC step equal to q and DC at q/8. The encoder is the mode-0 encoder with the
flat step table, and its body lands in place after a 16-byte header.

**Refinement.**

- M is the largest distance from the threshold, in half steps, of a base voxel on
  the wrong side.
- Every voxel whose base value lies within M of the threshold is a candidate, and
  gets one context-coded bit: "is this voxel on the wrong side?".
- A flipped voxel decodes to thr or thr − 1, the nearest value on the source's
  side.

**Contexts.** Each bit is coded with the mask coder's range coder under 3 072
contexts. The context holds:

- how close the base came to the threshold (6 classes);
- which of 6 causal neighbours ended up on the other side;
- how many of the 3 forward neighbours sit on the other side in the base;
- the base side itself.

**Far candidates.** Candidates more than 31 half steps out are 60-70 % of all
candidates but rarely flip. They are coded only inside 8³ blocks flagged for
it.

- The flags are coded first.
- This codes 2.3-3.3× fewer decisions for 2-4 % fewer bytes.
- Richer contexts were tried: forward-neighbour closeness, local gradient
  class, and the expected distance from the gradient. Each gained less than
  1 %; the 8³ block gating used for far candidates was another of these.

**Bit-exact base.** The refinement reads its contexts from the base
reconstruction, so encoder and decoder must agree on it to the bit. volcomp's
AVX2 and C kernels only agree to ±1 LSB, so mode 6 reconstructs its base with
new **strict kernels**:

- IEEE binary32 with no FMA contraction and no reassociation;
- the reference butterfly order;
- steps from a table rather than `powf`, whose result depends on the libm.

The AVX2 version runs the scalar operation sequence on 8 lines at once. The
C/AVX2 and clang/GCC builds decode the golden stream to the same hash, and 32
real chunks at three q decode identically under the AVX2 and C builds. The strict kernels run at the
same speed as the default ones: 0.5-0.6 ms per chunk.

**Guarantee.** For every voxel, `(source >= thr) == (decoded >= thr)`. The mask,
its skeleton, its boundary and its distance field are exactly the source's.

**Compatibility.**

- Old streams decode unchanged: mode 0 is byte for byte what it was, and the
  parse refactor is exercised by every existing test.
- A revision-3 reader refuses mode 6 cleanly: the mode byte is above its
  maximum.
- The zarr v3 layout is unchanged: 1024³ shards of 128³ chunks, codec chain
  exactly `[volcomp]`, configuration `{"mode": "surface", "q": Q}`.

The binary mask modes remain the special case for data that is only a mask. On a
0/255 input a surface chunk is exact as well, but mode 5 is the right tool there.

## 5. Results

The implemented codec (C, through the Python binding) against the existing
modes, on 12 cubes of 256³ (201 M voxels) per teacher, errors against the float
source (`final_eval.py`, `final_table.py`). "Surface qN" is the flat-law step N,
so surface q48 is about the size of mode-0 q16.

### recto

| candidate | bits/voxel | size vs q8 | dice | band MAE | band p99 | disp mean | disp p99 | bdist | sdf MAE | skel F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| dct q4 | 0.1179 | 1.823 | 0.9850 | 2.02 | 7.8 | 0.153 | 0.71 | 0.161 | 0.232 | 0.564 |
| **dct q8 (today)** | 0.0647 | 1.000 | 0.9807 | 2.61 | 10.2 | 0.195 | 0.91 | 0.209 | 0.318 | 0.521 |
| dct q16 | 0.0374 | 0.578 | 0.9757 | 3.36 | 13.3 | 0.245 | 1.12 | 0.265 | 0.417 | 0.473 |
| mask5 (mask only) | 0.0254 | 0.393 | 1 | 102.8 | 127 | 9.49 | 27.1 | 0 | 0 | 1 |
| surface q32 | 0.0759 | 1.174 | **1** | 2.59 | 10.1 | 0.177 | 0.81 | **0** | **0** | **1** |
| surface q48 | 0.0583 | 0.901 | **1** | 3.09 | 12.2 | 0.206 | 0.94 | **0** | **0** | **1** |
| **surface q64** | 0.0503 | **0.777** | **1** | 3.51 | 13.9 | 0.228 | 1.04 | **0** | **0** | **1** |
| surface q96 | 0.0430 | 0.665 | **1** | 4.22 | 17.0 | 0.265 | 1.22 | **0** | **0** | **1** |
| surface q128 | 0.0397 | 0.614 | **1** | 4.82 | 19.4 | 0.295 | 1.37 | **0** | **0** | **1** |

### m7

| candidate | bits/voxel | size vs q8 | dice | band MAE | band p99 | disp mean | disp p99 | bdist | sdf MAE | skel F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| dct q4 | 0.0785 | 1.501 | 0.9946 | 1.42 | 6.0 | 0.052 | 0.23 | 0.051 | 0.065 | 0.686 |
| **dct q8 (today)** | 0.0523 | 1.000 | 0.9917 | 2.21 | 9.2 | 0.079 | 0.36 | 0.079 | 0.101 | 0.642 |
| dct q16 | 0.0344 | 0.658 | 0.9873 | 3.42 | 13.9 | 0.120 | 0.53 | 0.119 | 0.151 | 0.607 |
| mask5 (mask only) | 0.0121 | 0.232 | 1 | 82.6 | 127 | 4.67 | 16.0 | 0 | 0 | 1 |
| surface q32 | 0.0530 | 1.014 | **1** | 2.23 | 9.3 | 0.079 | 0.36 | **0** | **0** | **1** |
| surface q48 | 0.0427 | 0.816 | **1** | 2.98 | 12.2 | 0.103 | 0.47 | **0** | **0** | **1** |
| **surface q64** | 0.0370 | **0.708** | **1** | 3.64 | 14.8 | 0.124 | 0.56 | **0** | **0** | **1** |
| surface q96 | 0.0309 | 0.592 | **1** | 4.84 | 19.5 | 0.161 | 0.72 | **0** | **0** | **1** |
| surface q128 | 0.0277 | 0.529 | **1** | 5.87 | 23.5 | 0.191 | 0.84 | **0** | **0** | **1** |

### Size of surface q48 / q64 / q96 as a fraction of q8, per region

| region | recto q8 bpv | recto | m7 q8 bpv | m7 |
|---|---|---|---|---|
| evalbox | 0.0664 | 0.835 / 0.725 / 0.610 | 0.0757 | 0.813 / 0.690 / 0.553 |
| bbox | 0.0579 | 0.738 / 0.609 / 0.485 | 0.0269 | 0.820 / 0.710 / 0.595 |
| r15k | 0.0751 | 0.929 / 0.796 / 0.685 | 0.0482 | 0.792 / 0.696 / 0.595 |
| r32k | 0.0689 | 1.030 / 0.904 / 0.797 | 0.0632 | 0.830 / 0.724 / 0.605 |
| r47k | 0.0648 | 0.985 / 0.862 / 0.757 | 0.0477 | 0.825 / 0.732 / 0.633 |
| r60k | 0.0549 | 0.850 / 0.733 / 0.623 | 0.0518 | 0.816 / 0.705 / 0.589 |

### Reading the table

- **Surface q32** is the soft quality of today's q8, with an exact mask. It is
  the same size on m7 and +17 % on recto.
- **Surface q64**, the recommended default, is **0.78× (recto) and 0.71× (m7)**
  of today's q8.
  - It has an exact mask, skeleton and distance field.
  - Its soft error in the band is about 3.5/255 (1.4 % of full scale), against
    2.2-2.6 today.
  - The iso-surface of the soft values moves by 0.12-0.23 voxel on average.
- **Surface q96-q128** is 0.53-0.67× of q8.
- **At equal size the two modes trade soft fidelity for the mask.**
  - Recto: surface q128 has band MAE 4.8 where mode-0 q16 has 3.4.
  - m7: surface q64 has band MAE 3.6 where mode-0 q16 has 3.4.
  - Pick mode 0 when only the soft values matter, and mode 6 whenever anything
    thresholds.

### Cost of the exact threshold

Recto's refinement is the expensive part: the base error near its shallow 0.5
crossing is often larger than the distance to the threshold.

- At q64 the refinement is 26-51 % of a recto chunk and 12-27 % of an m7 chunk
  (evalbox, r32k, r47k).
- Most of its bits go to voxels within 7 half steps of the threshold, where the
  base is on the wrong side 25-45 % of the time. That is information the base
  does not carry.

### Throughput

Single thread, forlindesk2 (Core Ultra 9, AVX2), 16 real chunks per teacher
(`bench_surf.c`).

| | recto encode | recto decode | m7 encode | m7 decode |
|---|---|---|---|---|
| mode 0, q8 | 1 672 MB/s | 3 455 MB/s | 1 904 MB/s | 4 184 MB/s |
| surface q48 | 402 | 961 | 612 | 1 478 |
| surface q64 | 406 | 958 | 619 | 1 519 |
| surface q96 | 413 | 936 | 619 | 1 478 |

Decode is 2.8-3.6× slower than q8. Per chunk, the strict base takes 0.5-0.6 ms
and the refinement 1.0-1.6 ms, which is about 7 ns per coded decision. At about
1 GB/s per core, a 1024³ shard decodes in about a second on one core.
`volcomp_decode_block` on a surface chunk decodes the whole chunk, because the
refinement is one adaptive pass.

## 6. Use

```python
from volcomp_zarr import VolcompCodec
z = zarr.create_array(store, shape=shape, chunks=(128,) * 3, shards=(1024,) * 3, dtype="uint8",
                      serializer=VolcompCodec(mode="surface", q=64), compressors=None, fill_value=0)
```

```c
volcomp_surface_encode(src, 64.0f, 128, enc, VOLCOMP_SURFACE_ENCODE_BOUND, &n);
volcomp_decode(enc, n, dst, VOLCOMP_CHUNK_VOXELS);          /* like any chunk */
```

## 7. Reproducing

The scripts in `tools/surface_study/` were run on forlindesk2 with the usrm2
venv (for inference) and a separate venv (numpy, scipy, scikit-image, numba,
zarr). Their paths point at `/vesuvius/usrm2/volcomp_surface/`.

| step | scripts |
|---|---|
| pick regions and run the teachers | `pick_regions.py`, `gen_float.py` |
| store statistics, float statistics, raw costs | `measure.py`, `fstats.py`, `rawcost.py` |
| metrics and data access | `common.py` |
| candidate prototypes | `ctxcoder.py`, `nl.py`, `run1.py` … `run6.py` |
| refinement context studies | `fixexp2.py`, `fixexp3.py`, `fixexp6.py` |
| the final table | `final_eval.py`, `final_table.py`, `rebytes.py` |
| the test fixture | `mkfixture.py` |
| throughput | `bench_surf.c` |

## 8. Not done / caveats

- **External readers must learn mode 6 before stores are written with it.**
  vc3d's embedded volcomp reader is an example. A revision-3 reader rejects the
  chunk; it does not misread it.
- **Encode is about 4× slower than q8** (0.4-0.6 GB/s). The strict base decode
  and the candidate scan in the encoder are not yet tuned.
- **The existing q=8 stores were not re-encoded** (the task forbids rewriting
  them in place). Re-encoding from them would stack two quantisers anyway; the
  numbers above are from float source.
- **Per-chunk q choice (rate control) is not implemented.** Neither is an
  adaptive threshold set (several thresholds, for example 0.45 and 0.5).
