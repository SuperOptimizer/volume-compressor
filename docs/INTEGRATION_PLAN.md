# volume-compressor — Integration Plan

**Status:** draft, pre-audit. Gated behind the AUDIT (#22) which freezes the feature list.
**Premise:** we have ~26 experiments measured *in isolation against an RLGR baseline*. Integration
is NOT a `git merge` of branches (they're self-contained bake-off engines). It is **building one
real codec with the kept features wired into the actual pipeline, then MEASURING THE COMPOSED
STACK** — because composed gains ≠ sum of isolated gains.

---

## 0. The core risk: "measured-in-isolation vs RLGR" staleness

Almost every kept feature (context-coder, RDOQ, dead-zone tuning, zero-structure, D4-lattice,
EG2024 3-band) was measured with **RLGR as the baseline entropy coder**. But the chosen DEFAULT
is now the **coefficient context coder**. So **every isolated gain number is provisional** until
re-measured on the context-coder stack. The integration must re-measure each kept feature's
*incremental* contribution on the real composed pipeline and **prune any whose gain evaporated.**

---

## 1. Composition map (what stacks / overlaps / conflicts)

Pipeline order (encode):
`gather 16³ atom → DCT-16³ → [LFNST low-freq secondary?] → quant{HF-matrix + dead-zone(0.80/0.40) + RDOQ + [D4-lattice low-freq?]} → entropy{context-coder default | RLGR alt} + [zero-structure CBF/EOB?] → per-16³ Lagrangian rate control`
Container/decode: `M1 region-scoped shared tables · inter-LOD residual (+anchor LOD hedge) · deblock (decode post-filter)`

### 🟢 Clean / orthogonal (compose freely)
- DCT-16³ · M1 chunk-model · archive container · deblock · inter-LOD — different layers, no shared mechanism.
- D4 lattice (low-freq group only, decode-identical to scalar) — clean swap.
- 16³ random-access invariant — preserved by every kept feature (verified touched=1 each).

### 🟡 Overlapping — gains will NOT fully add; MUST measure composed
- **Quant-stage pile-up:** HF-matrix + dead-zone-tuning + RDOQ + D4-lattice all modify quantization, all measured separately. RDOQ's round-to-zero overlaps dead-zone widening; HF-matrix vs edge-weight both protect HF. **Re-measure the composed quant stack; expect sub-additive.**
- **Context-coder × RDOQ:** RDOQ's λ/rate-model was tuned for RLGR. On the context coder it's miscalibrated → **re-tune λ or the +10-15% shrinks.**
- **Context-coder × zero-structure:** context coder already models significance; sub-cube CBF+EOB overlaps. **Measure INCREMENTAL on context coder — likely << the ~3-6% measured vs RLGR.**

### 🔴 Conflicts / supersessions (pick, don't stack)
- **Entropy backends are mutually exclusive:** context-coder (default) vs RLGR (fast-decode alt) vs 3-band-rANS (third option). NOT stackable. The EG2024 **3-band split's +25-35% was vs RLGR-on-rANS — it does NOT stack with the context coder and is likely DEAD WEIGHT if context-coder is default.** Decide: is the 3-band-rANS path even kept?
- **Inter-LOD vs independent-LOD-fetch:** resolved by format hedge (top-LOD anchor + per-LOD independent/residual flag).
- **Spatial/directional tricks (z-aniso, directional-basis, edge-weight):** all NIX/PARK — the global HF quant matrix already does the frequency-domain perceptual work; do NOT include.

---

## 2. The kept feature set (provisional — audit confirms)

**★ SINGLE-PIPELINE MANDATE (user 2026-06-02):** ONE frozen pipeline, NO optional features / selectable
paths / profiles / user-pickable modes. This kills the old "compile-time toolkit" framing (that was for
exploration only). Consequences below.

**Definite KEEP (clean, confirmed):** DCT-16³ · HF-protecting quant matrix · dead-zone(0.80/0.40) ·
M1 region-scoped chunk · per-16³ rate control via **closed-form q-from-λ (THE rate controller — replaces
multi-q probe; 8-10× faster, hits high-ratio targets the probe fails at, equal/better quality; drop probe
+ feedback variant; likely fixes the failing ratectrl ctest)** · deblock(0.3-0.5) · 16³ random-access ·
**D4-lattice low-freq quant (+0.9..1.75 dB iso-rate, ZERO decode cost — KEEP; cross-check additivity with RDOQ at integration)** ·
**context coder = THE SOLE entropy stage (RLGR dropped from codec, repo-reference only)** ·
**inter-LOD residual = ALWAYS-ON, no per-LOD flag (user-locked). HARD format requirement: top-LOD anchor
always independent (to bootstrap the chain) + spatially co-located residual chains with explicit parent
refs in index. Cold jump to a fine LOD decodes the coarse→fine chain — accepted.**

**DROPPED by single-pipeline mandate:** RLGR code path · EG2024 3-band-rANS split (was a rANS-alternate-path
crutch; rANS path gone → it goes; its low-AC/high-AC insight should live in the context coder's contexts) ·
fast-rc encode profile · inter-LOD per-LOD flag · the toolkit-config layer.

**KEEP pending composition re-measure:** RDOQ (re-tune λ for context coder) · D4-lattice low-freq quant ·
dead-zone tuning (vs RDOQ overlap).

**KEEP pending round-3:** LFNST low-freq secondary transform · zero-structure CBF/EOB (incremental on context coder).

**Likely DROP on composition:** EG2024 3-band split (RLGR-moot, context-coder default makes it dead weight) —
unless the 3-band-rANS path is kept as a separate profile.

**NIX (confirmed):** curve-grouping · cross-atom prediction · DC-subvolume(ratio-neutral) · sign-modeling ·
z-anisotropy · directional-basis · edge-weight · learned-tables · VQ-codebook · bitplane/SNR-scalable ·
content-adaptive quant · perceptual-in-loop · 9/7 wavelet · lapped transform · adaptive-transform-mode.

---

## 3. Integration phases

**Phase I — clean core (one real codec, not bake-off engines).**
Build the production codec around the single source-of-truth pipeline: one DCT-16³, one quant module,
one entropy module (context-coder default + RLGR alt selectable), M1 chunk container, per-16³ RC, deblock.
No `#include`-the-.c bake-off hacks. Wire the per-16³ q-field into the chunk header (the outstanding
rate-control header change). Build green, round-trip + touched=1 tests pass. **Measure this as the new baseline.**

**Phase II — compose the quant-stage features, measure incremental.**
Add RDOQ (re-tune λ for context coder) → measure. Add D4-lattice → measure. Confirm dead-zone tuning still
helps on top of RDOQ (or fold/drop). **Prune anything whose incremental gain on the composed stack is noise.**

**Phase III — fold in round-3 winners (if any) + inter-LOD.**
LFNST and zero-structure (if round-3 keeps them) — measure incremental on the context coder. Wire inter-LOD
residual coding with the top-LOD-anchor + per-LOD-flag format hedge.

**Phase IV — composed end-to-end validation.**
Full multi-LOD archive (0/..N/), real PHerc Paris 4, vs c3d AND c4d on a NON-stitched native volume (the
stitched cube unfairly penalized c3d). Report the COMPOSED ratio/quality/decode-speed — the real numbers.
Confirm encode speed is acceptable (least-important axis; one-time, parallel).

**Phase V — freeze the format.**
Lock the archive byte layout per the format-lit hedges: u64 offsets, 64-byte atom alignment + 4KiB
footer/region alignment, dual-magic footer + versioned tag-based metadata, xxHash-per-atom + CRC32C-index,
top-LOD anchor + per-LOD independent/residual flag. Document the frozen format FROM the code.

---

## 4. How integration actually happens (mechanics)

- Build on a fresh `integration` branch off clean `main`.
- Pull each kept feature's *implementation* from its `exp-*`/`r2-*`/`r3-*` branch, but RE-IMPLEMENT it
  into the single production pipeline (not the isolated bench engine). The branches are reference, not merge sources.
- One integrator agent (or a small sequence) does this deliberately — NOT parallel agents on the same files
  (the collision lesson). Each phase commits + re-measures before the next.
- Keep `main` clean; merge `integration` → `main` only after Phase IV validation passes.

---

## 5. Open questions the AUDIT (#22) must resolve before Phase I

1. **Is the 3-band-rANS path kept at all?** (If context-coder is the sole default, the EG2024 split is dead weight.)
2. **Does RDOQ survive re-tuning on the context coder, and does it still beat dead-zone tuning alone?** (overlap)
3. **Does zero-structure add anything on the context coder** (which already models significance)?
4. **fast-rc:** ship as an optional fast-encode profile or drop? (encode = least important)
5. ~~ink-quality validation gap~~ **DECIDED (user 2026-06-02): forget it.** Freeze quality on PROXY metrics
   (PSNR/MAE/L∞/p99/MS-SSIM/GMSD/HaarPSI); do NOT build an ink-detection harness or gate on ink-AUC.
   Design is already conservative on HF (what the domain review said ink needs); proxies are sufficient.
