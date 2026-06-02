#!/usr/bin/env bash
# Chunk-size sweep (PLAN §5 Phase-1 item 3 / §2 "Chunk size").
#
# VC_CHUNK_SIDE is a COMPILE-TIME constant (src/config.h: #define VC_CHUNK_SIDE,
# overridable via -D). So for each side in {32,64,128,256} we RECOMPILE the
# Phase-0 pipeline + the sweep worker, then run it on the SAME real PHerc Paris 4
# data (8 high-res 128^3 center chunks tiled, and a coarse 384^3 downscaled
# sub-region). Emits one combined TSV table, then a formatted table.
#
# This script only compiles src/* (the existing Phase-0 lib) + harness/sweep_metrics.c.
# It does NOT touch CMakeLists.txt, bench.c, or any other agent's files.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/harness/sweep_results.tsv}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

CC="${CC:-gcc}"
CFLAGS="-O3 -ffp-contract=fast -ffast-math -std=c23 -Wall -Wextra -I$ROOT/include"

# The pipeline sources (codec/transform/entropy/ratectrl/config/blocks) are OWNED
# by other agents and may be MID-EDIT (uncommitted, not compilable) in the working
# tree. To get a stable, reproducible build of the Phase-0 pipeline we materialise
# those sources from a pinned git ref (default: the Phase-0 commit). Our own files
# (metrics.c, sweep_metrics.c) come from the working tree.
PIN_REF="${VC_PIN_REF:-e6c86f9}"   # Phase-0: end-to-end DCT+Rice codec
PIN="$WORK/pin"
mkdir -p "$PIN"
git -C "$ROOT" archive "$PIN_REF" | tar -x -C "$PIN"
# Overlay OUR current metrics.c (the file we own) onto the pinned tree.
cp "$ROOT/src/metrics/metrics.c" "$PIN/src/metrics/metrics.c"
CFLAGS="-O3 -ffp-contract=fast -ffast-math -std=c23 -Wall -Wextra -I$PIN/include"

# Phase-0 library sources (compile-time pipeline) built from the pinned ref tree.
# NOTE: we never compile bench.c.
LIB_SRC=(
  "$PIN/src/codec.c"
  "$PIN/src/transform/dct_int.c"
  "$PIN/src/entropy/rice.c"
  "$PIN/src/ratectrl/fixed.c"
  "$PIN/src/metrics/metrics.c"
)

HIRES_DIR="$ROOT/data/hires_chunks"
COARSE_FILE="$ROOT/data/coarse/coarse_z15_y8_x8_3x3x3.u8"
# coarse file is a 3x3x3 grid of 128^3 = 384^3.
COARSE_DIM=384

# q values. Phase-0 Rice+DCT ratio saturates ~7.5x on this (noisy X-ray) data, so
# 10x/20x/50x are above the achievable ceiling; we sweep a q spread that brackets
# the realised operating points (low/mid/high distortion) and report achieved ratio.
# (The "10x/20x/50x target" reduces in practice to "max ratio the pipeline yields".)
QVALS=(8 16 32 64 128)

echo "# chunk-size sweep on real PHerc Paris 4 data  ($(date -u +%FT%TZ))" > "$OUT"
echo -e "dataset\tchunk\tq\tratio\tpsnr\tmsssim\tgmsd\tencMBs\tdecMBs\tfaces\tbstep" >> "$OUT"

for SIDE in 32 64 128 256; do
  echo ">>> building pipeline at VC_CHUNK_SIDE=$SIDE ..." >&2
  OBJS=()
  for s in "${LIB_SRC[@]}"; do
    o="$WORK/$(basename "$s" .c).$SIDE.o"
    # shellcheck disable=SC2086
    $CC $CFLAGS -DVC_CHUNK_SIDE=$SIDE -c "$s" -o "$o"
    OBJS+=("$o")
  done
  BIN="$WORK/sweep.$SIDE"
  # shellcheck disable=SC2086
  $CC $CFLAGS -DVC_CHUNK_SIDE=$SIDE "$ROOT/harness/sweep_metrics.c" "${OBJS[@]}" -lm -o "$BIN"

  echo ">>> running side=$SIDE on hires + coarse ..." >&2
  "$BIN" hires  "$HIRES_DIR"   "${QVALS[@]}" >> "$OUT"
  "$BIN" coarse "$COARSE_FILE" "$COARSE_DIM" "$COARSE_DIM" "$COARSE_DIM" "${QVALS[@]}" >> "$OUT"
done

echo ">>> sweep complete -> $OUT" >&2
echo
echo "==== chunk-size sweep table (real PHerc Paris 4 data) ===="
# Pretty-print with column, skipping comment lines.
grep -v '^#' "$OUT" | column -t
