#!/usr/bin/env bash
# Chunk-model (block-grid) bake-off sweep on real PHerc Paris 4 data.
#
# Runs harness/bench_chunkmodel over a q (base dead-zone step) sweep so the
# {stencil x traversal x edge x chunk-size x entropy} matrix can be read at
# equal-PSNR points and compared against c3d/c4d (harness/compare_c3d_c4d.*).
# bench_chunkmodel itself sweeps the whole matrix at one q; this drives the q
# axis so the equal-PSNR ratio curve falls out.
#
# Usage:  harness/sweep_chunkmodel.sh [repo_root] [q ...]
# Default q set brackets ~35-48 dB on hires-256 / coarse-256.
set -euo pipefail
ROOT="${1:-.}"
shift || true
QS=("$@"); [ ${#QS[@]} -eq 0 ] && QS=(4 8 12 16 24 32 48 64)

BIN="$ROOT/build/bench_chunkmodel"
if [ ! -x "$BIN" ]; then
  echo "building bench_chunkmodel..." >&2
  cmake -S "$ROOT" -B "$ROOT/build" >/dev/null
  cmake --build "$ROOT/build" --target bench_chunkmodel >/dev/null
fi

for q in "${QS[@]}"; do
  echo "############################ q(base-step) = $q ############################"
  "$BIN" "$ROOT" "$q"
done
