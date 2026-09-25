#!/usr/bin/env bash
# Build web/dist/volcomp_viewer.html: compile volcomp.h (via src/shim.c) to
# WebAssembly with Emscripten, then inline the wasm (base64), the worker and the
# main script into one self-contained HTML file.
# Needs emcc (emsdk; sourced from ~/emsdk if not on PATH) and python3.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v emcc >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" >/dev/null 2>&1 || { echo "emcc not found (install emsdk)" >&2; exit 1; }
fi
mkdir -p build dist
# -msimd128: the strict (surface-base) kernels stay bit-exact under it and every
# decode matches the native C kernels bit for bit (web/test/wasm_check.mjs).
emcc -O3 -msimd128 -std=c2x -Wall -Wno-unused-function \
  -sSTANDALONE_WASM --no-entry -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 -sSTACK_SIZE=1048576 \
  -o build/volcomp.wasm src/shim.c
python3 build.py build/volcomp.wasm dist/volcomp_viewer.html
