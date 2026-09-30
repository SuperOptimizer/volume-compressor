#!/usr/bin/env bash
# Build web/dist/volcomp_viewer.html: compile volcomp.h (via src/shim.c) plus the
# zstd decoder to WebAssembly with Emscripten, then inline the wasm (base64), the
# worker and the main script into one self-contained HTML file.
# Needs emcc (emsdk; sourced from ~/emsdk if not on PATH), python3 and curl (the
# zstd release tarball is fetched once into build/ and checked against its sha256).
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v emcc >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" >/dev/null 2>&1 || { echo "emcc not found (install emsdk)" >&2; exit 1; }
fi
mkdir -p build dist
ZSTD_VER=1.5.7
ZSTD_SHA=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
Z=build/zstd-$ZSTD_VER
if [ ! -d "$Z/lib" ]; then
  curl -sSL -o "build/zstd-$ZSTD_VER.tar.gz" "https://github.com/facebook/zstd/releases/download/v$ZSTD_VER/zstd-$ZSTD_VER.tar.gz"
  echo "$ZSTD_SHA  build/zstd-$ZSTD_VER.tar.gz" | sha256sum -c --quiet
  tar -xzf "build/zstd-$ZSTD_VER.tar.gz" -C build
fi
# zstd: the decoder only (zarr "zstd" bytes->bytes codec after volcomp, as in the
# teacher-region stores), no legacy formats, no dictionary builder, no asm.
ZSRC=("$Z"/lib/common/{entropy_common,error_private,fse_decompress,xxhash,zstd_common}.c
      "$Z"/lib/decompress/{huf_decompress,zstd_ddict,zstd_decompress,zstd_decompress_block}.c)
# -msimd128: the strict (surface-base) kernels stay bit-exact under it and every
# decode matches the native C kernels bit for bit (web/test/wasm_check.mjs).
emcc -O3 -msimd128 -std=gnu2x -Wall -Wno-unused-function \
  -DVC_WITH_ZSTD -DZSTD_LEGACY_SUPPORT=0 -DZSTD_DISABLE_ASM -DZSTD_STRIP_ERROR_STRINGS -I"$Z/lib" \
  -sSTANDALONE_WASM --no-entry -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 -sSTACK_SIZE=1048576 \
  -o build/volcomp.wasm src/shim.c "${ZSRC[@]}"
python3 build.py build/volcomp.wasm dist/volcomp_viewer.html
