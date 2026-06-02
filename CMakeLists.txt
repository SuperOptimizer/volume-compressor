cmake_minimum_required(VERSION 3.20)
project(volume_compressor C)

# Pure C23, libc + libm only, no GPU, no hand intrinsics (PLAN §1, §7).
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compile-time chunk side (PLAN: 32/64/128/256 must all build; default 64).
set(VC_CHUNK_SIDE "64" CACHE STRING "chunk edge in voxels (32/64/128/256)")

# Compile-time block selection (the bake-off levers). Entropy coder and the
# HF-weighting policy of the quantizer are selected here; bench.sh sweeps them.
#   VC_ENTROPY: rice | rlgr | rans
#   VC_QWEIGHT: 0 (flat) | 1 (HF-protecting) | 2 (content-adaptive)
set(VC_ENTROPY "rice" CACHE STRING "entropy coder: rice|rlgr|rans")
set(VC_QWEIGHT "0" CACHE STRING "quant HF weighting: 0 flat|1 HF|2 adaptive")
set(VC_TRANSFORM "dct8" CACHE STRING "transform: dct8|dct16")

add_compile_options(-O3 -ffp-contract=fast -ffast-math -Wall -Wextra)
add_compile_definitions(VC_CHUNK_SIDE=${VC_CHUNK_SIDE})
add_compile_definitions(VC_QWEIGHT=${VC_QWEIGHT})

if(VC_ENTROPY STREQUAL "rice")
  add_compile_definitions(VC_ENTROPY_ENC=vc_rice_encode VC_ENTROPY_DEC=vc_rice_decode VC_CFG_ENTROPY_TAG=1)
elseif(VC_ENTROPY STREQUAL "rlgr")
  add_compile_definitions(VC_ENTROPY_ENC=vc_rlgr_encode VC_ENTROPY_DEC=vc_rlgr_decode VC_CFG_ENTROPY_TAG=2)
elseif(VC_ENTROPY STREQUAL "rans")
  add_compile_definitions(VC_ENTROPY_ENC=vc_rans_encode VC_ENTROPY_DEC=vc_rans_decode VC_CFG_ENTROPY_TAG=3)
else()
  message(FATAL_ERROR "VC_ENTROPY must be rice|rlgr|rans")
endif()

# Emit auto-vectorization reports so we can SEE which hot loops vectorized
# (PLAN §7). GCC: -fopt-info-vec; Clang: -Rpass=loop-vectorize.
option(VC_VEC_REPORT "emit vectorization diagnostics" ON)
if(VC_VEC_REPORT)
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fopt-info-vec-optimized)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Rpass=loop-vectorize)
  endif()
endif()

if(VC_TRANSFORM STREQUAL "dct8")
  set(VC_TRANSFORM_SRC src/transform/dct_int.c)
elseif(VC_TRANSFORM STREQUAL "dct16")
  set(VC_TRANSFORM_SRC src/transform/dct_int16.c)
  add_compile_definitions(VC_TRANSFORM_FWD=vc_dct_int16_fwd
                          VC_TRANSFORM_INV=vc_dct_int16_inv
                          VC_CODEC_TAG_TRANSFORM=2)
else()
  message(FATAL_ERROR "VC_TRANSFORM must be dct8|dct16")
endif()

set(VC_SRC
  src/codec.c
  ${VC_TRANSFORM_SRC}
  src/entropy/rice.c
  src/entropy/rlgr.c
  src/entropy/rans_static.c
  src/quant/subband.c
  src/ratectrl/fixed.c
  src/ratectrl/lagrangian.c
  src/metrics/metrics.c
  src/healing/none.c
  src/healing/deblock.c
  src/healing/cdef.c
)

add_library(vc STATIC ${VC_SRC})
target_include_directories(vc PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vc PUBLIC m)

add_executable(bench harness/bench.c)
target_link_libraries(bench PRIVATE vc)

# Round-trip / unit self-test.
add_executable(vc_test tests/test_roundtrip.c)
target_link_libraries(vc_test PRIVATE vc)

add_executable(vc_test_entropy tests/test_entropy.c)
target_link_libraries(vc_test_entropy PRIVATE vc)

# --- Boundary-healing (healing/) tests + bench -----------------------------
add_executable(vc_test_healing tests/test_healing.c)
target_link_libraries(vc_test_healing PRIVATE vc)

add_executable(bench_healing harness/bench_healing.c)
target_link_libraries(bench_healing PRIVATE vc)

# --- Phase-1 transform bake-off (transform/ work; self-contained) ----------
# These targets #include the candidate transform .c files directly with renamed
# symbols, so they do not need the dct_int.c already inside libvc; only metrics +
# libm. Covers dct-int8, dct-int16, dct-float8, dwt-9/7, zfp-lift4.
add_executable(vc_test_transforms tests/test_transforms.c src/metrics/metrics.c)
target_include_directories(vc_test_transforms PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vc_test_transforms PRIVATE m)

add_executable(bench_transforms harness/bench_transforms.c src/metrics/metrics.c)
target_include_directories(bench_transforms PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_transforms PRIVATE m)

# --- Rate-control allocator (ratectrl/) tests + bench ----------------------
# Guarded so an as-yet-uncreated source (in-progress on another branch) doesn't
# break configure for everyone.
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_ratectrl.c)
  add_executable(vc_test_ratectrl tests/test_ratectrl.c)
  target_link_libraries(vc_test_ratectrl PRIVATE vc)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/harness/bench_ratectrl.c)
  add_executable(bench_ratectrl harness/bench_ratectrl.c)
  target_link_libraries(bench_ratectrl PRIVATE vc)
endif()

# --- Phase-1 chunk-model (block-grid) bake-off (src/chunkmodel/) ------------
# Self-contained engine: blockgrid.c #includes dct_int16.c (renamed) and links
# rlgr.c + metrics.c. Does NOT touch codec.c. Runtime-parameterized so one binary
# sweeps the {stencil x traversal x edge x chunk-size x entropy} matrix.
set(VC_CM_SRC
  src/chunkmodel/blockgrid.c
  src/entropy/rlgr.c
  src/metrics/metrics.c
)
add_executable(vc_test_chunkmodel tests/test_chunkmodel.c ${VC_CM_SRC})
target_include_directories(vc_test_chunkmodel PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(vc_test_chunkmodel PRIVATE m)

add_executable(bench_chunkmodel harness/bench_chunkmodel.c ${VC_CM_SRC})
target_include_directories(bench_chunkmodel PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_chunkmodel PRIVATE m)

# --- HF / perceptual distortion-path bake-off (src/quant/ work; self-contained) -
# Includes the won DCT-16 transform + the 16^3 quant matrix directly, links the
# won RLGR + metrics. Does NOT touch codec.c / ratectrl / chunkmodel. Guarded so
# an absent source never breaks configure for others.
if(EXISTS ${CMAKE_SOURCE_DIR}/harness/bench_hfdist.c)
  add_executable(bench_hfdist harness/bench_hfdist.c
                 src/entropy/rlgr.c src/metrics/metrics.c)
  target_include_directories(bench_hfdist PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(bench_hfdist PRIVATE m)
endif()

# Sign-bit modeling bake-off (r2-sign-model). Self-contained; links won RLGR
# for the baseline archive size. Does NOT touch codec.c.
if(EXISTS ${CMAKE_SOURCE_DIR}/harness/bench_signmodel.c)
  add_executable(bench_signmodel harness/bench_signmodel.c src/entropy/rlgr.c)
  target_include_directories(bench_signmodel PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(bench_signmodel PRIVATE m)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_qmatrix16.c)
  add_executable(vc_test_qmatrix16 tests/test_qmatrix16.c src/entropy/rlgr.c)
  target_include_directories(vc_test_qmatrix16 PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(vc_test_qmatrix16 PRIVATE m)
endif()

enable_testing()
add_test(NAME roundtrip COMMAND vc_test)
add_test(NAME ratectrl COMMAND vc_test_ratectrl)
add_test(NAME entropy COMMAND vc_test_entropy)
add_test(NAME transforms COMMAND vc_test_transforms)
add_test(NAME healing COMMAND vc_test_healing)
add_test(NAME chunkmodel COMMAND vc_test_chunkmodel)
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_qmatrix16.c)
  add_test(NAME qmatrix16 COMMAND vc_test_qmatrix16)
endif()
