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

set(VC_SRC
  src/codec.c
  src/transform/dct_int.c
  src/entropy/rice.c
  src/entropy/rlgr.c
  src/entropy/rans_static.c
  src/quant/subband.c
  src/ratectrl/fixed.c
  src/metrics/metrics.c
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

enable_testing()
add_test(NAME roundtrip COMMAND vc_test)
add_test(NAME entropy COMMAND vc_test_entropy)
add_test(NAME transforms COMMAND vc_test_transforms)
