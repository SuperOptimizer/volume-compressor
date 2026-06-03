cmake_minimum_required(VERSION 3.20)
project(volume_compressor C)

# Frozen single-pipeline codec: pure C23, libc + libm only, no GPU, no hand
# intrinsics (compiler-autovectorized only). The whole codec is one drop-in
# unit: src/vc/vc.{c,h}. Everything else is tests/benches.
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- SIMD / ISA targeting -------------------------------------------------
# The codec relies on compiler autovectorization (no hand intrinsics). The DCT
# and quant loops vectorize far better with a modern ISA: measured on AVX-512,
# decode ~2.3x and encode ~2.4x vs the x86-64 baseline (SSE2). So target the
# build machine's native ISA BY DEFAULT, but make it configurable so a
# distributor building portable binaries (or cross-compiling) can override.
#   -DVC_MARCH=native        (default) tune for THIS machine — fastest, not portable
#   -DVC_MARCH=x86-64-v3     AVX2+FMA baseline — portable to ~2015+ x86
#   -DVC_MARCH=x86-64-v4     AVX-512 baseline
#   -DVC_MARCH=armv8-a+simd  ARM NEON
#   -DVC_MARCH=              (empty) no -march, pure baseline
# Correctness is identical across ISAs (integer DCT, same results); only speed
# changes. The SOURCE IS ARCH-GENERIC: no intrinsics, no #ifdef per-ISA paths,
# no arch-specific code anywhere — just plain autovectorizable C. AVX2, AVX-512,
# and ARM NEON are all reached from the SAME source via this one compiler flag.
set(VC_MARCH "native" CACHE STRING "target ISA for -march (native|x86-64-v3|x86-64-v4|armv8-a+simd|<empty>)")

# Portable single binary via function multiversioning (target_clones): the hot
# decode kernels are cloned for AVX-512/AVX2/generic and the best is chosen at
# runtime — ONE binary runs optimally on any x86, no -march needed (decode ~80%
# of native; encode runs generic, which is fine since encode is the producer's
# one-time cost on a known machine). Turning this ON also clears -march so the
# non-cloned code stays generic/portable. Default OFF = max-speed native build.
option(VC_PORTABLE "portable multiversioned binary (runtime AVX dispatch, no -march)" OFF)

add_compile_options(-O3 -ffp-contract=fast -ffast-math -funroll-all-loops -Wall -Wextra)
if(VC_PORTABLE)
  add_compile_definitions(VC_PORTABLE)
elseif(VC_MARCH)
  add_compile_options(-march=${VC_MARCH})
endif()

# Link-time optimization: MEASURED +22% decode (lets the compiler inline the
# DCT/dequant/range-decoder across TU boundaries and specialize). On by default;
# the dominant compiler-level win after -march. (PGO / __attribute__((hot)) /
# likely-unlikely were ALL measured to be noise once LTO is on — LTO subsumes
# them — so they're deliberately not used.) Hot DCT buffers are aligned(64) in
# the source for AVX-512 aligned load/store. -fvisibility=hidden trims the
# exported symbol surface to the vc_* public API (hygiene; speed-neutral).
option(VC_LTO "link-time optimization (measured +22% decode)" ON)
if(VC_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _ipo OUTPUT _ipo_msg)
  if(_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  endif()
endif()
add_compile_options(-fvisibility=hidden)

# Coarse multithreading via OpenMP: parallel-for over the independent atom grid
# in decode (and the per-LOD encode). Atoms are independent + thread-local scratch
# → near-linear scaling, the codec stays single-threaded/reentrant per atom. ON if
# OpenMP is found; the #pragma omp lines are no-ops without it (single-thread).
option(VC_OPENMP "coarse multithreading (OpenMP parallel-for over atoms)" ON)
if(VC_OPENMP)
  find_package(OpenMP)
  if(OpenMP_C_FOUND)
    link_libraries(OpenMP::OpenMP_C)
    add_compile_options(${OpenMP_C_FLAGS})
  endif()
endif()

# Optional: emit autovectorization diagnostics (which hot loops vectorized).
option(VC_VEC_REPORT "emit vectorization diagnostics" OFF)
if(VC_VEC_REPORT)
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fopt-info-vec-optimized)
  elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Rpass=loop-vectorize -Rpass-missed=loop-vectorize)
  endif()
endif()

# --- the codec: one unit ---------------------------------------------------
find_package(Threads REQUIRED)
add_library(vc STATIC src/vc/vc.c)
target_include_directories(vc PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(vc PUBLIC m Threads::Threads)
# Test hooks (downsampling helper used to form LOD pyramids in tests).
target_compile_definitions(vc PUBLIC VC_TESTHOOKS)

# Shared library for language bindings (e.g. the Python/ctypes exporter).
add_library(vc_shared SHARED src/vc/vc.c)
set_target_properties(vc_shared PROPERTIES OUTPUT_NAME vc)
target_include_directories(vc_shared PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(vc_shared PUBLIC m Threads::Threads)
target_compile_definitions(vc_shared PUBLIC VC_TESTHOOKS)

# --- tools -----------------------------------------------------------------

# Vendored libs3 (S3 client) + libcurl are REQUIRED: the exporter reads source
# zarr from S3 and writes the archive to S3. Built into the exporter directly.
find_package(CURL REQUIRED)

if(EXISTS ${CMAKE_SOURCE_DIR}/tools/vc_export_stream.c)
  add_executable(vc_export_stream
    tools/vc_export_stream.c
    tools/third_party/cJSON.c
    tools/third_party/libs3/libs3.c)
  target_include_directories(vc_export_stream PRIVATE tools/third_party/libs3)
  target_link_libraries(vc_export_stream PRIVATE vc CURL::libcurl Threads::Threads)
endif()

# --- tests / benches -------------------------------------------------------
enable_testing()
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_vc.c)
  add_executable(vc_test tests/test_vc.c)
  target_link_libraries(vc_test PRIVATE vc)
  add_test(NAME roundtrip COMMAND vc_test)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_sparse.c)
  add_executable(vc_test_sparse tests/test_sparse.c)
  target_link_libraries(vc_test_sparse PRIVATE vc)
  add_test(NAME sparse COMMAND vc_test_sparse)
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_realdata.c)
  add_executable(vc_test_realdata tests/test_realdata.c)
  target_link_libraries(vc_test_realdata PRIVATE vc)
  # real-data test needs the decoded raw volume; gated by VC_TESTDATA env var
  add_test(NAME realdata COMMAND vc_test_realdata)
endif()
