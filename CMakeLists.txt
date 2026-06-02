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

add_compile_options(-O3 -ffp-contract=fast -ffast-math -Wall -Wextra)
add_compile_definitions(VC_CHUNK_SIDE=${VC_CHUNK_SIDE})

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
enable_testing()
add_test(NAME roundtrip COMMAND vc_test)
