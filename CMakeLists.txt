cmake_minimum_required(VERSION 3.28)
project(volcomp VERSION 1.0.0 LANGUAGES C)

# volcomp is a single header (volcomp.h). This build produces only the tests,
# the CLI, the benchmark and the fuzzers that consume it.

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_definitions(_GNU_SOURCE)

if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(WARNING "volcomp targets clang; got ${CMAKE_C_COMPILER_ID}")
endif()

option(VOLCOMP_WERROR "Treat warnings as errors" ON)
option(VOLCOMP_NATIVE "Tune for the host CPU (-march=native)" OFF)
option(VOLCOMP_STATIC_AVX2 "Compile with -mavx2 -mfma so the AVX2 kernels are called directly (no runtime dispatch; binary needs AVX2)" OFF)
option(VOLCOMP_NO_AVX2 "Build the C kernels only (define VOLCOMP_NO_AVX2); for testing the portable path" OFF)
option(VOLCOMP_BENCH "Build volcomp-bench" OFF)
option(VOLCOMP_FUZZ "Build libFuzzer targets" OFF)
set(VOLCOMP_SANITIZE "" CACHE STRING "Sanitizer set: '' | asan-ubsan")
set_property(CACHE VOLCOMP_SANITIZE PROPERTY STRINGS "" asan-ubsan)

add_library(volcomp INTERFACE)
target_include_directories(volcomp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(volcomp INTERFACE m)
target_compile_options(volcomp INTERFACE
  -Wall -Wextra -Wconversion -Wshadow -Wimplicit-fallthrough -Wformat=2 -Wvla -Wdouble-promotion
  $<$<BOOL:${VOLCOMP_WERROR}>:-Werror>
  # volcomp.h needs no arch flags: on x86-64 it compiles its AVX2 kernels with a
  # target attribute and picks them at runtime when the CPU has AVX2+FMA.
  $<$<BOOL:${VOLCOMP_STATIC_AVX2}>:-mavx2>
  $<$<BOOL:${VOLCOMP_STATIC_AVX2}>:-mfma>
  $<$<BOOL:${VOLCOMP_NATIVE}>:-march=native>)
target_compile_definitions(volcomp INTERFACE $<$<BOOL:${VOLCOMP_NO_AVX2}>:VOLCOMP_NO_AVX2>)
if(VOLCOMP_SANITIZE STREQUAL "asan-ubsan")
  target_compile_options(volcomp INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(volcomp INTERFACE -fsanitize=address,undefined)
elseif(NOT VOLCOMP_SANITIZE STREQUAL "")
  message(FATAL_ERROR "Unknown VOLCOMP_SANITIZE value: ${VOLCOMP_SANITIZE}")
endif()

# ---- CLI ----
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/cli/volcomp.c)
  add_executable(volcomp_cli tools/cli/volcomp.c tools/bench/metrics.c)
  set_target_properties(volcomp_cli PROPERTIES OUTPUT_NAME volcomp)
  target_include_directories(volcomp_cli PRIVATE tools/bench)
  target_link_libraries(volcomp_cli PRIVATE volcomp volcomp_surface_deps)
endif()

# ---- surface-pack dependencies (libzstd for the blosc1 reader, pthreads) ----
# Only `volcomp surface-pack` needs them; the rest of the CLI stays dependency-free.
find_package(Threads REQUIRED)
find_library(ZSTD_LIBRARY zstd)
find_path(ZSTD_INCLUDE_DIR zstd.h)
add_library(volcomp_surface_deps INTERFACE)
target_link_libraries(volcomp_surface_deps INTERFACE Threads::Threads)
if(ZSTD_LIBRARY AND ZSTD_INCLUDE_DIR)
  target_link_libraries(volcomp_surface_deps INTERFACE ${ZSTD_LIBRARY})
  target_include_directories(volcomp_surface_deps INTERFACE ${ZSTD_INCLUDE_DIR})
  target_compile_definitions(volcomp_surface_deps INTERFACE VOLCOMP_HAVE_ZSTD=1)
else()
  message(WARNING "libzstd not found: `volcomp surface-pack` will refuse blosc/zstd chunks")
endif()

# ---- bench ----
if(VOLCOMP_BENCH)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/bench/main.c)
    add_executable(volcomp-bench tools/bench/main.c tools/bench/corpus.c tools/bench/metrics.c)
    target_link_libraries(volcomp-bench PRIVATE volcomp)
    find_library(ZSTD_LIB zstd)
    if(ZSTD_LIB)
      target_link_libraries(volcomp-bench PRIVATE ${ZSTD_LIB})
      target_compile_definitions(volcomp-bench PRIVATE VOLCOMP_HAVE_ZSTD=1)
    endif()
  endif()
endif()

# ---- tests ----
enable_testing()
file(GLOB VOLCOMP_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.c)
foreach(src ${VOLCOMP_TESTS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src} tools/bench/metrics.c)
  target_include_directories(${name} PRIVATE tools/bench tools/cli)
  target_link_libraries(${name} PRIVATE volcomp volcomp_surface_deps)
  add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # test_lossless encodes and decodes 1000 chunks; that is slow under sanitizers
  set_tests_properties(${name} PROPERTIES TIMEOUT 1200)
endforeach()

# C-kernel variants of the golden, codec and surface-chunk tests (VOLCOMP_NO_AVX2), so a
# normal x86-64 build also exercises the portable path.
foreach(name test_golden test_codec test_surfchunk)
  add_executable(${name}_c tests/${name}.c tools/bench/metrics.c)
  target_include_directories(${name}_c PRIVATE tools/bench tools/cli)
  target_link_libraries(${name}_c PRIVATE volcomp volcomp_surface_deps)
  target_compile_definitions(${name}_c PRIVATE VOLCOMP_NO_AVX2)
  add_test(NAME ${name}_c COMMAND ${name}_c WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(${name}_c PROPERTIES TIMEOUT 300)
endforeach()

# ---- fuzz ----
if(VOLCOMP_FUZZ)
  foreach(tgt d_chunk rt_chunk rt_mask rt_mask_ll)
    add_executable(fuzz_${tgt} fuzz/${tgt}.c)
    target_link_libraries(fuzz_${tgt} PRIVATE volcomp)
    target_compile_options(fuzz_${tgt} PRIVATE -fsanitize=fuzzer,address,undefined -g -O1)
    target_link_options(fuzz_${tgt} PRIVATE -fsanitize=fuzzer,address,undefined)
  endforeach()
endif()

# shared library with exported entry points for ctypes / FFI (python/volcomp_zarr)
add_library(volcomp_shim SHARED python/volcomp_shim.c)
target_link_libraries(volcomp_shim PRIVATE volcomp)
set_target_properties(volcomp_shim PROPERTIES OUTPUT_NAME volcomp C_VISIBILITY_PRESET hidden)
