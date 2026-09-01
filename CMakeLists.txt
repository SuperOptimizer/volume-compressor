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
  -mavx2 -mfma # required by volcomp.h (explicit AVX2 kernels)
  $<$<BOOL:${VOLCOMP_NATIVE}>:-march=native>)
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
  target_link_libraries(volcomp_cli PRIVATE volcomp)
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
  target_link_libraries(${name} PRIVATE volcomp)
  add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(${name} PROPERTIES TIMEOUT 300)
endforeach()

# ---- fuzz ----
if(VOLCOMP_FUZZ)
  foreach(tgt d_chunk rt_chunk)
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
