cmake_minimum_required(VERSION 3.24)
project(
  cuEin
  VERSION 0.1.0
  LANGUAGES CXX)

# clang-tidy reads build/compile_commands.json; the pre-commit hook needs it.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Unlike /usr/bin/c++, clang-tidy and clangd do not know Apple's default SDK.
# Spelling the sysroot out changes nothing about the build and lets them find
# the standard headers.
if(APPLE AND NOT CMAKE_OSX_SYSROOT)
  execute_process(
    COMMAND xcrun --show-sdk-path
    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

option(CUEIN_BUILD_TESTS "Build cuEin tests" ON)
option(CUEIN_BUILD_BENCHMARKS "Build cuEin benchmarks" OFF)
option(CUEIN_ENABLE_JIT "Enable cuTENSOR JIT compilation" OFF)

set(CUEIN_MAX_RANK
    12
    CACHE
      STRING
      "Largest tensor rank cuEin can describe; the inline capacity of an index")

configure_file(cmake/config.hpp.in
               "${PROJECT_BINARY_DIR}/include/cuein/config.hpp" @ONLY)

# Header-only: the parsing and index layers need no CUDA. The cuTENSOR glue is
# compiled, so callers never include <cutensor.h>.
add_library(cuein INTERFACE)
add_library(cuEin::cuein ALIAS cuein)
target_include_directories(
  cuein
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>)
target_compile_features(cuein INTERFACE cxx_std_20)

if(CUEIN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
