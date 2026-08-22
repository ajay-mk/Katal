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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDAToolkit QUIET)
find_package(cuTENSOR QUIET)

# The parsing and index layers need no CUDA, so without cuTENSOR the library is
# still usable and testable as headers alone. The cuTENSOR glue is compiled, so
# callers never include <cutensor.h>.
if(cuTENSOR_FOUND AND CUDAToolkit_FOUND)
  set(CUEIN_HAVE_CUTENSOR ON)
  set(cuein_scope PUBLIC)
  add_library(cuein src/context.cpp src/plan.cpp)
  target_link_libraries(cuein PRIVATE cuEin::cutensor CUDA::cudart)
else()
  set(CUEIN_HAVE_CUTENSOR OFF)
  set(cuein_scope INTERFACE)
  add_library(cuein INTERFACE)
  message(STATUS "cuEin: cuTENSOR not found, building the header-only layers")
endif()

add_library(cuEin::cuein ALIAS cuein)
target_include_directories(
  cuein ${cuein_scope} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_features(cuein ${cuein_scope} cxx_std_20)

if(CUEIN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS cuein EXPORT cuEinTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${PROJECT_BINARY_DIR}/include/cuein/config.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cuein)
install(
  EXPORT cuEinTargets
  NAMESPACE cuEin::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuEin)
install(FILES cmake/FindcuTENSOR.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuEin)

configure_package_config_file(
  cmake/cuEinConfig.cmake.in "${PROJECT_BINARY_DIR}/cuEinConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuEin)

# 0.x minor versions are not compatible with one another.
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cuEinConfigVersion.cmake"
  COMPATIBILITY SameMinorVersion)

install(FILES "${PROJECT_BINARY_DIR}/cuEinConfig.cmake"
              "${PROJECT_BINARY_DIR}/cuEinConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuEin)
