cmake_minimum_required(VERSION 3.24)
project(
  Katal
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

option(KATAL_BUILD_TESTS "Build Katal tests" ON)
option(KATAL_BUILD_BENCHMARKS "Build GPU timing benchmarks" OFF)
option(KATAL_ENABLE_LOGGING
       "Log Katal cache and planning diagnostics to stderr" OFF)

set(KATAL_MAX_RANK
    12
    CACHE
      STRING
      "Largest tensor rank Katal can describe; the inline capacity of an index")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(KatalWarnings)
find_package(CUDAToolkit QUIET)
find_package(cuTENSOR QUIET)

# enumstr names enums in diagnostics and in test failures. It is header-only and
# private to the compiled library and the tests, so an installed Katal does not
# need it. An installed enumstr is used when found;
# FETCHCONTENT_SOURCE_DIR_ENUMSTR points the fetch at a local copy.
if((cuTENSOR_FOUND AND CUDAToolkit_FOUND) OR KATAL_BUILD_TESTS)
  find_package(enumstr 0.2 CONFIG QUIET)
  if(NOT enumstr_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      enumstr
      GIT_REPOSITORY https://github.com/ajay-mk/enumstr.git
      GIT_TAG v0.2.0
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(enumstr)
  endif()
endif()

# The parsing and index layers need no CUDA, so without cuTENSOR the library is
# still usable and testable as headers alone. The cuTENSOR glue is compiled, so
# callers never include <cutensor.h>.
if(cuTENSOR_FOUND AND CUDAToolkit_FOUND)
  set(KATAL_HAVE_CUTENSOR ON)
  set(katal_scope PUBLIC)
  add_library(katal src/context.cpp src/cutensor_backend.cpp src/plan.cpp
                    src/tensor.cpp)
  target_compile_definitions(
    katal PRIVATE KATAL_ENABLE_LOGGING=$<BOOL:${KATAL_ENABLE_LOGGING}>)
  target_link_libraries(katal PRIVATE Katal::cutensor CUDA::cudart)
  target_link_libraries(katal PRIVATE $<BUILD_INTERFACE:enumstr::enumstr>)
  katal_target_warnings(katal)
else()
  set(KATAL_HAVE_CUTENSOR OFF)
  set(katal_scope INTERFACE)
  add_library(katal INTERFACE)
  message(STATUS "Katal: cuTENSOR not found, building the header-only layers")
endif()

# After the backend is known, so the header can say whether it was built.
configure_file(cmake/config.hpp.in
               "${PROJECT_BINARY_DIR}/include/katal/config.hpp" @ONLY)

add_library(Katal::katal ALIAS katal)
target_include_directories(
  katal ${katal_scope} $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_features(katal ${katal_scope} cxx_std_20)

add_subdirectory(examples)

if(KATAL_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(KATAL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS katal EXPORT KatalTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${PROJECT_BINARY_DIR}/include/katal/config.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/katal)
install(
  EXPORT KatalTargets
  NAMESPACE Katal::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Katal)
install(FILES cmake/FindcuTENSOR.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Katal)

configure_package_config_file(
  cmake/KatalConfig.cmake.in "${PROJECT_BINARY_DIR}/KatalConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Katal)

# 0.x minor versions are not compatible with one another.
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/KatalConfigVersion.cmake"
  COMPATIBILITY SameMinorVersion)

install(FILES "${PROJECT_BINARY_DIR}/KatalConfig.cmake"
              "${PROJECT_BINARY_DIR}/KatalConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Katal)
