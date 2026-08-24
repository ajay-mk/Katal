cmake_minimum_required(VERSION 3.24)
project(
  DeviceEin
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

option(DEVICEEIN_BUILD_TESTS "Build DeviceEin tests" ON)
option(DEVICEEIN_BUILD_BENCHMARKS "Build DeviceEin benchmarks" OFF)
option(DEVICEEIN_ENABLE_JIT "Enable cuTENSOR JIT compilation" OFF)
option(DEVICEEIN_ENABLE_HDF5 "Build the optional HDF5 reader" ON)

set(DEVICEEIN_MAX_RANK
    12
    CACHE
      STRING
      "Largest tensor rank DeviceEin can describe; the inline capacity of an index"
)

configure_file(cmake/config.hpp.in
               "${PROJECT_BINARY_DIR}/include/deviceein/config.hpp" @ONLY)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDAToolkit QUIET)
find_package(cuTENSOR QUIET)

# The parsing and index layers need no CUDA, so without cuTENSOR the library is
# still usable and testable as headers alone. The cuTENSOR glue is compiled, so
# callers never include <cutensor.h>.
if(cuTENSOR_FOUND AND CUDAToolkit_FOUND)
  set(DEVICEEIN_HAVE_CUTENSOR ON)
  set(deviceein_scope PUBLIC)
  add_library(deviceein src/context.cpp src/plan.cpp)
  target_link_libraries(deviceein PRIVATE DeviceEin::cutensor CUDA::cudart)
else()
  set(DEVICEEIN_HAVE_CUTENSOR OFF)
  set(deviceein_scope INTERFACE)
  add_library(deviceein INTERFACE)
  message(
    STATUS "DeviceEin: cuTENSOR not found, building the header-only layers")
endif()

add_library(DeviceEin::deviceein ALIAS deviceein)
target_include_directories(
  deviceein ${deviceein_scope}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_features(deviceein ${deviceein_scope} cxx_std_20)

# Optional and separate from the core library, so a consumer that does not read
# HDF5 never links it. MODULE mode because the config package HDF5 ships names
# its targets differently across versions.
if(DEVICEEIN_ENABLE_HDF5)
  find_package(
    HDF5 MODULE
    COMPONENTS C
    QUIET)
endif()

if(DEVICEEIN_ENABLE_HDF5 AND HDF5_FOUND)
  set(DEVICEEIN_HAVE_HDF5 ON)
  add_library(deviceein_hdf5 src/hdf5.cpp)
  add_library(DeviceEin::hdf5 ALIAS deviceein_hdf5)
  target_link_libraries(
    deviceein_hdf5
    PUBLIC DeviceEin::deviceein
    PRIVATE HDF5::HDF5)
  target_compile_features(deviceein_hdf5 PUBLIC cxx_std_20)
else()
  set(DEVICEEIN_HAVE_HDF5 OFF)
  message(STATUS "DeviceEin: HDF5 not found or disabled, skipping the reader")
endif()

add_subdirectory(examples)

if(DEVICEEIN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS deviceein EXPORT DeviceEinTargets)
if(DEVICEEIN_HAVE_HDF5)
  install(TARGETS deviceein_hdf5 EXPORT DeviceEinTargets)
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${PROJECT_BINARY_DIR}/include/deviceein/config.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/deviceein)
install(
  EXPORT DeviceEinTargets
  NAMESPACE DeviceEin::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceEin)
install(FILES cmake/FindcuTENSOR.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceEin)

configure_package_config_file(
  cmake/DeviceEinConfig.cmake.in "${PROJECT_BINARY_DIR}/DeviceEinConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceEin)

# 0.x minor versions are not compatible with one another.
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/DeviceEinConfigVersion.cmake"
  COMPATIBILITY SameMinorVersion)

install(FILES "${PROJECT_BINARY_DIR}/DeviceEinConfig.cmake"
              "${PROJECT_BINARY_DIR}/DeviceEinConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeviceEin)
