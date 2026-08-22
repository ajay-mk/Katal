# cuTENSOR ships no CMake config package, and on some clusters it has no module
# of its own either -- NVHPC bundles it. Hint with -DCUEIN_CUTENSOR_ROOT=... or
# let $NVHPC point the way.
#
# Defines cuEin::cutensor when found.

set(CUEIN_CUTENSOR_ROOT
    ""
    CACHE PATH "Root of a cuTENSOR installation (contains include/ and lib/)")

set(_cutensor_hints)
foreach(_root "${CUEIN_CUTENSOR_ROOT}" "$ENV{CUTENSOR_ROOT}")
  if(_root)
    list(APPEND _cutensor_hints "${_root}")
  endif()
endforeach()

# NVHPC keeps it under math_libs/<cuda>/targets/<arch>/.
if(DEFINED ENV{NVHPC})
  file(GLOB _nvhpc_math "$ENV{NVHPC}/Linux_*/*/math_libs/*/targets/*")
  # NVHPC ships one tree per CUDA version; take the newest, since an older
  # cuTENSOR against a newer CUDA runtime is asking for trouble.
  list(
    SORT _nvhpc_math
    COMPARE NATURAL
    ORDER DESCENDING)
  list(APPEND _cutensor_hints ${_nvhpc_math})
endif()

find_path(
  cuTENSOR_INCLUDE_DIR cutensor.h
  HINTS ${_cutensor_hints}
  PATH_SUFFIXES include)

find_library(
  cuTENSOR_LIBRARY cutensor
  HINTS ${_cutensor_hints}
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuTENSOR REQUIRED_VARS cuTENSOR_LIBRARY
                                                         cuTENSOR_INCLUDE_DIR)

if(cuTENSOR_FOUND AND NOT TARGET cuEin::cutensor)
  add_library(cuEin::cutensor UNKNOWN IMPORTED)
  set_target_properties(
    cuEin::cutensor
    PROPERTIES IMPORTED_LOCATION "${cuTENSOR_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${cuTENSOR_INCLUDE_DIR}")
endif()

mark_as_advanced(cuTENSOR_INCLUDE_DIR cuTENSOR_LIBRARY)
