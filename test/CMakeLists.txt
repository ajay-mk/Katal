# Prefer an installed Catch2 so an offline machine can build against a local
# copy; set FETCHCONTENT_SOURCE_DIR_CATCH2 to point at one.
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()
include(Catch)

# Every test_*.cpp here becomes an executable; catch_discover_tests then
# registers each TEST_CASE inside it as its own CTest case. Adding a test file
# is the whole registration step.
file(GLOB CUEIN_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

# The GPU tests need a device to run on, not just cuTENSOR to link against.
if(NOT CUEIN_HAVE_CUTENSOR)
  list(FILTER CUEIN_TEST_SOURCES EXCLUDE REGEX "test_gpu")
endif()

foreach(src IN LISTS CUEIN_TEST_SOURCES)
  cmake_path(GET src STEM name)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE cuEin::cuein Catch2::Catch2WithMain)
  if(name MATCHES "test_gpu")
    target_link_libraries(${name} PRIVATE CUDA::cudart)
  endif()
  catch_discover_tests(${name})
endforeach()
