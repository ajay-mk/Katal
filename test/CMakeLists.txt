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
file(GLOB KATAL_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

# The GPU tests need a device to run on, not just cuTENSOR to link against.
if(NOT KATAL_HAVE_CUTENSOR)
  list(FILTER KATAL_TEST_SOURCES EXCLUDE REGEX "test_gpu")
endif()

foreach(src IN LISTS KATAL_TEST_SOURCES)
  cmake_path(GET src STEM name)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE Katal::katal Catch2::Catch2WithMain)
  # The GPU cases SKIP themselves when no device is present, but Catch2's exit
  # code for an all-skipped run does not match what catch_discover_tests sets as
  # SKIP_RETURN_CODE, so CTest calls them failures. Labelling them lets a
  # machine without a device exclude them: ctest --label-exclude gpu.
  if(name MATCHES "test_gpu")
    target_link_libraries(${name} PRIVATE CUDA::cudart)
    catch_discover_tests(${name} PROPERTIES LABELS gpu)
  else()
    catch_discover_tests(${name})
  endif()
endforeach()
