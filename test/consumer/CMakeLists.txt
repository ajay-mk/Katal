# A stand-in for a downstream project. It is built out of tree against an
# installed cuEin, which is the only way to find out whether the install rules
# and the exported config actually work.
cmake_minimum_required(VERSION 3.24)
project(cuein_consumer LANGUAGES CXX)

find_package(cuEin CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE cuEin::cuein)

if(cuEin_HAS_CUTENSOR)
  target_compile_definitions(consumer PRIVATE CONSUMER_HAS_BACKEND)
endif()
