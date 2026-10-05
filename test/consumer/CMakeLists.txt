# A stand-in for a downstream project. It is built out of tree against an
# installed Katal, which is the only way to find out whether the install rules
# and the exported config actually work.
cmake_minimum_required(VERSION 3.24)
project(katal_consumer LANGUAGES CXX)

find_package(Katal CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Katal::katal)

if(Katal_HAS_CUTENSOR)
  target_compile_definitions(consumer PRIVATE CONSUMER_HAS_CUTENSOR)
endif()
