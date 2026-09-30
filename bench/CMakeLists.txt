if(NOT KATAL_HAVE_CUTENSOR)
  return()
endif()

foreach(benchmark grid overhead shapes)
  add_executable(${benchmark} ${benchmark}.cpp)
  target_link_libraries(${benchmark} PRIVATE Katal::katal CUDA::cudart)
endforeach()
target_link_libraries(shapes PRIVATE CUDA::cublas)
