if(NOT KATAL_HAVE_CUTENSOR)
  return()
endif()

foreach(benchmark grid overhead shapes)
  add_executable(${benchmark} ${benchmark}.cpp)
  target_link_libraries(${benchmark} PRIVATE Katal::katal CUDA::cudart)
  katal_target_warnings(${benchmark})
endforeach()
target_link_libraries(shapes PRIVATE CUDA::cublas)

# Calls cuTENSOR and cuBLAS directly as well as Katal, so it links all three.
add_executable(vs_cutensor vs_cutensor.cpp)
target_link_libraries(vs_cutensor PRIVATE Katal::katal Katal::cutensor
                                          CUDA::cudart CUDA::cublas)
katal_target_warnings(vs_cutensor)
