# Examples need a GPU backend to run against, but no data files: every tensor is
# generated in the example itself.
if(NOT KATAL_HAVE_CUTENSOR)
  return()
endif()

foreach(example tensor_storage ccsd_energy ccd_residual)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE Katal::katal CUDA::cudart)
endforeach()
