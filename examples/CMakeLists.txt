# Examples need a GPU backend to run against, but no data files: every tensor is
# generated in the example itself.
if(NOT DEVICEEIN_HAVE_CUTENSOR)
  return()
endif()

foreach(example ccsd_energy ccd_residual ccsd_residual)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE DeviceEin::deviceein CUDA::cudart)
endforeach()
