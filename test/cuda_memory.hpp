#ifndef KATAL_TEST_CUDA_MEMORY_HPP_
#define KATAL_TEST_CUDA_MEMORY_HPP_

#include <cstdio>
#include <exception>

#include <cuda_runtime.h>

namespace katal::test {

// A destructor cannot throw; a bad free must still fail the test process.
inline void free_device(void* data) noexcept {
  if (data == nullptr) return;
  if (const auto status = cudaFree(data); status != cudaSuccess) {
    std::fprintf(stderr, "cudaFree: %s\n", cudaGetErrorString(status));
    std::terminate();
  }
}

struct DeviceFree {
  void operator()(void* data) const noexcept { free_device(data); }
};

}  // namespace katal::test

#endif  // KATAL_TEST_CUDA_MEMORY_HPP_
