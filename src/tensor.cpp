#include <new>

#include <cuda_runtime.h>

#include <katal/tensor.hpp>

namespace katal::detail {

void* allocate_device_bytes(std::size_t bytes, StreamRef stream) {
  void* data = nullptr;
  if (cudaMallocAsync(&data, bytes, stream.get()) != cudaSuccess)
    throw std::bad_alloc();
  return data;
}

void deallocate_device_bytes(void* data, StreamRef stream) noexcept {
  if (data != nullptr) (void)cudaFreeAsync(data, stream.get());
}

}  // namespace katal::detail
