#include <exception>
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

void* allocate_managed_bytes(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* data = nullptr;
  const cudaError_t status = cudaMallocManaged(&data, bytes);
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw Error(Errc::CudaError, {}, cudaGetErrorString(status));
  return data;
}

void deallocate_managed_bytes(void* data, StreamRef stream) noexcept {
  if (data == nullptr) return;
  // Last-owner release can happen while a kernel still borrows this storage.
  // Cleanup cannot throw, and a failed wait must not permit premature freeing.
  if (cudaStreamSynchronize(stream.get()) != cudaSuccess) std::terminate();
  if (cudaFree(data) != cudaSuccess) std::terminate();
}

}  // namespace katal::detail
