#include <new>

#include "cuda_status.hpp"
#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

namespace katal::detail {

void* allocate_device_bytes(std::size_t bytes, StreamRef stream) {
  if (bytes == 0) return nullptr;
  void* data = nullptr;
  const cudaError_t status = cudaMallocAsync(&data, bytes, stream.get());
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();  // reported here, so not left for the caller
    throw std::bad_alloc();
  }
  if (status != cudaSuccess)
    throw Error(Errc::CudaError, {}, cudaGetErrorString(status));
  return data;
}

void deallocate_device_bytes(void* data, StreamRef stream) noexcept {
  if (data != nullptr) util::check_release(cudaFreeAsync(data, stream.get()));
}

void* allocate_managed_bytes(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* data = nullptr;
  const cudaError_t status = cudaMallocManaged(&data, bytes);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();  // reported here, so not left for the caller
    throw std::bad_alloc();
  }
  if (status != cudaSuccess)
    throw Error(Errc::CudaError, {}, cudaGetErrorString(status));
  return data;
}

void deallocate_managed_bytes(void* data, StreamRef stream) noexcept {
  if (data == nullptr) return;
  // Last-owner release can happen while a kernel still borrows this storage.
  // Cleanup cannot throw, and a failed wait must not permit premature freeing.
  if (const cudaError_t synced = cudaStreamSynchronize(stream.get());
      synced != cudaSuccess) {
    util::check_release(synced);
    return;
  }
  util::check_release(cudaFree(data));
}

}  // namespace katal::detail

namespace katal {

// The default workspace allocator is CUDA's, not any one backend's.
Workspace CudaAsyncAllocator::allocate(uint64_t bytes, StreamRef stream) const {
  Workspace workspace;
  workspace.ptr = detail::allocate_device_bytes(bytes, stream);
  workspace.size = bytes;
  return workspace;
}

void CudaAsyncAllocator::deallocate(Workspace workspace,
                                    StreamRef stream) const noexcept {
  detail::deallocate_device_bytes(workspace.ptr, stream);
}

}  // namespace katal
