// The same checked matrix product with pooled and managed tensor storage.
#include <array>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

namespace {
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

template <class Allocator>
void matrix_product(const char* label) {
  using Tensor = katal::Tensor<Allocator>;
  Tensor a(katal::Dtype::F64, {2, 3});
  Tensor b(katal::Dtype::F64, {3, 2});
  Tensor d(katal::Dtype::F64, {2, 2});

  // Columns are contiguous. Explicit transfers keep the two paths identical.
  const std::array<double, 6> av{1, 4, 2, 5, 3, 6};
  const std::array<double, 6> bv{1, 3, 5, 2, 4, 6};
  check_cuda(
      cudaMemcpy(a.data(), av.data(), sizeof(av), cudaMemcpyHostToDevice));
  check_cuda(
      cudaMemcpy(b.data(), bv.data(), sizeof(bv), cudaMemcpyHostToDevice));

  katal::einsum("ab,bc->ac", 1.0, a.ref(), b.ref(), 0.0, d.ref());
  check_cuda(cudaStreamSynchronize(nullptr));
  std::array<double, 4> result{};
  check_cuda(cudaMemcpy(result.data(), d.data(), sizeof(result),
                        cudaMemcpyDeviceToHost));
  const std::array<double, 4> expected{22, 49, 28, 64};
  if (result != expected) throw std::runtime_error("matrix product mismatch");
  std::printf("%s: verified [22, 49, 28, 64] (column-major)\n", label);
}
}  // namespace

int main() {
  try {
    const katal::Runtime runtime;
    matrix_product<katal::PooledDeviceAllocator<std::byte>>("pooled");
    matrix_product<katal::ManagedAllocator<std::byte>>("managed");
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
