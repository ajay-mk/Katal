#include <array>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

TEST_CASE("owning tensor views support a GPU contraction", "[tensor][gpu]") {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    SKIP("no CUDA device");

  katal::Tensor<> a(katal::Dtype::F64, {2, 3});
  katal::Tensor<> b(katal::Dtype::F64, {3, 2});
  katal::Tensor<> d(katal::Dtype::F64, {2, 2});
  const double av[]{1, 4, 2, 5, 3, 6};
  const double bv[]{1, 3, 5, 2, 4, 6};
  REQUIRE(cudaMemcpy(a.data(), av, sizeof(av), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(cudaMemcpy(b.data(), bv, sizeof(bv), cudaMemcpyHostToDevice) ==
          cudaSuccess);

  katal::einsum("ab,bc->ac", 1.0, a.ref(), b.ref(), 0.0, d.ref());
  std::array<double, 4> got{};
  REQUIRE(cudaMemcpy(got.data(), d.data(), sizeof(got),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  const std::array<double, 4> expected{22, 49, 28, 64};
  REQUIRE(got == expected);
}
