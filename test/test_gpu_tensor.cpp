#include <array>
#include <complex>
#include <limits>
#include <new>

#include "enum_names.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

TEST_CASE("GPU allocator size guards", "[tensor][gpu]") {
  katal::PooledDeviceAllocator<double> pooled;
  katal::ManagedAllocator<double> managed;
  const auto too_many = std::numeric_limits<std::size_t>::max();
  REQUIRE_THROWS_AS(pooled.allocate(too_many), std::bad_array_new_length);
  REQUIRE_THROWS_AS(managed.allocate(too_many), std::bad_array_new_length);
  REQUIRE(pooled.allocate(0) == nullptr);
  REQUIRE(managed.allocate(0) == nullptr);
  pooled.deallocate(nullptr, 0);
  managed.deallocate(nullptr, 0);
}

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

TEST_CASE("whole-tensor reductions return to the host", "[tensor][gpu]") {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    SKIP("no CUDA device");

  katal::Tensor<> a(katal::Dtype::F64, {2, 3});
  const double av[]{1, -4, 2, 5, 3, -6};
  REQUIRE(cudaMemcpy(a.data(), av, sizeof(av), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(a.sum<double>() == 1.0);
  REQUIRE(a.max<double>() == 5.0);
  REQUIRE(a.min<double>() == -6.0);
  REQUIRE(katal::test::thrown_code([&] { (void)a.sum<float>(); }) ==
          katal::Errc::DtypeMismatch);

  // One element is its own sum, maximum and minimum.
  katal::Tensor<> one(katal::Dtype::F32, {});
  const float ov = 7;
  REQUIRE(cudaMemcpy(one.data(), &ov, sizeof(ov), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(one.sum<float>() == 7.0f);
  REQUIRE(one.max<float>() == 7.0f);

  katal::Tensor<> z(katal::Dtype::C64, {3});
  const std::complex<double> zv[]{{1, 2}, {3, -1}, {-2, 4}};
  REQUIRE(cudaMemcpy(z.data(), zv, sizeof(zv), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(z.sum<std::complex<double>>() == std::complex<double>(2, 5));
}

TEST_CASE("a tensor's conjugate feeds a contraction", "[tensor][gpu]") {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    SKIP("no CUDA device");

  // d = sum_i conj(z[i]) * z[i] = |z|^2.
  katal::Tensor<> z(katal::Dtype::C64, {2});
  katal::Tensor<> d(katal::Dtype::C64, {});
  const std::complex<double> zv[]{{1, 2}, {3, -1}};
  REQUIRE(cudaMemcpy(z.data(), zv, sizeof(zv), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  katal::einsum("i,i->", 1.0, z.conj(), z, 0.0, d);
  std::complex<double> got;
  REQUIRE(cudaMemcpy(&got, d.data(), sizeof(got), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  REQUIRE(got == std::complex<double>(15, 0));
}
