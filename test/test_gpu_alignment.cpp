#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cuda_memory.hpp"
#include "enum_names.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

// make_ref used to assert 256-byte alignment for every pointer. A sub-block
// such as `base + offset` then let cuTENSOR pick a vectorized kernel the
// address cannot support, which gives wrong answers or a misaligned access.
TEST_CASE("a contraction on offset pointers is correct", "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");

  // Each operand starts one element past a 256-byte boundary, so the largest
  // alignment any of them satisfies is 8 bytes. `elements` is a whole number
  // of 256-byte blocks, which keeps every operand at the same offset.
  constexpr std::size_t n = 32, offset = 1;
  constexpr std::size_t block = 256 / sizeof(double);
  constexpr std::size_t elements = (offset + n * n + block - 1) / block * block;
  std::vector<double> a(n * n), b(n * n);
  for (std::size_t i = 0; i < n * n; ++i) {
    a[i] = static_cast<double>(i % 7) - 3;
    b[i] = static_cast<double>(i % 5) - 2;
  }

  double* base = nullptr;
  REQUIRE(cudaMalloc(&base, 3 * elements * sizeof(double)) == cudaSuccess);
  const std::unique_ptr<double, katal::test::DeviceFree> storage(base);
  double* const da = base + offset;
  double* const db = base + elements + offset;
  double* const dd = base + 2 * elements + offset;
  REQUIRE(cudaMemcpy(da, a.data(), n * n * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  REQUIRE(cudaMemcpy(db, b.data(), n * n * sizeof(double),
                     cudaMemcpyHostToDevice) == cudaSuccess);

  const std::array<std::int64_t, 2> extent{n, n};
  const katal::ConstTensorRef ra = katal::make_ref(
      static_cast<const double*>(da), katal::Dtype::F64, extent);
  const katal::ConstTensorRef rb = katal::make_ref(
      static_cast<const double*>(db), katal::Dtype::F64, extent);
  const katal::TensorRef rd = katal::make_ref(dd, katal::Dtype::F64, extent);
  REQUIRE(ra.alignment == 8);
  REQUIRE(rb.alignment == 8);
  REQUIRE(rd.alignment == 8);

  katal::einsum("ij,jk->ik", 1.0, ra, rb, 0.0, rd);
  std::vector<double> got(n * n);
  REQUIRE(cudaMemcpy(got.data(), dd, n * n * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);

  // Row-major, the make_ref default. Small integers, so the sums are exact.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k) {
      double expected = 0;
      for (std::size_t j = 0; j < n; ++j)
        expected += a[i * n + j] * b[j * n + k];
      REQUIRE(got[i * n + k] == expected);
    }
}

TEST_CASE("a plan refuses a pointer less aligned than it was built for",
          "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");

  // Planned from shapes alone, so it assumes 256-byte alignment.
  const std::array<std::int64_t, 2> extent{4, 4};
  const auto t = katal::make_ref(nullptr, katal::Dtype::F64, extent);
  const katal::Plan plan = katal::make_plan(
      katal::default_context(), katal::Equation::parse("ij,jk->ik"), t, t, t);

  // The check runs before anything is queued, so no memory is touched.
  const auto* misaligned = reinterpret_cast<const void*>(std::uintptr_t{8});
  try {
    plan.execute(1.0, misaligned, nullptr, 0.0, nullptr, nullptr, {});
    FAIL("expected an alignment rejection");
  } catch (const katal::Error& e) {
    REQUIRE(e.code() == katal::Errc::UnsupportedPattern);
  }
}
