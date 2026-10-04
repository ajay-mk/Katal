#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

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
  // alignment any of them satisfies is 8 bytes.
  constexpr std::size_t n = 32, offset = 1, elements = offset + n * n;
  std::vector<double> a(n * n), b(n * n);
  for (std::size_t i = 0; i < n * n; ++i) {
    a[i] = static_cast<double>(i % 7) - 3;
    b[i] = static_cast<double>(i % 5) - 2;
  }

  double* base = nullptr;
  REQUIRE(cudaMalloc(&base, 3 * elements * sizeof(double)) == cudaSuccess);
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
  REQUIRE(rd.alignment == 8);

  katal::einsum("ij,jk->ik", 1.0, ra, rb, 0.0, rd);
  std::vector<double> got(n * n);
  REQUIRE(cudaMemcpy(got.data(), dd, n * n * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaFree(base) == cudaSuccess);

  // Row-major, the make_ref default. Small integers, so the sums are exact.
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < n; ++k) {
      double expected = 0;
      for (std::size_t j = 0; j < n; ++j)
        expected += a[i * n + j] * b[j * n + k];
      REQUIRE(got[i * n + k] == expected);
    }
}
