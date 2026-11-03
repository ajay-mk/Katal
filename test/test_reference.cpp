#include <array>
#include <complex>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include "reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <katal/equation.hpp>
#include <katal/tensor_ref.hpp>

using Catch::Matchers::WithinRel;
using katal::Dtype;
using katal::Equation;
using katal::Layout;
using katal::TensorRef;
using katal::test::reference_einsum;

namespace {

TensorRef ref_of(std::vector<double>& v, std::span<const std::int64_t> extent,
                 Layout layout = Layout::ColumnMajor) {
  return katal::make_ref(v.data(), Dtype::F64, extent, layout);
}

// 1, 2, 3, ... so that a misindexed element shows up as a wrong value rather
// than as a plausible one.
std::vector<double> ramp(std::size_t n) {
  std::vector<double> v(n);
  std::iota(v.begin(), v.end(), 1.0);
  return v;
}

}  // namespace

TEST_CASE("reference matmul agrees with an explicit loop", "[reference]") {
  constexpr std::int64_t m = 2, k = 3, n = 4;
  const std::array<std::int64_t, 2> ae{m, k}, be{k, n}, de{m, n};

  std::vector<double> a = ramp(m * k), b = ramp(k * n), d(m * n, 0.0);
  const TensorRef ar = ref_of(a, ae), br = ref_of(b, be);
  TensorRef dr = ref_of(d, de);

  const std::array<TensorRef, 2> in{ar, br};
  reference_einsum<double>(Equation::parse("ik,kj->ij"), 1.0, in, 0.0, dr);

  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j) {
      double want = 0.0;
      for (std::int64_t p = 0; p < k; ++p) want += a[i + m * p] * b[p + k * j];
      REQUIRE_THAT(d[i + m * j], WithinRel(want, 1e-14));
    }
}

TEST_CASE("reference handles the unary lowerings", "[reference]") {
  const std::array<std::int64_t, 4> shape{2, 3, 4, 5};
  std::vector<double> a = ramp(2 * 3 * 4 * 5);
  const TensorRef ar = ref_of(a, shape);

  SECTION("permutation moves each element exactly once") {
    const std::array<std::int64_t, 4> pe{2, 4, 3, 5};
    std::vector<double> d(a.size(), 0.0);
    TensorRef dr = ref_of(d, pe);
    const std::array<TensorRef, 1> in{ar};
    reference_einsum<double>(Equation::parse("ijkl->ikjl"), 1.0, in, 0.0, dr);

    for (std::int64_t i = 0; i < 2; ++i)
      for (std::int64_t j = 0; j < 3; ++j)
        for (std::int64_t k = 0; k < 4; ++k)
          for (std::int64_t l = 0; l < 5; ++l)
            REQUIRE(d[i + 2 * (k + 4 * (j + 3 * l))] ==
                    a[i + 2 * (j + 3 * (k + 4 * l))]);
  }

  SECTION("reduction sums the dropped modes") {
    const std::array<std::int64_t, 2> re{2, 3};
    std::vector<double> d(2 * 3, 0.0);
    TensorRef dr = ref_of(d, re);
    const std::array<TensorRef, 1> in{ar};
    reference_einsum<double>(Equation::parse("ijkl->ij"), 1.0, in, 0.0, dr);

    for (std::int64_t i = 0; i < 2; ++i)
      for (std::int64_t j = 0; j < 3; ++j) {
        double want = 0.0;
        for (std::int64_t k = 0; k < 4; ++k)
          for (std::int64_t l = 0; l < 5; ++l)
            want += a[i + 2 * (j + 3 * (k + 4 * l))];
        REQUIRE_THAT(d[i + 2 * j], WithinRel(want, 1e-13));
      }
  }

  SECTION("full reduction") {
    std::vector<double> d(1, 0.0);
    TensorRef dr = ref_of(d, {});
    const std::array<TensorRef, 1> in{ar};
    reference_einsum<double>(Equation::parse("ijkl->"), 1.0, in, 0.0, dr);
    REQUIRE_THAT(d[0],
                 WithinRel(std::accumulate(a.begin(), a.end(), 0.0), 1e-13));
  }
}

TEST_CASE("reference honours alpha and beta", "[reference]") {
  const std::array<std::int64_t, 2> e{2, 3};
  std::vector<double> a = ramp(6), d(6, 100.0);
  const TensorRef ar = ref_of(a, e);
  TensorRef dr = ref_of(d, e);
  const std::array<TensorRef, 1> in{ar};

  reference_einsum<double>(Equation::parse("ij->ij"), 2.0, in, 0.5, dr);
  for (std::size_t i = 0; i < d.size(); ++i)
    REQUIRE_THAT(d[i], WithinRel(2.0 * a[i] + 0.5 * 100.0, 1e-14));
}

TEST_CASE("reference agrees across layouts", "[reference]") {
  // The same logical matrices, one pair stored row major and one column
  // major. Same equation, so the answers must match element for element.
  constexpr std::int64_t m = 2, k = 3, n = 4;
  const std::array<std::int64_t, 2> ae{m, k}, be{k, n}, de{m, n};

  std::vector<double> a_row = ramp(m * k), b_row = ramp(k * n);
  std::vector<double> a_col(m * k), b_col(k * n);
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < k; ++j) a_col[j * m + i] = a_row[i * k + j];
  for (std::int64_t i = 0; i < k; ++i)
    for (std::int64_t j = 0; j < n; ++j) b_col[j * k + i] = b_row[i * n + j];

  std::vector<double> d_row(m * n, 0.0), d_col(m * n, 0.0);
  TensorRef dr_row = ref_of(d_row, de, Layout::RowMajor);
  TensorRef dr_col = ref_of(d_col, de);

  const std::array<TensorRef, 2> row{ref_of(a_row, ae, Layout::RowMajor),
                                     ref_of(b_row, be, Layout::RowMajor)};
  const std::array<TensorRef, 2> col{ref_of(a_col, ae), ref_of(b_col, be)};

  const Equation eq = Equation::parse("ik,kj->ij");
  reference_einsum<double>(eq, 1.0, row, 0.0, dr_row);
  reference_einsum<double>(eq, 1.0, col, 0.0, dr_col);

  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j)
      REQUIRE_THAT(d_col[j * m + i], WithinRel(d_row[i * n + j], 1e-14));
}

TEST_CASE("reference works in complex arithmetic", "[reference]") {
  using c64 = std::complex<double>;
  const std::array<std::int64_t, 2> e{2, 2};
  std::vector<c64> a{{1, 1}, {2, 0}, {0, 3}, {1, -1}};
  std::vector<c64> b{{1, 0}, {0, 1}, {2, 2}, {1, 1}};
  std::vector<c64> d(4, c64{});

  const TensorRef ar = katal::make_ref(a.data(), Dtype::C64, e);
  const TensorRef br = katal::make_ref(b.data(), Dtype::C64, e);
  TensorRef dr = katal::make_ref(d.data(), Dtype::C64, e);

  const std::array<TensorRef, 2> in{ar, br};
  reference_einsum<c64>(Equation::parse("ik,kj->ij"), c64{1, 0}, in, c64{}, dr);

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      c64 want{};
      for (int p = 0; p < 2; ++p) want += a[i + 2 * p] * b[p + 2 * j];
      REQUIRE_THAT(d[i + 2 * j].real(), WithinRel(want.real(), 1e-14));
      REQUIRE_THAT(d[i + 2 * j].imag(), WithinRel(want.imag(), 1e-14));
    }
}

TEST_CASE("reference rejects inconsistent extents", "[reference]") {
  const std::array<std::int64_t, 2> ae{2, 3}, be{4, 5}, de{2, 5};
  std::vector<double> a = ramp(6), b = ramp(20), d(10, 0.0);
  TensorRef dr = ref_of(d, de);
  const std::array<TensorRef, 2> in{ref_of(a, ae), ref_of(b, be)};

  REQUIRE_THROWS_AS(
      reference_einsum<double>(Equation::parse("ik,kj->ij"), 1.0, in, 0.0, dr),
      katal::Error);
}
