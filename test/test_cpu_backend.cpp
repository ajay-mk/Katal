// Numbers through the whole frontend, with the CPU reference backend in place
// of cuTENSOR, so no device is needed. Each result is checked against the
// brute-force reference_einsum or an explicit loop.

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "cpu_backend.hpp"
#include "enum_names.hpp"
#include "numeric.hpp"
#include "reference.hpp"
#include "thrown.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <katal/context.hpp>
#include <katal/equation.hpp>
#include <katal/plan.hpp>
#include <katal/tensor_ref.hpp>

namespace {

using namespace katal;
using katal::test::close;
using katal::test::count;
using katal::test::cpu_context;
using katal::test::HostAllocator;
using katal::test::random_values;
using katal::test::reference_einsum;

using Shape = std::vector<std::int64_t>;

template <class T>
TensorRef ref(std::vector<T>& v, const Shape& shape,
              Layout layout = Layout::RowMajor) {
  return make_ref(static_cast<void*>(v.data()), dtype_of<T>, shape, layout);
}

/// `d = alpha * <equation over a and b> + beta * d` through einsum on the CPU
/// backend, against reference_einsum on the same data.
template <class T>
void check_contraction(const char* equation, const Shape& sa, const Shape& sb,
                       const Shape& sd, Layout layout = Layout::RowMajor) {
  CAPTURE(equation, layout);
  const Context context = cpu_context();
  std::vector<T> a = random_values<T>(count(sa), 1);
  std::vector<T> b = random_values<T>(count(sb), 2);
  std::vector<T> got = random_values<T>(count(sd), 3);
  std::vector<T> want = got;
  const T alpha(1.5), beta(-0.5);

  const std::array<TensorRef, 2> in{ref(a, sa, layout), ref(b, sb, layout)};
  reference_einsum<T>(Equation::parse(equation), alpha, in, beta,
                      ref(want, sd, layout));
  einsum(equation, alpha, in[0], in[1], beta, ref(got, sd, layout), {}, {},
         context, HostAllocator{});
  CHECK(close(got, want));
}

/// The one-operand einsum: a permutation, a sum over dropped labels, a
/// trace or a copy.
template <class T>
void check_one_operand(const char* equation, const Shape& sa, const Shape& sd) {
  CAPTURE(equation);
  const Context context = cpu_context();
  std::vector<T> a = random_values<T>(count(sa), 4);
  std::vector<T> got = random_values<T>(count(sd), 5);
  std::vector<T> want = got;
  const T alpha(2), beta(0.25);

  const std::array<TensorRef, 1> in{ref(a, sa)};
  reference_einsum<T>(Equation::parse(equation), alpha, in, beta,
                      ref(want, sd));
  einsum(equation, alpha, in[0], beta, ref(got, sd), {}, {}, context,
         HostAllocator{});
  CHECK(close(got, want));
}

}  // namespace

TEMPLATE_TEST_CASE("the CPU backend contracts as the reference does",
                   "[cpu_backend]", float, double, std::complex<float>,
                   std::complex<double>) {
  using T = TestType;
  SECTION("a matrix product, in either layout") {
    check_contraction<T>("ij,jk->ik", {3, 4}, {4, 5}, {3, 5});
    check_contraction<T>("ij,jk->ik", {3, 4}, {4, 5}, {3, 5},
                         Layout::ColumnMajor);
  }
  SECTION("a batched product with a permuted result") {
    check_contraction<T>("bij,bjk->kib", {2, 3, 4}, {2, 4, 5}, {5, 3, 2});
  }
  SECTION("an outer product") {
    check_contraction<T>("i,j->ij", {3}, {4}, {3, 4});
  }
  SECTION("a full contraction to a scalar") {
    check_contraction<T>("ij,ij->", {3, 4}, {3, 4}, {});
  }
  SECTION("labels in one input alone are summed through the workspace") {
    check_contraction<T>("ijk,jl->l", {2, 3, 4}, {3, 5}, {5});
    check_contraction<T>("ab,bcd->ad", {2, 3}, {3, 4, 5}, {2, 5});
  }
  SECTION("a diagonal input") {
    check_contraction<T>("ii,ik->k", {4, 4}, {4, 3}, {3});
  }
  SECTION("a four-index contraction") {
    check_contraction<T>("ijab,abkl->ijkl", {2, 3, 4, 2}, {4, 2, 3, 2},
                         {2, 3, 3, 2});
  }
}

TEMPLATE_TEST_CASE("the CPU backend runs the one-operand einsum",
                   "[cpu_backend]", float, double, std::complex<float>,
                   std::complex<double>) {
  using T = TestType;
  check_one_operand<T>("ijk->kji", {2, 3, 4}, {4, 3, 2});
  check_one_operand<T>("ijk->j", {2, 3, 4}, {3});
  check_one_operand<T>("ii->", {5, 5}, {});
  check_one_operand<T>("ij->ij", {3, 4}, {3, 4});
}

TEST_CASE("the CPU backend reads a strided view", "[cpu_backend]") {
  // The top-left 3x4 block of a row-major 5x6 buffer, times a dense 4x2.
  const Context context = cpu_context();
  std::vector<double> big = random_values<double>(5 * 6, 6);
  std::vector<double> b = random_values<double>(4 * 2, 7);
  std::vector<double> got(3 * 2, 0.0), want(3 * 2, 0.0);
  const Shape sa{3, 4}, sb{4, 2}, sd{3, 2};
  const std::array<std::int64_t, 2> stride{6, 1};

  TensorRef a = ref(big, sa);
  a.stride = stride;
  const std::array<TensorRef, 2> in{a, ref(b, sb)};
  reference_einsum<double>(Equation::parse("ij,jk->ik"), 1.0, in, 0.0,
                           ref(want, sd));
  einsum("ij,jk->ik", 1.0, in[0], in[1], 0.0, ref(got, sd), {}, {}, context,
         HostAllocator{});
  CHECK(close(got, want));
}

TEMPLATE_TEST_CASE("the CPU backend reduces with each operator",
                   "[cpu_backend]", float, double) {
  using T = TestType;
  const Context context = cpu_context();
  const Shape sa{3, 4}, sd{3};
  std::vector<T> a = random_values<T>(count(sa), 8);

  struct Case {
    BinaryOp op;
    T (*combine)(T, T);
  };
  const std::array<Case, 4> cases{{
      {BinaryOp::Add, [](T x, T y) { return x + y; }},
      {BinaryOp::Multiply, [](T x, T y) { return x * y; }},
      {BinaryOp::Max, [](T x, T y) { return std::fmax(x, y); }},
      {BinaryOp::Min, [](T x, T y) { return std::fmin(x, y); }},
  }};
  for (const Case& c : cases) {
    CAPTURE(c.op);
    std::vector<T> got{T(1), T(2), T(3)};
    std::vector<T> want(3);
    for (std::size_t i = 0; i < 3; ++i) {
      T r = a[i * 4];
      for (std::size_t j = 1; j < 4; ++j) r = c.combine(r, a[i * 4 + j]);
      // alpha scales the reduced value, then beta * d is added.
      want[i] = T(2) * r + T(0.5) * got[i];
    }
    reduce("ij->i", c.op, T(2), ref(a, sa), T(0.5), ref(got, sd), {}, {},
           context, HostAllocator{});
    CHECK(close(got, want));
  }
}

TEMPLATE_TEST_CASE("the CPU backend combines elementwise", "[cpu_backend]",
                   float, double, std::complex<float>, std::complex<double>) {
  using T = TestType;
  const Context context = cpu_context();
  const Shape sd{3, 4}, sb{4};
  std::vector<T> a = random_values<T>(count(sd), 9);
  std::vector<T> b = random_values<T>(count(sb), 10);
  std::vector<T> got = random_values<T>(count(sd), 11);
  const std::vector<T> before = got;
  const T alpha(2), beta(-1), gamma(0.5);

  SECTION("a sum with b broadcast along i, accumulating into d") {
    std::vector<T> want(got.size());
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 4; ++j)
        want[i * 4 + j] =
            alpha * a[i * 4 + j] + beta * b[j] + gamma * before[i * 4 + j];
    elementwise("ij,j->ij", BinaryOp::Add, alpha, ref(a, sd), beta, ref(b, sb),
                gamma, ref(got, sd), {}, {}, context, HostAllocator{});
    CHECK(close(got, want));
  }

  if constexpr (std::is_floating_point_v<T>) {
    SECTION("the larger and the smaller of the two") {
      std::vector<T> lo(got.size()), want_hi(got.size()), want_lo(got.size());
      for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j) {
          const T x = alpha * a[i * 4 + j], y = beta * b[j];
          want_hi[i * 4 + j] = std::fmax(x, y);
          want_lo[i * 4 + j] = std::fmin(x, y);
        }
      max("ij,j->ij", alpha, ref(a, sd), beta, ref(b, sb), ref(got, sd), {}, {},
          context, HostAllocator{});
      min("ij,j->ij", alpha, ref(a, sd), beta, ref(b, sb), ref(lo, sd), {}, {},
          context, HostAllocator{});
      CHECK(got == want_hi);
      CHECK(lo == want_lo);
    }
  }

  SECTION("a product with a transposed input") {
    // d[i][j] = (alpha * a^T[i][j]) * (beta * b2[i][j]), with a read as 4x3.
    const Shape st{4, 3};
    std::vector<T> b2 = random_values<T>(count(sd), 12);
    std::vector<T> want(got.size());
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 4; ++j)
        want[i * 4 + j] = (alpha * a[j * 3 + i]) * (beta * b2[i * 4 + j]);
    multiply("ji,ij->ij", alpha, ref(a, st), beta, ref(b2, sd), ref(got, sd),
             {}, {}, context, HostAllocator{});
    CHECK(close(got, want));
  }
}

TEST_CASE("the CPU backend permutes, broadcasts and converts",
          "[cpu_backend]") {
  const Context context = cpu_context();

  SECTION("a permutation scaled by alpha") {
    const Shape sa{2, 3, 4}, sd{4, 2, 3};
    std::vector<double> a = random_values<double>(count(sa), 15);
    std::vector<double> got(count(sd)), want(count(sd), 0.0);
    const std::array<TensorRef, 1> in{ref(a, sa)};
    reference_einsum<double>(Equation::parse("ijk->kij"), 3.0, in, 0.0,
                             ref(want, sd));
    permute("ijk->kij", 3.0, in[0], ref(got, sd), {}, {}, context,
            HostAllocator{});
    CHECK(close(got, want));
  }

  SECTION("a broadcast along a new label") {
    const Shape sa{3}, sd{3, 2};
    std::vector<double> a{1, 2, 3}, got(6);
    permute("i->ij", 1.0, ref(a, sa), ref(got, sd), {}, {}, context,
            HostAllocator{});
    CHECK(got == std::vector<double>{1, 1, 2, 2, 3, 3});
  }

  SECTION("F64 to F32") {
    const Shape s{2, 2};
    std::vector<double> a{1.0 / 3, 2.0 / 3, 1e-50, -4.0};
    std::vector<float> got(4);
    permute("ij->ji", 1.0, ref(a, s), ref(got, s), {}, {}, context,
            HostAllocator{});
    CHECK(got ==
          std::vector<float>{float(1.0 / 3), 0.0F, float(2.0 / 3), -4.0F});
  }
}

TEST_CASE("the CPU backend applies unary operators to inputs",
          "[cpu_backend]") {
  const Context context = cpu_context();

  SECTION("a conjugated input in a contraction") {
    using C = std::complex<double>;
    const Shape sa{2, 3}, sb{3, 2}, sd{2, 2};
    std::vector<C> a = random_values<C>(6, 16), b = random_values<C>(6, 17);
    std::vector<C> got(4), want(4);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t k = 0; k < 2; ++k)
        for (std::size_t j = 0; j < 3; ++j)
          want[i * 2 + k] += std::conj(a[i * 3 + j]) * b[j * 2 + k];
    einsum("ij,jk->ik", 1.0, conj(ref(a, sa)), ref(b, sb), 0.0, ref(got, sd),
           {}, {}, context, HostAllocator{});
    CHECK(close(got, want));
  }

  SECTION("an operator on each elementwise input") {
    const Shape s{5};
    std::vector<double> a{0, 1, 4, 9, 16}, b{-2, -1, 0, 1, 2}, got(5);
    add("i,i->i", 1.0, apply(UnaryOp::Sqrt, ref(a, s)), 1.0,
        apply(UnaryOp::Relu, ref(b, s)), ref(got, s), {}, {}, context,
        HostAllocator{});
    CHECK(got == std::vector<double>{0, 1, 2, 4, 6});
  }

  SECTION("the largest magnitude in each row") {
    const Shape sa{2, 3}, sd{2};
    std::vector<double> a{1, -7, 3, -2, 5, 4}, got(2);
    reduce("ij->i", BinaryOp::Max, 1.0, apply(UnaryOp::Abs, ref(a, sa)), 0.0,
           ref(got, sd), {}, {}, context, HostAllocator{});
    CHECK(got == std::vector<double>{7, 5});
  }
}

TEST_CASE("the CPU backend refuses what it cannot compute", "[cpu_backend]") {
  using katal::test::thrown_code;
  const Context context = cpu_context();
  const std::array<std::int64_t, 2> e{2, 2};
  const std::array<std::int64_t, 1> r{2};

  SECTION("half precision") {
    CHECK(thrown_code([&] {
            make_plan(context, Equation::parse("ij->ji"),
                      make_ref(nullptr, Dtype::F16, e),
                      make_ref(nullptr, Dtype::F16, e));
          }) == Errc::UnsupportedPattern);
  }

  SECTION("the maximum of complex values") {
    CHECK(thrown_code([&] {
            make_plan(context, Equation::parse("ij->i"), BinaryOp::Max,
                      make_ref(nullptr, Dtype::C64, e),
                      make_ref(nullptr, Dtype::C64, r));
          }) == Errc::UnsupportedPattern);
  }

  SECTION("a real-only operator on a complex input") {
    CHECK(thrown_code([&] {
            make_plan(context, Equation::parse("ij->ij"),
                      apply(UnaryOp::Relu, make_ref(nullptr, Dtype::C64, e)),
                      make_ref(nullptr, Dtype::C64, e));
          }) == Errc::UnsupportedPattern);
  }

  SECTION("a complex input under a real compute type") {
    PlanOptions options;
    options.compute = Dtype::F64;
    CHECK(thrown_code([&] {
            make_plan(context, Equation::parse("ij->ij"),
                      make_ref(nullptr, Dtype::C64, e),
                      make_ref(nullptr, Dtype::C64, e), options);
          }) == Errc::UnsupportedPattern);
  }
}
