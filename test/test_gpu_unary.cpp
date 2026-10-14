#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "enum_names.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

bool have_gpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<T>& host) : size_(host.size()) {
    REQUIRE(cudaMalloc(&ptr_, bytes()) == cudaSuccess);
    REQUIRE(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  ~DeviceBuffer() { cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const noexcept { return ptr_; }

  std::vector<T> to_host() const {
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<T> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return host;
  }

 private:
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }

  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

using katal::Dtype;

}  // namespace

TEST_CASE("a conjugated input in a contraction", "[unary][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using C = std::complex<double>;

  // d[i][k] = sum_j conj(a[i][j]) * b[j][k], row-major 2x3 by 3x2.
  const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 2}, ed{2, 2};
  const std::vector<C> a{{1, 1}, {2, -1}, {0, 3}, {1, 0}, {-2, 2}, {1, 1}};
  const std::vector<C> b{{1, 0}, {0, 1}, {2, 1}, {1, -1}, {0, 2}, {3, 0}};
  const DeviceBuffer<C> da(a), db(b), dd(std::vector<C>(4));

  katal::einsum("ij,jk->ik", 1.0,
                katal::conj(katal::make_ref(da.get(), Dtype::C64, ea)),
                katal::make_ref(db.get(), Dtype::C64, eb), 0.0,
                katal::make_ref(dd.get(), Dtype::C64, ed));
  const std::vector<C> got = dd.to_host();
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 2; ++k) {
      C want{};
      for (int j = 0; j < 3; ++j)
        want += std::conj(a[i * 3 + j]) * b[j * 2 + k];
      REQUIRE(got[i * 2 + k] == want);
    }
}

TEST_CASE("an operator on an elementwise input", "[unary][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // d = sqrt(a) + b over perfect squares, so the result is exact.
  const std::array<std::int64_t, 1> e{5};
  const std::vector<float> a{0, 1, 4, 9, 16}, b{10, 20, 30, 40, 50};
  const DeviceBuffer<float> da(a), db(b), dd(std::vector<float>(5));

  katal::add("i,i->i", 1.0f,
             katal::apply(katal::UnaryOp::Sqrt,
                          katal::make_ref(da.get(), Dtype::F32, e)),
             1.0f, katal::make_ref(db.get(), Dtype::F32, e),
             katal::make_ref(dd.get(), Dtype::F32, e));
  const std::vector<float> got = dd.to_host();
  for (std::size_t i = 0; i < a.size(); ++i)
    REQUIRE(got[i] == std::sqrt(a[i]) + b[i]);
}

TEST_CASE("reductions other than a sum", "[unary][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // Row-major 3x4, reduced over j.
  const std::array<std::int64_t, 2> ea{3, 4};
  const std::array<std::int64_t, 1> ed{3};
  const std::vector<double> a{3, -1, 4, 1, -5, 9, 2, 6, 5, 3, -5, 8};
  const DeviceBuffer<double> da(a);
  const katal::ConstTensorRef ar = katal::make_ref(da.get(), Dtype::F64, ea);

  struct Case {
    katal::BinaryOp op;
    double (*combine)(double, double);
  };
  const std::array<Case, 3> cases{{
      {katal::BinaryOp::Max,
       [](double x, double y) { return std::fmax(x, y); }},
      {katal::BinaryOp::Min,
       [](double x, double y) { return std::fmin(x, y); }},
      {katal::BinaryOp::Multiply, [](double x, double y) { return x * y; }},
  }};
  for (const Case& c : cases) {
    const DeviceBuffer<double> dd(std::vector<double>(3, 0.0));
    katal::reduce("ij->i", c.op, 1.0, ar, 0.0,
                  katal::make_ref(dd.get(), Dtype::F64, ed));
    const std::vector<double> got = dd.to_host();
    for (int i = 0; i < 3; ++i) {
      double want = a[i * 4];
      for (int j = 1; j < 4; ++j) want = c.combine(want, a[i * 4 + j]);
      REQUIRE(got[i] == want);
    }
  }

  SECTION("with an operator on the input: the largest magnitude per row") {
    const DeviceBuffer<double> dd(std::vector<double>(3, 0.0));
    katal::reduce("ij->i", katal::BinaryOp::Max, 1.0,
                  katal::apply(katal::UnaryOp::Abs, ar), 0.0,
                  katal::make_ref(dd.get(), Dtype::F64, ed));
    REQUIRE(dd.to_host() == std::vector<double>{4, 9, 8});
  }
}

TEST_CASE("a dedicated permutation", "[unary][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 2> e23{2, 3}, e32{3, 2};
  const std::vector<double> a{1, 2, 3, 4, 5, 6};
  const DeviceBuffer<double> da(a);
  const katal::ConstTensorRef ar = katal::make_ref(da.get(), Dtype::F64, e23);

  SECTION("a transpose, scaled and negated") {
    const DeviceBuffer<double> dd(std::vector<double>(6));
    katal::permute("ij->ji", 2.0, katal::apply(katal::UnaryOp::Neg, ar),
                   katal::make_ref(dd.get(), Dtype::F64, e32));
    REQUIRE(dd.to_host() == std::vector<double>{-2, -8, -4, -10, -6, -12});
  }

  SECTION("a broadcast along a new label") {
    const std::array<std::int64_t, 1> e3{3};
    const DeviceBuffer<double> dv(std::vector<double>{7, 8, 9});
    const DeviceBuffer<double> dd(std::vector<double>(6));
    katal::permute("j->ij", 1.0, katal::make_ref(dv.get(), Dtype::F64, e3),
                   katal::make_ref(dd.get(), Dtype::F64, e23));
    REQUIRE(dd.to_host() == std::vector<double>{7, 8, 9, 7, 8, 9});
  }

  SECTION("a conversion to a narrower type") {
    const DeviceBuffer<float> dd(std::vector<float>(6));
    katal::permute("ij->ji", 1.0, ar,
                   katal::make_ref(dd.get(), Dtype::F32, e32));
    REQUIRE(dd.to_host() == std::vector<float>{1, 4, 2, 5, 3, 6});
  }

  SECTION("a conversion to a wider type") {
    const DeviceBuffer<float> df(std::vector<float>{1, 2, 3, 4, 5, 6});
    const DeviceBuffer<double> dd(std::vector<double>(6));
    katal::permute("ij->ji", 1.0, katal::make_ref(df.get(), Dtype::F32, e23),
                   katal::make_ref(dd.get(), Dtype::F64, e32));
    REQUIRE(dd.to_host() == std::vector<double>{1, 4, 2, 5, 3, 6});
  }
}

TEST_CASE("unary and permutation misuse", "[unary][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using katal::test::thrown_code;

  const std::array<std::int64_t, 2> e{2, 2};
  const DeviceBuffer<double> da(std::vector<double>(4, 1.0));
  const DeviceBuffer<double> dd(std::vector<double>(4, 0.0));
  const katal::TensorRef a = katal::make_ref(da.get(), Dtype::F64, e);
  const katal::TensorRef d = katal::make_ref(dd.get(), Dtype::F64, e);
  const katal::Context& context = katal::default_context();

  SECTION("an operator on the output") {
    katal::TensorRef marked = d;
    marked.op = katal::UnaryOp::Abs;
    REQUIRE(thrown_code([&] {
              katal::make_plan(context, katal::Equation::parse("ij->ji"), a,
                               marked);
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("a permutation that would sum") {
    const std::array<std::int64_t, 1> e1{2};
    REQUIRE(thrown_code([&] {
              katal::make_permutation_plan(
                  context, katal::Equation::parse("ij->i"), a,
                  katal::make_ref(dd.get(), Dtype::F64, e1));
            }) == katal::Errc::UnsupportedPattern);
  }

  const katal::Plan plan = katal::make_permutation_plan(
      context, katal::Equation::parse("ij->ji"), a, d);

  SECTION("a permutation with beta") {
    REQUIRE(thrown_code([&] {
              plan.execute(1.0, da.get(), 1.0, dd.get(), dd.get(), {});
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("a permutation in place") {
    REQUIRE(thrown_code([&] {
              plan.execute(1.0, dd.get(), 0.0, dd.get(), dd.get(), {});
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("the cache tells operators, reductions and permutations apart") {
    const auto sum = katal::cached_plan(context, "ij->ji", a, d);
    const auto max =
        katal::cached_plan(context, "ij->ji", katal::BinaryOp::Max, a, d);
    const auto perm = katal::cached_permutation_plan(context, "ij->ji", a, d);
    const auto neg = katal::cached_plan(
        context, "ij->ji", katal::apply(katal::UnaryOp::Neg, a), d);
    REQUIRE(sum != max);
    REQUIRE(sum != perm);
    REQUIRE(sum != neg);
    REQUIRE(perm == katal::cached_permutation_plan(context, "ij->ji", a, d));
  }
}
