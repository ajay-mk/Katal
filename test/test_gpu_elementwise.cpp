#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "enum_names.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

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

/// Distinct small integers, so every result below is exact in f32 and f64.
template <class T>
std::vector<T> iota(std::size_t n, T start = 1) {
  std::vector<T> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = start + static_cast<T>(i);
  return v;
}

/// Runs a plan built from the equation and op on the buffers, for an
/// elementwise result checked element by element.
template <class T>
std::vector<T> run(std::string_view equation, katal::BinaryOp op, double alpha,
                   const DeviceBuffer<T>& a, std::span<const std::int64_t> ea,
                   double beta, const DeviceBuffer<T>& b,
                   std::span<const std::int64_t> eb, double gamma,
                   const DeviceBuffer<T>& d, std::span<const std::int64_t> ed,
                   katal::Dtype type) {
  const katal::Plan plan = katal::make_plan(
      katal::default_context(), katal::Equation::parse(equation), op,
      katal::make_ref(a.get(), type, ea), katal::make_ref(b.get(), type, eb),
      katal::make_ref(d.get(), type, ed));
  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size(), {});
  plan.execute(alpha, a.get(), beta, b.get(), gamma, d.get(), d.get(),
               scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  return d.to_host();
}

}  // namespace

TEST_CASE("elementwise add permutes one input against the other",
          "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // d[i][j][a] = t[j][i][a] - t[i][j][a], row-major 3x3x4.
  constexpr std::int64_t n = 3, v = 4;
  const std::array<std::int64_t, 3> e{n, n, v};
  const std::vector<double> t = iota<double>(n * n * v);
  const DeviceBuffer<double> dt(t);
  const DeviceBuffer<double> dd(std::vector<double>(t.size(), 0.0));

  const std::vector<double> got =
      run<double>("jia,ija->ija", katal::BinaryOp::Add, 1.0, dt, e, -1.0, dt, e,
                  0.0, dd, e, katal::Dtype::F64);
  for (std::int64_t i = 0; i < n; ++i)
    for (std::int64_t j = 0; j < n; ++j)
      for (std::int64_t a = 0; a < v; ++a)
        REQUIRE(got[(i * n + j) * v + a] ==
                t[(j * n + i) * v + a] - t[(i * n + j) * v + a]);
}

TEST_CASE("elementwise broadcast and accumulate", "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // d[i][j] = 2 * a[i][j] + 3 * b[j] + 0.5 * d[i][j].
  constexpr std::int64_t rows = 3, cols = 5;
  const std::array<std::int64_t, 2> ea{rows, cols};
  const std::array<std::int64_t, 1> eb{cols};
  const std::vector<double> a = iota<double>(rows * cols);
  const std::vector<double> b = iota<double>(cols, 10);
  const std::vector<double> d0 = iota<double>(rows * cols, 2);
  const DeviceBuffer<double> da(a), db(b), dd(d0);

  const std::vector<double> got =
      run<double>("ij,j->ij", katal::BinaryOp::Add, 2.0, da, ea, 3.0, db, eb,
                  0.5, dd, ea, katal::Dtype::F64);
  for (std::int64_t i = 0; i < rows; ++i)
    for (std::int64_t j = 0; j < cols; ++j)
      REQUIRE(got[i * cols + j] ==
              2.0 * a[i * cols + j] + 3.0 * b[j] + 0.5 * d0[i * cols + j]);
}

TEST_CASE("an outer sum broadcasts both inputs into a separate output",
          "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // d[i][j] = a[i] + 2 * b[j] + c[i][j], with c and d distinct.
  constexpr std::int64_t rows = 3, cols = 4;
  const std::array<std::int64_t, 1> ea{rows}, eb{cols};
  const std::array<std::int64_t, 2> ed{rows, cols};
  const std::vector<double> a = iota<double>(rows);
  const std::vector<double> b = iota<double>(cols, 10);
  const std::vector<double> c = iota<double>(rows * cols, 100);
  const DeviceBuffer<double> da(a), db(b), dc(c);
  const DeviceBuffer<double> dd(std::vector<double>(c.size(), 0.0));

  const katal::Plan plan = katal::make_plan(
      katal::default_context(), katal::Equation::parse("i,j->ij"),
      katal::BinaryOp::Add, katal::make_ref(da.get(), katal::Dtype::F64, ea),
      katal::make_ref(db.get(), katal::Dtype::F64, eb),
      katal::make_ref(dd.get(), katal::Dtype::F64, ed));
  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size(), {});
  plan.execute(1.0, da.get(), 2.0, db.get(), 1.0, dc.get(), dd.get(),
               scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<double> got = dd.to_host();
  for (std::int64_t i = 0; i < rows; ++i)
    for (std::int64_t j = 0; j < cols; ++j)
      REQUIRE(got[i * cols + j] == a[i] + 2.0 * b[j] + c[i * cols + j]);
  REQUIRE(dc.to_host() == c);
}

TEST_CASE("each binary operator", "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  constexpr std::int64_t n = 6;
  const std::array<std::int64_t, 1> e{n};
  // Crossing values, so max and min each pick from both inputs.
  const std::vector<float> a{1, 5, 2, 8, 3, 0};
  const std::vector<float> b{4, 2, 6, 1, 3, 7};
  const DeviceBuffer<float> da(a), db(b);

  struct Case {
    katal::BinaryOp op;
    float (*expect)(float, float);
  };
  const std::array<Case, 4> cases{{
      {katal::BinaryOp::Add, [](float x, float y) { return x + y; }},
      {katal::BinaryOp::Multiply, [](float x, float y) { return x * y; }},
      {katal::BinaryOp::Max, [](float x, float y) { return std::fmax(x, y); }},
      {katal::BinaryOp::Min, [](float x, float y) { return std::fmin(x, y); }},
  }};
  for (const Case& c : cases) {
    const DeviceBuffer<float> dd(std::vector<float>(n, 0.0f));
    const std::vector<float> got = run<float>(
        "i,i->i", c.op, 2.0, da, e, 1.0, db, e, 0.0, dd, e, katal::Dtype::F32);
    for (std::int64_t i = 0; i < n; ++i)
      REQUIRE(got[i] == c.expect(2.0f * a[i], b[i]));
  }
}

TEST_CASE("a zero gamma does not read the output", "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // A NaN read through gamma = 0 would survive as NaN.
  constexpr std::int64_t n = 8;
  const std::array<std::int64_t, 1> e{n};
  const std::vector<double> a = iota<double>(n), b = iota<double>(n, 3);
  const DeviceBuffer<double> da(a), db(b);
  const DeviceBuffer<double> dd(
      std::vector<double>(n, std::numeric_limits<double>::quiet_NaN()));

  const std::vector<double> got =
      run<double>("i,i->i", katal::BinaryOp::Add, 1.0, da, e, 1.0, db, e, 0.0,
                  dd, e, katal::Dtype::F64);
  for (std::int64_t i = 0; i < n; ++i) REQUIRE(got[i] == a[i] + b[i]);
}

TEST_CASE("the named calls take owning tensors", "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  katal::Tensor<> a(katal::Dtype::F64, {2, 3});
  katal::Tensor<> b(katal::Dtype::F64, {3});
  katal::Tensor<> d(katal::Dtype::F64, {2, 3});
  const std::array<double, 6> av{1, 2, 3, 4, 5, 6};
  const std::array<double, 3> bv{10, 20, 30};
  REQUIRE(cudaMemcpy(a.data(), av.data(), sizeof(av), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  REQUIRE(cudaMemcpy(b.data(), bv.data(), sizeof(bv), cudaMemcpyHostToDevice) ==
          cudaSuccess);

  // Tensor is column-major: element (i, j) of a 2x3 tensor sits at i + 2j.
  std::array<double, 6> d_host{};
  const auto check = [&](auto expect) {
    std::array<double, 6> got{};
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(got.data(), d.data(), sizeof(got),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        REQUIRE(got[i + 2 * j] ==
                expect(av[i + 2 * j], bv[j], d_host[i + 2 * j]));
    d_host = got;
  };

  katal::add("ij,j->ij", 1.0, a, -1.0, b, d);
  check([](double x, double y, double) { return x - y; });

  katal::multiply("ij,j->ij", 1.0, a, 1.0, b, d);
  check([](double x, double y, double) { return x * y; });

  katal::max("ij,j->ij", 10.0, a, 1.0, b, d);
  check([](double x, double y, double) { return std::fmax(10.0 * x, y); });

  katal::min("ij,j->ij", 10.0, a, 1.0, b, d);
  check([](double x, double y, double) { return std::fmin(10.0 * x, y); });

  // gamma = 1 accumulates into what min left behind.
  katal::elementwise("ij,j->ij", katal::BinaryOp::Add, 1.0, a, 0.0, b, 1.0, d);
  check([](double x, double, double old) { return x + old; });
}

TEST_CASE("elementwise plans reject misuse", "[elementwise][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using katal::test::thrown_code;

  const std::array<std::int64_t, 2> e{2, 2};
  const DeviceBuffer<double> da(std::vector<double>(4, 1.0));
  const DeviceBuffer<double> dd(std::vector<double>(4, 0.0));
  const katal::TensorRef a = katal::make_ref(da.get(), katal::Dtype::F64, e);
  const katal::TensorRef d = katal::make_ref(dd.get(), katal::Dtype::F64, e);
  const katal::Context& context = katal::default_context();

  SECTION("an equation that sums over a label") {
    REQUIRE(thrown_code([&] {
              katal::make_plan(context, katal::Equation::parse("ij,jk->ik"),
                               katal::BinaryOp::Add, a, a, d);
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("a label with two extents") {
    const std::array<std::int64_t, 1> three{3};
    const DeviceBuffer<double> db(std::vector<double>(3, 1.0));
    const katal::TensorRef b =
        katal::make_ref(db.get(), katal::Dtype::F64, three);
    REQUIRE(thrown_code([&] {
              katal::make_plan(context, katal::Equation::parse("ij,j->ij"),
                               katal::BinaryOp::Add, a, b, d);
            }) == katal::Errc::ExtentMismatch);
  }

  const katal::Plan plan =
      katal::make_plan(context, katal::Equation::parse("ij,ji->ij"),
                       katal::BinaryOp::Add, a, a, d);

  SECTION("an input that is the output") {
    REQUIRE(thrown_code([&] {
              plan.execute(1.0, dd.get(), 1.0, da.get(), 0.0, dd.get(),
                           dd.get(), {});
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("a zero scale on an operator other than Add") {
    const katal::Plan max =
        katal::make_plan(context, katal::Equation::parse("ij,ji->ij"),
                         katal::BinaryOp::Max, a, a, d);
    REQUIRE(thrown_code([&] {
              max.execute(1.0, da.get(), 0.0, da.get(), 0.0, dd.get(), dd.get(),
                          {});
            }) == katal::Errc::UnsupportedPattern);
    REQUIRE(thrown_code([&] {
              katal::multiply("ij,ji->ij", 0.0, a, 1.0, a, d);
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("the einsum execute overloads") {
    REQUIRE(thrown_code([&] {
              plan.execute(1.0, da.get(), da.get(), 0.0, dd.get(), dd.get(),
                           {});
            }) == katal::Errc::UnsupportedPattern);
    REQUIRE(thrown_code([&] {
              plan.execute(1.0, da.get(), 0.0, dd.get(), dd.get(), {});
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("an einsum plan refuses the elementwise execute") {
    const katal::Plan contraction =
        katal::make_plan(context, katal::Equation::parse("ij,jk->ik"), a, a, d);
    REQUIRE(thrown_code([&] {
              contraction.execute(1.0, da.get(), 1.0, da.get(), 0.0, dd.get(),
                                  dd.get(), {});
            }) == katal::Errc::UnsupportedPattern);
  }

  SECTION("the cache keeps elementwise and contraction plans apart") {
    const auto product = katal::cached_plan(context, "ij,ij->ij", a, a, d);
    const auto sum =
        katal::cached_plan(context, "ij,ij->ij", katal::BinaryOp::Add, a, a, d);
    const auto max =
        katal::cached_plan(context, "ij,ij->ij", katal::BinaryOp::Max, a, a, d);
    REQUIRE(product != sum);
    REQUIRE(sum != max);
    REQUIRE(sum == katal::cached_plan(context, "ij,ij->ij",
                                      katal::BinaryOp::Add, a, a, d));
  }
}
