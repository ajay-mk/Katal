#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

#include "enum_names.hpp"
#include "reference.hpp"
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

/// Small integers, so every sum below is exact in f64.
std::vector<double> ramp(std::size_t n, double start) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = start + static_cast<double>(i % 7);
  return v;
}

std::size_t elements(std::span<const std::int64_t> extent) {
  std::size_t n = 1;
  for (const std::int64_t e : extent) n *= static_cast<std::size_t>(e);
  return n;
}

/// Runs a two-operand einsum on the device and on the host reference, and
/// compares them exactly.
///
/// With `abs_a`, a is read through UnaryOp::Abs, which the reference gets by
/// taking magnitudes up front; a starts negative so the order matters.
void check_binary(std::string_view equation, std::span<const std::int64_t> ea,
                  std::span<const std::int64_t> eb,
                  std::span<const std::int64_t> ed,
                  katal::Layout layout = katal::Layout::ColumnMajor,
                  bool abs_a = false) {
  std::vector<double> a = ramp(elements(ea), abs_a ? -3.0 : 1.0);
  std::vector<double> b = ramp(elements(eb), -2.0);
  std::vector<double> want(elements(ed), 0.0);
  const DeviceBuffer<double> da(a), db(b), dd(want);

  katal::ConstTensorRef ra =
      katal::make_ref(da.get(), katal::Dtype::F64, ea, layout);
  if (abs_a) ra = katal::apply(katal::UnaryOp::Abs, ra);
  katal::einsum(equation, 1.0, ra,
                katal::make_ref(db.get(), katal::Dtype::F64, eb, layout), 0.0,
                katal::make_ref(dd.get(), katal::Dtype::F64, ed, layout));

  if (abs_a)
    for (double& x : a) x = std::abs(x);
  const std::array<katal::TensorRef, 2> in{
      katal::make_ref(a.data(), katal::Dtype::F64, ea, layout),
      katal::make_ref(b.data(), katal::Dtype::F64, eb, layout)};
  katal::test::reference_einsum<double>(
      katal::Equation::parse(equation), 1.0, in, 0.0,
      katal::make_ref(want.data(), katal::Dtype::F64, ed, layout));
  REQUIRE(dd.to_host() == want);
}

}  // namespace

TEST_CASE("a trace and a diagonal", "[diagonal][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 2> e{3, 3};
  const std::vector<double> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const DeviceBuffer<double> da(a);
  const katal::ConstTensorRef ar =
      katal::make_ref(da.get(), katal::Dtype::F64, e);

  SECTION("trace") {
    const DeviceBuffer<double> dd(std::vector<double>{0.0});
    katal::einsum("ii->", 1.0, ar, 0.0,
                  katal::make_ref(dd.get(), katal::Dtype::F64,
                                  std::span<const std::int64_t>{}));
    REQUIRE(dd.to_host() == std::vector<double>{15});
  }

  SECTION("diagonal") {
    const std::array<std::int64_t, 1> e1{3};
    const DeviceBuffer<double> dd(std::vector<double>(3, 0.0));
    katal::einsum("ii->i", 1.0, ar, 0.0,
                  katal::make_ref(dd.get(), katal::Dtype::F64, e1));
    REQUIRE(dd.to_host() == std::vector<double>{1, 5, 9});
  }

  SECTION("a diagonal read elementwise") {
    const std::array<std::int64_t, 1> e1{3};
    const DeviceBuffer<double> dv(std::vector<double>{10, 20, 30});
    const DeviceBuffer<double> dd(std::vector<double>(3, 0.0));
    katal::add("ii,i->i", 1.0, ar, 1.0,
               katal::make_ref(dv.get(), katal::Dtype::F64, e1),
               katal::make_ref(dd.get(), katal::Dtype::F64, e1));
    REQUIRE(dd.to_host() == std::vector<double>{11, 25, 39});
  }
}

TEST_CASE("a diagonal inside a contraction", "[diagonal][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  const std::array<std::int64_t, 3> ea{3, 3, 4};
  const std::array<std::int64_t, 2> eb{4, 5}, ed{3, 5};
  check_binary("iij,jk->ik", ea, eb, ed);
}

TEST_CASE("a diagonal in row-major order", "[diagonal][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  const std::array<std::int64_t, 3> ea{3, 3, 4};
  const std::array<std::int64_t, 2> eb{4, 5}, ed{3, 5};
  check_binary("iij,jk->ik", ea, eb, ed, katal::Layout::RowMajor);
}

TEST_CASE("a free reduction fused into a contraction", "[diagonal][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  SECTION("summing an input down to a scalar") {
    const std::array<std::int64_t, 2> ea{3, 4};
    const std::array<std::int64_t, 1> eb{5}, ed{5};
    check_binary("ij,k->k", ea, eb, ed);
  }

  SECTION("in row-major order") {
    const std::array<std::int64_t, 3> ea{3, 4, 5};
    const std::array<std::int64_t, 2> eb{5, 2}, ed{3, 2};
    check_binary("ijk,kl->il", ea, eb, ed, katal::Layout::RowMajor);
  }

  SECTION("with an operator applied before the sum") {
    const std::array<std::int64_t, 3> ea{3, 4, 5};
    const std::array<std::int64_t, 2> eb{5, 2}, ed{3, 2};
    check_binary("ijk,kl->il", ea, eb, ed, katal::Layout::ColumnMajor, true);
  }

  SECTION("on one input") {
    const std::array<std::int64_t, 3> ea{3, 4, 5};
    const std::array<std::int64_t, 2> eb{5, 2}, ed{3, 2};
    check_binary("ijk,kl->il", ea, eb, ed);
  }

  SECTION("on both inputs") {
    const std::array<std::int64_t, 3> ea{3, 4, 5}, eb{5, 6, 2};
    const std::array<std::int64_t, 2> ed{3, 2};
    check_binary("ijk,kml->il", ea, eb, ed);
  }

  SECTION("summing a diagonal out of one input") {
    // sum_i a[i][i][j] * b[j][k], with j shared as a batch label.
    const std::array<std::int64_t, 3> ea{3, 3, 4};
    const std::array<std::int64_t, 2> eb{4, 5}, ed{4, 5};
    check_binary("iij,jk->jk", ea, eb, ed);
  }
}

TEST_CASE("a fused reduction keeps its input in the workspace",
          "[diagonal][gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using katal::test::thrown_code;

  const std::array<std::int64_t, 3> ea{3, 4, 5};
  const std::array<std::int64_t, 2> eb{5, 2}, ed{3, 2};
  const DeviceBuffer<double> da(ramp(60, 1.0)), db(ramp(10, 1.0)),
      dd(std::vector<double>(6, 0.0));
  const katal::Plan plan = katal::make_plan(
      katal::default_context(), katal::Equation::parse("ijk,kl->il"),
      katal::make_ref(da.get(), katal::Dtype::F64, ea),
      katal::make_ref(db.get(), katal::Dtype::F64, eb),
      katal::make_ref(dd.get(), katal::Dtype::F64, ed));

  // The reduced a is 3x5 doubles.
  REQUIRE(plan.workspace_size() >= 3 * 5 * sizeof(double));

  REQUIRE(thrown_code([&] {
            plan.execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(), {});
          }) == katal::Errc::WorkspaceTooSmall);

  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size() + 256, {});
  katal::Workspace shifted = scratch.get();
  shifted.ptr = static_cast<std::byte*>(shifted.ptr) + 8;
  shifted.size -= 8;
  REQUIRE(thrown_code([&] {
            plan.execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(),
                         shifted);
          }) == katal::Errc::UnsupportedPattern);
}
