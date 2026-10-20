// Three-operand contractions on the device, checked against the host
// reference einsum.

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "cuda_memory.hpp"
#include "reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

bool have_gpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::int64_t elements(std::span<const std::int64_t> extent) {
  std::int64_t n = 1;
  for (std::int64_t e : extent) n *= e;
  return n;
}

/// Distinct values that repeat only every 97 elements, so a misplaced mode
/// shows up in the result.
template <class T>
std::vector<T> ramp(std::int64_t n, double scale) {
  std::vector<T> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double x = scale * static_cast<double>((i % 97) + 1) / 97.0;
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>)
      v[i] = T(x, -0.5 * x);
    else
      v[i] = static_cast<T>(x);
  }
  return v;
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<T>& host) : size_(host.size()) {
    T* data = nullptr;
    REQUIRE(cudaMalloc(&data, size_ * sizeof(T)) == cudaSuccess);
    ptr_.reset(data);
    REQUIRE(cudaMemcpy(ptr_.get(), host.data(), size_ * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const noexcept { return ptr_.get(); }

  std::vector<T> to_host() const {
    std::vector<T> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_.get(), size_ * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }

 private:
  std::unique_ptr<T, katal::test::DeviceFree> ptr_;
  std::size_t size_ = 0;
};

/// Shapes and layout for one case. Equal extents hide layout and stride bugs,
/// so the cases below keep them distinct.
struct Shapes {
  std::span<const std::int64_t> a, b, c, e;
  katal::Layout layout = katal::Layout::RowMajor;
};

/// Runs the equation through the three-operand einsum and again through the
/// host reference, and requires the two to agree to a relative `tolerance`
/// of the largest result.
template <class T>
void agrees_with_reference(std::string_view equation, const Shapes& s, T alpha,
                           T beta, double tolerance) {
  const katal::Dtype type = katal::dtype_of<T>;
  const std::vector<T> a = ramp<T>(elements(s.a), 1.0);
  const std::vector<T> b = ramp<T>(elements(s.b), 0.5);
  const std::vector<T> c = ramp<T>(elements(s.c), 0.75);
  const std::vector<T> e0 = ramp<T>(elements(s.e), 0.25);

  DeviceBuffer<T> da(a), db(b), dc(c), de(e0);
  katal::einsum(equation, alpha, katal::make_ref(da.get(), type, s.a, s.layout),
                katal::make_ref(db.get(), type, s.b, s.layout),
                katal::make_ref(dc.get(), type, s.c, s.layout), beta,
                katal::make_ref(de.get(), type, s.e, s.layout));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<T> got = de.to_host();

  std::vector<T> want = e0;
  const std::array<katal::TensorRef, 3> in{
      katal::make_ref(const_cast<T*>(a.data()), type, s.a, s.layout),
      katal::make_ref(const_cast<T*>(b.data()), type, s.b, s.layout),
      katal::make_ref(const_cast<T*>(c.data()), type, s.c, s.layout)};
  katal::test::reference_einsum<T>(
      katal::Equation::parse(equation), alpha, in, beta,
      katal::make_ref(want.data(), type, s.e, s.layout));

  REQUIRE(got.size() == want.size());
  double scale = 0.0;
  for (const T& w : want) scale = std::max(scale, double(std::abs(w)));
  for (std::size_t i = 0; i < got.size(); ++i)
    REQUIRE(double(std::abs(got[i] - want[i])) <= tolerance * scale);
}

}  // namespace

TEST_CASE("gpu three-operand contraction matches the host reference",
          "[gpu][trinary]") {
  if (!have_gpu()) SKIP("no CUDA device");

  SECTION("a chain of two matrix products") {
    const std::array<std::int64_t, 2> a{2, 3}, b{3, 4}, c{4, 5}, e{2, 5};
    agrees_with_reference<double>("ij,jk,kl->il", {a, b, c, e}, 1.0, 0.0,
                                  1e-12);
  }

  SECTION("column-major operands") {
    const std::array<std::int64_t, 2> a{2, 3}, b{3, 4}, c{4, 5}, e{2, 5};
    agrees_with_reference<double>("ij,jk,kl->il",
                                  {a, b, c, e, katal::Layout::ColumnMajor}, 1.0,
                                  0.0, 1e-12);
  }

  SECTION("alpha and beta accumulate into the output") {
    const std::array<std::int64_t, 2> a{2, 3}, b{3, 4}, c{4, 5}, e{2, 5};
    agrees_with_reference<double>("ij,jk,kl->il", {a, b, c, e}, 2.0, 0.5,
                                  1e-12);
  }

  SECTION("a coupled-cluster shape") {
    const std::array<std::int64_t, 4> a{2, 3, 4, 5}, b{4, 5, 3, 2},
        c{3, 2, 4, 3}, e{2, 3, 4, 3};
    agrees_with_reference<double>("ijab,abmn,mnkl->ijkl", {a, b, c, e}, 1.0,
                                  0.0, 1e-12);
  }

  SECTION("a batch label in every operand") {
    const std::array<std::int64_t, 3> a{2, 3, 4}, b{2, 4, 5}, c{2, 5, 3},
        e{2, 3, 3};
    agrees_with_reference<double>("bij,bjk,bkl->bil", {a, b, c, e}, 1.0, 0.0,
                                  1e-12);
  }

  SECTION("a label in one input alone is summed out of it first") {
    const std::array<std::int64_t, 3> a{2, 3, 4};
    const std::array<std::int64_t, 2> b{4, 5}, c{5, 3}, e{2, 3};
    agrees_with_reference<double>("izj,jk,kl->il", {a, b, c, e}, 1.0, 0.0,
                                  1e-12);
  }

  SECTION("single precision") {
    const std::array<std::int64_t, 2> a{2, 3}, b{3, 4}, c{4, 5}, e{2, 5};
    agrees_with_reference<float>("ij,jk,kl->il", {a, b, c, e}, 1.0F, 0.0F,
                                 1e-5);
  }

  SECTION("complex double") {
    using C = std::complex<double>;
    const std::array<std::int64_t, 2> a{2, 3}, b{3, 4}, c{4, 5}, e{2, 5};
    agrees_with_reference<C>("ij,jk,kl->il", {a, b, c, e}, C(0.75, -0.5),
                             C(-0.25, 0.5), 1e-12);
  }

  SECTION("complex single") {
    using C = std::complex<float>;
    const std::array<std::int64_t, 2> a{2, 3}, b{3, 4}, c{4, 5}, e{2, 5};
    agrees_with_reference<C>("ij,jk,kl->il", {a, b, c, e}, C(1.0F, 0.0F),
                             C(0.0F, 0.0F), 1e-5);
  }
}

TEST_CASE("a three-operand plan writes e apart from d", "[gpu][trinary]") {
  if (!have_gpu()) SKIP("no CUDA device");
  // [[1, 2], [3, 4]] times the identity times [[1, 0], [0, 2]].
  DeviceBuffer<double> a(std::vector<double>{1, 2, 3, 4});
  DeviceBuffer<double> b(std::vector<double>{1, 0, 0, 1});
  DeviceBuffer<double> c(std::vector<double>{1, 0, 0, 2});
  DeviceBuffer<double> d(std::vector<double>{10, 20, 30, 40});
  DeviceBuffer<double> e(std::vector<double>(4, -1.0));
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto ref = [&](const DeviceBuffer<double>& t) {
    return katal::make_ref(t.get(), katal::Dtype::F64, extent);
  };
  const katal::Plan plan = katal::make_plan(
      katal::default_context(), katal::Equation::parse("ij,jk,kl->il"), ref(a),
      ref(b), ref(c), ref(e));
  const katal::util::ScopedWorkspace scratch(
      katal::CudaAsyncAllocator{}, plan.workspace_size(), katal::StreamRef{});
  plan.execute(1.0, a.get(), b.get(), c.get(), 0.5, d.get(), e.get(),
               scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(e.to_host() == std::vector<double>{6, 14, 18, 28});
  REQUIRE(d.to_host() == std::vector<double>{10, 20, 30, 40});
}
