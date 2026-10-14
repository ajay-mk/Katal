#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "cuda_memory.hpp"
#include "enum_names.hpp"
#include "reference.hpp"
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
    T* data = nullptr;
    REQUIRE(cudaMalloc(&data, bytes()) == cudaSuccess);
    ptr_.reset(data);
    REQUIRE(cudaMemcpy(ptr_.get(), host.data(), bytes(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const noexcept { return ptr_.get(); }

  std::vector<T> to_host() const {
    std::vector<T> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_.get(), bytes(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }

 private:
  std::size_t bytes() const noexcept { return size_ * sizeof(T); }

  std::unique_ptr<T, katal::test::DeviceFree> ptr_;
  std::size_t size_ = 0;
};

/// Transpose-transpose-GEMM-transpose with the largest workspace preference
/// is the pairing most likely to need scratch.
katal::Plan workspace_hungry_plan(const katal::TensorRef& a,
                                  const katal::TensorRef& b,
                                  const katal::TensorRef& d) {
  katal::PlanOptions options;
  options.algo = katal::Algo::Ttgt;
  options.workspace = katal::WorkspacePreference::Max;
  return katal::make_plan(katal::default_context(),
                          katal::Equation::parse("ijab,cdba->ijdc"), a, b, d,
                          options);
}

// i, j, c and d are 8; a and b are 16. b is laid out c, d, b, a.
constexpr std::array<std::int64_t, 4> hungry_a{8, 8, 16, 16},
    hungry_b{8, 8, 16, 16}, hungry_d{8, 8, 8, 8};

void require_workspace_too_small(const std::function<void()>& run) {
  try {
    run();
    FAIL("expected WorkspaceTooSmall");
  } catch (const katal::Error& e) {
    REQUIRE(e.code() == katal::Errc::WorkspaceTooSmall);
  }
}

}  // namespace

TEST_CASE("execute rejects a workspace smaller than the plan needs", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  const auto ar = katal::make_ref(nullptr, katal::Dtype::F64, hungry_a);
  const auto br = katal::make_ref(nullptr, katal::Dtype::F64, hungry_b);
  const auto dr = katal::make_ref(nullptr, katal::Dtype::F64, hungry_d);
  const katal::Plan plan = workspace_hungry_plan(ar, br, dr);
  if (plan.workspace_size() == 0) SKIP("this plan needs no workspace");

  // The check comes before any device memory is touched, so null operands
  // are enough to reach it.
  SECTION("one byte short") {
    const katal::Workspace short_by_one{nullptr, plan.workspace_size() - 1};
    require_workspace_too_small([&] {
      plan.execute(1.0, nullptr, nullptr, 0.0, nullptr, nullptr, short_by_one);
    });
  }
  SECTION("none at all") {
    require_workspace_too_small([&] {
      plan.execute(1.0, nullptr, nullptr, 0.0, nullptr, nullptr, {});
    });
  }
}

TEST_CASE("execute accepts a workspace of exactly the plan's size", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  const auto elements = [](std::span<const std::int64_t> extent) {
    std::int64_t n = 1;
    for (std::int64_t e : extent) n *= e;
    return static_cast<std::size_t>(n);
  };
  std::vector<double> a(elements(hungry_a)), b(elements(hungry_b));
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = double(i % 7) - 3.0;
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = double(i % 5) - 2.0;
  const DeviceBuffer<double> da(a), db(b),
      dd(std::vector<double>(elements(hungry_d), 0.0));

  const auto ar = katal::make_ref(da.get(), katal::Dtype::F64, hungry_a);
  const auto br = katal::make_ref(db.get(), katal::Dtype::F64, hungry_b);
  const auto dr = katal::make_ref(dd.get(), katal::Dtype::F64, hungry_d);
  const katal::Plan plan = workspace_hungry_plan(ar, br, dr);
  if (plan.workspace_size() == 0) SKIP("this plan needs no workspace");
  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size(), {});
  plan.execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(), scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<double> d = dd.to_host();

  // Integer inputs keep every sum exact, so the comparison can be exact too.
  std::vector<double> want(d.size(), 0.0);
  auto ha = katal::make_ref(a.data(), katal::Dtype::F64, hungry_a);
  auto hb = katal::make_ref(b.data(), katal::Dtype::F64, hungry_b);
  auto hd = katal::make_ref(want.data(), katal::Dtype::F64, hungry_d);
  const std::array<katal::TensorRef, 2> in{ha, hb};
  katal::test::reference_einsum<double>(
      katal::Equation::parse("ijab,cdba->ijdc"), 1.0, in, 0.0, hd);
  for (std::size_t i = 0; i < d.size(); ++i) REQUIRE(d[i] == want[i]);
}

TEST_CASE("a complex tensor contracts with a real one", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using C = std::complex<double>;

  // d[i][k] = sum_j a[i][j] * b[j][k], complex a, real b, row-major.
  const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 2}, ed{2, 2};
  const std::vector<C> a{{1, 1}, {2, -1}, {0, 3}, {1, 0}, {-2, 2}, {1, 1}};
  const std::vector<double> b{1, 2, 3, 4, 5, 6};
  const DeviceBuffer<C> da(a), dd(std::vector<C>(4));
  const DeviceBuffer<double> db(b);

  katal::einsum("ij,jk->ik", 1.0,
                katal::make_ref(da.get(), katal::Dtype::C64, ea),
                katal::make_ref(db.get(), katal::Dtype::F64, eb), 0.0,
                katal::make_ref(dd.get(), katal::Dtype::C64, ed));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<C> got = dd.to_host();
  for (int i = 0; i < 2; ++i)
    for (int k = 0; k < 2; ++k) {
      C want{};
      for (int j = 0; j < 3; ++j) want += a[i * 3 + j] * b[j * 2 + k];
      REQUIRE(got[i * 2 + k] == want);
    }
}
