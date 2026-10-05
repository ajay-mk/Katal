#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

bool have_gpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Half and bfloat16 bit patterns for small integers, which both formats hold
/// exactly, so the conversions need no rounding.
std::uint16_t to_half(float value) {
  if (value == 0.0f) return 0;
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const auto exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  const auto mantissa = static_cast<std::uint16_t>((bits >> 13) & 0x3ffu);
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint16_t>(exponent << 10) | mantissa);
}

float from_half(std::uint16_t half) {
  if ((half & 0x7fffu) == 0) return 0.0f;
  const std::uint32_t sign = (half & 0x8000u) << 16;
  const std::uint32_t exponent = ((half >> 10) & 0x1fu) - 15 + 127;
  const std::uint32_t mantissa = (half & 0x3ffu) << 13;
  return std::bit_cast<float>(sign | (exponent << 23) | mantissa);
}

std::uint16_t to_bf16(float value) {
  return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}

float from_bf16(std::uint16_t bf16) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(bf16) << 16);
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

/// 16-bit storage, whose element type the host has no arithmetic for.
using DeviceBytes = DeviceBuffer<std::uint16_t>;

/// Converts host values to a 16-bit type and copies them to the device.
template <class ToBits>
DeviceBytes upload(const std::vector<float>& values, ToBits to_bits) {
  std::vector<std::uint16_t> bits(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) bits[i] = to_bits(values[i]);
  return DeviceBytes(bits);
}

/// Builds a plan, skipping when the device lacks a kernel for the type. Older
/// devices lack some 16-bit kernels; any other failure, the inexpressible-
/// scalar one in particular, is a real one.
template <class MakePlan>
katal::Plan plan_or_skip(MakePlan make) {
  std::optional<katal::Plan> plan;
  try {
    plan.emplace(make());
  } catch (const katal::Error& e) {
    if (e.code() == katal::Errc::UnsupportedPattern)
      SKIP("this device has no kernel for the type: " << e.what());
    throw;
  }
  return std::move(*plan);
}

/// Small integers, which both 16-bit formats hold exactly.
std::vector<float> small_integers(std::size_t n, std::size_t period,
                                  std::size_t offset = 0) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = static_cast<float>((i + offset) % period);
  return v;
}

/// `d = alpha * a.b + beta * d` for 4x4 matrices in a 16-bit type, against an
/// exact host result. Every partial sum is a small integer, so the 16-bit
/// result is exact; a beta other than zero also checks that d was read back
/// in the right type.
template <class ToBits, class FromBits>
void half_precision_product(katal::Dtype type, ToBits to_bits,
                            FromBits from_bits, double alpha, double beta) {
  constexpr std::int64_t n = 4;
  const std::array<std::int64_t, 2> extent{n, n};
  const std::vector<float> a = small_integers(n * n, 3);
  const std::vector<float> b = small_integers(n * n, 4, 1);
  const std::vector<float> d0 = small_integers(n * n, 5);
  const DeviceBytes da = upload(a, to_bits), db = upload(b, to_bits),
                    dd = upload(d0, to_bits);

  const katal::TensorRef ar = katal::make_ref(da.get(), type, extent);
  const katal::TensorRef br = katal::make_ref(db.get(), type, extent);
  const katal::TensorRef dr = katal::make_ref(dd.get(), type, extent);
  const katal::Plan plan = plan_or_skip([&] {
    return katal::make_plan(katal::default_context(),
                            katal::Equation::parse("ab,bc->ac"), ar, br, dr);
  });

  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size(), {});
  plan.execute(alpha, da.get(), db.get(), beta, dd.get(), dd.get(),
               scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<std::uint16_t> got = dd.to_host();

  // Row-major: d[i][k] = alpha * sum_j a[i][j] * b[j][k] + beta * d0[i][k].
  for (std::int64_t i = 0; i < n; ++i)
    for (std::int64_t k = 0; k < n; ++k) {
      float sum = 0.0f;
      for (std::int64_t j = 0; j < n; ++j) sum += a[i * n + j] * b[j * n + k];
      const auto want = static_cast<float>(alpha * sum + beta * d0[i * n + k]);
      REQUIRE(from_bits(got[i * n + k]) == want);
    }
}

/// `d = alpha * sum_b a[a][b] + beta * d` over a 4x8 tensor in a 16-bit type,
/// which goes through the one-operand plan and its own scalar type.
template <class ToBits, class FromBits>
void half_precision_reduction(katal::Dtype type, ToBits to_bits,
                              FromBits from_bits, double alpha, double beta) {
  constexpr std::int64_t rows = 4, cols = 8;
  const std::array<std::int64_t, 2> ea{rows, cols};
  const std::array<std::int64_t, 1> ed{rows};
  const std::vector<float> a = small_integers(rows * cols, 3);
  const std::vector<float> d0 = small_integers(rows, 5, 2);
  const DeviceBytes da = upload(a, to_bits), dd = upload(d0, to_bits);

  const katal::TensorRef ar = katal::make_ref(da.get(), type, ea);
  const katal::TensorRef dr = katal::make_ref(dd.get(), type, ed);
  const katal::Plan plan = plan_or_skip([&] {
    return katal::make_plan(katal::default_context(),
                            katal::Equation::parse("ab->a"), ar, dr);
  });

  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size(), {});
  plan.execute(alpha, da.get(), beta, dd.get(), dd.get(), scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<std::uint16_t> got = dd.to_host();

  for (std::int64_t i = 0; i < rows; ++i) {
    float sum = 0.0f;
    for (std::int64_t j = 0; j < cols; ++j) sum += a[i * cols + j];
    const auto want = static_cast<float>(alpha * sum + beta * d0[i]);
    REQUIRE(from_bits(got[i]) == want);
  }
}

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

constexpr std::array<std::int64_t, 4> hungry_a{8, 8, 16, 16},
    hungry_b{16, 16, 8, 8}, hungry_d{8, 8, 8, 8};

void require_workspace_too_small(const std::function<void()>& run) {
  try {
    run();
    FAIL("expected WorkspaceTooSmall");
  } catch (const katal::Error& e) {
    REQUIRE(e.code() == katal::Errc::WorkspaceTooSmall);
  }
}

}  // namespace

TEST_CASE("half-precision contractions execute", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  // alpha = 1, beta = 0 is the plain product; the second pair checks that
  // alpha and beta reach cuTENSOR as their values, not just as 1 and 0.
  const auto [alpha, beta] =
      GENERATE(std::pair{1.0, 0.0}, std::pair{2.0, -1.0});
  CAPTURE(alpha, beta);
  SECTION("F16") {
    half_precision_product(katal::Dtype::F16, to_half, from_half, alpha, beta);
  }
  SECTION("Bf16") {
    half_precision_product(katal::Dtype::Bf16, to_bf16, from_bf16, alpha, beta);
  }
}

TEST_CASE("half-precision reductions execute", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  SECTION("F16") {
    half_precision_reduction(katal::Dtype::F16, to_half, from_half, 2.0, -1.0);
  }
  SECTION("Bf16") {
    half_precision_reduction(katal::Dtype::Bf16, to_bf16, from_bf16, 2.0, -1.0);
  }
}

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
