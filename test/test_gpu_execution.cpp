#include <array>
#include <bit>
#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>
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

class DeviceBytes {
 public:
  explicit DeviceBytes(const std::vector<std::uint16_t>& host)
      : size_(host.size()) {
    REQUIRE(cudaMalloc(&ptr_, bytes()) == cudaSuccess);
    REQUIRE(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  ~DeviceBytes() { cudaFree(ptr_); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;

  void* get() const noexcept { return ptr_; }

  std::vector<std::uint16_t> to_host() const {
    std::vector<std::uint16_t> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    return host;
  }

 private:
  std::size_t bytes() const noexcept { return size_ * sizeof(std::uint16_t); }

  void* ptr_ = nullptr;
  std::size_t size_ = 0;
};

/// A 4x4 matrix product in a 16-bit type, against an exact host result.
template <class ToBits, class FromBits>
void half_precision_product(katal::Dtype type, ToBits to_bits,
                            FromBits from_bits) {
  constexpr std::int64_t n = 4;
  const std::array<std::int64_t, 2> extent{n, n};
  std::vector<float> a(n * n), b(n * n);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(i % 3);
    b[i] = static_cast<float>((i + 1) % 4);
  }

  std::vector<std::uint16_t> ha(a.size()), hb(b.size()), hd(a.size(), 0);
  for (std::size_t i = 0; i < a.size(); ++i) {
    ha[i] = to_bits(a[i]);
    hb[i] = to_bits(b[i]);
  }
  const DeviceBytes da(ha), db(hb), dd(hd);

  const katal::TensorRef ar = katal::make_ref(da.get(), type, extent);
  const katal::TensorRef br = katal::make_ref(db.get(), type, extent);
  const katal::TensorRef dr = katal::make_ref(dd.get(), type, extent);

  std::optional<katal::Plan> plan;
  try {
    plan.emplace(katal::make_plan(katal::default_context(),
                                  katal::Equation::parse("ab,bc->ac"), ar, br,
                                  dr));
  } catch (const katal::Error& e) {
    // Older devices lack some 16-bit kernels. Any other failure, the
    // inexpressible-scalar one in particular, is a real one.
    if (e.code() == katal::Errc::UnsupportedPattern)
      SKIP("this device has no kernel for the type: " << e.what());
    throw;
  }

  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan->workspace_size(), {});
  plan->execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(),
                scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<std::uint16_t> got = dd.to_host();

  // Row-major: d[i][k] = sum_j a[i][j] * b[j][k]. Every partial sum is a
  // small integer, so the 16-bit result is exact.
  for (std::int64_t i = 0; i < n; ++i)
    for (std::int64_t k = 0; k < n; ++k) {
      float want = 0.0f;
      for (std::int64_t j = 0; j < n; ++j) want += a[i * n + j] * b[j * n + k];
      REQUIRE(from_bits(got[i * n + k]) == want);
    }
}

}  // namespace

TEST_CASE("half-precision contractions execute", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  SECTION("F16") {
    half_precision_product(katal::Dtype::F16, to_half, from_half);
  }
  SECTION("Bf16") {
    half_precision_product(katal::Dtype::Bf16, to_bf16, from_bf16);
  }
}

TEST_CASE("execute rejects a workspace smaller than the plan needs", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  const std::array<std::int64_t, 4> ea{8, 8, 16, 16}, eb{16, 16, 8, 8},
      ed{8, 8, 8, 8};
  const auto ar = katal::make_ref(nullptr, katal::Dtype::F64, ea);
  const auto br = katal::make_ref(nullptr, katal::Dtype::F64, eb);
  const auto dr = katal::make_ref(nullptr, katal::Dtype::F64, ed);

  // Transpose-transpose-GEMM-transpose with every algorithm allowed is the
  // pairing most likely to need scratch.
  katal::PlanOptions options;
  options.algo = katal::Algo::Ttgt;
  options.workspace = katal::WorkspacePreference::Max;
  const katal::Plan plan = katal::make_plan(
      katal::default_context(), katal::Equation::parse("ijab,cdba->ijdc"), ar,
      br, dr, options);
  if (plan.workspace_size() == 0) SKIP("this plan needs no workspace");

  // The check comes before any device memory is touched, so null operands
  // are enough to reach it.
  const katal::Workspace short_by_one{nullptr, plan.workspace_size() - 1};
  try {
    plan.execute(1.0, nullptr, nullptr, 0.0, nullptr, nullptr, short_by_one);
    FAIL("expected WorkspaceTooSmall");
  } catch (const katal::Error& e) {
    REQUIRE(e.code() == katal::Errc::WorkspaceTooSmall);
  }
}
