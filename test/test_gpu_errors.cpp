#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

namespace {

bool have_device() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

}  // namespace

TEST_CASE("a rank error from planning quotes the whole equation", "[gpu]") {
  if (!have_device()) SKIP("no CUDA device");
  const katal::Context context;
  const std::array<std::int64_t, 2> matrix{2, 2};
  const std::array<std::int64_t, 3> cube{2, 2, 2};
  const auto two = katal::make_ref(nullptr, katal::Dtype::F64, matrix);
  const auto three = katal::make_ref(nullptr, katal::Dtype::F64, cube);
  const auto equation = katal::Equation::parse("ij,jk->ik");
  try {
    katal::make_plan(context, equation, two, three, two);
    FAIL("expected a rank mismatch");
  } catch (const katal::Error& e) {
    REQUIRE(e.code() == katal::Errc::RankMismatch);
    REQUIRE(e.equation() == "ij,jk->ik");
    REQUIRE(std::string(e.what()).find("tensor \"jk\"") != std::string::npos);
  }
}

TEST_CASE("the default workspace allocator reports exhaustion as bad_alloc",
          "[gpu]") {
  if (!have_device()) SKIP("no CUDA device");
  const katal::CudaAsyncAllocator allocator;
  REQUIRE_THROWS_AS(allocator.allocate(std::uint64_t{1} << 62, {}),
                    std::bad_alloc);
  // The failure was reported by the throw, so it is not left behind too.
  REQUIRE(cudaGetLastError() == cudaSuccess);
}

// Runs in its own process under CTest, since it breaks the CUDA context on
// purpose. Every cleanup path below used to call std::terminate, which killed
// the process during unwinding before the original error could be reported.
TEST_CASE("cleanup after a sticky device fault does not terminate", "[gpu]") {
  if (!have_device()) SKIP("no CUDA device");

  std::optional<katal::Tensor<>> pooled;
  std::optional<katal::Tensor<katal::ManagedAllocator<std::byte>>> managed;
  pooled.emplace(katal::Dtype::F64, katal::Extents{64, 64});
  managed.emplace(katal::Dtype::F64, katal::Extents{64, 64});
  const katal::CudaAsyncAllocator allocator;
  const katal::Workspace workspace = allocator.allocate(1024, {});

  // A 256-byte aligned address that nothing maps, so the kernel that writes
  // the result faults.
  void* const unmapped = reinterpret_cast<void*>(std::uintptr_t{1} << 46);
  const std::array<std::int64_t, 2> extent{64, 64};
  const auto input = pooled->ref();
  const auto output = katal::make_ref(unmapped, katal::Dtype::F64, extent,
                                      katal::Layout::ColumnMajor);
  try {
    katal::einsum("ij,jk->ik", 1.0, input, input, 0.0, output);
  } catch (const katal::Error&) {
    // Some launches report the fault here rather than at the sync below.
  }
  const cudaError_t fault = cudaDeviceSynchronize();
  if (fault != cudaErrorIllegalAddress && fault != cudaErrorLaunchFailure)
    SKIP("could not provoke a sticky fault; got " +
         std::string(cudaGetErrorName(fault)));

  allocator.deallocate(workspace, {});
  pooled.reset();
  managed.reset();
  SUCCEED("every cleanup returned");
}
