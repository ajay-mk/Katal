#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>

#include "enum_names.hpp"
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

// Runs in its own process under CTest because the fault destroys its CUDA
// context.
TEST_CASE("cleanup during sticky-fault unwinding does not terminate", "[gpu]") {
  if (!have_device()) SKIP("no CUDA device");
  struct DeviceFault {};
  try {
    katal::Tensor<> pooled(katal::Dtype::F64, {64, 64});
    katal::Tensor<katal::ManagedAllocator<std::byte>> managed(katal::Dtype::F64,
                                                              {64, 64});
    const katal::util::ScopedWorkspace workspace(katal::CudaAsyncAllocator{},
                                                 1024, katal::StreamRef{});
    // Aligned but unmapped, so the output kernel faults.
    void* const unmapped = reinterpret_cast<void*>(std::uintptr_t{1} << 46);
    const std::array<std::int64_t, 2> extent{64, 64};
    const auto input = pooled.ref();
    const auto output = katal::make_ref(unmapped, katal::Dtype::F64, extent,
                                        katal::Layout::ColumnMajor);
    std::exception_ptr launch_error;
    try {
      katal::einsum("ij,jk->ik", 1.0, input, input, 0.0, output);
    } catch (const katal::Error& error) {
      // A launch may report the device fault before the explicit wait.
      if (error.code() != katal::Errc::CudaError &&
          error.code() != katal::Errc::CutensorError)
        throw;
      launch_error = std::current_exception();
    }
    const auto fault = cudaDeviceSynchronize();
    if (fault != cudaErrorIllegalAddress && fault != cudaErrorLaunchFailure) {
      if (launch_error) std::rethrow_exception(launch_error);
      REQUIRE(fault == cudaSuccess);
      SKIP("the unmapped write did not trigger a sticky device fault");
    }
    throw DeviceFault{};
  } catch (const DeviceFault&) {
    SUCCEED("the original fault reached its handler after cleanup");
  }
}
