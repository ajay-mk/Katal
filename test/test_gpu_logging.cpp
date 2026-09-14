#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

class CaptureStderr {
 public:
  CaptureStderr() : previous_(std::cerr.rdbuf(output_.rdbuf())) {}
  ~CaptureStderr() { std::cerr.rdbuf(previous_); }
  CaptureStderr(const CaptureStderr&) = delete;
  CaptureStderr& operator=(const CaptureStderr&) = delete;
  std::string str() const { return output_.str(); }

 private:
  std::ostringstream output_;
  std::streambuf* previous_;
};

}  // namespace

TEST_CASE("cached_plan reports cache reuse when logging is enabled",
          "[gpu][logging]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");

  katal::Config config;
  config.plan_cache_entries = 1;
  const katal::Context context(config);
  const std::array<std::int64_t, 2> extents{2, 2};
  const auto tensor = katal::make_ref(nullptr, katal::Dtype::F64, extents);
  std::string output;
  {
    const CaptureStderr capture;
    const auto first =
        katal::cached_plan(context, "ij,jk->ik", tensor, tensor, tensor);
    const auto second =
        katal::cached_plan(context, "ij,jk->ik", tensor, tensor, tensor);
    REQUIRE(first == second);
    const auto different =
        katal::cached_plan(context, "ij,kj->ik", tensor, tensor, tensor);
    REQUIRE(first != different);
    config.plan_cache_entries = 0;
    const katal::Context uncached(config);
    const auto bypass_first =
        katal::cached_plan(uncached, "ij,jk->ik", tensor, tensor, tensor);
    const auto bypass_second =
        katal::cached_plan(uncached, "ij,jk->ik", tensor, tensor, tensor);
    REQUIRE(bypass_first != bypass_second);
    output = capture.str();
  }
#if KATAL_ENABLE_LOGGING
  const auto count = [&](std::string_view event) {
    std::size_t result = 0, pos = 0;
    while ((pos = output.find(event, pos)) != std::string::npos) {
      ++result;
      pos += event.size();
    }
    return result;
  };
  REQUIRE(count("[katal] cache.miss") == 2);
  REQUIRE(count("[katal] cache.insert") == 2);
  REQUIRE(count("[katal] cache.hit") == 1);
  REQUIRE(count("[katal] cache.clear") == 1);
  REQUIRE(count("[katal] cache.bypass") == 2);
  REQUIRE(count("[katal] plan.created") == 4);
  REQUIRE(output.find("removed=1") != std::string::npos);
  REQUIRE(output.find("equation=\"ij,jk->ik\"") != std::string::npos);
  REQUIRE(output.find("a=F64[2,2]") != std::string::npos);
  REQUIRE(output.find("workspace_bytes=") != std::string::npos);
  REQUIRE(output.find("host_plan_us=") != std::string::npos);
#else
  REQUIRE(output.empty());
#endif
}

TEST_CASE("einsum diagnostics preserve binary and unary results",
          "[gpu][logging]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  const katal::Context context;
  using Managed = katal::ManagedAllocator<std::byte>;
  katal::Tensor<Managed> a(katal::Dtype::F64, {2, 2});
  katal::Tensor<Managed> b(katal::Dtype::F64, {2, 2});
  katal::Tensor<Managed> d(katal::Dtype::F64, {2, 2});
  katal::Tensor<Managed> transposed(katal::Dtype::F64, {2, 2});
  for (std::size_t i = 0; i < a.size(); ++i) {
    static_cast<double*>(a.data())[i] = 1.0;
    static_cast<double*>(b.data())[i] = 2.0;
  }
  std::string output;
  {
    const CaptureStderr capture;
    katal::einsum("ij,jk->ik", 1.0, a.ref(), b.ref(), 0.0, d.ref(), {}, {},
                  context);
    katal::einsum("ij->ji", 1.0, d.ref(), 0.0, transposed.ref(), {}, {},
                  context);
    REQUIRE(cudaStreamSynchronize(nullptr) == cudaSuccess);
    output = capture.str();
  }
  for (std::size_t i = 0; i < d.size(); ++i) {
    REQUIRE(static_cast<const double*>(d.data())[i] == 4.0);
    REQUIRE(static_cast<const double*>(transposed.data())[i] == 4.0);
  }
#if KATAL_ENABLE_LOGGING
  REQUIRE(output.find("equation=\"ij,jk->ik\"") != std::string::npos);
  REQUIRE(output.find("equation=\"ij->ji\"") != std::string::npos);
  REQUIRE(output.find("output=F64[2,2]") != std::string::npos);
#else
  REQUIRE(output.empty());
#endif
}
