#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "enum_names.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

using katal::test::thrown_code;

int device_count() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess ? devices : 0;
}

}  // namespace

// Planning used to run under the cache lock, so these threads serialized.
// Worth running under -fsanitize=thread as well as plainly.
TEST_CASE("concurrent cached_plan calls share one plan per key", "[gpu]") {
  if (device_count() == 0) SKIP("no CUDA device");
  const katal::Context context;
  const std::array<std::int64_t, 2> square{4, 4}, wide{4, 8};
  const auto a = katal::make_ref(nullptr, katal::Dtype::F64, square);
  const auto b = katal::make_ref(nullptr, katal::Dtype::F64, wide);

  constexpr std::size_t threads = 8, rounds = 16;
  std::vector<std::shared_ptr<const katal::Plan>> same(threads * rounds);
  std::vector<std::shared_ptr<const katal::Plan>> mixed(threads * rounds);
  {
    std::vector<std::jthread> pool;
    for (std::size_t t = 0; t < threads; ++t)
      pool.emplace_back([&, t] {
        for (std::size_t r = 0; r < rounds; ++r) {
          same[t * rounds + r] =
              katal::cached_plan(context, "ij,jk->ik", a, a, a);
          // Odd threads ask for a different key, which must not wait on or
          // disturb the shared one.
          mixed[t * rounds + r] =
              t % 2 == 0 ? katal::cached_plan(context, "ij,jk->ik", a, a, a)
                         : katal::cached_plan(context, "ij,jk->ik", a, b, b);
        }
      });
  }
  for (const auto& plan : same) REQUIRE(plan == same.front());
  for (std::size_t t = 0; t < threads; ++t)
    for (std::size_t r = 0; r < rounds; ++r) {
      const auto& plan = mixed[t * rounds + r];
      if (t % 2 == 0)
        REQUIRE(plan == same.front());
      else
        REQUIRE(plan == mixed[rounds]);
    }
  REQUIRE(mixed[rounds] != same.front());
}

TEST_CASE("a full plan cache evicts the least recently used plan", "[gpu]") {
  if (device_count() == 0) SKIP("no CUDA device");
  katal::Config config;
  config.plan_cache_entries = 2;
  const katal::Context context(config);
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto t = katal::make_ref(nullptr, katal::Dtype::F64, extent);

  const auto first = katal::cached_plan(context, "ij,jk->ik", t, t, t);
  const auto second = katal::cached_plan(context, "ij,kj->ik", t, t, t);
  // Touch the first, so the second is now the least recently used.
  REQUIRE(katal::cached_plan(context, "ij,jk->ik", t, t, t) == first);
  const auto third = katal::cached_plan(context, "ji,jk->ik", t, t, t);

  REQUIRE(katal::cached_plan(context, "ij,jk->ik", t, t, t) == first);
  REQUIRE(katal::cached_plan(context, "ji,jk->ik", t, t, t) == third);
  REQUIRE(katal::cached_plan(context, "ij,kj->ik", t, t, t) != second);
}

TEST_CASE("initialize after the default context exists throws", "[gpu]") {
  if (device_count() == 0) SKIP("no CUDA device");
  katal::finalize();
  katal::initialize();
  katal::Config config;
  config.plan_cache_entries = 0;
  REQUIRE(thrown_code([&] { katal::initialize(config); }) ==
          katal::Errc::InvalidState);
  katal::finalize();
  REQUIRE_NOTHROW(katal::initialize(config));
  katal::finalize();
}

TEST_CASE("a plan rejects a device other than its context's", "[gpu]") {
  if (device_count() < 2) SKIP("needs two CUDA devices");
  // Put device 0 back however this test ends, so later cases are unaffected.
  struct RestoreDevice {
    ~RestoreDevice() { cudaSetDevice(0); }
  } restore;
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  const katal::Context context;
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto t = katal::make_ref(nullptr, katal::Dtype::F64, extent);
  const auto equation = katal::Equation::parse("ij,jk->ik");
  const katal::Plan plan = katal::make_plan(context, equation, t, t, t);
  katal::cached_plan(context, "ij,jk->ik", t, t, t);

  REQUIRE(cudaSetDevice(1) == cudaSuccess);
  const auto rejects = [](auto call) {
    try {
      call();
      FAIL("expected a device mismatch");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == katal::Errc::CudaError);
      REQUIRE(std::string(e.what()).find("device 0") != std::string::npos);
    }
  };
  rejects(
      [&] { plan.execute(1.0, nullptr, nullptr, 0.0, nullptr, nullptr, {}); });
  rejects([&] { katal::make_plan(context, equation, t, t, t); });
  // Already cached, so this is a hit that never reaches planning.
  rejects([&] { katal::cached_plan(context, "ij,jk->ik", t, t, t); });
}
