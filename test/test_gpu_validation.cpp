#include <array>

#include "enum_names.hpp"
#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

using katal::test::thrown_code;

TEST_CASE("plan construction validates ranks before inspecting extents",
          "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  const katal::Context context;
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto good = katal::make_ref(nullptr, katal::Dtype::F64, extent);
  const auto bad = katal::make_ref(nullptr, katal::Dtype::F64, {});
  const auto unary = katal::Equation::parse("ij->ji");
  const auto binary = katal::Equation::parse("ij,jk->ik");
  const auto reject = [](auto make) {
    try {
      make();
      FAIL("expected rank mismatch");
    } catch (const katal::Error& e) {
      REQUIRE(e.code() == katal::Errc::RankMismatch);
    }
  };
  reject([&] { katal::make_plan(context, unary, bad, good); });
  reject([&] { katal::make_plan(context, unary, good, bad); });
  reject([&] { katal::make_plan(context, binary, bad, good, good); });
  reject([&] { katal::make_plan(context, binary, good, bad, good); });
  reject([&] { katal::make_plan(context, binary, good, good, bad); });
}

TEST_CASE("zero cache capacity disables reuse", "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  katal::Config config;
  config.plan_cache_entries = 0;
  const katal::Context context(config);
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto tensor = katal::make_ref(nullptr, katal::Dtype::F64, extent);
  const auto first =
      katal::cached_plan(context, "ij,jk->ik", tensor, tensor, tensor);
  const auto second =
      katal::cached_plan(context, "ij,jk->ik", tensor, tensor, tensor);
  REQUIRE(first.get() != second.get());
}

TEST_CASE("plan cache distinguishes execution descriptors and options",
          "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  const katal::Context context;
  const std::array<std::int64_t, 2> extent{2, 2}, padded{3, 1};
  const auto tensor = katal::make_ref(nullptr, katal::Dtype::F64, extent);
  const auto original =
      katal::cached_plan(context, "ij,jk->ik", tensor, tensor, tensor);
  auto changed = tensor;
  changed.layout = katal::Layout::ColumnMajor;
  REQUIRE(katal::cached_plan(context, "ij,jk->ik", changed, tensor, tensor) !=
          original);
  changed = tensor;
  changed.stride = padded;
  REQUIRE(katal::cached_plan(context, "ij,jk->ik", changed, tensor, tensor) !=
          original);
  changed = tensor;
  changed.alignment = 128;
  REQUIRE(katal::cached_plan(context, "ij,jk->ik", changed, tensor, tensor) !=
          original);
  katal::PlanOptions options;
  options.workspace = katal::WorkspacePreference::Max;
  REQUIRE(katal::cached_plan(context, "ij,jk->ik", tensor, tensor, tensor,
                             options) != original);
}

TEST_CASE("plan construction accepts equation-wide labels beyond max rank",
          "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  if (katal::max_rank < 9) SKIP("needs rank-nine tensors");
  const katal::Context context;
  const std::array<std::int64_t, 9> input{2, 2, 2, 2, 2, 2, 2, 2, 2};
  const std::array<std::int64_t, 8> output{2, 2, 2, 2, 2, 2, 2, 2};
  const auto a = katal::make_ref(nullptr, katal::Dtype::F64, input);
  const auto d = katal::make_ref(nullptr, katal::Dtype::F64, output);
  REQUIRE_NOTHROW(katal::make_plan(
      context, katal::Equation::parse("abcdefghi,efghijklm->abcdjklm"), a, a,
      d));
}

TEST_CASE("a context remains usable after backend setup failures", "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  const katal::Context context;
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto tensor = katal::make_ref(nullptr, katal::Dtype::F64, extent);
  auto invalid = tensor;
  invalid.type = static_cast<katal::Dtype>(-1);
  const auto eq = katal::Equation::parse("ij,jk->ik");
  for (int i = 0; i < 8; ++i)
    REQUIRE_THROWS_AS(katal::make_plan(context, eq, tensor, invalid, tensor),
                      katal::Error);
  katal::PlanOptions options;
  options.algo = static_cast<katal::Algo>(-1);
  REQUIRE_THROWS_AS(
      katal::make_plan(context, eq, tensor, tensor, tensor, options),
      katal::Error);
  REQUIRE_NOTHROW(katal::make_plan(context, eq, tensor, tensor, tensor));
}

TEST_CASE("Runtime rejects competing owners and premature finalization",
          "[gpu]") {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    SKIP("no CUDA device");
  katal::finalize();
  {
    const katal::Runtime runtime;
    const auto* context = &katal::default_context();
    REQUIRE(thrown_code([] { katal::Runtime{}; }) == katal::Errc::InvalidState);
    REQUIRE(thrown_code([] { katal::finalize(); }) ==
            katal::Errc::InvalidState);
    REQUIRE(katal::initialized());
    REQUIRE(&katal::default_context() == context);
  }
  REQUIRE_FALSE(katal::initialized());
  katal::initialize();
  REQUIRE(thrown_code([] { katal::Runtime{}; }) == katal::Errc::InvalidState);
  REQUIRE(katal::initialized());
  katal::finalize();
}
