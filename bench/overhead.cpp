// Compares hoisted-plan execution with cached and uncached einsum calls.
// Synthetic zero-filled tensors; this driver reports timings, not accuracy.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "util.hpp"
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

using benchmark::Buffer;
using benchmark::extents;
using benchmark::product;

double time_best(const std::function<void()>& run) {
  run();
  example::check(cudaDeviceSynchronize());
  example::Timer probe_timer;
  run();
  const double probe = probe_timer.seconds() * 1e3;

  const int iterations =
      std::clamp(static_cast<int>(100.0 / std::max(probe, 1e-3)), 5, 500);

  double best = 1e30;
  for (int repeat = 0; repeat < 3; ++repeat) {
    example::Timer timer;
    for (int i = 0; i < iterations; ++i) run();
    const double elapsed = timer.seconds() * 1e3;
    best = std::min(best, static_cast<double>(elapsed) / iterations);
  }

  return best;
}

/// Host time per call in microseconds, which the GPU-event timings above hide
/// whenever the kernels outlast the enqueueing. Few enough calls that the
/// launch queue does not fill and block the host.
double host_us(const std::function<void()>& run) {
  run();
  example::check(cudaDeviceSynchronize());
  constexpr int calls = 200;
  double best = 1e30;
  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < calls; ++i) run();
    const auto stop = std::chrono::steady_clock::now();
    example::check(cudaDeviceSynchronize());
    best = std::min(
        best, std::chrono::duration<double, std::micro>(stop - start).count() /
                  calls);
  }
  return best;
}

struct Shape {
  std::vector<std::int64_t> a, b, d;
};

Shape shape(const std::string& equation, std::int64_t o, std::int64_t v) {
  const auto arrow = equation.find("->");
  const auto comma = equation.find(',');
  return {extents(equation.substr(0, comma), o, v),
          extents(equation.substr(comma + 1, arrow - comma - 1), o, v),
          extents(equation.substr(arrow + 2), o, v)};
}

void call_overhead(const std::string& equation) {
  std::printf("  %4s %5s %10s %10s %10s %10s %9s %9s\n", "o", "v", "hoisted",
              "einsum", "default", "no cache", "host us", "factor");
  std::printf("  %s\n", std::string(80, '-').c_str());

  for (const std::int64_t o : {8, 16}) {
    for (const std::int64_t v : {8, 16, 32, 64, 128}) {
      const Shape s = shape(equation, o, v);
      const Buffer a(product(s.a)), b(product(s.b)), d(product(s.d));
      const auto ra = katal::make_ref(a.get(), katal::Dtype::F64, s.a);
      const auto rb = katal::make_ref(b.get(), katal::Dtype::F64, s.b);
      const auto rd = katal::make_ref(d.get(), katal::Dtype::F64, s.d);

      const katal::Context context;
      const katal::Plan plan = katal::make_plan(
          context, katal::Equation::parse(equation), ra, rb, rd);
      const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
          {}, plan.workspace_size(), {});
      const katal::Workspace workspace = scratch.get();

      const double hoisted = time_best([&] {
        plan.execute(1.0, a.get(), b.get(), 0.0, d.get(), d.get(), workspace);
      });
      const auto one_shot_call = [&] {
        katal::einsum(equation, 1.0, ra, rb, 0.0, rd, {}, {}, context);
      };
      const double one_shot = time_best(one_shot_call);
      // The default-context path, which is what a call site that names no
      // context pays.
      const double by_default =
          time_best([&] { katal::einsum(equation, 1.0, ra, rb, 0.0, rd); });
      const double host = host_us(one_shot_call);

      katal::Config uncached;
      uncached.plan_cache_entries = 0;
      const katal::Context no_cache(uncached);
      const double without = time_best([&] {
        katal::einsum(equation, 1.0, ra, rb, 0.0, rd, {}, {}, no_cache);
      });

      std::printf("  %4lld %5lld %10.4f %10.4f %10.4f %10.4f %9.2f %8.2fx\n",
                  (long long)o, (long long)v, hoisted, one_shot, by_default,
                  without, host, one_shot / hoisted);
    }
    std::printf("  %s\n", std::string(80, '-').c_str());
  }
}

/// A loop that synchronizes every iteration, as a convergence check does.
/// With CUDA's default release threshold of 0 the pool hands the einsum
/// scratch back at each synchronization; Config::async_pool_release_threshold
/// keeps it. WorkspacePreference::Max asks for a kernel that uses scratch.
void synchronized_calls(const std::string& equation) {
  std::printf("\n  synchronize after every einsum, WorkspacePreference::Max\n");
  std::printf("  %4s %5s %12s %12s %12s\n", "o", "v", "scratch MiB",
              "release 0", "keep all");
  std::printf("  %s\n", std::string(52, '-').c_str());

  katal::PlanOptions options;
  options.workspace = katal::WorkspacePreference::Max;

  for (const std::int64_t v : {32, 64, 128}) {
    const std::int64_t o = 16;
    const Shape s = shape(equation, o, v);
    const Buffer a(product(s.a)), b(product(s.b)), d(product(s.d));
    const auto ra = katal::make_ref(a.get(), katal::Dtype::F64, s.a);
    const auto rb = katal::make_ref(b.get(), katal::Dtype::F64, s.b);
    const auto rd = katal::make_ref(d.get(), katal::Dtype::F64, s.d);

    // The threshold belongs to the device, so it outlives each context; it
    // is put back to CUDA's default of 0 after the point.
    double ms[2] = {};
    std::uint64_t scratch = 0;
    for (const std::uint64_t threshold :
         {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()}) {
      katal::Config config;
      config.async_pool_release_threshold = threshold;
      const katal::Context context(config);
      scratch = katal::cached_plan(context, equation, ra, rb, rd, options)
                    ->workspace_size();
      ms[threshold == 0 ? 0 : 1] = time_best([&] {
        katal::einsum(equation, 1.0, ra, rb, 0.0, rd, options, {}, context);
        example::check(cudaDeviceSynchronize());
      });
    }
    int device = 0;
    example::check(cudaGetDevice(&device));
    cudaMemPool_t pool = nullptr;
    example::check(cudaDeviceGetDefaultMemPool(&pool, device));
    std::uint64_t release_all = 0;
    example::check(cudaMemPoolSetAttribute(
        pool, cudaMemPoolAttrReleaseThreshold, &release_all));

    std::printf("  %4lld %5lld %12.2f %12.4f %12.4f\n", (long long)o,
                (long long)v, static_cast<double>(scratch) / (1 << 20), ms[0],
                ms[1]);
  }
}

}  // namespace

int run() {
  cudaDeviceProp prop{};
  example::check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device  %s\n", prop.name);
  std::printf(
      "times in ms per call; host us is the CPU cost of one cached "
      "einsum\n\n");

  const std::string equation = "ijab,abcd->ijcd";
  call_overhead(equation);
  synchronized_calls(equation);
  return 0;
}

int main() {
  try {
    const katal::Runtime runtime;
    return run();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
