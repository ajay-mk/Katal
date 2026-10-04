// Compares hoisted-plan execution with cached and uncached einsum calls.
// Synthetic zero-filled tensors; this driver reports timings, not accuracy.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
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

}  // namespace

int run() {
  cudaDeviceProp prop{};
  example::check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device  %s\n\n", prop.name);
  std::printf("  %4s %5s %10s %10s %10s %9s %9s\n", "o", "v", "hoisted",
              "einsum", "no cache", "overhead", "factor");
  std::printf("  %s\n", std::string(72, '-').c_str());

  const std::string equation = "ijab,abcd->ijcd";

  for (const std::int64_t o : {8, 16}) {
    for (const std::int64_t v : {8, 16, 32, 64, 128}) {
      const auto arrow = equation.find("->");
      const auto comma = equation.find(',');
      const auto ea = extents(equation.substr(0, comma), o, v);
      const auto eb =
          extents(equation.substr(comma + 1, arrow - comma - 1), o, v);
      const auto ed = extents(equation.substr(arrow + 2), o, v);

      const Buffer a(product(ea)), b(product(eb)), d(product(ed));
      const auto ra = katal::make_ref(a.get(), katal::Dtype::F64, ea);
      const auto rb = katal::make_ref(b.get(), katal::Dtype::F64, eb);
      const auto rd = katal::make_ref(d.get(), katal::Dtype::F64, ed);

      const katal::Context context;
      const katal::Plan plan = katal::make_plan(
          context, katal::Equation::parse(equation), ra, rb, rd);
      const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
          {}, plan.workspace_size(), {});
      const katal::Workspace workspace = scratch.get();

      const double hoisted = time_best([&] {
        plan.execute(1.0, a.get(), b.get(), 0.0, d.get(), d.get(), workspace);
      });
      const double one_shot = time_best([&] {
        katal::einsum(equation, 1.0, ra, rb, 0.0, rd, {}, {}, context);
      });

      katal::Config uncached;
      uncached.plan_cache_entries = 0;
      const katal::Context no_cache(uncached);
      const double without = time_best([&] {
        katal::einsum(equation, 1.0, ra, rb, 0.0, rd, {}, {}, no_cache);
      });

      std::printf("  %4lld %5lld %10.4f %10.4f %10.4f %9.4f %8.2fx\n",
                  (long long)o, (long long)v, hoisted, one_shot, without,
                  one_shot - hoisted, one_shot / hoisted);
    }
    std::printf("  %s\n", std::string(72, '-').c_str());
  }
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
