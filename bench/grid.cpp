// Times two particle-particle ladder layouts over fixed occupied/virtual grids.
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

/// Kernels shorter than this are reported, but their throughput is a measure
/// of launch overhead rather than of arithmetic.
constexpr double kLaunchBoundMs = 0.05;

using benchmark::Buffer;
using benchmark::extents;
using benchmark::product;

/// One probe run sets the iteration count, so every point spends about the
/// same wall time being measured regardless of its size.
double time_best(const std::function<void()>& run) {
  run();
  example::check(cudaDeviceSynchronize());
  example::Timer probe_timer;
  run();
  const double probe = probe_timer.seconds() * 1e3;

  const int iterations =
      std::clamp(static_cast<int>(100.0 / std::max(probe, 1e-3)), 3, 500);

  double best = 1e30;
  for (int repeat = 0; repeat < 3; ++repeat) {
    example::Timer timer;
    for (int i = 0; i < iterations; ++i) run();
    const double elapsed = timer.seconds() * 1e3;
    best = std::min(best, static_cast<double>(elapsed) / iterations);
  }

  return best;
}

/// Returns false when the point could not be run, e.g. out of device memory,
/// so the sweep can report it and move on to the remaining points.
bool run_point(const char* spelling, const std::string& equation,
               std::int64_t o, std::int64_t v) {
  const auto arrow = equation.find("->");
  const auto comma = equation.find(',');
  const std::vector<std::int64_t> ea = extents(equation.substr(0, comma), o, v);
  const std::vector<std::int64_t> eb =
      extents(equation.substr(comma + 1, arrow - comma - 1), o, v);
  const std::vector<std::int64_t> ed =
      extents(equation.substr(arrow + 2), o, v);

  const double gib = static_cast<double>(product(eb)) * 8.0 / (1 << 30);
  const double flops = 2.0 * static_cast<double>(o) * o * v * v * v * v;

  try {
    const Buffer a(product(ea)), b(product(eb)), d(product(ed));
    const katal::TensorRef ra = katal::make_ref(a.get(), katal::Dtype::F64, ea);
    const katal::TensorRef rb = katal::make_ref(b.get(), katal::Dtype::F64, eb);
    const katal::TensorRef rd = katal::make_ref(d.get(), katal::Dtype::F64, ed);

    const katal::Plan plan = katal::make_plan(
        katal::default_context(), katal::Equation::parse(equation), ra, rb, rd);
    const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
        {}, plan.workspace_size(), {});
    const katal::Workspace workspace = scratch.get();

    const double ms = time_best([&] {
      plan.execute(1.0, a.get(), b.get(), 0.0, d.get(), d.get(), workspace);
    });

    const double tflops = flops / (ms * 1e-3) / 1e12;
    std::printf("  %4lld %5lld  %-5s %8.1f %10.4f %9.2f   %s\n", (long long)o,
                (long long)v, spelling, gib, ms, tflops,
                ms < kLaunchBoundMs ? "launch bound" : "");
    return true;
  } catch (const std::exception& e) {
    std::printf("  %4lld %5lld  %-5s %8.1f   %s\n", (long long)o, (long long)v,
                spelling, gib, e.what());
    return false;
  }
}

}  // namespace

int run() {
  cudaDeviceProp prop{};
  example::check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device  %s\n\n", prop.name);

  std::printf("  %4s %5s  %-5s %8s %10s %9s\n", "o", "v", "spell", "B (GiB)",
              "ms", "TFLOP/s");
  std::printf("  %s\n", std::string(72, '-').c_str());

  int failed = 0;
  for (const std::int64_t o : {4, 8, 16}) {
    for (const std::int64_t v : {8, 16, 32, 64, 128, 256}) {
      failed += !run_point("acbd", "ijcd,acbd->ijab", o, v);
      failed += !run_point("abcd", "ijab,abcd->ijcd", o, v);
    }
    std::printf("  %s\n", std::string(72, '-').c_str());
  }
  if (failed > 0) std::printf("\n%d point(s) failed\n", failed);
  return failed > 0 ? 1 : 0;
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
