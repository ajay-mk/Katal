// Compares particle-particle ladder shapes and layouts with a cuBLAS DGEMM.
// Synthetic zero-filled tensors; this driver reports timings, not accuracy.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "util.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace {

void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("benchmark: cuBLAS call failed, status " +
                             std::to_string(static_cast<int>(status)));
}

void destroy_blas(cublasHandle_t handle) noexcept {
  if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) std::terminate();
}

constexpr int kWarmup = 3;
constexpr int kTimed = 20;
constexpr int kRepeats = 3;

struct Shape {
  const char* name;
  const char* equation;
  std::int64_t o;
  std::int64_t v;
};

using benchmark::Buffer;
using benchmark::extents;
using benchmark::product;

double time_best(const std::function<void()>& run) {
  for (int i = 0; i < kWarmup; ++i) run();
  example::check(cudaDeviceSynchronize());

  double best = 1e30;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    example::Timer timer;
    for (int i = 0; i < kTimed; ++i) run();
    const double elapsed = timer.seconds() * 1e3;
    best = std::min(best, static_cast<double>(elapsed) / kTimed);
  }

  return best;
}

void report(const char* label, double milliseconds, double flops,
            const char* note) {
  const double tflops = flops / (milliseconds * 1e-3) / 1e12;
  std::printf("  %-30s %9.3f %9.2f   %s\n", label, milliseconds, tflops, note);
}

}  // namespace

int run() {
  cudaDeviceProp prop{};
  example::check(cudaGetDeviceProperties(&prop, 0));
  std::printf("device  %s  sm_%d%d\n\n", prop.name, prop.major, prop.minor);

  const std::vector<Shape> shapes{
      {"A  o=15 v=93  acbd", "ijcd,acbd->ijab", 15, 93},
      {"B  o=15 v=96  acbd", "ijcd,acbd->ijab", 15, 96},
      {"C  o=15 v=93  abcd", "ijab,abcd->ijcd", 15, 93},
      {"D  o=30 v=93  acbd", "ijcd,acbd->ijab", 30, 93},
  };

  std::printf("  %-30s %9s %9s   %s\n", "run", "ms", "TFLOP/s", "note");
  std::printf("  %s\n", std::string(78, '-').c_str());

  for (const Shape& shape : shapes) {
    const std::string equation = shape.equation;
    const auto arrow = equation.find("->");
    const auto comma = equation.find(',');
    const std::vector<std::int64_t> ea =
        extents(equation.substr(0, comma), shape.o, shape.v);
    const std::vector<std::int64_t> eb = extents(
        equation.substr(comma + 1, arrow - comma - 1), shape.o, shape.v);
    const std::vector<std::int64_t> ed =
        extents(equation.substr(arrow + 2), shape.o, shape.v);

    const Buffer a(product(ea)), b(product(eb)), d(product(ed));
    const katal::TensorRef ra = katal::make_ref(a.get(), katal::Dtype::F64, ea);
    const katal::TensorRef rb = katal::make_ref(b.get(), katal::Dtype::F64, eb);
    const katal::TensorRef rd = katal::make_ref(d.get(), katal::Dtype::F64, ed);

    const double flops = 2.0 * static_cast<double>(shape.o) * shape.o *
                         shape.v * shape.v * shape.v * shape.v;

    const katal::Plan plan = katal::make_plan(
        katal::default_context(), katal::Equation::parse(equation), ra, rb, rd);
    const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
        {}, plan.workspace_size(), {});
    const katal::Workspace workspace = scratch.get();

    const double milliseconds = time_best([&] {
      plan.execute(1.0, a.get(), b.get(), 0.0, d.get(), d.get(), workspace);
    });
    report(shape.name, milliseconds, flops, equation.c_str());
  }

  std::printf("  %s\n", std::string(78, '-').c_str());

  // The ceiling: a plain DGEMM of the shape the ladder term becomes once the
  // integrals are permuted, which in a real iteration is done once and reused.
  // Timing only, on arbitrary data.
  cublasHandle_t blas = nullptr;
  check(cublasCreate(&blas));
  const std::unique_ptr<std::remove_pointer_t<cublasHandle_t>,
                        decltype(&destroy_blas)>
      owner(blas, &destroy_blas);

  for (const Shape& shape : {shapes[0], shapes[1], shapes[3]}) {
    const std::int64_t m = shape.o * shape.o;
    const std::int64_t k = shape.v * shape.v;
    const std::int64_t n = shape.v * shape.v;

    const Buffer a(m * k), b(k * n), c(m * n);
    const double alpha = 1.0;
    const double beta = 0.0;

    // Column-major C = A B, the same layout as the Katal runs above.
    const double milliseconds = time_best([&] {
      check(cublasDgemm(
          blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(m),
          static_cast<int>(n), static_cast<int>(k), &alpha,
          static_cast<const double*>(a.get()), static_cast<int>(m),
          static_cast<const double*>(b.get()), static_cast<int>(k), &beta,
          static_cast<double*>(c.get()), static_cast<int>(m)));
    });

    char label[64];
    std::snprintf(label, sizeof(label), "dgemm  %lldx%lldx%lld", (long long)m,
                  (long long)n, (long long)k);
    report(label, milliseconds, 2.0 * static_cast<double>(m) * n * k,
           "pre-permuted, timing only");
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
