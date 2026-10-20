// Katal against hand-written cuTENSOR, over a size sweep.
//
// For each problem the same contraction runs four ways:
//   cutensor  plan built once by hand with the raw API, then cutensorContract
//   hoisted   katal::make_plan once, then Plan::execute
//   einsum    katal::einsum on a context with the plan cache on (the default)
//   no cache  katal::einsum on a context with Katal's plan cache off
//
// The raw path describes the operands exactly as the cuTENSOR samples do:
// column-major, no explicit strides, with the modes listed fastest first. A
// row-major tensor with its modes reversed is the same memory, so all four
// paths should select the same kernel, and any difference is what Katal
// costs on top. cuBLAS runs the same product as one GEMM, as a ceiling.
// Times are GPU-event wall time per call, best of five batches with the
// paths taking turns, after warm-up. Host time per call is only measured
// for kernels under 1 ms. Tensors hold a constant nonzero byte pattern.
//
//   vs_cutensor [f64|f32 ...]     default: f64 f32

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "util.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutensor.h>

#include <katal/plan.hpp>

namespace {

void check(cutensorStatus_t status) {
  if (status != CUTENSOR_STATUS_SUCCESS)
    throw katal::Error(katal::Errc::CudaError, "",
                       cutensorGetErrorString(status));
}

void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw katal::Error(katal::Errc::CudaError, "",
                       "cuBLAS status " + std::to_string(int(status)));
}

struct Problem {
  std::string equation;  // row-major, Katal's convention
  std::map<char, std::int64_t> extent;
  std::string label;
};

std::vector<std::int64_t> extents_of(const std::string& modes,
                                     const std::map<char, std::int64_t>& e) {
  std::vector<std::int64_t> result;
  for (char m : modes) result.push_back(e.at(m));
  return result;
}

std::int64_t volume(const std::vector<std::int64_t>& extent) {
  std::int64_t n = 1;
  for (std::int64_t x : extent) n *= x;
  return n;
}

class Device {
 public:
  explicit Device(std::size_t bytes) {
    example::check(cudaMalloc(&ptr_, bytes));
    // 0x3f in every byte: about 0.75 as f32, 4.8e-4 as f64. Nonzero, so the
    // kernels do not see an all-zero input, and small enough not to overflow.
    example::check(cudaMemset(ptr_, 0x3f, bytes));
  }
  ~Device() { cudaFree(ptr_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

/// Per-call milliseconds of one batch, after a warm-up. A batch lasts about
/// 100 ms, or at least three calls, so event resolution does not matter.
double batch_ms(const std::function<void()>& run, int calls) {
  example::Timer t;
  for (int i = 0; i < calls; ++i) run();
  return t.seconds() * 1e3 / calls;
}

/// Best per-call milliseconds for each path. The paths take turns, round by
/// round, so clock or thermal drift during a case affects them all alike
/// instead of whichever ran last.
std::vector<double> time_paths(
    const std::vector<std::function<void()>>& paths) {
  std::vector<int> calls;
  for (const auto& run : paths) {
    for (int i = 0; i < 2; ++i) run();
    example::check(cudaDeviceSynchronize());
    const double probe = batch_ms(run, 1);
    calls.push_back(
        std::clamp(static_cast<int>(100.0 / std::max(probe, 1e-3)), 3, 2000));
  }
  std::vector<double> best(paths.size(), 1e30);
  for (int round = 0; round < 5; ++round)
    for (std::size_t k = 0; k < paths.size(); ++k)
      best[k] = std::min(best[k], batch_ms(paths[k], calls[k]));
  return best;
}

/// CPU time to enqueue one call, in microseconds. Few enough calls that the
/// launch queue never fills and blocks the host. Only meaningful when the
/// kernel is short; longer ones are skipped (NaN), since timing them just
/// waits on the GPU.
double host_us(const std::function<void()>& run, double kernel_ms) {
  if (kernel_ms > 1.0) return std::nan("");
  run();
  example::check(cudaDeviceSynchronize());
  constexpr int calls = 100;
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

/// The hand-written path: descriptors, operation, preference, workspace
/// estimate and plan, as in cuTENSOR's contraction sample.
struct RawPlan {
  cutensorTensorDescriptor_t a{}, b{}, d{};
  cutensorOperationDescriptor_t op{};
  cutensorPlanPreference_t pref{};
  cutensorPlan_t plan{};
  std::uint64_t workspace = 0;

  RawPlan(cutensorHandle_t h, const std::string& ma, const std::string& mb,
          const std::string& md, const std::map<char, std::int64_t>& e,
          cutensorDataType_t type, cutensorComputeDescriptor_t compute) {
    // Reversed: fastest mode first, column-major, null strides.
    auto describe = [&](cutensorTensorDescriptor_t* desc, std::string modes,
                        std::vector<int32_t>& out) {
      std::reverse(modes.begin(), modes.end());
      const auto ext = extents_of(modes, e);
      out.assign(modes.begin(), modes.end());
      check(cutensorCreateTensorDescriptor(h, desc, (uint32_t)modes.size(),
                                           ext.data(), nullptr, type, 256));
    };
    describe(&a, ma, mode_a);
    describe(&b, mb, mode_b);
    describe(&d, md, mode_d);
    check(cutensorCreateContraction(
        h, &op, a, mode_a.data(), CUTENSOR_OP_IDENTITY, b, mode_b.data(),
        CUTENSOR_OP_IDENTITY, d, mode_d.data(), CUTENSOR_OP_IDENTITY, d,
        mode_d.data(), compute));
    check(cutensorCreatePlanPreference(h, &pref, CUTENSOR_ALGO_DEFAULT,
                                       CUTENSOR_JIT_MODE_NONE));
    std::uint64_t estimate = 0;
    check(cutensorEstimateWorkspaceSize(h, op, pref, CUTENSOR_WORKSPACE_DEFAULT,
                                        &estimate));
    check(cutensorCreatePlan(h, &plan, op, pref, estimate));
    check(cutensorPlanGetAttribute(h, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                                   &workspace, sizeof(workspace)));
  }
  ~RawPlan() {
    cutensorDestroyPlan(plan);
    cutensorDestroyPlanPreference(pref);
    cutensorDestroyOperationDescriptor(op);
    cutensorDestroyTensorDescriptor(d);
    cutensorDestroyTensorDescriptor(b);
    cutensorDestroyTensorDescriptor(a);
  }
  RawPlan(const RawPlan&) = delete;
  RawPlan& operator=(const RawPlan&) = delete;

  std::vector<int32_t> mode_a, mode_b, mode_d;
};

void run_problem(cutensorHandle_t handle, cublasHandle_t blas, const Problem& p,
                 katal::Dtype dtype, const katal::Context& cached,
                 const katal::Context& uncached) {
  const auto arrow = p.equation.find("->");
  const auto comma = p.equation.find(',');
  const std::string ma = p.equation.substr(0, comma);
  const std::string mb = p.equation.substr(comma + 1, arrow - comma - 1);
  const std::string md = p.equation.substr(arrow + 2);
  const auto ea = extents_of(ma, p.extent);
  const auto eb = extents_of(mb, p.extent);
  const auto ed = extents_of(md, p.extent);

  const bool f64 = dtype == katal::Dtype::F64;
  const std::size_t word = f64 ? 8 : 4;
  const std::size_t bytes = (volume(ea) + volume(eb) + volume(ed)) * word;
  std::size_t free_bytes = 0, total_bytes = 0;
  example::check(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (bytes > free_bytes * 0.8) {
    std::printf("%-18s %-22s %s  skipped: needs %.1f GiB\n", p.equation.c_str(),
                p.label.c_str(), f64 ? "f64" : "f32", bytes / double(1 << 30));
    return;
  }

  double flop = 2.0;
  for (const auto& kv : p.extent) flop *= double(kv.second);

  const Device a(volume(ea) * word), b(volume(eb) * word), d(volume(ed) * word);
  const auto ra = katal::make_ref(static_cast<const void*>(a.get()), dtype, ea);
  const auto rb = katal::make_ref(static_cast<const void*>(b.get()), dtype, eb);
  const auto rd = katal::make_ref(d.get(), dtype, ed);

  // Raw cuTENSOR. Plan time is the second build, which hits cuTENSOR's own
  // plan cache as a Katal einsum miss would after warm-up.
  const auto type = f64 ? CUTENSOR_R_64F : CUTENSOR_R_32F;
  const auto compute =
      f64 ? CUTENSOR_COMPUTE_DESC_64F : CUTENSOR_COMPUTE_DESC_32F;
  auto t0 = std::chrono::steady_clock::now();
  RawPlan raw(handle, ma, mb, md, p.extent, type, compute);
  const double raw_plan_cold_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0)
                                      .count();
  void* raw_ws = nullptr;
  if (raw.workspace) example::check(cudaMalloc(&raw_ws, raw.workspace));
  const double one = 1.0, zero = 0.0;
  const float onef = 1.0f, zerof = 0.0f;
  const void* alpha = f64 ? static_cast<const void*>(&one) : &onef;
  const void* beta = f64 ? static_cast<const void*>(&zero) : &zerof;
  auto raw_call = [&] {
    check(cutensorContract(handle, raw.plan, alpha, a.get(), b.get(), beta,
                           d.get(), d.get(), raw_ws, raw.workspace, nullptr));
  };

  // Katal, hoisted.
  t0 = std::chrono::steady_clock::now();
  const katal::Plan plan =
      katal::make_plan(cached, katal::Equation::parse(p.equation), ra, rb, rd);
  const double katal_plan_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count();
  const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
      {}, plan.workspace_size(), {});
  const katal::Workspace ws = scratch.get();
  auto hoisted_call = [&] {
    plan.execute(1.0, a.get(), b.get(), 0.0, d.get(), d.get(), ws);
  };
  auto einsum_call = [&] {
    katal::einsum(p.equation, 1.0, ra, rb, 0.0, rd, {}, {}, cached);
  };
  auto uncached_call = [&] {
    katal::einsum(p.equation, 1.0, ra, rb, 0.0, rd, {}, {}, uncached);
  };

  // cuBLAS on the same buffers, as the ceiling for what the hardware does
  // with this matrix product. Both problems are one GEMM over row-major
  // data: ij,jk->ik is D = A B, and abcd,ijcd->ijab is D = B A^T with
  // (ij), (ab) and (cd) as the matrix dimensions. cuBLAS is column-major, so
  // it computes D^T.
  const bool ppl = p.equation == "abcd,ijcd->ijab";
  const int gm = static_cast<int>(ppl ? ed[0] * ed[1] : ed[0]);  // rows of D
  const int gn = static_cast<int>(ppl ? ed[2] * ed[3] : ed[1]);  // cols of D
  const int gk = static_cast<int>(ppl ? ea[2] * ea[3] : ea[1]);  // contracted
  auto blas_call = [&] {
    if (f64) {
      if (ppl)
        check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, gn, gm, gk, &one,
                          static_cast<const double*>(a.get()), gk,
                          static_cast<const double*>(b.get()), gk, &zero,
                          static_cast<double*>(d.get()), gn));
      else
        check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, gn, gm, gk, &one,
                          static_cast<const double*>(b.get()), gn,
                          static_cast<const double*>(a.get()), gk, &zero,
                          static_cast<double*>(d.get()), gn));
    } else {
      if (ppl)
        check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, gn, gm, gk, &onef,
                          static_cast<const float*>(a.get()), gk,
                          static_cast<const float*>(b.get()), gk, &zerof,
                          static_cast<float*>(d.get()), gn));
      else
        check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, gn, gm, gk, &onef,
                          static_cast<const float*>(b.get()), gn,
                          static_cast<const float*>(a.get()), gk, &zerof,
                          static_cast<float*>(d.get()), gn));
    }
  };

  const auto t = time_paths(
      {raw_call, hoisted_call, einsum_call, uncached_call, blas_call});
  const double t_raw = t[0], t_hoisted = t[1], t_einsum = t[2],
               t_uncached = t[3], t_blas = t[4];
  const double h_raw = host_us(raw_call, t_raw);
  const double h_einsum = host_us(einsum_call, t_raw);
  const double h_uncached = host_us(uncached_call, t_raw);

  auto tf = [&](double ms) { return flop / (ms * 1e-3) / 1e12; };
  std::printf(
      "%-18s %-22s %s %9.2f | %9.4f %7.2f | %9.4f %7.2f | %9.4f %7.2f | "
      "%9.4f %7.2f | %9.4f %7.2f | %6.3f | %7.1f %7.1f %8.1f | %8.2f %8.2f | "
      "%8.2f %8.2f\n",
      p.equation.c_str(), p.label.c_str(), f64 ? "f64" : "f32", flop / 1e9,
      t_raw, tf(t_raw), t_hoisted, tf(t_hoisted), t_einsum, tf(t_einsum),
      t_uncached, tf(t_uncached), t_blas, tf(t_blas), t_einsum / t_raw, h_raw,
      h_einsum, h_uncached, raw_plan_cold_ms, katal_plan_ms,
      raw.workspace / double(1 << 20), plan.workspace_size() / double(1 << 20));
  std::fflush(stdout);

  if (raw_ws) example::check(cudaFree(raw_ws));
}

std::vector<Problem> problems() {
  std::vector<Problem> list;
  for (std::int64_t n : {128, 256, 512, 1024, 2048, 4096, 8192, 12288})
    list.push_back({"ij,jk->ik",
                    {{'i', n}, {'j', n}, {'k', n}},
                    "n=" + std::to_string(n)});
  // Particle-particle ladder in the layout sequant-katal's ppl demo uses.
  const std::pair<std::int64_t, std::int64_t> ov[] = {
      {4, 32},   {8, 48},   {8, 64},   {16, 64},  {16, 96},
      {16, 128}, {24, 128}, {32, 128}, {24, 160}, {32, 160},
      {32, 192}, {48, 192}, {32, 224}, {64, 192}};
  for (auto [o, v] : ov)
    list.push_back(
        {"abcd,ijcd->ijab",
         {{'a', v}, {'b', v}, {'c', v}, {'d', v}, {'i', o}, {'j', o}},
         "o=" + std::to_string(o) + " v=" + std::to_string(v)});
  return list;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<katal::Dtype> types;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "f64") == 0) types.push_back(katal::Dtype::F64);
      if (std::strcmp(argv[i], "f32") == 0) types.push_back(katal::Dtype::F32);
    }
    if (types.empty()) types = {katal::Dtype::F64, katal::Dtype::F32};

    const katal::Runtime runtime;
    cudaDeviceProp prop{};
    example::check(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    std::printf(
        "device %s  sm_%d%d  %d SMs  %.0f MHz  %.1f GiB  cuTENSOR %zu\n",
        prop.name, prop.major, prop.minor, prop.multiProcessorCount,
        clock_khz / 1e3, prop.totalGlobalMem / double(1 << 30),
        cutensorGetVersion());
    std::printf(
        "columns: GFLOP | cutensor ms TF/s | hoisted ms TF/s | einsum ms TF/s "
        "| no-cache ms TF/s | cuBLAS ms TF/s | einsum/cutensor | host us: "
        "cutensor einsum "
        "no-cache | plan ms: raw-cold katal | workspace MiB: raw katal\n");

    cutensorHandle_t handle{};
    check(cutensorCreate(&handle));
    cublasHandle_t blas{};
    check(cublasCreate(&blas));
    const katal::Context cached;
    katal::Config off;
    off.plan_cache_entries = 0;
    const katal::Context uncached(off);

    for (katal::Dtype t : types)
      for (const Problem& p : problems()) {
        try {
          run_problem(handle, blas, p, t, cached, uncached);
        } catch (const std::exception& error) {
          std::printf("%-18s %-22s failed: %s\n", p.equation.c_str(),
                      p.label.c_str(), error.what());
        }
      }
    cublasDestroy(blas);
    cutensorDestroy(handle);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
