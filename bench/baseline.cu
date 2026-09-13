// Hand-written cuTENSOR contraction, timed. This is the reference Katal
// gets measured against: whatever the wrapper costs has to show up as a
// difference from these numbers.
//
// It answers three questions. What does setting up a contraction cost, cold
// and warm, against what the kernel itself costs? What does the workspace
// preference buy? And does handing cuTENSOR the fastest-varying mode first
// actually matter, or is reversing a row-major operand only a correctness
// concern?
//
// Build with bench/build_baseline.sh, which knows where NVHPC keeps
// cuTENSOR and emits code for both the A100 and H200 partitions.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#define CHECK_CUDA(expr)                                      \
  do {                                                        \
    const cudaError_t err_ = (expr);                          \
    if (err_ != cudaSuccess) {                                \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                 \
      std::exit(1);                                           \
    }                                                         \
  } while (0)

#define CHECK_CUTENSOR(expr)                                  \
  do {                                                        \
    const cutensorStatus_t st_ = (expr);                      \
    if (st_ != CUTENSOR_STATUS_SUCCESS) {                     \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                   cutensorGetErrorString(st_));              \
      std::exit(1);                                           \
    }                                                         \
  } while (0)

namespace {

struct Problem {
  const char* name;
  std::vector<int32_t> mode_a, mode_b, mode_d;
  std::map<int32_t, int64_t> extent;
};

std::vector<int64_t> extents_of(const std::vector<int32_t>& modes,
                                const std::map<int32_t, int64_t>& extent) {
  std::vector<int64_t> e;
  for (int32_t m : modes) e.push_back(extent.at(m));
  return e;
}

// Row major: the last mode varies fastest.
std::vector<int64_t> row_major_strides(const std::vector<int64_t>& extent) {
  std::vector<int64_t> s(extent.size(), 1);
  for (size_t i = extent.size(); i-- > 1;) s[i - 1] = s[i] * extent[i];
  return s;
}

template <class T>
std::vector<T> reversed(std::vector<T> v) {
  return std::vector<T>(v.rbegin(), v.rend());
}

int64_t elements(const std::vector<int64_t>& extent) {
  int64_t n = 1;
  for (int64_t e : extent) n *= e;
  return n;
}

__global__ void fill(double* p, int64_t n, double scale) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = scale * (double)((i % 97) + 1) / 97.0;
}

void launch_fill(double* p, int64_t n, double scale) {
  const int block = 256;
  fill<<<(unsigned)((n + block - 1) / block), block>>>(p, n, scale);
  CHECK_CUDA(cudaGetLastError());
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

/// Descriptors, plan preference, workspace estimate and plan: everything the
/// one-shot path pays on every call.
struct Setup {
  cutensorTensorDescriptor_t desc_a{}, desc_b{}, desc_d{};
  cutensorOperationDescriptor_t op{};
  cutensorPlanPreference_t pref{};
  cutensorPlan_t plan{};
  uint64_t workspace_size = 0;
};

Setup build(
    cutensorHandle_t handle, const std::vector<int32_t>& mode_a,
    const std::vector<int64_t>& ext_a, const std::vector<int64_t>& str_a,
    const std::vector<int32_t>& mode_b, const std::vector<int64_t>& ext_b,
    const std::vector<int64_t>& str_b, const std::vector<int32_t>& mode_d,
    const std::vector<int64_t>& ext_d, const std::vector<int64_t>& str_d,
    cutensorWorksizePreference_t worksize) {
  Setup s;
  constexpr uint32_t align = 256;

  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
      handle, &s.desc_a, (uint32_t)mode_a.size(), ext_a.data(), str_a.data(),
      CUTENSOR_R_64F, align));
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
      handle, &s.desc_b, (uint32_t)mode_b.size(), ext_b.data(), str_b.data(),
      CUTENSOR_R_64F, align));
  CHECK_CUTENSOR(cutensorCreateTensorDescriptor(
      handle, &s.desc_d, (uint32_t)mode_d.size(), ext_d.data(), str_d.data(),
      CUTENSOR_R_64F, align));

  CHECK_CUTENSOR(cutensorCreateContraction(
      handle, &s.op, s.desc_a, mode_a.data(), CUTENSOR_OP_IDENTITY, s.desc_b,
      mode_b.data(), CUTENSOR_OP_IDENTITY, s.desc_d, mode_d.data(),
      CUTENSOR_OP_IDENTITY, s.desc_d, mode_d.data(),
      CUTENSOR_COMPUTE_DESC_64F));

  CHECK_CUTENSOR(cutensorCreatePlanPreference(
      handle, &s.pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
  CHECK_CUTENSOR(cutensorEstimateWorkspaceSize(handle, s.op, s.pref, worksize,
                                               &s.workspace_size));
  CHECK_CUTENSOR(
      cutensorCreatePlan(handle, &s.plan, s.op, s.pref, s.workspace_size));

  return s;
}

void destroy(const Setup& s) {
  cutensorDestroyPlan(s.plan);
  cutensorDestroyPlanPreference(s.pref);
  cutensorDestroyOperationDescriptor(s.op);
  cutensorDestroyTensorDescriptor(s.desc_d);
  cutensorDestroyTensorDescriptor(s.desc_b);
  cutensorDestroyTensorDescriptor(s.desc_a);
}

const char* worksize_name(cutensorWorksizePreference_t w) {
  switch (w) {
    case CUTENSOR_WORKSPACE_MIN:
      return "MIN";
    case CUTENSOR_WORKSPACE_MAX:
      return "MAX";
    default:
      return "DEFAULT";
  }
}

}  // namespace

int main() {
  constexpr int warmup = 3;
  constexpr int iterations = 20;

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s  sm_%d%d\n\n", prop.name, prop.major, prop.minor);

  const std::vector<Problem> problems{
      // Control: a plain matrix product.
      {"ij,jk->ik",
       {'i', 'j'},
       {'j', 'k'},
       {'i', 'k'},
       {{'i', 4096}, {'j', 4096}, {'k', 4096}}},
      // A coupled-cluster shape: the particle-particle ladder term.
      {"ijab,abcd->ijcd",
       {'i', 'j', 'a', 'b'},
       {'a', 'b', 'c', 'd'},
       {'i', 'j', 'c', 'd'},
       {{'i', 16}, {'j', 16}, {'a', 96}, {'b', 96}, {'c', 96}, {'d', 96}}},
  };

  cutensorHandle_t handle{};
  CHECK_CUTENSOR(cutensorCreate(&handle));

  std::printf("%-18s %-8s %-9s %10s %10s %10s %10s %9s %s\n", "equation",
              "order", "worksize", "cold(ms)", "warm(ms)", "kernel(ms)",
              "TFLOP/s", "ws(MiB)", "checksum");
  std::printf("%s\n", std::string(110, '-').c_str());

  for (const Problem& p : problems) {
    const std::vector<int64_t> ea = extents_of(p.mode_a, p.extent);
    const std::vector<int64_t> eb = extents_of(p.mode_b, p.extent);
    const std::vector<int64_t> ed = extents_of(p.mode_d, p.extent);

    double* da = nullptr;
    double* db = nullptr;
    double* dd = nullptr;
    CHECK_CUDA(cudaMalloc(&da, elements(ea) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&db, elements(eb) * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dd, elements(ed) * sizeof(double)));
    launch_fill(da, elements(ea), 1.0);
    launch_fill(db, elements(eb), 0.5);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 2 * the product of every distinct extent, which is exact for a
    // contraction and needs no API call.
    double flop = 2.0;
    for (const auto& kv : p.extent) flop *= (double)kv.second;

    for (const bool reverse : {false, true}) {
      const auto ma = reverse ? reversed(p.mode_a) : p.mode_a;
      const auto mb = reverse ? reversed(p.mode_b) : p.mode_b;
      const auto md = reverse ? reversed(p.mode_d) : p.mode_d;
      const auto xa = reverse ? reversed(ea) : ea;
      const auto xb = reverse ? reversed(eb) : eb;
      const auto xd = reverse ? reversed(ed) : ed;
      const auto sa =
          reverse ? reversed(row_major_strides(ea)) : row_major_strides(ea);
      const auto sb =
          reverse ? reversed(row_major_strides(eb)) : row_major_strides(eb);
      const auto sd =
          reverse ? reversed(row_major_strides(ed)) : row_major_strides(ed);

      for (const cutensorWorksizePreference_t ws :
           {CUTENSOR_WORKSPACE_MIN, CUTENSOR_WORKSPACE_DEFAULT,
            CUTENSOR_WORKSPACE_MAX}) {
        auto t0 = std::chrono::steady_clock::now();
        Setup cold = build(handle, ma, xa, sa, mb, xb, sb, md, xd, sd, ws);
        const double cold_s = seconds_since(t0);

        t0 = std::chrono::steady_clock::now();
        Setup warm = build(handle, ma, xa, sa, mb, xb, sb, md, xd, sd, ws);
        const double warm_s = seconds_since(t0);
        destroy(warm);

        void* workspace = nullptr;
        if (cold.workspace_size > 0)
          CHECK_CUDA(cudaMalloc(&workspace, cold.workspace_size));

        const double alpha = 1.0;
        const double beta = 0.0;
        for (int i = 0; i < warmup; ++i)
          CHECK_CUTENSOR(cutensorContract(handle, cold.plan, &alpha, da, db,
                                          &beta, dd, dd, workspace,
                                          cold.workspace_size, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEvent_t start{};
        cudaEvent_t stop{};
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i)
          CHECK_CUTENSOR(cutensorContract(handle, cold.plan, &alpha, da, db,
                                          &beta, dd, dd, workspace,
                                          cold.workspace_size, nullptr));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float total_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
        const double kernel_ms = total_ms / iterations;

        // Reversing the modes must not change the answer; if it does, the
        // relabelling is not the no-op it is claimed to be.
        std::vector<double> host(elements(ed));
        CHECK_CUDA(cudaMemcpy(host.data(), dd, host.size() * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double checksum = 0.0;
        for (double v : host) checksum += v;

        std::printf("%-18s %-8s %-9s %10.3f %10.3f %10.3f %10.2f %9.1f %.6e\n",
                    p.name, reverse ? "reversed" : "as-written",
                    worksize_name(ws), cold_s * 1e3, warm_s * 1e3, kernel_ms,
                    flop / (kernel_ms * 1e-3) / 1e12,
                    (double)cold.workspace_size / (1024.0 * 1024.0), checksum);

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        if (workspace) CHECK_CUDA(cudaFree(workspace));
        destroy(cold);
      }
    }

    std::printf("%s\n", std::string(110, '-').c_str());
    CHECK_CUDA(cudaFree(dd));
    CHECK_CUDA(cudaFree(db));
    CHECK_CUDA(cudaFree(da));
  }

  CHECK_CUTENSOR(cutensorDestroy(handle));
  return 0;
}
