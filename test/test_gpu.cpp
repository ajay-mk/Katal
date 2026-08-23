#include <array>
#include <cstdint>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

#include "reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>

#include <cuein/plan.hpp>

using Catch::Matchers::WithinRel;

namespace {

bool have_gpu() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::int64_t elements(std::span<const std::int64_t> extent) {
  std::int64_t n = 1;
  for (std::int64_t e : extent) n *= e;
  return n;
}

template <class T>
std::vector<T> ramp(std::int64_t n, double scale) {
  std::vector<T> v(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = static_cast<T>(scale * ((i % 97) + 1) / 97.0);
  return v;
}

template <class T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(const std::vector<T>& host) : size_(host.size()) {
    REQUIRE(cudaMalloc(&ptr_, size_ * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(ptr_, host.data(), size_ * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  ~DeviceBuffer() { cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const noexcept { return ptr_; }

  std::vector<T> to_host() const {
    std::vector<T> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_, size_ * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

/// Runs the equation on the GPU and again through the host reference, which
/// never reverses anything, and requires the two to agree.
template <class T>
void agrees_with_reference(std::string_view equation,
                           std::span<const std::int64_t> ea,
                           std::span<const std::int64_t> eb,
                           std::span<const std::int64_t> ed, T alpha, T beta,
                           double tolerance) {
  const std::vector<T> a = ramp<T>(elements(ea), 1.0);
  const std::vector<T> b = ramp<T>(elements(eb), 0.5);
  const std::vector<T> d0 = ramp<T>(elements(ed), 0.25);

  DeviceBuffer<T> da(a), db(b), dd(d0);
  const cuein::TensorRef ar = cuein::make_ref(da.get(), cuein::dtype_of<T>, ea);
  const cuein::TensorRef br = cuein::make_ref(db.get(), cuein::dtype_of<T>, eb);
  const cuein::TensorRef dr = cuein::make_ref(dd.get(), cuein::dtype_of<T>, ed);

  cuein::einsum(equation, alpha, ar, br, beta, dr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<T> got = dd.to_host();

  std::vector<T> want = d0;
  const cuein::TensorRef ha =
      cuein::make_ref(const_cast<T*>(a.data()), cuein::dtype_of<T>, ea);
  const cuein::TensorRef hb =
      cuein::make_ref(const_cast<T*>(b.data()), cuein::dtype_of<T>, eb);
  const cuein::TensorRef hd =
      cuein::make_ref(want.data(), cuein::dtype_of<T>, ed);
  const std::array<cuein::TensorRef, 2> in{ha, hb};
  cuein::test::reference_einsum<T>(cuein::Equation::parse(equation), alpha, in,
                                   beta, hd);

  REQUIRE(got.size() == want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
    REQUIRE_THAT(static_cast<double>(got[i]),
                 WithinRel(static_cast<double>(want[i]), tolerance));
}

}  // namespace

TEST_CASE("gpu contraction matches the host reference", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // Asymmetric throughout: equal extents hide layout and stride bugs.
  SECTION("matrix product") {
    const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 4}, ed{2, 4};
    agrees_with_reference<double>("ij,jk->ik", ea, eb, ed, 1.0, 0.0, 1e-12);
  }

  SECTION("coupled-cluster shape") {
    const std::array<std::int64_t, 4> ea{2, 3, 4, 5}, eb{4, 5, 6, 7},
        ed{2, 3, 6, 7};
    agrees_with_reference<double>("ijab,abcd->ijcd", ea, eb, ed, 1.0, 0.0,
                                  1e-12);
  }

  SECTION("batched") {
    const std::array<std::int64_t, 3> ea{2, 3, 4}, eb{2, 4, 5}, ed{2, 3, 5};
    agrees_with_reference<double>("bij,bjk->bik", ea, eb, ed, 1.0, 0.0, 1e-12);
  }

  SECTION("outer product") {
    const std::array<std::int64_t, 2> ea{2, 3}, eb{4, 5};
    const std::array<std::int64_t, 4> ed{2, 3, 4, 5};
    agrees_with_reference<double>("ij,kl->ijkl", ea, eb, ed, 1.0, 0.0, 1e-12);
  }

  SECTION("alpha and beta accumulate into the output") {
    const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 4}, ed{2, 4};
    agrees_with_reference<double>("ij,jk->ik", ea, eb, ed, 2.0, 0.5, 1e-12);
  }

  SECTION("single precision") {
    const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 4}, ed{2, 4};
    agrees_with_reference<float>("ij,jk->ik", ea, eb, ed, 1.0F, 0.0F, 1e-5);
  }
}

namespace {

/// The one-operand overloads, against the same host reference.
template <class T>
void unary_agrees_with_reference(std::string_view equation,
                                 std::span<const std::int64_t> ea,
                                 std::span<const std::int64_t> ed, T alpha,
                                 T beta, double tolerance) {
  const std::vector<T> a = ramp<T>(elements(ea), 1.0);
  const std::vector<T> d0 = ramp<T>(elements(ed), 0.25);

  DeviceBuffer<T> da(a), dd(d0);
  const cuein::TensorRef ar = cuein::make_ref(da.get(), cuein::dtype_of<T>, ea);
  const cuein::TensorRef dr = cuein::make_ref(dd.get(), cuein::dtype_of<T>, ed);

  cuein::einsum(equation, alpha, ar, beta, dr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<T> got = dd.to_host();

  std::vector<T> want = d0;
  const cuein::TensorRef ha =
      cuein::make_ref(const_cast<T*>(a.data()), cuein::dtype_of<T>, ea);
  const cuein::TensorRef hd =
      cuein::make_ref(want.data(), cuein::dtype_of<T>, ed);
  const std::array<cuein::TensorRef, 1> in{ha};
  cuein::test::reference_einsum<T>(cuein::Equation::parse(equation), alpha, in,
                                   beta, hd);

  REQUIRE(got.size() == want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
    REQUIRE_THAT(static_cast<double>(got[i]),
                 WithinRel(static_cast<double>(want[i]), tolerance));
}

}  // namespace

TEST_CASE("gpu unary lowerings match the host reference", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 4> shape{2, 3, 4, 5};

  SECTION("permutation") {
    const std::array<std::int64_t, 4> ed{2, 4, 3, 5};
    unary_agrees_with_reference<double>("ijkl->ikjl", shape, ed, 1.0, 0.0,
                                        1e-12);
  }

  SECTION("reduction") {
    const std::array<std::int64_t, 2> ed{2, 3};
    unary_agrees_with_reference<double>("ijkl->ij", shape, ed, 1.0, 0.0, 1e-12);
  }

  SECTION("full reduction to a scalar") {
    unary_agrees_with_reference<double>("ijkl->", shape, {}, 1.0, 0.0, 1e-12);
  }

  SECTION("copy") {
    unary_agrees_with_reference<double>("ijkl->ijkl", shape, shape, 1.0, 0.0,
                                        1e-12);
  }

  SECTION("reduction accumulating with alpha and beta") {
    const std::array<std::int64_t, 2> ed{2, 3};
    unary_agrees_with_reference<double>("ijkl->ij", shape, ed, 2.0, 0.5, 1e-12);
  }

  SECTION("permutation in single precision") {
    const std::array<std::int64_t, 4> ed{2, 4, 3, 5};
    unary_agrees_with_reference<float>("ijkl->ikjl", shape, ed, 1.0F, 0.0F,
                                       1e-5);
  }
}

TEST_CASE("a plan can be built once and executed many times", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 4}, ed{2, 4};
  const std::vector<double> a = ramp<double>(6, 1.0);
  const std::vector<double> b = ramp<double>(12, 0.5);
  const std::vector<double> zero(8, 0.0);

  DeviceBuffer<double> da(a), db(b), dd(zero);
  const cuein::TensorRef ar = cuein::make_ref(da.get(), cuein::Dtype::F64, ea);
  const cuein::TensorRef br = cuein::make_ref(db.get(), cuein::Dtype::F64, eb);
  const cuein::TensorRef dr = cuein::make_ref(dd.get(), cuein::Dtype::F64, ed);

  const cuein::Equation eq = cuein::Equation::parse("ij,jk->ik");
  const cuein::Plan plan =
      cuein::make_plan(cuein::default_context(), eq, ar, br, dr);

  void* scratch = nullptr;
  if (plan.workspace_size() > 0)
    REQUIRE(cudaMalloc(&scratch, plan.workspace_size()) == cudaSuccess);
  const cuein::Workspace ws{scratch, plan.workspace_size()};

  // beta = 1 twice over a zeroed output must give exactly twice one pass.
  plan.execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(), ws);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<double> once = dd.to_host();

  plan.execute(1.0, da.get(), db.get(), 1.0, dd.get(), dd.get(), ws);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<double> twice = dd.to_host();

  for (std::size_t i = 0; i < once.size(); ++i)
    REQUIRE_THAT(twice[i], WithinRel(2.0 * once[i], 1e-12));

  cudaFree(scratch);
}

TEST_CASE("a caller's allocator supplies the scratch", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // Counts calls and forwards to plain cudaMalloc, so a failure to route
  // through the allocator shows up as a zero count rather than a wrong answer.
  struct CountingAllocator {
    int* allocations;
    int* deallocations;
    cuein::Workspace allocate(std::uint64_t bytes, cuein::StreamRef) {
      ++*allocations;
      void* ptr = nullptr;
      REQUIRE(cudaMalloc(&ptr, bytes) == cudaSuccess);
      return {ptr, bytes};
    }
    void deallocate(cuein::Workspace workspace, cuein::StreamRef) {
      ++*deallocations;
      cudaFree(workspace.ptr);
    }
  };
  static_assert(cuein::WorkspaceAllocator<CountingAllocator>);

  const std::array<std::int64_t, 4> ea{2, 3, 4, 5}, eb{4, 5, 6, 7},
      ed{2, 3, 6, 7};
  const std::vector<double> a = ramp<double>(elements(ea), 1.0);
  const std::vector<double> b = ramp<double>(elements(eb), 0.5);
  const std::vector<double> d(elements(ed), 0.0);
  DeviceBuffer<double> da(a), db(b), dd(d);

  const cuein::TensorRef ar = cuein::make_ref(da.get(), cuein::Dtype::F64, ea);
  const cuein::TensorRef br = cuein::make_ref(db.get(), cuein::Dtype::F64, eb);
  const cuein::TensorRef dr = cuein::make_ref(dd.get(), cuein::Dtype::F64, ed);

  int allocations = 0, deallocations = 0;
  cuein::einsum("ijab,abcd->ijcd", 1.0, ar, br, 0.0, dr, {},
                cuein::default_context(),
                CountingAllocator{&allocations, &deallocations});
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // The plan may need no scratch, but if it allocated it must have released.
  REQUIRE(allocations == deallocations);

  const std::vector<double> got = dd.to_host();
  bool nonzero = false;
  for (double v : got) nonzero = nonzero || v != 0.0;
  REQUIRE(nonzero);
}

TEST_CASE("the plan cache returns the same plan for the same problem",
          "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 4}, ed{2, 4};
  const std::array<std::int64_t, 2> wide_b{3, 5}, wide_d{2, 5};
  const std::vector<double> a = ramp<double>(6, 1.0);
  const std::vector<double> b = ramp<double>(20, 0.5);
  const std::vector<double> d(10, 0.0);
  DeviceBuffer<double> da(a), db(b), dd(d);

  const cuein::TensorRef ar = cuein::make_ref(da.get(), cuein::Dtype::F64, ea);
  const cuein::TensorRef br = cuein::make_ref(db.get(), cuein::Dtype::F64, eb);
  const cuein::TensorRef dr = cuein::make_ref(dd.get(), cuein::Dtype::F64, ed);
  const cuein::TensorRef wbr =
      cuein::make_ref(db.get(), cuein::Dtype::F64, wide_b);
  const cuein::TensorRef wdr =
      cuein::make_ref(dd.get(), cuein::Dtype::F64, wide_d);

  const cuein::Context& context = cuein::default_context();
  const auto first = cuein::cached_plan(context, "ij,jk->ik", ar, br, dr);
  const auto again = cuein::cached_plan(context, "ij,jk->ik", ar, br, dr);
  REQUIRE(first.get() == again.get());

  // A different extent is a different problem, so it must not hit.
  const auto wider = cuein::cached_plan(context, "ij,jk->ik", ar, wbr, wdr);
  REQUIRE(wider.get() != first.get());

  // So is a different equation. The transpose of a 2x3 is a 3x2.
  const std::array<std::int64_t, 2> transposed{3, 2};
  const cuein::TensorRef tr =
      cuein::make_ref(dd.get(), cuein::Dtype::F64, transposed);
  const auto permuted = cuein::cached_plan(context, "ij->ji", ar, tr);
  REQUIRE(permuted.get() != first.get());
}

TEST_CASE("every algorithm gives the same answer", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // The coupled-cluster shape, since that is where the algorithms differ.
  const std::array<std::int64_t, 4> ea{2, 3, 4, 5}, eb{4, 5, 6, 7},
      ed{2, 3, 6, 7};
  const std::vector<double> a = ramp<double>(elements(ea), 1.0);
  const std::vector<double> b = ramp<double>(elements(eb), 0.5);
  const std::vector<double> zero(elements(ed), 0.0);

  DeviceBuffer<double> da(a), db(b);
  const cuein::Equation eq = cuein::Equation::parse("ijab,abcd->ijcd");

  std::vector<double> reference;
  for (const cuein::Algo algo :
       {cuein::Algo::Default, cuein::Algo::DefaultPatient, cuein::Algo::Gett,
        cuein::Algo::Tgett, cuein::Algo::Ttgt})
    for (const cuein::WorkspacePreference workspace :
         {cuein::WorkspacePreference::Min, cuein::WorkspacePreference::Default,
          cuein::WorkspacePreference::Max}) {
      DeviceBuffer<double> dd(zero);
      const cuein::TensorRef ar =
          cuein::make_ref(da.get(), cuein::Dtype::F64, ea);
      const cuein::TensorRef br =
          cuein::make_ref(db.get(), cuein::Dtype::F64, eb);
      const cuein::TensorRef dr =
          cuein::make_ref(dd.get(), cuein::Dtype::F64, ed);

      cuein::PlanOptions options;
      options.algo = algo;
      options.workspace = workspace;

      // Not every algorithm is available for every problem and workspace, and
      // being told so is the correct outcome rather than a failure.
      try {
        const cuein::Plan plan =
            cuein::make_plan(cuein::default_context(), eq, ar, br, dr, options);
        void* scratch = nullptr;
        if (plan.workspace_size() > 0)
          REQUIRE(cudaMalloc(&scratch, plan.workspace_size()) == cudaSuccess);
        plan.execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(),
                     {scratch, plan.workspace_size()});
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cudaFree(scratch);
      } catch (const cuein::Error&) {
        continue;
      }

      const std::vector<double> got = dd.to_host();
      if (reference.empty()) {
        reference = got;
        continue;
      }
      for (std::size_t i = 0; i < got.size(); ++i)
        REQUIRE_THAT(got[i], WithinRel(reference[i], 1e-12));
    }

  REQUIRE_FALSE(reference.empty());
}

TEST_CASE("gpu rejections name the problem", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 2> ea{2, 3}, eb{5, 4}, ed{2, 4};
  const std::vector<double> a = ramp<double>(6, 1.0);
  const std::vector<double> b = ramp<double>(20, 0.5);
  const std::vector<double> d(8, 0.0);
  DeviceBuffer<double> da(a), db(b), dd(d);

  const cuein::TensorRef ar = cuein::make_ref(da.get(), cuein::Dtype::F64, ea);
  const cuein::TensorRef br = cuein::make_ref(db.get(), cuein::Dtype::F64, eb);
  const cuein::TensorRef dr = cuein::make_ref(dd.get(), cuein::Dtype::F64, ed);

  // j is 3 in the first operand and 5 in the second.
  REQUIRE_THROWS_AS(cuein::einsum("ij,jk->ik", 1.0, ar, br, 0.0, dr),
                    cuein::Error);
  // A free reduction beside a contraction stays rejected.
  REQUIRE_THROWS_AS(cuein::einsum("ijk,kl->il", 1.0, ar, br, 0.0, dr),
                    cuein::Error);
  // Arity mismatches between the equation and the overload are caught.
  REQUIRE_THROWS_AS(cuein::einsum("ij,jk->ik", 1.0, ar, 0.0, dr), cuein::Error);
  REQUIRE_THROWS_AS(cuein::einsum("ij->ji", 1.0, ar, br, 0.0, dr),
                    cuein::Error);
}
