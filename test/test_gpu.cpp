#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

#include "cuda_memory.hpp"
#include "reference.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>

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
    T* data = nullptr;
    REQUIRE(cudaMalloc(&data, size_ * sizeof(T)) == cudaSuccess);
    ptr_.reset(data);
    REQUIRE(cudaMemcpy(ptr_.get(), host.data(), size_ * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() const noexcept { return ptr_.get(); }

  std::vector<T> to_host() const {
    std::vector<T> host(size_);
    REQUIRE(cudaMemcpy(host.data(), ptr_.get(), size_ * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }

 private:
  std::unique_ptr<T, decltype(&katal::test::free_device)> ptr_{
      nullptr, &katal::test::free_device};
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
  const katal::TensorRef ar = katal::make_ref(da.get(), katal::dtype_of<T>, ea);
  const katal::TensorRef br = katal::make_ref(db.get(), katal::dtype_of<T>, eb);
  const katal::TensorRef dr = katal::make_ref(dd.get(), katal::dtype_of<T>, ed);

  katal::einsum(equation, alpha, ar, br, beta, dr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<T> got = dd.to_host();

  std::vector<T> want = d0;
  const katal::TensorRef ha =
      katal::make_ref(const_cast<T*>(a.data()), katal::dtype_of<T>, ea);
  const katal::TensorRef hb =
      katal::make_ref(const_cast<T*>(b.data()), katal::dtype_of<T>, eb);
  const katal::TensorRef hd =
      katal::make_ref(want.data(), katal::dtype_of<T>, ed);
  const std::array<katal::TensorRef, 2> in{ha, hb};
  katal::test::reference_einsum<T>(katal::Equation::parse(equation), alpha, in,
                                   beta, hd);

  REQUIRE(got.size() == want.size());
  for (std::size_t i = 0; i < got.size(); ++i)
    REQUIRE_THAT(static_cast<double>(got[i]),
                 WithinRel(static_cast<double>(want[i]), tolerance));
}

}  // namespace

TEST_CASE("complex padded tensors execute across layouts", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  using Complex = std::complex<double>;
  std::vector<Complex> a(10), b(6), initial(8, Complex{-17.0, 3.0});
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] = {double(i + 1), -0.5 * double(i)};
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = {0.25 * double(i + 1), 1.0};
  DeviceBuffer<Complex> da(a), db(b), dd(initial);
  const std::array<std::int64_t, 2> ea{2, 3}, eb{3, 2}, ed{2, 2};
  const std::array<std::int64_t, 2> sa{5, 1}, sd{1, 4};
  auto ar = katal::make_ref(da.get(), katal::Dtype::C64, ea);
  ar.stride = sa;
  auto br = katal::make_ref(db.get(), katal::Dtype::C64, eb,
                            katal::Layout::ColumnMajor);
  auto dr = katal::make_ref(dd.get(), katal::Dtype::C64, ed,
                            katal::Layout::ColumnMajor);
  dr.stride = sd;
  const Complex alpha{0.75, -0.5}, beta{-0.25, 0.5};
  katal::einsum("ij,jk->ik", alpha, ar, br, beta, dr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  auto expected = initial;
  ar.data = a.data();
  br.data = b.data();
  dr.data = expected.data();
  const std::array<katal::TensorRef, 2> inputs{ar, br};
  katal::test::reference_einsum<Complex>(katal::Equation::parse("ij,jk->ik"),
                                         alpha, inputs, beta, dr);
  const auto got = dd.to_host();
  for (std::size_t i = 0; i < got.size(); ++i)
    REQUIRE(std::abs(got[i] - expected[i]) < 1e-11);
}

TEST_CASE("a retained plan executes after cache eviction", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");
  katal::Config config;
  config.plan_cache_entries = 1;
  const katal::Context context(config);
  DeviceBuffer<double> a(std::vector<double>{1, 2, 3, 4});
  DeviceBuffer<double> b(std::vector<double>{5, 6, 7, 8});
  DeviceBuffer<double> d(std::vector<double>(4));
  const std::array<std::int64_t, 2> extent{2, 2};
  const auto ar = katal::make_ref(a.get(), katal::Dtype::F64, extent);
  const auto br = katal::make_ref(b.get(), katal::Dtype::F64, extent);
  const auto dr = katal::make_ref(d.get(), katal::Dtype::F64, extent);
  const auto retained = katal::cached_plan(context, "ij,jk->ik", ar, br, dr);
  const auto evict = katal::cached_plan(context, "ij->ji", ar, dr);
  const auto rebuilt = katal::cached_plan(context, "ij,jk->ik", ar, br, dr);
  REQUIRE(rebuilt.get() != retained.get());
  const katal::util::ScopedWorkspace scratch(katal::CudaAsyncAllocator{},
                                             retained->workspace_size(),
                                             katal::StreamRef{});
  retained->execute(1.0, a.get(), b.get(), 0.0, d.get(), d.get(),
                    scratch.get());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(d.to_host() == std::vector<double>{19, 22, 43, 50});
}

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
  const katal::TensorRef ar = katal::make_ref(da.get(), katal::dtype_of<T>, ea);
  const katal::TensorRef dr = katal::make_ref(dd.get(), katal::dtype_of<T>, ed);

  katal::einsum(equation, alpha, ar, beta, dr);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<T> got = dd.to_host();

  std::vector<T> want = d0;
  const katal::TensorRef ha =
      katal::make_ref(const_cast<T*>(a.data()), katal::dtype_of<T>, ea);
  const katal::TensorRef hd =
      katal::make_ref(want.data(), katal::dtype_of<T>, ed);
  const std::array<katal::TensorRef, 1> in{ha};
  katal::test::reference_einsum<T>(katal::Equation::parse(equation), alpha, in,
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
  const katal::TensorRef ar = katal::make_ref(da.get(), katal::Dtype::F64, ea);
  const katal::TensorRef br = katal::make_ref(db.get(), katal::Dtype::F64, eb);
  const katal::TensorRef dr = katal::make_ref(dd.get(), katal::Dtype::F64, ed);

  const katal::Equation eq = katal::Equation::parse("ij,jk->ik");
  const katal::Plan plan =
      katal::make_plan(katal::default_context(), eq, ar, br, dr);

  void* scratch = nullptr;
  if (plan.workspace_size() > 0)
    REQUIRE(cudaMalloc(&scratch, plan.workspace_size()) == cudaSuccess);
  const std::unique_ptr<void, decltype(&katal::test::free_device)> storage(
      scratch, &katal::test::free_device);
  const katal::Workspace ws{storage.get(), plan.workspace_size()};

  // beta = 1 twice over a zeroed output must give exactly twice one pass.
  plan.execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(), ws);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<double> once = dd.to_host();

  plan.execute(1.0, da.get(), db.get(), 1.0, dd.get(), dd.get(), ws);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const std::vector<double> twice = dd.to_host();

  for (std::size_t i = 0; i < once.size(); ++i)
    REQUIRE_THAT(twice[i], WithinRel(2.0 * once[i], 1e-12));
}

TEST_CASE("a caller's allocator supplies the scratch", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  // Counts calls and forwards to plain cudaMalloc, so a failure to route
  // through the allocator shows up as a zero count rather than a wrong answer.
  struct CountingAllocator {
    int* allocations;
    int* deallocations;
    katal::Workspace allocate(std::uint64_t bytes, katal::StreamRef) {
      ++*allocations;
      void* ptr = nullptr;
      REQUIRE(cudaMalloc(&ptr, bytes) == cudaSuccess);
      return {ptr, bytes};
    }
    void deallocate(katal::Workspace workspace, katal::StreamRef) {
      ++*deallocations;
      katal::test::free_device(workspace.ptr);
    }
  };
  static_assert(katal::WorkspaceAllocator<CountingAllocator>);

  const std::array<std::int64_t, 4> ea{2, 3, 4, 5}, eb{4, 5, 6, 7},
      ed{2, 3, 6, 7};
  const std::vector<double> a = ramp<double>(elements(ea), 1.0);
  const std::vector<double> b = ramp<double>(elements(eb), 0.5);
  const std::vector<double> d(elements(ed), 0.0);
  DeviceBuffer<double> da(a), db(b), dd(d);

  const katal::TensorRef ar = katal::make_ref(da.get(), katal::Dtype::F64, ea);
  const katal::TensorRef br = katal::make_ref(db.get(), katal::Dtype::F64, eb);
  const katal::TensorRef dr = katal::make_ref(dd.get(), katal::Dtype::F64, ed);

  int allocations = 0, deallocations = 0;
  katal::einsum("ijab,abcd->ijcd", 1.0, ar, br, 0.0, dr, {}, {},
                katal::default_context(),
                CountingAllocator{&allocations, &deallocations});
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // The plan may need no scratch, but if it allocated it must have released.
  REQUIRE(allocations == deallocations);

  const std::vector<double> got = dd.to_host();
  std::vector<double> want(d.size(), 0.0);
  const std::array<katal::TensorRef, 2> host{
      katal::make_ref(const_cast<double*>(a.data()), katal::Dtype::F64, ea),
      katal::make_ref(const_cast<double*>(b.data()), katal::Dtype::F64, eb)};
  katal::test::reference_einsum<double>(
      katal::Equation::parse("ijab,abcd->ijcd"), 1.0, host, 0.0,
      katal::make_ref(want.data(), katal::Dtype::F64, ed));
  for (std::size_t i = 0; i < got.size(); ++i)
    REQUIRE_THAT(got[i], WithinRel(want[i], 1e-12));
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

  const katal::TensorRef ar = katal::make_ref(da.get(), katal::Dtype::F64, ea);
  const katal::TensorRef br = katal::make_ref(db.get(), katal::Dtype::F64, eb);
  const katal::TensorRef dr = katal::make_ref(dd.get(), katal::Dtype::F64, ed);
  const katal::TensorRef wbr =
      katal::make_ref(db.get(), katal::Dtype::F64, wide_b);
  const katal::TensorRef wdr =
      katal::make_ref(dd.get(), katal::Dtype::F64, wide_d);

  const katal::Context& context = katal::default_context();
  const auto first = katal::cached_plan(context, "ij,jk->ik", ar, br, dr);
  const auto again = katal::cached_plan(context, "ij,jk->ik", ar, br, dr);
  REQUIRE(first.get() == again.get());

  // A different extent is a different problem, so it must not hit.
  const auto wider = katal::cached_plan(context, "ij,jk->ik", ar, wbr, wdr);
  REQUIRE(wider.get() != first.get());

  // So is a different equation. The transpose of a 2x3 is a 3x2.
  const std::array<std::int64_t, 2> transposed{3, 2};
  const katal::TensorRef tr =
      katal::make_ref(dd.get(), katal::Dtype::F64, transposed);
  const auto permuted = katal::cached_plan(context, "ij->ji", ar, tr);
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
  const katal::Equation eq = katal::Equation::parse("ijab,abcd->ijcd");

  std::vector<double> reference(zero.size(), 0.0);
  const std::array<katal::TensorRef, 2> host{
      katal::make_ref(const_cast<double*>(a.data()), katal::Dtype::F64, ea),
      katal::make_ref(const_cast<double*>(b.data()), katal::Dtype::F64, eb)};
  katal::test::reference_einsum<double>(
      eq, 1.0, host, 0.0,
      katal::make_ref(reference.data(), katal::Dtype::F64, ed));
  bool default_ran = false;
  for (const katal::Algo algo :
       {katal::Algo::Default, katal::Algo::DefaultPatient, katal::Algo::Gett,
        katal::Algo::Tgett, katal::Algo::Ttgt})
    for (const katal::WorkspacePreference workspace :
         {katal::WorkspacePreference::Min, katal::WorkspacePreference::Default,
          katal::WorkspacePreference::Max}) {
      DeviceBuffer<double> dd(zero);
      const katal::TensorRef ar =
          katal::make_ref(da.get(), katal::Dtype::F64, ea);
      const katal::TensorRef br =
          katal::make_ref(db.get(), katal::Dtype::F64, eb);
      const katal::TensorRef dr =
          katal::make_ref(dd.get(), katal::Dtype::F64, ed);

      katal::PlanOptions options;
      options.algo = algo;
      options.workspace = workspace;

      // Only an explicitly unsupported planning combination may be skipped.
      std::optional<katal::Plan> plan;
      try {
        plan.emplace(katal::make_plan(katal::default_context(), eq, ar, br, dr,
                                      options));
      } catch (const katal::Error& error) {
        REQUIRE(error.code() == katal::Errc::UnsupportedPattern);
        REQUIRE_FALSE((algo == katal::Algo::Default &&
                       workspace == katal::WorkspacePreference::Default));
        continue;
      }
      const katal::util::ScopedWorkspace scratch(katal::CudaAsyncAllocator{},
                                                 plan->workspace_size(),
                                                 katal::StreamRef{});
      plan->execute(1.0, da.get(), db.get(), 0.0, dd.get(), dd.get(),
                    scratch.get());
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      if (algo == katal::Algo::Default &&
          workspace == katal::WorkspacePreference::Default)
        default_ran = true;

      const std::vector<double> got = dd.to_host();
      for (std::size_t i = 0; i < got.size(); ++i)
        REQUIRE_THAT(got[i], WithinRel(reference[i], 1e-12));
    }

  REQUIRE(default_ran);
}

TEST_CASE("gpu rejections name the problem", "[gpu]") {
  if (!have_gpu()) SKIP("no CUDA device");

  const std::array<std::int64_t, 2> ea{2, 3}, eb{5, 4}, ed{2, 4};
  const std::vector<double> a = ramp<double>(6, 1.0);
  const std::vector<double> b = ramp<double>(20, 0.5);
  const std::vector<double> d(8, 0.0);
  DeviceBuffer<double> da(a), db(b), dd(d);

  const katal::TensorRef ar = katal::make_ref(da.get(), katal::Dtype::F64, ea);
  const katal::TensorRef br = katal::make_ref(db.get(), katal::Dtype::F64, eb);
  const katal::TensorRef dr = katal::make_ref(dd.get(), katal::Dtype::F64, ed);

  // j is 3 in the first operand and 5 in the second.
  REQUIRE_THROWS_AS(katal::einsum("ij,jk->ik", 1.0, ar, br, 0.0, dr),
                    katal::Error);
  // A free reduction beside a contraction stays rejected.
  REQUIRE_THROWS_AS(katal::einsum("ijk,kl->il", 1.0, ar, br, 0.0, dr),
                    katal::Error);
  // Arity mismatches between the equation and the overload are caught.
  REQUIRE_THROWS_AS(katal::einsum("ij,jk->ik", 1.0, ar, 0.0, dr), katal::Error);
  REQUIRE_THROWS_AS(katal::einsum("ij->ji", 1.0, ar, br, 0.0, dr),
                    katal::Error);
}
