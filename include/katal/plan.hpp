#ifndef KATAL_PLAN_HPP_
#define KATAL_PLAN_HPP_

#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <katal/context.hpp>
#include <katal/equation.hpp>
#include <katal/options.hpp>
#include <katal/tensor_ref.hpp>

namespace katal {

/// Scratch Katal does not own.
struct Workspace {
  /// Device memory owned by the caller, which must stay alive until the
  /// stream the plan ran on has completed.
  void* ptr = nullptr;
  /// Bytes available at `ptr`, at least `Plan::workspace_size`.
  uint64_t size = 0;
};

/// Supplies scratch for a plan. A caller with its own device pool implements
/// this and hands it over instead of Katal allocating on its own.
///
/// Taken by value, so an implementation should be a cheap handle that refers
/// to the pool rather than owning it.
/// Deallocation must not throw; it is also called during exception unwinding.
template <class A>
concept WorkspaceAllocator =
    requires(A allocator, uint64_t bytes, StreamRef stream, Workspace w) {
      { allocator.allocate(bytes, stream) } -> std::same_as<Workspace>;
      { allocator.deallocate(w, stream) } -> std::same_as<void>;
    };

/// The default: stream-ordered cudaMallocAsync and cudaFreeAsync. Its
/// definitions live in the compiled library so this header stays free of CUDA.
class CudaAsyncAllocator {
 public:
  Workspace allocate(uint64_t bytes, StreamRef stream) const;
  void deallocate(Workspace workspace, StreamRef stream) const noexcept;
};

namespace util {

/// Releases scratch on scope exit, including when execute throws.
template <WorkspaceAllocator A>
class ScopedWorkspace {
 public:
  ScopedWorkspace(A allocator, uint64_t bytes, StreamRef stream)
      : allocator_(allocator), stream_(stream) {
    if (bytes > 0) workspace_ = allocator_.allocate(bytes, stream);
  }

  ~ScopedWorkspace() {
    if (workspace_.ptr != nullptr) allocator_.deallocate(workspace_, stream_);
  }

  ScopedWorkspace(const ScopedWorkspace&) = delete;
  ScopedWorkspace& operator=(const ScopedWorkspace&) = delete;

  Workspace get() const noexcept { return workspace_; }

 private:
  A allocator_;
  Workspace workspace_;
  StreamRef stream_;
};

}  // namespace util

/// A selected kernel, ready to run. Building one is the expensive part; running
/// it is not, so a residual evaluation should build its plans once and execute
/// them every iteration.
/// Borrows its Context: the Context must outlive the Plan and all submitted
/// work. A cached shared_ptr retains the Plan, not its Context. Runs only on
/// its Context's device, and a moved-from Plan may only be destroyed or
/// assigned to.
class Plan {
 public:
  Plan(Plan&&) noexcept;
  Plan& operator=(Plan&&) noexcept;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  ~Plan();

  /// Bytes of scratch `execute` needs. Exact rather than estimated: it is
  /// what the selected kernel was planned against. Often zero.
  uint64_t workspace_size() const noexcept;

  /// The equation this plan came from, for diagnostics. The view is valid only
  /// while the plan is.
  std::string_view equation() const noexcept;

  /// Computes `d = alpha * <equation over a and b> + beta * c`.
  ///
  /// The pointers are device memory shaped as the TensorRefs this plan was
  /// built from; anything else is undefined rather than diagnosed. `c` and `d`
  /// may be the same pointer, which is how to accumulate.
  ///
  /// Throws `Errc::WorkspaceTooSmall` if `workspace` is smaller than
  /// `workspace_size()`, before any work is queued.
  ///
  /// Const and allocation-free, so one plan can be shared across streams and
  /// run on a hot path.
  void execute(Scalar alpha, const void* a, const void* b, Scalar beta,
               const void* c, void* d, Workspace workspace,
               StreamRef stream = {}) const;

  /// The one-operand form, for a permutation, a reduction or a copy. Throws
  /// if this plan was built from two operands.
  void execute(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
               Workspace workspace, StreamRef stream = {}) const;

  class Impl;

 private:
  explicit Plan(std::unique_ptr<Impl> impl);
  const Impl& impl() const noexcept;
  friend Plan make_plan(const Context&, const Equation&, const TensorRef&,
                        const TensorRef&, const TensorRef&, PlanOptions);
  friend Plan make_plan(const Context&, const Equation&, const TensorRef&,
                        const TensorRef&, PlanOptions);

  std::unique_ptr<Impl> impl_;
};

/// Selects a kernel for this equation at these shapes. Far more expensive than
/// executing it, so it belongs outside any loop.
///
/// Throws `Errc::UnsupportedPattern` for a pattern Katal does not lower,
/// `Errc::ExtentMismatch` when a shared label disagrees across operands, and
/// `Errc::CutensorError` when cuTENSOR refuses the descriptor.
/// A cuTENSOR NOT_SUPPORTED status is reported as UnsupportedPattern.
Plan make_plan(const Context& context, const Equation& equation,
               const TensorRef& a, const TensorRef& b, const TensorRef& d,
               PlanOptions options = {});

/// The one-operand form. Permutation, reduction and copy all go through
/// cuTENSOR's reduction, its only unary entry point that accepts beta.
Plan make_plan(const Context& context, const Equation& equation,
               const TensorRef& a, const TensorRef& d,
               PlanOptions options = {});

/// Looks up a plan in the context's cache, building it on a miss. Returns a
/// shared plan so the caller keeps it alive even if the cache evicts it, and
/// so a disabled cache costs the same call.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, const TensorRef& a,
    const TensorRef& b, const TensorRef& d,
    std::optional<PlanOptions> options = {});

/// The one-operand form.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, const TensorRef& a,
    const TensorRef& d, std::optional<PlanOptions> options = {});

/// Look up a cached plan, allocate scratch and run, in one call.
///
/// Parsing and planning happen on a cache miss. Every call still looks up the
/// plan and allocates scratch; hoist a Plan and its workspace out of a hot loop
/// to avoid that overhead.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void einsum(std::string_view equation, Scalar alpha, const TensorRef& a,
            const TensorRef& b, Scalar beta, const TensorRef& d,
            std::optional<PlanOptions> options = {}, StreamRef stream = {},
            const Context& context = default_context(), A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, a, b, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, b.data, beta, d.data, d.data, scratch.get(),
                stream);
}

/// The one-operand form: a permutation, a reduction, or a copy.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void einsum(std::string_view equation, Scalar alpha, const TensorRef& a,
            Scalar beta, const TensorRef& d,
            std::optional<PlanOptions> options = {}, StreamRef stream = {},
            const Context& context = default_context(), A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, a, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, beta, d.data, d.data, scratch.get(), stream);
}

}  // namespace katal

#endif  // KATAL_PLAN_HPP_
