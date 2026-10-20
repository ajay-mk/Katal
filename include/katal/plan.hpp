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

/// How an elementwise plan combines its two inputs, or how a reduction
/// combines the elements it reduces.
enum class BinaryOp {
  Add,
  Multiply,
  /// The larger of the two.
  Max,
  /// The smaller of the two.
  Min,
};

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

  /// Bytes of scratch `execute` needs. Exact rather than estimated: what the
  /// selected kernels were planned against, plus room for any input summed
  /// ahead of a contraction. Often zero.
  uint64_t workspace_size() const;

  /// The equation this plan came from, for diagnostics. The view is valid only
  /// while the plan is.
  std::string_view equation() const;

  /// Computes `d = alpha * <equation over a and b> + beta * c`.
  ///
  /// The pointers are device memory shaped as the TensorRefs this plan was
  /// built from; anything else is undefined rather than diagnosed, except
  /// that a pointer less aligned than its TensorRef was throws
  /// `Errc::UnsupportedPattern`. `c` and `d` may be the same pointer, which is
  /// how to accumulate.
  ///
  /// Throws `Errc::WorkspaceTooSmall` if `workspace` is smaller than
  /// `workspace_size()`, before any work is queued. A plan that sums a label
  /// out of one input first keeps that input in the workspace, and throws
  /// `Errc::UnsupportedPattern` unless `workspace.ptr` is 256-byte aligned.
  ///
  /// Const and allocation-free, so one plan can be shared across streams and
  /// run on a hot path.
  void execute(Scalar alpha, const void* a, const void* b, Scalar beta,
               const void* c, void* d, Workspace workspace,
               StreamRef stream = {}) const;

  /// The three-operand form: `e = alpha * <equation over a, b and c> + beta *
  /// d`, contracted in one kernel. Named as cuTENSOR names it, with d the
  /// tensor added in and e the result; `d` and `e` may be the same pointer,
  /// which is how to accumulate. Throws if this plan was not built from three
  /// operands. Otherwise as the two-operand form.
  void execute(Scalar alpha, const void* a, const void* b, const void* c,
               Scalar beta, const void* d, void* e, Workspace workspace,
               StreamRef stream = {}) const;

  /// The one-operand form, for a permutation, a reduction or a copy. Throws
  /// if this plan was built from two operands. A plan from
  /// `make_permutation_plan` computes `d = alpha * a`, ignores `c`, and
  /// throws `Errc::UnsupportedPattern` if beta is not zero or if the bytes
  /// `a` addresses overlap those `d` does.
  void execute(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
               Workspace workspace, StreamRef stream = {}) const;

  /// The elementwise form: `d = op(alpha * a, beta * b) + gamma * c`, where
  /// an input lacking a label of the result is broadcast along it. Throws if
  /// this plan is not elementwise.
  ///
  /// `c` and `d` may be the same pointer, which is how to accumulate; when
  /// gamma is zero, `c` is not read. `a` and `b` must not overlap `d`, and
  /// passing either as `d` throws `Errc::UnsupportedPattern`. A zero alpha or
  /// beta skips that input, so only `BinaryOp::Add` accepts one; the other
  /// operators throw `Errc::UnsupportedPattern`.
  void execute(Scalar alpha, const void* a, Scalar beta, const void* b,
               Scalar gamma, const void* c, void* d, Workspace workspace,
               StreamRef stream = {}) const;

  class Impl;

 private:
  explicit Plan(std::unique_ptr<Impl> impl);
  const Impl& impl() const;
  friend Plan make_plan(const Context&, const Equation&, const ConstTensorRef&,
                        const ConstTensorRef&, const ConstTensorRef&,
                        PlanOptions);
  friend Plan make_plan(const Context&, const Equation&, const ConstTensorRef&,
                        const ConstTensorRef&, const ConstTensorRef&,
                        const ConstTensorRef&, PlanOptions);
  friend Plan make_plan(const Context&, const Equation&, const ConstTensorRef&,
                        const ConstTensorRef&, PlanOptions);
  friend Plan make_plan(const Context&, const Equation&, BinaryOp,
                        const ConstTensorRef&, const ConstTensorRef&,
                        const ConstTensorRef&, PlanOptions);
  friend Plan make_plan(const Context&, const Equation&, BinaryOp,
                        const ConstTensorRef&, const ConstTensorRef&,
                        PlanOptions);
  friend Plan make_permutation_plan(const Context&, const Equation&,
                                    const ConstTensorRef&,
                                    const ConstTensorRef&, PlanOptions);

  std::unique_ptr<Impl> impl_;
};

/// Selects a kernel for this equation at these shapes. Far more expensive than
/// executing it, so it belongs outside any loop.
///
/// Throws `Errc::UnsupportedPattern` for a pattern Katal does not lower,
/// `Errc::ExtentMismatch` when a shared label disagrees across operands, and
/// `Errc::CutensorError` when cuTENSOR refuses the descriptor.
/// A cuTENSOR NOT_SUPPORTED status is reported as UnsupportedPattern.
///
/// Only shapes, types and alignments are read; no data pointer is, so every
/// operand, the output included, is taken read-only.
Plan make_plan(const Context& context, const Equation& equation,
               const ConstTensorRef& a, const ConstTensorRef& b,
               const ConstTensorRef& d, PlanOptions options = {});

/// The three-operand form, `e = alpha * A.B.C + beta * d`, as one kernel
/// through cuTENSOR's trinary contraction rather than two pairwise ones, so no
/// intermediate is stored. `"ij,jk,kl->il"` is a chain of two matrix
/// products. A label in one input alone is summed out of it first, as in the
/// two-operand form. A pattern or type cuTENSOR's trinary contraction does not
/// support throws `Errc::UnsupportedPattern`; Katal does not fall back to
/// pairwise contractions.
Plan make_plan(const Context& context, const Equation& equation,
               const ConstTensorRef& a, const ConstTensorRef& b,
               const ConstTensorRef& c, const ConstTensorRef& e,
               PlanOptions options = {});

/// The one-operand form. Permutation, reduction and copy all go through
/// cuTENSOR's reduction, its only unary entry point that accepts beta.
Plan make_plan(const Context& context, const Equation& equation,
               const ConstTensorRef& a, const ConstTensorRef& d,
               PlanOptions options = {});

/// The elementwise form, `d = op(alpha * a, beta * b) + gamma * c`. The
/// equation names how the inputs line up with the result, as in
/// `"jiab,ijab->ijab"` or, broadcasting b along i and j, `"ijab,ab->ijab"`.
/// Every input label must appear in the result; see
/// `Equation::classify_elementwise`.
Plan make_plan(const Context& context, const Equation& equation, BinaryOp op,
               const ConstTensorRef& a, const ConstTensorRef& b,
               const ConstTensorRef& d, PlanOptions options = {});

/// The one-operand form with a choice of reduction: `d = alpha * reduce(a) +
/// beta * c`, combining the elements over each label the result drops with
/// `reduce` instead of summing them. `"ij->i"` with `BinaryOp::Max` takes
/// each row's maximum.
Plan make_plan(const Context& context, const Equation& equation,
               BinaryOp reduce, const ConstTensorRef& a,
               const ConstTensorRef& d, PlanOptions options = {});

/// A dedicated permutation, `d = alpha * a`, through cuTENSOR's own
/// permutation kernel rather than a reduction. The result may also
/// broadcast `a` along labels it lacks, as in `"i->ij"`, and may have a
/// different type, such as F32 from F16 or F64; unless `options.compute` is
/// set, a conversion accumulates in the wider of the two types. See
/// `Equation::classify_permutation`.
Plan make_permutation_plan(const Context& context, const Equation& equation,
                           const ConstTensorRef& a, const ConstTensorRef& d,
                           PlanOptions options = {});

/// Looks up a plan in the context's cache, building it on a miss. Returns a
/// shared plan so the caller keeps it alive even if the cache evicts it, and
/// so a disabled cache costs the same call.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, const ConstTensorRef& a,
    const ConstTensorRef& b, const ConstTensorRef& d,
    std::optional<PlanOptions> options = {});

/// The three-operand form.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, const ConstTensorRef& a,
    const ConstTensorRef& b, const ConstTensorRef& c, const ConstTensorRef& e,
    std::optional<PlanOptions> options = {});

/// The one-operand form.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, const ConstTensorRef& a,
    const ConstTensorRef& d, std::optional<PlanOptions> options = {});

/// The elementwise form.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, BinaryOp op,
    const ConstTensorRef& a, const ConstTensorRef& b, const ConstTensorRef& d,
    std::optional<PlanOptions> options = {});

/// The one-operand form with a choice of reduction.
std::shared_ptr<const Plan> cached_plan(
    const Context& context, std::string_view equation, BinaryOp reduce,
    const ConstTensorRef& a, const ConstTensorRef& d,
    std::optional<PlanOptions> options = {});

/// The permutation form.
std::shared_ptr<const Plan> cached_permutation_plan(
    const Context& context, std::string_view equation, const ConstTensorRef& a,
    const ConstTensorRef& d, std::optional<PlanOptions> options = {});

/// Look up a cached plan, allocate scratch and run, in one call.
///
/// Parsing and planning happen on a cache miss. Every call still looks up the
/// plan and allocates scratch; hoist a Plan and its workspace out of a hot loop
/// to avoid that overhead.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void einsum(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
            const ConstTensorRef& b, Scalar beta, const TensorRef& d,
            std::optional<PlanOptions> options = {}, StreamRef stream = {},
            const Context& context = default_context(), A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, a, b, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, b.data, beta, d.data, d.data, scratch.get(),
                stream);
}

/// The three-operand form: `e = alpha * A.B.C + beta * e` in one kernel, as
/// in `einsum("ij,jk,kl->il", 1.0, a, b, c, 0.0, e)`.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void einsum(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
            const ConstTensorRef& b, const ConstTensorRef& c, Scalar beta,
            const TensorRef& e, std::optional<PlanOptions> options = {},
            StreamRef stream = {}, const Context& context = default_context(),
            A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, a, b, c, e, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, b.data, c.data, beta, e.data, e.data,
                scratch.get(), stream);
}

/// The one-operand form: a permutation, a reduction, or a copy.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void einsum(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
            Scalar beta, const TensorRef& d,
            std::optional<PlanOptions> options = {}, StreamRef stream = {},
            const Context& context = default_context(), A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, a, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, beta, d.data, d.data, scratch.get(), stream);
}

/// `d = alpha * reduce(a) + beta * d` over the labels the result drops, with
/// the same caching as einsum. `reduce("ij->i", BinaryOp::Max, 1.0, a, 0.0,
/// d)` takes each row's maximum.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void reduce(std::string_view equation, BinaryOp op, Scalar alpha,
            const ConstTensorRef& a, Scalar beta, const TensorRef& d,
            std::optional<PlanOptions> options = {}, StreamRef stream = {},
            const Context& context = default_context(), A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, op, a, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, beta, d.data, d.data, scratch.get(), stream);
}

/// `d = alpha * a`, permuted, broadcast and converted to d's type, through a
/// dedicated permutation kernel. `a` must not overlap `d`, which throws. To
/// accumulate into d,
/// use the one-operand einsum.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void permute(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
             const TensorRef& d, std::optional<PlanOptions> options = {},
             StreamRef stream = {}, const Context& context = default_context(),
             A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_permutation_plan(context, equation, a, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, 0.0, d.data, d.data, scratch.get(), stream);
}

/// `d = op(alpha * a, beta * b) + gamma * d`, with the same caching as
/// einsum. A label an input lacks is broadcast, so `"ij,j->ij"` adds a row
/// vector to every row. `a` and `b` must not be `d`.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void elementwise(std::string_view equation, BinaryOp op, Scalar alpha,
                 const ConstTensorRef& a, Scalar beta, const ConstTensorRef& b,
                 Scalar gamma, const TensorRef& d,
                 std::optional<PlanOptions> options = {}, StreamRef stream = {},
                 const Context& context = default_context(), A allocator = {}) {
  const std::shared_ptr<const Plan> plan =
      cached_plan(context, equation, op, a, b, d, options);
  const util::ScopedWorkspace<A> scratch(allocator, plan->workspace_size(),
                                         stream);
  plan->execute(alpha, a.data, beta, b.data, gamma, d.data, d.data,
                scratch.get(), stream);
}

/// `d = alpha * a + beta * b`, overwriting d. For example,
/// `add("jiab,ijab->ijab", 1.0, t, -1.0, t, d)` antisymmetrizes t.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void add(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
         Scalar beta, const ConstTensorRef& b, const TensorRef& d,
         std::optional<PlanOptions> options = {}, StreamRef stream = {},
         const Context& context = default_context(), A allocator = {}) {
  elementwise(equation, BinaryOp::Add, alpha, a, beta, b, 0.0, d, options,
              stream, context, allocator);
}

/// `d = (alpha * a) * (beta * b)` elementwise, overwriting d.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void multiply(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
              Scalar beta, const ConstTensorRef& b, const TensorRef& d,
              std::optional<PlanOptions> options = {}, StreamRef stream = {},
              const Context& context = default_context(), A allocator = {}) {
  elementwise(equation, BinaryOp::Multiply, alpha, a, beta, b, 0.0, d, options,
              stream, context, allocator);
}

/// `d = max(alpha * a, beta * b)` elementwise, overwriting d.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void max(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
         Scalar beta, const ConstTensorRef& b, const TensorRef& d,
         std::optional<PlanOptions> options = {}, StreamRef stream = {},
         const Context& context = default_context(), A allocator = {}) {
  elementwise(equation, BinaryOp::Max, alpha, a, beta, b, 0.0, d, options,
              stream, context, allocator);
}

/// `d = min(alpha * a, beta * b)` elementwise, overwriting d.
template <WorkspaceAllocator A = CudaAsyncAllocator>
void min(std::string_view equation, Scalar alpha, const ConstTensorRef& a,
         Scalar beta, const ConstTensorRef& b, const TensorRef& d,
         std::optional<PlanOptions> options = {}, StreamRef stream = {},
         const Context& context = default_context(), A allocator = {}) {
  elementwise(equation, BinaryOp::Min, alpha, a, beta, b, 0.0, d, options,
              stream, context, allocator);
}

}  // namespace katal

#endif  // KATAL_PLAN_HPP_
