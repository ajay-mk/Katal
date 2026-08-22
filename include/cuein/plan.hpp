#ifndef CUEIN_PLAN_HPP_
#define CUEIN_PLAN_HPP_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <cuein/context.hpp>
#include <cuein/equation.hpp>
#include <cuein/tensor_ref.hpp>

namespace cuein {

/// Scratch cuEin does not own.
struct Workspace {
  /// Device memory owned by the caller, which must stay alive until the
  /// stream the plan ran on has completed.
  void* ptr = nullptr;
  /// Bytes available at `ptr`, at least `Plan::workspace_size`.
  uint64_t size = 0;
};

struct PlanOptions {
  /// Accumulation type. Unset follows the output tensor, which is the only
  /// choice cuTENSOR always accepts; asking for f64 accumulation over f32
  /// storage, for instance, is not a supported combination. Set it to widen
  /// deliberately, or to opt into a reduced-precision mode such as TF32.
  std::optional<Dtype> compute;
  /// 0 means cuTENSOR's DEFAULT preference, which is never the worst choice
  /// and is measurably better than MIN on some shapes.
  uint64_t workspace_limit = 0;
};

/// A selected kernel, ready to run. Building one is the expensive part; running
/// it is not, so a residual evaluation should build its plans once and execute
/// them every iteration.
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
  /// The pointers are device memory laid out as the TensorRefs the plan was
  /// built from described; passing differently shaped memory is undefined
  /// rather than diagnosed. `c` and `d` may be the same pointer, which is the
  /// usual way to accumulate.
  ///
  /// Const because executing does not mutate the plan, which is what lets one
  /// plan be shared across streams. Allocation-free, so it is safe on a hot
  /// path.
  void execute(Scalar alpha, const void* a, const void* b, Scalar beta,
               const void* c, void* d, Workspace workspace,
               StreamRef stream = {}) const;

  class Impl;

 private:
  explicit Plan(std::unique_ptr<Impl> impl);
  friend Plan make_plan(const Context&, const Equation&, const TensorRef&,
                        const TensorRef&, const TensorRef&, PlanOptions);

  std::unique_ptr<Impl> impl_;
};

/// Selects a kernel for this equation at these shapes. Costs on the order of
/// a hundred microseconds the first time and around ten on a cuTENSOR plan
/// cache hit, so it belongs outside any loop that runs it.
///
/// Throws `Error` with `Errc::UnsupportedPattern` for a pattern cuEin does not
/// lower, `Errc::ExtentMismatch` when a shared label disagrees across
/// operands, and `Errc::CutensorError` when cuTENSOR refuses the descriptor.
Plan make_plan(const Context& context, const Equation& equation,
               const TensorRef& a, const TensorRef& b, const TensorRef& d,
               PlanOptions options = {});

/// Parse, plan, allocate scratch and run, in one call.
///
/// Convenient, and it pays for parsing and descriptor construction on every
/// call, plus a stream-ordered allocation for the workspace. That is a small
/// fraction of a large contraction and most of a small one, so a loop over
/// tile-sized work should build a `Plan` once and execute it instead.
void einsum(std::string_view equation, Scalar alpha, const TensorRef& a,
            const TensorRef& b, Scalar beta, const TensorRef& d,
            StreamRef stream = {}, const Context& context = default_context());

}  // namespace cuein

#endif  // CUEIN_PLAN_HPP_
