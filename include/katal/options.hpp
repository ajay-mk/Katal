#ifndef KATAL_OPTIONS_HPP_
#define KATAL_OPTIONS_HPP_

#include <cstdint>
#include <optional>

#include <katal/tensor_ref.hpp>

namespace katal {

/// Which algorithm family cuTENSOR should use. `Default` lets its performance
/// model choose, which is right unless you are measuring.
enum class Algo {
  Default,
  /// A slower, more thorough selection than Default.
  DefaultPatient,
  /// GEMM-like tensor contraction, needing no extra memory.
  Gett,
  /// Transpose one operand, then Gett.
  Tgett,
  /// Transpose, transpose, GEMM, transpose. Needs workspace, so a Min
  /// workspace preference can put it out of reach.
  Ttgt,
};

/// How much scratch cuTENSOR may assume when it picks a kernel. This gates
/// which algorithms are available, so it is not independent of `Algo`.
enum class WorkspacePreference {
  /// Least memory; at least one algorithm remains available.
  Min,
  /// High performance without an extravagant footprint.
  Default,
  /// Every algorithm available. Choose it when footprint does not matter.
  Max,
};

/// How to plan a contraction, as opposed to what to contract.
///
/// A Context carries one of these as its default, so a program can set a policy
/// once for `cached_plan` and `einsum`. An explicit option overrides that
/// policy. `make_plan` uses its own argument, defaulting to PlanOptions{};
/// it does not inherit the Context policy.
struct PlanOptions {
  /// Accumulation type. Unset follows the output tensor, except that F16 and
  /// Bf16 outputs accumulate in F32; those are the choices cuTENSOR always
  /// accepts. Asking for f64 accumulation over f32 storage, for instance, is
  /// not a supported combination. Set it to widen deliberately where
  /// supported. TF32 is not exposed by this interface.
  std::optional<Dtype> compute;

  /// Caps the scratch a plan may use. 0 takes whatever `workspace` suggests.
  uint64_t workspace_limit = 0;

  Algo algo = Algo::Default;

  WorkspacePreference workspace = WorkspacePreference::Default;

  /// Picks the n-th best kernel by cuTENSOR's own model, 0 being its first
  /// choice. Kernel variants differ in tiling and blocking, so this is as
  /// close to a tiling knob as cuTENSOR offers. Unset leaves the choice alone.
  std::optional<int32_t> kernel_rank;

  /// Lets cuTENSOR compile a kernel for this exact problem. Ampere and newer.
  /// Costs seconds at plan time.
  bool jit = false;
};

}  // namespace katal

#endif  // KATAL_OPTIONS_HPP_
