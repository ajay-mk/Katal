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
/// once instead of at every call site. Passing one to `make_plan` or `einsum`
/// overrides that default for the one call.
struct PlanOptions {
  /// Accumulation type. Unset follows the output tensor, which is the only
  /// choice cuTENSOR always accepts; asking for f64 accumulation over f32
  /// storage, for instance, is not a supported combination. Set it to widen
  /// deliberately, or to opt into a reduced-precision mode such as TF32.
  std::optional<Dtype> compute;

  /// Caps the scratch a plan may use. 0 takes whatever `workspace` suggests.
  uint64_t workspace_limit = 0;

  Algo algo = Algo::Default;

  /// Never the worst choice on any shape measured so far.
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
