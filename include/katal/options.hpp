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

/// A faster, lower-precision way for cuTENSOR to compute, beyond what the
/// `compute` type alone can say. Each runs on tensor cores and needs a
/// recent GPU and particular storage types, according to cuTENSOR's
/// contraction table; anything else is refused at plan time as
/// `Errc::UnsupportedPattern`. Katal passes these through untested.
enum class Precision {
  /// Exactly the `compute` type. The default.
  Native,
  /// F32 or C32 data in TF32: 10 mantissa bits. Ampere and newer.
  Tf32,
  /// F32 or C32 data in three TF32 terms: more precise than TF32, less than
  /// F32. Ampere and newer.
  Tf32x3,
  /// F32 or C32 data in three bf16 terms, 23 mantissa bits. Hopper and
  /// newer.
  Bf16x9,
  /// F32 or C32 data in two f16 terms, 23 mantissa bits. Blackwell and newer.
  F16x4,
  /// F64 or C64 data emulated with int8 terms, chosen at run time to be at
  /// least as accurate as F64. Hopper and newer.
  Int8x8,
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
  /// supported; `precision` covers TF32 and the other tensor-core modes.
  std::optional<Dtype> compute;

  /// Caps the scratch a plan may use, including any input it sums ahead of a
  /// contraction. 0 takes whatever `workspace` suggests.
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

  /// Replaces `compute` with a tensor-core mode when not Native. Opt in per
  /// call, or once as the Context's default.
  Precision precision = Precision::Native;
};

}  // namespace katal

#endif  // KATAL_OPTIONS_HPP_
