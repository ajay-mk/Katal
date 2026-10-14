#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "context_impl.hpp"
#include "log.hpp"

#include <katal/plan.hpp>

#if KATAL_ENABLE_LOGGING
#include <chrono>
#endif

namespace katal {
namespace {

/// The cache key, built as one string so that std::hash and string equality
/// do the work: an exact key means no collision handling to get wrong.
void append(std::string& key, const void* bytes, std::size_t n) {
  if (n != 0) key.append(static_cast<const char*>(bytes), n);
}

void append_tensor(std::string& key, const ConstTensorRef& t) {
  const auto type = static_cast<int32_t>(t.type);
  const auto layout = static_cast<int32_t>(t.layout);
  append(key, &type, sizeof(type));
  append(key, &layout, sizeof(layout));
  append(key, &t.alignment, sizeof(t.alignment));
  const auto op = static_cast<int32_t>(t.op);
  append(key, &op, sizeof(op));
  const auto extents = static_cast<uint32_t>(t.extent.size());
  const auto strides = static_cast<uint32_t>(t.stride.size());
  append(key, &extents, sizeof(extents));
  append(key, &strides, sizeof(strides));
  append(key, t.extent.data(), t.extent.size_bytes());
  append(key, t.stride.data(), t.stride.size_bytes());
}

/// `op` is -1 for an einsum plan, which the tensor count already tells apart
/// from a one-operand one; the BinaryOp for an elementwise plan or a
/// reduction, which the tensor count tells apart too; and `permutation_key`
/// for a permutation.
constexpr int32_t permutation_key = 1000;

std::string plan_key(std::string_view equation, PlanOptions options,
                     std::initializer_list<const ConstTensorRef*> tensors,
                     int32_t op = -1) {
  std::string key(equation);
  key.push_back('\0');
  append(key, &op, sizeof(op));
  // Only the two-operand einsum contracts; every other form ignores the
  // precision, so it must not split the cache. A tensor-core mode replaces
  // the compute type, which then must not split it either.
  if (op != -1 || tensors.size() != 3) options.precision = Precision::Native;
  if (options.precision != Precision::Native) options.compute.reset();
  const int32_t compute =
      options.compute ? static_cast<int32_t>(*options.compute) : -1;
  append(key, &compute, sizeof(compute));
  append(key, &options.workspace_limit, sizeof(options.workspace_limit));
  const auto algo = static_cast<int32_t>(options.algo);
  const auto workspace = static_cast<int32_t>(options.workspace);
  const int32_t rank = options.kernel_rank.value_or(-1);
  const auto jit = static_cast<int32_t>(options.jit);
  const auto precision = static_cast<int32_t>(options.precision);
  append(key, &precision, sizeof(precision));
  append(key, &algo, sizeof(algo));
  append(key, &workspace, sizeof(workspace));
  append(key, &rank, sizeof(rank));
  append(key, &jit, sizeof(jit));
  for (const ConstTensorRef* t : tensors) append_tensor(key, *t);
  return key;
}

/// A label's byte value is already a unique, operand-independent mode id.
std::vector<int32_t> mode_ids(const util::Index& modes) {
  std::vector<int32_t> ids;
  ids.reserve(modes.rank());
  for (const auto label : modes)
    ids.push_back(static_cast<unsigned char>(label));
  return ids;
}

/// A shared label must have one extent. cuTENSOR would reject a mismatch too,
/// but without naming the label responsible.
void check_extents(
    const Equation& equation,
    std::initializer_list<std::pair<const util::Index*, const ConstTensorRef*>>
        operands) {
  std::array<std::int64_t, 256> extent{};
  for (const auto& [modes, tensor] : operands)
    for (std::size_t p = 0; p < modes->rank(); ++p) {
      const auto slot = static_cast<unsigned char>((*modes)[p]);
      const std::int64_t e = tensor->extent[p];
      if (e <= 0)
        throw Error(Errc::ExtentMismatch, equation.text(),
                    "GPU operands require positive extents");
      if (extent[slot] != 0 && extent[slot] != e)
        throw Error(Errc::ExtentMismatch, equation.text(),
                    "label \"" + std::string(1, (*modes)[p]) +
                        "\" has extent " + std::to_string(extent[slot]) +
                        " in one operand and " + std::to_string(e) +
                        " in another");
      extent[slot] = e;
    }
}

/// The bytes a tensor addresses, as offsets from its data pointer: `low` is
/// its lowest addressed byte and `high` one past its highest. A negative
/// stride reaches below the pointer.
struct ByteSpan {
  std::int64_t low = 0;
  std::int64_t high = 0;
};

std::int64_t element_bytes(Dtype type) {
  switch (type) {
    case Dtype::F16:
    case Dtype::Bf16:
      return 2;
    case Dtype::F32:
      return 4;
    case Dtype::F64:
    case Dtype::C32:
      return 8;
    case Dtype::C64:
      return 16;
  }
  return 16;
}

/// Extents are positive by now; see check_extents. A view whose reach does
/// not fit std::int64_t cannot be compared, so it is refused.
ByteSpan byte_span(const backend::Operand& t, std::string_view equation) {
  constexpr std::int64_t most = std::numeric_limits<std::int64_t>::max();
  const auto overflow = [&] {
    return Error(Errc::ExtentMismatch, equation,
                 "a tensor's byte span overflows std::int64_t");
  };
  const std::int64_t bytes = element_bytes(t.type);
  ByteSpan span{0, bytes};
  for (std::size_t i = 0; i < t.extent.size(); ++i) {
    const std::int64_t steps = t.extent[i] - 1;
    if (steps == 0) continue;
    // |steps * stride * bytes| <= most, checked without overflowing.
    const std::int64_t limit = most / bytes / steps;
    if (t.stride[i] > limit || t.stride[i] < -limit) throw overflow();
    const std::int64_t reach = steps * t.stride[i] * bytes;
    if (reach < 0) {
      if (span.low < -most - reach) throw overflow();
      span.low += reach;
    } else {
      if (span.high > most - reach) throw overflow();
      span.high += reach;
    }
  }
  return span;
}

/// Whether two planned tensors at these pointers share any byte. Compared as
/// integers, since the pointers may come from different allocations.
bool overlaps(const void* x, ByteSpan xs, const void* y, ByteSpan ys) {
  const auto at = [](const void* p, std::int64_t offset) {
    return static_cast<std::intmax_t>(reinterpret_cast<std::uintptr_t>(p)) +
           offset;
  };
  return at(x, xs.low) < at(y, ys.high) && at(y, ys.low) < at(x, xs.high);
}

/// What cuTENSOR assumes of every base pointer it is given without being told
/// otherwise, and so where each scratch tensor in a workspace starts.
constexpr std::uint64_t scratch_alignment = 256;

std::uint64_t round_up(std::uint64_t bytes) {
  return (bytes + scratch_alignment - 1) / scratch_alignment *
         scratch_alignment;
}

}  // namespace

/// The frontend of a plan: everything about an equation that does not depend
/// on which library runs it. Shapes are checked and lowered to a
/// backend::Operation here; the backend's Kernel holds the rest.
class Plan::Impl {
 public:
  /// Which execute overload a plan answers to.
  enum class Form { Contraction, OneOperand, Elementwise, Permutation };

  /// Selects the permutation constructor, whose arguments match the
  /// one-operand einsum's.
  struct PermutationTag {};

  Impl(const Context& context, const Equation& equation,
       const ConstTensorRef& a, const ConstTensorRef& b,
       const ConstTensorRef& d, const PlanOptions& options)
      : Impl(context, equation, Form::Contraction) {
    const Classification classification = equation.classify();
    if (classification.kind != Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "only binary contraction is implemented so far");

    backend::Operation op;
    op.kind = backend::OpKind::Contraction;
    op.inputs = {operand(equation.operand(0), a),
                 operand(equation.operand(1), b)};
    op.output = operand(equation.result(), d);
    check_extents(equation, {{&equation.operand(0), &a},
                             {&equation.operand(1), &b},
                             {&equation.result(), &d}});
    // A label in one input alone is summed out of it first, into scratch,
    // since a contraction kernel only sums labels both inputs share.
    prereduce(0, op.inputs[0], classification.reduced_a);
    prereduce(1, op.inputs[1], classification.reduced_b);
    const PlanOptions kernel_options = scratch_budget(options);
    plan_prereductions(kernel_options);
    finish(std::move(op), d, kernel_options);
    alignment_ = {a.alignment, b.alignment, d.alignment};
  }

  /// The one-operand form: a permutation, a reduction or a copy.
  Impl(const Context& context, const Equation& equation,
       const ConstTensorRef& a, const ConstTensorRef& d,
       const PlanOptions& options)
      : Impl(context, equation, BinaryOp::Add, a, d, options) {}

  /// The one-operand form, reducing with `reduce` rather than a sum.
  Impl(const Context& context, const Equation& equation, BinaryOp reduce,
       const ConstTensorRef& a, const ConstTensorRef& d,
       const PlanOptions& options)
      : Impl(context, equation, Form::OneOperand) {
    if (equation.classify().kind == Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "this equation has two operands; use the two-operand form");

    backend::Operation op;
    op.kind = backend::OpKind::Reduction;
    op.binary = reduce;
    op.inputs = {operand(equation.operand(0), a)};
    op.output = operand(equation.result(), d);
    check_extents(equation,
                  {{&equation.operand(0), &a}, {&equation.result(), &d}});
    finish(std::move(op), d, options);
    alignment_ = {a.alignment, 0, d.alignment};
  }

  /// The elementwise form.
  Impl(const Context& context, const Equation& equation, BinaryOp binary,
       const ConstTensorRef& a, const ConstTensorRef& b,
       const ConstTensorRef& d, const PlanOptions& options)
      : Impl(context, equation, Form::Elementwise) {
    equation.classify_elementwise();

    backend::Operation op;
    op.kind = backend::OpKind::Elementwise;
    op.binary = binary;
    binary_ = binary;
    op.inputs = {operand(equation.operand(0), a),
                 operand(equation.operand(1), b)};
    op.output = operand(equation.result(), d);
    check_extents(equation, {{&equation.operand(0), &a},
                             {&equation.operand(1), &b},
                             {&equation.result(), &d}});
    finish(std::move(op), d, options);
    alignment_ = {a.alignment, b.alignment, d.alignment};
  }

  /// A dedicated permutation, `d = alpha * a`.
  Impl(const Context& context, const Equation& equation, PermutationTag,
       const ConstTensorRef& a, const ConstTensorRef& d,
       const PlanOptions& options)
      : Impl(context, equation, Form::Permutation) {
    equation.classify_permutation();

    backend::Operation op;
    op.kind = backend::OpKind::Permutation;
    op.inputs = {operand(equation.operand(0), a)};
    op.output = operand(equation.result(), d);
    check_extents(equation,
                  {{&equation.operand(0), &a}, {&equation.result(), &d}});
    span_a_ = byte_span(op.inputs[0], equation_);
    span_d_ = byte_span(op.output, equation_);
    PlanOptions widened = options;
    if (!widened.compute) widened.compute = conversion_compute(a.type, d.type);
    finish(std::move(op), d, widened);
    alignment_ = {a.alignment, 0, d.alignment};
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  uint64_t workspace_size() const noexcept { return workspace_size_; }
  std::string_view equation() const noexcept { return equation_; }

  Form form() const noexcept { return form_; }

  void contract(Scalar alpha, const void* a, const void* b, Scalar beta,
                const void* c, void* d, Workspace workspace,
                StreamRef stream) const {
    backend_->check_device(equation_, "plan");
    check_alignment("a", a, alignment_[0]);
    check_alignment("b", b, alignment_[1]);
    check_alignment("c", c, alignment_[2]);
    check_alignment("d", d, alignment_[2]);
    check_workspace(workspace);
    std::array<const void*, 2> inputs{a, b};
    Workspace scratch = workspace;
    if (scratch_offset_ > 0) {
      if (alignment_of(workspace.ptr) < scratch_alignment)
        throw Error(Errc::UnsupportedPattern, equation_,
                    "the workspace is aligned to " +
                        std::to_string(alignment_of(workspace.ptr)) +
                        " bytes but this plan keeps reduced inputs in it and "
                        "needs " +
                        std::to_string(scratch_alignment));
      auto* base = static_cast<std::byte*>(workspace.ptr);
      scratch = {base + scratch_offset_, workspace.size - scratch_offset_};
      const std::array<Scalar, 2> overwrite{1.0, 0.0};
      for (std::size_t k = 0; k < pre_.size(); ++k) {
        if (pre_[k].kernel == nullptr) continue;
        void* reduced = base + pre_[k].offset;
        const std::array<const void*, 1> input{inputs[k]};
        pre_[k].kernel->run(overwrite, input, reduced, reduced, scratch,
                            stream);
        inputs[k] = reduced;
      }
    }
    const std::array<Scalar, 2> scalars{alpha, beta};
    kernel_->run(scalars, inputs, c, d, scratch, stream);
  }

  void reduce(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
              Workspace workspace, StreamRef stream) const {
    backend_->check_device(equation_, "plan");
    check_alignment("a", a, alignment_[0]);
    check_alignment("c", c, alignment_[2]);
    check_alignment("d", d, alignment_[2]);
    check_workspace(workspace);
    const std::array<Scalar, 2> scalars{alpha, beta};
    const std::array<const void*, 1> inputs{a};
    kernel_->run(scalars, inputs, c, d, workspace, stream);
  }

  void permute(Scalar alpha, const void* a, Scalar beta, void* d,
               Workspace workspace, StreamRef stream) const {
    if (beta.value() != 0.0)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "a permutation plan writes d = alpha * a and cannot add "
                  "beta * c; use einsum for that");
    backend_->check_device(equation_, "plan");
    // cuTENSOR requires the elements read from a not to overlap those written
    // to d, so a pointer into the same buffer is refused, not only d itself.
    if (overlaps(a, span_a_, d, span_d_))
      throw Error(Errc::UnsupportedPattern, equation_,
                  "pointer a overlaps the output; a permutation runs out of "
                  "place");
    check_alignment("a", a, alignment_[0]);
    check_alignment("d", d, alignment_[2]);
    check_workspace(workspace);
    const std::array<Scalar, 1> scalars{alpha};
    const std::array<const void*, 1> inputs{a};
    kernel_->run(scalars, inputs, nullptr, d, workspace, stream);
  }

  void combine(Scalar alpha, const void* a, Scalar beta, const void* b,
               Scalar gamma, const void* c, void* d, Workspace workspace,
               StreamRef stream) const {
    backend_->check_device(equation_, "plan");
    // cuTENSOR requires the inputs not to overlap the output. Only the exact
    // alias is cheap to spot; it is also the likely mistake.
    if (a == d || b == d)
      throw Error(Errc::UnsupportedPattern, equation_,
                  std::string("pointer ") + (a == d ? "a" : "b") +
                      " is also the output; an elementwise input must not "
                      "overlap d");
    // cuTENSOR skips an input whose scale is zero instead of combining a zero
    // tensor. A sum is the same either way; the other operators are not, so
    // refuse rather than return max(alpha * a) for max(alpha * a, 0). Tested
    // after narrowing to the compute precision, where 1e-50 is already zero.
    const bool narrow = compute_ != Dtype::F64 && compute_ != Dtype::C64;
    const auto is_zero = [narrow](Scalar s) {
      const std::complex<double> v = s.value();
      if (!narrow) return v == 0.0;
      return static_cast<float>(v.real()) == 0.0F &&
             static_cast<float>(v.imag()) == 0.0F;
    };
    if (binary_ != BinaryOp::Add && (is_zero(alpha) || is_zero(beta)))
      throw Error(Errc::UnsupportedPattern, equation_,
                  std::string(is_zero(alpha) ? "alpha" : "beta") +
                      " is zero, which drops that input instead of using "
                      "zeros; only Add allows a zero scale");
    check_alignment("a", a, alignment_[0]);
    check_alignment("b", b, alignment_[1]);
    check_alignment("c", c, alignment_[2]);
    check_alignment("d", d, alignment_[2]);
    check_workspace(workspace);
    const std::array<Scalar, 3> scalars{alpha, beta, gamma};
    const std::array<const void*, 2> inputs{a, b};
    kernel_->run(scalars, inputs, c, d, workspace, stream);
  }

 private:
  Impl(const Context& context, const Equation& equation, Form form)
      : backend_(&context.impl().backend()),
        equation_(equation.text()),
        form_(form) {
    backend_->check_device(equation_, "plan");
  }

  /// Puts a tensor into descriptor order, with labels as mode ids. A label
  /// repeated within the tensor becomes one mode whose stride is the sum of
  /// the repeats', which walks the diagonal.
  backend::Operand operand(const util::Index& labels,
                           const ConstTensorRef& tensor) const {
    const ModeOrder order = mode_order(labels, tensor, equation_);
    const std::vector<int32_t> ids = mode_ids(order.modes);
    backend::Operand result;
    for (std::size_t p = 0; p < ids.size(); ++p) {
      const auto seen =
          std::find(result.modes.begin(), result.modes.end(), ids[p]);
      if (seen == result.modes.end()) {
        result.modes.push_back(ids[p]);
        result.extent.push_back(order.extent[p]);
        result.stride.push_back(order.stride[p]);
        continue;
      }
      const auto q = static_cast<std::size_t>(seen - result.modes.begin());
      if (result.extent[q] != order.extent[p])
        throw Error(Errc::ExtentMismatch, equation_,
                    "label \"" + std::string(1, order.modes[p]) +
                        "\" is repeated with extents " +
                        std::to_string(result.extent[q]) + " and " +
                        std::to_string(order.extent[p]) +
                        "; a diagonal needs them equal");
      if (__builtin_add_overflow(result.stride[q], order.stride[p],
                                 &result.stride[q]))
        throw Error(Errc::ExtentMismatch, equation_,
                    "label \"" + std::string(1, order.modes[p]) +
                        "\" is repeated with strides whose sum overflows");
    }
    result.type = tensor.type;
    result.alignment = tensor.alignment;
    result.op = tensor.op;
    return result;
  }

  void finish(backend::Operation op, const ConstTensorRef& d,
              const PlanOptions& options) {
    if (d.op != UnaryOp::Identity)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "the output carries a unary operator; only inputs can");
    op.compute = compute_type(options, d.type);
    compute_ = op.compute;
    // The tensor-core modes exist only in cuTENSOR's contraction table, so a
    // Context-wide default must not break the other kinds of operation.
    if (op.kind == backend::OpKind::Contraction)
      op.precision = options.precision;
    op.equation = equation_;
    kernel_ = backend_->plan(op, options);

    std::uint64_t kernel_scratch = kernel_->workspace_size();
    for (const Prereduction& pre : pre_)
      if (pre.kernel != nullptr)
        kernel_scratch = std::max(kernel_scratch, pre.kernel->workspace_size());
    workspace_size_ = scratch_offset_ + kernel_scratch;
  }

  /// Lays out summing `labels` out of `input` into a dense scratch tensor,
  /// and makes `input` that tensor. The operator `input` was read through
  /// goes with it, since it applies before the sum. plan_prereductions picks
  /// the kernel once every buffer's place in the workspace is known.
  void prereduce(std::size_t k, backend::Operand& input,
                 const util::Index& labels) {
    if (labels.rank() == 0) return;
    const std::vector<int32_t> drop = mode_ids(labels);

    backend::Operand reduced;
    reduced.type = input.type;
    reduced.alignment = scratch_alignment;
    for (std::size_t p = 0; p < input.modes.size(); ++p)
      if (std::find(drop.begin(), drop.end(), input.modes[p]) == drop.end()) {
        reduced.modes.push_back(input.modes[p]);
        reduced.extent.push_back(input.extent[p]);
      }
    reduced.stride =
        dense_strides(reduced.extent, Layout::ColumnMajor, equation_);
    constexpr std::int64_t max = std::numeric_limits<std::int64_t>::max();
    std::int64_t elements = 1;
    for (const std::int64_t e : reduced.extent) {
      if (e > 0 && elements > max / e)
        throw Error(Errc::ExtentMismatch, equation_,
                    "a reduced input is too large to address");
      elements *= e;
    }
    const std::int64_t size = element_bytes(reduced.type);
    if (elements > max / size)
      throw Error(Errc::ExtentMismatch, equation_,
                  "a reduced input is too large to address");

    backend::Operation& op = pre_[k].operation;
    op.kind = backend::OpKind::Reduction;
    op.inputs = {input};
    op.output = reduced;
    // The storage type's own default, not options.compute: that names the
    // contraction's accumulation, and a reduction supports fewer pairings
    // (an F64 tensor summed in F32 is a contraction-only combination).
    op.compute = compute_type({}, reduced.type);
    op.equation = equation_;
    // Each buffer fits std::int64_t; keep their sum there too, so offsets
    // and workspace_size() cannot wrap.
    const std::uint64_t bytes =
        round_up(static_cast<std::uint64_t>(elements * size));
    constexpr auto most = static_cast<std::uint64_t>(max);
    if (bytes > most || scratch_offset_ > most - bytes)
      throw Error(Errc::ExtentMismatch, equation_,
                  "the reduced inputs are too large to address");
    pre_[k].offset = scratch_offset_;
    scratch_offset_ += bytes;
    input = reduced;
  }

  /// `options` with workspace_limit lowered by the reduced inputs, which
  /// share the workspace with the kernels' own scratch. Throws if the
  /// reduced inputs alone exceed it.
  PlanOptions scratch_budget(const PlanOptions& options) const {
    PlanOptions budget = options;
    if (options.workspace_limit == 0 || scratch_offset_ == 0) return budget;
    if (scratch_offset_ > options.workspace_limit)
      throw Error(Errc::WorkspaceTooSmall, equation_,
                  "the reduced inputs need " + std::to_string(scratch_offset_) +
                      " bytes of workspace but workspace_limit is " +
                      std::to_string(options.workspace_limit));
    // A limit of zero means none, so an exhausted budget asks for one byte,
    // which only a kernel needing no scratch fits.
    budget.workspace_limit =
        std::max<std::uint64_t>(options.workspace_limit - scratch_offset_, 1);
    return budget;
  }

  void plan_prereductions(const PlanOptions& options) {
    // Algorithm and kernel choices name contraction kernels; leave them to
    // the contraction.
    PlanOptions reduce_options;
    reduce_options.workspace_limit = options.workspace_limit;
    reduce_options.workspace = options.workspace;
    reduce_options.jit = options.jit;
    for (Prereduction& pre : pre_)
      if (!pre.operation.inputs.empty())
        pre.kernel = backend_->plan(pre.operation, reduce_options);
  }

  /// Unset follows the output, except that half-precision output accumulates
  /// in single precision: that is the pairing cuTENSOR supports for both. A
  /// frontend policy, so that every backend gets the same default.
  static Dtype compute_type(const PlanOptions& options, Dtype output) {
    if (options.compute) return *options.compute;
    if (output == Dtype::F16 || output == Dtype::Bf16) return Dtype::F32;
    return output;
  }

  /// A permutation that converts accumulates in the wider of its two types,
  /// which is what cuTENSOR requires: F64 to F32 runs in double precision,
  /// not in the output's single. Unset when the types match, so the usual
  /// default applies.
  static std::optional<Dtype> conversion_compute(Dtype a, Dtype d) {
    if (a == d) return std::nullopt;
    const auto wide = [](Dtype t) {
      return t == Dtype::F64 || t == Dtype::C64;
    };
    const auto complex = [](Dtype t) {
      return t == Dtype::C32 || t == Dtype::C64;
    };
    const bool either_complex = complex(a) || complex(d);
    if (wide(a) || wide(d)) return either_complex ? Dtype::C64 : Dtype::F64;
    return either_complex ? Dtype::C32 : Dtype::F32;
  }

  /// The kernel was chosen for the alignment the plan was built with; a less
  /// aligned pointer would make it read or write misaligned, so refuse it.
  void check_alignment(const char* name, const void* ptr,
                       std::uint32_t planned) const {
    if (alignment_of(ptr) < planned)
      throw Error(Errc::UnsupportedPattern, equation_,
                  std::string("pointer ") + name + " is aligned to " +
                      std::to_string(alignment_of(ptr)) +
                      " bytes but this plan was built for " +
                      std::to_string(planned));
  }

  void check_workspace(Workspace workspace) const {
    if (workspace.size < workspace_size())
      throw Error(Errc::WorkspaceTooSmall, equation_,
                  "the workspace has " + std::to_string(workspace.size) +
                      " bytes but this plan needs " +
                      std::to_string(workspace_size()));
  }

  /// One input summed into scratch before the contraction runs.
  struct Prereduction {
    /// Empty inputs when the operand needs no prereduction.
    backend::Operation operation;
    std::unique_ptr<backend::Kernel> kernel;
    /// Where the reduced input sits in the workspace.
    std::uint64_t offset = 0;
  };

  const backend::Backend* backend_ = nullptr;
  std::unique_ptr<backend::Kernel> kernel_;
  /// For a and b; empty unless the contraction has a label in one input
  /// alone.
  std::array<Prereduction, 2> pre_;
  /// Where the kernels' own scratch starts, after any reduced inputs.
  std::uint64_t scratch_offset_ = 0;
  std::uint64_t workspace_size_ = 0;
  /// Planned alignment of a, b and d, which c shares; b is unused by reduce.
  std::array<std::uint32_t, 3> alignment_{};
  std::string equation_;
  Form form_ = Form::Contraction;
  /// The elementwise operator; unused by the other forms.
  BinaryOp binary_ = BinaryOp::Add;
  /// The accumulation type, whose precision alpha and beta are narrowed to.
  Dtype compute_ = Dtype::F64;
  /// The bytes a and d address; set only for a permutation.
  ByteSpan span_a_;
  ByteSpan span_d_;
};

namespace {

std::string form_name(Plan::Impl::Form form) {
  switch (form) {
    case Plan::Impl::Form::Contraction:
      return "two-operand";
    case Plan::Impl::Form::OneOperand:
      return "one-operand";
    case Plan::Impl::Form::Elementwise:
      return "elementwise";
    case Plan::Impl::Form::Permutation:
      return "one-operand";
  }
  return "matching";
}

}  // namespace

Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;
Plan::~Plan() = default;

const Plan::Impl& Plan::impl() const noexcept {
  assert(impl_ != nullptr && "katal: use of a moved-from Plan");
  return *impl_;
}

uint64_t Plan::workspace_size() const noexcept {
  return impl().workspace_size();
}
std::string_view Plan::equation() const noexcept { return impl().equation(); }

void Plan::execute(Scalar alpha, const void* a, const void* b, Scalar beta,
                   const void* c, void* d, Workspace workspace,
                   StreamRef stream) const {
  const Impl& plan = impl();
  if (plan.form() != Impl::Form::Contraction)
    throw Error(Errc::UnsupportedPattern, plan.equation(),
                "this plan is not a contraction; use the " +
                    form_name(plan.form()) + " execute");
  plan.contract(alpha, a, b, beta, c, d, workspace, stream);
}

void Plan::execute(Scalar alpha, const void* a, Scalar beta, const void* c,
                   void* d, Workspace workspace, StreamRef stream) const {
  const Impl& plan = impl();
  if (plan.form() == Impl::Form::Permutation)
    return plan.permute(alpha, a, beta, d, workspace, stream);
  if (plan.form() != Impl::Form::OneOperand)
    throw Error(Errc::UnsupportedPattern, plan.equation(),
                "this plan takes two operands; use the " +
                    form_name(plan.form()) + " execute");
  plan.reduce(alpha, a, beta, c, d, workspace, stream);
}

void Plan::execute(Scalar alpha, const void* a, Scalar beta, const void* b,
                   Scalar gamma, const void* c, void* d, Workspace workspace,
                   StreamRef stream) const {
  const Impl& plan = impl();
  if (plan.form() != Impl::Form::Elementwise)
    throw Error(Errc::UnsupportedPattern, plan.equation(),
                "this plan is not elementwise; use the " +
                    form_name(plan.form()) + " execute");
  plan.combine(alpha, a, beta, b, gamma, c, d, workspace, stream);
}

Plan make_plan(const Context& context, const Equation& equation,
               const ConstTensorRef& a, const ConstTensorRef& b,
               const ConstTensorRef& d, PlanOptions options) {
#if KATAL_ENABLE_LOGGING
  const auto start = std::chrono::steady_clock::now();
#endif
  Plan plan(std::make_unique<Plan::Impl>(context, equation, a, b, d, options));
#if KATAL_ENABLE_LOGGING
  const auto elapsed = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count();
#endif
  KATAL_LOG("plan.created device="
            << context.device() << " equation="
            << util::log_quote(equation.text()) << " a=" << util::log_tensor(a)
            << " b=" << util::log_tensor(b) << " output=" << util::log_tensor(d)
            << ' ' << util::log_options(options) << " workspace_bytes="
            << plan.workspace_size() << " host_plan_us=" << elapsed);
  return plan;
}

Plan make_plan(const Context& context, const Equation& equation,
               const ConstTensorRef& a, const ConstTensorRef& d,
               PlanOptions options) {
#if KATAL_ENABLE_LOGGING
  const auto start = std::chrono::steady_clock::now();
#endif
  Plan plan(std::make_unique<Plan::Impl>(context, equation, a, d, options));
#if KATAL_ENABLE_LOGGING
  const auto elapsed = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count();
#endif
  KATAL_LOG("plan.created device="
            << context.device()
            << " equation=" << util::log_quote(equation.text())
            << " a=" << util::log_tensor(a) << " output=" << util::log_tensor(d)
            << ' ' << util::log_options(options) << " workspace_bytes="
            << plan.workspace_size() << " host_plan_us=" << elapsed);
  return plan;
}

Plan make_plan(const Context& context, const Equation& equation, BinaryOp op,
               const ConstTensorRef& a, const ConstTensorRef& b,
               const ConstTensorRef& d, PlanOptions options) {
#if KATAL_ENABLE_LOGGING
  const auto start = std::chrono::steady_clock::now();
#endif
  Plan plan(
      std::make_unique<Plan::Impl>(context, equation, op, a, b, d, options));
#if KATAL_ENABLE_LOGGING
  const auto elapsed = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count();
#endif
  KATAL_LOG("plan.created device="
            << context.device()
            << " equation=" << util::log_quote(equation.text())
            << " op=" << util::log_binary(op) << " a=" << util::log_tensor(a)
            << " b=" << util::log_tensor(b) << " output=" << util::log_tensor(d)
            << ' ' << util::log_options(options) << " workspace_bytes="
            << plan.workspace_size() << " host_plan_us=" << elapsed);
  return plan;
}

std::shared_ptr<const Plan> cached_plan(const Context& context,
                                        std::string_view equation,
                                        const ConstTensorRef& a,
                                        const ConstTensorRef& b,
                                        const ConstTensorRef& d,
                                        std::optional<PlanOptions> options) {
  // An unset override takes the context's policy, which is where a program
  // sets planning once instead of at every call site.
  const PlanOptions resolved =
      options.value_or(context.impl().config().plan_options);
  return context.impl().cached(plan_key(equation, resolved, {&a, &b, &d}), [&] {
    return make_plan(context, Equation::parse(equation), a, b, d, resolved);
  });
}

std::shared_ptr<const Plan> cached_plan(const Context& context,
                                        std::string_view equation,
                                        const ConstTensorRef& a,
                                        const ConstTensorRef& d,
                                        std::optional<PlanOptions> options) {
  const PlanOptions resolved =
      options.value_or(context.impl().config().plan_options);
  return context.impl().cached(plan_key(equation, resolved, {&a, &d}), [&] {
    return make_plan(context, Equation::parse(equation), a, d, resolved);
  });
}

std::shared_ptr<const Plan> cached_plan(const Context& context,
                                        std::string_view equation, BinaryOp op,
                                        const ConstTensorRef& a,
                                        const ConstTensorRef& b,
                                        const ConstTensorRef& d,
                                        std::optional<PlanOptions> options) {
  const PlanOptions resolved =
      options.value_or(context.impl().config().plan_options);
  return context.impl().cached(
      plan_key(equation, resolved, {&a, &b, &d}, static_cast<int32_t>(op)),
      [&] {
        return make_plan(context, Equation::parse(equation), op, a, b, d,
                         resolved);
      });
}

Plan make_plan(const Context& context, const Equation& equation,
               BinaryOp reduce, const ConstTensorRef& a,
               const ConstTensorRef& d, PlanOptions options) {
#if KATAL_ENABLE_LOGGING
  const auto start = std::chrono::steady_clock::now();
#endif
  Plan plan(
      std::make_unique<Plan::Impl>(context, equation, reduce, a, d, options));
#if KATAL_ENABLE_LOGGING
  const auto elapsed = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count();
#endif
  KATAL_LOG("plan.created device="
            << context.device()
            << " equation=" << util::log_quote(equation.text())
            << " reduce=" << util::log_binary(reduce)
            << " a=" << util::log_tensor(a) << " output=" << util::log_tensor(d)
            << ' ' << util::log_options(options) << " workspace_bytes="
            << plan.workspace_size() << " host_plan_us=" << elapsed);
  return plan;
}

Plan make_permutation_plan(const Context& context, const Equation& equation,
                           const ConstTensorRef& a, const ConstTensorRef& d,
                           PlanOptions options) {
#if KATAL_ENABLE_LOGGING
  const auto start = std::chrono::steady_clock::now();
#endif
  Plan plan(std::make_unique<Plan::Impl>(
      context, equation, Plan::Impl::PermutationTag{}, a, d, options));
#if KATAL_ENABLE_LOGGING
  const auto elapsed = std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - start)
                           .count();
#endif
  KATAL_LOG("plan.created device="
            << context.device()
            << " equation=" << util::log_quote(equation.text())
            << " permutation a=" << util::log_tensor(a) << " output="
            << util::log_tensor(d) << ' ' << util::log_options(options)
            << " workspace_bytes=" << plan.workspace_size()
            << " host_plan_us=" << elapsed);
  return plan;
}

std::shared_ptr<const Plan> cached_plan(const Context& context,
                                        std::string_view equation,
                                        BinaryOp reduce,
                                        const ConstTensorRef& a,
                                        const ConstTensorRef& d,
                                        std::optional<PlanOptions> options) {
  const PlanOptions resolved =
      options.value_or(context.impl().config().plan_options);
  return context.impl().cached(
      plan_key(equation, resolved, {&a, &d}, static_cast<int32_t>(reduce)),
      [&] {
        return make_plan(context, Equation::parse(equation), reduce, a, d,
                         resolved);
      });
}

std::shared_ptr<const Plan> cached_permutation_plan(
    const Context& context, std::string_view equation, const ConstTensorRef& a,
    const ConstTensorRef& d, std::optional<PlanOptions> options) {
  const PlanOptions resolved =
      options.value_or(context.impl().config().plan_options);
  return context.impl().cached(
      plan_key(equation, resolved, {&a, &d}, permutation_key), [&] {
        return make_permutation_plan(context, Equation::parse(equation), a, d,
                                     resolved);
      });
}

}  // namespace katal
