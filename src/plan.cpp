#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
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
  const int32_t compute =
      options.compute ? static_cast<int32_t>(*options.compute) : -1;
  append(key, &compute, sizeof(compute));
  append(key, &options.workspace_limit, sizeof(options.workspace_limit));
  const auto algo = static_cast<int32_t>(options.algo);
  const auto workspace = static_cast<int32_t>(options.workspace);
  const int32_t rank = options.kernel_rank.value_or(-1);
  const auto jit = static_cast<int32_t>(options.jit);
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

}  // namespace

/// The front end of a plan: everything about an equation that does not depend
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
    finish(std::move(op), d, options);
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

  uint64_t workspace_size() const noexcept { return kernel_->workspace_size(); }
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
    const std::array<Scalar, 2> scalars{alpha, beta};
    const std::array<const void*, 2> inputs{a, b};
    kernel_->run(scalars, inputs, c, d, workspace, stream);
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

  /// Puts a tensor into descriptor order, with labels as mode ids.
  backend::Operand operand(const util::Index& labels,
                           const ConstTensorRef& tensor) const {
    const ModeOrder order = mode_order(labels, tensor, equation_);
    backend::Operand result;
    result.modes = mode_ids(order.modes);
    result.extent = order.extent;
    result.stride = order.stride;
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
    op.compute = compute_type(options, d);
    compute_ = op.compute;
    op.equation = equation_;
    kernel_ = backend_->plan(op, options);
  }

  /// Unset follows the output, except that half-precision output accumulates
  /// in single precision: that is the pairing cuTENSOR supports for both. A
  /// front-end policy, so that every backend gets the same default.
  static Dtype compute_type(const PlanOptions& options,
                            const ConstTensorRef& d) {
    if (options.compute) return *options.compute;
    if (d.type == Dtype::F16 || d.type == Dtype::Bf16) return Dtype::F32;
    return d.type;
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

  const backend::Backend* backend_ = nullptr;
  std::unique_ptr<backend::Kernel> kernel_;
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
