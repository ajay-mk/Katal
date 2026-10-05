#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
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
  const auto extents = static_cast<uint32_t>(t.extent.size());
  const auto strides = static_cast<uint32_t>(t.stride.size());
  append(key, &extents, sizeof(extents));
  append(key, &strides, sizeof(strides));
  append(key, t.extent.data(), t.extent.size_bytes());
  append(key, t.stride.data(), t.stride.size_bytes());
}

std::string plan_key(std::string_view equation, PlanOptions options,
                     std::initializer_list<const ConstTensorRef*> tensors) {
  std::string key(equation);
  key.push_back('\0');
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

}  // namespace

/// The front end of a plan: everything about an equation that does not depend
/// on which library runs it. Shapes are checked and lowered to a
/// backend::Operation here; the backend's Kernel holds the rest.
class Plan::Impl {
 public:
  Impl(const Context& context, const Equation& equation,
       const ConstTensorRef& a, const ConstTensorRef& b,
       const ConstTensorRef& d, const PlanOptions& options)
      : Impl(context, equation, 2) {
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
      : Impl(context, equation, 1) {
    if (equation.classify().kind == Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "this equation has two operands; use the two-operand form");

    backend::Operation op;
    op.kind = backend::OpKind::Reduction;
    op.inputs = {operand(equation.operand(0), a)};
    op.output = operand(equation.result(), d);
    check_extents(equation,
                  {{&equation.operand(0), &a}, {&equation.result(), &d}});
    finish(std::move(op), d, options);
    alignment_ = {a.alignment, 0, d.alignment};
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  uint64_t workspace_size() const noexcept { return kernel_->workspace_size(); }
  std::string_view equation() const noexcept { return equation_; }

  std::size_t n_operands() const noexcept { return n_operands_; }

  void contract(Scalar alpha, const void* a, const void* b, Scalar beta,
                const void* c, void* d, Workspace workspace,
                StreamRef stream) const {
    backend_->check_device(equation_, "plan");
    check_alignment("a", a, alignment_[0]);
    check_alignment("b", b, alignment_[1]);
    check_alignment("c", c, alignment_[2]);
    check_alignment("d", d, alignment_[2]);
    check_workspace(workspace);
    const std::array<const void*, 2> inputs{a, b};
    kernel_->run(alpha, inputs.data(), beta, c, d, workspace, stream);
  }

  void reduce(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
              Workspace workspace, StreamRef stream) const {
    backend_->check_device(equation_, "plan");
    check_alignment("a", a, alignment_[0]);
    check_alignment("c", c, alignment_[2]);
    check_alignment("d", d, alignment_[2]);
    check_workspace(workspace);
    kernel_->run(alpha, &a, beta, c, d, workspace, stream);
  }

 private:
  Impl(const Context& context, const Equation& equation, std::size_t n_operands)
      : backend_(&context.impl().backend()),
        equation_(equation.text()),
        n_operands_(n_operands) {
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
    return result;
  }

  void finish(backend::Operation op, const ConstTensorRef& d,
              const PlanOptions& options) {
    op.compute = compute_type(options, d);
    op.equation = equation_;
    kernel_ = backend_->plan(op, options);
  }

  /// Unset follows the output, except that half-precision output accumulates
  /// in single precision: that is the pairing cuTENSOR supports for both.
  static Dtype compute_type(const PlanOptions& options,
                            const ConstTensorRef& d) {
    if (options.compute) return *options.compute;
    if (d.type == Dtype::F16 || d.type == Dtype::Bf16) return Dtype::F32;
    return d.type;
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
  std::size_t n_operands_ = 2;
};

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
  if (plan.n_operands() != 2)
    throw Error(Errc::UnsupportedPattern, plan.equation(),
                "this plan takes one operand; use the one-operand execute");
  plan.contract(alpha, a, b, beta, c, d, workspace, stream);
}

void Plan::execute(Scalar alpha, const void* a, Scalar beta, const void* c,
                   void* d, Workspace workspace, StreamRef stream) const {
  const Impl& plan = impl();
  if (plan.n_operands() != 1)
    throw Error(Errc::UnsupportedPattern, plan.equation(),
                "this plan takes two operands; use the two-operand execute");
  plan.reduce(alpha, a, beta, c, d, workspace, stream);
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

}  // namespace katal
