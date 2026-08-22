#include <cstddef>
#include <utility>
#include <vector>

#include "context_impl.hpp"
#include "cutensor_status.hpp"
#include "cutensor_types.hpp"

#include <cuein/plan.hpp>

namespace cuein {
namespace {

/// Every distinct label in the equation. A label's position here is the
/// int32 mode id cuTENSOR sees, which keeps the ids consistent across operands.
util::Index all_labels(const Equation& equation) {
  util::Index all = equation.result();
  for (std::size_t k = 0; k < equation.n_operands(); ++k)
    all = all | equation.operand(k);
  return all;
}

std::vector<int32_t> mode_ids(const util::Index& modes,
                              const util::Index& all) {
  std::vector<int32_t> ids;
  ids.reserve(modes.rank());
  for (std::size_t i = 0; i < modes.rank(); ++i)
    for (std::size_t j = 0; j < all.rank(); ++j)
      if (all[j] == modes[i]) {
        ids.push_back(static_cast<int32_t>(j));
        break;
      }
  return ids;
}

/// A shared label must have one extent. cuTENSOR would reject a mismatch too,
/// but without naming the label responsible.
void check_extents(
    const Equation& equation, const util::Index& all,
    std::initializer_list<std::pair<const util::Index*, const TensorRef*>>
        operands) {
  std::vector<std::int64_t> extent(all.rank(), 0);
  for (const auto& [modes, tensor] : operands)
    for (std::size_t p = 0; p < modes->rank(); ++p) {
      std::size_t slot = all.rank();
      for (std::size_t j = 0; j < all.rank(); ++j)
        if (all[j] == (*modes)[p]) {
          slot = j;
          break;
        }
      const std::int64_t e = tensor->extent[p];
      if (extent[slot] != 0 && extent[slot] != e)
        throw Error(Errc::ExtentMismatch, equation.text(),
                    "label \"" + std::string((*modes)[p].view()) +
                        "\" has extent " + std::to_string(extent[slot]) +
                        " in one operand and " + std::to_string(e) +
                        " in another");
      extent[slot] = e;
    }
}

/// Frees the scratch the one-shot path allocates, including when execute
/// throws.
class ScopedWorkspace {
 public:
  ScopedWorkspace(uint64_t size, CUstream_st* stream, std::string_view equation)
      : stream_(stream) {
    if (size == 0) return;
    CUEIN_CHECK_CUDA(cudaMallocAsync(&workspace_.ptr, size, stream_), equation);
    workspace_.size = size;
  }

  ~ScopedWorkspace() {
    if (workspace_.ptr != nullptr) cudaFreeAsync(workspace_.ptr, stream_);
  }

  ScopedWorkspace(const ScopedWorkspace&) = delete;
  ScopedWorkspace& operator=(const ScopedWorkspace&) = delete;

  Workspace get() const noexcept { return workspace_; }

 private:
  Workspace workspace_;
  CUstream_st* stream_ = nullptr;
};

}  // namespace

class Plan::Impl {
 public:
  Impl(const Context& context, const Equation& equation, const TensorRef& a,
       const TensorRef& b, const TensorRef& d, PlanOptions options)
      : handle_(context.impl().handle()),
        equation_(equation.text()),
        scalar_type_(d.type) {
    const Classification classification = equation.classify();
    if (classification.kind != Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "only binary contraction is implemented so far");

    const util::Index all = all_labels(equation);
    check_extents(equation, all,
                  {{&equation.operand(0), &a},
                   {&equation.operand(1), &b},
                   {&equation.result(), &d}});

    const ModeOrder ma = mode_order(equation.operand(0), a);
    const ModeOrder mb = mode_order(equation.operand(1), b);
    const ModeOrder md = mode_order(equation.result(), d);
    const std::vector<int32_t> ia = mode_ids(ma.modes, all);
    const std::vector<int32_t> ib = mode_ids(mb.modes, all);
    const std::vector<int32_t> id = mode_ids(md.modes, all);

    describe(&desc_a_, ma, a);
    describe(&desc_b_, mb, b);
    describe(&desc_d_, md, d);

    // C and D share a descriptor: beta scales the output in place.
    CUEIN_CHECK(cutensorCreateContraction(
                    handle_, &op_, desc_a_, ia.data(), CUTENSOR_OP_IDENTITY,
                    desc_b_, ib.data(), CUTENSOR_OP_IDENTITY, desc_d_,
                    id.data(), CUTENSOR_OP_IDENTITY, desc_d_, id.data(),
                    util::to_compute(options.compute.value_or(d.type))),
                equation_);
    CUEIN_CHECK(
        cutensorCreatePlanPreference(handle_, &pref_, CUTENSOR_ALGO_DEFAULT,
                                     CUTENSOR_JIT_MODE_NONE),
        equation_);
    CUEIN_CHECK(
        cutensorEstimateWorkspaceSize(
            handle_, op_, pref_, CUTENSOR_WORKSPACE_DEFAULT, &workspace_size_),
        equation_);
    if (options.workspace_limit != 0) workspace_size_ = options.workspace_limit;
    CUEIN_CHECK(
        cutensorCreatePlan(handle_, &plan_, op_, pref_, workspace_size_),
        equation_);
  }

  ~Impl() {
    if (plan_ != nullptr) cutensorDestroyPlan(plan_);
    if (pref_ != nullptr) cutensorDestroyPlanPreference(pref_);
    if (op_ != nullptr) cutensorDestroyOperationDescriptor(op_);
    if (desc_d_ != nullptr) cutensorDestroyTensorDescriptor(desc_d_);
    if (desc_b_ != nullptr) cutensorDestroyTensorDescriptor(desc_b_);
    if (desc_a_ != nullptr) cutensorDestroyTensorDescriptor(desc_a_);
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  uint64_t workspace_size() const noexcept { return workspace_size_; }
  std::string_view equation() const noexcept { return equation_; }

  void execute(Scalar alpha, const void* a, const void* b, Scalar beta,
               const void* c, void* d, Workspace workspace,
               StreamRef stream) const {
    // ponytail: fixed buffers, so this path allocates nothing. 16 bytes holds
    // the widest scalar cuTENSOR takes.
    alignas(16) unsigned char alpha_bytes[16];
    alignas(16) unsigned char beta_bytes[16];
    alpha.write_as(scalar_type_, alpha_bytes);
    beta.write_as(scalar_type_, beta_bytes);

    CUEIN_CHECK(
        cutensorContract(handle_, plan_, alpha_bytes, a, b, beta_bytes, c, d,
                         workspace.ptr, workspace.size, stream.get()),
        equation_);
  }

 private:
  void describe(cutensorTensorDescriptor_t* desc, const ModeOrder& modes,
                const TensorRef& tensor) {
    CUEIN_CHECK(cutensorCreateTensorDescriptor(
                    handle_, desc, static_cast<uint32_t>(modes.modes.rank()),
                    modes.extent.data(), modes.stride.data(),
                    util::to_cutensor(tensor.type), tensor.alignment),
                equation_);
  }

  cutensorHandle_t handle_ = nullptr;
  cutensorTensorDescriptor_t desc_a_ = nullptr;
  cutensorTensorDescriptor_t desc_b_ = nullptr;
  cutensorTensorDescriptor_t desc_d_ = nullptr;
  cutensorOperationDescriptor_t op_ = nullptr;
  cutensorPlanPreference_t pref_ = nullptr;
  cutensorPlan_t plan_ = nullptr;
  uint64_t workspace_size_ = 0;
  std::string equation_;
  Dtype scalar_type_ = Dtype::F64;
};

Plan::Plan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;
Plan::~Plan() = default;

uint64_t Plan::workspace_size() const noexcept {
  return impl_->workspace_size();
}
std::string_view Plan::equation() const noexcept { return impl_->equation(); }

void Plan::execute(Scalar alpha, const void* a, const void* b, Scalar beta,
                   const void* c, void* d, Workspace workspace,
                   StreamRef stream) const {
  impl_->execute(alpha, a, b, beta, c, d, workspace, stream);
}

Plan make_plan(const Context& context, const Equation& equation,
               const TensorRef& a, const TensorRef& b, const TensorRef& d,
               PlanOptions options) {
  return Plan(
      std::make_unique<Plan::Impl>(context, equation, a, b, d, options));
}

void einsum(std::string_view equation, Scalar alpha, const TensorRef& a,
            const TensorRef& b, Scalar beta, const TensorRef& d,
            StreamRef stream, const Context& context) {
  const Equation parsed = Equation::parse(equation);
  const Plan plan = make_plan(context, parsed, a, b, d);
  const ScopedWorkspace workspace(plan.workspace_size(), stream.get(),
                                  equation);
  plan.execute(alpha, a.data, b.data, beta, d.data, d.data, workspace.get(),
               stream);
}

}  // namespace cuein
