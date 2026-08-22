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
                    "label \"" + std::string(1, (*modes)[p]) +
                        "\" has extent " + std::to_string(extent[slot]) +
                        " in one operand and " + std::to_string(e) +
                        " in another");
      extent[slot] = e;
    }
}

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
    select_kernel(options);
  }

  /// The one-operand form. cuTENSOR's permutation entry point takes neither
  /// beta nor a workspace, so a reduction covers all three unary lowerings
  /// with one code path: reducing over no modes is a permutation.
  ///
  /// A copy therefore runs a reduction kernel rather than cudaMemcpyAsync.
  /// Add the memcpy path if the copy case ever matters.
  Impl(const Context& context, const Equation& equation, const TensorRef& a,
       const TensorRef& d, PlanOptions options)
      : handle_(context.impl().handle()),
        equation_(equation.text()),
        scalar_type_(d.type),
        n_operands_(1) {
    if (equation.classify().kind == Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "this equation has two operands; use the two-operand form");

    const util::Index all = all_labels(equation);
    check_extents(equation, all,
                  {{&equation.operand(0), &a}, {&equation.result(), &d}});

    const ModeOrder ma = mode_order(equation.operand(0), a);
    const ModeOrder md = mode_order(equation.result(), d);
    const std::vector<int32_t> ia = mode_ids(ma.modes, all);
    const std::vector<int32_t> id = mode_ids(md.modes, all);

    describe(&desc_a_, ma, a);
    describe(&desc_d_, md, d);

    CUEIN_CHECK(cutensorCreateReduction(
                    handle_, &op_, desc_a_, ia.data(), CUTENSOR_OP_IDENTITY,
                    desc_d_, id.data(), CUTENSOR_OP_IDENTITY, desc_d_,
                    id.data(), CUTENSOR_OP_ADD,
                    util::to_compute(options.compute.value_or(d.type))),
                equation_);
    select_kernel(options);
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

  std::size_t n_operands() const noexcept { return n_operands_; }

  void contract(Scalar alpha, const void* a, const void* b, Scalar beta,
                const void* c, void* d, Workspace workspace,
                StreamRef stream) const {
    // Stack objects, so this path still allocates nothing.
    const util::ScalarBytes alpha_bytes(alpha, scalar_type_);
    const util::ScalarBytes beta_bytes(beta, scalar_type_);

    CUEIN_CHECK(cutensorContract(handle_, plan_, alpha_bytes.data(), a, b,
                                 beta_bytes.data(), c, d, workspace.ptr,
                                 workspace.size, stream.get()),
                equation_);
  }

  void reduce(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
              Workspace workspace, StreamRef stream) const {
    const util::ScalarBytes alpha_bytes(alpha, scalar_type_);
    const util::ScalarBytes beta_bytes(beta, scalar_type_);

    CUEIN_CHECK(
        cutensorReduce(handle_, plan_, alpha_bytes.data(), a, beta_bytes.data(),
                       c, d, workspace.ptr, workspace.size, stream.get()),
        equation_);
  }

 private:
  /// Plan preference, workspace estimate and kernel selection, shared by both
  /// arities.
  void select_kernel(const PlanOptions& options) {
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
  std::size_t n_operands_ = 2;
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
  if (impl_->n_operands() != 2)
    throw Error(Errc::UnsupportedPattern, impl_->equation(),
                "this plan takes one operand; use the one-operand execute");
  impl_->contract(alpha, a, b, beta, c, d, workspace, stream);
}

void Plan::execute(Scalar alpha, const void* a, Scalar beta, const void* c,
                   void* d, Workspace workspace, StreamRef stream) const {
  if (impl_->n_operands() != 1)
    throw Error(Errc::UnsupportedPattern, impl_->equation(),
                "this plan takes two operands; use the two-operand execute");
  impl_->reduce(alpha, a, beta, c, d, workspace, stream);
}

Plan make_plan(const Context& context, const Equation& equation,
               const TensorRef& a, const TensorRef& b, const TensorRef& d,
               PlanOptions options) {
  return Plan(
      std::make_unique<Plan::Impl>(context, equation, a, b, d, options));
}

Plan make_plan(const Context& context, const Equation& equation,
               const TensorRef& a, const TensorRef& d, PlanOptions options) {
  return Plan(std::make_unique<Plan::Impl>(context, equation, a, d, options));
}

Workspace CudaAsyncAllocator::allocate(uint64_t bytes, StreamRef stream) const {
  Workspace workspace;
  CUEIN_CHECK_CUDA(cudaMallocAsync(&workspace.ptr, bytes, stream.get()), "");
  workspace.size = bytes;
  return workspace;
}

void CudaAsyncAllocator::deallocate(Workspace workspace,
                                    StreamRef stream) const {
  // Destructors are noexcept; a failure here has nowhere useful to go.
  cudaFreeAsync(workspace.ptr, stream.get());
}

}  // namespace cuein
