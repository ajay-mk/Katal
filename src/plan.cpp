#include <cstddef>
#include <utility>
#include <vector>

#include "context_impl.hpp"
#include "cutensor_status.hpp"
#include "cutensor_types.hpp"

#include <deviceein/plan.hpp>

namespace deviceein {
namespace {

/// The cache key, built as one string so that std::hash and string equality
/// do the work: an exact key means no collision handling to get wrong.
void append(std::string& key, const void* bytes, std::size_t n) {
  key.append(static_cast<const char*>(bytes), n);
}

void append_tensor(std::string& key, const TensorRef& t) {
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
                     std::initializer_list<const TensorRef*> tensors) {
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
  for (const TensorRef* t : tensors) append_tensor(key, *t);
  return key;
}

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
    DEVICEEIN_CHECK(cutensorCreateContraction(
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

    DEVICEEIN_CHECK(cutensorCreateReduction(
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

    DEVICEEIN_CHECK(cutensorContract(handle_, plan_, alpha_bytes.data(), a, b,
                                     beta_bytes.data(), c, d, workspace.ptr,
                                     workspace.size, stream.get()),
                    equation_);
  }

  void reduce(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
              Workspace workspace, StreamRef stream) const {
    const util::ScalarBytes alpha_bytes(alpha, scalar_type_);
    const util::ScalarBytes beta_bytes(beta, scalar_type_);

    DEVICEEIN_CHECK(
        cutensorReduce(handle_, plan_, alpha_bytes.data(), a, beta_bytes.data(),
                       c, d, workspace.ptr, workspace.size, stream.get()),
        equation_);
  }

 private:
  /// Plan preference, workspace estimate and kernel selection, shared by both
  /// arities.
  void select_kernel(const PlanOptions& options) {
    DEVICEEIN_CHECK(
        cutensorCreatePlanPreference(
            handle_, &pref_, util::to_cutensor(options.algo),
            options.jit ? CUTENSOR_JIT_MODE_DEFAULT : CUTENSOR_JIT_MODE_NONE),
        equation_);

    if (options.kernel_rank) {
      const int32_t rank = *options.kernel_rank;
      DEVICEEIN_CHECK(cutensorPlanPreferenceSetAttribute(
                          handle_, pref_, CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK,
                          &rank, sizeof(rank)),
                      equation_);
    }

    uint64_t estimate = 0;
    DEVICEEIN_CHECK(cutensorEstimateWorkspaceSize(
                        handle_, op_, pref_,
                        util::to_cutensor(options.workspace), &estimate),
                    equation_);
    const uint64_t limit =
        options.workspace_limit != 0 ? options.workspace_limit : estimate;
    DEVICEEIN_CHECK(cutensorCreatePlan(handle_, &plan_, op_, pref_, limit),
                    equation_);

    // The estimate above is what planning was allowed to assume. Ask the plan
    // what it actually needs, so a caller does not over-allocate.
    DEVICEEIN_CHECK(cutensorPlanGetAttribute(
                        handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                        &workspace_size_, sizeof(workspace_size_)),
                    equation_);
  }

  void describe(cutensorTensorDescriptor_t* desc, const ModeOrder& modes,
                const TensorRef& tensor) {
    DEVICEEIN_CHECK(
        cutensorCreateTensorDescriptor(
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

std::shared_ptr<const Plan> cached_plan(const Context& context,
                                        std::string_view equation,
                                        const TensorRef& a, const TensorRef& b,
                                        const TensorRef& d,
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
                                        const TensorRef& a, const TensorRef& d,
                                        std::optional<PlanOptions> options) {
  const PlanOptions resolved =
      options.value_or(context.impl().config().plan_options);
  return context.impl().cached(plan_key(equation, resolved, {&a, &d}), [&] {
    return make_plan(context, Equation::parse(equation), a, d, resolved);
  });
}

Workspace CudaAsyncAllocator::allocate(uint64_t bytes, StreamRef stream) const {
  Workspace workspace;
  DEVICEEIN_CHECK_CUDA(cudaMallocAsync(&workspace.ptr, bytes, stream.get()),
                       "");
  workspace.size = bytes;
  return workspace;
}

void CudaAsyncAllocator::deallocate(Workspace workspace,
                                    StreamRef stream) const {
  // Destructors are noexcept; a failure here has nowhere useful to go.
  cudaFreeAsync(workspace.ptr, stream.get());
}

}  // namespace deviceein
