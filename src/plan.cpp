#include <array>
#include <cassert>
#include <cstddef>
#include <new>
#include <utility>
#include <vector>

#include "context_impl.hpp"
#include "cuda_status.hpp"
#include "cutensor_status.hpp"
#include "cutensor_types.hpp"
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

class Plan::Impl {
 public:
  Impl(const Context& context, const Equation& equation,
       const ConstTensorRef& a, const ConstTensorRef& b,
       const ConstTensorRef& d, PlanOptions options)
      : Impl(context, equation, 2) {
    const Classification classification = equation.classify();
    if (classification.kind != Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "only binary contraction is implemented so far");

    const ModeOrder ma = mode_order(equation.operand(0), a, equation_);
    const ModeOrder mb = mode_order(equation.operand(1), b, equation_);
    const ModeOrder md = mode_order(equation.result(), d, equation_);
    check_extents(equation, {{&equation.operand(0), &a},
                             {&equation.operand(1), &b},
                             {&equation.result(), &d}});
    const std::vector<int32_t> ia = mode_ids(ma.modes);
    const std::vector<int32_t> ib = mode_ids(mb.modes);
    const std::vector<int32_t> id = mode_ids(md.modes);

    describe(&desc_a_, ma, a);
    describe(&desc_b_, mb, b);
    describe(&desc_d_, md, d);

    // C and D share a descriptor: beta scales the output in place.
    KATAL_CHECK(cutensorCreateContraction(
                    handle_, &op_, desc_a_, ia.data(), CUTENSOR_OP_IDENTITY,
                    desc_b_, ib.data(), CUTENSOR_OP_IDENTITY, desc_d_,
                    id.data(), CUTENSOR_OP_IDENTITY, desc_d_, id.data(),
                    util::to_compute(compute_type(options, d))),
                equation_);
    read_scalar_type();
    select_kernel(options);
  }

  /// The one-operand form. cuTENSOR's permutation entry point takes neither
  /// beta nor a workspace, so a reduction covers all three unary lowerings
  /// with one code path: reducing over no modes is a permutation.
  ///
  /// A copy therefore runs a reduction kernel rather than cudaMemcpyAsync.
  /// Add the memcpy path if the copy case ever matters.
  Impl(const Context& context, const Equation& equation,
       const ConstTensorRef& a, const ConstTensorRef& d, PlanOptions options)
      : Impl(context, equation, 1) {
    if (equation.classify().kind == Lowering::Contract)
      throw Error(Errc::UnsupportedPattern, equation_,
                  "this equation has two operands; use the two-operand form");

    const ModeOrder ma = mode_order(equation.operand(0), a, equation_);
    const ModeOrder md = mode_order(equation.result(), d, equation_);
    check_extents(equation,
                  {{&equation.operand(0), &a}, {&equation.result(), &d}});
    const std::vector<int32_t> ia = mode_ids(ma.modes);
    const std::vector<int32_t> id = mode_ids(md.modes);

    describe(&desc_a_, ma, a);
    describe(&desc_d_, md, d);

    KATAL_CHECK(
        cutensorCreateReduction(
            handle_, &op_, desc_a_, ia.data(), CUTENSOR_OP_IDENTITY, desc_d_,
            id.data(), CUTENSOR_OP_IDENTITY, desc_d_, id.data(),
            CUTENSOR_OP_ADD, util::to_compute(compute_type(options, d))),
        equation_);
    read_scalar_type();
    select_kernel(options);
  }

  ~Impl() {
    util::ScopedDevice on_device(device_);
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
    check_device();
    check_workspace(workspace);
    // Stack objects, so this path still allocates nothing.
    const util::ScalarBytes alpha_bytes(alpha, scalar_type_);
    const util::ScalarBytes beta_bytes(beta, scalar_type_);

    KATAL_CHECK(cutensorContract(handle_, plan_, alpha_bytes.data(), a, b,
                                 beta_bytes.data(), c, d, workspace.ptr,
                                 workspace.size, stream.get()),
                equation_);
  }

  void reduce(Scalar alpha, const void* a, Scalar beta, const void* c, void* d,
              Workspace workspace, StreamRef stream) const {
    check_device();
    check_workspace(workspace);
    const util::ScalarBytes alpha_bytes(alpha, scalar_type_);
    const util::ScalarBytes beta_bytes(beta, scalar_type_);

    KATAL_CHECK(
        cutensorReduce(handle_, plan_, alpha_bytes.data(), a, beta_bytes.data(),
                       c, d, workspace.ptr, workspace.size, stream.get()),
        equation_);
  }

 private:
  /// The handle and every descriptor belong to the device the context was
  /// built on; handing them to another device's stream is undefined.
  /// cudaGetDevice is a host-side read, so this costs the hot path little.
  void check_device() const {
    int current = 0;
    KATAL_CHECK_CUDA(cudaGetDevice(&current), equation_);
    if (current != device_)
      throw Error(Errc::CudaError, equation_,
                  "this plan belongs to device " + std::to_string(device_) +
                      " but device " + std::to_string(current) + " is current");
  }

  void check_workspace(Workspace workspace) const {
    if (workspace.size < workspace_size_)
      throw Error(Errc::WorkspaceTooSmall, equation_,
                  "the workspace has " + std::to_string(workspace.size) +
                      " bytes but this plan needs " +
                      std::to_string(workspace_size_));
  }

  // Completing the delegated constructor makes ~Impl run if backend setup
  // in either public constructor throws after acquiring a resource.
  Impl(const Context& context, const Equation& equation, std::size_t n_operands)
      : handle_(context.impl().handle()),
        device_(context.device()),
        equation_(equation.text()),
        n_operands_(n_operands) {
    check_device();
  }

  /// Unset follows the output, except that half-precision output accumulates
  /// in single precision: that is the pairing cuTENSOR supports for both.
  static Dtype compute_type(const PlanOptions& options,
                            const ConstTensorRef& d) {
    if (options.compute) return *options.compute;
    if (d.type == Dtype::F16 || d.type == Dtype::Bf16) return Dtype::F32;
    return d.type;
  }

  /// alpha and beta are passed in the operation's scalar type, which follows
  /// the compute type rather than the storage, so ask cuTENSOR for it. Checked
  /// here so that an inexpressible one fails planning, not every execute.
  void read_scalar_type() {
    cutensorDataType_t type{};
    KATAL_CHECK(cutensorOperationDescriptorGetAttribute(
                    handle_, op_, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                    &type, sizeof(type)),
                equation_);
    scalar_type_ = util::from_cutensor(type);
    const util::ScalarBytes probe(1.0, scalar_type_);
  }

  /// Plan preference, workspace estimate and kernel selection, shared by both
  /// arities.
  void select_kernel(const PlanOptions& options) {
    KATAL_CHECK(
        cutensorCreatePlanPreference(
            handle_, &pref_, util::to_cutensor(options.algo),
            options.jit ? CUTENSOR_JIT_MODE_DEFAULT : CUTENSOR_JIT_MODE_NONE),
        equation_);

    if (options.kernel_rank) {
      const int32_t rank = *options.kernel_rank;
      KATAL_CHECK(cutensorPlanPreferenceSetAttribute(
                      handle_, pref_, CUTENSOR_PLAN_PREFERENCE_KERNEL_RANK,
                      &rank, sizeof(rank)),
                  equation_);
    }

    uint64_t estimate = 0;
    KATAL_CHECK(cutensorEstimateWorkspaceSize(
                    handle_, op_, pref_, util::to_cutensor(options.workspace),
                    &estimate),
                equation_);
    const uint64_t limit =
        options.workspace_limit != 0 ? options.workspace_limit : estimate;
    KATAL_CHECK(cutensorCreatePlan(handle_, &plan_, op_, pref_, limit),
                equation_);

    // The estimate above is what planning was allowed to assume. Ask the plan
    // what it actually needs, so a caller does not over-allocate.
    KATAL_CHECK(cutensorPlanGetAttribute(
                    handle_, plan_, CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                    &workspace_size_, sizeof(workspace_size_)),
                equation_);
  }

  void describe(cutensorTensorDescriptor_t* desc, const ModeOrder& modes,
                const ConstTensorRef& tensor) {
    KATAL_CHECK(cutensorCreateTensorDescriptor(
                    handle_, desc, static_cast<uint32_t>(modes.modes.rank()),
                    modes.extent.data(), modes.stride.data(),
                    util::to_cutensor(tensor.type), tensor.alignment),
                equation_);
  }

  cutensorHandle_t handle_ = nullptr;
  int device_ = 0;
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

Workspace CudaAsyncAllocator::allocate(uint64_t bytes, StreamRef stream) const {
  Workspace workspace;
  const cudaError_t status =
      cudaMallocAsync(&workspace.ptr, bytes, stream.get());
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();  // reported here, so not left for the caller
    throw std::bad_alloc();
  }
  KATAL_CHECK_CUDA(status, "");
  workspace.size = bytes;
  return workspace;
}

void CudaAsyncAllocator::deallocate(Workspace workspace,
                                    StreamRef stream) const noexcept {
  util::check_release(cudaFreeAsync(workspace.ptr, stream.get()));
}

}  // namespace katal
