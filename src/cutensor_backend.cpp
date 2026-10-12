#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "backend.hpp"
#include "cuda_status.hpp"
#include "cutensor_status.hpp"
#include "cutensor_types.hpp"

namespace katal::backend {
namespace {

cutensorOperator_t to_cutensor(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return CUTENSOR_OP_ADD;
    case BinaryOp::Multiply:
      return CUTENSOR_OP_MUL;
    case BinaryOp::Max:
      return CUTENSOR_OP_MAX;
    case BinaryOp::Min:
      return CUTENSOR_OP_MIN;
  }
  return CUTENSOR_OP_ADD;
}

class CutensorKernel final : public Kernel {
 public:
  CutensorKernel(cutensorHandle_t handle, int device,
                 const Operation& operation, const PlanOptions& options)
      : CutensorKernel(handle, device, operation.equation) {
    const Operand& a = operation.inputs.at(0);
    const Operand& d = operation.output;
    describe(&desc_a_, a);
    describe(&desc_d_, d);
    const auto compute = util::to_compute(operation.compute);

    switch (operation.kind) {
      case OpKind::Contraction: {
        if (operation.inputs.size() != 2)
          throw Error(Errc::UnsupportedPattern, operation.equation,
                      "a contraction takes two inputs");
        const Operand& b = operation.inputs[1];
        describe(&desc_b_, b);
        // C and D share a descriptor: beta scales the output in place.
        KATAL_CHECK(cutensorCreateContraction(
                        handle_, &op_, desc_a_, a.modes.data(),
                        CUTENSOR_OP_IDENTITY, desc_b_, b.modes.data(),
                        CUTENSOR_OP_IDENTITY, desc_d_, d.modes.data(),
                        CUTENSOR_OP_IDENTITY, desc_d_, d.modes.data(), compute),
                    equation_);
        break;
      }
      case OpKind::Reduction:
        // cuTENSOR's permutation entry point takes neither beta nor a
        // workspace, so a reduction covers permute, reduce and copy with one
        // code path: reducing over no modes is a permutation. A copy
        // therefore runs a reduction kernel rather than cudaMemcpyAsync.
        KATAL_CHECK(
            cutensorCreateReduction(
                handle_, &op_, desc_a_, a.modes.data(), CUTENSOR_OP_IDENTITY,
                desc_d_, d.modes.data(), CUTENSOR_OP_IDENTITY, desc_d_,
                d.modes.data(), CUTENSOR_OP_ADD, compute),
            equation_);
        break;
      case OpKind::Elementwise: {
        if (operation.inputs.size() != 2)
          throw Error(Errc::UnsupportedPattern, operation.equation,
                      "an elementwise operation takes two inputs");
        const Operand& b = operation.inputs[1];
        describe(&desc_b_, b);
        // cuTENSOR's binary form requires C to share D's descriptor, which
        // would leave no input free to be permuted or broadcast on that side.
        // The trinary form frees both inputs, with D as C for the gamma term.
        KATAL_CHECK(
            cutensorCreateElementwiseTrinary(
                handle_, &op_, desc_a_, a.modes.data(), CUTENSOR_OP_IDENTITY,
                desc_b_, b.modes.data(), CUTENSOR_OP_IDENTITY, desc_d_,
                d.modes.data(), CUTENSOR_OP_IDENTITY, desc_d_, d.modes.data(),
                to_cutensor(operation.binary), CUTENSOR_OP_ADD, compute),
            equation_);
        break;
      }
    }
    kind_ = operation.kind;
    read_scalar_type();
    select_kernel(options);
  }

  ~CutensorKernel() override {
    // cuTENSOR objects must be destroyed on the device they were made on,
    // and the caller may have another device current by now.
    util::ScopedDevice on_device(device_);
    if (plan_ != nullptr) cutensorDestroyPlan(plan_);
    if (pref_ != nullptr) cutensorDestroyPlanPreference(pref_);
    if (op_ != nullptr) cutensorDestroyOperationDescriptor(op_);
    if (desc_d_ != nullptr) cutensorDestroyTensorDescriptor(desc_d_);
    if (desc_b_ != nullptr) cutensorDestroyTensorDescriptor(desc_b_);
    if (desc_a_ != nullptr) cutensorDestroyTensorDescriptor(desc_a_);
  }

  std::uint64_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void run(std::span<const Scalar> scalars, std::span<const void* const> inputs,
           const void* c, void* d, Workspace workspace,
           StreamRef stream) const override {
    // Stack objects, so this path still allocates nothing.
    const util::ScalarBytes alpha_bytes(scalars[0], scalar_type_);
    const util::ScalarBytes beta_bytes(scalars[1], scalar_type_);
    switch (kind_) {
      case OpKind::Contraction:
        KATAL_CHECK(
            cutensorContract(handle_, plan_, alpha_bytes.data(), inputs[0],
                             inputs[1], beta_bytes.data(), c, d, workspace.ptr,
                             workspace.size, stream.get()),
            equation_);
        return;
      case OpKind::Reduction:
        KATAL_CHECK(cutensorReduce(handle_, plan_, alpha_bytes.data(),
                                   inputs[0], beta_bytes.data(), c, d,
                                   workspace.ptr, workspace.size, stream.get()),
                    equation_);
        return;
      case OpKind::Elementwise: {
        const util::ScalarBytes gamma_bytes(scalars[2], scalar_type_);
        KATAL_CHECK(cutensorElementwiseTrinaryExecute(
                        handle_, plan_, alpha_bytes.data(), inputs[0],
                        beta_bytes.data(), inputs[1], gamma_bytes.data(), c, d,
                        stream.get()),
                    equation_);
        return;
      }
    }
  }

 private:
  // Completing this delegated constructor first makes the destructor run if
  // setup in the public one throws after acquiring a descriptor.
  CutensorKernel(cutensorHandle_t handle, int device, std::string equation)
      : handle_(handle), device_(device), equation_(std::move(equation)) {}

  void describe(cutensorTensorDescriptor_t* desc, const Operand& operand) {
    KATAL_CHECK(cutensorCreateTensorDescriptor(
                    handle_, desc, static_cast<uint32_t>(operand.modes.size()),
                    operand.extent.data(), operand.stride.data(),
                    util::to_cutensor(operand.type), operand.alignment),
                equation_);
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

  /// Plan preference, workspace estimate and kernel selection.
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

  cutensorHandle_t handle_ = nullptr;
  int device_ = 0;
  std::string equation_;
  OpKind kind_ = OpKind::Contraction;
  cutensorTensorDescriptor_t desc_a_ = nullptr;
  cutensorTensorDescriptor_t desc_b_ = nullptr;
  cutensorTensorDescriptor_t desc_d_ = nullptr;
  cutensorOperationDescriptor_t op_ = nullptr;
  cutensorPlanPreference_t pref_ = nullptr;
  cutensorPlan_t plan_ = nullptr;
  uint64_t workspace_size_ = 0;
  Dtype scalar_type_ = Dtype::F64;
};

class CutensorBackend final : public Backend {
 public:
  explicit CutensorBackend(const Config& config) {
    KATAL_CHECK(cutensorCreate(&handle_), "");
    std::unique_ptr<std::remove_pointer_t<cutensorHandle_t>,
                    decltype(&cutensorDestroy)>
        guard(handle_, &cutensorDestroy);
    if (config.plan_cache_capacity > 0)
      KATAL_CHECK(
          cutensorHandleResizePlanCache(
              handle_, static_cast<uint32_t>(config.plan_cache_capacity)),
          "");
    KATAL_CHECK_CUDA(cudaGetDevice(&device_), "");
    handle_ = guard.release();
  }

  // Destructors are noexcept and swallow errors; there is nothing useful to do
  // with a failure here.
  ~CutensorBackend() override {
    if (handle_ == nullptr) return;
    util::ScopedDevice on_device(device_);
    cutensorDestroy(handle_);
  }

  int device() const noexcept override { return device_; }

  /// The handle and every descriptor belong to the device the backend was
  /// built on; handing them to another device's stream is undefined.
  /// cudaGetDevice is a host-side read, so this costs the hot path little.
  void check_device(std::string_view equation,
                    std::string_view owner) const override {
    int current = 0;
    KATAL_CHECK_CUDA(cudaGetDevice(&current), equation);
    if (current != device_)
      throw Error(Errc::CudaError, equation,
                  "this " + std::string(owner) + " belongs to device " +
                      std::to_string(device_) + " but device " +
                      std::to_string(current) + " is current");
  }

  std::unique_ptr<Kernel> plan(const Operation& operation,
                               const PlanOptions& options) const override {
    return std::make_unique<CutensorKernel>(handle_, device_, operation,
                                            options);
  }

 private:
  cutensorHandle_t handle_ = nullptr;
  int device_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_cutensor_backend(const Config& config) {
  return std::make_unique<CutensorBackend>(config);
}

}  // namespace katal::backend
