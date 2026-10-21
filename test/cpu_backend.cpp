#include "cpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "context_impl.hpp"
#include "numeric.hpp"
#include "reference.hpp"
#include <enumstr.hpp>

#include <katal/error.hpp>
#include <katal/scalar.hpp>
#include <katal/tensor_ref.hpp>

namespace katal::test {
namespace {

using backend::Kernel;
using backend::Operand;
using backend::Operation;
using backend::OpKind;

bool is_complex(Dtype type) { return type == Dtype::C32 || type == Dtype::C64; }

bool is_supported(Dtype type) {
  return type != Dtype::F16 && type != Dtype::Bf16;
}

template <class T>
inline constexpr bool is_complex_v = !std::is_same_v<T, real_of<T>>;

/// `value` as a T. Planning refuses a complex tensor under a real compute
/// type, so the imaginary part dropped here is always zero.
template <class T, class U>
T cast(U value) {
  if constexpr (is_complex_v<T>) {
    using R = typename T::value_type;
    return T(static_cast<R>(std::real(value)),
             static_cast<R>(std::imag(value)));
  } else {
    return static_cast<T>(std::real(value));
  }
}

/// The element at `offset` elements from `base`, widened or narrowed to T.
template <class T>
T load(const void* base, Dtype type, std::int64_t offset) {
  switch (type) {
    case Dtype::F32:
      return cast<T>(static_cast<const float*>(base)[offset]);
    case Dtype::F64:
      return cast<T>(static_cast<const double*>(base)[offset]);
    case Dtype::C32:
      return cast<T>(static_cast<const std::complex<float>*>(base)[offset]);
    case Dtype::C64:
      return cast<T>(static_cast<const std::complex<double>*>(base)[offset]);
    case Dtype::F16:
    case Dtype::Bf16:
      break;  // refused at plan time
  }
  return T{};
}

template <class T>
void store(void* base, Dtype type, std::int64_t offset, T value) {
  switch (type) {
    case Dtype::F32:
      static_cast<float*>(base)[offset] = cast<float>(value);
      return;
    case Dtype::F64:
      static_cast<double*>(base)[offset] = cast<double>(value);
      return;
    case Dtype::C32:
      static_cast<std::complex<float>*>(base)[offset] =
          cast<std::complex<float>>(value);
      return;
    case Dtype::C64:
      static_cast<std::complex<double>*>(base)[offset] =
          cast<std::complex<double>>(value);
      return;
    case Dtype::F16:
    case Dtype::Bf16:
      return;  // refused at plan time
  }
}

template <class T>
using Unary = T (*)(T);

template <class T>
using Binary = T (*)(T, T);

/// The unary operators every compute type has, as tensor_ref.hpp defines
/// them, or null for one that has no complex form.
template <class T>
Unary<T> any_unary(UnaryOp op) {
  switch (op) {
    case UnaryOp::Identity:
      return [](T x) { return x; };
    case UnaryOp::Conj:
      if constexpr (is_complex_v<T>)
        return [](T x) { return std::conj(x); };
      else
        return [](T x) { return x; };
    case UnaryOp::Neg:
      return [](T x) { return -x; };
    case UnaryOp::Abs:
      return [](T x) { return T(std::abs(x)); };
    case UnaryOp::Sqrt:
      return [](T x) { return std::sqrt(x); };
    case UnaryOp::Rcp:
      return [](T x) { return T(1) / x; };
    case UnaryOp::Exp:
      return [](T x) { return std::exp(x); };
    case UnaryOp::Log:
      return [](T x) { return std::log(x); };
    case UnaryOp::Sin:
      return [](T x) { return std::sin(x); };
    case UnaryOp::Cos:
      return [](T x) { return std::cos(x); };
    case UnaryOp::Tan:
      return [](T x) { return std::tan(x); };
    case UnaryOp::Sinh:
      return [](T x) { return std::sinh(x); };
    case UnaryOp::Cosh:
      return [](T x) { return std::cosh(x); };
    case UnaryOp::Tanh:
      return [](T x) { return std::tanh(x); };
    case UnaryOp::Asin:
      return [](T x) { return std::asin(x); };
    case UnaryOp::Acos:
      return [](T x) { return std::acos(x); };
    case UnaryOp::Atan:
      return [](T x) { return std::atan(x); };
    case UnaryOp::Asinh:
      return [](T x) { return std::asinh(x); };
    case UnaryOp::Acosh:
      return [](T x) { return std::acosh(x); };
    case UnaryOp::Atanh:
      return [](T x) { return std::atanh(x); };
    default:
      return nullptr;
  }
}

/// The unary operators only a real compute type has, or null.
template <class T>
Unary<T> real_unary(UnaryOp op) {
  switch (op) {
    case UnaryOp::Ceil:
      return [](T x) { return std::ceil(x); };
    case UnaryOp::Floor:
      return [](T x) { return std::floor(x); };
    case UnaryOp::Relu:
      return [](T x) { return std::max(x, T(0)); };
    case UnaryOp::Sigmoid:
      return [](T x) { return T(1) / (T(1) + std::exp(-x)); };
    case UnaryOp::Mish:
      return [](T x) { return x * std::tanh(std::log1p(std::exp(x))); };
    case UnaryOp::Swish:
      return [](T x) { return x / (T(1) + std::exp(-x)); };
    case UnaryOp::SoftPlus:
      return [](T x) { return std::log1p(std::exp(x)); };
    case UnaryOp::SoftSign:
      return [](T x) { return x / (std::abs(x) + T(1)); };
    default:
      return nullptr;
  }
}

/// `op` in compute type T, or null when T has no such operator.
template <class T>
Unary<T> unary(UnaryOp op) {
  if (const Unary<T> f = any_unary<T>(op)) return f;
  if constexpr (is_complex_v<T>)
    return nullptr;
  else
    return real_unary<T>(op);
}

/// `op` in compute type T, or null for Max and Min on complex values, which
/// have no order.
template <class T>
Binary<T> binary(BinaryOp op) {
  switch (op) {
    case BinaryOp::Add:
      return [](T x, T y) { return x + y; };
    case BinaryOp::Multiply:
      return [](T x, T y) { return x * y; };
    case BinaryOp::Max:
      if constexpr (is_complex_v<T>)
        return nullptr;
      else
        return [](T x, T y) { return std::fmax(x, y); };
    case BinaryOp::Min:
      if constexpr (is_complex_v<T>)
        return nullptr;
      else
        return [](T x, T y) { return std::fmin(x, y); };
  }
  return nullptr;
}

Error refuse(const Operation& operation, const std::string& why) {
  return Error(Errc::UnsupportedPattern, operation.equation,
               "the CPU backend " + why);
}

/// Runs one Operation in compute type T.
///
/// The loop modes are the output's, followed by every mode only the inputs
/// have. Each output element is visited once: the trailing modes are walked
/// for it, summed for a contraction or combined with the operation's operator
/// for a reduction, and the result is scaled and written. An elementwise
/// operation or a permutation has no trailing modes, since every input mode
/// is an output mode, and an input lacking an output mode has a zero stride
/// along it, which is the broadcast.
template <class T>
class CpuKernel final : public Kernel {
 public:
  /// Throws `Errc::UnsupportedPattern` for an operator T has no form of.
  explicit CpuKernel(const Operation& operation) : kind_(operation.kind) {
    const auto add_modes = [this](const Operand& operand) {
      for (std::size_t p = 0; p < operand.modes.size(); ++p)
        if (std::find(modes_.begin(), modes_.end(), operand.modes[p]) ==
            modes_.end()) {
          modes_.push_back(operand.modes[p]);
          extent_.push_back(operand.extent[p]);
        }
    };
    add_modes(operation.output);
    outer_ = modes_.size();
    for (const Operand& input : operation.inputs) add_modes(input);

    for (const Operand& input : operation.inputs) {
      const Unary<T> op = unary<T>(input.op);
      if (op == nullptr)
        throw refuse(operation,
                     "has no complex form of " + enumstr::to_string(input.op));
      inputs_.push_back({input.type, op, strides(input)});
    }
    output_ = {operation.output.type, nullptr, strides(operation.output)};

    // A contraction sums its trailing modes. A permutation has none.
    const bool own = operation.kind == OpKind::Reduction ||
                     operation.kind == OpKind::Elementwise;
    combine_ = binary<T>(own ? operation.binary : BinaryOp::Add);
    if (combine_ == nullptr)
      throw refuse(operation, "cannot order complex values for " +
                                  enumstr::to_string(operation.binary));
  }

  std::uint64_t workspace_size() const noexcept override { return 0; }

  void run(std::span<const Scalar> scalars, std::span<const void* const> inputs,
           const void* c, void* d, Workspace /*workspace*/,
           StreamRef /*stream*/) const override {
    const T alpha = cast<T>(scalars[0].value());
    const T beta =
        kind_ == OpKind::Permutation ? T{} : cast<T>(scalars[1].value());
    // What scales c: beta, except for an elementwise operation, whose beta
    // scales its second input.
    const T c_scale =
        kind_ == OpKind::Elementwise ? cast<T>(scalars[2].value()) : beta;

    std::vector<std::int64_t> index(modes_.size(), 0);
    const std::span<std::int64_t> outer(index.data(), outer_);
    const std::span<std::int64_t> inner(index.data() + outer_,
                                        index.size() - outer_);
    const std::span<const std::int64_t> outer_extent(extent_.data(), outer_);
    const std::span<const std::int64_t> inner_extent(extent_.data() + outer_,
                                                     extent_.size() - outer_);
    do {
      T value{};
      if (kind_ == OpKind::Elementwise) {
        value = combine_(alpha * read(0, inputs, index),
                         beta * read(1, inputs, index));
      } else {
        std::fill(inner.begin(), inner.end(), 0);
        bool first = true;
        T sum{};
        do {
          T term = read(0, inputs, index);
          if (kind_ == OpKind::Contraction)
            for (std::size_t k = 1; k < inputs_.size(); ++k)
              term *= read(k, inputs, index);
          sum = first ? term : combine_(sum, term);
          first = false;
        } while (advance_counter(inner, inner_extent));
        value = alpha * sum;
      }
      const std::int64_t at = offset(output_, index);
      // A zero scale does not read c, which may then hold anything.
      if (c_scale != T{}) value += c_scale * load<T>(c, output_.type, at);
      store(d, output_.type, at, value);
    } while (advance_counter(outer, outer_extent));
  }

 private:
  /// One tensor as the kernel walks it.
  struct View {
    Dtype type = Dtype::F64;
    /// Applied as an input is read; null for the output.
    Unary<T> op = nullptr;
    /// Along every loop mode, zero along a mode the tensor lacks.
    std::vector<std::int64_t> stride;
  };

  std::vector<std::int64_t> strides(const Operand& operand) const {
    std::vector<std::int64_t> stride(modes_.size(), 0);
    for (std::size_t p = 0; p < operand.modes.size(); ++p) {
      const auto m = static_cast<std::size_t>(
          std::find(modes_.begin(), modes_.end(), operand.modes[p]) -
          modes_.begin());
      stride[m] = operand.stride[p];
    }
    return stride;
  }

  static std::int64_t offset(const View& v,
                             const std::vector<std::int64_t>& index) {
    std::int64_t at = 0;
    for (std::size_t m = 0; m < index.size(); ++m) at += index[m] * v.stride[m];
    return at;
  }

  T read(std::size_t k, std::span<const void* const> inputs,
         const std::vector<std::int64_t>& index) const {
    const View& v = inputs_[k];
    return v.op(load<T>(inputs[k], v.type, offset(v, index)));
  }

  OpKind kind_;
  /// How an elementwise operation combines its inputs, or how the trailing
  /// modes combine: Add for a contraction.
  Binary<T> combine_ = nullptr;
  std::vector<std::int32_t> modes_;
  std::vector<std::int64_t> extent_;
  /// How many of `modes_` are the output's.
  std::size_t outer_ = 0;
  std::vector<View> inputs_;
  View output_;
};

class CpuBackend final : public backend::Backend {
 public:
  /// No CUDA device: this backend runs on the host.
  int device() const noexcept override { return -1; }

  void check_device(std::string_view /*equation*/,
                    std::string_view /*owner*/) const override {}

  std::unique_ptr<Kernel> plan(const Operation& operation,
                               const PlanOptions& /*options*/) const override {
    check(operation);
    switch (operation.compute) {
      case Dtype::F32:
        return std::make_unique<CpuKernel<float>>(operation);
      case Dtype::F64:
        return std::make_unique<CpuKernel<double>>(operation);
      case Dtype::C32:
        return std::make_unique<CpuKernel<std::complex<float>>>(operation);
      case Dtype::C64:
        return std::make_unique<CpuKernel<std::complex<double>>>(operation);
      case Dtype::F16:
      case Dtype::Bf16:
        break;  // refused by check
    }
    throw refuse(operation, "got an unrecognized compute type");
  }

 private:
  /// Refuses at plan time the shapes and types run could not handle. The
  /// kernel refuses the operators.
  static void check(const Operation& operation) {
    // A contraction takes two inputs or three, and multiplies them all.
    const std::size_t n = operation.inputs.size();
    const bool counted =
        operation.kind == OpKind::Contraction
            ? n == 2 || n == 3
            : n == (operation.kind == OpKind::Elementwise ? 2U : 1U);
    if (!counted) throw refuse(operation, "got the wrong number of inputs");

    std::vector<const Operand*> tensors;
    for (const Operand& input : operation.inputs) tensors.push_back(&input);
    tensors.push_back(&operation.output);

    if (!is_supported(operation.compute))
      throw refuse(operation, "computes only in F32, F64, C32 or C64");
    for (const Operand* t : tensors) {
      if (!is_supported(t->type))
        throw refuse(operation, "stores only F32, F64, C32 or C64");
      if (is_complex(t->type) && !is_complex(operation.compute))
        throw refuse(operation,
                     "cannot read or write a complex tensor in a real "
                     "compute type");
      if (t->extent.size() != t->modes.size() ||
          t->stride.size() != t->modes.size())
        throw refuse(operation,
                     "got modes, extents and strides that differ in length");
    }
    if (is_complex(operation.compute) && !is_complex(operation.output.type))
      throw refuse(operation, "cannot write a complex result to a real output");
    if (operation.output.op != UnaryOp::Identity)
      throw refuse(operation, "cannot apply an operator to the output");

    // Every mode has one extent in every tensor, and only a contraction or a
    // reduction may drop an input mode from the output.
    const bool drops = operation.kind == OpKind::Contraction ||
                       operation.kind == OpKind::Reduction;
    const Operand& out = operation.output;
    std::map<std::int32_t, std::int64_t> extent;
    for (const Operand* t : tensors)
      for (std::size_t p = 0; p < t->modes.size(); ++p) {
        const auto [it, added] = extent.emplace(t->modes[p], t->extent[p]);
        if (!added && it->second != t->extent[p])
          throw Error(Errc::ExtentMismatch, operation.equation,
                      "the CPU backend got a mode with two extents");
        if (!drops && std::find(out.modes.begin(), out.modes.end(),
                                t->modes[p]) == out.modes.end())
          throw refuse(operation, "cannot drop an input mode");
      }
  }
};

}  // namespace

std::unique_ptr<backend::Backend> make_cpu_backend() {
  return std::make_unique<CpuBackend>();
}

Context cpu_context(Config config) {
  return Context::Impl::with_backend(make_cpu_backend(), std::move(config));
}

namespace {

constexpr std::align_val_t host_alignment{256};

}  // namespace

Workspace HostAllocator::allocate(std::uint64_t bytes,
                                  StreamRef /*stream*/) const {
  if (bytes == 0) return {};
  return {::operator new(static_cast<std::size_t>(bytes), host_alignment),
          bytes};
}

void HostAllocator::deallocate(Workspace workspace,
                               StreamRef /*stream*/) const noexcept {
  if (workspace.ptr != nullptr)
    ::operator delete(workspace.ptr, host_alignment);
}

}  // namespace katal::test
