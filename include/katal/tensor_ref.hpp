#ifndef KATAL_TENSOR_REF_HPP_
#define KATAL_TENSOR_REF_HPP_

#include <bit>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>

#include <katal/config.hpp>
#include <katal/error.hpp>
#include <katal/util/index.hpp>

namespace katal {

enum class Dtype { F32, F64, C32, C64, F16, Bf16 };

/// Undefined for anything else, so an unsupported storage type fails here
/// rather than deeper in. F16 and Bf16 have no portable C++ spelling and so
/// have no mapping; pass those as a Dtype directly.
template <class T>
struct DtypeOf;

template <>
struct DtypeOf<float> {
  static constexpr Dtype value = Dtype::F32;
};
template <>
struct DtypeOf<double> {
  static constexpr Dtype value = Dtype::F64;
};
template <>
struct DtypeOf<std::complex<float>> {
  static constexpr Dtype value = Dtype::C32;
};
template <>
struct DtypeOf<std::complex<double>> {
  static constexpr Dtype value = Dtype::C64;
};

template <class T>
inline constexpr Dtype dtype_of = DtypeOf<T>::value;

namespace util {

/// Bytes in one element of `type`.
inline std::size_t size_of(Dtype type) {
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
  throw Error(Errc::DtypeMismatch, {}, "unrecognized tensor data type");
}

}  // namespace util

/// Which mode varies fastest in memory: the last one for RowMajor, the first
/// for ColumnMajor. The flag declares the convention the extent and stride
/// arrays are written in, whether or not those strides are dense.
/// ColumnMajor is the default for `TensorRef`, `make_ref` and a
/// value-initialized `Layout{}`, as it is for `Tensor` storage, cuTENSOR,
/// hipTensor, Eigen and BLAS.
enum class Layout { ColumnMajor, RowMajor };

/// An elementwise function applied to a tensor as it is read, without
/// changing its data. These are cuTENSOR's unary operators. Which ones a
/// given operation and type accepts is up to the backend; a refusal is
/// `Errc::UnsupportedPattern` at plan time.
enum class UnaryOp {
  Identity,
  Conj,
  Neg,
  Abs,
  Sqrt,
  Rcp,  ///< 1 / x
  Exp,
  Log,  ///< natural log
  Sin,
  Cos,
  Tan,
  Sinh,
  Cosh,
  Tanh,
  Asin,
  Acos,
  Atan,
  Asinh,
  Acosh,
  Atanh,
  Ceil,
  Floor,
  Relu,
  Sigmoid,   ///< 1 / (1 + exp(-x))
  Mish,      ///< x * tanh(softplus(x))
  Swish,     ///< x * sigmoid(x)
  SoftPlus,  ///< log(exp(x) + 1)
  SoftSign,  ///< x / (abs(x) + 1)
};

/// Extents or strides for one tensor. Fixed capacity, so building a
/// descriptor allocates nothing.
using Extents = util::StaticVector<std::int64_t, max_rank>;

/// A tensor Katal does not own: a pointer plus a shape. Trivially copyable,
/// passed by value. Spelled through `TensorRef`, which may be written, and
/// `ConstTensorRef`, which is only read; a TensorRef converts to a
/// ConstTensorRef wherever an input is expected.
template <class Pointer>
struct BasicTensorRef {
  /// Device memory, owned by the caller and required to outlive every use.
  Pointer data = nullptr;
  /// Element type of `data`, which the compute type may exceed.
  Dtype type = Dtype::F64;
  /// One extent per mode, in the order the equation names them. The span is a
  /// view: the array behind it must outlive the reference.
  std::span<const std::int64_t> extent;
  /// Strides in **elements**, not bytes, one per mode. Empty means dense in
  /// `layout`, which is the usual case.
  std::span<const std::int64_t> stride;
  /// Which convention `extent` and `stride` are written in.
  Layout layout = Layout::ColumnMajor;
  /// Alignment required by the plan, in bytes. Execution rejects pointers
  /// that do not satisfy it.
  /// `make_ref` and `Tensor::ref` derive it from the address.
  std::uint32_t alignment = 256;
  /// Applied to each element as an input is read; see `apply` and `conj`.
  /// An output must leave it as `Identity`.
  UnaryOp op = UnaryOp::Identity;

  /// Any reference can be read through.
  operator BasicTensorRef<const void*>() const noexcept  // NOLINT: implicit
    requires std::same_as<Pointer, void*>
  {
    return {data, type, extent, stride, layout, alignment, op};
  }
};

using TensorRef = BasicTensorRef<void*>;
using ConstTensorRef = BasicTensorRef<const void*>;

/// The same input, read through `op`: `reduce("ij->i", BinaryOp::Max, 1.0,
/// apply(UnaryOp::Abs, a), 0.0, d)` takes the largest magnitude in each row.
/// Replaces any operator the reference already carried. Contractions accept
/// only Identity and Conj.
inline ConstTensorRef apply(UnaryOp op, ConstTensorRef tensor) noexcept {
  tensor.op = op;
  return tensor;
}

/// The same input, complex conjugated as it is read.
inline ConstTensorRef conj(const ConstTensorRef& tensor) noexcept {
  return apply(UnaryOp::Conj, tensor);
}

/// The largest power of two that divides `data`, capped at 256 so that
/// equivalent buffers share plans: CUDA allocations are 256-byte aligned, and
/// cuTENSOR gains nothing beyond that. A null pointer reads as 256, so a plan
/// built from shapes alone matches a fresh allocation.
inline std::uint32_t alignment_of(const void* data) noexcept {
  constexpr int cap = 8;
  const auto address = reinterpret_cast<std::uintptr_t>(data);
  if (address == 0) return std::uint32_t{1} << cap;
  const int bits = std::countr_zero(address);
  return std::uint32_t{1} << (bits < cap ? bits : cap);
}

inline TensorRef make_ref(void* data, Dtype type,
                          std::span<const std::int64_t> extent,
                          Layout layout = Layout::ColumnMajor) {
  return TensorRef{data, type, extent, {}, layout, alignment_of(data)};
}

/// For an input that is only read.
inline ConstTensorRef make_ref(const void* data, Dtype type,
                               std::span<const std::int64_t> extent,
                               Layout layout = Layout::ColumnMajor) {
  return ConstTensorRef{data, type, extent, {}, layout, alignment_of(data)};
}

/// Shapes alone, for planning before any buffer exists. Without this a null
/// pointer would match both forms above.
inline TensorRef make_ref(std::nullptr_t, Dtype type,
                          std::span<const std::int64_t> extent,
                          Layout layout = Layout::ColumnMajor) {
  return make_ref(static_cast<void*>(nullptr), type, extent, layout);
}

/// Strides, in elements, of a dense tensor with these extents. Throws
/// `Errc::ExtentMismatch` for an extent that is not positive or a stride that
/// overflows. `equation_text` is quoted in that error, when there is one.
inline Extents dense_strides(std::span<const std::int64_t> extent,
                             Layout layout,
                             std::string_view equation_text = {}) {
  for (const auto e : extent)
    if (e <= 0)
      throw Error(Errc::ExtentMismatch, equation_text,
                  "tensor extents must be positive");
  const auto product = [&](std::int64_t stride, std::int64_t e) {
    if (stride > std::numeric_limits<std::int64_t>::max() / e)
      throw Error(Errc::ExtentMismatch, equation_text,
                  "a dense stride overflows std::int64_t");
    return stride * e;
  };
  Extents stride;
  for (std::size_t i = 0; i < extent.size(); ++i) stride.push_back(1);

  if (layout == Layout::RowMajor)
    for (std::size_t i = extent.size(); i-- > 1;)
      stride[i - 1] = product(stride[i], extent[i]);
  else
    for (std::size_t i = 1; i < extent.size(); ++i)
      stride[i] = product(stride[i - 1], extent[i - 1]);

  return stride;
}

/// Mode labels, extents and strides in the order cuTENSOR expects, with the
/// first mode varying fastest.
struct ModeOrder {
  /// Labels in descriptor order, which is reversed from the equation's order
  /// for a row-major operand.
  util::Index modes;
  /// Extents permuted to match `modes`.
  Extents extent;
  /// Strides permuted to match `modes`, in elements.
  Extents stride;
};

/// Puts a tensor into that order. A row-major tensor gets its labels, extents
/// and strides reversed together, which moves no data: the descriptor still
/// describes the same memory, and a contraction is invariant under a
/// consistent renaming of its modes. Reversing is what stops cuTENSOR from
/// seeing the slowest-varying mode first and picking a kernel that transposes.
///
/// `equation_text` is the whole equation `modes` came from, which a rank error
/// quotes; the message names the tensor by its own labels.
inline ModeOrder mode_order(const util::Index& modes, const ConstTensorRef& t,
                            std::string_view equation_text = {}) {
  // Built only on failure, so a valid call allocates nothing for it.
  const auto tensor = [&] { return "tensor \"" + std::string(modes) + "\""; };
  if (modes.rank() != t.extent.size())
    throw Error(Errc::RankMismatch, equation_text,
                tensor() + " has " + std::to_string(modes.rank()) +
                    " labels but " + std::to_string(t.extent.size()) +
                    " extents");
  if (!t.stride.empty() && t.stride.size() != t.extent.size())
    throw Error(Errc::RankMismatch, equation_text,
                tensor() + " has " + std::to_string(t.extent.size()) +
                    " extents but " + std::to_string(t.stride.size()) +
                    " strides");

  Extents given_stride;
  if (t.stride.empty())
    given_stride = dense_strides(t.extent, t.layout, equation_text);
  else
    for (std::int64_t s : t.stride) given_stride.push_back(s);

  const bool reverse = t.layout == Layout::RowMajor;
  ModeOrder m;
  for (std::size_t i = 0; i < modes.rank(); ++i) {
    const std::size_t j = reverse ? modes.rank() - 1 - i : i;
    m.modes.push_back(modes[j]);
    m.extent.push_back(t.extent[j]);
    m.stride.push_back(given_stride[j]);
  }
  return m;
}

}  // namespace katal

#endif  // KATAL_TENSOR_REF_HPP_
