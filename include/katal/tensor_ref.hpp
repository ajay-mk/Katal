#ifndef KATAL_TENSOR_REF_HPP_
#define KATAL_TENSOR_REF_HPP_

#include <complex>
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

/// Which mode varies fastest in memory: the last one for RowMajor, the first
/// for ColumnMajor. The flag declares the convention the extent and stride
/// arrays are written in, whether or not those strides are dense.
enum class Layout { RowMajor, ColumnMajor };

/// Extents or strides for one tensor. Fixed capacity, so building a
/// descriptor allocates nothing.
using Extents = util::StaticVector<std::int64_t, max_rank>;

/// A tensor Katal does not own: a pointer plus a shape. Trivially copyable,
/// passed by value.
struct TensorRef {
  /// Device memory, owned by the caller and required to outlive every use.
  void* data = nullptr;
  /// Element type of `data`, which the compute type may exceed.
  Dtype type = Dtype::F64;
  /// One extent per mode, in the order the equation names them. The span is a
  /// view: the array behind it must outlive the reference.
  std::span<const std::int64_t> extent;
  /// Strides in **elements**, not bytes, one per mode. Empty means dense in
  /// `layout`, which is the usual case.
  std::span<const std::int64_t> stride;
  /// Which convention `extent` and `stride` are written in.
  Layout layout = Layout::RowMajor;
  /// Alignment of `data` in bytes, asserted rather than enforced: a value
  /// larger than the pointer actually satisfies is a way to get wrong answers.
  std::uint32_t alignment = 256;
};

inline TensorRef make_ref(void* data, Dtype type,
                          std::span<const std::int64_t> extent,
                          Layout layout = Layout::RowMajor) {
  return TensorRef{data, type, extent, {}, layout, 256};
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
/// quotes; the message names the operand's own labels.
inline ModeOrder mode_order(const util::Index& modes, const TensorRef& t,
                            std::string_view equation_text = {}) {
  const std::string operand = "operand \"" + std::string(modes) + "\"";
  if (modes.rank() != t.extent.size())
    throw Error(Errc::RankMismatch, equation_text,
                operand + " has " + std::to_string(modes.rank()) +
                    " labels but " + std::to_string(t.extent.size()) +
                    " extents");
  if (!t.stride.empty() && t.stride.size() != t.extent.size())
    throw Error(Errc::RankMismatch, equation_text,
                operand + " has " + std::to_string(t.extent.size()) +
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
