#ifndef CUEIN_TENSOR_REF_HPP_
#define CUEIN_TENSOR_REF_HPP_

#include <complex>
#include <cstdint>
#include <span>

#include <cuein/config.hpp>
#include <cuein/error.hpp>
#include <cuein/util/index.hpp>

namespace cuein {

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

using Extents = util::StaticVector<std::int64_t, max_rank>;

/// A tensor cuEin does not own: a pointer plus a shape. Trivially copyable,
/// passed by value.
struct TensorRef {
  void* data = nullptr;
  Dtype type = Dtype::F64;
  std::span<const std::int64_t> extent;
  std::span<const std::int64_t> stride;  ///< empty means dense in `layout`
  Layout layout = Layout::RowMajor;
  std::uint32_t alignment = 256;
};

inline TensorRef make_ref(void* data, Dtype type,
                          std::span<const std::int64_t> extent,
                          Layout layout = Layout::RowMajor) {
  return TensorRef{data, type, extent, {}, layout, 256};
}

/// Strides, in elements, of a dense tensor with these extents.
inline Extents dense_strides(std::span<const std::int64_t> extent,
                             Layout layout) {
  Extents stride;
  for (std::size_t i = 0; i < extent.size(); ++i) stride.push_back(1);

  if (layout == Layout::RowMajor)
    for (std::size_t i = extent.size(); i-- > 1;)
      stride[i - 1] = stride[i] * extent[i];
  else
    for (std::size_t i = 1; i < extent.size(); ++i)
      stride[i] = stride[i - 1] * extent[i - 1];

  return stride;
}

/// Mode labels, extents and strides in the order cuTENSOR expects, with the
/// first mode varying fastest.
struct ModeOrder {
  util::Index modes;
  Extents extent;
  Extents stride;
};

/// Puts a tensor into that order. A row-major tensor gets its labels, extents
/// and strides reversed together, which moves no data: the descriptor still
/// describes the same memory, and a contraction is invariant under a
/// consistent renaming of its modes. Reversing is what stops cuTENSOR from
/// seeing the slowest-varying mode first and picking a kernel that transposes.
inline ModeOrder mode_order(const util::Index& modes, const TensorRef& t) {
  if (modes.rank() != t.extent.size())
    throw Error(Errc::RankMismatch, std::string(modes),
                "the operand has " + std::to_string(modes.rank()) +
                    " labels but " + std::to_string(t.extent.size()) +
                    " extents");
  if (!t.stride.empty() && t.stride.size() != t.extent.size())
    throw Error(Errc::RankMismatch, std::string(modes),
                "the operand has " + std::to_string(t.extent.size()) +
                    " extents but " + std::to_string(t.stride.size()) +
                    " strides");

  Extents given_stride;
  if (t.stride.empty())
    given_stride = dense_strides(t.extent, t.layout);
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

}  // namespace cuein

#endif  // CUEIN_TENSOR_REF_HPP_
