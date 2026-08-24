#ifndef DEVICEEIN_TEST_REFERENCE_HPP_
#define DEVICEEIN_TEST_REFERENCE_HPP_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <deviceein/equation.hpp>
#include <deviceein/tensor_ref.hpp>

namespace deviceein::test {

/// Advances a mixed-radix counter. Returns false once it wraps, so a rank-zero
/// counter runs its body exactly once in a do-while.
///
/// Not named `advance`: both argument types live in namespace std, so an
/// unqualified call also finds std::advance, which wins on libstdc++.
inline bool advance_counter(std::span<std::int64_t> counter,
                            std::span<const std::int64_t> extent) {
  for (std::size_t i = counter.size(); i-- > 0;) {
    if (++counter[i] < extent[i]) return true;
    counter[i] = 0;
  }
  return false;
}

inline Extents strides_of(const TensorRef& t) {
  if (t.stride.empty()) return dense_strides(t.extent, t.layout);
  Extents s;
  for (std::int64_t v : t.stride) s.push_back(v);
  return s;
}

inline std::size_t position_of(const util::Index& idx, util::Label l) {
  for (std::size_t i = 0; i < idx.rank(); ++i)
    if (idx[i] == l) return i;
  return idx.rank();
}

/// Brute-force einsum over host memory: `out = alpha * <equation> + beta *
/// out`.
///
/// Not part of the library. It exists so the tests have an answer to compare
/// against, and it is written for obviousness rather than speed: one walk over
/// the cartesian product of every distinct label handles contraction,
/// batching, outer products, permutation and reduction with no special cases.
///
/// Labels are read in the equation's own order, not descriptor order, so a
/// result that matches this also confirms that reversing a row-major operand
/// preserved its meaning.
template <class T>
void reference_einsum(const Equation& eq, T alpha,
                      std::span<const TensorRef> in, T beta,
                      const TensorRef& out) {
  // Every distinct label, and the extent each one runs over.
  util::Index all = eq.result();
  for (std::size_t k = 0; k < eq.n_operands(); ++k) all = all | eq.operand(k);

  std::vector<std::int64_t> extent(all.rank(), 0);
  const auto bind = [&](const util::Index& idx,
                        std::span<const std::int64_t> e) {
    for (std::size_t p = 0; p < idx.rank(); ++p) {
      const std::size_t k = position_of(all, idx[p]);
      if (extent[k] != 0 && extent[k] != e[p])
        throw Error(Errc::ExtentMismatch, eq.text(),
                    "label \"" + std::string(1, idx[p]) +
                        "\" has two different extents");
      extent[k] = e[p];
    }
  };
  for (std::size_t k = 0; k < in.size(); ++k) bind(eq.operand(k), in[k].extent);
  bind(eq.result(), out.extent);

  for (std::int64_t e : extent)
    if (e == 0) return;

  // Where each operand's label sits in the counter above.
  std::vector<std::vector<std::size_t>> slot(in.size());
  for (std::size_t k = 0; k < in.size(); ++k)
    for (std::size_t p = 0; p < eq.operand(k).rank(); ++p)
      slot[k].push_back(position_of(all, eq.operand(k)[p]));

  std::vector<std::size_t> out_slot;
  for (std::size_t p = 0; p < eq.result().rank(); ++p)
    out_slot.push_back(position_of(all, eq.result()[p]));

  std::vector<Extents> in_stride;
  for (const TensorRef& t : in) in_stride.push_back(strides_of(t));
  const Extents out_stride = strides_of(out);
  T* d = static_cast<T*>(out.data);

  // out *= beta, spelled as a store when beta is zero so that uninitialised
  // memory cannot leak a NaN into the result.
  {
    std::vector<std::int64_t> c(out.extent.size(), 0);
    do {
      std::int64_t off = 0;
      for (std::size_t p = 0; p < c.size(); ++p) off += c[p] * out_stride[p];
      d[off] = (beta == T{}) ? T{} : beta * d[off];
    } while (advance_counter(c, out.extent));
  }

  std::vector<std::int64_t> c(all.rank(), 0);
  do {
    T term = alpha;
    for (std::size_t k = 0; k < in.size(); ++k) {
      std::int64_t off = 0;
      for (std::size_t p = 0; p < slot[k].size(); ++p)
        off += c[slot[k][p]] * in_stride[k][p];
      term *= static_cast<const T*>(in[k].data)[off];
    }
    std::int64_t off = 0;
    for (std::size_t p = 0; p < out_slot.size(); ++p)
      off += c[out_slot[p]] * out_stride[p];
    d[off] += term;
  } while (advance_counter(c, extent));
}

}  // namespace deviceein::test

#endif  // DEVICEEIN_TEST_REFERENCE_HPP_
