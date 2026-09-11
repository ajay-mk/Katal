#ifndef KATAL_UTIL_INDEX_HPP_
#define KATAL_UTIL_INDEX_HPP_

// Index-set design follows TiledArray's Einsum::Index (ValeevGroup/tiledarray).
// Independently implemented; no TiledArray source is used.

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include <katal/config.hpp>
#include <katal/util/static_vector.hpp>

namespace katal::util {

/// A mode label. One character: with a rank cap in the low tens there are
/// more letters and digits than a tensor can have modes.
using Label = char;

/// Maps destination slots to source slots: `to[i]` is `from[p[i]]`.
using Permutation = StaticVector<std::size_t, katal::max_rank>;

/// An ordered list of mode labels. Owns no memory, so it is usable in constant
/// evaluation and cheap to copy.
///
/// Duplicates are representable (`"ii"` has rank 2) because the diagonal and
/// trace patterns need them; rejecting them is the equation layer's job.
class Index {
 public:
  Index() = default;

  /// Whitespace is ignored, so `"ij, jk -> ik"` and `"ij,jk->ik"` spell the
  /// same equation.
  constexpr explicit Index(std::string_view s) {
    for (char c : s)
      if (c != ' ' && c != '\t') push_back(c);
  }

  constexpr std::size_t rank() const noexcept { return labels_.size(); }

  constexpr Label operator[](std::size_t i) const noexcept {
    return labels_[i];
  }

  constexpr bool contains(Label l) const noexcept {
    for (Label x : labels_)
      if (x == l) return true;
    return false;
  }

  constexpr auto begin() const noexcept { return labels_.begin(); }
  constexpr auto end() const noexcept { return labels_.end(); }

  constexpr void push_back(Label l) {
    if (labels_.size() == labels_.capacity())
      throw std::length_error("index rank exceeds the configured maximum rank");
    labels_.push_back(l);
  }

  explicit operator std::string() const {
    return std::string(labels_.begin(), labels_.end());
  }

  friend constexpr bool operator==(const Index&, const Index&) = default;

 private:
  StaticVector<Label, katal::max_rank> labels_;
};

/// Intersection, in `a`'s order. Never sorts: the resulting label order is the
/// mode order handed to the descriptor, and reordering it can cost cuTENSOR an
/// internal transpose.
constexpr Index operator&(const Index& a, const Index& b) {
  Index r;
  for (Label l : a)
    if (b.contains(l) && !r.contains(l)) r.push_back(l);
  return r;
}

/// Union: `a`'s labels in order, then the labels of `b` that are new.
constexpr Index operator|(const Index& a, const Index& b) {
  Index r;
  for (Label l : a)
    if (!r.contains(l)) r.push_back(l);
  for (Label l : b)
    if (!r.contains(l)) r.push_back(l);
  return r;
}

/// Difference, in `a`'s order.
constexpr Index operator-(const Index& a, const Index& b) {
  Index r;
  for (Label l : a)
    if (!b.contains(l) && !r.contains(l)) r.push_back(l);
  return r;
}

/// Concatenation. Unlike the set operators this keeps duplicates; it is the
/// building primitive, not a set operation.
constexpr Index operator+(const Index& a, const Index& b) {
  Index r = a;
  for (Label l : b) r.push_back(l);
  return r;
}

/// Symmetric difference: the implicit result index when an equation omits `->`.
constexpr Index operator^(const Index& a, const Index& b) {
  return (a - b) + (b - a);
}

/// The permutation taking `from` to `to`, i.e. `p` with `to[i] == from[p[i]]`.
/// Throws `std::invalid_argument` unless the two are permutations of each
/// other; a silently wrong permutation would be silently wrong numerics.
constexpr Permutation permute_from_to(const Index& from, const Index& to) {
  if (from.rank() != to.rank())
    throw std::invalid_argument("katal: permute_from_to rank mismatch");

  Permutation p;
  // ponytail: O(n^2) with a used-slot mask, which is what makes it correct for
  // repeated labels. Tensor ranks are single digits.
  StaticVector<char, katal::max_rank> used;
  for (std::size_t i = 0; i < from.rank(); ++i) used.push_back(0);

  for (Label l : to) {
    std::size_t j = 0;
    for (; j < from.rank(); ++j)
      if (!used[j] && from[j] == l) break;
    if (j == from.rank())
      throw std::invalid_argument(
          "katal: permute_from_to has no unused source label matching the "
          "destination");
    used[j] = 1;
    p.push_back(j);
  }
  return p;
}

}  // namespace katal::util

#endif  // KATAL_UTIL_INDEX_HPP_
