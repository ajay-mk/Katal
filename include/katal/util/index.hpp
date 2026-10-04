#ifndef KATAL_UTIL_INDEX_HPP_
#define KATAL_UTIL_INDEX_HPP_

// Index-set design follows TiledArray's Einsum::Index (ValeevGroup/tiledarray).
// Independently implemented; no TiledArray source is used.

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <string_view>

#include <katal/config.hpp>
#include <katal/error.hpp>
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

  /// ASCII whitespace is ignored, so `"ij, jk -> ik"` and `"ij,jk->ik"` spell
  /// the same equation, as does one split across lines in a raw string.
  constexpr explicit Index(std::string_view s) {
    for (char c : s)
      if (!is_space(c)) push_back(c);
  }

  /// The characters `std::isspace` accepts in the C locale. Spelled out
  /// because `std::isspace` is neither constexpr nor locale-independent.
  static constexpr bool is_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
           c == '\r';
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
      throw Error(
          Errc::RankMismatch, {},
          "an index has more than " + std::to_string(max_rank) + " labels");
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

/// Symmetric difference: `a`'s labels missing from `b`, then `b`'s missing from
/// `a`. Concatenates, so the result can exceed either operand's rank.
constexpr Index operator^(const Index& a, const Index& b) {
  return (a - b) + (b - a);
}

/// The permutation taking `from` to `to`, i.e. `p` with `to[i] == from[p[i]]`.
/// Throws `Errc::RankMismatch` or `Errc::UnsupportedPattern` unless the two
/// are permutations of each other; a silently wrong permutation would be
/// silently wrong numerics.
constexpr Permutation permute_from_to(const Index& from, const Index& to) {
  if (from.rank() != to.rank())
    throw Error(Errc::RankMismatch, {},
                "permute_from_to: \"" + std::string(from) + "\" and \"" +
                    std::string(to) + "\" differ in rank");

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
      throw Error(Errc::UnsupportedPattern, {},
                  "permute_from_to: \"" + std::string(to) +
                      "\" is not a permutation of \"" + std::string(from) +
                      "\"");
    used[j] = 1;
    p.push_back(j);
  }
  return p;
}

}  // namespace katal::util

#endif  // KATAL_UTIL_INDEX_HPP_
