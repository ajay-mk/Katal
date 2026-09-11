#ifndef KATAL_UTIL_STATIC_VECTOR_HPP_
#define KATAL_UTIL_STATIC_VECTOR_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace katal::util {

/// A fixed-capacity sequence that never allocates, sized for the single-digit
/// ranks tensors actually have. Throws rather than growing.
///
/// C++26 spells this std::inplace_vector; delete this header once that can be
/// required. `T` must be default-constructible, since the whole array is
/// value-initialized so instances work in constant evaluation.
template <class T, std::size_t N>
class StaticVector {
 public:
  constexpr StaticVector() = default;

  constexpr StaticVector(std::initializer_list<T> init) {
    for (const T& v : init) push_back(v);
  }

  static constexpr std::size_t capacity() noexcept { return N; }
  constexpr std::size_t size() const noexcept { return size_; }
  constexpr bool empty() const noexcept { return size_ == 0; }

  constexpr const T& operator[](std::size_t i) const noexcept {
    return data_[i];
  }
  constexpr T& operator[](std::size_t i) noexcept { return data_[i]; }

  constexpr const T* data() const noexcept { return data_.data(); }

  constexpr auto begin() const noexcept { return data_.begin(); }
  constexpr auto end() const noexcept { return data_.begin() + size_; }

  constexpr void push_back(const T& v) {
    if (size_ == N)
      throw std::length_error("katal: static_vector capacity exceeded");
    data_[size_++] = v;
  }

  /// Compares contents, so a trailing default-constructed element is a real
  /// element and not padding.
  friend constexpr bool operator==(const StaticVector& a,
                                   const StaticVector& b) {
    return a.size_ == b.size_ && std::equal(a.begin(), a.end(), b.begin());
  }

 private:
  std::array<T, N> data_{};
  std::size_t size_ = 0;
};

}  // namespace katal::util

#endif  // KATAL_UTIL_STATIC_VECTOR_HPP_
