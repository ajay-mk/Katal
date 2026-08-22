#ifndef CUEIN_UTIL_STATIC_VECTOR_HPP_
#define CUEIN_UTIL_STATIC_VECTOR_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace cuein::util {

/// A sequence with a fixed capacity that never allocates, sized for the
/// single-digit ranks tensors actually have. It throws rather than growing.
///
/// This is what C++26 spells std::inplace_vector. Delete this header once the
/// project can require C++26 and typedef it instead.
///
/// `T` must be default-constructible; the whole array is value-initialized so
/// that instances work in constant evaluation.
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
      throw std::length_error("cuein: static_vector capacity exceeded");
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

}  // namespace cuein::util

#endif  // CUEIN_UTIL_STATIC_VECTOR_HPP_
