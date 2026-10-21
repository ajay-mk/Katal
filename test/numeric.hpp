#ifndef KATAL_TEST_NUMERIC_HPP_
#define KATAL_TEST_NUMERIC_HPP_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

/// Test data and a tolerance for comparing results that were summed in
/// different orders.
namespace katal::test {

template <class T>
struct RealOf {
  using type = T;
};
template <class T>
struct RealOf<std::complex<T>> {
  using type = T;
};
template <class T>
using real_of = typename RealOf<T>::type;

/// The elements a dense tensor of this shape holds.
inline std::size_t count(const std::vector<std::int64_t>& shape) {
  std::size_t n = 1;
  for (const std::int64_t e : shape) n *= static_cast<std::size_t>(e);
  return n;
}

/// `n` values drawn uniformly from [lo, hi), the same for the same seed. A
/// complex value draws its real and imaginary parts separately.
template <class T>
std::vector<T> random_values(std::size_t n, std::uint32_t seed,
                             double lo = -1.0, double hi = 1.0) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<double> draw(lo, hi);
  std::vector<T> values(n);
  for (T& v : values) {
    using R = real_of<T>;
    if constexpr (std::is_same_v<T, R>) {
      v = static_cast<R>(draw(engine));
    } else {
      const auto re = static_cast<R>(draw(engine));
      v = T(re, static_cast<R>(draw(engine)));
    }
  }
  return values;
}

/// Relative tolerance for a result accumulated in T: loose enough for a sum
/// of a few hundred terms taken in another order, tight enough to catch a
/// misplaced element.
template <class T>
constexpr double tolerance() {
  return std::is_same_v<real_of<T>, float> ? 1e-5 : 1e-12;
}

/// Whether `got` matches `want` to `tol`, relative to the largest magnitude
/// in `want`, so that a cancelling element is not held to a tighter bound
/// than its neighbours. An all-zero `want` makes the bound absolute.
template <class T>
bool close(const std::vector<T>& got, const std::vector<T>& want,
           double tol = tolerance<T>()) {
  if (got.size() != want.size()) return false;
  double scale = 0.0;
  for (const T& w : want) scale = std::max(scale, double(std::abs(w)));
  if (scale == 0.0) scale = 1.0;
  for (std::size_t i = 0; i < got.size(); ++i)
    if (!(double(std::abs(got[i] - want[i])) <= tol * scale)) return false;
  return true;
}

}  // namespace katal::test

#endif  // KATAL_TEST_NUMERIC_HPP_
