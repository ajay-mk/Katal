#ifndef KATAL_EXAMPLES_UTIL_HPP_
#define KATAL_EXAMPLES_UTIL_HPP_

// Scaffolding shared by the examples, kept out of them so each file reads as
// the equations it evaluates and little else.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace example {

/// A tensor on the device filled with random values. Katal never allocates
/// device memory, so owning it is the caller's job.
template <class T>
class RandomTensor {
 public:
  RandomTensor(std::vector<std::int64_t> extent, unsigned seed)
      : extent_(std::move(extent)) {
    std::int64_t count = 1;
    for (std::int64_t e : extent_) count *= e;

    std::vector<T> host(static_cast<std::size_t>(count));
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> value(-0.05, 0.05);
    for (T& x : host) {
      if constexpr (std::is_same_v<T, double>)
        x = value(engine);
      else
        x = T(value(engine), value(engine));
    }

    cudaMalloc(&data_, host.size() * sizeof(T));
    cudaMemcpy(data_, host.data(), host.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  }

  ~RandomTensor() { cudaFree(data_); }
  RandomTensor(const RandomTensor&) = delete;
  RandomTensor& operator=(const RandomTensor&) = delete;

  /// Converts to the non-owning view Katal takes, so a RandomTensor can be
  /// passed straight to a contraction. It must outlive the call.
  operator katal::TensorRef() const {  // NOLINT: implicit on purpose
    return katal::make_ref(data_, katal::dtype_of<T>, extent_);
  }

  T scalar() const {
    T value{};
    cudaMemcpy(&value, data_, sizeof(value), cudaMemcpyDeviceToHost);
    return value;
  }

 private:
  void* data_ = nullptr;
  std::vector<std::int64_t> extent_;
};

/// Running total of the arithmetic the examples have asked for, so a residual
/// can report what it achieved without a separate list of its own equations.
inline double flops = 0.0;

/// Two operations per contracted element, over the product of every distinct
/// extent in the equation. The equation is parsed with Katal's own parser,
/// so the count cannot drift from what was actually run.
inline double count(std::string_view equation, const katal::TensorRef& a,
                    const katal::TensorRef& b) {
  const katal::Equation parsed = katal::Equation::parse(equation);
  katal::util::Index all = parsed.result();
  for (std::size_t k = 0; k < parsed.n_operands(); ++k)
    all = all | parsed.operand(k);

  const auto extent_of = [&](katal::util::Label label) -> std::int64_t {
    for (std::size_t p = 0; p < parsed.operand(0).rank(); ++p)
      if (parsed.operand(0)[p] == label) return a.extent[p];
    for (std::size_t p = 0; p < parsed.operand(1).rank(); ++p)
      if (parsed.operand(1)[p] == label) return b.extent[p];
    return 1;
  };

  double product = 2.0;
  for (std::size_t i = 0; i < all.rank(); ++i)
    product *= static_cast<double>(extent_of(all[i]));
  return product;
}

/// Shorthand for katal::einsum that also tallies the work. A plain alias
/// will not do: einsum is an overload set, not a single function.
inline void ein(std::string_view equation, katal::Scalar alpha,
                const katal::TensorRef& a, const katal::TensorRef& b,
                katal::Scalar beta, const katal::TensorRef& d) {
  flops += count(equation, a, b);
  katal::einsum(equation, alpha, a, b, beta, d);
}

/// The one-operand form: a permutation or a copy, which moves data rather than
/// doing arithmetic, so it adds nothing to the tally.
inline void ein(std::string_view equation, katal::Scalar alpha,
                const katal::TensorRef& a, katal::Scalar beta,
                const katal::TensorRef& d) {
  katal::einsum(equation, alpha, a, beta, d);
}

/// Wall time on the device, in seconds.
class Timer {
 public:
  Timer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }
  ~Timer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  double seconds() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    float milliseconds = 0.0F;
    cudaEventElapsedTime(&milliseconds, start_, stop_);
    return milliseconds / 1e3;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

inline void report(const char* what, double seconds) {
  std::printf("%s: %.2f GFLOP in %.2f ms  (%.2f TFLOP/s)\n", what, flops / 1e9,
              seconds * 1e3, flops / seconds / 1e12);
}

}  // namespace example

#endif  // KATAL_EXAMPLES_UTIL_HPP_
