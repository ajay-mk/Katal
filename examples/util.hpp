#ifndef KATAL_EXAMPLES_UTIL_HPP_
#define KATAL_EXAMPLES_UTIL_HPP_

// Scaffolding shared by the examples, kept out of them so each file reads as
// the equations it evaluates and little else.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <katal/plan.hpp>

namespace example {

inline void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw katal::Error(katal::Errc::CudaError, "", cudaGetErrorString(status));
}

inline void free_device(void* pointer) noexcept {
  if (cudaFree(pointer) != cudaSuccess) std::terminate();
}

inline void destroy_event(cudaEvent_t event) noexcept {
  if (cudaEventDestroy(event) != cudaSuccess) std::terminate();
}

/// Caller-owned device storage filled with random values. This demonstrates
/// passing external buffers to Katal; tensor_storage.cpp uses katal::Tensor.
template <class T>
class RandomTensor {
 public:
  RandomTensor(std::vector<std::int64_t> extent, unsigned seed)
      : extent_(std::move(extent)) {
    std::size_t count = 1;
    for (std::int64_t e : extent_) {
      if (e <= 0)
        throw std::invalid_argument("example: extents must be positive");
      const auto size = static_cast<std::size_t>(e);
      if (count > std::numeric_limits<std::size_t>::max() / sizeof(T) / size)
        throw std::length_error("example: tensor is too large");
      count *= size;
    }

    std::vector<T> host(static_cast<std::size_t>(count));
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> value(-0.05, 0.05);
    for (T& x : host) {
      if constexpr (std::is_same_v<T, double>)
        x = value(engine);
      else
        x = T(value(engine), value(engine));
    }

    void* pointer = nullptr;
    check(cudaMalloc(&pointer, host.size() * sizeof(T)));
    data_.reset(pointer);
    check(cudaMemcpy(data_.get(), host.data(), host.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
  }

  RandomTensor(const RandomTensor&) = delete;
  RandomTensor& operator=(const RandomTensor&) = delete;

  /// Converts to the non-owning view Katal takes, so a RandomTensor can be
  /// passed straight to a contraction. It must outlive the submitted GPU work.
  operator katal::TensorRef() const {  // NOLINT: implicit on purpose
    return katal::make_ref(data_.get(), katal::dtype_of<T>, extent_);
  }

  T scalar() const {
    T value{};
    check(
        cudaMemcpy(&value, data_.get(), sizeof(value), cudaMemcpyDeviceToHost));
    return value;
  }

 private:
  std::unique_ptr<void, decltype(&free_device)> data_{nullptr, &free_device};
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
  std::string all(parsed.result());
  for (std::size_t k = 0; k < parsed.n_operands(); ++k)
    for (const auto label : parsed.operand(k))
      if (all.find(label) == std::string::npos) all.push_back(label);

  const auto extent_of = [&](katal::util::Label label) -> std::int64_t {
    for (std::size_t p = 0; p < parsed.operand(0).rank(); ++p)
      if (parsed.operand(0)[p] == label) return a.extent[p];
    for (std::size_t p = 0; p < parsed.operand(1).rank(); ++p)
      if (parsed.operand(1)[p] == label) return b.extent[p];
    return 1;
  };

  double product = 2.0;
  for (std::size_t i = 0; i < all.size(); ++i)
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
  Timer() { check(cudaEventRecord(start_.get())); }
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

  double seconds() {
    check(cudaEventRecord(stop_.get()));
    check(cudaEventSynchronize(stop_.get()));
    float milliseconds = 0.0F;
    check(cudaEventElapsedTime(&milliseconds, start_.get(), stop_.get()));
    return milliseconds / 1e3;
  }

 private:
  using Event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>,
                                decltype(&destroy_event)>;
  static Event make_event() {
    cudaEvent_t event = nullptr;
    check(cudaEventCreate(&event));
    return Event(event, &destroy_event);
  }
  Event start_ = make_event();
  Event stop_ = make_event();
};

inline void report(const char* what, double seconds) {
  std::printf("%s: %.2f GFLOP in %.2f ms  (%.2f TFLOP/s)\n", what, flops / 1e9,
              seconds * 1e3, flops / seconds / 1e12);
}

}  // namespace example

#endif  // KATAL_EXAMPLES_UTIL_HPP_
