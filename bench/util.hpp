#ifndef KATAL_BENCH_UTIL_HPP_
#define KATAL_BENCH_UTIL_HPP_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "../examples/util.hpp"

namespace benchmark {

inline std::vector<std::int64_t> extents(std::string_view labels,
                                         std::int64_t occupied,
                                         std::int64_t virtuals) {
  std::vector<std::int64_t> result;
  for (char label : labels)
    result.push_back(label == 'i' || label == 'j' ? occupied : virtuals);
  return result;
}

// Shapes are fixed by the drivers, rather than supplied by external input.
inline std::int64_t product(const std::vector<std::int64_t>& extent) {
  std::int64_t result = 1;
  for (std::int64_t value : extent) result *= value;
  return result;
}

class Buffer {
 public:
  explicit Buffer(std::int64_t elements) {
    void* pointer = nullptr;
    example::check(cudaMalloc(&pointer, elements * sizeof(double)));
    data_.reset(pointer);
    example::check(cudaMemset(data_.get(), 0, elements * sizeof(double)));
  }

  void* get() const noexcept { return data_.get(); }

 private:
  std::unique_ptr<void, decltype(&example::free_device)> data_{
      nullptr, &example::free_device};
};

}  // namespace benchmark

#endif  // KATAL_BENCH_UTIL_HPP_
