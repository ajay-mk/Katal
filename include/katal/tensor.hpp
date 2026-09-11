#ifndef KATAL_TENSOR_HPP_
#define KATAL_TENSOR_HPP_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <katal/context.hpp>
#include <katal/tensor_ref.hpp>

namespace katal {

namespace detail {

void* allocate_device_bytes(std::size_t bytes, StreamRef stream);
void deallocate_device_bytes(void* data, StreamRef stream) noexcept;

}  // namespace detail

/// A standard allocator backed by the device's stream-ordered memory pool.
template <class T>
class PooledDeviceAllocator {
 public:
  using value_type = T;

  PooledDeviceAllocator() = default;
  explicit PooledDeviceAllocator(StreamRef stream) : stream_(stream) {}

  template <class U>
  PooledDeviceAllocator(const PooledDeviceAllocator<U>& other) noexcept
      : stream_(other.stream()) {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    return static_cast<T*>(
        detail::allocate_device_bytes(n * sizeof(T), stream_));
  }

  void deallocate(T* data, std::size_t) noexcept {
    detail::deallocate_device_bytes(data, stream_);
  }

  StreamRef stream() const noexcept { return stream_; }

  template <class U>
  friend bool operator==(const PooledDeviceAllocator& a,
                         const PooledDeviceAllocator<U>& b) noexcept {
    return a.stream().get() == b.stream().get();
  }

 private:
  StreamRef stream_;
};

/// Dense tensor metadata plus shared ownership of its allocation.
template <class Allocator = PooledDeviceAllocator<std::byte>>
class Tensor {
 public:
  using allocator_type = Allocator;

  static_assert(
      std::is_same_v<typename std::allocator_traits<Allocator>::value_type,
                     std::byte>,
      "katal::Tensor allocators must allocate std::byte");

  explicit Tensor(Dtype type, Extents extent, Allocator allocator = {})
      : type_(type), extent_(std::move(extent)), size_(volume()) {
    const std::size_t element_bytes = size_of(type_);
    if (size_ > std::numeric_limits<std::size_t>::max() / element_bytes)
      throw std::length_error("katal: tensor allocation is too large");
    bytes_ = size_ * element_bytes;
    if (bytes_ != 0)
      storage_ = std::make_shared<Storage>(bytes_, std::move(allocator));
  }

  Dtype type() const noexcept { return type_; }
  const Extents& extents() const noexcept { return extent_; }
  Layout layout() const noexcept { return Layout::ColumnMajor; }
  std::size_t size() const noexcept { return size_; }
  std::size_t bytes() const noexcept { return bytes_; }

  void* data() noexcept { return storage_ ? storage_->data() : nullptr; }
  const void* data() const noexcept {
    return storage_ ? storage_->data() : nullptr;
  }

  TensorRef ref() const noexcept {
    std::uint32_t alignment = 1;
    if (const auto address = reinterpret_cast<std::uintptr_t>(data());
        address != 0) {
      const int bits = std::countr_zero(address);
      // Cap at CUDA's allocation alignment so equivalent buffers share plans.
      alignment <<= bits < 8 ? bits : 8;
    }
    return TensorRef{const_cast<void*>(data()),
                     type_,
                     {extent_.data(), extent_.size()},
                     {},
                     Layout::ColumnMajor,
                     alignment};
  }

 private:
  using Traits = std::allocator_traits<Allocator>;

  class Storage {
   public:
    Storage(std::size_t bytes, Allocator allocator)
        : allocator_(std::move(allocator)), bytes_(bytes) {
      data_ = Traits::allocate(allocator_, bytes_);
    }

    ~Storage() { Traits::deallocate(allocator_, data_, bytes_); }

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* data() noexcept { return std::to_address(data_); }
    const void* data() const noexcept { return std::to_address(data_); }

   private:
    Allocator allocator_;
    std::size_t bytes_;
    typename Traits::pointer data_;
  };

  static std::size_t size_of(Dtype type) {
    switch (type) {
      case Dtype::F16:
      case Dtype::Bf16:
        return 2;
      case Dtype::F32:
        return 4;
      case Dtype::F64:
      case Dtype::C32:
        return 8;
      case Dtype::C64:
        return 16;
    }
    throw std::invalid_argument("katal: unrecognized tensor data type");
  }

  std::size_t volume() const {
    for (std::int64_t e : extent_) {
      if (e < 0) throw std::invalid_argument("katal: negative tensor extent");
      if (e == 0) return 0;
    }

    std::size_t result = 1;
    for (std::int64_t e : extent_) {
      const auto n = static_cast<std::size_t>(e);
      if (result > std::numeric_limits<std::size_t>::max() / n)
        throw std::length_error("katal: tensor extent product is too large");
      result *= n;
    }
    return result;
  }

  Dtype type_;
  Extents extent_;
  std::size_t size_;
  std::size_t bytes_ = 0;
  std::shared_ptr<Storage> storage_;
};

}  // namespace katal

#endif  // KATAL_TENSOR_HPP_
