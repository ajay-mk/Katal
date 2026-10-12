#ifndef KATAL_TENSOR_HPP_
#define KATAL_TENSOR_HPP_

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

#include <katal/context.hpp>
#include <katal/plan.hpp>
#include <katal/tensor_ref.hpp>

namespace katal {

namespace detail {

void* allocate_device_bytes(std::size_t bytes, StreamRef stream);
void deallocate_device_bytes(void* data, StreamRef stream) noexcept;
void* allocate_managed_bytes(std::size_t bytes);
void deallocate_managed_bytes(void* data, StreamRef stream) noexcept;
/// Copies device bytes to the host once `stream` reaches the copy, and waits.
void copy_to_host(void* host, const void* device, std::size_t bytes,
                  StreamRef stream);

/// `"abc->"` for rank 3: the equation that reduces every label away. Letters
/// first, then any other byte the grammar accepts as a label, so every rank
/// an equation can spell has one.
inline std::string full_reduction_equation(std::size_t rank) {
  std::string text;
  const auto take = [&](unsigned char b) {
    const char c = static_cast<char>(b);
    if (text.size() < rank && c != ',' && c != '-' && c != '>' &&
        !util::Index::is_space(c) && text.find(c) == std::string::npos)
      text.push_back(c);
  };
  for (unsigned char b = 'a'; b <= 'z'; ++b) take(b);
  for (unsigned char b = 'A'; b <= 'Z'; ++b) take(b);
  for (int b = 1; b < 256; ++b) take(static_cast<unsigned char>(b));
  if (text.size() < rank)
    throw Error(Errc::RankMismatch, {},
                "too many modes to label in one equation");
  return text + "->";
}

}  // namespace detail

/// A standard allocator backed by the device's stream-ordered memory pool.
/// All GPU uses must run on this stream, which must outlive the allocation.
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

/// Host/device storage whose GPU uses must all run on the supplied stream.
/// The stream must outlive the allocation; deallocation waits for its work.
template <class T>
class ManagedAllocator {
 public:
  using value_type = T;

  ManagedAllocator() = default;
  explicit ManagedAllocator(StreamRef stream) : stream_(stream) {}

  template <class U>
  ManagedAllocator(const ManagedAllocator<U>& other) noexcept
      : stream_(other.stream()) {}

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    return static_cast<T*>(detail::allocate_managed_bytes(n * sizeof(T)));
  }

  void deallocate(T* data, std::size_t) noexcept {
    detail::deallocate_managed_bytes(data, stream_);
  }

  StreamRef stream() const noexcept { return stream_; }

  template <class U>
  friend bool operator==(const ManagedAllocator& a,
                         const ManagedAllocator<U>& b) noexcept {
    return a.stream().get() == b.stream().get();
  }

 private:
  StreamRef stream_;
};

/// Dense tensor metadata plus shared ownership of its allocation.
/// Copies alias the data. A ref() also borrows this object's extent metadata.
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
    // Too large to allocate is out of memory, as the allocators report it.
    if (size_ > std::numeric_limits<std::size_t>::max() / element_bytes)
      throw std::bad_array_new_length();
    bytes_ = size_ * element_bytes;
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

  /// A reference to write through. Copies share storage, so this is how an
  /// output is passed; a const Tensor gives only a ConstTensorRef. A
  /// temporary has none, so a reference cannot be stored from one and then
  /// outlive the extents it borrows.
  TensorRef ref() & noexcept {
    return make_ref(data(), type_, {extent_.data(), extent_.size()},
                    Layout::ColumnMajor);
  }
  ConstTensorRef ref() const& noexcept {
    return make_ref(data(), type_, {extent_.data(), extent_.size()},
                    Layout::ColumnMajor);
  }
  TensorRef ref() && = delete;
  ConstTensorRef ref() const&& = delete;

  /// Lets a Tensor be passed straight to einsum, make_plan and cached_plan,
  /// with the same const and temporary rules as ref().
  operator TensorRef() & noexcept { return ref(); }
  operator ConstTensorRef() const& noexcept { return ref(); }
  operator TensorRef() && = delete;
  operator ConstTensorRef() const&& = delete;

  /// This tensor, complex conjugated as it is read, for passing to einsum:
  /// `einsum("ij,jk->ik", 1.0, a.conj(), b, 0.0, d)`. The data is unchanged.
  /// The same as `katal::conj(a)`.
  ConstTensorRef conj() const& noexcept { return katal::conj(ref()); }
  ConstTensorRef conj() const&& = delete;

  /// The sum of every element, returned to the host. T is the element type,
  /// so `a.sum<double>()` on an F64 tensor; another T is `DtypeMismatch`.
  /// Like the other whole-tensor reductions, this runs on `stream` and blocks
  /// until the result is back, using the default context. To keep it on the
  /// device, reduce only some labels or use another context, call `reduce`
  /// or the one-operand einsum.
  template <class T>
  T sum(StreamRef stream = {}) const {
    return reduce_all<T>(BinaryOp::Add, stream);
  }

  /// The largest element, returned to the host. Real types only.
  template <std::floating_point T>
  T max(StreamRef stream = {}) const {
    return reduce_all<T>(BinaryOp::Max, stream);
  }

  /// The smallest element, returned to the host. Real types only.
  template <std::floating_point T>
  T min(StreamRef stream = {}) const {
    return reduce_all<T>(BinaryOp::Min, stream);
  }

 private:
  template <class T>
  T reduce_all(BinaryOp op, StreamRef stream) const {
    if (dtype_of<T> != type_)
      throw Error(Errc::DtypeMismatch, {},
                  "the result type must be the tensor's element type");
    T result{};
    if (extent_.empty()) {
      // One element: it is its own sum, maximum and minimum.
      detail::copy_to_host(&result, data(), sizeof(T), stream);
      return result;
    }
    const util::ScopedWorkspace<CudaAsyncAllocator> out({}, sizeof(T), stream);
    reduce(detail::full_reduction_equation(extent_.size()), op, 1.0, ref(), 0.0,
           make_ref(out.get().ptr, type_, {}, Layout::ColumnMajor), {}, stream);
    detail::copy_to_host(&result, out.get().ptr, sizeof(T), stream);
    return result;
  }

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
    throw Error(Errc::DtypeMismatch, {}, "unrecognized tensor data type");
  }

  /// Every extent must be positive: an empty tensor has nothing for a GPU
  /// operation to read, and rejecting it here is clearer than at planning.
  std::size_t volume() const {
    for (std::int64_t e : extent_)
      if (e <= 0)
        throw Error(Errc::ExtentMismatch, {},
                    "tensor extents must be positive");

    std::size_t result = 1;
    for (std::int64_t e : extent_) {
      const auto n = static_cast<std::size_t>(e);
      if (result > std::numeric_limits<std::size_t>::max() / n)
        throw std::bad_array_new_length();
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
