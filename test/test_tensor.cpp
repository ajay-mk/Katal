#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include <katal/tensor.hpp>

TEST_CASE("zero and negative tensor extents are rejected", "[tensor]") {
  using Tensor = katal::Tensor<std::allocator<std::byte>>;
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {3, 0, 4}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {0, -1}), std::invalid_argument);
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {-1, 0}), std::invalid_argument);
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {2, 0, -1}),
                    std::invalid_argument);
}

TEST_CASE("a rank-zero tensor holds one element", "[tensor]") {
  using Tensor = katal::Tensor<std::allocator<std::byte>>;
  const Tensor scalar(katal::Dtype::F64, {});
  REQUIRE(scalar.size() == 1);
  REQUIRE(scalar.bytes() == sizeof(double));
  REQUIRE(scalar.data() != nullptr);
}

TEST_CASE("tensor allocation sizes are checked before allocating", "[tensor]") {
  using Tensor = katal::Tensor<std::allocator<std::byte>>;
  constexpr auto large = std::numeric_limits<std::int64_t>::max();
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {large, 3}), std::length_error);
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {large}), std::length_error);
  REQUIRE_THROWS_AS(Tensor(static_cast<katal::Dtype>(99), {2}),
                    std::invalid_argument);
}

namespace {
struct AllocationCounts {
  int allocations = 0;
  int deallocations = 0;
};

template <class T>
struct CountingAllocator {
  using value_type = T;
  AllocationCounts* counts;

  T* allocate(std::size_t n) {
    auto* data = std::allocator<T>{}.allocate(n);
    ++counts->allocations;
    return data;
  }

  void deallocate(T* data, std::size_t n) noexcept {
    ++counts->deallocations;
    std::allocator<T>{}.deallocate(data, n);
  }
};
}  // namespace

TEST_CASE("tensor copies and moves retain storage until the last owner",
          "[tensor]") {
  using Tensor = katal::Tensor<CountingAllocator<std::byte>>;
  AllocationCounts counts;
  std::optional<Tensor> survivor;
  {
    Tensor tensor(katal::Dtype::F64, {2, 3}, {&counts});
    Tensor copy = tensor;
    Tensor moved = std::move(copy);
    REQUIRE(moved.data() == tensor.data());
    REQUIRE(moved.size() == 6);
    static_cast<std::byte*>(moved.data())[0] = std::byte{42};
    REQUIRE(static_cast<const std::byte*>(tensor.data())[0] == std::byte{42});
    survivor.emplace(std::move(moved));
    REQUIRE(counts.allocations == 1);
    REQUIRE(counts.deallocations == 0);
  }
  REQUIRE(counts.deallocations == 0);
  REQUIRE(static_cast<const std::byte*>(survivor->data())[0] == std::byte{42});
  survivor.reset();
  REQUIRE(counts.deallocations == 1);
}

TEST_CASE("managed allocators preserve their stream when rebound", "[tensor]") {
  using Bytes = katal::ManagedAllocator<std::byte>;
  using Doubles = std::allocator_traits<Bytes>::rebind_alloc<double>;

  STATIC_REQUIRE(std::is_same_v<Doubles, katal::ManagedAllocator<double>>);
  const Bytes bytes;
  const Doubles doubles(bytes);
  REQUIRE(bytes == doubles);
  REQUIRE(doubles.stream().get() == bytes.stream().get());
}

TEST_CASE("an owning tensor exposes a shared execution view", "[tensor]") {
  using Tensor = katal::Tensor<std::allocator<std::byte>>;

  const katal::Extents extent{2, 3, 4};
  Tensor tensor(katal::Dtype::F64, extent);

  REQUIRE(tensor.type() == katal::Dtype::F64);
  REQUIRE(tensor.extents() == extent);
  REQUIRE(tensor.layout() == katal::Layout::ColumnMajor);
  REQUIRE(tensor.size() == 24);
  REQUIRE(tensor.bytes() == 24 * sizeof(double));
  REQUIRE(tensor.data() != nullptr);

  const Tensor copy = tensor;
  REQUIRE(copy.data() == tensor.data());

  const katal::TensorRef ref = tensor.ref();
  REQUIRE(ref.data == tensor.data());
  REQUIRE(ref.type == tensor.type());
  REQUIRE(ref.extent.size() == extent.size());
  for (std::size_t i = 0; i < extent.size(); ++i)
    REQUIRE(ref.extent[i] == extent[i]);
  REQUIRE(ref.stride.empty());
  REQUIRE(ref.layout == tensor.layout());
  REQUIRE(ref.alignment >= alignof(double));
  REQUIRE(reinterpret_cast<std::uintptr_t>(ref.data) % ref.alignment == 0);
}
