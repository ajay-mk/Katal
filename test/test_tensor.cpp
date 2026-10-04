#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#include "thrown.hpp"
#include <catch2/catch_test_macros.hpp>

#include <katal/tensor.hpp>

using katal::test::thrown_code;

TEST_CASE("zero and negative tensor extents are rejected", "[tensor]") {
  using Tensor = katal::Tensor<std::allocator<std::byte>>;
  using katal::Errc;
  for (const katal::Extents& e :
       {katal::Extents{3, 0, 4}, katal::Extents{0, -1}, katal::Extents{-1, 0},
        katal::Extents{2, 0, -1}})
    REQUIRE(thrown_code([&] { Tensor(katal::Dtype::F64, e); }) ==
            Errc::ExtentMismatch);
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
  // Too large to allocate is out of memory, reported as the allocators do.
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {large, 3}), std::bad_alloc);
  REQUIRE_THROWS_AS(Tensor(katal::Dtype::F64, {large}), std::bad_alloc);
  REQUIRE(thrown_code([] { Tensor(static_cast<katal::Dtype>(99), {2}); }) ==
          katal::Errc::DtypeMismatch);
}

TEST_CASE("extents beyond the maximum rank are a katal::Error", "[tensor]") {
  katal::Extents e;
  for (std::size_t i = 0; i < katal::max_rank; ++i) e.push_back(1);
  REQUIRE(thrown_code([&] { e.push_back(1); }) == katal::Errc::RankMismatch);
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

  // Copies share storage, so a const Tensor still reads the same data, but
  // only through a reference that cannot be written.
  STATIC_REQUIRE(std::is_same_v<decltype(copy.ref()), katal::ConstTensorRef>);
  STATIC_REQUIRE(std::is_same_v<decltype(tensor.ref()), katal::TensorRef>);
  REQUIRE(copy.ref().data == tensor.data());
}
