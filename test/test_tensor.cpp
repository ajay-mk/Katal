#include <cstddef>
#include <cstdint>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include <katal/tensor.hpp>

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
