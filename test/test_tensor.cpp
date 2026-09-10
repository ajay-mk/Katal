#include <cstddef>
#include <memory>

#include <catch2/catch_test_macros.hpp>

#include <deviceein/tensor.hpp>

TEST_CASE("an owning tensor exposes a shared execution view", "[tensor]") {
  using Tensor = deviceein::Tensor<std::allocator<std::byte>>;

  const deviceein::Extents extent{2, 3, 4};
  Tensor tensor(deviceein::Dtype::F64, extent);

  REQUIRE(tensor.type() == deviceein::Dtype::F64);
  REQUIRE(tensor.extents() == extent);
  REQUIRE(tensor.layout() == deviceein::Layout::ColumnMajor);
  REQUIRE(tensor.size() == 24);
  REQUIRE(tensor.bytes() == 24 * sizeof(double));
  REQUIRE(tensor.data() != nullptr);

  const Tensor copy = tensor;
  REQUIRE(copy.data() == tensor.data());

  const deviceein::TensorRef ref = tensor.ref();
  REQUIRE(ref.data == tensor.data());
  REQUIRE(ref.type == tensor.type());
  REQUIRE(ref.extent.size() == extent.size());
  for (std::size_t i = 0; i < extent.size(); ++i)
    REQUIRE(ref.extent[i] == extent[i]);
  REQUIRE(ref.stride.empty());
  REQUIRE(ref.layout == tensor.layout());
}
