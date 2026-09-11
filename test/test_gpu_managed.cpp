#include <cstddef>
#include <memory>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

TEST_CASE("managed tensors share host and GPU storage", "[tensor][gpu]") {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
    SKIP("no CUDA device");

  cudaStream_t raw_stream = nullptr;
  REQUIRE(cudaStreamCreate(&raw_stream) == cudaSuccess);
  const auto destroy = [](cudaStream_t stream) { cudaStreamDestroy(stream); };
  const std::unique_ptr<CUstream_st, decltype(destroy)> stream(raw_stream,
                                                               destroy);
  using Allocator = katal::ManagedAllocator<std::byte>;
  using Tensor = katal::Tensor<Allocator>;
  const Allocator allocator(stream.get());
  const katal::ManagedAllocator<double> rebound(allocator);
  REQUIRE(allocator == rebound);
  REQUIRE_FALSE(allocator == Allocator{});

  Tensor output(katal::Dtype::F64, {3, 2}, allocator);
  struct Unwind {};
  try {
    Tensor input(katal::Dtype::F64, {2, 3}, allocator);
    auto* values = static_cast<double*>(input.data());
    for (std::size_t i = 0; i < input.size(); ++i)
      values[i] = static_cast<double>(i + 1);
    const Tensor shared = input;
    REQUIRE(shared.data() == input.data());

    katal::einsum("ij->ji", 1.0, input.ref(), 0.0, output.ref(), {},
                      stream.get());
    throw Unwind{};
  } catch (const Unwind&) {
  }

  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  const auto* values = static_cast<const double*>(output.data());
  const double expected[]{1, 3, 5, 2, 4, 6};
  for (std::size_t i = 0; i < output.size(); ++i)
    REQUIRE(values[i] == expected[i]);
}
