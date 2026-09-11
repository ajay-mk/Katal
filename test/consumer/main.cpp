// Compiled by a plain host compiler, never by nvcc. Its job is to fail loudly
// if Katal's public headers drag CUDA into a consumer translation unit.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

#ifdef CUTENSOR_MAJOR
#error "cuTENSOR leaked into a consumer translation unit"
#endif
#ifdef CUDART_VERSION
#error "the CUDA runtime leaked into a consumer translation unit"
#endif

int main() {
  const katal::Equation equation = katal::Equation::parse("ijab,abcd->ijcd");
  const katal::Classification classification = equation.classify();
  if (classification.kind != katal::Lowering::Contract) return 1;
  if (std::string(classification.contracted) != "ab") return 2;

  const std::array<std::int64_t, 4> extent{2, 3, 4, 5};
  const katal::TensorRef tensor =
      katal::make_ref(nullptr, katal::Dtype::F64, extent);
  if (katal::mode_order(equation.operand(0), tensor).extent !=
      katal::Extents{5, 4, 3, 2})
    return 3;

  using Tensor = katal::Tensor<std::allocator<std::byte>>;
  const Tensor owned(katal::Dtype::F64, katal::Extents{2, 3});
  if (owned.bytes() != 6 * sizeof(double)) return 4;
  if (owned.ref().data != owned.data()) return 5;

#ifdef CONSUMER_HAS_BACKEND
  // A real call into the compiled library, so the link is exercised. It needs
  // no GPU: nothing has been initialised yet.
  if (katal::initialized()) return 6;
  std::printf("consumer: linked the cuTENSOR backend\n");
#else
  std::printf("consumer: header-only build, no cuTENSOR\n");
#endif

  std::printf("consumer: ok\n");
  return 0;
}
