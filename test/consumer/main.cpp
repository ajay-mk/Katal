// Compiled by a plain host compiler, never by nvcc. Its job is to fail loudly
// if cuEin's public headers drag CUDA into a consumer translation unit.

#include <array>
#include <cstdint>
#include <cstdio>

#include <cuein/plan.hpp>

#ifdef CUTENSOR_MAJOR
#error "cuTENSOR leaked into a consumer translation unit"
#endif
#ifdef CUDART_VERSION
#error "the CUDA runtime leaked into a consumer translation unit"
#endif

int main() {
  const cuein::Equation equation = cuein::Equation::parse("ijab,abcd->ijcd");
  const cuein::Classification classification = equation.classify();
  if (classification.kind != cuein::Lowering::Contract) return 1;
  if (std::string(classification.contracted) != "ab") return 2;

  const std::array<std::int64_t, 4> extent{2, 3, 4, 5};
  const cuein::TensorRef tensor =
      cuein::make_ref(nullptr, cuein::Dtype::F64, extent);
  if (cuein::mode_order(equation.operand(0), tensor).extent !=
      cuein::Extents{5, 4, 3, 2})
    return 3;

#ifdef CONSUMER_HAS_BACKEND
  // A real call into the compiled library, so the link is exercised. It needs
  // no GPU: nothing has been initialised yet.
  if (cuein::initialized()) return 4;
  std::printf("consumer: linked the cuTENSOR backend\n");
#else
  std::printf("consumer: header-only build, no cuTENSOR\n");
#endif

  std::printf("consumer: ok\n");
  return 0;
}
