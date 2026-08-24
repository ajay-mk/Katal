# Examples

## Matrix multiply

```cpp
#include <cuein/plan.hpp>

using cuein::make_ref;

void matmul(double* a, double* b, double* c, std::int64_t m, std::int64_t k,
            std::int64_t n) {
  const std::int64_t a_extent[] = {m, k};
  const std::int64_t b_extent[] = {k, n};
  const std::int64_t c_extent[] = {m, n};

  cuein::einsum("ik,kj->ij", 1.0, make_ref(a, cuein::Dtype::F64, a_extent),
                make_ref(b, cuein::Dtype::F64, b_extent), 0.0,
                make_ref(c, cuein::Dtype::F64, c_extent));
}
```

`a`, `b`, and `c` are device pointers; cuEin allocates no memory of its own.

## Transpose

The one-operand form covers a permutation, a reduction, or a copy:

```cpp
void transpose(double* a, double* a_t, std::int64_t m, std::int64_t n) {
  const std::int64_t extent[] = {m, n};
  const std::int64_t extent_t[] = {n, m};
  cuein::einsum("ij->ji", 1.0, make_ref(a, cuein::Dtype::F64, extent), 0.0,
                make_ref(a_t, cuein::Dtype::F64, extent_t));
}
```

## More

`examples/` in the repository has full worked programs — closed-shell CCD and
CCSD residuals and a CCSD correlation energy — built against real-sized
tensors.
