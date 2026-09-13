# Examples

## Pooled and managed Tensor storage

Start with `examples/tensor_storage.cpp`, built as
`build/examples/tensor_storage` when CUDA and cuTENSOR are available. It
initializes column-major matrices, computes `ab,bc->ac`, waits for the GPU,
and verifies every output value.

The same function uses `katal::Tensor<Allocator>` with either
`katal::PooledDeviceAllocator<std::byte>` or
`katal::ManagedAllocator<std::byte>`. Both paths use explicit transfers so
that only allocation changes. Managed memory is not pooled and does not
remove synchronization requirements. See [ownership and allocator lifetimes](tensors.md)
for the storage contract. The examples below instead demonstrate
passing caller-owned buffers through `TensorRef`.

## Doubles contribution to the correlation energy

With singles set to zero, the closed-shell correlation energy uses two
contractions over every index, direct and exchange, reducing to a single
number. This example does not include the singles contributions to CCSD:

```
E = 2 sum_ijab <ij|ab> t[i,j,a,b] - sum_ijab <ij|ba> t[i,j,a,b]
```

```cpp
#include <cuda_runtime.h>

#include <katal/plan.hpp>

using katal::make_ref;

// i, j index occupied orbitals; a, b virtual ones.
double correlation_energy(double* t2, double* oovv, double* energy,
                           std::int64_t o, std::int64_t v) {
  const std::int64_t amp_extent[] = {o, o, v, v};
  auto t = make_ref(t2, katal::Dtype::F64, amp_extent);
  auto oovv_ref = make_ref(oovv, katal::Dtype::F64, amp_extent);
  auto e = make_ref(energy, katal::Dtype::F64, {});  // rank zero: one number

  katal::einsum("ijab,ijab->", 1.0, t, oovv_ref, 0.0, e);
  double direct = 0.0;
  if (const auto status = cudaMemcpy(&direct, energy, sizeof(direct),
                                      cudaMemcpyDeviceToHost);
      status != cudaSuccess)
    throw katal::Error(katal::Errc::CudaError, "", cudaGetErrorString(status));

  // The exchange term swaps the two virtual labels of the integrals.
  katal::einsum("ijab,ijba->", 1.0, t, oovv_ref, 0.0, e);
  double exchange = 0.0;
  if (const auto status = cudaMemcpy(&exchange, energy, sizeof(exchange),
                                      cudaMemcpyDeviceToHost);
      status != cudaSuccess)
    throw katal::Error(katal::Errc::CudaError, "", cudaGetErrorString(status));

  return 2 * direct - exchange;
}
```

## Permuting an integral block

CC intermediates are often a stored integral block reordered into the index
order the intermediate needs. The one-operand form of `einsum` covers this —
here, `(ki|lj)` becomes CCD's `Woooo`:

```cpp
void build_woooo(double* oooo, double* woooo, std::int64_t o) {
  const std::int64_t extent[] = {o, o, o, o};
  katal::einsum("kilj->klij", 1.0, make_ref(oooo, katal::Dtype::F64, extent),
                0.0, make_ref(woooo, katal::Dtype::F64, extent));
}
```

## More

`examples/` also contains a CCD contraction workload and the doubles-only
correlation-energy example above. These are execution examples, not validated
electronic-structure solvers. The handwritten CCSD residual has been removed;
a SeQuant-generated replacement is future work.
