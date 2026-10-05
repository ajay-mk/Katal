# Katal

A C++20 einsum front-end for GPU tensor libraries, currently NVIDIA
cuTENSOR.

**Work in progress.** Equation parsing, mode classification, and the cuTENSOR
backend for contraction, permutation, reduction, and copy are implemented and
tested, with plan caching and a pluggable workspace allocator. The API is not
stable yet.

## Quick start

```cpp
#include <cuda_runtime.h>
#include <katal/plan.hpp>
#include <katal/tensor.hpp>

int main() {
  const katal::Runtime runtime;  // owns the cuTENSOR handle for this scope

  // Device tensors, column-major. Fill them with cudaMemcpy.
  katal::Tensor<> t2(katal::Dtype::F64, {4, 4, 20, 20});  // i j a b
  katal::Tensor<> v(katal::Dtype::F64, {20, 20, 20, 20});  // a b c d
  katal::Tensor<> r(katal::Dtype::F64, {4, 4, 20, 20});   // i j c d

  // r = 0.5 * t2 . v + 1.0 * r, planned once and cached by shape.
  katal::einsum("ijab,abcd->ijcd", 0.5, t2, v, 1.0, r);

  // einsum is asynchronous: wait before reading r or leaving the Runtime.
  cudaDeviceSynchronize();
}
```

For a hot loop, build the `Plan` once with `katal::make_plan` and call
`execute` each iteration. See [equations](docs/equations.md) for the notation,
including how a result without `->` is ordered, and [errors](docs/errors.md)
for what each call throws.

## Building

Needs a C++20 compiler and CMake 3.24 or newer. GPU execution also requires
the CUDA Toolkit and cuTENSOR; without them, only the host-side layers build.

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Documentation

- [Overview](docs/index.md)
- [Building, testing, and development](docs/building.md)
- [Equations](docs/equations.md)
- [Errors](docs/errors.md)
- [Tensor ownership and allocators](docs/tensors.md)
- [Examples](docs/examples.md)

## License

BSD-3-Clause.
