<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/katal-logo-dark.svg">
    <img alt="Katal" src="docs/assets/katal-logo-light.svg" width="360">
  </picture>
</p>

A C++20 einsum frontend for GPU tensor libraries.

Libraries like cuTENSOR run tensor contractions fast, but every one of them
needs its own descriptors, plan and workspace written by hand, and a real
application can have hundreds. With Katal you write the equation instead:

```cpp
katal::einsum("ijab,abcd->ijcd", 0.5, t2, v, 1.0, r);
```

Katal parses the equation, builds the matching plan for the backend and
caches it by equation and shape, so the code reads like the math it computes
and a hot loop pays for setup once. NVIDIA cuTENSOR is the backend today.

> [!NOTE]
> Katal is a work in progress and its API is not stable yet.

## What it covers

- **Contractions**, including diagonal and trace inputs (`"iij,jk->ik"`) and
  labels summed out of one input first.
- **Elementwise operations** with broadcasting: `katal::add`, `multiply`,
  `max` and `min`, or `katal::elementwise` with an accumulating term.
- **Reductions** by sum, product, max or min, and **permutations**, including
  type conversion.
- **Operators on inputs** such as `katal::conj(a)` or
  `katal::apply(katal::UnaryOp::Abs, a)`, applied as the input is read.
- **Plans you can hold**: `make_plan` and `execute` hoist setup out of a loop,
  and execution is const and allocation-free.
- **Your memory**: bring a workspace allocator, or let Katal use
  `cudaMallocAsync`. `Tensor` owns pooled device or managed storage, and
  `TensorRef` borrows yours.

## Quick start

```cpp
#include <cuda_runtime.h>
#include <katal/plan.hpp>
#include <katal/tensor.hpp>

int main() {
  const katal::Runtime runtime;  // owns the backend handle for this scope

  // Device tensors, column-major. Fill t2 and v with cudaMemcpy.
  katal::Tensor<> t2(katal::Dtype::F64, {4, 4, 20, 20});  // i j a b
  katal::Tensor<> v(katal::Dtype::F64, {20, 20, 20, 20});  // a b c d
  katal::Tensor<> r(katal::Dtype::F64, {4, 4, 20, 20});   // i j c d

  // r = 0.5 * t2 . v, planned once and cached by shape.
  katal::einsum("ijab,abcd->ijcd", 0.5, t2, v, 0.0, r);

  // einsum is asynchronous: wait before reading r or leaving the Runtime.
  cudaDeviceSynchronize();
}
```

## Building

Needs a C++20 compiler, CMake 3.24 or newer, Ninja, the CUDA Toolkit and
cuTENSOR.

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="<cuda-toolkit>" -DKATAL_CUTENSOR_ROOT="<cutensor>"
cmake --build build
ctest --test-dir build --output-on-failure
```

Leave out the two roots when CUDA and cuTENSOR are on the standard search
paths. After installing, a downstream project uses
`find_package(Katal CONFIG REQUIRED)` and links `Katal::katal`.

## Documentation

Start with the [tutorial](docs/tutorial/first-contraction.md), or see the
[documentation overview](docs/index.md) for guides, reference and
background.

## License

BSD-3-Clause.
