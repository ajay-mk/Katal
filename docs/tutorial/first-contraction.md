# Your first contraction

In this tutorial you will install Katal, build a small program against it,
and use it to multiply two matrices on the GPU. Then you will accumulate into
the result and change the equation, to see how one string turns into
another computation.

You need a machine with an NVIDIA GPU, the CUDA Toolkit, cuTENSOR, a C++20
compiler, CMake 3.24 or newer and Ninja.

## Install Katal

From a checkout of Katal, configure, build and install into a prefix of your
choice. This tutorial uses `$HOME/katal`:

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DKATAL_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$HOME/katal"
cmake --build build
cmake --install build
```

The configure step prints `Katal: cuTENSOR not found` if it could not find
cuTENSOR. If you see that, follow [build and install](../how-to/build.md) to
point CMake at it, then come back.

## Set up a project

Make a new directory with two files in it. The first is `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(first_contraction LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(Katal CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(first main.cpp)
target_link_libraries(first PRIVATE Katal::katal CUDA::cudart)
```

The second is `main.cpp`. It makes three matrices on the device, copies
values into two of them, and multiplies them with one call:

```cpp
#include <array>
#include <cstdio>

#include <cuda_runtime.h>

#include <katal/plan.hpp>
#include <katal/tensor.hpp>

int main() {
  const katal::Runtime runtime;

  katal::Tensor<> a(katal::Dtype::F64, {2, 3});
  katal::Tensor<> b(katal::Dtype::F64, {3, 2});
  katal::Tensor<> d(katal::Dtype::F64, {2, 2});

  // Column-major: each column is contiguous.
  const std::array<double, 6> av{1, 4, 2, 5, 3, 6};
  const std::array<double, 6> bv{1, 3, 5, 2, 4, 6};
  cudaMemcpy(a.data(), av.data(), sizeof(av), cudaMemcpyHostToDevice);
  cudaMemcpy(b.data(), bv.data(), sizeof(bv), cudaMemcpyHostToDevice);

  katal::einsum("ab,bc->ac", 1.0, a, b, 0.0, d);

  std::array<double, 4> result{};
  cudaMemcpy(result.data(), d.data(), sizeof(result), cudaMemcpyDeviceToHost);
  std::printf("%g %g %g %g\n", result[0], result[1], result[2], result[3]);
}
```

## Build and run it

```sh
cmake -S . -B build -GNinja -DCMAKE_PREFIX_PATH="$HOME/katal"
cmake --build build
./build/first
```

If you had to point Katal's build at CUDA or cuTENSOR, pass the same
`-DCUDAToolkit_ROOT` and `-DKATAL_CUTENSOR_ROOT` here too: the installed
package looks for both again.

The program prints:

```
22 49 28 64
```

That is the product of the two matrices, read column by column:

```
[1 2 3]   [1 2]   [22 28]
[4 5 6] . [3 4] = [49 64]
          [5 6]
```

`einsum` returns before the GPU has finished. The copy back to the host
waited for it here, because `cudaMemcpy` on the default stream runs after
the work queued before it.

## Accumulate into the result

Change the `einsum` line so that it runs twice, the second time with a beta
of one:

```cpp
  katal::einsum("ab,bc->ac", 1.0, a, b, 0.0, d);
  katal::einsum("ab,bc->ac", 1.0, a, b, 1.0, d);
```

Rebuild and run. Every value has doubled:

```
44 98 56 128
```

The second call computed `d = 1.0 * a.b + 1.0 * d`. It also reused the plan
the first call built, because the equation and the shapes were the same.

## Change the computation

Now delete the second call and change the equation of the first, to
transpose the result as it is written:

```cpp
  katal::einsum("ab,bc->ca", 1.0, a, b, 0.0, d);
```

Rebuild and run:

```
22 28 49 64
```

The numbers are the same as the first run, laid out the other way round.
The same call, with a different string, produced a different contraction
and a new plan.

## What you have done

You built a program against an installed Katal, ran a contraction on the
GPU, accumulated into its result, and changed what it computes by editing
the equation. From here:

- [Equations](../reference/equations.md) lists everything the notation can
  say, including diagonals and labels summed out of one input.
- [Operations](../reference/operations.md) covers elementwise operations,
  reductions and permutations.
- [Hoist a plan out of a loop](../how-to/hoist-a-plan.md) shows how to stop
  paying for the cache lookup in a hot loop.
