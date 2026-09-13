# Katal

A C++20 einsum front-end for GPU tensor libraries, currently NVIDIA
cuTENSOR.

**Work in progress.** Equation parsing, mode classification, and the cuTENSOR
backend for contraction, permutation, reduction, and copy are implemented and
tested, with plan caching and a pluggable workspace allocator. See
`examples/` for CCSD-shaped calls. The API is not stable yet.

## Building

Needs a C++20 compiler and CMake 3.24 or newer. Catch2 v3 is used for the
tests; an installed copy is preferred and one is fetched if none is found.
GPU execution also needs the CUDA Toolkit and cuTENSOR. CUDA 12.6 is the
toolkit used by the CUDA compile-check CI job. Without those dependencies,
CMake builds only the host-side layers and omits the GPU tests and examples;
a passing host-only test run does not validate GPU execution.

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="<cuda-toolkit>" -DKATAL_CUTENSOR_ROOT="<cutensor>"
cmake --build build
ctest --test-dir build --output-on-failure
```

Omit the two dependency hints if CUDA and cuTENSOR are in standard search
paths, or to build the host-only layers on a machine without them. HDF5 is
optional and separate from the core library. A downstream project uses
`find_package(Katal REQUIRED)` and links `Katal::katal` after installation.

| Option | Default | Effect |
| --- | --- | --- |
| `KATAL_BUILD_TESTS` | `ON` | build the test executables |
| `KATAL_BUILD_BENCHMARKS` | `OFF` | build the benchmark harness |
| `KATAL_ENABLE_JIT` | `OFF` | allow cuTENSOR JIT compilation |
| `KATAL_ENABLE_HDF5` | `ON` | build the optional HDF5 reader when found |
| `KATAL_MAX_RANK` | `12` | largest describable tensor rank |

## First example

With a GPU backend configured, run `build/examples/tensor_storage`. It
allocates two matrices, initializes them, evaluates `ab,bc->ac`, waits for
the default stream, and checks the result. The same function runs with
`PooledDeviceAllocator<std::byte>` and `ManagedAllocator<std::byte>`; only
the allocator type changes. It is a correctness example, not a benchmark.

## Design and ownership

- `Tensor<Allocator>` owns dense, column-major storage. Copies share the
  allocation; they do not copy its values. The allocator follows standard
  allocator semantics and must allocate `std::byte`.
- `TensorRef` borrows data and shape/stride metadata. Keep the data alive
  until GPU work completes, and the metadata alive while using the view.
  In particular, `tensor.ref()` borrows that Tensor object's extents, even
  when another Tensor still owns the shared allocation.
- `Context` owns the cuTENSOR handle and plan cache. `Plan` holds a selected
  operation ready for repeated execution. `einsum` uses cached plans;
  an explicit Plan lets callers hoist setup out of a loop.
- Tensor allocation and execution workspace are separate policies. Changing
  a Tensor to managed storage does not change the temporary device workspace
  that `einsum` obtains through its workspace allocator.

Use `katal::Runtime` to scope the default context, or manage an explicit
Context. This walkthrough targets one CUDA device. A stream is borrowed,
not owned; keep it alive through allocation use and release.

## Managed tensor storage

`katal::Tensor<>` uses pooled device memory by default. To allocate CUDA
managed memory, include `<katal/tensor.hpp>` and use
`katal::Tensor<katal::ManagedAllocator<std::byte>>`. Tensor copies share
the allocation; `ref()` borrows it for execution.

The allocator accepts a `StreamRef` (default: the default CUDA stream). All GPU
uses of that allocation must use that stream, which must outlive the storage.
Wait for GPU work before accessing the data on the CPU, and finish CPU writes
before submitting GPU work. Last-owner release waits for the stream before
freeing managed memory. A CUDA cleanup failure terminates rather than throwing
from destruction or freeing storage that may still be in use.

Managed allocations use `cudaMallocManaged` / `cudaFree`; they are not pooled.
Pooled device allocations use `cudaMallocAsync` / `cudaFreeAsync` on their
allocator's stream, without a host-side wait during release. Allocation
exhaustion throws `std::bad_alloc`; other CUDA allocation errors throw
`katal::Error`. Both allocators terminate on a failed cleanup call.

## Validation

Run `ctest --test-dir build --output-on-failure` on a CUDA GPU for the complete
configured suite. On a CUDA-equipped build machine without a GPU, use
`--label-exclude gpu`; that is a compile/link check, not device validation.
`bash tools/test-install.sh` separately installs into a temporary prefix,
builds a downstream consumer, and checks that CUDA headers do not leak through
the public API. Set `CUDAToolkit_ROOT` and `CUTENSOR_ROOT` in the environment
when validating GPU dependencies outside standard search paths.

## Development

```sh
pre-commit install
```

That wires up formatting, include grouping, and a clang-tidy naming check
before each commit. The clang-tidy hook reads `build/compile_commands.json`,
so configure once before committing.

## License

BSD-3-Clause.

Independent project. Not affiliated with or endorsed by NVIDIA.
