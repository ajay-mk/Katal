# Katal

A C++20 einsum front-end for GPU tensor libraries, currently NVIDIA
cuTENSOR.

**Work in progress.** Equation parsing, mode classification, and the cuTENSOR
backend for contraction, permutation, reduction, and copy are implemented and
tested, with plan caching and a pluggable workspace allocator. See
`examples/` for CCSD-shaped calls. Documentation will follow once the API
settles.

## Building

Needs a C++20 compiler and CMake 3.24 or newer. Catch2 v3 is used for the
tests; an installed copy is preferred and one is fetched if none is found.

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build --output-on-failure
```

| Option | Default | Effect |
| --- | --- | --- |
| `KATAL_BUILD_TESTS` | `ON` | build the test executables |
| `KATAL_BUILD_BENCHMARKS` | `OFF` | build the benchmark harness |
| `KATAL_ENABLE_JIT` | `OFF` | allow cuTENSOR JIT compilation |
| `KATAL_MAX_RANK` | `12` | largest describable tensor rank |

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
