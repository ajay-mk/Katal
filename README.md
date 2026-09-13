# Katal

A C++20 einsum front-end for GPU tensor libraries, currently NVIDIA
cuTENSOR.

**Work in progress.** Equation parsing, mode classification, and the cuTENSOR
backend for contraction, permutation, reduction, and copy are implemented and
tested, with plan caching and a pluggable workspace allocator. The API is not
stable yet.

## Quick start

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
- [Tensor ownership and allocators](docs/tensors.md)
- [Examples](docs/examples.md)

## License

BSD-3-Clause.

Independent project. Not affiliated with or endorsed by NVIDIA.
