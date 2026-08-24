# DeviceEin

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
| `DEVICEEIN_BUILD_TESTS` | `ON` | build the test executables |
| `DEVICEEIN_BUILD_BENCHMARKS` | `OFF` | build the benchmark harness |
| `DEVICEEIN_ENABLE_JIT` | `OFF` | allow cuTENSOR JIT compilation |
| `DEVICEEIN_MAX_RANK` | `12` | largest describable tensor rank |

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
