# cuEin

A C++20 einsum front-end for NVIDIA cuTENSOR.

**Work in progress.** The layers that need no CUDA — equation parsing, mode
classification, and tensor description — are implemented and tested. The
cuTENSOR backend is not written yet, so there is no working `einsum` call.
Documentation will follow once the API settles.

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
| `CUEIN_BUILD_TESTS` | `ON` | build the test executables |
| `CUEIN_BUILD_BENCHMARKS` | `OFF` | build the benchmark harness |
| `CUEIN_ENABLE_JIT` | `OFF` | allow cuTENSOR JIT compilation |
| `CUEIN_MAX_RANK` | `12` | largest describable tensor rank |

## Development

```sh
pre-commit install
```

That wires up formatting, include grouping, and a clang-tidy naming check
before each commit, plus a Conventional Commits check on the message. The
clang-tidy hook reads `build/compile_commands.json`, so configure once before
committing.

## License

BSD-3-Clause.

Independent project. Not affiliated with or endorsed by NVIDIA.
