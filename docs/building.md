# Building and testing

## Dependencies

Needs a C++20 compiler and CMake 3.24 or newer. Catch2 v3 is used for the
tests; an installed copy is preferred and one is fetched if none is found.
GPU execution also needs the CUDA Toolkit and cuTENSOR. CUDA 12.6 is the
toolkit used by the CUDA compile-check CI job. Without those dependencies,
CMake builds only the host-side layers and omits the GPU tests and examples;
a passing host-only test run does not validate GPU execution.

## Build

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="<cuda-toolkit>" -DKATAL_CUTENSOR_ROOT="<cutensor>"
cmake --build build
ctest --test-dir build --output-on-failure
```

Omit the two dependency hints if CUDA and cuTENSOR are in standard search
paths, or to build the host-only layers on a machine without them. A downstream project uses
`find_package(Katal REQUIRED)` and links `Katal::katal` after installation.

| Option | Default | Effect |
| --- | --- | --- |
| `KATAL_BUILD_TESTS` | `ON` | build the test executables |
| `KATAL_MAX_RANK` | `12` | largest describable tensor rank |

With a GPU backend configured, start with the
[pooled and managed Tensor example](examples.md#pooled-and-managed-tensor-storage).

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

The documentation uses MkDocs Material with a Doxygen API reference. With
`mkdocs-material`, Doxygen, and Graphviz installed, run the same checks as CI:

```sh
doxygen Doxyfile
mkdocs build --strict
```

The generated site is written to `docs-build/site/`.
