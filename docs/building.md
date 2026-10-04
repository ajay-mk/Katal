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
| `KATAL_BUILD_BENCHMARKS` | `OFF` | build the GPU timing drivers in `bench/` |
| `KATAL_ENABLE_LOGGING` | `OFF` | compile cache and planning diagnostics into the GPU backend |
| `KATAL_MAX_RANK` | `12` | largest describable tensor rank |
| `KATAL_WARNINGS_AS_ERRORS` | `OFF` | treat compiler warnings in Katal's own targets as errors; CI turns it on |
| `KATAL_CUTENSOR_ROOT` | empty | cuTENSOR install root; `$CUTENSOR_ROOT` and an NVHPC tree under `$NVHPC` are also searched |

Katal's own targets always build with `-Wall -Wextra -Wpedantic` (`/W4` on
MSVC). These flags are private and never reach a consumer of `Katal::katal`.

### Host-only installs

Without CUDA and cuTENSOR, Katal installs as headers only. The equation,
index and tensor-metadata layers work. `Context`, `Plan`, `einsum` and the
device allocators are still declared but have no definitions, so code using
them fails to link. Check for the backend before using it:

- in CMake, `Katal_HAS_CUTENSOR` is set after `find_package(Katal)`;
- in C++, `KATAL_HAVE_CUTENSOR` from `<katal/config.hpp>` is `1` or `0`.

With a GPU backend configured, start with the
[pooled and managed Tensor example](examples.md#pooled-and-managed-tensor-storage).

## Validation

### Optional diagnostics

Build the GPU backend with `-DKATAL_ENABLE_LOGGING=ON` to write Katal
diagnostics to stderr. No Logger object or runtime setup is needed.
The default `OFF` build removes logging calls, formatting, and timers at
compile time. This does not change cuTENSOR's independent logging settings.

Messages start with `[katal]`. Cache events are `cache.hit`, `cache.miss`,
`cache.bypass` (caching disabled), `cache.insert`, and `cache.clear` (the full
cache is cleared). The `key` field is a diagnostic hash for correlating events
within a run, not a persistent or collision-free operation ID. Hit and
insertion records include the equation; tensor values are never printed.

`plan.created` records a successful Plan construction: equation, tensor shapes,
data types, strides, alignment, planning options, required workspace bytes,
and `host_plan_us`. This time covers construction in `make_plan`, not equation
parsing, cache lookup, logging output, or GPU execution. No GPU synchronization
is added. A retained Plan's execution does not perform a cache lookup and
therefore produces no cache event.

Planning options use the enum values from `options.hpp`: `algo` is
`0=Default`, `1=DefaultPatient`, `2=Gett`, `3=Tgett`, `4=Ttgt`;
`workspace_preference` is `0=Min`, `1=Default`, `2=Max`. `compute=output`
uses the output type and `kernel_rank=-1` leaves kernel rank unspecified.
Empty strides mean the tensor's dense layout. A workspace requirement of zero
needs no scratch allocation.

Diagnostics are best-effort: output failures do not replace normal operation
errors. Enabled logging adds host overhead, including output during cache
access; leave it off for performance measurements.

cuTENSOR has its own logging, enabled separately before launching a program:

```sh
CUTENSOR_LOG_LEVEL=4 CUTENSOR_LOG_FILE=cutensor.log ./your_program
```

Use level `5` for API tracing. Katal does not enable or intercept this logger,
and does not claim to report cuTENSOR's internal cache hit counts.
See NVIDIA's [logging guide](https://docs.nvidia.com/cuda/cutensor/latest/user_guide.html#logging).

### Build and test checks

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

## Timing benchmarks

With CUDA and cuTENSOR available, configure with `-DKATAL_BUILD_BENCHMARKS=ON`
to build `bench/grid`, `bench/overhead`, and `bench/shapes`.
`grid` compares two contraction layouts over occupied/virtual dimensions;
`overhead` compares hoisted execution with cached and uncached einsum;
`shapes` compares contraction shapes/layouts with cuBLAS DGEMM.

These drivers use synthetic zero-filled FP64 tensors and report minimum device
elapsed times across repeated trials. They are timing tools, not numerical
correctness tests or full SeQuant residual benchmarks. They require a CUDA GPU;
CUDA/cuBLAS failures exit unsuccessfully. Without CUDA and cuTENSOR, enabling
the option leaves these targets unbuilt and the host-only build available.
