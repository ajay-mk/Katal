# Build, test and install

## Check the dependencies

| Dependency | Needed for | Notes |
| --- | --- | --- |
| C++20 compiler, CMake 3.24+, Ninja | everything | any CMake generator works; the commands below use Ninja |
| CUDA Toolkit | the GPU backend | CI compile-checks against CUDA 12.6 |
| cuTENSOR | the GPU backend | 2.0 or newer; CI uses 2.8.1 |
| [Catch2](https://github.com/catchorg/Catch2) v3 | tests | fetched if no installed copy is found |
| [enumstr](https://github.com/ajay-mk/enumstr) | GPU backend and tests | header-only; fetched if no installed copy is found |

If CMake cannot find cuTENSOR, it prints `Katal: cuTENSOR not found, building
the header-only layers` and builds only equation parsing and classification,
with no GPU backend, GPU tests or examples. That build is not useful on its
own yet, so treat the message as a sign that the paths below need setting.

## Build

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="<cuda-toolkit>" -DKATAL_CUTENSOR_ROOT="<cutensor>"
cmake --build build
```

Leave out the two roots when CUDA and cuTENSOR are on the standard search
paths. cuTENSOR is also looked for under `$CUTENSOR_ROOT` and in an NVHPC
tree under `$NVHPC`.

Every option is listed under [CMake options](../reference/configuration.md#cmake-options).

## Run the tests

```sh
ctest --test-dir build --output-on-failure
```

On a CUDA GPU this runs the complete suite. On a build machine with CUDA but
no GPU, add `--label-exclude gpu`; that checks compiling and linking, not
device execution. It still checks numbers: `test_cpu_backend` runs einsum,
reductions, elementwise operations and permutations through a reference
backend of plain host loops in place of cuTENSOR. On a GPU,
`test_gpu_cpu_backend` compares cuTENSOR's reductions, elementwise
operations, permutations and unary operators with that backend's. It lives
in `test/` and is not part of the installed library.

## Install and use from another project

```sh
cmake --install build --prefix "<prefix>"
```

In the downstream project's `CMakeLists.txt`:

```cmake
find_package(Katal CONFIG REQUIRED)
target_link_libraries(app PRIVATE Katal::katal)
```

and configure it with `-DCMAKE_PREFIX_PATH="<prefix>"`. The installed
package looks for CUDA and cuTENSOR again, so pass the same
`-DCUDAToolkit_ROOT` and `-DKATAL_CUTENSOR_ROOT` (or set `$CUTENSOR_ROOT`)
if Katal's own build needed them. Katal's public headers do not include CUDA
headers, so a target that only calls Katal needs no CUDA include path. A
target that also calls the CUDA runtime itself, as the
[tutorial](../tutorial/first-contraction.md) does, links `CUDA::cudart` from
`find_package(CUDAToolkit)` too.

To check an install end to end, run `bash tools/test-install.sh`. It installs
into a temporary prefix, builds a downstream consumer with a plain host
compiler, and fails if a CUDA header leaks through the public API. Set
`CUDAToolkit_ROOT` and `CUTENSOR_ROOT` in the environment when CUDA or
cuTENSOR is outside the standard search paths.

## Run the examples

The programs in `examples/` are built into `build/examples/` and generate
their own input:

| Program | Shows |
| --- | --- |
| `tensor_storage` | a checked matrix product with pooled and managed `Tensor` storage |
| `ccsd_energy` | a doubles-only correlation energy, in F64 and C64 |
| `ccd_residual [o] [v]` | a closed-shell CCD doubles residual, with optional occupied and virtual sizes |

`tensor_storage` checks its result. The other two run on random tensors and
show how the equations are written, not a working solver.
