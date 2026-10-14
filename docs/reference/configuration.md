# Configuration

## CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `KATAL_BUILD_TESTS` | `ON` | build the test executables |
| `KATAL_BUILD_BENCHMARKS` | `OFF` | build the GPU timing drivers in `bench/` |
| `KATAL_ENABLE_LOGGING` | `OFF` | compile cache and planning [diagnostics](diagnostics.md) into the GPU backend |
| `KATAL_MAX_RANK` | `12` | largest describable tensor rank |
| `KATAL_WARNINGS_AS_ERRORS` | `OFF` | treat compiler warnings in Katal's own targets as errors; CI turns it on |
| `KATAL_CUTENSOR_ROOT` | empty | cuTENSOR install root; `$CUTENSOR_ROOT` and an NVHPC tree under `$NVHPC` are also searched |

Katal's own targets always build with `-Wall -Wextra -Wpedantic` (`/W4` on
MSVC). These flags are private and never reach a consumer of `Katal::katal`.

## Context configuration

`katal::Config` is read once, when a context is built, whether by
`Context(config)`, `initialize(config)` or `Runtime(config)`:

| Field | Default | Meaning |
| --- | --- | --- |
| `plan_cache_entries` | `128` | plans Katal keeps by equation and shape; `0` disables the cache, and a full cache evicts the least recently used plan |
| `plan_cache_capacity` | `256` | entries in cuTENSOR's own kernel-selection cache |
| `plan_options` | `{}` | the `PlanOptions` that `einsum`, the other one-shot calls and `cached_plan` use unless a call passes its own |

## Plan options

`katal::PlanOptions` says how to plan, as opposed to what to compute:

| Field | Default | Meaning |
| --- | --- | --- |
| `compute` | unset | accumulation type; unset follows the output, with F16 and BF16 accumulating in F32 |
| `precision` | `Native` | a tensor-core mode (`Tf32`, `Tf32x3`, `Bf16x9`, `F16x4`, `Int8x8`) for contractions only; see [data types](tensors.md#data-types) |
| `workspace_limit` | `0` | cap on scratch bytes, including inputs summed ahead of a contraction; `0` takes what `workspace` suggests |
| `workspace` | `Default` | cuTENSOR's workspace preference: `Min`, `Default` or `Max` |
| `algo` | `Default` | cuTENSOR's algorithm choice: `Default`, `DefaultPatient`, `Gett`, `Tgett` or `Ttgt` |
| `kernel_rank` | unset | pick the n-th best kernel by cuTENSOR's model, `0` being its first choice |
| `jit` | `false` | let cuTENSOR compile a kernel for this exact problem; costs seconds at plan time, Ampere and newer |

A call that takes `std::optional<PlanOptions>` uses the context's
`plan_options` when it is left empty. `make_plan` takes a plain
`PlanOptions`, defaulting to `PlanOptions{}`, and does not inherit the
context's policy.
