# Hoist a plan out of a loop

Every `einsum` call looks up its plan in the context's cache and allocates
any scratch it needs. In a loop that runs the same equation on the same
shapes, build the plan and its workspace once and call `execute`, which is
const and allocation-free.

The snippets below run on `stream`, a `cudaStream_t` you own; pass `{}` for
the default stream.

## Build the plan once

```cpp
const katal::Context& context = katal::default_context();
const katal::Plan plan = katal::make_plan(
    context, katal::Equation::parse("ijab,abcd->ijcd"), t2, v, r);
```

`make_plan` reads only shapes, types and alignment, never data. It uses the
`PlanOptions` you pass it, or the defaults; it does not inherit the context's
`Config::plan_options`.

## Allocate its workspace once

```cpp
const katal::util::ScopedWorkspace<katal::CudaAsyncAllocator> scratch(
    {}, plan.workspace_size(), stream);
```

`workspace_size()` is often zero, in which case nothing is allocated. The
workspace is released when `scratch` goes out of scope, including during
unwinding. To use your own pool instead, see
[use your own workspace allocator](memory.md#use-your-own-workspace-allocator).

## Execute in the loop

```cpp
for (int iter = 0; iter < n; ++iter)
  plan.execute(0.5, t2.data(), v.data(), 1.0, r.data(), r.data(),
               scratch.get(), stream);
```

`execute` takes raw device pointers shaped as the tensors the plan was built
from. Passing `r.data()` as both `c` and `d` accumulates into `r`.

## Other kinds of plan

| Operation | Build with | Run with |
| --- | --- | --- |
| reduction, permutation or copy (one operand) | `make_plan(context, eq, a, d)` | `execute(alpha, a, beta, c, d, workspace)` |
| reduction by `Multiply`, `Max` or `Min` | `make_plan(context, eq, op, a, d)` | the one-operand `execute` |
| elementwise | `make_plan(context, eq, op, a, b, d)` | `execute(alpha, a, beta, b, gamma, c, d, workspace)` |
| dedicated permutation | `make_permutation_plan(context, eq, a, d)` | the one-operand `execute`, with a beta of zero |

To keep a plan in the cache but still skip the lookup, take the
`shared_ptr<const Plan>` that `cached_plan` returns and hold on to it. The
plan borrows its context; see [ownership and lifetimes](../explanation/design.md#ownership-and-lifetimes).
