# Design

## From an equation to a kernel

A call such as `einsum("ijab,abcd->ijcd", ...)` is parsed and classified
(contraction, reduction, permutation, copy or elementwise), then lowered to
a backend-neutral operation: modes, extents, strides and types. The backend
turns that into a kernel and reports the workspace it needs. With cuTENSOR,
that last step is where descriptors are built and the kernel search runs,
and it is the expensive part. The result is a `Plan`. Running it only
queues its kernels with this call's pointers and scales.

## Two caches

- **Katal's plan cache**, in each `Context`, is keyed on the exact equation
  text, the operation, the planning options, and every tensor's type,
  extents, strides, alignment and unary operator. Data pointers are not in
  the key, so new data with the same layout reuses the plan, and a hit skips
  every step above. `cached_plan` and the one-shot calls built on it, such
  as `einsum`, use it; `make_plan` does not.
- **cuTENSOR's plan cache** remembers kernel selections. Its keys are the
  descriptors themselves, so it saves the kernel search but not building
  the description.

A `Plan` you hold skips both, which is why hot loops should
[hoist their plans](../how-to/hoist-a-plan.md).

## The backend seam

Parsing, classification and lowering know nothing about cuTENSOR. A second
backend, for another vendor's library or the CPU, implements one internal
interface and leaves every public call unchanged. The seam is not public yet,
and cuTENSOR is the only backend. Some limits are cuTENSOR's passed through:
which unary operators a contraction accepts, which type combinations run,
and what a zero elementwise scale means.

## Why the API looks this way

- **Einsum is the core.** One notation covers contractions, reductions,
  permutations, diagonals and broadcasting. The direct calls (`add`,
  `multiply`, `reduce` and the rest) name the operation instead of
  inferring it, and `permute` uses cuTENSOR's dedicated permutation kernel.
- **Every call is one explicit operation.** No operator overloading and no
  lazy expression graph: those would have to choose result layouts,
  temporaries and evaluation order out of sight. Each call names its output,
  scales and stream.
- **Planning and running are separate.** `make_plan` reads only shapes,
  types and alignment, and `Plan::execute` is const and allocation-free, so
  a plan can be built once, shared across streams and run on a hot path.
- **The caller owns memory.** Operands can be any device buffer, and
  workspace comes from a replaceable allocator.

## Ownership and lifetimes

Calls queue GPU work and return before it runs, and GPU work holds raw
pointers. So the data behind every operand, and every workspace, must stay
alive until the work that uses it is done, whoever owns it.

- A `Plan` borrows its `Context`, and a cached `shared_ptr<const Plan>`
  keeps only the plan alive. The context must outlive its plans and their
  work.
- A context is tied to the CUDA device that was current when it was built.
  Using it with another device current is `Errc::CudaError`, so each device
  gets its own context. Contexts and plans switch to their own device when
  destroyed, so they can go out of scope with any device current.
- The default context can be scoped with a `Runtime`, or set up eagerly
  with `initialize(config)` (which puts the slow one-time cuTENSOR setup
  where you choose) and released with `finalize()` once all plans and GPU
  work are done. With neither, it is built on first use and deliberately
  leaked at exit, because destroying a CUDA library handle from a static
  destructor races CUDA's own teardown. Its `Config` is read once, which is
  why configuring it a second time is `Errc::InvalidState` rather than
  silently ignored.
- Destructors never throw. A failure while releasing memory terminates,
  since carrying on could free memory a kernel is still using. The one
  exception is a CUDA context already lost while an exception unwinds: the
  memory is gone with the context, so cleanup returns and lets the original
  error through. See [cleanup](../reference/errors.md#cleanup).
