# Tensor ownership and allocators

## Ownership and execution

- `Tensor<Allocator>` owns dense, column-major storage. Copies share the
  allocation; they do not copy its values. The allocator follows standard
  allocator semantics and must allocate `std::byte`.
- `TensorRef` borrows data and shape/stride metadata. Keep the data alive
  until GPU work completes, and the metadata alive while using the view.
  In particular, `tensor.ref()` borrows that Tensor object's extents, even
  when another Tensor still owns the shared allocation.
- `ConstTensorRef` is the read-only form. Inputs and planning take it, and a
  `TensorRef` converts to it, so only the output of `einsum` must be writable.
  `make_ref` returns a `ConstTensorRef` for a pointer to const, and a const
  Tensor's `ref()` does too. `make_ref` and `ref()` record the largest
  alignment the address satisfies, up to 256 bytes, so a pointer into the
  middle of an allocation is described correctly. A plan built from a
  `nullptr` ref assumes 256, and executing it on a less aligned pointer
  throws `Errc::UnsupportedPattern` rather than running the wrong kernel.
  A type of your own that converts to `TensorRef` should also convert to
  `ConstTensorRef`, since C++ will not chain the two conversions for an input.
- A Tensor converts to the reference `ref()` would return, so it can be
  passed to `einsum`, `make_plan` and `cached_plan` directly:
  `einsum("ij,jk->ik", 1.0, a, b, 0.0, d)`. A temporary Tensor neither
  converts nor has a `ref()`, so a reference cannot be stored from one and
  outlive the extents it borrows.
- `Context` owns the cuTENSOR handle and plan cache. `Plan` holds a selected
  operation ready for repeated execution. `einsum` uses cached plans;
  an explicit Plan lets callers hoist setup out of a loop.
  A Plan borrows its Context: keep the Context alive until all its Plans are
  destroyed and their submitted GPU work is finished. Keeping a cached
  `shared_ptr<Plan>` does not retain the Context.
- Tensor allocation and execution workspace are separate policies. Changing
  a Tensor to managed storage does not change the temporary device workspace
  that `einsum` obtains through its workspace allocator.

Use `katal::Runtime` to scope the default context, or manage an explicit
Context. This walkthrough targets one CUDA device. A stream is borrowed,
not owned; keep it alive through allocation use and release. Construct a Runtime
before using the default context. It rejects an already initialized context or
another Runtime, and `finalize()` is rejected while a Runtime owns the context.
With explicit initialization, finish all Plans and GPU work before `finalize()`.
`initialize()` throws once the default context exists, since its `Config` could
no longer take effect; call `finalize()` first to change it.

A context serves the CUDA device that was current when it was built. Planning
or executing with a different device current throws `Errc::CudaError`, so a
program that drives several devices builds one context per device. Contexts
and plans switch to their own device when destroyed, so they can go out of
scope with any device current.

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
from destruction or freeing storage that may still be in use. The exception is
a failure that means the CUDA context is already lost while an exception is
unwinding: the memory goes with the context, so cleanup returns and the original
error can reach its handler. Cleanup outside unwinding terminates on any CUDA
failure, including one already caught by the caller.

Managed allocations use `cudaMallocManaged` / `cudaFree`; they are not pooled.
Pooled device allocations use `cudaMallocAsync` / `cudaFreeAsync` on their
allocator's stream, without a host-side wait during release. Allocation
exhaustion throws `std::bad_alloc`, as does a tensor too large to address;
other CUDA allocation errors throw `katal::Error`. Every other failure is a
`katal::Error`, including an extent that is not positive
(`Errc::ExtentMismatch`), a rank beyond `katal::max_rank`
(`Errc::RankMismatch`), and misuse of the default-context lifecycle, such as
creating a `Runtime` while the default context exists (`Errc::InvalidState`).

See the [pooled and managed Tensor example](examples.md#pooled-and-managed-tensor-storage)
for a complete matrix product using either allocator.
