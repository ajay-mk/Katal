# Tensor ownership and allocators

## Ownership and execution

- `Tensor<Allocator>` owns dense, column-major storage. Copies share the
  allocation; they do not copy its values. The allocator follows standard
  allocator semantics and must allocate `std::byte`.
- `TensorRef` borrows data and shape/stride metadata. Keep the data alive
  until GPU work completes, and the metadata alive while using the view.
  In particular, `tensor.ref()` borrows that Tensor object's extents, even
  when another Tensor still owns the shared allocation.
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
from destruction or freeing storage that may still be in use.

Managed allocations use `cudaMallocManaged` / `cudaFree`; they are not pooled.
Pooled device allocations use `cudaMallocAsync` / `cudaFreeAsync` on their
allocator's stream, without a host-side wait during release. Allocation
exhaustion throws `std::bad_alloc`; other CUDA allocation errors throw
`katal::Error`. Both allocators terminate on a failed cleanup call.

See the [pooled and managed Tensor example](examples.md#pooled-and-managed-tensor-storage)
for a complete matrix product using either allocator.
