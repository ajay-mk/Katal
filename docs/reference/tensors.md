# Tensors

## Data types

Katal is tested with four storage types: `F32`, `F64`, `C32` and `C64`.

!!! warning "Untested"
    `F16` and `Bf16` are accepted and passed to cuTENSOR as they are, as are
    the tensor-core compute modes in `PlanOptions::precision` (TF32 and the
    like, which apply to contractions only and are ignored by every other
    operation). None of those are tested. Whether a combination runs is up
    to cuTENSOR and the GPU, and one it refuses is `UnsupportedPattern` at
    plan time. The emulated modes (`Bf16x9`, `F16x4`, `Int8x8`) also need a
    CUDA version that provides them; on one that does not, cuTENSOR quietly
    computes in native F32 or F64 instead.

Inputs of different types may meet where cuTENSOR allows it, such as a C64
tensor contracted with an F64 one into C64.

## Tensors and references

- `Tensor<Allocator>` owns dense, column-major storage. Copies share the
  allocation; they do not copy its values. The allocator follows standard
  allocator semantics and must allocate `std::byte`.
- `TensorRef` borrows data and shape/stride metadata. Keep the data alive
  until GPU work completes, and the metadata alive while using the view.
  In particular, `tensor.ref()` borrows that Tensor object's extents, even
  when another Tensor still owns the shared allocation. A `TensorRef` is
  column-major unless its `layout` says otherwise, as is one from `make_ref`;
  C-ordered data needs `Layout::RowMajor`.
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
- `a.conj()` is `a` read as its complex conjugate, for passing to `einsum`,
  the same as `katal::conj(a)`. `a.sum<T>()`, `a.max<T>()` and `a.min<T>()`
  reduce the whole tensor and return the value to the host, blocking until
  it arrives; `T` is the element type, and `max` and `min` take real types
  only. To keep the result on the device or reduce only some labels, use
  `reduce` or the one-operand `einsum`.
- Tensor allocation and execution workspace are separate policies. Changing
  a Tensor to managed storage does not change the temporary device workspace
  a call obtains through its workspace allocator; see
  [use your own workspace allocator](../how-to/memory.md#use-your-own-workspace-allocator).

## Pooled and managed storage

`katal::Tensor<>` uses pooled device memory by default. To allocate CUDA
managed memory, include `<katal/tensor.hpp>` and use
`katal::Tensor<katal::ManagedAllocator<std::byte>>`. Tensor copies share
the allocation; `ref()` borrows it for execution.

The allocator accepts a `StreamRef` (default: the default CUDA stream). All GPU
uses of that allocation must use that stream, which must outlive the storage.
Wait for GPU work before accessing the data on the CPU, and finish CPU writes
before submitting GPU work. Last-owner release waits for the stream before
freeing managed memory. A CUDA failure during release follows the
[cleanup](errors.md#cleanup) rules.

Managed allocations use `cudaMallocManaged` / `cudaFree`; they are not pooled.
Pooled device allocations use `cudaMallocAsync` / `cudaFreeAsync` on their
allocator's stream, without a host-side wait during release. Allocation
exhaustion throws `std::bad_alloc`, as does a tensor too large to address;
other CUDA allocation errors throw `katal::Error`. Every other failure is a
`katal::Error`, including an extent that is not positive
(`Errc::ExtentMismatch`) and a rank beyond `katal::max_rank`
(`Errc::RankMismatch`).

To switch a tensor to managed storage, see
[use managed memory](../how-to/memory.md#use-managed-memory-for-a-tensor).
