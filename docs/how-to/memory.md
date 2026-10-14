# Bring your own memory

Tensor storage and the scratch a call needs are separate. Each can come from
Katal's defaults or from you.

## Pass your own device buffers

Any device pointer can be an operand. Describe it with `make_ref`:

```cpp
const std::int64_t extent[] = {m, n};
auto a = katal::make_ref(device_ptr, katal::Dtype::F64, extent);
```

`make_ref` describes the buffer as row-major unless you pass
`katal::Layout::ColumnMajor` as a fourth argument; `Tensor` storage is always
column-major. The data must stay alive until the GPU work that reads it is
done, and the `extent` array as long as the reference is used.

## Use managed memory for a tensor

`katal::Tensor<>` uses pooled device memory. To make a tensor the CPU can
also address, use CUDA managed memory instead:

```cpp
#include <katal/tensor.hpp>

katal::Tensor<katal::ManagedAllocator<std::byte>> a(katal::Dtype::F64,
                                                    {64, 64});
```

The allocator uses the default CUDA stream unless you give it another one:

```cpp
using Managed = katal::ManagedAllocator<std::byte>;
katal::Tensor<Managed> a(katal::Dtype::F64, {64, 64}, Managed(stream));
```

Then:

- Use that stream for every GPU use of the tensor, and keep the stream alive
  longer than the storage.
- Wait for GPU work before reading or writing the data on the CPU, and finish
  CPU writes before submitting GPU work.

Managed memory is not pooled, and it does not change the workspace a call
allocates. That still comes from the call's workspace allocator.

`examples/tensor_storage.cpp` runs the same matrix product with both
allocators; see [run the examples](build.md#run-the-examples).

## Use your own workspace allocator

By default, each call allocates the scratch its plan needs with
`cudaMallocAsync` and frees it with `cudaFreeAsync`, ordered on the call's
stream. To draw scratch from a pool you already manage, give Katal an
allocator for it.

### Write the allocator

Any type that models `katal::WorkspaceAllocator` works. It needs two members:

```cpp
struct PoolAllocator {
  MyPool* pool;  // a handle, not the pool itself

  katal::Workspace allocate(std::uint64_t bytes,
                            katal::StreamRef stream) const {
    return {pool->allocate(bytes, stream.get()), bytes};
  }
  void deallocate(katal::Workspace w, katal::StreamRef stream) const noexcept {
    pool->free(w.ptr, stream.get());
  }
};
```

- `allocate` returns a `Workspace`: device memory of at least `bytes`, and
  its size.
- `deallocate` is called as soon as the work is queued, not when it has
  run, so it must be stream-ordered: the memory must not be reused until
  `stream` reaches that point (as `cudaFreeAsync` does), or it must
  synchronize first. The same holds for a workspace you pass to a hoisted
  plan yourself.
- `deallocate` must not throw. It also runs while an exception unwinds.
- The allocator is taken by value, so make it a cheap handle to the pool.
- A plan that sums a label out of one input before a contraction keeps that
  input in its workspace and needs it aligned to 256 bytes.

### Pass it to a call

The allocator is the last argument of every call:

```cpp
katal::einsum("ij,jk->ik", 1.0, a, b, 0.0, d, {}, stream,
              katal::default_context(), PoolAllocator{&pool});
```

For a [hoisted plan](hoist-a-plan.md), allocate `plan.workspace_size()`
bytes yourself, wrap them in a `katal::Workspace`, and pass that to
`execute`.
