# Errors

Katal reports failures two ways, and only two:

- **`std::bad_alloc`** when memory runs out, from a tensor allocator, the
  default workspace allocator, or a tensor too large to address.
- **`katal::Error`** for everything else. It derives from
  `std::runtime_error`, and carries an `Errc` to branch on and, when there
  is one, the equation that failed.

```cpp
try {
  katal::einsum("ij,jk->ik", 1.0, a, b, 0.0, d);
} catch (const katal::Error& e) {
  if (e.code() == katal::Errc::UnsupportedPattern) { /* fall back */ }
  std::fprintf(stderr, "%s\n", e.what());  // quotes e.equation()
}
```

## Error codes

| `Errc`               | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `ParseError`         | The equation is malformed: an empty operand, a stray arrow, or a separator inside a label. |
| `RankMismatch`       | Labels and extents disagree in count, or a rank exceeds `max_rank`.     |
| `ExtentMismatch`     | A label has two extents, an extent is not positive, or a shape overflows. |
| `UnsupportedPattern` | Well formed but not lowered, or a call that does not fit its plan; see [below](#unsupportedpattern). |
| `DtypeMismatch`      | An operand or scalar representation is unsupported.               |
| `WorkspaceTooSmall`  | `execute` got less workspace than `Plan::workspace_size()`, or cuTENSOR reported too little. |
| `CutensorError`      | cuTENSOR failed; the message carries its text.                          |
| `CudaError`          | A CUDA call failed, or a plan ran with another device current.          |
| `InvalidState`       | The process-wide context was misused: a second `initialize()`, a `Runtime` when a context exists, or `finalize()` while a `Runtime` owns it. Also a failed [assertion](#assertions) in a build that throws on one. |

### UnsupportedPattern

The code covers both equations Katal cannot lower and calls it refuses to
run:

- an equation Katal does not lower, with the labels responsible in the
  message (see [what is rejected](equations.md#what-is-rejected)), or
  cuTENSOR's `NOT_SUPPORTED`;
- an `execute` pointer less aligned than the plan was built for;
- an elementwise input that is the output, or a permutation input that
  overlaps it;
- a zero scale on an elementwise operator other than `Add`;
- a permutation given a beta;
- an output carrying a unary operator;
- an `execute` overload that does not match the plan;
- a workspace under 256-byte alignment for a plan that keeps a reduced input
  in it.

## Assertions

Katal checks two kinds of condition, with two macros from
`<katal/assert.hpp>`:

- **`KATAL_ENFORCE`** checks what a caller passed in, such as an `execute`
  pointer's alignment or the workspace size. It is never compiled out, and a
  failure always throws `katal::Error` with the code documented for it.
- **`KATAL_ASSERT`** checks what Katal itself guarantees, or misuse the API
  calls undefined, such as using a moved-from `Context` or `Plan`. A failure
  is a bug, so what it does is fixed when Katal is configured, by
  `KATAL_ASSERT_BEHAVIOR`:

| Behavior | A failed `KATAL_ASSERT` |
| --- | --- |
| `THROW` | throws `katal::Error` with `Errc::InvalidState`; the message names the check and its source location |
| `ABORT` | prints the same to stderr and calls `std::abort` |
| `IGNORE` | is compiled out; the condition is not evaluated |

The default is `IGNORE` for `Release` and `MinSizeRel` builds and `ABORT`
otherwise. `katal::assert_behavior` says which one a build has.

## Scalars

Scalars are converted to the backend's compute type. This conversion can narrow
precision, and a complex scalar's imaginary part is discarded for real compute
types; `DtypeMismatch` does not diagnose these losses.

## Non-throwing overloads

`Equation::parse` and `Equation::classify` also have overloads that report
through a `std::error_code` and never throw. The code compares equal to the
`Errc` the throwing overload would have raised. Any other exception, such as
`std::bad_alloc`, comes back as `ParseError` from `parse` and
`UnsupportedPattern` from `classify`.

## Cleanup

Destructors and deallocators never throw. A CUDA failure while releasing
memory terminates the program, because continuing could free memory a kernel
still uses. The exception is a failure that means the CUDA context is already
lost, such as after an illegal-address fault, while an exception is unwinding:
cleanup then returns so the original error can reach its handler. Outside
unwinding, including explicit cleanup after catching a fault, it terminates.

Cleanup can detect active unwinding, but cannot identify the exception. If an
unrelated exception is unwinding when the context is lost, cleanup also
suppresses that device fault. When a device fault must be reported on its own,
check for it, for example with `cudaStreamSynchronize`, before leaving the
scope that owns the tensors and plans.
