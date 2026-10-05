# Errors

Katal reports failures three ways:

- **`std::bad_alloc`** when memory runs out, from a tensor allocator, the
  default workspace allocator, or a tensor too large to address.
- **`katal::Error`** for everything else. It derives from
  `std::runtime_error`, and carries an `Errc` to branch on and, when there
  is one, the equation that failed.
- **`std::logic_error`** for misusing the process-wide context, described
  below.

```cpp
try {
  katal::einsum("ij,jk->ik", 1.0, a, b, 0.0, d);
} catch (const katal::Error& e) {
  if (e.code() == katal::Errc::UnsupportedPattern) { /* fall back */ }
  std::fprintf(stderr, "%s\n", e.what());  // quotes e.equation()
}
```

| `Errc`               | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `ParseError`         | The equation is malformed: an empty operand, a stray arrow.             |
| `RankMismatch`       | Labels and extents disagree in count, or a rank exceeds `max_rank`.     |
| `ExtentMismatch`     | A label has two extents, an extent is not positive, or a shape overflows. |
| `UnsupportedPattern` | Well formed but not lowered; cuTENSOR's NOT_SUPPORTED; or `execute` got a pointer less aligned than the plan. |
| `DtypeMismatch`      | The compute type cannot hold the operands, alpha or beta.               |
| `WorkspaceTooSmall`  | `execute` got less workspace than `Plan::workspace_size()`.             |
| `CutensorError`      | cuTENSOR failed; the message carries its text.                          |
| `CudaError`          | A CUDA call failed, or a plan ran with another device current.          |

Misusing the process-wide context throws `std::logic_error`: a second
`initialize()`, a `Runtime` when a context exists, or `finalize()` while a
`Runtime` owns it.

## Without exceptions in hand

`Equation::parse` and `Equation::classify` also have overloads that report
through a `std::error_code` and never throw. The code compares equal to the
`Errc` the throwing overload would have raised; any other exception, such as
`std::bad_alloc`, comes back as `ParseError` from `parse` and
`UnsupportedPattern` from `classify`:

```cpp
std::error_code ec;
const auto eq = katal::Equation::parse(text, ec);
if (ec == katal::Errc::RankMismatch) { /* ... */ }
```

## Cleanup

Destructors and deallocators never throw. A CUDA failure while releasing
memory terminates the program, because continuing could free memory a kernel
still uses. The exception is a failure that means the CUDA context is already
lost, such as after an illegal-address fault: cleanup then returns, so the
original error can still reach a handler.
