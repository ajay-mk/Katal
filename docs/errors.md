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

| `Errc`               | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `ParseError`         | The equation is malformed: an empty operand, a stray arrow, or a separator inside a label. |
| `RankMismatch`       | Labels and extents disagree in count, or a rank exceeds `max_rank`.     |
| `ExtentMismatch`     | A label has two extents, an extent is not positive, or a shape overflows. |
| `UnsupportedPattern` | Well formed but not lowered; cuTENSOR's NOT_SUPPORTED; `execute` got a pointer less aligned than the plan, an elementwise or permutation input that is the output, a zero scale on an elementwise operator other than Add, a permutation given a beta, an output with a unary operator, or an overload that does not match the plan. |
| `DtypeMismatch`      | An operand or scalar representation is unsupported.               |
| `WorkspaceTooSmall`  | `execute` got less workspace than `Plan::workspace_size()`, or cuTENSOR reported too little. |
| `CutensorError`      | cuTENSOR failed; the message carries its text.                          |
| `CudaError`          | A CUDA call failed, or a plan ran with another device current.          |
| `InvalidState`       | The process-wide context was misused: a second `initialize()`, a `Runtime` when a context exists, or `finalize()` while a `Runtime` owns it. |

Scalars are converted to the backend's compute type. This conversion can narrow
precision, and a complex scalar's imaginary part is discarded for real compute
types; `DtypeMismatch` does not diagnose these losses.

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
lost, such as after an illegal-address fault, while an exception is unwinding:
cleanup then returns so the original error can reach its handler. Outside
unwinding, including explicit cleanup after catching a fault, it terminates.

Cleanup can detect active unwinding, but cannot identify the exception. If an
unrelated exception is unwinding when the context is lost, cleanup also
suppresses that device fault. Check asynchronous CUDA errors explicitly before
leaving the scope when they must be reported separately.
