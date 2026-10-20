# Operations

Every operation takes an [equation](equations.md) naming how the inputs line
up with the result, scale factors (`alpha`, `beta`, `gamma`), and device
tensors as `TensorRef`s or `Tensor`s. Each call looks up a cached plan,
allocates any scratch it needs, queues the work on a stream and returns
without waiting for it.

| Call | Computes |
| --- | --- |
| `einsum(eq, alpha, a, b, beta, d)` | `d = alpha * contract(a, b) + beta * d` |
| `einsum(eq, alpha, a, b, c, beta, e)` | `e = alpha * contract(a, b, c) + beta * e`, in one kernel |
| `einsum(eq, alpha, a, beta, d)` | `d = alpha * reduce_or_permute(a) + beta * d` |
| `reduce(eq, op, alpha, a, beta, d)` | the one-operand einsum, combining with `op` instead of a sum |
| `permute(eq, alpha, a, d)` | `d = alpha * a`, permuted, broadcast or converted |
| `elementwise(eq, op, alpha, a, beta, b, gamma, d)` | `d = op(alpha * a, beta * b) + gamma * d` |
| `add`, `multiply`, `max`, `min` | `elementwise` with that `op` and `gamma = 0` |

Each call also takes optional `PlanOptions`, a stream, a `Context` and a
workspace allocator, in that order after the output. To build a plan once and
run it many times, see [hoist a plan out of a loop](../how-to/hoist-a-plan.md).

## Operators on inputs

Any input can be read through one of cuTENSOR's unary operators, without
changing its data. `katal::conj(t)` conjugates and `katal::apply(op, t)`
takes any `UnaryOp`, such as `Abs`, `Sqrt`, `Exp` or `Neg`:

```cpp
// d[i][k] = sum_j conj(a[i][j]) * b[j][k]
katal::einsum("ij,jk->ik", 1.0, katal::conj(a), b, 0.0, d);
```

The operator travels in the `ConstTensorRef` (its `op` field), so it works
with every call and `make_plan` overload, and plans with different
operators are cached apart. An output must not carry one.

Which operators an operation accepts for a type is up to cuTENSOR.
Contractions (two- and three-operand einsum) take only `Identity` and
`Conj`; the elementwise calls, reductions and permutations take the rest.
cuTENSOR refuses an operator it does not support when the plan is built,
which Katal reports as `UnsupportedPattern` or, for some refusals,
`CutensorError`.

## Elementwise operations

`katal::elementwise` pairs elements by label instead of summing over shared
ones, and computes `d = op(alpha * a, beta * b) + gamma * d` with `op` one of
`BinaryOp::Add`, `Multiply`, `Max` or `Min`:

```cpp
// d[i][j][a][b] = t[j][i][a][b] - t[i][j][a][b]
katal::add("jiab,ijab->ijab", 1.0, t, -1.0, t, d);

// Broadcasting: add the vector v to every row of m, then accumulate.
katal::elementwise("ij,j->ij", katal::BinaryOp::Add, 1.0, m, 1.0, v, 1.0, d);
```

`add`, `multiply`, `max` and `min` are the same call with `gamma = 0`, so
they overwrite `d` without reading it.

The equation takes exactly two inputs. A label an input lacks is broadcast
along it, and every input label must appear in the result; an equation that
would sum over a label is `UnsupportedPattern`, since that is einsum's job.
Neither input may be `d` itself.

`alpha` and `beta` must be nonzero except for `Add`. cuTENSOR skips an input
whose scale is zero rather than treating it as zeros, so `max(x, 0)` written
as `katal::max(eq, 1.0, x, 0.0, y, d)` would come back as `x`. Katal refuses
that with `UnsupportedPattern`; for a ReLU, pass a tensor of zeros with a
scale of one.

The hoisted form is in [hoist a plan out of a loop](../how-to/hoist-a-plan.md#other-kinds-of-plan).

## Reductions

`katal::reduce` is the one-operand einsum with a choice of how the dropped
labels combine: `BinaryOp::Add`, `Multiply`, `Max` or `Min`.

```cpp
// d[i] = max_j |a[i][j]|
katal::reduce("ij->i", katal::BinaryOp::Max, 1.0,
              katal::apply(katal::UnaryOp::Abs, a), 0.0, d);
```

A whole tensor can also be reduced to a host value with `a.sum<T>()`,
`a.max<T>()` or `a.min<T>()` on a `Tensor`; see
[tensors](tensors.md#tensors-and-references).

## Permutations

The one-operand einsum permutes through a reduction kernel, which is what
lets it take beta. `katal::permute(equation, alpha, a, d)` computes
`d = alpha * a` through cuTENSOR's dedicated permutation kernel instead. It
can also broadcast `a` along result labels it lacks, as in `"j->ij"`, and
write a different type than it reads, such as F32 from F16 or F64. Unless
`PlanOptions::compute` says otherwise, a conversion accumulates in the wider
of the two types, which is what cuTENSOR requires. It has no beta,
and `a` must not overlap `d`. The hoisted form is `make_permutation_plan`, run
with the one-operand `Plan::execute` and a beta of zero.
