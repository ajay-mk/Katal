# Equations

Katal takes einsum equations in NumPy's notation: comma-separated operands,
then optionally `->` and the result. `"ijab,abcd->ijcd"` contracts `a` and `b`.

## Grammar

- **Labels are single bytes.** Any byte other than `,`, `-`, `>` and ASCII
  whitespace is a label, so `"ij"` is two modes and `"ab"` is not one.
- **ASCII whitespace is ignored**, newlines included, so an equation may be
  spaced out or split across lines in a raw string. The arrow itself must be
  written `->`: `- >` is a `ParseError`.
- **Each operand and the result has at most `katal::max_rank` labels** (12 by
  default, set with `-DKATAL_MAX_RANK`). The equation as a whole may use more
  distinct labels than that.
- An empty operand, a second `->`, or `-` or `>` inside a label is a
  `ParseError`.

## The implicit result

Without `->`, the result is the labels that appear exactly once across the
operands, **sorted by byte value**, as NumPy and PyTorch do:

| Equation | Result | Lowers to                     |
|----------|--------|-------------------------------|
| `ij,jk`  | `ik`   | contraction over `j`          |
| `kj,ji`  | `ik`   | contraction, not `ki`         |
| `ba`     | `ab`   | transpose, not a copy         |
| `ba,AB`  | `ABab` | outer product; upper case first |

Write `->` whenever the result's mode order matters to you; it is never
inferred differently from what is written.

## What each equation lowers to

| Operands | Pattern                                   | Classification |
|----------|-------------------------------------------|----------------|
| two      | contracted, batch and free labels         | contraction   |
| one      | result drops labels                       | reduction     |
| one      | result reorders labels                    | permutation   |
| one      | result equals the operand                 | copy          |

A label repeated within one input reads that input along its diagonal:
`"ii->"` is a trace, `"ii->i"` the diagonal, and `"iij,jk->ik"` contracts
the diagonal of the first input. Katal merges the repeats into one mode whose
stride is the sum of theirs, so no copy is made. The repeats must have equal
extents.

With two operands, a label in one input alone (neither in the other nor in
the result) is summed out of that input first: `"ijk,kl->il"` sums j out of
the first input, then contracts k. The summed input is kept in the plan's
workspace, so such a plan has a nonzero `workspace_size()`, runs two kernels,
and needs its workspace aligned to 256 bytes, which `cudaMallocAsync` and
the default allocator give. `PlanOptions::workspace_limit` counts the summed
input too; a limit too small to hold it is `WorkspaceTooSmall` when the plan
is built. The summed input is stored in the input's own type and summed in
that type's default compute type, whatever `PlanOptions::compute` asks of the
contraction. For F16 and BF16 that means the partial sums are rounded to half
precision before the contraction reads them.

Katal rejects these with `UnsupportedPattern`, naming the labels responsible:

- a label repeated in the result (writing a diagonal),
- in einsum, a result label that appears in no input (broadcasting; see
  [Elementwise operations](#elementwise-operations) and
  [Permutations](#permutations)).

Three or more operands are also `UnsupportedPattern`, since they would need
a contraction path.

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
Contractions (two-operand einsum) take only `Identity` and `Conj`; the
elementwise calls, reductions and permutations take the rest. cuTENSOR
refuses an operator it does not support when the plan is built, which Katal
reports as `UnsupportedPattern` or, for some refusals, `CutensorError`.

## Reductions other than a sum

`katal::reduce` is the one-operand einsum with a choice of how the dropped
labels combine: `BinaryOp::Add`, `Multiply`, `Max` or `Min`.

```cpp
// d[i] = max_j |a[i][j]|
katal::reduce("ij->i", katal::BinaryOp::Max, 1.0,
              katal::apply(katal::UnaryOp::Abs, a), 0.0, d);
```

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

alpha and beta must be nonzero except for `Add`. cuTENSOR skips an input
whose scale is zero rather than treating it as zeros, so `max(x, 0)` written
as `katal::max(eq, 1.0, x, 0.0, y, d)` would come back as `x`. Katal refuses
that with `UnsupportedPattern`; for a ReLU, pass a tensor of zeros with a
scale of one.

To hoist the planning out of a loop, build the plan once with
`make_plan(context, equation, op, a, b, d)` and call the elementwise
`Plan::execute(alpha, a, beta, b, gamma, c, d, workspace)`. `c` shares `d`'s
layout and may be the same pointer.
