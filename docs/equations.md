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

The cuTENSOR backend executes all unary classifications through its reduction
operation, preserving `alpha * input + beta * output` for copies and permutations.

Katal rejects these with `UnsupportedPattern`, naming the labels responsible:

- a label repeated within one operand (a diagonal or trace) or in the result,
- in einsum, a result label that appears in no input (broadcasting; see
  [Elementwise operations](#elementwise-operations)),
- with two operands, a label in one input that is neither in the other nor in
  the result (a reduction fused into a contraction).

Three or more operands are also `UnsupportedPattern`, since they would need
a contraction path.

Unary operators such as conjugation are not available yet.

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

To hoist the planning out of a loop, build the plan once with
`make_plan(context, equation, op, a, b, d)` and call the elementwise
`Plan::execute(alpha, a, beta, b, gamma, c, d, workspace)`. `c` shares `d`'s
layout and may be the same pointer.
