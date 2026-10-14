# Equations

Katal takes einsum equations in NumPy's notation: comma-separated operands,
then optionally `->` and the result. `"ijab,abcd->ijcd"` contracts `a` and `b`.
This page covers the notation and what each equation computes; the calls
that take an equation are on [operations](operations.md).

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

## Diagonals and traces

A label repeated within one input reads that input along its diagonal:
`"ii->"` is a trace, `"ii->i"` the diagonal, and `"iij,jk->ik"` contracts
the diagonal of the first input. Katal merges the repeats into one mode whose
stride is the sum of theirs, so no copy is made. The repeats must have equal
extents.

## Labels summed out of one input

With two operands, a label in one input alone (neither in the other nor in
the result) is summed out of that input first: `"ijk,kl->il"` sums `j` out
of the first input, then contracts `k`.

- The plan runs one reduction per affected input before the contraction:
  two kernels, or three when both inputs have such labels, as in
  `"ijk,kml->il"`.
- The summed input is kept in the plan's workspace, so `workspace_size()` is
  nonzero and the workspace must be aligned to 256 bytes, which
  `cudaMallocAsync` and the default allocator give.
- `PlanOptions::workspace_limit` counts the summed input too. A limit too
  small to hold it is `WorkspaceTooSmall` when the plan is built.
- The summed input is stored in the input's own type and summed in that
  type's default compute type, whatever `PlanOptions::compute` asks of the
  contraction. For F16 and BF16 the partial sums are rounded to half
  precision before the contraction reads them.

## What is rejected

Katal rejects these with `UnsupportedPattern`, naming the labels responsible:

- a label repeated in the result (writing a diagonal),
- in einsum, a result label that appears in no input (broadcasting; use
  [elementwise operations](operations.md#elementwise-operations) or
  [permute](operations.md#permutations) for that).

Three or more operands are also `UnsupportedPattern`, since they would need
a contraction path.
