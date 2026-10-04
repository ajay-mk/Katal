# Equations

Katal takes einsum equations in NumPy's notation: comma-separated operands,
then optionally `->` and the result. `"ijab,abcd->ijcd"` contracts `a` and `b`.

## Grammar

- **Labels are single bytes.** Any byte other than `,`, `-`, `>` and ASCII
  whitespace is a label, so `"ij"` is two modes and `"ab"` is not one.
- **ASCII whitespace is ignored**, newlines included, so an equation may be
  spaced out or split across lines in a raw string.
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

| Operands | Pattern                                   | Kernel        |
|----------|-------------------------------------------|---------------|
| two      | contracted, batch and free labels         | contraction   |
| one      | result drops labels                       | reduction     |
| one      | result reorders labels                    | permutation   |
| one      | result equals the operand                 | copy          |

Katal rejects with `UnsupportedPattern`, naming the labels responsible:

- a label repeated within one operand (a diagonal or trace) or in the result,
- a result label that appears in no input (broadcasting),
- with two operands, a label in one input that is neither in the other nor in
  the result (a reduction fused into a contraction),
- three or more operands, which would need a contraction path.

There are no element-wise operators such as conjugation.
