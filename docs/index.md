# Katal

A C++20 einsum front-end for GPU tensor libraries, currently NVIDIA
cuTENSOR.

Quantum chemistry methods like coupled cluster are built from tensor
contractions over occupied and virtual orbital indices — hundreds of them in
a full CCSD or CCSDT implementation, each needing its own cuTENSOR descriptor
if written by hand. Katal parses an einsum equation such as `"ijab,ijab->"`
and picks and caches the matching cuTENSOR kernel, so a contraction reads as
the equation it computes instead of as descriptor-building boilerplate.

!!! note "Work in progress"
    Equation parsing, mode classification, and the cuTENSOR backend for
    contraction, permutation, reduction, and copy are implemented and
    tested. There is no stable API yet — expect breaking changes.

<div class="grid cards" markdown>

-   __Equation parsing__

    ---

    Einstein notation in, a `Classification` out: batch, contracted, and free
    modes per operand.

-   __cuTENSOR backend__

    ---

    Contraction, permutation, reduction, and copy all lower to cuTENSOR
    kernels.

-   __Plan caching__

    ---

    `cached_plan` reuses a built plan by equation and shape; `einsum` uses it
    under the hood.

-   __Pluggable workspace__

    ---

    Bring your own device allocator, or let Katal manage scratch with
    `cudaMallocAsync`.

</div>

[API reference](api/index.html){ .md-button }
