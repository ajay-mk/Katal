# cuEin

A C++20 einsum front-end for NVIDIA cuTENSOR.

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

    Bring your own device allocator, or let cuEin manage scratch with
    `cudaMallocAsync`.

</div>

[API reference](api/index.html){ .md-button }
