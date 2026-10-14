# Katal

A C++20 einsum frontend for GPU tensor libraries.

Libraries like cuTENSOR run tensor contractions fast, but every one of them
needs its own descriptors, plan and workspace written by hand, and a real
application can have hundreds. With Katal you write the equation instead:

```cpp
katal::einsum("ijab,abcd->ijcd", 0.5, t2, v, 1.0, r);
```

Katal parses the equation, builds the matching plan for the backend and
caches it by equation and shape, so the code reads like the math it computes
and a hot loop pays for setup once. NVIDIA cuTENSOR is the backend today.

!!! note "Work in progress"
    The operations described here are implemented and tested, but the API
    is not stable yet. Expect breaking changes.

<div class="grid cards" markdown>

-   __Tutorial__

    ---

    New to Katal? Build a program against it and run your first
    contraction.

    [Your first contraction](tutorial/first-contraction.md)

-   __How-to guides__

    ---

    Building and installing, hoisting plans out of loops, and bringing your
    own memory.

    [Build, test and install](how-to/build.md)

-   __Reference__

    ---

    The notation, every operation, configuration, tensors, errors and log
    output, plus the [API reference](api/index.html).

    [Equations](reference/equations.md)

-   __Design__

    ---

    How a call becomes a kernel, why the API looks the way it does, and the
    ownership rules behind it.

    [Design](explanation/design.md)

</div>
