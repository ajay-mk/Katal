# Diagnostics

Configure with `-DKATAL_ENABLE_LOGGING=ON` and Katal writes diagnostics to
stderr. The default `OFF` build compiles logging out entirely. Logging adds
host overhead, so leave it off when measuring performance.

## Events

Messages start with `[katal]`:

| Event | Meaning |
| --- | --- |
| `cache.hit` | a cached plan was reused; includes the equation |
| `cache.miss` | no cached plan matched |
| `cache.bypass` | caching is disabled |
| `cache.insert` | a new plan was cached; includes the equation |
| `cache.evict` | the least recently used plan made room for a new one |
| `plan.created` | a plan was built |

The `key` field is a diagnostic hash for correlating events within a run,
not a persistent or collision-free operation ID. Tensor values are never
printed. A retained plan's execution does no cache lookup and so produces no
cache event.

## Plan records

`plan.created` records the equation, tensor shapes, data types, strides,
alignment, planning options, required workspace bytes, and `host_plan_us`.
That time covers construction in `make_plan`, not equation parsing, cache
lookup, logging output or GPU execution, and no GPU synchronization is added.
Empty strides mean the tensor's dense layout, and a workspace of zero needs
no scratch allocation.

Planning options are printed by name, as in `algo=Gett precision=Tf32`.
`compute=output` means the accumulation type follows the output (F32 for an
F16 or BF16 output), and `kernel_rank=-1` leaves the kernel rank unspecified.

## Limits

Diagnostics are best-effort: an output failure never replaces an
operation's own error. cuTENSOR's own logger is separate, set with
`CUTENSOR_LOG_LEVEL` and `CUTENSOR_LOG_FILE`; see NVIDIA's
[logging guide](https://docs.nvidia.com/cuda/cutensor/latest/user_guide.html#logging).
