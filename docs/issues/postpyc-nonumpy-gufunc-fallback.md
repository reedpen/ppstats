# postpyc issue — DRAFT, not yet filed (2026-07-15)

Proposed title: "Interpreted no-numpy fallback calls @guvectorize kernels
without their out parameter".

---

## Summary

`postpyc`'s interpreted broadcast machinery has a no-numpy fallback that is
only correct for scalar `@vectorize` kernels. For `@guvectorize` kernels,
which receive their result buffer as a trailing ``out`` array parameter, the
fallback calls the kernel with the user's inputs only, so every gufunc raises
``TypeError`` in an environment without numpy:

```
File ".../postpyc/ufunc.py", line 272, in _broadcast_call
    return fn(*args)
TypeError: mean() missing 1 required positional argument: 'out'
```

The numpy path (`ufunc.py` line ~257) allocates `call_outputs` and passes
them; the `except ImportError` fallback (`ufunc.py` line ~270) does not:

```python
    except ImportError:
        # NumPy not available — fall back to direct scalar call.
        return fn(*args)
```

Found while verifying `ppstats` (PostSciPy Target 1, descriptive reductions)
in a numpy-free environment: numpy is an optional extra for both ppspecial
and ppstats, but every `(n)->()` reduction is unusable interpreted without
it, e.g. `ppstats.mean([1.0, 2.0])`.

## Minimal reproducer

```python
# nonumpy_gufunc_repro.py — run in an environment WITHOUT numpy
from postyp import Array, Float64
from postpyc import guvectorize


@guvectorize([], "(n)->()")
def total(a: Array[Float64], out: Array[Float64]) -> None:
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i]
    out[0] = acc


print(total([1.0, 2.0, 3.0]))  # TypeError: total() missing ... 'out'
```

Observed with postpyc 0.3.0 (PyPI). Expected: the fallback allocates a plain
Python list (or array) for each output core dimension, passes it to the
kernel, and unwraps `()`-shaped outputs to a scalar — mirroring the numpy
path's semantics for single un-batched calls, or at minimum raises an
explicit "interpreted gufuncs require numpy" error instead of a confusing
``TypeError``.
