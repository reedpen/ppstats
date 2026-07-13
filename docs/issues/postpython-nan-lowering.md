# postpython issue — FILED as [openteams-ai/postpython#36](https://github.com/openteams-ai/postpython/issues/36) (2026-07-13)

Title: "NAN/INF constants from postpyc.math fail to lower: emitted as bare
'nan'/'inf' identifiers in C". Body archived below.

---

## Summary

`postpyc.math.NAN` (and by the same mechanism `INF`) is valid POST Python that
runs interpreted but fails to compile: the backend emits the constant via its
`repr()`, producing the bare C identifier `nan`, which collides with libm's
`double nan(const char *)`.

Found while implementing `ppstats._descriptive` (Target 1, descriptive
reductions) — scipy-compatible domain behavior wants `out[0] = NAN` for e.g.
`gmean` of a vector containing a negative element.

## Minimal reproducer

```python
# nan_repro.py
from postyp import Array, Float64
from postpyc import guvectorize
from postpyc.math import NAN


@guvectorize([], "(n)->()")
def always_nan(a: Array[Float64], out: Array[Float64]) -> None:
    out[0] = NAN
```

```python
from pathlib import Path
from postpyc.build import build_file
build_file(Path("nan_repro.py"), output=Path("nan_repro.so"))
```

## Observed

Interpreted mode: works (`always_nan([1.0])` → `nan`).

Compiled (postpyc 0.3.0 from PyPI, cc = gcc, linux-64):

```
postpyc.build.BuildError: C compiler failed (cc -O2 -fPIC -c .../00-nan_repro.c ...):
.../00-nan_repro.c: In function 'always_nan':
.../00-nan_repro.c:105:23: error: incompatible types when initializing type 'double' using type '__attribute__((const)) double (*)(const char *)'
  105 |         double _k17 = nan;
      |                       ^~~
```

## Expected

Either lower these to the C99 macros (`NAN` / `INFINITY` from `<math.h>`) or to
expressions like `(0.0/0.0)` / `HUGE_VAL`, or reject with an explicit
unsupported-feature diagnostic (PP9xx) per the cardinal rule — anything but an
opaque C compiler error. `repr(float('nan'))` is not valid C source, so any
float constant whose repr is `nan`/`inf`/`-inf` needs a special case at
emission.

## Workaround in library code

`ppstats._descriptive` currently produces NaN arithmetically from array
elements — `out[0] = (a[i] - a[i]) / (a[i] - a[i])` — which lowers fine and
matches interpreted-mode numpy semantics, but reads as a workaround. Tracked in
ppstats' ROADMAP to revert once this lands.
