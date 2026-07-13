# ppstats

Statistical functions and distributions, in POST Python.

`ppstats` reimplements `scipy.stats` in
[POST Python](https://github.com/openteams-ai/postpython) — every kernel is
fully-typed Python that runs under the standard CPython interpreter **and**
compiles ahead-of-time to native code (a plain C shared library and a NumPy
ufunc extension module) with the POST Python reference compiler.

Status: **Active** — descriptive-statistics reductions are implemented and
verified in both execution modes (see `ROADMAP.md` Target 1). Part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
to rebuild SciPy one subpackage at a time as the compiler's proving ground.

## Implemented functions

| Family | Module | Functions |
|---|---|---|
| Descriptive | `_descriptive` | `mean`, `variance`, `gmean`, `hmean`, `moment(a, order)`, `skew`, `kurtosis`, `sem(a, ddof)`, `zscore(a, ddof)` |

All are reduction gufuncs (`(n)->()`, `(n),()->()`, or `(n),()->(n)`) that
broadcast over batch dimensions. Numeric scipy parameters are positional
inputs; boolean scipy modes are fixed to their scipy defaults (`skew`
biased, `kurtosis` Fisher + biased). `mean`/`variance` mirror
`numpy.mean`/`numpy.var` and are exposed as documented conveniences.

```python
import numpy as np
from ppstats import skew, sem, zscore

skew([1.0, 2.0, 3.0, 4.0, 10.0])          # 1.1384199576606167
sem([1.0, 2.0, 3.0, 4.0, 5.0], 1)         # 0.7071067811865476  (ddof=1, scipy default)
zscore(np.array([[1., 2., 3.], [4., 6., 8.]]), 0)   # batches broadcast
```

### Building

```bash
pixi install -e dev
pixi run -e dev test            # test suite (interpreted + compiled modes)
pixi run -e dev build-native    # plain C shared library + header + manifest
pixi run -e dev build-prefix    # libppstats layout under dist/prefix
pixi run -e dev build-ext       # ppstats_native, a NumPy-ufunc extension
```

When `ppstats_native` is importable, `import ppstats` prefers the compiled
gufuncs automatically (`ppstats.__native_available__`).

Primary compiler pressure this package generates: cross-package POST dependencies (ppspecial), reduction gufuncs.

## Start here

1. Read the [POST Python spec](https://github.com/openteams-ai/postpython/blob/main/docs/spec.md)
   and the [PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
   (package map, working rules, capability matrix).
2. Copy the layout of [ppspecial](https://github.com/openteams-ai/ppspecial),
   the exemplar package: `ppstats/` sources, `tests/`,
   `scripts/build_native.py`, `scripts/build_ext.py`, a pixi workspace with
   `test` / `build-native` / `build-ext` tasks, a git dependency on
   postpython, and a `ROADMAP.md` tracking targets and upstream requests.
3. Start with a slice from "Compiles today" below; land it as a small PR with
   tests in both execution modes.

## First slices

### Compiles today

- ~~Descriptive reductions as gufuncs `(n)->()`: mean, variance, skew, kurtosis, moment, gmean, hmean, sem; `zscore` as `(n)->(n)`~~ **Done** (ROADMAP Target 1)
- Distribution kernels built on ppspecial (a cross-package POST dependency — the first!): `norm` pdf/cdf/ppf via `ndtr`/`ndtri`, `logistic` via `expit`/`logit`, plus expon, uniform, laplace, cauchy — **next up** (ROADMAP Target 3)

### Blocked on compiler capabilities

File these as [postpython issues](https://github.com/openteams-ai/postpython/issues)
with minimal reproducers when you start on them — the filing is part of the
work and drives the compiler roadmap.

- `rvs` sampling — needs a POST RNG model (design discussion in postpython first)
- `fit` — needs ppoptimize / callable parameters
- `rv_continuous`-style class framework — needs structs

## Working rules (summary)

- Pure POST Python: no compiler-specific escape hatches; every kernel runs
  interpreted and compiled.
- `scipy` is the reference, never a runtime dependency. Tests may use it
  optionally; prefer deterministic hardcoded reference values.
- Compiler gaps go upstream as postpython issues with reproducers, not
  silent workarounds.
- Verify against a postpython checkout on `main`.
- Document accuracy targets and reference sources per function.

The full rules and the definition of done live in the
[PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md).
