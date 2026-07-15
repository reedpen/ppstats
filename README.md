# ppstats

Statistical functions and distributions, in POST Python.

`ppstats` reimplements `scipy.stats` in
[POST Python](https://github.com/openteams-ai/postpython) — every kernel is
fully-typed Python that runs under the standard CPython interpreter **and**
compiles ahead-of-time to native code (a plain C shared library and a NumPy
ufunc extension module) with the POST Python reference compiler.

Status: **Active** — descriptive reductions and six continuous distribution
families are verified in interpreted, native-library, and NumPy-ufunc modes.
Part of the
[PostSciPy effort](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
to rebuild SciPy one subpackage at a time as the compiler's proving ground.

## Implemented functions

| Family | Module | Functions |
|---|---|---|
| Descriptive | `_descriptive` | `mean`, `variance`, `gmean`, `hmean`, `moment(a, order)`, `skew`, `kurtosis`, `sem(a, ddof)`, `zscore(a, ddof)` |
| Continuous distributions | `_distributions` | `<name>_pdf`, `<name>_cdf`, `<name>_ppf` for `norm`, `logistic`, `expon`, `uniform`, `laplace`, and `cauchy` |

Descriptive functions are reduction gufuncs (`(n)->()`, `(n),()->()`, or
`(n),()->(n)`) that broadcast over batch dimensions. Numeric scipy
parameters are positional inputs; boolean scipy modes are fixed to their
scipy defaults (`skew` biased, `kurtosis` Fisher + biased).
`mean`/`variance` mirror `numpy.mean`/`numpy.var` and are exposed as
documented conveniences. Distribution functions are scalar ufuncs.

```python
import numpy as np
from ppstats import skew, sem, zscore

skew([1.0, 2.0, 3.0, 4.0, 10.0])          # 1.1384199576606167
sem([1.0, 2.0, 3.0, 4.0, 5.0], 1)         # 0.7071067811865476  (ddof=1, scipy default)
zscore(np.array([[1., 2., 3.], [4., 6., 8.]]), 0)   # batches broadcast
```

Distribution `loc` and `scale` parameters are explicit positional inputs and
broadcast like the value input:

```python
from ppstats import norm_cdf, logistic_ppf

norm_cdf(1.5, 0.5, 2.0)       # 0.69146247... (x, loc, scale)
logistic_ppf(0.8, 0.5, 2.0)   # 3.27258872... (q, loc, scale)
```

Normal CDF/PPF accuracy is inherited from `ppspecial` and validated to
`2e-7` relative on the reference cases; other distribution kernels match
SciPy references to `1e-12` relative on the test grid. Scales must be
positive. Until POST Python lowers IEEE infinity constants, unbounded PPF
endpoints use finite `±1e308` sentinels.

NumPy is an optional extra (`ppstats[numpy]`), but interpreted execution of
the descriptive **reduction** gufuncs currently requires it: postpyc 0.3.0's
no-numpy fallback calls gufunc kernels without their output buffer
(reproducer archived at `docs/issues/postpyc-nonumpy-gufunc-fallback.md`).
The scalar distribution ufuncs run interpreted without numpy.

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
- ~~Distribution kernels built on ppspecial (the first cross-package POST dependency): `norm`, `logistic`, `expon`, `uniform`, `laplace`, and `cauchy` pdf/cdf/ppf~~ **Done** (ROADMAP Target 3)

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
- Verify against released postpyc by default; use postpython `main` when
  chasing an unreleased compiler feature.
- Document accuracy targets and reference sources per function.

The full rules and the definition of done live in the
[PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md).
