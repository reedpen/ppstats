# ppstats / postpyc Roadmap

This roadmap is the coordination document for building `ppstats` as a
POST Python implementation of `scipy.stats`, following the
[PostSciPy roadmap](https://github.com/openteams-ai/postpython/blob/main/postscipy-roadmap.md)
and the [ppspecial](https://github.com/openteams-ai/ppspecial) exemplar
layout.

Primary compiler pressure this package generates: reduction gufuncs
(`(n)->()`), and — starting with Target 3 — cross-**package** POST
dependencies (ppstats → ppspecial).

## Working Rules

- Keep `ppstats` source free of compiler-specific escape hatches; every
  kernel runs interpreted and compiles with the reference compiler.
- scipy is the reference, never a runtime dependency. Tests hardcode
  deterministic reference values with a comment citing the producing
  scipy call.
- Compiler gaps go upstream as postpython issues with minimal
  reproducers; reference them from this file.
- Default verification target is the released `postpyc` from PyPI
  (currently 0.3.0); use a postpython checkout on `main` only when
  chasing unreleased features, and confirm the branch first.
- Each target lands as a reviewable PR with tests in both execution
  modes and a status update here.

## Status Legend

- `Done` / `Active` / `Ready` / `Blocked` / `Later` (as in ppspecial).

## Target 0: Baseline Package Health

Status: `Active`

Acceptance criteria:

- `pixi run -e dev test` passes (interpreted + compiled-extension tests).
- `pixi run -e dev build-native` reports `_descriptive`, `_distributions`,
  and the full package building natively with
  header/manifest sidecars.
- `pixi run -e dev build-ext` registers every public function as a
  `numpy.ufunc`.

## Target 1: Descriptive Reductions

Status: `Done` (verified against postpyc 0.3.0 from PyPI)

Functions: `mean`, `variance`, `gmean`, `hmean`, `moment(a, order)`,
`skew`, `kurtosis`, `sem(a, ddof)` — all `(n)->()` or `(n),()->()` —
and `zscore(a, ddof)` as `(n),()->(n)`.

Conventions (decided in CONTEXT.md):

- Numeric scipy parameters are positional gufunc inputs (ppspecial's
  `polygamma(n, x)` precedent). Boolean scipy modes are fixed to their
  scipy defaults: `skew` biased, `kurtosis` Fisher + biased.
- `mean`/`variance` mirror numpy (scipy has no top-level equivalents)
  and are exposed as documented conveniences.
- `gmean`/`hmean` are unweighted (scipy's `weights=` dropped for now).

Accuracy:

- Definitional formulas, two-pass central moments, naive summation:
  worst-case error grows O(n·ε) with vector length, versus numpy's
  pairwise summation. On the test vectors (n ≤ 6) everything matches
  scipy 1.18.0 references to ≤ 1e-12 relative; compiled and interpreted
  modes agree to ≤ 1e-13 relative on batch grids.
- Domain behavior documented in `_descriptive.py`: gmean/hmean yield
  0.0 for a zero element and NaN for a negative element; skew/kurtosis/
  zscore of a constant vector yield NaN — all matching scipy.

Deferred variants: unbiased `skew`/`kurtosis` (`bias=False`), Pearson
kurtosis, weighted g/h-means, `nan_policy`.

## Target 2: Accuracy and Compatibility Harness

Status: `Ready`

Mirror ppspecial's Target 3: per-function accuracy table, optional
scipy-backed comparison mode, compensated/pairwise summation decision
for long vectors, edge-case coverage (±0, infinities, NaN propagation,
single-element and empty vectors).

## Target 3: Distributions via ppspecial (cross-package POST)

Status: `Done` (verified against postpyc 0.3.0 + ppspecial 0.1.2)

The first cross-**package** POST dependency in the
ecosystem; exercises postpython
[#14](https://github.com/openteams-ai/postpython/issues/14) (dependency
resolution) and [#13](https://github.com/openteams-ai/postpython/issues/13)
(cross-module inlining).

Implemented as 18 scalar ufuncs in `_distributions`: `norm` pdf/cdf/ppf
via ppspecial's `ndtr`/`ndtri`, `logistic` via `expit`/`logit`, plus
`expon`, `uniform`, `laplace`, and `cauchy`. Every kernel takes explicit
`loc` and `scale` positional inputs. ppspecial 0.1.2 is pinned as a Git
dependency and all build targets pass its installed source root as a POST
`search_path` until #14 lands.

Accuracy: normal CDF inherits ppspecial's erfc error bound (<1.2e-7
absolute), and normal CDF/PPF match scipy 1.18.0 references within 2e-7
relative on the test cases. Direct-formula PDFs and all other CDF/PPFs
match within 1e-12 relative. Compiled and interpreted modes agree within
1e-13 relative across the test grid.

Compatibility boundary: `scale > 0` and `q in [0, 1]` are current
preconditions. Unbounded PPF endpoints use ±1e308 sentinels until
postpython#36 permits IEEE infinities. Invalid-parameter/NaN behavior is
part of Target 2.

## Target 4: Rank and Order Statistics

Status: `Later`

`iqr`, `median_abs_deviation`, `percentileofscore`, trimmed statistics
(`tmean`, `tvar`, …). Needs sorting/selection inside kernels — expect
to generate compiler pressure on local array workspaces (spec §7).

## Target 5: Correlation

Status: `Later`

`pearsonr` is expressible today as `(n),(n)->()`; result objects
(`statistic`, `pvalue`) want structs, and p-values want Target 3's
distribution CDFs first.

## Blocked on named compiler capabilities

File the postpython issue with a minimal reproducer when starting each:

- `rvs` sampling — needs a POST RNG model (design discussion first).
- `fit` — needs callable parameters (ppoptimize is the lead driver).
- `describe` / `rv_continuous`-style framework — needs structs
  (`@dataclass` → C struct).

## postpython Request Backlog

Needs discovered from ppstats. Items become issues in
`openteams-ai/postpython` when actively worked.

### Found during Target 1

- **`postpyc.math.NAN` / `INF` constants fail to lower**: the C backend
  emits them via `repr()` as the bare identifiers `nan` / `inf`;
  `nan` collides with libm's `double nan(const char *)` and fails to
  compile (`error: incompatible types when initializing type 'double'`).
  Interpreted mode works, so this is an interpreted/compiled divergence
  for valid POST source. Minimal reproducer: a `(n)->()` gufunc whose
  body does `out[0] = NAN; return`. Workaround in `_descriptive.py`:
  NaN produced arithmetically as `(a[i]-a[i])/(a[i]-a[i])`; revert when
  fixed. **Filed:**
  [postpython#36](https://github.com/openteams-ai/postpython/issues/36).

### Expected from Target 3

- Cross-package dependency resolution and wheel story
  ([#14](https://github.com/openteams-ai/postpython/issues/14)).
- Cross-module/-package inlining for hot inner calls into ppspecial
  ([#13](https://github.com/openteams-ai/postpython/issues/13)).

## Publication Checklist for Each Milestone

Before marking a target `Done`, publish: source changes, tests,
interpreted test output, native build output, any postpython
issue/request generated by the work, and a status update here. For
compiler-dependent claims, include the postpyc version or commit used
for verification.
