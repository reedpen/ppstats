# ppstats — CONTEXT

Working notes and decisions for building `ppstats` (scipy.stats in POST Python),
modeled on [ppspecial](../ppspecial). Governed by the
[PostSciPy roadmap](../postpython/postscipy-roadmap.md).

## Fixed facts (verified in-repo)

- `ppstats/` starts bare: `README.md` + `.gitignore` + git repo. Everything else is to build.
- Exemplar layout (ppspecial): `pp<name>/` kernel modules (`_family.py`) + `__init__.py`
  that re-exports and swaps in `pp<name>_native` at import when present; `tests/` per family
  plus native ABI/ext/default tests; `scripts/build_native.py` + `scripts/build_ext.py`;
  pixi workspace with `test` / `build-native` / `build-prefix` / `build-ext` tasks;
  git dep on postpyc; `ROADMAP.md`.
- postpython is on `main` (`7c5ed08`, v0.2.1).
- Reduction gufuncs compile **today**: `@guvectorize([], "(n)->()")`, `(n)->(n)`, and
  gufunc-calling-gufunc all work (see `postpython/examples/guvectorize_norm.py`).
- Cross-module linking works today (ppspecial `_stats` links `_erf`). Cross-**package**
  dependency resolution + wheel story is postpython issue #14 (not landed); use git dep +
  POST `search_paths` at build time until then.
- Kernel scalar math: `from postpyc.math import exp, log, log1p, fabs, ...`; scalar type
  `from postyp import f64`; arrays `from postyp import Array, Float64`.

## Working rules (from the roadmap)

Pure POST Python, no escape hatches · scipy is reference not a runtime dep (tests may use it
optionally; prefer hardcoded reference values) · compiler gaps filed upstream with minimal
reproducers · verify against released postpyc by default (`main` for unreleased features) ·
accuracy is a documented deliverable · small reviewable landings · no binary wheels.

## Decisions

- **D1 — First landing = scaffolding + descriptive reductions (option B).**
  PR #1 = full ppspecial-style harness (pyproject, pixi tasks, both build scripts, package
  `__init__` + native fallback, CI) *plus* the self-contained descriptive-reduction gufuncs.
  No dependency on ppspecial in slice 1. Distributions (which force the cross-package
  dependency, #14) become slice 2 with its own focused landing.

- **D2 — Parameter model = option (ii), matching ppspecial precedent.**
  scipy numeric parameters become ordinary positional gufunc inputs (like ppspecial's
  `polygamma(n: i64, x: f64)`): `moment(a, order)` `(n),()->()`, `sem(a, ddof)` `(n),()->()`,
  `zscore(a, ddof)` `(n),()->(n)`. Boolean *mode flags* (`bias`, `fisher`) are fixed to
  scipy's default in the primary kernel — matches the bare `scipy.stats.f(a)` call. Variant
  kernels (unbiased skew, Pearson kurtosis) deferred to a later slice. ppspecial has no
  flag precedent (scipy.special has no such kwargs), so the flag treatment is new ground.

- **D3 — Slice-1 function slate (confirmed).**
  `mean` `(n)->()`, `variance` `(n)->()`, `gmean` `(n)->()`, `hmean` `(n)->()`,
  `moment(a,order)` `(n),()->()`, `skew` `(n)->()` (Fisher–Pearson g1, biased),
  `kurtosis` `(n)->()` (Fisher/excess, biased), `sem(a,ddof)` `(n),()->()`,
  `zscore(a,ddof)` `(n),()->(n)`.
  - `mean`/`variance` are numpy names (not scipy.stats) but exposed **publicly** as
    documented convenience kernels — they're needed internally by moment/skew/kurtosis/zscore.
  - `gmean`/`hmean` drop scipy's `weights=` (unweighted only).
  - **Deferred to a later slice:** `describe` (namedtuple → needs structs, blocked),
    `variation`, `gstd`, `iqr`, trimmed stats (`tmean`/`tvar`/…).

- **D4 — Module layout follows ppspecial's precedent: family groupings with our own names,
    NOT scipy's internal filenames.** (ppspecial's `_erf`/`_gamma`/`_bessel`/`_stats` are
    family names it invented; scipy.special is Cython/C with no such files.)
  - Slice 1: single family module `ppstats/_descriptive.py` (all 9 reduction kernels).
  - `ppstats/__init__.py`: copy ppspecial's re-export + `_prefer_native()` pattern, renamed
    to `ppstats_native`.
  - `scripts/build_native.py` `EXPECTED_NATIVE = ["_descriptive"]`.
  - Slice 2 distributions module gets a *family* name we pick (`_distributions.py` /
    `_continuous.py`) — not a copy of scipy's `_continuous_distns.py`.

## Test/verification precedent (from ppspecial — follow verbatim, not grilled)

- Reference values are **hardcoded deterministic constants** with a comment citing the scipy
  call that produced them; no runtime scipy import. numpy only `importorskip`'d in native-ext
  tests. Plus identity/symmetry checks (e.g. zscore mean≈0, std≈1).
- ppspecial ships **no CI** (roadmap Target 11 is aspirational; no `.github/` exists).

- **D5 — Dependencies mirror ppspecial; build against PyPI postpyc 0.3.0 by default.**
  `[project].dependencies = ["postpyc>=0.3.0", "postyp>=0.3.0"]`; pixi `pypi-dependencies`
  = `ppstats` editable; `[tool.pixi.dependencies]` = `python>=3.10` + `c-compiler`;
  `dev` feature = `pytest>=7` + `numpy`; same pixi tasks. No ppspecial dep in slice 1
  (added as a git dep in slice 2). Local `../postpython` checkout (v0.2.1, *behind* PyPI
  0.3.0) held in reserve for reproducers / unreleased-feature testing only.

- **D6 — No CI in slice 1 (option a); metadata mirrors ppspecial.**
  CI deferred to its own later landing (no exemplar to copy yet). pyproject: `name="ppstats"`,
  `version="0.1.0"`, `requires-python=">=3.10"`, ppspecial's classifiers, Repository
  `https://github.com/openteams-ai/ppstats`.
  `authors = [{name="Travis E. Oliphant"}, {name="Reed A. Pennock"}]`.

## Slice-1 deliverable (synthesis)

PR #1 = ppspecial-style harness + descriptive-reduction gufuncs:
- `ppstats/_descriptive.py`: `mean`, `variance`, `gmean`, `hmean`, `moment(a,order)`,
  `skew`, `kurtosis`, `sem(a,ddof)`, `zscore(a,ddof)` — all `@guvectorize`, flags fixed to
  scipy defaults, numeric params as `()` inputs.
- `ppstats/__init__.py`: re-export + `_prefer_native()` → `ppstats_native`.
- `scripts/build_native.py` (`EXPECTED_NATIVE=["_descriptive"]`) + `scripts/build_ext.py`.
- `pyproject.toml` (pixi workspace, tasks, deps per D5, metadata per D6).
- `tests/`: `test_descriptive.py` (hardcoded reference values + identity checks),
  `test_native_abi.py`, `test_native_ext.py`, `test_native_default.py` (adapted from ppspecial).
- `ROADMAP.md` (targets + upstream requests), README already exists.
- Definition of done (roadmap): interpreted tests pass · native `.so` builds · NumPy-ufunc
  ext builds · accuracy documented.

## Slice-1 implementation findings (2026-07-13)

- **Compiler bug found: `postpyc.math.NAN`/`INF` fail to lower** — emitted via `repr()`
  as bare `nan`/`inf` C identifiers; `nan` collides with libm's `double nan(const char*)`
  → C compile error. Interpreted works. **Filed as
  [postpython#36](https://github.com/openteams-ai/postpython/issues/36)** (reproducer
  archived at `docs/issues/postpython-nan-lowering.md`).
  Workaround: NaN produced arithmetically `(a[i]-a[i])/(a[i]-a[i])`; revert when fixed.
- Interpreted gufuncs effectively require numpy: the no-numpy fallback calls the kernel
  without its `out` parameter, so list inputs route through numpy → elements are
  `np.float64` → 0/0 quietly gives nan (keeps both modes equivalent).
- `skew` uses `m3 / (m2*sqrt(m2))`, `kurtosis` `m4/(m2*m2) - 3` — avoids `**`.
- **Plain (non-decorated) POST helpers with array parameters compile fine** and are
  callable from gufunc kernels: `_mean(a: Array[Float64]) -> Float64` / `_sumsq(a, m)`
  deduplicate the mean/sum-of-squares loops across the seven kernels that need them.
- `moment` computes negative orders as reciprocal powers — matches scipy 1.18 (which
  *computes* them, it does not raise): `moment([1,2,3], -1)` → inf on a zero deviation.
- gmean/hmean scan for negatives **before** zeros so `gmean([0.0, -1.0])` → nan like
  scipy (a first-zero early return would wrongly give 0.0). Caught by /code-review.
- `sem`/`zscore` compute `n` as Float64 (`1.0 * len(a)`) so ddof ≥ n gives inf/nan
  rather than an integer-division divergence.
- Reference values generated with scipy 1.18.0 in a scratch venv; hardcoded per test
  with the producing call cited. Suite: 73 tests, both modes, zero warnings.
- Verified against PyPI postpyc 0.3.0. All four pixi tasks green
  (test / build-native / build-prefix / build-ext; ext registers all 9 as numpy.ufunc).

## Slice 2 implementation findings (2026-07-15)

- Added `_distributions.py` with 18 scalar ufuncs: pdf/cdf/ppf for
  `norm`, `logistic`, `expon`, `uniform`, `laplace`, and `cauchy`.
- Public names use `<family>_<method>` and take `(x_or_q, loc, scale)`;
  all three inputs broadcast. This applies D2's explicit numeric-parameter
  model to distribution parameters.
- `norm_cdf`/`norm_ppf` call ppspecial `ndtr`/`ndtri`; logistic CDF/PPF
  call `expit`/`logit`. ppspecial v0.1.2 is a Git dependency pinned at
  `435ecfe`, and build/test entry points discover its installed source root
  for POST `search_paths` (postpython #14 workaround).
- Normal CDF/PPF references pass at 2e-7 relative (inherited erfc
  approximation); direct-formula kernels pass at 1e-12. Interpreted and
  compiled modes agree at 1e-13 on the test grid.
- Current preconditions are positive scale and q in [0,1]. Unbounded PPF
  endpoints use ±1e308, consistent with ppspecial, until postpython#36
  allows IEEE infinity constants. Full invalid-parameter behavior remains
  Target 2 compatibility-harness work.

## Slice 2 review fixes (2026-07-15)

- **`uniform_pdf` NaN hole fixed.** The support test written on raw `x`
  (`x < loc or x > loc + scale`) compares False for NaN in any argument and
  fell through to `1/scale` — the only kernel of 18 not propagating NaN.
  Now standardizes to `z = (x-loc)/scale` first (scipy's structure) and
  guards with `postpyc.math.isnan` (compiles fine; #36 is only about the
  NAN/INF *constants*). `TestNanPropagation` pins all 18 kernels across all
  three argument positions.
- **Out-of-range q is pinned, not fixed.** `q <= 0` / `q >= 1` guards fold
  invalid q into the endpoint result (scipy returns NaN); can't do better
  until #36 lets kernels produce NaN. `TestPpfOutOfRangeQ` documents this.
  NaN in loc/scale at *sentinel endpoints* (e.g. `laplace_ppf(0.0, nan, s)`)
  also returns the sentinel — same debt.
- **Reference suites now import the family modules directly**
  (`ppstats._distributions` / `ppstats._descriptive`). Importing the package
  root let a stale `ppstats_native*.so` in the repo root silently substitute
  compiled kernels into the "interpreted" test task.
- Known mode divergence outside the documented `scale > 0` boundary:
  `scale = 0` raises ZeroDivisionError interpreted but returns NaN (with a
  RuntimeWarning) compiled. Left as-is; Target 2 scope.
- `build_prefix.py` now catches `BuildError` like its sibling scripts.

## Slice 2 review fixes, round 2 (2026-07-15)

- **`moment` orders 0 and 1 now return 1.0/0.0 exactly** (scipy's shortcut).
  Numerically summing deviations left cancellation residue —
  `moment([1e20, 1, 1], 1)` returned 2730.67. Early `return` inside a
  guvectorize kernel compiles fine.
- **A present-but-unloadable `ppstats_native` no longer breaks
  `import ppstats`.** The guard only caught `ModuleNotFoundError` for
  `ppstats_native` itself; an extension built against numpy in a numpy-free
  env (or any ABI mismatch) propagated ImportError out of `import ppstats`.
  Now warns and falls back to interpreted kernels. ppspecial's `__init__`
  has the same narrow guard — worth an upstream PR.
- **postpyc 0.3.0's no-numpy interpreted fallback cannot run gufuncs**
  (`ufunc.py` `except ImportError: return fn(*args)` drops the `out`
  parameter). Every descriptive reduction is unusable interpreted without
  numpy; scalar `@vectorize` distribution kernels are fine. Not fixable in
  ppstats without escape hatches. Reproducer drafted at
  `docs/issues/postpyc-nonumpy-gufunc-fallback.md` (not yet filed); numpy
  stays an optional extra (ppspecial parity), README documents the boundary.
- **PPF sentinel policy centralized** in `_unbounded_endpoint(q)` — plain
  POST helper, compiles when called from `@vectorize` kernels; single point
  of change when postpython#36 lands.
- `tests/test_native_ext.py` builds its throwaway extension with
  `sysconfig.get_config_var("EXT_SUFFIX")` instead of a hardcoded `.so`
  (Windows needs `.pyd` for `spec_from_file_location` to pick an extension
  loader).

## Benchmark findings (2026-07-16, ../ppstats-bench)

Standalone scipy-comparison harness (pixi env with scipy 1.18; separate
project because scipy stays out of this repo's deps). All 27 kernels match
scipy in both modes under an `atol + rtol*|ref|` comparison.

- **Accuracy docs corrected: the ndtr/ndtri-inherited error is absolute,
  not relative** (<1.2e-7 CDF; ~2.5e-7·scale PPF on a dense q grid vs
  scipy 1.18). `norm_ppf` crosses zero at `q = ndtr(-loc/scale)`, where an
  absolute error of ~2.6e-7 measured as 7.8e-3 *relative* — a pure-rtol
  check spuriously fails there. README, `_distributions.py` docstring, and
  ROADMAP now state the absolute error model; pinned by
  `TestNorm.test_ppf_error_is_absolute_near_the_zero_crossing`. Target 2's
  accuracy harness must use atol+rtol comparisons for all ppf kernels.
- Perf vs base scipy (native, linux-64): `skew`/`kurtosis`/`moment` 32–48×
  faster at every size; distributions 1.2–10× at N=1e6 (overhead-dominated
  2–21× at N=1e3). Losses worth Target-2/upstream attention: `mean` 0.4×
  on long vectors (numpy's pairwise summation is SIMD-optimized — feeds the
  existing summation decision), `norm_ppf` 0.7× (scipy's ndtri is faster
  *and* ~1e-9-accurate; candidate ppspecial upstream: AS241-style ndtri),
  `gmean` 0.8× at small batches.
