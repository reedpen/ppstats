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
reproducers · verify against postpython `main` · accuracy is a documented deliverable ·
small reviewable landings · no binary wheels.

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

## Slice 2 (later, not now)

ppspecial-backed distributions (`norm`/`logistic`/`expon`/`uniform`/`laplace`/`cauchy`) in a
family module — forces the cross-package dependency on ppspecial (git dep + POST search_paths,
postpython #14). Own focused PR.
