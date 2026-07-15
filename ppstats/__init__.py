"""ppstats — POST Python reimplementation of scipy.stats.

Each function is a @guvectorize reduction kernel, written in fully-typed
POST Python. The postpyc compiler lowers them to native shared-library
code; in interpreted mode they run via the pure-Python broadcast loop.
When the optional `ppstats_native` extension module is installed next to
this package, matching public functions are replaced with native NumPy
gufuncs at import time.

Function families implemented
------------------------------
Descriptive (_descriptive) : mean, variance, gmean, hmean, moment,
                              skew, kurtosis, sem, zscore
Distributions (_distributions): norm, logistic, expon, uniform, laplace,
                                 cauchy pdf/cdf/ppf

Roadmap (not yet implemented)
-----------------------------
Blocked on compiler features : rvs (RNG model), fit (callable params),
                                describe (structs)
"""

from importlib import import_module as _import_module
from warnings import warn as _warn

from ppstats._descriptive import (
    mean,
    variance,
    gmean,
    hmean,
    moment,
    skew,
    kurtosis,
    sem,
    zscore,
)
from ppstats._distributions import (
    norm_pdf,
    norm_cdf,
    norm_ppf,
    logistic_pdf,
    logistic_cdf,
    logistic_ppf,
    expon_pdf,
    expon_cdf,
    expon_ppf,
    uniform_pdf,
    uniform_cdf,
    uniform_ppf,
    laplace_pdf,
    laplace_cdf,
    laplace_ppf,
    cauchy_pdf,
    cauchy_cdf,
    cauchy_ppf,
)

__all__ = [
    # descriptive
    "mean", "variance", "gmean", "hmean",
    "moment", "skew", "kurtosis", "sem", "zscore",
    # distributions
    "norm_pdf", "norm_cdf", "norm_ppf",
    "logistic_pdf", "logistic_cdf", "logistic_ppf",
    "expon_pdf", "expon_cdf", "expon_ppf",
    "uniform_pdf", "uniform_cdf", "uniform_ppf",
    "laplace_pdf", "laplace_cdf", "laplace_ppf",
    "cauchy_pdf", "cauchy_cdf", "cauchy_ppf",
]

__native_available__ = False
__native_module__ = None


def _prefer_native() -> None:
    """Prefer compiled ufuncs when a sibling native extension is installed."""
    global __native_available__, __native_module__

    try:
        native = _import_module("ppstats_native")
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "ppstats_native":
            return
        # Present but unloadable (missing numpy, ABI mismatch, ...) must
        # degrade to the interpreted kernels, not break `import ppstats`.
        _warn(
            f"ppstats_native is present but failed to import ({exc}); "
            "falling back to interpreted kernels",
            RuntimeWarning,
            stacklevel=3,
        )
        return

    replaced = []
    for name in __all__:
        if hasattr(native, name):
            globals()[name] = getattr(native, name)
            replaced.append(name)

    if replaced:
        __native_available__ = True
        __native_module__ = native


_prefer_native()

del _prefer_native, _import_module
