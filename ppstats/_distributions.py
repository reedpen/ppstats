"""Continuous probability distributions — ppstats._distributions.

Public API
----------
Each family exposes ``<name>_pdf(x, loc, scale)``,
``<name>_cdf(x, loc, scale)``, and ``<name>_ppf(q, loc, scale)`` for normal,
logistic, exponential, uniform, Laplace, and Cauchy distributions.  They are
scalar ``@vectorize`` kernels, so values, locations, and scales broadcast.

Algorithms and accuracy
-----------------------
Normal CDF/PPF delegate to ``ppspecial.ndtr`` / ``ppspecial.ndtri`` and
logistic CDF/PPF to ``ppspecial.expit`` / ``ppspecial.logit``.  This is
ppstats' first cross-package POST dependency.  The normal CDF inherits
ppspecial's erfc approximation (<1.2e-7 absolute error); normal CDF/PPF are
tested to 2e-7 relative on the reference cases.  All PDFs and the remaining
CDF/PPF formulas use direct libm identities and match scipy 1.18.0 references
to 1e-12 relative on the test grid.

Domain conventions
------------------
``scale`` must be positive and PPF probabilities must lie in ``[0, 1]``.
Invalid parameters are reserved for the compatibility-harness slice.  Until
the compiler lowers IEEE infinity constants (postpython#36), unbounded PPF
endpoints use finite sentinels of magnitude ``1e308``, matching ppspecial's
current convention.  Uniform endpoints remain finite by definition.
"""

from postyp import f64
from postpyc import vectorize
from postpyc.math import PI, atan, exp, expm1, fabs, isnan, log, log1p, tan

# Import from ppspecial._stats rather than the package root: ppspecial's
# __init__ swaps in native ufuncs when ppspecial_native is installed, while
# both interpreted execution and the POST compiler need the kernel source.
from ppspecial._stats import expit, logit, ndtr, ndtri


__all__ = [
    "norm_pdf", "norm_cdf", "norm_ppf",
    "logistic_pdf", "logistic_cdf", "logistic_ppf",
    "expon_pdf", "expon_cdf", "expon_ppf",
    "uniform_pdf", "uniform_cdf", "uniform_ppf",
    "laplace_pdf", "laplace_cdf", "laplace_ppf",
    "cauchy_pdf", "cauchy_cdf", "cauchy_ppf",
]


def _unbounded_endpoint(q: f64) -> f64:
    """Sentinel PPF value at q <= 0 / q >= 1 for unbounded-support families.

    ±1e308 stands in for -inf/+inf until postpython#36 lets kernels lower
    IEEE infinity constants; when it lands, this is the one place to fix.
    """
    if q <= 0.0:
        return -1.0e308
    return 1.0e308


@vectorize
def norm_pdf(x: f64, loc: f64, scale: f64) -> f64:
    """Normal probability density with explicit location and scale."""
    z: f64 = (x - loc) / scale
    inv_sqrt_2pi: f64 = 0.3989422804014327
    return inv_sqrt_2pi * exp(-0.5 * z * z) / scale


@vectorize
def norm_cdf(x: f64, loc: f64, scale: f64) -> f64:
    """Normal cumulative distribution via ppspecial.ndtr."""
    return ndtr((x - loc) / scale)


@vectorize
def norm_ppf(q: f64, loc: f64, scale: f64) -> f64:
    """Normal percent-point function via ppspecial.ndtri."""
    if q <= 0.0 or q >= 1.0:
        return _unbounded_endpoint(q)
    return loc + scale * ndtri(q)


@vectorize
def logistic_pdf(x: f64, loc: f64, scale: f64) -> f64:
    """Logistic probability density with explicit location and scale."""
    z: f64 = (x - loc) / scale
    return expit(z) * expit(-z) / scale


@vectorize
def logistic_cdf(x: f64, loc: f64, scale: f64) -> f64:
    """Logistic cumulative distribution via ppspecial.expit."""
    return expit((x - loc) / scale)


@vectorize
def logistic_ppf(q: f64, loc: f64, scale: f64) -> f64:
    """Logistic percent-point function via ppspecial.logit."""
    if q <= 0.0 or q >= 1.0:
        return _unbounded_endpoint(q)
    return loc + scale * logit(q)


@vectorize
def expon_pdf(x: f64, loc: f64, scale: f64) -> f64:
    """Exponential probability density on ``x >= loc``."""
    if x < loc:
        return 0.0
    return exp(-(x - loc) / scale) / scale


@vectorize
def expon_cdf(x: f64, loc: f64, scale: f64) -> f64:
    """Exponential cumulative distribution on ``x >= loc``."""
    if x < loc:
        return 0.0
    return -expm1(-(x - loc) / scale)


@vectorize
def expon_ppf(q: f64, loc: f64, scale: f64) -> f64:
    """Exponential percent-point function."""
    if q <= 0.0:
        return loc
    if q >= 1.0:
        return _unbounded_endpoint(q)
    return loc - scale * log1p(-q)


@vectorize
def uniform_pdf(x: f64, loc: f64, scale: f64) -> f64:
    """Uniform probability density on ``loc <= x <= loc + scale``."""
    # Standardize first (as scipy does): a support test written directly on
    # x compares False for NaN in any argument and would fall through to the
    # in-support density instead of propagating NaN.
    z: f64 = (x - loc) / scale
    if isnan(z):
        return z
    if z < 0.0 or z > 1.0:
        return 0.0
    return 1.0 / scale


@vectorize
def uniform_cdf(x: f64, loc: f64, scale: f64) -> f64:
    """Uniform cumulative distribution."""
    if x <= loc:
        return 0.0
    if x >= loc + scale:
        return 1.0
    return (x - loc) / scale


@vectorize
def uniform_ppf(q: f64, loc: f64, scale: f64) -> f64:
    """Uniform percent-point function."""
    if q <= 0.0:
        return loc
    if q >= 1.0:
        return loc + scale
    return loc + scale * q


@vectorize
def laplace_pdf(x: f64, loc: f64, scale: f64) -> f64:
    """Laplace probability density with explicit location and scale."""
    return 0.5 * exp(-fabs((x - loc) / scale)) / scale


@vectorize
def laplace_cdf(x: f64, loc: f64, scale: f64) -> f64:
    """Laplace cumulative distribution."""
    z: f64 = (x - loc) / scale
    if z < 0.0:
        return 0.5 * exp(z)
    return 1.0 - 0.5 * exp(-z)


@vectorize
def laplace_ppf(q: f64, loc: f64, scale: f64) -> f64:
    """Laplace percent-point function."""
    if q <= 0.0 or q >= 1.0:
        return _unbounded_endpoint(q)
    if q < 0.5:
        return loc + scale * log(2.0 * q)
    return loc - scale * log(2.0 * (1.0 - q))


@vectorize
def cauchy_pdf(x: f64, loc: f64, scale: f64) -> f64:
    """Cauchy probability density with explicit location and scale."""
    z: f64 = (x - loc) / scale
    return 1.0 / (PI * scale * (1.0 + z * z))


@vectorize
def cauchy_cdf(x: f64, loc: f64, scale: f64) -> f64:
    """Cauchy cumulative distribution."""
    return 0.5 + atan((x - loc) / scale) / PI


@vectorize
def cauchy_ppf(q: f64, loc: f64, scale: f64) -> f64:
    """Cauchy percent-point function."""
    if q <= 0.0 or q >= 1.0:
        return _unbounded_endpoint(q)
    return loc + scale * tan(PI * (q - 0.5))
