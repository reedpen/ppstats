"""Descriptive statistics — ppstats._descriptive

Public API
----------
mean(a)         — arithmetic mean                       (n)->()
variance(a)     — population variance (ddof = 0)        (n)->()
gmean(a)        — geometric mean                        (n)->()
hmean(a)        — harmonic mean                         (n)->()
moment(a, k)    — k-th central moment, integer k        (n),()->()
skew(a)         — Fisher–Pearson g1, biased             (n)->()
kurtosis(a)     — Fisher (excess) kurtosis, biased      (n)->()
sem(a, ddof)    — standard error of the mean            (n),()->()
zscore(a, ddof) — standardized values                   (n),()->(n)

Conventions
-----------
Boolean scipy modes are fixed to their scipy defaults: ``skew`` is the
biased Fisher–Pearson coefficient (``scipy.stats.skew(a)``), ``kurtosis``
is the biased Fisher/excess definition (``scipy.stats.kurtosis(a)``).
Numeric scipy parameters are ordinary positional inputs (like
ppspecial's ``polygamma(n, x)``): ``moment`` takes its order, ``sem``
and ``zscore`` take ``ddof``. scipy defaults are ``sem(a, 1)`` and
``zscore(a, 0)``.

``mean`` and ``variance`` mirror ``numpy.mean`` / ``numpy.var`` (scipy
has no top-level equivalents); they are exposed as public conveniences
and used as building blocks throughout.

Algorithms
----------
Definitional formulas with two-pass central moments (mean first, then
deviations) and naive left-to-right summation; worst-case rounding
grows O(n·ε) with vector length. skew and kurtosis accumulate their
second/third/fourth powers in a single deviation pass.

Domain behavior
---------------
gmean/hmean: any negative element yields NaN; otherwise any zero
element yields 0.0 (matching scipy, including mixed zero/negative
input). moment computes negative orders as reciprocal powers, as
scipy does. skew/kurtosis/zscore of a constant vector are NaN.
NaN is produced arithmetically (0/0) because the reference compiler
cannot yet lower ``postpyc.math.NAN`` (upstream issue; see ROADMAP).
"""

from postyp import Array, Float64, Int64
from postpyc import guvectorize
from postpyc.math import exp, log, sqrt


# ---------------------------------------------------------------------------
# Shared helpers (plain POST functions, compiled with the kernels)
# ---------------------------------------------------------------------------

def _mean(a: Array[Float64]) -> Float64:
    """Arithmetic mean of a 1-D view."""
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += a[i]
    return acc / len(a)


def _sumsq(a: Array[Float64], m: Float64) -> Float64:
    """Sum of squared deviations from m."""
    acc: Float64 = 0.0
    for i in range(len(a)):
        d: Float64 = a[i] - m
        acc += d * d
    return acc


# ---------------------------------------------------------------------------
# Means
# ---------------------------------------------------------------------------

@guvectorize([], "(n)->()")
def mean(a: Array[Float64], out: Array[Float64]) -> None:
    """Arithmetic mean: sum(a) / n.  Mirrors numpy.mean."""
    out[0] = _mean(a)


@guvectorize([], "(n)->()")
def gmean(a: Array[Float64], out: Array[Float64]) -> None:
    """Geometric mean: exp(mean(log(a))).

    Matches scipy.stats.gmean for the unweighted case: any negative
    element gives NaN; otherwise any zero element gives 0.0.
    """
    for i in range(len(a)):
        if a[i] < 0.0:
            out[0] = (a[i] - a[i]) / (a[i] - a[i])   # NaN
            return
    for i in range(len(a)):
        if a[i] == 0.0:
            out[0] = 0.0
            return
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += log(a[i])
    out[0] = exp(acc / len(a))


@guvectorize([], "(n)->()")
def hmean(a: Array[Float64], out: Array[Float64]) -> None:
    """Harmonic mean: n / sum(1/a).

    Matches scipy.stats.hmean for the unweighted case: any negative
    element gives NaN; otherwise any zero element gives 0.0.
    """
    for i in range(len(a)):
        if a[i] < 0.0:
            out[0] = (a[i] - a[i]) / (a[i] - a[i])   # NaN
            return
    for i in range(len(a)):
        if a[i] == 0.0:
            out[0] = 0.0
            return
    acc: Float64 = 0.0
    for i in range(len(a)):
        acc += 1.0 / a[i]
    out[0] = len(a) / acc


# ---------------------------------------------------------------------------
# Central moments
# ---------------------------------------------------------------------------

@guvectorize([], "(n)->()")
def variance(a: Array[Float64], out: Array[Float64]) -> None:
    """Population variance (ddof = 0): mean((a - mean(a))**2).

    Mirrors numpy.var; equals moment(a, 2). Two-pass algorithm.
    """
    m: Float64 = _mean(a)
    out[0] = _sumsq(a, m) / len(a)


@guvectorize([], "(n),()->()")
def moment(a: Array[Float64], order: Int64, out: Array[Float64]) -> None:
    """k-th central moment: mean((a - mean(a))**k) for integer k.

    moment(a, 0) == 1.0 and moment(a, 1) == 0.0 by definition; like scipy,
    these are returned exactly rather than computed (numerically summing
    the deviations leaves cancellation residue, e.g. for [1e20, 1, 1]).
    Negative orders are reciprocal powers, as in scipy: a zero deviation
    then yields inf/NaN.
    """
    if order == 0:
        out[0] = 1.0
        return
    if order == 1:
        out[0] = 0.0
        return
    m: Float64 = _mean(a)
    acc: Float64 = 0.0
    for i in range(len(a)):
        d: Float64 = a[i] - m
        p: Float64 = 1.0
        if order < 0:
            for _ in range(0 - order):
                p *= d
            acc += 1.0 / p
        else:
            for _ in range(order):
                p *= d
            acc += p
    out[0] = acc / len(a)


# ---------------------------------------------------------------------------
# Shape statistics
# ---------------------------------------------------------------------------

@guvectorize([], "(n)->()")
def skew(a: Array[Float64], out: Array[Float64]) -> None:
    """Fisher–Pearson coefficient of skewness g1 = m3 / m2**1.5 (biased).

    Matches scipy.stats.skew(a) with the default bias=True. A constant
    input yields NaN (0/0).
    """
    m: Float64 = _mean(a)
    m2: Float64 = 0.0
    m3: Float64 = 0.0
    for i in range(len(a)):
        d: Float64 = a[i] - m
        d2: Float64 = d * d
        m2 += d2
        m3 += d2 * d
    m2 = m2 / len(a)
    m3 = m3 / len(a)
    out[0] = m3 / (m2 * sqrt(m2))


@guvectorize([], "(n)->()")
def kurtosis(a: Array[Float64], out: Array[Float64]) -> None:
    """Fisher (excess) kurtosis g2 = m4 / m2**2 - 3 (biased).

    Matches scipy.stats.kurtosis(a) with the defaults fisher=True,
    bias=True. A constant input yields NaN (0/0).
    """
    m: Float64 = _mean(a)
    m2: Float64 = 0.0
    m4: Float64 = 0.0
    for i in range(len(a)):
        d: Float64 = a[i] - m
        d2: Float64 = d * d
        m2 += d2
        m4 += d2 * d2
    m2 = m2 / len(a)
    m4 = m4 / len(a)
    out[0] = m4 / (m2 * m2) - 3.0


# ---------------------------------------------------------------------------
# Standard error / standardization
# ---------------------------------------------------------------------------

@guvectorize([], "(n),()->()")
def sem(a: Array[Float64], ddof: Int64, out: Array[Float64]) -> None:
    """Standard error of the mean: sqrt(var(a, ddof) / n).

    scipy default is sem(a, 1); ddof = 0 gives the population form.
    """
    m: Float64 = _mean(a)
    ss: Float64 = _sumsq(a, m)
    n: Float64 = 1.0 * len(a)
    out[0] = sqrt(ss / (n - ddof) / n)


@guvectorize([], "(n),()->(n)")
def zscore(a: Array[Float64], ddof: Int64, out: Array[Float64]) -> None:
    """Standardized values: (a - mean(a)) / std(a, ddof).

    scipy default is zscore(a, 0). A constant input yields NaN (0/0).
    """
    m: Float64 = _mean(a)
    ss: Float64 = _sumsq(a, m)
    n: Float64 = 1.0 * len(a)
    s: Float64 = sqrt(ss / (n - ddof))
    for i in range(len(a)):
        out[i] = (a[i] - m) / s
