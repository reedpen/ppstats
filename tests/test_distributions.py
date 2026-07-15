"""Continuous distribution kernels against scipy 1.18.0 references.

Imports come from ``ppstats._distributions`` (not the package root) so these
tests always exercise the interpreted kernel source: the root namespace swaps
in ``ppstats_native`` when a built extension is importable, which would make
this suite silently validate a possibly stale binary instead.  Compiled
behavior is covered by test_native_ext.py and test_native_abi.py.
"""

import math
import pytest

from ppstats._distributions import (
    cauchy_cdf,
    cauchy_pdf,
    cauchy_ppf,
    expon_cdf,
    expon_pdf,
    expon_ppf,
    logistic_cdf,
    logistic_pdf,
    logistic_ppf,
    laplace_cdf,
    laplace_pdf,
    laplace_ppf,
    norm_cdf,
    norm_pdf,
    norm_ppf,
    uniform_cdf,
    uniform_pdf,
    uniform_ppf,
)


def close(got, expected, *, rtol=1e-12, atol=1e-14):
    return got == pytest.approx(expected, rel=rtol, abs=atol)


class TestNorm:
    def test_pdf_cdf_ppf_with_location_and_scale(self):
        # scipy.stats.norm.pdf/cdf(1.5, loc=0.5, scale=2.0)
        assert close(norm_pdf(1.5, 0.5, 2.0), 0.17603266338214976)
        # ppspecial.ndtr inherits erfc's <1.2e-7 absolute error bound.
        assert close(norm_cdf(1.5, 0.5, 2.0), 0.6914624612740131, rtol=2e-7)

        # scipy.stats.norm.ppf(0.8, loc=0.5, scale=2.0)
        assert close(norm_ppf(0.8, 0.5, 2.0), 2.1832424671458286, rtol=2e-7)

    def test_standard_distribution_identities(self):
        assert close(norm_pdf(0.0, 0.0, 1.0), 0.3989422804014327)
        assert norm_cdf(0.0, 0.0, 1.0) == 0.5
        assert close(norm_ppf(norm_cdf(1.25, 0.0, 1.0), 0.0, 1.0), 1.25, rtol=1e-9)


class TestLogistic:
    def test_pdf_cdf_ppf_with_location_and_scale(self):
        # scipy.stats.logistic.pdf/cdf(1.5, loc=0.5, scale=2.0)
        assert close(logistic_pdf(1.5, 0.5, 2.0), 0.11750185610079725)
        assert close(logistic_cdf(1.5, 0.5, 2.0), 0.6224593312018546)

        # scipy.stats.logistic.ppf(0.8, loc=0.5, scale=2.0)
        assert close(logistic_ppf(0.8, 0.5, 2.0), 3.2725887222397816)

    def test_cdf_ppf_roundtrip(self):
        q = logistic_cdf(-1.25, 0.25, 1.5)
        assert close(logistic_ppf(q, 0.25, 1.5), -1.25)


class TestExpon:
    def test_pdf_cdf_ppf_with_location_and_scale(self):
        # scipy.stats.expon.pdf/cdf(1.5, loc=0.5, scale=2.0)
        assert close(expon_pdf(1.5, 0.5, 2.0), 0.3032653298563167)
        assert close(expon_cdf(1.5, 0.5, 2.0), 0.3934693402873666)

        # scipy.stats.expon.ppf(0.8, loc=0.5, scale=2.0)
        assert close(expon_ppf(0.8, 0.5, 2.0), 3.718875824868201)

    def test_support_and_roundtrip(self):
        assert expon_pdf(-1.0, 0.0, 1.0) == 0.0
        assert expon_cdf(-1.0, 0.0, 1.0) == 0.0
        q = expon_cdf(2.5, 0.5, 2.0)
        assert close(expon_ppf(q, 0.5, 2.0), 2.5)


class TestUniform:
    def test_pdf_cdf_ppf_with_location_and_scale(self):
        # scipy.stats.uniform.pdf/cdf(1.5, loc=0.5, scale=2.0)
        assert uniform_pdf(1.5, 0.5, 2.0) == 0.5
        assert uniform_cdf(1.5, 0.5, 2.0) == 0.5

        # scipy.stats.uniform.ppf(0.8, loc=0.5, scale=2.0)
        assert close(uniform_ppf(0.8, 0.5, 2.0), 2.1)

    def test_support_boundaries(self):
        assert uniform_pdf(0.5, 0.5, 2.0) == 0.5
        assert uniform_pdf(2.5, 0.5, 2.0) == 0.5
        assert uniform_pdf(2.6, 0.5, 2.0) == 0.0
        assert uniform_cdf(0.4, 0.5, 2.0) == 0.0
        assert uniform_cdf(2.6, 0.5, 2.0) == 1.0


class TestLaplace:
    def test_pdf_cdf_ppf_with_location_and_scale(self):
        # scipy.stats.laplace.pdf/cdf(1.5, loc=0.5, scale=2.0)
        assert close(laplace_pdf(1.5, 0.5, 2.0), 0.15163266492815836)
        assert close(laplace_cdf(1.5, 0.5, 2.0), 0.6967346701436833)

        # scipy.stats.laplace.ppf(0.8, loc=0.5, scale=2.0)
        assert close(laplace_ppf(0.8, 0.5, 2.0), 2.3325814637483107)

    def test_symmetry_and_roundtrip(self):
        assert laplace_cdf(0.5, 0.5, 2.0) == 0.5
        assert close(laplace_pdf(-1.0, 0.5, 2.0), laplace_pdf(2.0, 0.5, 2.0))
        q = laplace_cdf(-1.25, 0.25, 1.5)
        assert close(laplace_ppf(q, 0.25, 1.5), -1.25)


class TestCauchy:
    def test_pdf_cdf_ppf_with_location_and_scale(self):
        # scipy.stats.cauchy.pdf/cdf(1.5, loc=0.5, scale=2.0)
        assert close(cauchy_pdf(1.5, 0.5, 2.0), 0.12732395447351627)
        assert close(cauchy_cdf(1.5, 0.5, 2.0), 0.6475836176504333)

        # scipy.stats.cauchy.ppf(0.8, loc=0.5, scale=2.0)
        assert close(cauchy_ppf(0.8, 0.5, 2.0), 3.252763840942347)

    def test_symmetry_and_roundtrip(self):
        assert cauchy_cdf(0.5, 0.5, 2.0) == 0.5
        assert close(cauchy_pdf(-1.0, 0.5, 2.0), cauchy_pdf(2.0, 0.5, 2.0))
        q = cauchy_cdf(-1.25, 0.25, 1.5)
        assert close(cauchy_ppf(q, 0.25, 1.5), -1.25)


class TestPpfEndpoints:
    @pytest.mark.parametrize(
        "ppf",
        [norm_ppf, logistic_ppf, laplace_ppf, cauchy_ppf],
    )
    def test_unbounded_endpoints_use_finite_sentinels(self, ppf):
        lower = ppf(0.0, 0.5, 2.0)
        upper = ppf(1.0, 0.5, 2.0)
        assert math.isfinite(lower) and lower < -1e307
        assert math.isfinite(upper) and upper > 1e307

    def test_exponential_and_uniform_endpoints(self):
        assert expon_ppf(0.0, 0.5, 2.0) == 0.5
        assert math.isfinite(expon_ppf(1.0, 0.5, 2.0))
        assert uniform_ppf(0.0, 0.5, 2.0) == 0.5
        assert uniform_ppf(1.0, 0.5, 2.0) == 2.5


ALL_KERNELS = [
    norm_pdf, norm_cdf, norm_ppf,
    logistic_pdf, logistic_cdf, logistic_ppf,
    expon_pdf, expon_cdf, expon_ppf,
    uniform_pdf, uniform_cdf, uniform_ppf,
    laplace_pdf, laplace_cdf, laplace_ppf,
    cauchy_pdf, cauchy_cdf, cauchy_ppf,
]


class TestNanPropagation:
    """A NaN in any argument yields NaN, never a plausible finite value.

    Interior values (x=1.5, q=0.8) keep PPF calls off the sentinel-endpoint
    branches, whose q<=0 / q>=1 guards return constants before touching
    loc/scale (see TestPpfOutOfRangeQ).
    """

    NAN = float("nan")

    @pytest.mark.parametrize("fn", ALL_KERNELS, ids=lambda fn: fn.__name__)
    def test_nan_value_or_q(self, fn):
        assert math.isnan(fn(self.NAN, 0.5, 2.0))

    @pytest.mark.parametrize("fn", ALL_KERNELS, ids=lambda fn: fn.__name__)
    def test_nan_loc(self, fn):
        value = 0.8 if fn.__name__.endswith("_ppf") else 1.5
        assert math.isnan(fn(value, self.NAN, 2.0))

    @pytest.mark.parametrize("fn", ALL_KERNELS, ids=lambda fn: fn.__name__)
    def test_nan_scale(self, fn):
        value = 0.8 if fn.__name__.endswith("_ppf") else 1.5
        assert math.isnan(fn(value, 0.5, self.NAN))


class TestPpfOutOfRangeQ:
    """Pins current out-of-range behavior; scipy parity is deferred debt.

    scipy returns NaN for q outside [0, 1].  The kernels cannot produce an
    IEEE NaN constant until postpython#36 is fixed, so the q<=0 / q>=1
    guards fold invalid q into the q=0 / q=1 result.  Revisit with the
    Target 2 compatibility harness once #36 lands.
    """

    @pytest.mark.parametrize(
        "ppf",
        [norm_ppf, logistic_ppf, expon_ppf, uniform_ppf, laplace_ppf, cauchy_ppf],
        ids=lambda fn: fn.__name__,
    )
    def test_invalid_q_folds_to_endpoint_result(self, ppf):
        assert ppf(-0.5, 0.5, 2.0) == ppf(0.0, 0.5, 2.0)
        assert ppf(1.5, 0.5, 2.0) == ppf(1.0, 0.5, 2.0)
