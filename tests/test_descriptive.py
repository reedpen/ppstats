"""Tests for ppstats descriptive-statistics reductions.

Reference values are hardcoded, pre-computed with scipy 1.18.0 /
numpy (cited per value) so the suite runs without scipy installed.
Boolean-mode defaults follow scipy: skew is the biased Fisher–Pearson
g1, kurtosis is the biased Fisher (excess) definition.
"""

import math
import warnings

import pytest

from ppstats import (
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

# Fixed sample vectors (float64-exact inputs)
A1 = [1.0, 2.0, 3.0, 4.0, 5.0]           # symmetric
A2 = [1.0, 2.0, 3.0, 4.0, 10.0]          # right-skewed
A3 = [0.5, 1.5, 2.5, 3.5, 4.5, 10.0]     # longer, skewed


def close(a, b, rtol=1e-12, atol=1e-300):
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


class TestMean:
    def test_symmetric(self):
        assert close(mean(A1), 3.0)                      # np.mean(A1)

    def test_skewed(self):
        assert close(mean(A2), 4.0)                      # np.mean(A2)

    def test_longer(self):
        assert close(mean(A3), 3.75)                     # np.mean(A3)


class TestVariance:
    def test_symmetric(self):
        assert close(variance(A1), 2.0)                  # np.var(A1)

    def test_skewed(self):
        assert close(variance(A2), 10.0)                 # np.var(A2)

    def test_longer(self):
        assert close(variance(A3), 9.479166666666666)    # np.var(A3)

    def test_matches_second_central_moment(self):
        assert close(variance(A3), moment(A3, 2))


class TestGmean:
    def test_powers_of_two(self):
        assert close(gmean([2.0, 4.0, 8.0]), 4.0)        # scipy.stats.gmean([2,4,8])

    def test_symmetric(self):
        assert close(gmean(A1), 2.6051710846973517)      # scipy.stats.gmean(A1)

    def test_skewed(self):
        assert close(gmean(A2), 2.9925557394776896)      # scipy.stats.gmean(A2)

    def test_longer(self):
        assert close(gmean(A3), 2.5805580817082)         # scipy.stats.gmean(A3)

    def test_zero_element_gives_zero(self):
        assert gmean([2.0, 0.0, 4.0]) == 0.0             # scipy.stats.gmean([2,0,4])

    def test_negative_dominates_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert math.isnan(gmean([0.0, -1.0]))        # scipy.stats.gmean([0,-1]) → nan


class TestHmean:
    def test_exact_fraction(self):
        # scipy.stats.hmean([1,2,4]) == 3 / (1 + 1/2 + 1/4) == 12/7
        assert close(hmean([1.0, 2.0, 4.0]), 12.0 / 7.0)

    def test_symmetric(self):
        assert close(hmean(A1), 2.18978102189781)        # scipy.stats.hmean(A1)

    def test_skewed(self):
        assert close(hmean(A2), 2.290076335877863)       # scipy.stats.hmean(A2)

    def test_longer(self):
        assert close(hmean(A3), 1.632829373650108)       # scipy.stats.hmean(A3)

    def test_zero_element_gives_zero(self):
        assert hmean([1.0, 0.0, 2.0]) == 0.0             # scipy.stats.hmean([1,0,2])

    def test_negative_dominates_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert math.isnan(hmean([0.0, -1.0]))        # scipy.stats.hmean([0,-1]) → nan


class TestMoment:
    def test_order_zero_is_one(self):
        assert close(moment(A1, 0), 1.0)                 # scipy.stats.moment(A1, order=0)

    def test_order_one_is_zero(self):
        assert moment(A1, 1) == 0.0                      # scipy.stats.moment(A1, order=1)

    def test_second(self):
        assert close(moment(A2, 2), 10.0)                # scipy.stats.moment(A2, order=2)

    def test_third(self):
        assert close(moment(A2, 3), 36.0)                # scipy.stats.moment(A2, order=3)

    def test_fourth(self):
        assert close(moment(A2, 4), 278.8)               # scipy.stats.moment(A2, order=4)

    def test_third_symmetric_vanishes(self):
        assert abs(moment(A1, 3)) < 1e-12                # scipy.stats.moment(A1, order=3)

    def test_fourth_longer(self):
        assert close(moment(A3, 4), 277.6393229166667)   # scipy.stats.moment(A3, order=4)

    def test_negative_order_reciprocal(self):
        # scipy.stats.moment([1,2,4], order=-1) / order=-2
        assert close(moment([1.0, 2.0, 4.0], -1), -1.0499999999999996, rtol=1e-10)
        assert close(moment([1.0, 2.0, 4.0], -2), 3.3074999999999974, rtol=1e-10)

    def test_negative_order_zero_deviation_is_inf(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # scipy.stats.moment([1,2,3], order=-1) → inf (zero deviation)
            assert math.isinf(moment([1.0, 2.0, 3.0], -1))


class TestSkew:
    def test_symmetric_is_zero(self):
        assert abs(skew(A1)) < 1e-12                     # scipy.stats.skew(A1)

    def test_skewed(self):
        assert close(skew(A2), 1.1384199576606167)       # scipy.stats.skew(A2)

    def test_longer(self):
        assert close(skew(A3), 1.124304844072362)        # scipy.stats.skew(A3)

    def test_constant_input_is_nan(self):
        with warnings.catch_warnings():
            # interpreted mode surfaces the intentional 0/0 as a RuntimeWarning
            warnings.simplefilter("ignore", RuntimeWarning)
            assert math.isnan(skew([3.0, 3.0, 3.0]))     # scipy.stats.skew([3,3,3]) → nan


class TestKurtosis:
    def test_symmetric(self):
        assert close(kurtosis(A1), -1.3)                 # scipy.stats.kurtosis(A1)

    def test_skewed(self):
        assert close(kurtosis(A2), -0.21199999999999974) # scipy.stats.kurtosis(A2)

    def test_longer(self):
        assert close(kurtosis(A3), 0.08987320371935814)  # scipy.stats.kurtosis(A3)

    def test_constant_input_is_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert math.isnan(kurtosis([3.0, 3.0, 3.0])) # scipy.stats.kurtosis([3,3,3]) → nan


class TestSem:
    def test_default_ddof_one(self):
        assert close(sem(A1, 1), 0.7071067811865476)     # scipy.stats.sem(A1)

    def test_ddof_zero(self):
        assert close(sem(A1, 0), 0.6324555320336759)     # scipy.stats.sem(A1, ddof=0)

    def test_skewed(self):
        assert close(sem(A2, 1), 1.5811388300841895)     # scipy.stats.sem(A2)

    def test_longer(self):
        assert close(sem(A3, 1), 1.3768926368215257)     # scipy.stats.sem(A3)
        assert close(sem(A3, 0), 1.256925260749863)      # scipy.stats.sem(A3, ddof=0)


class TestZscore:
    # scipy.stats.zscore(A1) — population std (ddof=0) by default
    REF_A1_DDOF0 = [
        -1.4142135623730951,
        -0.7071067811865476,
        0.0,
        0.7071067811865476,
        1.4142135623730951,
    ]
    REF_A1_DDOF1 = [
        -1.2649110640673518,
        -0.6324555320336759,
        0.0,
        0.6324555320336759,
        1.2649110640673518,
    ]  # scipy.stats.zscore(A1, ddof=1)

    def test_ddof_zero(self):
        out = zscore(A1, 0)
        for got, ref in zip(out, self.REF_A1_DDOF0):
            assert close(got, ref, rtol=1e-10, atol=1e-12)

    def test_ddof_one(self):
        out = zscore(A1, 1)
        for got, ref in zip(out, self.REF_A1_DDOF1):
            assert close(got, ref, rtol=1e-10, atol=1e-12)

    def test_output_is_standardized(self):
        out = list(zscore(A3, 0))
        n = len(out)
        m = sum(out) / n
        v = sum((z - m) ** 2 for z in out) / n
        assert abs(m) < 1e-12
        assert close(v, 1.0, rtol=1e-10)


class TestBroadcasting:
    """Batch (2-D) inputs reduce along the last axis, gufunc-style."""

    np = pytest.importorskip("numpy")

    def test_mean_batch(self):
        np = self.np
        batch = np.array([A1, A2])
        out = mean(batch)
        np.testing.assert_allclose(out, [3.0, 4.0], rtol=1e-12)

    def test_skew_batch(self):
        np = self.np
        batch = np.array([A1, A2])
        out = skew(batch)
        np.testing.assert_allclose(out, [0.0, 1.1384199576606167], rtol=1e-10, atol=1e-12)

    def test_moment_batch_with_scalar_order(self):
        np = self.np
        batch = np.array([A1, A2])
        out = moment(batch, 2)
        np.testing.assert_allclose(out, [2.0, 10.0], rtol=1e-12)

    def test_zscore_batch_shape(self):
        np = self.np
        batch = np.array([A1, A2])
        out = zscore(batch, 0)
        assert out.shape == batch.shape
        np.testing.assert_allclose(out.mean(axis=-1), [0.0, 0.0], atol=1e-12)
