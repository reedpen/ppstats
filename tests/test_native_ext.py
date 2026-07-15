"""Compiled NumPy extension vs. interpreted mode.

Builds ppstats as a CPython extension module once per session, then
verifies that every registered gufunc agrees with the interpreted POST
Python implementation, including batch broadcasting behavior.

Interpreted references come from ppstats._descriptive directly — the
package namespace may already prefer an installed native module.
"""

import importlib.util
import shutil
import warnings

import pytest
import ppspecial

np = pytest.importorskip("numpy")

from ppstats import _descriptive as interp
from ppstats import _distributions as dist_interp

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")

PUBLIC = [
    "mean", "variance", "gmean", "hmean",
    "moment", "skew", "kurtosis", "sem", "zscore",
    "norm_pdf", "norm_cdf", "norm_ppf",
    "logistic_pdf", "logistic_cdf", "logistic_ppf",
    "expon_pdf", "expon_cdf", "expon_ppf",
    "uniform_pdf", "uniform_cdf", "uniform_ppf",
    "laplace_pdf", "laplace_cdf", "laplace_ppf",
    "cauchy_pdf", "cauchy_cdf", "cauchy_ppf",
]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    from postpyc.build import build_file
    from pathlib import Path
    import ppstats

    out_dir = tmp_path_factory.mktemp("ppstats-ext")
    ext = build_file(
        Path(ppstats.__file__),
        ext_module=True,
        module_name="ppstats_native_test",
        output=out_dir / "ppstats_native_test.so",
        search_paths=[Path(ppspecial.__file__).resolve().parent.parent],
    )
    spec = importlib.util.spec_from_file_location("ppstats_native_test", str(ext))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_public_gufuncs_registered(native):
    registered = {n for n in dir(native) if not n.startswith("_")}
    assert set(PUBLIC) <= registered
    for name in PUBLIC:
        assert isinstance(getattr(native, name), np.ufunc), name


# Sample batches: rows are independent reduction inputs.
BATCH = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 2.0, 3.0, 4.0, 10.0],
    [0.5, 1.5, 2.5, 3.5, 9.0],
    [2.0, 4.0, 8.0, 16.0, 32.0],
])

PLAIN_REDUCTIONS = ["mean", "variance", "gmean", "hmean", "skew", "kurtosis"]


@pytest.mark.parametrize("name", PLAIN_REDUCTIONS)
def test_compiled_matches_interpreted(native, name):
    compiled = getattr(native, name)
    interpreted = getattr(interp, name)
    expected = np.array([interpreted(row) for row in BATCH])
    np.testing.assert_allclose(compiled(BATCH), expected, rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_moment_matches_interpreted(native, order):
    expected = np.array([interp.moment(row, order) for row in BATCH])
    np.testing.assert_allclose(native.moment(BATCH, order), expected, rtol=1e-13)


@pytest.mark.parametrize("ddof", [0, 1])
def test_sem_matches_interpreted(native, ddof):
    expected = np.array([interp.sem(row, ddof) for row in BATCH])
    np.testing.assert_allclose(native.sem(BATCH, ddof), expected, rtol=1e-13)


@pytest.mark.parametrize("ddof", [0, 1])
def test_zscore_matches_interpreted(native, ddof):
    expected = np.stack([interp.zscore(row, ddof) for row in BATCH])
    got = native.zscore(BATCH, ddof)
    assert got.shape == BATCH.shape
    np.testing.assert_allclose(got, expected, rtol=1e-13)


def test_batch_reduction_shape(native):
    # (2, 3, 5) reduces along the last axis → (2, 3)
    stacked = np.stack([BATCH[:3], BATCH[1:]])
    out = native.mean(stacked)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, stacked.mean(axis=-1), rtol=1e-13)


def test_ufunc_out_parameter_and_docstring(native):
    buffer = np.empty(BATCH.shape[0])
    result = native.mean(BATCH, out=buffer)
    assert result is buffer
    assert "arithmetic mean" in native.mean.__doc__.lower()


def test_constant_input_is_nan_in_both_modes(native):
    const = np.full(4, 7.5)
    with warnings.catch_warnings():
        # the intentional 0/0 NaN trips numpy's invalid-value warning
        warnings.simplefilter("ignore", RuntimeWarning)
        assert np.isnan(native.skew(const))
        assert np.isnan(native.kurtosis(const))
        assert np.isnan(interp.skew(const))
        assert np.isnan(interp.kurtosis(const))


DISTRIBUTION_CASES = [
    ("norm_pdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("norm_cdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("norm_ppf", np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 0.5, 2.0),
    ("logistic_pdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("logistic_cdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("logistic_ppf", np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 0.5, 2.0),
    ("expon_pdf", np.array([-1.0, 0.5, 2.0]), 0.5, 2.0),
    ("expon_cdf", np.array([-1.0, 0.5, 2.0]), 0.5, 2.0),
    ("expon_ppf", np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 0.5, 2.0),
    ("uniform_pdf", np.array([-1.0, 0.5, 2.0, 3.0]), 0.5, 2.0),
    ("uniform_cdf", np.array([-1.0, 0.5, 2.0, 3.0]), 0.5, 2.0),
    ("uniform_ppf", np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 0.5, 2.0),
    ("laplace_pdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("laplace_cdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("laplace_ppf", np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 0.5, 2.0),
    ("cauchy_pdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("cauchy_cdf", np.array([-2.0, 0.0, 1.5]), 0.5, 2.0),
    ("cauchy_ppf", np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 0.5, 2.0),
]


@pytest.mark.parametrize("name,value,loc,scale", DISTRIBUTION_CASES)
def test_distribution_compiled_matches_interpreted(native, name, value, loc, scale):
    compiled = getattr(native, name)
    interpreted = getattr(dist_interp, name)
    np.testing.assert_allclose(
        compiled(value, loc, scale),
        interpreted(value, loc, scale),
        rtol=1e-13,
        atol=1e-300,
    )


def test_distribution_parameters_broadcast(native):
    x = np.array([-1.0, 0.0, 1.0])
    scale = np.array([1.0, 2.0, 4.0])
    expected = dist_interp.logistic_cdf(x, 0.0, scale)
    np.testing.assert_allclose(native.logistic_cdf(x, 0.0, scale), expected, rtol=1e-13)


def test_distribution_cdf_ppf_array_roundtrip(native):
    x = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    q = native.cauchy_cdf(x, 0.25, 1.5)
    np.testing.assert_allclose(
        native.cauchy_ppf(q, 0.25, 1.5),
        x,
        rtol=1e-13,
        atol=1e-13,
    )
