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

np = pytest.importorskip("numpy")

from ppstats import _descriptive as interp

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")

PUBLIC = [
    "mean", "variance", "gmean", "hmean",
    "moment", "skew", "kurtosis", "sem", "zscore",
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
