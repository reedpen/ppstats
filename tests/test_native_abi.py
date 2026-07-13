"""Plain C ABI shared-library smoke tests.

This is intentionally separate from tests/test_native_ext.py.  These tests
load the package shared library with ctypes and call exported C functions
directly, without importing a CPython extension module or NumPy ufunc layer.

The reduction kernels take POST array views (`__pp_array`, spec §9.2), so
the tests build those structs by hand around plain C double buffers.
"""

from __future__ import annotations

import ctypes
import json
import math
import shutil
from pathlib import Path

import pytest

import ppstats

cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")

pytestmark = pytest.mark.skipif(cc is None, reason="No C compiler available")


class PPArray(ctypes.Structure):
    """Mirror of the generated `__pp_array` view struct (spec §9.2)."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("ndim", ctypes.c_int64),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("offset_bytes", ctypes.c_int64),
    ]


def pp_array_1d(values):
    """Build a 1-D `__pp_array` over a fresh C double buffer.

    Returns (view, buffer) — keep the buffer alive for the call duration.
    """
    n = len(values)
    buf = (ctypes.c_double * max(n, 1))(*values)
    shape = (ctypes.c_int64 * 1)(n)
    strides = (ctypes.c_int64 * 1)(ctypes.sizeof(ctypes.c_double))
    view = PPArray(
        ctypes.cast(buf, ctypes.c_void_p),
        1,
        shape,
        strides,
        0,
    )
    return view, (buf, shape, strides)


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    from postpyc.build import build_file

    out_dir = tmp_path_factory.mktemp("ppstats-native-abi")
    lib_path = build_file(
        Path(ppstats.__file__),
        output=out_dir / "ppstats.so",
        emit_header=True,
        emit_manifest=True,
    )
    return {
        "path": lib_path,
        "lib": ctypes.CDLL(str(lib_path)),
        "header": lib_path.with_suffix(".h").read_text(),
        "manifest": json.loads(lib_path.with_suffix(".json").read_text()),
    }


def _reduce(lib, export_name: str, values, *scalars):
    """Call a `(n)[,()]->()` reduction through its stable pp_* C symbol."""
    fn = getattr(lib, f"pp_{export_name}")
    fn.restype = None
    a_view, a_keep = pp_array_1d(values)
    out_view, out_keep = pp_array_1d([0.0])
    args = [ctypes.byref(a_view)]
    for s in scalars:
        args.append(ctypes.c_int64(s))
    args.append(ctypes.byref(out_view))
    args.append(ctypes.c_int64(len(values)))
    fn(*args)
    return out_keep[0][0]


A1 = [1.0, 2.0, 3.0, 4.0, 5.0]
A2 = [1.0, 2.0, 3.0, 4.0, 10.0]

REDUCTION_CASES = [
    # (export, input, scalar args, expected, tol) — refs as in test_descriptive.py
    ("mean", A1, (), 3.0, 1e-14),
    ("variance", A2, (), 10.0, 1e-13),
    ("gmean", [2.0, 4.0, 8.0], (), 4.0, 1e-13),
    ("hmean", [1.0, 2.0, 4.0], (), 12.0 / 7.0, 1e-13),
    ("moment", A2, (3,), 36.0, 1e-12),
    ("skew", A2, (), 1.1384199576606167, 1e-12),
    ("kurtosis", A2, (), -0.21199999999999974, 1e-12),
    ("sem", A1, (1,), 0.7071067811865476, 1e-13),
]


@pytest.mark.parametrize(
    "name,values,scalars,expected,tol",
    REDUCTION_CASES,
    ids=[c[0] for c in REDUCTION_CASES],
)
def test_package_shared_library_exports_reduction_kernels(
    native_artifact,
    name,
    values,
    scalars,
    expected,
    tol,
):
    got = _reduce(native_artifact["lib"], name, values, *scalars)
    assert got == pytest.approx(expected, abs=tol, rel=tol)


def test_zscore_writes_full_output_vector(native_artifact):
    lib = native_artifact["lib"]
    fn = lib.pp_zscore
    fn.restype = None

    a_view, a_keep = pp_array_1d(A1)
    out_view, out_keep = pp_array_1d([0.0] * len(A1))
    fn(
        ctypes.byref(a_view),
        ctypes.c_int64(0),
        ctypes.byref(out_view),
        ctypes.c_int64(len(A1)),
    )
    got = list(out_keep[0])
    # scipy.stats.zscore(A1)
    ref = [-1.4142135623730951, -0.7071067811865476, 0.0,
           0.7071067811865476, 1.4142135623730951]
    for g, r in zip(got, ref):
        assert g == pytest.approx(r, abs=1e-12)
    # standardized: mean 0, population std 1
    assert math.fsum(got) == pytest.approx(0.0, abs=1e-12)


def test_constant_input_yields_nan_through_c_abi(native_artifact):
    got = _reduce(native_artifact["lib"], "skew", [3.0, 3.0, 3.0])
    assert math.isnan(got)


def test_header_declares_reduction_exports(native_artifact):
    header = native_artifact["header"]
    assert "typedef struct __pp_array {" in header
    assert "void pp_mean(__pp_array* a, __pp_array* out, int64_t pp_dim_n);" in header
    assert (
        "void pp_moment(__pp_array* a, int64_t order, __pp_array* out, int64_t pp_dim_n);"
        in header
    )
    assert (
        "void pp_zscore(__pp_array* a, int64_t ddof, __pp_array* out, int64_t pp_dim_n);"
        in header
    )


def test_manifest_describes_exported_abi(native_artifact):
    manifest = native_artifact["manifest"]
    assert manifest["post_abi"] == 1
    assert manifest["artifact"] == "ppstats"

    exports = {entry["name"]: entry for entry in manifest["exports"]}
    for name in ppstats.__all__:
        assert name in exports, name
        assert exports[name]["c_symbol"] == f"pp_{name}"
