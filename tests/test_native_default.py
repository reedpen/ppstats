"""Default import behavior for optional native acceleration."""

import subprocess
import sys
import textwrap


def test_package_import_prefers_available_native_module():
    script = textwrap.dedent(
        """
        import sys
        import types

        native = types.ModuleType("ppstats_native")
        native.mean = lambda a: "native-mean"
        native.skew = lambda a: "native-skew"
        native.norm_pdf = lambda x, loc, scale: "native-norm-pdf"
        sys.modules["ppstats_native"] = native

        import ppstats

        assert ppstats.__native_available__ is True
        assert ppstats.__native_module__ is native
        assert ppstats.mean([1.0]) == "native-mean"
        assert ppstats.skew([1.0]) == "native-skew"
        assert ppstats.norm_pdf(0.0, 0.0, 1.0) == "native-norm-pdf"
        # names absent from the native module keep the interpreted kernel
        assert ppstats.hmean([1.0, 2.0, 4.0]) != "native-hmean"
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)


def test_broken_native_module_degrades_to_interpreted_kernels():
    """A present-but-unloadable extension (missing numpy, ABI mismatch)
    must warn and fall back, not break ``import ppstats``."""
    script = textwrap.dedent(
        """
        import sys
        import types
        import warnings

        class _BrokenFinder:
            def find_spec(self, name, path=None, target=None):
                if name == "ppstats_native":
                    raise ImportError("numpy._core.multiarray failed to import")
                return None

        sys.meta_path.insert(0, _BrokenFinder())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import ppstats

        assert ppstats.__native_available__ is False
        assert any("failed to import" in str(w.message) for w in caught)
        assert ppstats.mean([1.0, 2.0, 3.0]) == 2.0
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
