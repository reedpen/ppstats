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
        sys.modules["ppstats_native"] = native

        import ppstats

        assert ppstats.__native_available__ is True
        assert ppstats.__native_module__ is native
        assert ppstats.mean([1.0]) == "native-mean"
        assert ppstats.skew([1.0]) == "native-skew"
        # names absent from the native module keep the interpreted kernel
        assert ppstats.hmean([1.0, 2.0, 4.0]) != "native-hmean"
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
