"""Build ppstats as an importable CPython extension module.

Produces `ppstats_native.<ext-suffix>` in the repository root: a real
extension module whose attributes are numpy.ufunc objects (generalized
ufuncs for the reduction kernels) registered from the compiled kernels
(postpyc `ext_module=True` output). This target is intentionally
separate from the plain C shared library built by
scripts/build_native.py; both link the same compiled translation units.
"""

import importlib.util
import sys
from pathlib import Path

from postpyc.build import build_file, BuildError
import ppspecial

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ENTRY = REPO_ROOT / "ppstats" / "__init__.py"
MODULE_NAME = "ppstats_native"
PPSPECIAL_ROOT = Path(ppspecial.__file__).resolve().parent.parent


def main() -> int:
    try:
        ext_path = build_file(
            PACKAGE_ENTRY,
            ext_module=True,
            module_name=MODULE_NAME,
            search_paths=[PPSPECIAL_ROOT],
        )
    except BuildError as exc:
        print("extension build FAILED:")
        print("  " + "\n  ".join(str(exc).splitlines()[:8]))
        return 1

    # Move the artifact to the repo root and smoke-test the import.
    target = REPO_ROOT / ext_path.name
    if ext_path != target:
        ext_path.replace(target)

    spec = importlib.util.spec_from_file_location(MODULE_NAME, str(target))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ufuncs = sorted(n for n in dir(module) if not n.startswith("_"))
    print(f"built {target.name}")
    print(f"registered {len(ufuncs)} ufuncs: {', '.join(ufuncs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
