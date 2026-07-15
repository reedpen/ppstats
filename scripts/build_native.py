"""Compile each ppstats kernel module to a native shared library.

Every module must build. The whole package is also built into a single
shared library from ppstats/__init__.py with stable C ABI sidecars.
Any failure exits non-zero.
"""

import sys
import tempfile
from pathlib import Path

from postpyc.build import build_file, BuildError
import ppspecial

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "ppstats"
PACKAGE_ENTRY = PACKAGE_DIR / "__init__.py"

EXPECTED_NATIVE = ["_descriptive", "_distributions"]
PPSPECIAL_ROOT = Path(ppspecial.__file__).resolve().parent.parent


def main() -> int:
    out_dir = Path(tempfile.mkdtemp(prefix="ppstats-native-"))
    failures = []

    for name in EXPECTED_NATIVE:
        source = PACKAGE_DIR / f"{name}.py"
        try:
            lib = build_file(
                source,
                output=out_dir / f"{name}.so",
                search_paths=[PPSPECIAL_ROOT],
            )
            print(f"  {name:14s} OK       -> {lib}")
        except BuildError as exc:
            failures.append(name)
            print(f"  {name:14s} FAILED")
            print("    " + "\n    ".join(str(exc).splitlines()[:6]))

    # The full package: one shared library with all translation units
    # plus stable C ABI sidecars (`ppstats.h`, `ppstats.json`).
    try:
        lib = build_file(
            PACKAGE_ENTRY,
            output=out_dir / "ppstats.so",
            emit_header=True,
            emit_manifest=True,
            search_paths=[PPSPECIAL_ROOT],
        )
        print(f"  {'package':14s} OK       -> {lib}")
        print(f"  {'header':14s} OK       -> {lib.with_suffix('.h')}")
        print(f"  {'manifest':14s} OK       -> {lib.with_suffix('.json')}")
    except BuildError as exc:
        failures.append("package")
        print(f"  {'package':14s} FAILED")
        print("    " + "\n    ".join(str(exc).splitlines()[:6]))

    if failures:
        print(f"\n{len(failures)} build(s) failed: {failures}")
        return 1
    print("\nAll modules and the full package compile natively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
