"""Build ppstats into the package-manager prefix layout."""

import platform
from pathlib import Path

import ppspecial
from postpyc.build import build_file


REPO_ROOT = Path(__file__).resolve().parent.parent
PREFIX = REPO_ROOT / "dist" / "prefix"
PPSPECIAL_ROOT = Path(ppspecial.__file__).resolve().parent.parent
LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"


def main() -> int:
    lib_dir = PREFIX / "lib"
    include_dir = PREFIX / "include"
    share_dir = PREFIX / "share" / "postpyc"
    for directory in (lib_dir, include_dir, share_dir):
        directory.mkdir(parents=True, exist_ok=True)

    lib = build_file(
        REPO_ROOT / "ppstats" / "__init__.py",
        output=lib_dir / f"libppstats{LIB_SUFFIX}",
        emit_header=True,
        emit_manifest=True,
        module_name="ppstats",
        search_paths=[PPSPECIAL_ROOT],
    )
    header = lib.with_suffix(".h")
    manifest = lib.with_suffix(".json")
    installed_header = include_dir / "ppstats.h"
    installed_manifest = share_dir / "ppstats.json"
    header.replace(installed_header)
    manifest.replace(installed_manifest)
    print(lib)
    print(installed_header)
    print(installed_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
