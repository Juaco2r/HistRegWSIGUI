#!/usr/bin/env python3
"""Build HistRegGUI with PyInstaller on the current operating system."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from importlib.metadata import distribution
from pathlib import Path

import PyInstaller.__main__


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def add_data(source: Path, destination: str) -> str:
    return f"{source}{os.pathsep}{destination}"


def package_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False




def validate_pkg_resources_compatibility() -> None:
    """Fail early when setuptools no longer provides PyInstaller's API."""
    try:
        import pkg_resources
    except ImportError as exc:
        raise RuntimeError(
            "pkg_resources is unavailable. Install setuptools==81.0.0 before building."
        ) from exc

    if not hasattr(pkg_resources, "NullProvider"):
        raise RuntimeError(
            "The installed pkg_resources does not provide NullProvider. "
            "PyInstaller's runtime hook requires setuptools==81.0.0 for this build."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--platform-label", default=platform.system().lower())
    parser.add_argument("--architecture", default=platform.machine().lower())
    args = parser.parse_args()

    validate_pkg_resources_compatibility()

    build_root = REPOSITORY_ROOT / "build"
    dist_root = REPOSITORY_ROOT / "dist"
    shutil.rmtree(build_root, ignore_errors=True)
    shutil.rmtree(dist_root, ignore_errors=True)
    build_root.mkdir(parents=True, exist_ok=True)

    build_info_path = build_root / "build_info.json"
    build_info_path.write_text(
        json.dumps(
            {
                "variant": args.variant,
                "platform": args.platform_label,
                "architecture": args.architecture,
                "torch_variant": "CUDA-enabled" if args.variant == "cuda" else "CPU-only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Use the installed PyPI package instead of relying on an untracked vendored
    # directory. Including the source tree as data is intentional: DeeperHistReg
    # has several top-level imports that resolve from its package directory.
    deeperhistreg_dir = Path(distribution("deeperhistreg").locate_file("deeperhistreg"))
    if not deeperhistreg_dir.exists():
        raise FileNotFoundError(f"Installed DeeperHistReg package not found: {deeperhistreg_dir}")

    pyinstaller_args = [
        str(REPOSITORY_ROOT / "src" / "histreggui" / "app.py"),
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--name",
        "HistRegGUI",
        "--paths",
        str(REPOSITORY_ROOT / "src"),
        "--distpath",
        str(dist_root),
        "--workpath",
        str(build_root / "pyinstaller"),
        "--specpath",
        str(build_root),
        "--add-data",
        add_data(deeperhistreg_dir, "deeperhistreg"),
        "--add-data",
        add_data(build_info_path, "histreggui"),
        "--add-data",
        add_data(REPOSITORY_ROOT / "README.md", "."),
        "--add-data",
        add_data(REPOSITORY_ROOT / "LICENSE", "."),
        "--add-data",
        add_data(REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md", "."),
        "--collect-submodules",
        "deeperhistreg",
        "--copy-metadata",
        "deeperhistreg",
    ]

    if sys.platform == "darwin":
        pyinstaller_args.extend(
            ["--osx-bundle-identifier", "org.juaco2r.histreggui"]
        )

    # These packages distribute native libraries/resources in wheels. Collecting
    # them explicitly makes the release portable on clean machines.
    if package_exists("openslide_bin"):
        pyinstaller_args.extend(["--collect-all", "openslide_bin"])
    if package_exists("_libvips"):
        # pyvips-binary exposes a native extension module rather than a Python
        # package. A hidden import lets PyInstaller inspect and collect its
        # dependent libvips libraries.
        pyinstaller_args.extend(["--hidden-import", "_libvips"])

    # Dynamic registrations and scientific backends that PyInstaller may not
    # discover from DeeperHistReg's string-based factory lookup.
    for package_name in (
        "torchio",
        "SimpleITK",
        "skimage",
        "sklearn",
        "cv2",
    ):
        if package_exists(package_name):
            pyinstaller_args.extend(["--collect-submodules", package_name])

    PyInstaller.__main__.run(pyinstaller_args)

    # The analysis work directory duplicates many large scientific/CUDA files.
    # It is not needed once the distributable folder has been produced.
    shutil.rmtree(build_root / "pyinstaller", ignore_errors=True)


if __name__ == "__main__":
    main()
