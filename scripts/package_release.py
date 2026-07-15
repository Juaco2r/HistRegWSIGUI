#!/usr/bin/env python3
"""Create a release archive from the current platform's PyInstaller output."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
RELEASE = ROOT / "release-assets"


def zip_directory(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in source.rglob("*"):
            archive.write(path, arcname=path.relative_to(source.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()

    RELEASE.mkdir(parents=True, exist_ok=True)
    variant = args.variant.upper()
    base_name = f"HistRegGUI-{args.platform}-{args.architecture}-{variant}"

    if sys.platform == "darwin":
        source = DIST / "HistRegGUI.app"
        if not source.exists():
            raise FileNotFoundError(source)
        output = RELEASE / f"{base_name}.zip"
        subprocess.run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(source), str(output)],
            check=True,
        )
    elif sys.platform.startswith("win"):
        source = DIST / "HistRegGUI"
        if not source.exists():
            raise FileNotFoundError(source)
        output = RELEASE / f"{base_name}.zip"
        zip_directory(source, output)
    else:
        source = DIST / "HistRegGUI"
        if not source.exists():
            raise FileNotFoundError(source)
        output = RELEASE / f"{base_name}.tar.gz"
        with tarfile.open(output, "w:gz") as archive:
            archive.add(source, arcname=source.name, recursive=True)

    print(output)


if __name__ == "__main__":
    main()
