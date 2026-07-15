#!/usr/bin/env python3
"""Launch the packaged application in non-GUI self-test mode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def executable_path() -> Path:
    if sys.platform == "darwin":
        return ROOT / "dist" / "HistRegGUI.app" / "Contents" / "MacOS" / "HistRegGUI"
    if sys.platform.startswith("win"):
        return ROOT / "dist" / "HistRegGUI" / "HistRegGUI.exe"
    return ROOT / "dist" / "HistRegGUI" / "HistRegGUI"


def main() -> None:
    executable = executable_path()
    if not executable.exists():
        raise FileNotFoundError(f"Packaged executable not found: {executable}")

    result = subprocess.run(
        [str(executable), "--self-test"],
        cwd=executable.parent,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Packaged executable self-test failed with exit code {result.returncode}."
        )
    print(f"Packaged executable self-test passed: {executable}")


if __name__ == "__main__":
    main()
