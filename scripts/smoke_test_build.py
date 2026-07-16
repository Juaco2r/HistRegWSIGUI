#!/usr/bin/env python3
"""Launch the packaged application in non-GUI self-test mode."""

from __future__ import annotations

import json
import os
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

    result_file = ROOT / "build" / "packaged-self-test.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.unlink(missing_ok=True)

    env = os.environ.copy()
    # Prevent an already-frozen parent environment from being inherited if this
    # script is ever called from another packaged process.
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"

    result = subprocess.run(
        [
            str(executable),
            "--self-test",
            "--self-test-output",
            str(result_file),
        ],
        cwd=executable.parent,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
        check=False,
    )

    if result.stdout:
        print("--- packaged stdout ---")
        print(result.stdout.rstrip())
    if result.stderr:
        print("--- packaged stderr ---", file=sys.stderr)
        print(result.stderr.rstrip(), file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "Packaged executable self-test failed with exit code "
            f"{result.returncode}. See stdout/stderr above."
        )

    if not result_file.exists():
        raise RuntimeError(
            "Packaged executable exited successfully but did not create the "
            f"self-test result file: {result_file}"
        )

    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid packaged self-test result: {exc}") from exc

    if payload.get("status") != "ok":
        raise RuntimeError(f"Packaged self-test reported failure: {payload}")
    if int(payload.get("preset_count", 0)) < 1:
        raise RuntimeError(f"No registration presets were packaged: {payload}")
    if payload.get("pillow_tkinter_finder") != "ok":
        raise RuntimeError(f"Pillow Tk bridge was not packaged: {payload}")
    if payload.get("pillow_tkinter_finder_alias") != "ok":
        raise RuntimeError(f"Pillow Tk compatibility alias was not packaged: {payload}")
    if payload.get("batch_registration") != "ok":
        raise RuntimeError(f"Batch registration support was not packaged: {payload}")
    if payload.get("cascading_registration") != "ok":
        raise RuntimeError(f"Cascading registration support was not packaged: {payload}")
    if payload.get("streamed_registration_downsample") != "ok":
        raise RuntimeError(f"Registration downsample support was not packaged: {payload}")
    if payload.get("if_he_registration_guides") != "ok":
        raise RuntimeError(f"IF/H&E registration-guide support was not packaged: {payload}")
    if payload.get("scientific_multichannel_zcyx") != "ok":
        raise RuntimeError(f"Scientific multichannel merge support was not packaged: {payload}")
    if payload.get("version") != "1.0":
        raise RuntimeError(f"Unexpected packaged application version: {payload}")
    if int(payload.get("supported_extension_count", 0)) < 10:
        raise RuntimeError(f"Image-format support table is incomplete: {payload}")

    print(json.dumps(payload, indent=2))
    print(f"Packaged executable self-test passed: {executable}")


if __name__ == "__main__":
    main()
