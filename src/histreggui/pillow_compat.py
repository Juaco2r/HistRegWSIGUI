from __future__ import annotations

"""Pillow/Tk compatibility helpers for source and PyInstaller builds.

Pillow's private Tk discovery module is named ``PIL._tkinter_finder``.  Some
older code and packaging diagnostics refer to ``PIL.tkinter_finder`` without
the leading underscore.  Registering an alias keeps both names working while
the PyInstaller hook bundles the real module.
"""

import importlib
import sys
from types import ModuleType


def install_pillow_tkinter_finder_alias() -> ModuleType:
    """Import Pillow's Tk finder and expose the legacy non-underscore alias.

    Returns the imported module.  A clear RuntimeError is raised if the Pillow
    installation truly does not provide the finder, rather than allowing a
    less useful ``ModuleNotFoundError`` later when ``ImageTk`` is initialized.
    """

    try:
        module = importlib.import_module("PIL._tkinter_finder")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow's Tk compatibility module PIL._tkinter_finder is missing. "
            "Reinstall Pillow or use the official packaged application."
        ) from exc

    sys.modules.setdefault("PIL.tkinter_finder", module)
    return module
