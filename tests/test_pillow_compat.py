from __future__ import annotations

import sys

from histreggui.pillow_compat import install_pillow_tkinter_finder_alias


def test_pillow_tkinter_finder_alias_is_available() -> None:
    module = install_pillow_tkinter_finder_alias()
    assert module.__name__ == "PIL._tkinter_finder"
    assert sys.modules["PIL.tkinter_finder"] is module
