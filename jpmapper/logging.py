"""
Centralised logging helpers for JPMapper
----------------------------------------

* Always import *this* module before anything that logs.
* Exports a `console` Rich Console you can reuse for colourful output.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["setup", "console"]

# ---------------------------------------------------------------------------#
# Rich console usable throughout the code-base
# ---------------------------------------------------------------------------#
console: Final[Console] = Console(theme=None, highlight=False)


def setup(level: int | str = logging.INFO) -> None:
    """
    Configure root logging exactly **once**.

    Calling `setup()` multiple times is safe – further calls are ignored.
    """
    root = logging.getLogger()
    if getattr(root, "_jpmapper_configured", False):  # idempotent
        return

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    # Remove any existing handlers that *basicConfig* may have added.
    root.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        markup=False,
    )
    root.setLevel(level)
    root.addHandler(rich_handler)

    # Mark as configured so we don't do it again
    root._jpmapper_configured = True  # type: ignore[attr-defined]


# Auto-configure at import so early log messages aren’t lost
setup()
