"""
Package-wide logging helpers for JPMapper.

Importing this module sets up Rich-style logging immediately, *and* exposes
a shared `console` instance that the rest of the codebase can use for pretty
terminal output.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal, overload

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["console", "setup"]

# --------------------------------------------------------------------------- #
# A single Rich console for the whole project
# --------------------------------------------------------------------------- #
console = Console()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@overload
def setup(
    level: int | str = "INFO",
    *,
    logfile: str | Path | None = None,
    force: bool = False,
) -> None: ...
@overload
def setup(
    level: Literal[10, 20, 30, 40, 50] = 20,
    *,
    logfile: str | Path | None = None,
    force: bool = False,
) -> None: ...
def setup(  # type: ignore[override]
    level: int | str = "INFO",
    *,
    logfile: str | Path | None = None,
    force: bool = False,
) -> None:
    """
    Initialise Rich logging.

    Parameters
    ----------
    level
        Log-level name or numeric value (default ``"INFO"``).
    logfile
        Optional path – if given, adds a rotating file handler alongside Rich.
    force
        When running inside e.g. pytest (which configures logging itself) set
        ``force=True`` so our config replaces the existing root handlers.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    # Rich handler for colourful terminal logs
    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=False,
            show_path=False,
            log_time_format="[%d/%m/%y %H:%M:%S]",
        )
    ]

    # Optional plain-text file log
    if logfile is not None:
        fh = logging.handlers.RotatingFileHandler(
            logfile, mode="a", maxBytes=2**20, backupCount=3, encoding="utf-8"
        )
        fmt_file = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        fh.setFormatter(logging.Formatter(fmt_file, datefmt="%Y-%m-%d %H:%M:%S"))
        handlers.append(fh)

    logging.basicConfig(
        level=level,
        format="%(message)s",  # Rich formats the record itself
        handlers=handlers,
        force=force,
    )


# Initialise at import time so `import jpmapper.logging` is enough
setup()
