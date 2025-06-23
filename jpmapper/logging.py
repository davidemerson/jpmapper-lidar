"""
Centralised logging helpers for JPMapper
----------------------------------------

* Always import *this* module before anything that logs.
* Exports a `console` Rich Console you can reuse for colourful output.
"""

from __future__ import annotations

import logging
import sys
from typing import Final, Optional, TextIO, Union

from rich.console import Console
from rich.logging import RichHandler

__all__ = ["setup", "console", "get_logger", "set_log_level"]

# ---------------------------------------------------------------------------#
# Rich console usable throughout the code-base
# ---------------------------------------------------------------------------#
console: Final[Console] = Console(theme=None, highlight=False)


def setup(level: Union[int, str] = logging.INFO) -> None:
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


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name. If None, returns the 'jpmapper' root logger.

    Returns:
        Configured logger instance
    """
    if name is None:
        return logging.getLogger("jpmapper")
    return logging.getLogger(f"jpmapper.{name}")


def set_log_level(level: Union[int, str], stream: Optional[TextIO] = None, 
                 format: Optional[str] = None) -> None:
    """
    Set the log level for the root logger.

    Args:
        level: Log level (either as integer or string like 'INFO', 'DEBUG', etc.)
        stream: Optional stream to output logs to (e.g., sys.stdout)
        format: Optional log format string
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    
    # Use basicConfig for compatibility with the tests
    kwargs = {'level': level}
    if stream is not None:
        kwargs['stream'] = stream
    if format is not None:
        kwargs['format'] = format
        
    logging.basicConfig(**kwargs)

# Auto-configure at import so early log messages aren’t lost
setup()
