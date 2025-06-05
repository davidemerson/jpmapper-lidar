"""Centralized logging setup using *rich* for pretty CLI output and JSON option."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Final

from rich.console import Console
from rich.logging import RichHandler

_JSON_ENV: Final[str] = "JPMAPPER_LOG_JSON"

DEFAULT_LEVEL = logging.INFO


def _build_rich_handler() -> RichHandler:  # pretty coloured output
    return RichHandler(rich_tracebacks=True, show_path=False)


def _build_json_handler() -> logging.Handler:  # machine‑readable logs
    class _JSON(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            log = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "msg": record.getMessage(),
                "logger": record.name,
            }
            print(json.dumps(log, separators=(",", ":")))

    return _JSON()


def setup() -> None:
    """Initialise root logger once; idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured

    json_mode = os.getenv(_JSON_ENV, "0") in {"1", "true", "yes"}
    handler: logging.Handler = _build_json_handler() if json_mode else _build_rich_handler()

    logging.basicConfig(
        level=DEFAULT_LEVEL,
        format="%(message)s",  # actual formatting done by handler
        handlers=[handler],
    )