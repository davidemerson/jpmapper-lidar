"""Load configuration from ~/.jpmapper.yml, env vars, and CLI defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

_CONFIG_ENV = "JPMAPPER_CONFIG_FILE"
_DEFAULT_PATH = Path.home() / ".jpmapper.yml"


@dataclass(slots=True)
class Config:  # extend as needed
    bbox: tuple[float, float, float, float] = (-74.47, 40.48, -73.35, 41.03)  # NYC default
    epsg: int = 6539  # NY‑Long‐Island ftUS


_DEF = Config()


def load() -> Config:
    path = Path(os.getenv(_CONFIG_ENV, str(_DEFAULT_PATH)))
    if not path.exists():
        return _DEF

    with path.open("r", encoding="utf8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return Config(
        bbox=tuple(raw.get("bbox", _DEF.bbox)),
        epsg=int(raw.get("epsg", _DEF.epsg)),
    )