"""Top‑level package for jpmapper."""
from __future__ import annotations

__all__: list[str] = [
    "__version__",
    "api",
    "exceptions",
]

__version__ = "0.1.0"

# Import the API so it can be accessed as jpmapper.api
from jpmapper import api
from jpmapper import exceptions