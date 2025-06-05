"""Top‑level entry point for the `jpmapper` umbrella command."""
from __future__ import annotations

import importlib
import logging

import typer

from jpmapper.logging import setup as _setup_logging

# ---------------------------------------------------------------------------
# Initialise global logging as early as possible
# ---------------------------------------------------------------------------
_setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typer root application.  Sub‑commands are attached lazily to keep start‑up
# time low when users only need a single operation.
# ---------------------------------------------------------------------------
app = typer.Typer(
    help="JPMapper CLI – LiDAR filtering, rasterisation, and link‑analysis toolkit.",
    add_help_option=True,
)


def _lazy(module: str):  # helper to defer heavy imports
    return importlib.import_module(module)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """Display help when invoked without a sub‑command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Sub‑commands – currently only *filter*; rasterise/analyse coming next
# ---------------------------------------------------------------------------
app.add_typer(
    _lazy("jpmapper.cli.filter").app,
    name="filter",
    help="Filter LAS/LAZ tiles by bounding box",
)


if __name__ == "__main__":  # pragma: no cover
    app()
```python
"""CLI sub‑package exposed via *typer*."""