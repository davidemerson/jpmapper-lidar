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
    help=(
        "JPMapper CLI – LiDAR filtering, rasterization, and link‑analysis "
        "toolkit. Use sub‑commands like `jpmapper filter`, `jpmapper rasterize`, "
        "and (soon) `jpmapper analyze`."
    ),
    add_help_option=True,
)


def _lazy(module: str):  # helper to defer heavy imports
    """Import *module* only when its command is first invoked."""
    return importlib.import_module(module)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """Show help when invoked without a sub‑command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Sub‑commands
# ---------------------------------------------------------------------------
# Filter LAS/LAZ tiles by bounding box
a = _lazy("jpmapper.cli.filter").app  # noqa: N816 – single‑letter alias
app.add_typer(a, name="filter", help=getattr(a, "info", {}).help if hasattr(a, "info") else None)

# Rasterize command (only tile sub‑command right now)
try:
    r = _lazy("jpmapper.cli.rasterize").app  # noqa: N816
    app.add_typer(r, name="rasterize", help=getattr(r, "info", {}).help if hasattr(r, "info") else None)
except ModuleNotFoundError:  # rasterize module not yet implemented
    logger.debug("rasterize CLI not available yet – skipped")


if __name__ == "__main__":  # pragma: no cover
    app()