"""Top-level entry point for `jpmapper` CLI."""
from __future__ import annotations

import importlib
import logging
from typing import Final

import typer

from jpmapper.logging import setup as _setup_logging

# ────────────────────────────────────────────────────────────────────────────
# Initialise global Rich logging *once*
# ────────────────────────────────────────────────────────────────────────────
_setup_logging()
logger: Final = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Helper: lazy import to keep start-up fast
# ────────────────────────────────────────────────────────────────────────────
def _lazy(module: str):  # noqa: D401
    """Return imported *module* only on first access."""
    return importlib.import_module(module)


# ────────────────────────────────────────────────────────────────────────────
# Root Typer application
# ────────────────────────────────────────────────────────────────────────────
app = typer.Typer(
    help=(
        "JPMapper CLI – LiDAR filtering, rasterisation, and "
        "point-to-point link analysis toolkit."
    ),
    add_help_option=True,
)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """Show help if invoked without a sub-command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ────────────────────────────────────────────────────────────────────────────
# Sub-commands – attached lazily
# ────────────────────────────────────────────────────────────────────────────
for _name in ("filter", "rasterize", "analyze"):
    try:
        mod = _lazy(f"jpmapper.cli.{_name}")
        app.add_typer(mod.app, name=_name, help=mod.app.info.help)  # type: ignore[attr-defined]
    except ModuleNotFoundError:
        logger.warning("%s CLI not found – did you add jpmapper/cli/%s.py?", _name, _name)
