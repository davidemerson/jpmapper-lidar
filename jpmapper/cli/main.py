"""Top-level entry point for the `jpmapper` umbrella command."""
from __future__ import annotations

import importlib
import logging

import typer

from jpmapper.logging import setup as _setup_logging

# ---------------------------------------------------------------------------
# Initialise global logging ASAP
# ---------------------------------------------------------------------------
_setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root Typer application
# ---------------------------------------------------------------------------
app = typer.Typer(
    help=(
        "JPMapper CLI – LiDAR filtering, rasterization and link-analysis toolkit.\n\n"
        "Sub-commands:\n"
        "  • jpmapper filter      – clip LAS/LAZ tiles by bounding-box\n"
        "  • jpmapper rasterize   – first-return DSM generation\n"
        "  • jpmapper analyze     – LOS/Fresnel analysis for point-to-point links"
    ),
    add_help_option=True,
)


def _lazy(module: str):  # helper to defer heavy imports
    return importlib.import_module(module)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:  # noqa: D401
    """Show help when invoked without a sub-command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Register sub-commands – attached lazily so the CLI stays snappy
# ---------------------------------------------------------------------------
for name in ("filter", "rasterize", "analyze"):
    try:
        mod = _lazy(f"jpmapper.cli.{name}")
        app.add_typer(mod.app, name=name, help=mod.app.info.help)
    except ModuleNotFoundError:
        logger.warning("%s CLI not found – did you add jpmapper/cli/%s.py?", name, name)


if __name__ == "__main__":  # pragma: no cover
    app()
