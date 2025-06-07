"""Top-level entry point for the `jpmapper` umbrella command."""
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
# Typer root application
# ---------------------------------------------------------------------------
app = typer.Typer(
    help=(
        "JPMapper CLI – LiDAR filtering, rasterisation and link-analysis toolkit.  "
        "Use sub-commands such as `jpmapper filter`, `jpmapper rasterize`, "
        "`jpmapper analyze`, …"
    ),
    add_help_option=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lazy(module: str):  # defer heavy imports until the sub-command is invoked
    return importlib.import_module(module)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """Show help when invoked without a sub-command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Sub-commands  –  attach lazily so import-time stays fast
# ---------------------------------------------------------------------------
for _name in ("filter", "rasterize", "analyze"):
    try:
        mod = _lazy(f"jpmapper.cli.{_name}")
        app.add_typer(
            mod.app,
            name=_name,
            help=getattr(mod.app, "info", {}).help if hasattr(mod.app, "info") else None,
        )
    except ModuleNotFoundError:
        logger.warning("%s CLI not found – did you add jpmapper/cli/%s.py?", _name, _name)
        continue

if __name__ == "__main__":  # pragma: no cover
    app()
