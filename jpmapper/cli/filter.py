"""`jpmapper filter` – LAS/LAZ bounding-box filter."""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from jpmapper import config as _config
from jpmapper.api import filter_by_bbox

logger = logging.getLogger(__name__)

# Keep default Typer help so `jpmapper filter --help` prints usage.
app = typer.Typer(
    help="Filter LAS/LAZ tiles by bounding box.",
)


@app.command(
    "bbox",
    help="Select .las/.laz tiles whose extent intersects the configured bounding box "
    "and optionally copy them to a destination directory.",
)
def bbox_command(
    src: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Directory containing .las/.laz tiles",
    ),
    dst: Path | None = typer.Option(
        None,
        "--dst",
        help="Optional destination directory for filtered tiles",
    ),
):
    """Run the bounding-box filter on *src*."""
    cfg = _config.load()
    bbox = cfg.bbox

    tiles = list(src.glob("*.la?[sz]"))
    selected = filter_by_bbox(tiles, bbox=bbox, dst_dir=dst)

    typer.secho(
        f"Selected {len(selected)} of {len(tiles)} tiles inside bbox {bbox}",
        fg=typer.colors.GREEN,
    )
