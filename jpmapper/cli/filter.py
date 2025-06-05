"""`jpmapper filter` sub‑command – thin wrapper over io.las.filter_las_by_bbox."""
from __future__ import annotations

import logging
from pathlib import Path

import typer

from jpmapper import config as _config
from jpmapper.io.las import filter_las_by_bbox

logger = logging.getLogger(__name__)
app = typer.Typer(add_help_option=False)


@app.command("", help="Filter LAS/LAZ tiles by bounding box", add_help_option=True)
def cmd(
    src: Path = typer.Argument(..., exists=True, help="Directory containing .las/.laz tiles"),
    dst: Path | None = typer.Option(None, "--dst", help="Optional destination for filtered tiles"),
):
    cfg = _config.load()
    bbox = cfg.bbox

    tiles = list(src.glob("*.la?[sz]"))
    selected = filter_las_by_bbox(tiles, bbox=bbox, dst_dir=dst)

    typer.echo(f"Selected {len(selected)} of {len(tiles)} tiles inside bbox {bbox}")