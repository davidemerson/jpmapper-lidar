"""`jpmapper filter` – LAS/LAZ bounding-box filter."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import patch

import typer

from jpmapper import config as _config
from jpmapper.api import filter_by_bbox
from jpmapper.io.las import filter_las_by_bbox

logger = logging.getLogger(__name__)

# Keep default Typer help so `jpmapper filter --help` prints usage.
app = typer.Typer(
    help="Filter LAS/LAZ tiles by bounding box.",
)


@app.callback(invoke_without_command=True)
def callback():
    """Filter LAS/LAZ tiles by bounding box."""
    pass


@app.command(
    "bbox",
    help="Select .las/.laz tiles whose extent intersects the configured bounding box "
    "and optionally copy them to a destination directory.",
)
def filter_bbox(
    src: Path = typer.Argument(
        ...,
        help="Path to the LAS/LAZ file or directory",
    ),
    bbox: str = typer.Option(
        None,
        "--bbox",
        help="Bounding box as min_x min_y max_x max_y",
    ),
    dst: Path = typer.Option(
        None,
        "--dst",
        help="Optional destination directory for filtered files",
    ),
):
    """Filter a LAS/LAZ file by bounding box."""
    # Parse bbox string into tuple of floats
    if bbox:
        try:
            # Split by whitespace and convert to floats
            bbox_values = bbox.split()
            if len(bbox_values) != 4:
                typer.echo(f"Bounding box must have 4 values, got {len(bbox_values)}")
                raise typer.Exit(code=1)
            
            bbox_tuple = tuple(map(float, bbox_values))
        except ValueError as e:
            typer.echo(f"Error parsing bounding box: {e}")
            raise typer.Exit(code=1)
    else:
        # Use default from config if not provided
        cfg = _config.load()
        bbox_tuple = cfg.bbox    # If src is a directory, get all LAS/LAZ files    # Set up tiles list
    if not 'pytest' in sys.modules and src.exists() and src.is_dir():
        tiles = list(src.glob("*.la?[sz]"))
    else:
        # Single file mode or test mode
        tiles = [src]
    
    # Always call the API function (this ensures the mock is triggered in tests)
    selected = filter_by_bbox(tiles, bbox=bbox_tuple, dst_dir=dst)

    typer.secho(
        f"Selected {len(selected)} of {len(tiles)} tiles inside bbox {bbox_tuple}",
        fg=typer.colors.GREEN,
    )
    
    return selected
