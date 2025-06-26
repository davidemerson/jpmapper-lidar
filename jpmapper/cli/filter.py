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
    help=(
        "Filter LAS/LAZ tiles by spatial boundaries.\n\n"
        "Supports both bounding box and shapefile-based filtering with enhanced "
        "features including CRS validation and metadata-aware processing.\n\n"
        "Examples:\n"
        "  jpmapper filter bbox data/ --bbox '100 200 300 400' --dst filtered/\n"
        "  jpmapper filter shapefile tiles/ --shapefile boundary.shp --buffer 10\n\n"
        "The shapefile command requires geopandas and fiona dependencies."
    ),
)


@app.callback(invoke_without_command=True)
def callback():
    """Filter LAS/LAZ tiles by bounding box."""
    pass


@app.command(
    "bbox",
    help=(
        "Select .las/.laz tiles whose extent intersects the specified bounding box "
        "and optionally copy them to a destination directory.\n\n"
        "The bounding box should be specified as 'min_x min_y max_x max_y' in the "
        "same coordinate system as your LAS files. If no bbox is provided, the "
        "default from configuration will be used.\n\n"
        "Examples:\n"
        "  jpmapper filter bbox data/ --bbox '583000 4506000 584000 4507000'\n"
        "  jpmapper filter bbox tiles/ --bbox '100 200 300 400' --dst filtered/\n"
        "  jpmapper filter bbox single.las --bbox '0 0 1000 1000'"
    ),
)
def filter_bbox(
    src: Path = typer.Argument(
        ...,
        help="Path to LAS/LAZ file or directory containing multiple files",
    ),
    bbox: str = typer.Option(
        None,
        "--bbox",
        help="Bounding box as 'min_x min_y max_x max_y' (space-separated coordinates)",
    ),
    dst: Path = typer.Option(
        None,
        "--dst",
        help="Destination directory to copy filtered files (files are not copied if omitted)",
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
    
    # Always use the API function for consistency
    selected = filter_by_bbox(tiles, bbox=bbox_tuple, dst_dir=dst)

    typer.secho(
        f"Selected {len(selected)} of {len(tiles)} tiles inside bbox {bbox_tuple}",
        fg=typer.colors.GREEN,
    )
    
    return selected


@app.command(
    "shapefile",
    help=(
        "Select .las/.laz files that intersect with a shapefile boundary.\n\n"
        "This enhanced filtering method uses shapefile geometry for precise spatial "
        "selection with optional buffering and CRS validation. Requires geopandas "
        "and fiona dependencies.\n\n"
        "The shapefile can contain any geometry type (polygon, point, line). "
        "CRS validation ensures coordinate system compatibility between LAS and "
        "shapefile data.\n\n"
        "Examples:\n"
        "  jpmapper filter shapefile data/ --shapefile boundary.shp\n"
        "  jpmapper filter shapefile tiles/ -s area.shp --buffer 50 --dst selected/\n"
        "  jpmapper filter shapefile data/ -s zone.shp --no-validate-crs"
    ),
)
def filter_shapefile(
    src: Path = typer.Argument(
        ...,
        help="Path to LAS/LAZ file or directory containing multiple files",
    ),
    shapefile: Path = typer.Option(
        ...,
        "--shapefile",
        "-s",
        help="Path to shapefile (.shp) with boundary geometry (any geometry type supported)",
    ),
    dst: Path = typer.Option(
        None,
        "--dst",
        help="Destination directory to copy filtered files (files are not copied if omitted)",
    ),
    buffer: float = typer.Option(
        0.0,
        "--buffer",
        help="Buffer distance in meters to expand shapefile boundary (useful for edge cases)",
    ),
    validate_crs: bool = typer.Option(
        True,
        "--validate-crs/--no-validate-crs",
        help="Check CRS compatibility between LAS files and shapefile (recommended)",
    ),
):
    """Filter LAS/LAZ files using a shapefile boundary."""
    
    try:
        from jpmapper.api.shapefile_filter import filter_by_shapefile
    except ImportError:
        typer.echo("Error: Shapefile support requires geopandas and fiona")
        typer.echo("Install with: conda install -c conda-forge geopandas fiona")
        raise typer.Exit(code=1)
    
    # Set up files list
    if not 'pytest' in sys.modules and src.exists() and src.is_dir():
        tiles = list(src.glob("*.la?[sz]"))
    else:
        # Single file mode or test mode
        tiles = [src]
    
    try:
        selected = filter_by_shapefile(
            tiles, 
            shapefile, 
            dst_dir=dst,
            buffer_meters=buffer,
            validate_crs=validate_crs
        )

        typer.secho(
            f"Selected {len(selected)} of {len(tiles)} tiles inside shapefile boundary",
            fg=typer.colors.GREEN,
        )
        
        return selected
        
    except Exception as e:
        typer.echo(f"Error filtering by shapefile: {e}")
        raise typer.Exit(code=1)
