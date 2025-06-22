"""`jpmapper rasterize` – rasterize LAS/LAZ files to DSM GeoTIFFs."""
from __future__ import annotations

from pathlib import Path

import typer

from jpmapper.api import rasterize_tile

app = typer.Typer(help="Rasterize LAS/LAZ to DSM GeoTIFF tiles.")


@app.command("tile", help="Rasterize a single LAS/LAZ tile.")
def tile_command(
    src: Path = typer.Argument(..., exists=True, readable=True, help="Source LAS/LAZ file"),
    dst: Path = typer.Argument(..., help="Destination GeoTIFF file (overwritten if exists)"),
    epsg: int | None = typer.Option(
        None,
        "--epsg",
        help="EPSG code for output CRS. If omitted, CRS is auto-detected from LAS header.",
    ),
    resolution: float = typer.Option(
        0.1,
        "--resolution",
        help="Cell size in metres (default: 0.1 m)",
    ),
):
    """CLI wrapper around :pyfunc:`jpmapper.io.raster.rasterize_tile`."""
    rasterize_tile(src, dst, epsg, resolution=resolution)
