"""`jpmapper rasterize` – rasterize LAS/LAZ files to DSM GeoTIFFs."""
from __future__ import annotations

from pathlib import Path

import typer

from jpmapper.io.raster import rasterize_tile

app = typer.Typer(help="Rasterize LAS/LAZ to DSM GeoTIFF tiles.")


@app.command("tile", help="Rasterize a single LAS/LAZ tile.")
def tile_command(
    src: Path = typer.Argument(..., exists=True, readable=True, help="Source LAS/LAZ file"),
    dst: Path = typer.Argument(..., help="Destination GeoTIFF file (will be overwritten)"),
    epsg: int = typer.Option(6539, help="EPSG code for output CRS (default: 6539 – NAD83 / NY Long Island)"),
    resolution: float = typer.Option(1.0, help="Cell size in metres (default: 1.0)"),
):
    """CLI wrapper around :pyfunc:`jpmapper.io.raster.rasterize_tile`."""

    rasterize_tile(src, dst, epsg, resolution=resolution)