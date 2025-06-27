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
        "JPMapper CLI – Advanced LiDAR processing toolkit for filtering, "
        "rasterisation, and point-to-point link analysis.\n\n"
        "Features metadata-aware rasterization, shapefile-based filtering, "
        "and comprehensive line-of-sight analysis for RF planning.\n\n"
        "Quick start:\n"
        "  jpmapper filter bbox data/ --bbox '100 200 300 400'\n"
        "  jpmapper rasterize tile input.las output.tif\n"
        "  jpmapper analyze csv links.csv --las-dir data/\n\n"
        "For detailed help on any command, use: jpmapper COMMAND --help"
    ),
    add_help_option=True,
)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):  # noqa: D401
    """Show help if invoked without a sub-command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ────────────────────────────────────────────────────────────────────────────
app.add_typer(_lazy("jpmapper.cli.filter").app, name="filter")
app.add_typer(_lazy("jpmapper.cli.rasterize").app, name="rasterize")
app.add_typer(_lazy("jpmapper.cli.analyze").app, name="analyze")

@app.command()
def debug_dsm(
    dsm_path: str = typer.Argument(..., help="Path to DSM file"),
    coords: str = typer.Argument(..., help="Coordinates to test as 'x,y' or multiple as 'x1,y1;x2,y2'"),
) -> None:
    """Debug DSM sampling at specific coordinates."""
    import rasterio
    import numpy as np
    
    coord_pairs = []
    for coord_str in coords.split(';'):
        x, y = map(float, coord_str.split(','))
        coord_pairs.append((x, y))
    
    print(f"🔍 Testing DSM sampling at {len(coord_pairs)} coordinate(s)")
    print(f"📁 DSM file: {dsm_path}")
    
    try:
        with rasterio.open(dsm_path) as ds:
            print(f"📊 DSM info:")
            print(f"   Shape: {ds.shape}")
            print(f"   Bounds: {ds.bounds}")
            print(f"   CRS: {ds.crs}")
            print(f"   NoData: {ds.nodata}")
            print(f"   Transform: {ds.transform}")
            
            # Sample a small area to check nodata distribution
            sample_size = min(1000, ds.width, ds.height)
            sample_data = ds.read(1, window=((0, sample_size), (0, sample_size)))
            nodata_pct = 100 * np.sum(sample_data == ds.nodata) / sample_data.size
            print(f"   NoData %: {nodata_pct:.1f}% (in {sample_size}x{sample_size} sample)")
            
            print(f"\n🎯 Testing coordinates:")
            for i, (x, y) in enumerate(coord_pairs):
                print(f"  {i+1}. ({x:.1f}, {y:.1f})")
                
                # Check bounds
                in_bounds = (ds.bounds.left <= x <= ds.bounds.right and 
                           ds.bounds.bottom <= y <= ds.bounds.top)
                print(f"     In bounds: {in_bounds}")
                
                if in_bounds:
                    # Sample elevation
                    try:
                        sampled_values = list(ds.sample([(x, y)], 1))
                        elevation = sampled_values[0][0]
                        is_nodata = (ds.nodata is not None and elevation == ds.nodata)
                        print(f"     Elevation: {elevation:.1f}m {'(NODATA)' if is_nodata else '(valid)'}")
                        
                        # Try pixel coordinates
                        row, col = ds.index(x, y)
                        print(f"     Pixel: row={row}, col={col}")
                        
                    except Exception as e:
                        print(f"     Error sampling: {e}")
                else:
                    print(f"     Outside bounds: bounds are {ds.bounds}")
    
    except Exception as e:
        print(f"❌ Error opening DSM: {e}")

# ────────────────────────────────────────────────────────────────────────────
# Load commands conditionally
# ────────────────────────────────────────────────────────────────────────────
app.add_typer(_lazy("jpmapper.cli.filter").app, name="filter")
app.add_typer(_lazy("jpmapper.cli.rasterize").app, name="rasterize")
app.add_typer(_lazy("jpmapper.cli.analyze").app, name="analyze")
