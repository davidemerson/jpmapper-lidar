"""`jpmapper analyze` – batch LOS / Fresnel analysis."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Optional

import typer

from jpmapper.analysis.los import is_clear
from jpmapper.io import raster as r

app = typer.Typer(help="Analyze LOS / Fresnel clearance for link pairs.")


def _build_temp_dsm(las_dir: Path, epsg: int, res: float, workers: int | None) -> Path:
    tmp = Path(tempfile.mkdtemp())
    tif_dir = tmp / "tiles"
    tif_dir.mkdir()
    tifs = r.rasterize_dir_parallel(
        las_dir, tif_dir, epsg=epsg, resolution=res, workers=workers
    )
    dsm = tmp / "mosaic.tif"
    r.merge_tiles(tifs, dsm)
    return dsm


@app.command()
def csv(
    points_csv: Path = typer.Argument(..., exists=True, readable=True),
    las_dir: Optional[Path] = typer.Option(
        None, "--las-dir", help="Folder of LAS tiles (builds DSM if given)"
    ),
    dsm: Optional[Path] = typer.Option(
        None, "--dsm", exists=True, readable=True, help="Pre-built DSM"
    ),
    freq: float = typer.Option(5.8, "--freq", help="GHz (default 5.8)"),
    epsg: int = typer.Option(6539, "--epsg", help="CRS for rasterization"),
    res: float = typer.Option(0.1, "--resolution", help="DSM cell size (m)"),
    workers: int | None = typer.Option(
        None, "--workers", help="Parallel rasterization processes (default = cpu count)"
    ),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write JSON summary"),
):
    """CSV must contain cols: point_a_lat, point_a_lon, point_b_lat, point_b_lon."""
    if dsm is None:
        if las_dir is None:
            raise typer.BadParameter("Need either --dsm or --las-dir")
        typer.echo("⏳  Rasterizing LAS tiles …")
        dsm = _build_temp_dsm(las_dir, epsg, res, workers)
        typer.echo(f"✅  Built DSM {dsm}")

    results = []
    with points_csv.open() as fh:
        for row in csv.DictReader(fh):
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            clear = is_clear(dsm, a, b, freq)
            results.append({"a": a, "b": b, "clear": clear})
            typer.echo(f"{a} ➜ {b} : {'CLEAR' if clear else 'BLOCKED'}")

    if json_out:
        json_out.write_text(json.dumps(results, indent=2))
        typer.echo(f"Wrote JSON → {json_out}")
