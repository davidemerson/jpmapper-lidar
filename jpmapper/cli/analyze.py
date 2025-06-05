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


def _build_temp_dsm(las_dir: Path) -> Path:
    tmp = Path(tempfile.mkdtemp())
    tif_paths = []
    for las in las_dir.glob("*.las"):
        tif = tmp / (las.stem + ".tif")
        r.rasterize_tile(las, tif, epsg=6539, resolution=0.1)
        tif_paths.append(tif)
    dsm = tmp / "mosaic.tif"
    r.merge_tiles(tif_paths, dsm)
    return dsm


@app.command()
def csv(
    points_csv: Path = typer.Argument(
        ..., exists=True, readable=True, help="CSV with point pairs"
    ),
    las_dir: Optional[Path] = typer.Option(
        None,
        "--las-dir",
        exists=True,
        readable=True,
        help="Folder of LAS tiles; builds DSM mosaic on-the-fly if provided.",
    ),
    dsm: Optional[Path] = typer.Option(
        None,
        "--dsm",
        exists=True,
        readable=True,
        help="Pre-built DSM GeoTIFF; skips rasterization step.",
    ),
    freq: float = typer.Option(5.8, "--freq", help="Frequency in GHz"),
    json_out: Optional[Path] = typer.Option(
        None, "--json", help="Write summary JSON to file"
    ),
):
    """Process CSV file with columns point_a_lat,…,frequency_ghz,expected_clear."""
    if dsm is None:
        if las_dir is None:
            raise typer.BadParameter("Either --dsm or --las-dir must be supplied")
        typer.echo("Building temporary DSM from LAS directory…")
        dsm = _build_temp_dsm(las_dir)

    results = []
    with points_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            ok = is_clear(dsm, a, b, freq_ghz=freq)
            results.append({"a": a, "b": b, "clear": ok})

            typer.echo(
                f"{a} → {b}: {'CLEAR' if ok else 'BLOCKED'} "
                f"(expected {row.get('expected_clear')})"
            )

    if json_out:
        json_out.write_text(json.dumps(results, indent=2))
        typer.echo(f"Wrote JSON results to {json_out}")
