"""
`jpmapper analyze` – batch line-of-sight / Fresnel analysis for point pairs.

Usage example
-------------
jpmapper analyze csv tests/data/points.csv \
        --las-dir tests/data/las \
        --json report.json \
        --map-html report.html
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
from rich.table import Table
from rich.text import Text

from jpmapper.logging import console
from jpmapper.analysis.los import is_clear
from jpmapper.io import raster as r

app = typer.Typer(add_help_option=True, no_args_is_help=True)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _dsm_linear_unit(crs) -> str:
    # Returns e.g. "ftus" or "metre"
    return crs.axis_info[0].unit_name.lower().replace(" ", "")


def _convert_to_raster_units(metres: float, units: str) -> float:
    """Convert *metres* to the DSM's linear units."""
    if units in ("m", "metre", "meter"):
        return metres
    # US-survey-ft is 0.304800609601 m
    if units in ("us_survey_foot", "usfoot", "ftus", "foot_us"):
        return metres / 0.304800609601
    raise ValueError(f"Unknown linear unit {units!r}")


def _build_dsm(
    las_dir: Path,
    epsg: int,
    res_m: float,
    cache: Path,
) -> Tuple[Path, str]:
    """Build (or load cached) first-return DSM mosaic and return path + units."""
    crs = r._crs_for_epsg(epsg)
    units = _dsm_linear_unit(crs)
    res_native = _convert_to_raster_units(res_m, units)

    dsm_path = r.cached_mosaic(
        las_dir=las_dir,
        cache=cache,
        epsg=epsg,
        resolution=res_native,
        first_return=True,
    )
    return dsm_path, units


def _rich_table() -> Table:
    tbl = Table(title="Link Analysis")
    tbl.add_column("Idx", justify="right")
    tbl.add_column("Clear", justify="center")
    tbl.add_column("Mast(m)", justify="right")
    tbl.add_column("ClrMin(m)", justify="right")
    tbl.add_column("WorstOff(m)", justify="right")
    tbl.add_column("Samples", justify="right")
    tbl.add_column("Aground", justify="right")
    tbl.add_column("Bground", justify="right")
    tbl.add_column("Snap(m)", justify="right")
    return tbl


# --------------------------------------------------------------------------- #
# CSV sub-command
# --------------------------------------------------------------------------- #
@app.command("csv", help="Analyse every row in a CSV of point pairs.")
def analyze_csv(
    points_csv: Path = typer.Argument(..., exists=True, readable=True),
    *,
    las_dir: Path = typer.Option(..., exists=True, file_okay=False),
    epsg: int = typer.Option(6539, help="EPSG code for LAS data / DSM"),
    res: float = typer.Option(0.1, help="Desired cell size in **metres**"),
    max_mast: int = typer.Option(5, help="Maximum mast height to try (m)"),
    json_out: Path | None = typer.Option(
        None, "--json", help="Write results as JSON list to this file"
    ),
    map_html: Path | None = typer.Option(
        None, "--map-html", help="Create an interactive Leaflet map"
    ),
) -> None:
    # --------------------------------------------------------------------- build/load DSM
    cache = las_dir.parent / "first_return_dsm.tif"
    console.print("[yellow]Building / loading first-return DSM…[/yellow]")
    dsm_path, units = _build_dsm(las_dir, epsg, res, cache)
    console.print(f"[cyan]DSM ready – units: {units.upper()} (EPSG {epsg})[/cyan]")

    # --------------------------------------------------------------------- iterate rows
    tbl = _rich_table()
    records: List[Dict[str, Any]] = []

    with points_csv.open(newline="") as fh, r.open_dsm(dsm_path) as ds:
        for idx, row in enumerate(csv.DictReader(fh), 1):
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            f_ghz = float(row.get("frequency_ghz", 5.8))

            try:
                clear, mast, clr_min, worst_off, n, g_a, g_b, snap = is_clear(
                    ds, a, b, f_ghz, max_height_m=max_mast
                )
                colour = "green" if clear else "red"
                tbl.add_row(
                    f"{idx}", Text("✔" if clear else "✘", style=colour),
                    f"{mast:+.1f}", f"{clr_min:.1f}", f"{worst_off:+.1f}",
                    str(n), f"{g_a:.1f}", f"{g_b:.1f}", f"{snap:.1f}"
                )
            except r.PointOutOfBounds as exc:
                # Off-DSM ➜ warn & mark N/A
                console.print(
                    f"[yellow]Row {idx}: {exc} – skipping (outside DSM)[/yellow]"
                )
                tbl.add_row(f"{idx}", "NA", *["-"] * 7)
                clear = False
                mast = clr_min = worst_off = n = g_a = g_b = snap = None

            records.append(
                dict(
                    idx=idx,
                    point_a=a,
                    point_b=b,
                    freq_ghz=f_ghz,
                    clear=clear,
                    mast_m=mast,
                    clearance_min_m=clr_min,
                    worst_obstruction_m=worst_off,
                    samples=n,
                    ground_a_m=g_a,
                    ground_b_m=g_b,
                    snap_m=snap,
                )
            )

    # --------------------------------------------------------------------- output
    console.print(tbl)

    if json_out:
        json_out.write_text(json.dumps(records, indent=2))
        console.print(f"[green]JSON written → {json_out}[/green]")

    if map_html:
        from jpmapper.reporting.map import write_map  # lazy import
        write_map(records, dsm_path, map_html)
        console.print(f"[green]Map written → {map_html}[/green]")
