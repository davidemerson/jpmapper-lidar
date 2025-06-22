"""
jpmapper analyze – point-to-point link analysis on LiDAR DSM
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any

import numpy as np
import typer
from rich.table import Table

from jpmapper.api import analyze_los, generate_profile
from jpmapper.io import raster as r
from jpmapper.logging import console, setup as _setup_logging

try:
    import folium
except ModuleNotFoundError:  # --map-html is optional
    folium = None

logger = logging.getLogger(__name__)
_setup_logging()

app = typer.Typer(
    help="Analyse a CSV of point-to-point links against a first-return DSM.",
    add_help_option=True,
)


# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#
def _convert_to_raster_units(value_m: float, units: str) -> float:
    """Metres → native DSM linear unit (ftus or metre)."""
    if units.lower() in {"metre", "meter", "m"}:
        return value_m
    if units.lower() in {"ftus", "us_survey_foot", "us-ft"}:
        return value_m / 0.3048006096012192
    raise ValueError(f"Unknown unit {units!r}")


def _dsm_linear_unit(crs) -> str:  # CRS can be pyproj or rasterio CRS
    unit = crs.axis_info[0].unit_name.lower()
    return "ftus" if "foot" in unit else "m"


def _build_dsm(las_dir: Path, epsg: int, res_m: float, cache: Path) -> Tuple[Path, str]:
    """Build (or load) first-return DSM mosaic. Returns (path, linear_unit)."""
    crs = r.crs_for_epsg(epsg)
    units = _dsm_linear_unit(crs)
    res_native = _convert_to_raster_units(res_m, units)

    dsm_path = r.cached_mosaic(
        las_dir,
        cache,
        epsg=epsg,
        resolution=res_native,
        first_return=True,
    )
    return dsm_path, units


def _write_json(rows: List[dict], out: Path) -> None:
    with out.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    console.print(f"[green]JSON written → {out}[/green]")


def _write_map_html(records: List[dict], bounds: Tuple[Tuple[float, float], ...], out: Path) -> None:
    if folium is None:
        console.print("[red]folium not installed – skipping HTML map[/red]")
        return

    # bbox → map centre
    lats = [lat for lat, lon in bounds]
    lons = [lon for lat, lon in bounds]
    m = folium.Map(location=[statistics.mean(lats), statistics.mean(lons)], zoom_start=12)

    folium.Polygon(locations=bounds, color="blue", weight=2, fill=False).add_to(m)

    for rec in records:
        if rec["Clear"] == "NA":
            colour = "gray"
        elif rec["Clear"]:
            colour = "green"
        else:
            colour = "red"
        folium.PolyLine(
            [rec["A"], rec["B"]],
            color=colour,
            tooltip=f"#{rec['Idx']} {rec['Mast(m)']} m",
        ).add_to(m)

    m.save(out)
    console.print(f"[green]HTML map written → {out}[/green]")


# ---------------------------------------------------------------------------#
# CLI command
# ---------------------------------------------------------------------------#
@app.command("csv")
def analyze_csv(
    points_csv: Path = typer.Argument(..., exists=True, readable=True, help="CSV with point pairs"),
    las_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Directory containing LAS/LAZ tiles"),
    epsg: int = typer.Option(6539, help="Target EPSG (default NYC Long-Island ftUS)"),
    res: float = typer.Option(0.10, help="Desired DSM resolution (metres)"),
    max_mast: int = typer.Option(5, help="Maximum mast height to test (metres)"),
    step: int = typer.Option(1, help="Mast-height step (metres)"),
    json_out: Path | None = typer.Option(None, help="Write raw results to JSON"),
    map_html: Path | None = typer.Option(None, help="Write interactive map (requires folium)"),
) -> None:
    """Analyse every row in the CSV and print a Rich summary table."""

    # ------------------------------------------------------------------ DSM
    cache = las_dir.parent / "first_return_dsm.tif"
    console.print("[yellow]Building / loading first-return DSM…[/yellow]")
    dsm_path, units = _build_dsm(las_dir, epsg, res, cache)
    console.print(f"[cyan]DSM ready – units: {units.upper()} (EPSG {epsg})[/cyan]")

    # ------------------------------------------------------------------ Table set-up
    table = Table(title="Link Analysis", show_lines=True)
    cols = [
        "Idx",
        "Clear",
        "Mast(m)",
        "ClrMin(m)",
        "WorstOff(m)",
        "Samples",
        "Aground",
        "Bground",
        "Snap(m)",    ]
    for c in cols:
        table.add_column(c, justify="right")
        
    records: List[dict] = []
    # ------------------------------------------------------------------ Iterate CSV
    with points_csv.open() as fh, r.open_read(dsm_path) as ds:
        for idx, row in enumerate(csv.DictReader(fh), 1):
            pt_a = float(row["point_a_lat"]), float(row["point_a_lon"])
            pt_b = float(row["point_b_lat"]), float(row["point_b_lon"])
            freq = float(row.get("frequency_ghz", 5.8))
            
            try:
                # Use the new API
                los_result = analyze_los(
                    dsm_path,
                    pt_a,
                    pt_b,
                    freq_ghz=freq,
                    max_mast_height_m=max_mast,
                    mast_height_step_m=step,
                )
                
                result = los_result["clear"]
                mast = los_result["mast_height_m"]
                gA = los_result["ground_a_m"]
                gB = los_result["ground_b_m"]
                snap_d = los_result["snap_distance_m"]
                
                # profile() is relatively cheap – only run if we already have DSM open
                dist, ground, fresnel = generate_profile(dsm_path, pt_a, pt_b, 256, freq)

                worst_off = float(np.nanmax(ground - fresnel))
                clr_min = float(np.nanmin(fresnel - ground))
                samples = len(ground)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error processing row %d: %s", idx, exc)
                result = "NA"
                mast = clr_min = worst_off = samples = gA = gB = snap_d = "-"
            # ------------------- accumulate
            records.append(
                {
                    "Idx": idx,
                    "Clear": result,
                    "Mast(m)": mast,
                    "ClrMin(m)": clr_min,
                    "WorstOff(m)": worst_off,
                    "Samples": samples,
                    "Aground": gA,
                    "Bground": gB,
                    "Snap(m)": snap_d,
                    "A": pt_a,
                    "B": pt_b,
                }
            )
            table.add_row(
                str(idx),
                str(result),
                f"{mast:.0f}" if isinstance(mast, (int, float)) else str(mast),
                f"{clr_min:.1f}" if isinstance(clr_min, (int, float)) else str(clr_min),
                f"{worst_off:.1f}" if isinstance(worst_off, (int, float)) else str(worst_off),
                str(samples),
                f"{gA:.1f}" if isinstance(gA, (int, float)) else str(gA),
                f"{gB:.1f}" if isinstance(gB, (int, float)) else str(gB),
                f"{snap_d:.1f}" if isinstance(snap_d, (int, float)) else str(snap_d),
            )

    console.print(table)

    # ------------------------------------------------------------------ outputs
    if json_out:
        _write_json(records, json_out)

    if map_html:
        # DSM bounds polygon in WGS84
        bounds = r.dsm_bounds_latlon(dsm_path)
        _write_map_html(records, bounds, map_html)
