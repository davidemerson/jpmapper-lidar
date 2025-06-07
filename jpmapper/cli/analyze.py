from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Tuple

import folium
import rasterio as rio
from rich.console import Console
from rich.table import Table

from jpmapper.analysis.los import analyze_link
from jpmapper.io import raster as r

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Analyse LOS/Fresnel clearance for point-to-point links.")


# … helper functions unchanged (see previous message) …


@app.command("csv")
def analyze_csv(
    points_csv: Path = typer.Argument(..., exists=True, readable=True),
    *,
    las_dir: Path = typer.Option(..., exists=True, file_okay=False),
    epsg: int = typer.Option(6539, help="EPSG of LiDAR tiles"),
    res: float = typer.Option(0.1, help="Desired DSM cell size (metres)"),
    max_mast: int = typer.Option(5, help="Max mast height to try (m)"),
    json_out: Path | None = typer.Option(None, "--json", help="Write JSON report"),
    map_html: Path | None = typer.Option(None, help="Save interactive HTML map"),
) -> None:
    # ── DSM build ────────────────────────────────────────────────────────────
    cache = las_dir.parent / "first_return_dsm.tif"
    console.print("[yellow]Building / loading first-return DSM…[/yellow]")
    dsm_path, units = _build_dsm(las_dir, epsg, res, cache)
    console.print(f"[cyan]DSM ready – units: {units.upper()} (EPSG {epsg})[/cyan]")

    # ── analysis table ───────────────────────────────────────────────────────
    table = Table(title="Link Analysis", show_lines=True)
    for c in (
        "Idx Clear Mast(m) ClrMin(m) WorstOff(m) Samples "
        "Aground Bground Snap(m) OOB(m)"
    ).split():
        table.add_column(c, justify="right")

    records = []
    m_points = []  # for optional map
    with rio.open(dsm_path) as ds, points_csv.open() as fh:
        for idx, row in enumerate(csv.DictReader(fh), 1):
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            f = float(row.get("frequency_ghz", 5.8))

            rec = analyze_link(
                ds,
                a,
                b,
                f,
                max_height_m=max_mast,
                raster_units=units,
            )
            records.append(rec)

            if not rec.get("coverage", True):
                table.add_row(
                    f"{idx}",
                    "NA",
                    *["-"] * 8,
                    f'{rec["oob_m"]:.0f}',
                )
                continue

            table.add_row(
                f"{idx}",
                "✔" if rec["clear"] else "✖",
                f'{rec["mast_height_m"]:+}',
                f'{rec["min_clearance_m"]:.1f}',
                f'{rec["worst_offset_m"]:.1f}',
                f'{rec["samples"]}',
                f'{rec["ground_a_m"]:.1f}',
                f'{rec["ground_b_m"]:.1f}',
                f'{rec["snap_distance_m"]:.1f}',
                "-",
            )
            m_points.append((*a, "A"))
            m_points.append((*b, "B"))

    console.print(table)

    # ── JSON export ──────────────────────────────────────────────────────────
    if json_out:
        with json_out.open("w") as jh:
            json.dump(records, jh, indent=2)
        console.print(f"[green]JSON written to {json_out}[/green]")

    # ── optional map ─────────────────────────────────────────────────────────
    if map_html:
        with rio.open(dsm_path) as ds:
            lon_min, lat_min, lon_max, lat_max = ds.bounds
            tf = rio.transform.xy(ds.transform, [0, ds.height], [0, ds.width])
        m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=11)
        # DSM footprint
        folium.Rectangle(
            [(lat_min, lon_min), (lat_max, lon_max)],
            color="blue",
            weight=2,
            fill=False,
            tooltip="DSM Extent",
        ).add_to(m)
        # Points
        for lat, lon, label in m_points:
            folium.CircleMarker(
                (lat, lon),
                radius=3,
                color="green",
                fill=True,
                tooltip=label,
            ).add_to(m)
        m.save(map_html)
        console.print(f"[green]Map saved to {map_html}[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
