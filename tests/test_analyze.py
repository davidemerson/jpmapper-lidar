"""LOS/Fresnel test – skips rows outside DSM, lists mismatches."""
from pathlib import Path
import csv
import textwrap
import pytest

from jpmapper.analysis.los import is_clear
from jpmapper.io import raster as r


@pytest.fixture(scope="session")
def dsm() -> Path:
    las_dir = Path(__file__).parent / "data" / "las"
    cache = Path(__file__).parent / "data" / "nyc_dsm_cache.tif"
    if not any(las_dir.glob("*.las")):
        pytest.skip("no LAS fixtures")
    return r.cached_mosaic(las_dir, cache, epsg=6539, resolution=0.1, workers=None)


def test_links_match_expected(dsm: Path):
    points_csv = Path(__file__).parent / "data" / "points.csv"
    mismatches = []

    with points_csv.open() as fh:
        for idx, row in enumerate(csv.DictReader(fh), 1):
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            expected = row["expected_clear"].lower() == "true"

            result, mast, gA, gB, snap_d = is_clear(dsm, a, b, freq_ghz=5.8)

            if gA is None:  # outside DSM – ignore
                continue
            if result != expected:
                mismatches.append(
                    f"{idx:02d}: snap={snap_d:.1f} m  gA={gA:.1f} gB={gB:.1f} "
                    f"calc={result} mast={mast if mast>=0 else '>5'} CSV={expected}"
                )

    if mismatches:
        pytest.fail(
            "Rows where calculation disagrees with CSV:\n"
            + textwrap.indent("\n".join(mismatches), "  ")
        )
