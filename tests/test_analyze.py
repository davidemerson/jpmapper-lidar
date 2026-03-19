"""LOS/Fresnel test – skips rows outside DSM, lists mismatches."""
from pathlib import Path
import csv
import textwrap
import pytest

from jpmapper.analysis.los import is_clear
from jpmapper.exceptions import NoDataError
from jpmapper.io import raster as r


@pytest.fixture(scope="session")
def dsm() -> Path:
    las_dir = Path(__file__).parent / "data" / "las"
    cache = Path(__file__).parent / "data" / "nyc_dsm_cache.tif"
    if not any(las_dir.glob("*.las")):
        pytest.skip("no LAS fixtures")
    try:
        return r.cached_mosaic(las_dir, cache, epsg=6539, resolution=0.1, workers=None)
    except Exception as e:
        pytest.skip(f"Could not create DSM (pdal not available?): {e}")


def test_links_match_expected(dsm: Path):
    points_csv = Path(__file__).parent / "data" / "points.csv"
    mismatches = []

    with points_csv.open() as fh:
        for idx, row in enumerate(csv.DictReader(fh), 1):
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            expected = row["expected_clear"].lower() == "true"

            try:
                result, mast, gA, gB, snap_d = is_clear(dsm, a, b, freq_ghz=5.8)
            except NoDataError:
                continue  # outside DSM coverage

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
