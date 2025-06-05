"""End-to-end LOS analysis on real LAS + CSV fixtures."""
from pathlib import Path
import csv
import pytest

from jpmapper.analysis.los import is_clear
from jpmapper.io import raster as r


@pytest.fixture(scope="session")
def dsm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # <-- FIXED: now points to tests/data/las/
    las_dir = Path(__file__).parent / "data" / "las"
    if not las_dir.exists():
        pytest.skip("no LAS fixtures at tests/data/las/")

    tif_dir = tmp_path_factory.mktemp("dsm")
    tifs = []
    for las in las_dir.glob("*.las"):
        tif = tif_dir / f"{las.stem}.tif"
        r.rasterize_tile(las, tif, epsg=6539, resolution=0.1)
        tifs.append(tif)

    if not tifs:
        pytest.skip("no LAS tiles found to rasterize")

    mosaic = tif_dir / "mosaic.tif"
    r.merge_tiles(tifs, mosaic)
    return mosaic


def test_all_links_clear(dsm: Path):
    points_csv = Path(__file__).parent / "data" / "points.csv"
    with points_csv.open() as fh:
        for row in csv.DictReader(fh):
            a = (float(row["point_a_lat"]), float(row["point_a_lon"]))
            b = (float(row["point_b_lat"]), float(row["point_b_lon"]))
            assert is_clear(dsm, a, b, freq_ghz=float(row["frequency_ghz"]))
