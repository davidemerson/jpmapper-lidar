"""Smoke-test rasterization on a real LAS fixture."""
from pathlib import Path
import pytest

from jpmapper.io import raster as r


def test_rasterize_smoke(tmp_path: Path):
    data_dir = Path(__file__).parent / "data" / "las"
    las_files = list(data_dir.glob("*.las"))

    if not las_files:
        pytest.skip("no LAS fixtures in tests/data/las/")

    src = las_files[0]
    dst = tmp_path / (src.stem + ".tif")

    # Use a known CRS (NY-Long-Island ftUS) instead of auto-detect.
    r.rasterize_tile(src, dst, epsg=6539, resolution=0.1)

    assert dst.exists() and dst.stat().st_size > 0
