"""Smoke-test rasterization on a tiny LAS stub."""
from pathlib import Path
import pytest

from jpmapper.io import raster as r


def test_rasterize_smoke(tmp_path: Path):
    data_dir = Path(__file__).parent / "data"
    src = data_dir / "sample.las"
    if not src.exists():
        pytest.skip("sample LAS fixture not present")

    dst = tmp_path / "sample.tif"
    r.rasterize_tile(src, dst, epsg=4326, resolution=2)
    assert dst.exists() and dst.stat().st_size > 0
