"""Smoke-test bbox filtering using whatever LAS file is in tests/data/las/."""
from pathlib import Path
import pytest
import laspy

from jpmapper.io.las import filter_las_by_bbox


def test_filter_selects_some(tmp_path: Path):
    data_dir = Path(__file__).parent / "data" / "las"
    las_files = list(data_dir.glob("*.las"))

    if not las_files:
        pytest.skip("no LAS fixtures in tests/data/las/")

    las = las_files[0]

    # Try to derive a bbox from the header; fall back to a giant bbox.
    try:
        with laspy.open(las) as r:
            h = r.header
            bbox = (h.mins[0], h.mins[1], h.maxs[0], h.maxs[1])
    except Exception:  # header unreadable or no mins/maxs
        bbox = (-1e9, -1e9, 1e9, 1e9)

    selected = filter_las_by_bbox([las], bbox=bbox)
    assert las in selected
