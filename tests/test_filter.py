"""Smoke-test the bounding-box filter using a real LAS fixture when available."""
from pathlib import Path
import pytest
import laspy

from jpmapper.io.las import filter_las_by_bbox


def test_filter_selects_some(tmp_path: Path):
    data_dir = Path(__file__).parent / "data"
    las = data_dir / "sample.las"

    if not las.exists():
        pytest.skip("sample LAS fixture not present")

    # Read the LAS header to build a tight bbox around the file itself
    with laspy.open(str(las)) as reader:
        h = reader.header
        bbox = (h.mins[0], h.mins[1], h.maxs[0], h.maxs[1])

    selected = filter_las_by_bbox([las], bbox=bbox)
    assert las in selected
