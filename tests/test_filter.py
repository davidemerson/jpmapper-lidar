from __future__ import annotations

from pathlib import Path

from jpmapper.io.las import filter_las_by_bbox


def test_filter_selects_some(tmp_path: Path):
    # dummy tiny LAS fixture copied into tmpdir (provide later)
    las = tmp_path / "one.las"
    las.write_bytes(b"LASF…")  # placeholder, real fixture needed

    selected = filter_las_by_bbox([las], bbox=(-180, -90, 180, 90))
    assert las in selected