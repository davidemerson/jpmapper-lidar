"""Tests for the jpmapper API."""
from pathlib import Path
import pytest

from jpmapper.api import filter_by_bbox, analyze_los


def test_filter_by_bbox():
    """Test the filter_by_bbox function with an empty input."""
    result = filter_by_bbox([], bbox=(-74.01, 40.70, -73.96, 40.75))
    assert result == []


def test_analyze_los(tmp_path: Path):
    """Test the analyze_los function raises appropriate error with invalid path."""
    non_existent_path = tmp_path / "non_existent.tif"
    
    with pytest.raises(FileNotFoundError):
        analyze_los(
            non_existent_path,
            (40.7128, -74.0060),
            (40.7614, -73.9776),
            freq_ghz=5.8
        )
