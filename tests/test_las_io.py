"""Tests for the LAS I/O operations."""
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from jpmapper.io.las import filter_las_by_bbox, _read_header
from jpmapper.exceptions import FilterError, FileFormatError, GeometryError


class TestLasIO:
    """Test suite for LAS I/O operations."""

    def test_filter_las_by_bbox_empty_list(self):
        """Test that filtering an empty list returns an empty list."""
        result = filter_las_by_bbox([], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert result == []

    def test_filter_las_by_bbox_invalid_bbox(self):
        """Test that invalid bounding boxes raise appropriate errors."""
        with pytest.raises(GeometryError, match="expected 4 coordinates"):
            filter_las_by_bbox([], bbox=(-74.01, 40.70, -73.96))

        with pytest.raises(GeometryError, match="min coordinates must be less than max"):
            filter_las_by_bbox([], bbox=(-73.96, 40.70, -74.01, 40.75))

        with pytest.raises(GeometryError, match="min coordinates must be less than max"):
            filter_las_by_bbox([], bbox=(-74.01, 40.75, -73.96, 40.70))

    def test_filter_las_by_bbox_file_not_found(self, tmp_path):
        """Test that non-existent files are skipped."""
        non_existent = tmp_path / "does_not_exist.las"
        result = filter_las_by_bbox([non_existent], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert result == []

    def test_filter_las_by_bbox_invalid_file_format(self, tmp_path):
        """Test that invalid LAS files raise FileFormatError."""
        bad_file = tmp_path / "bad.las"
        bad_file.write_bytes(b"not a LAS file")

        with patch('jpmapper.io.las._read_header', side_effect=FileFormatError("bad format")):
            with pytest.raises(FileFormatError):
                filter_las_by_bbox([bad_file], bbox=(-74.01, 40.70, -73.96, 40.75))

    @patch('jpmapper.io.las._read_header')
    def test_filter_las_by_bbox_inside_bbox(self, mock_read_header, tmp_path):
        """Test that files inside the bbox are selected."""
        mock_header = MagicMock()
        mock_header.mins = [-74.0, 40.71, 0]
        mock_header.maxs = [-73.97, 40.74, 100]
        mock_read_header.return_value = mock_header

        test_file = tmp_path / "tile.las"
        test_file.touch()

        result = filter_las_by_bbox([test_file], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert test_file in result

    @patch('jpmapper.io.las._read_header')
    def test_filter_las_by_bbox_outside_bbox(self, mock_read_header, tmp_path):
        """Test that files outside the bbox are not selected."""
        mock_header = MagicMock()
        mock_header.mins = [-75.0, 39.0, 0]
        mock_header.maxs = [-74.5, 39.5, 100]
        mock_read_header.return_value = mock_header

        test_file = tmp_path / "tile.las"
        test_file.touch()

        result = filter_las_by_bbox([test_file], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert test_file not in result

    @patch('jpmapper.io.las._read_header')
    def test_filter_las_by_bbox_with_dst_dir(self, mock_read_header, tmp_path):
        """Test that filtered files are copied to destination directory."""
        mock_header = MagicMock()
        mock_header.mins = [-74.0, 40.71, 0]
        mock_header.maxs = [-73.97, 40.74, 100]
        mock_read_header.return_value = mock_header

        src_file = tmp_path / "tile.las"
        src_file.write_bytes(b"fake las content")
        dst_dir = tmp_path / "output"

        result = filter_las_by_bbox([src_file], bbox=(-74.01, 40.70, -73.96, 40.75), dst_dir=dst_dir)
        assert len(result) == 1
        assert result[0].parent == dst_dir
        assert result[0].exists()
