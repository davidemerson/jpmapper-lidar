"""Tests for the raster I/O operations."""
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from jpmapper.io.raster import rasterize_tile, cached_mosaic
from jpmapper.exceptions import RasterizationError


class TestRasterIO:
    """Test suite for raster I/O operations."""

    def test_rasterize_tile_nonexistent_file(self, tmp_path):
        """Test that rasterizing a non-existent file raises an error."""
        src = tmp_path / "nonexistent.las"
        dst = tmp_path / "output.tif"

        with pytest.raises(FileNotFoundError):
            rasterize_tile(src, dst, epsg=6539, resolution=0.1)

    def test_rasterize_tile_invalid_resolution(self, tmp_path):
        """Test that invalid resolution parameters raise ValueError."""
        src = tmp_path / "empty.las"
        src.touch()
        dst = tmp_path / "output.tif"

        with pytest.raises(ValueError, match="Resolution must be positive"):
            rasterize_tile(src, dst, epsg=6539, resolution=-0.1)

        with pytest.raises(ValueError, match="Resolution must be positive"):
            rasterize_tile(src, dst, epsg=6539, resolution=0)

    @patch('jpmapper.io.raster._run_pdal')
    def test_rasterize_tile_pdal_error(self, mock_run_pdal, tmp_path):
        """Test that PDAL pipeline errors are handled appropriately."""
        src = tmp_path / "tile.las"
        src.touch()
        dst = tmp_path / "output.tif"

        error_msg = "readers.las: Couldn't read LAS header. File size insufficient."
        mock_run_pdal.side_effect = RuntimeError(error_msg)

        with pytest.raises(RasterizationError) as excinfo:
            rasterize_tile(src, dst, epsg=6539, resolution=0.1)

        assert "readers.las: Couldn't read LAS header" in str(excinfo.value)

    @patch('jpmapper.io.raster._run_pdal')
    def test_rasterize_tile_success(self, mock_run_pdal, tmp_path):
        """Test successful rasterization."""
        src = tmp_path / "tile.las"
        src.touch()
        dst = tmp_path / "output.tif"

        mock_run_pdal.return_value = None

        result = rasterize_tile(src, dst, epsg=6539, resolution=0.1)
        assert result == dst

    @patch('jpmapper.io.raster._run_pdal')
    @patch('jpmapper.io.raster._detect_epsg')
    def test_rasterize_tile_auto_detect_epsg(self, mock_detect, mock_run_pdal, tmp_path):
        """Test that EPSG is auto-detected when not provided."""
        src = tmp_path / "tile.las"
        src.touch()
        dst = tmp_path / "output.tif"

        mock_detect.return_value = 6539
        mock_run_pdal.return_value = None

        result = rasterize_tile(src, dst, resolution=0.1)
        assert result == dst
        mock_detect.assert_called_once_with(src)

    def test_cached_mosaic_cache_exists(self, tmp_path):
        """Test that cached_mosaic returns the cache file if it exists."""
        cache = tmp_path / "cache.tif"
        cache.touch()

        result = cached_mosaic(tmp_path, cache, epsg=6539, resolution=0.1)
        assert result == cache

    @patch('jpmapper.io.raster.merge_tiles')
    @patch('jpmapper.io.raster.rasterize_dir_parallel')
    def test_cached_mosaic_create_cache(self, mock_rasterize_dir, mock_merge, tmp_path):
        """Test that cached_mosaic creates the cache file if it doesn't exist."""
        las_dir = tmp_path / "las"
        las_dir.mkdir()
        las_file = las_dir / "tile.las"
        las_file.touch()

        cache = tmp_path / "cache.tif"

        mock_tif = tmp_path / "temp" / "tile.tif"
        mock_rasterize_dir.return_value = [mock_tif]

        result = cached_mosaic(las_dir, cache, epsg=6539, resolution=0.1)

        mock_rasterize_dir.assert_called_once()
        mock_merge.assert_called_once_with([mock_tif], cache)
        assert result == cache
