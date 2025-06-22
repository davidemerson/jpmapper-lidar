"""Tests for the raster I/O operations."""
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from jpmapper.io.raster import rasterize_tile, cached_mosaic
from jpmapper.exceptions import RasterizationError, FileFormatError


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
        # Create empty file to avoid FileNotFoundError
        src = tmp_path / "empty.las"
        src.touch()
        dst = tmp_path / "output.tif"
        
        with pytest.raises(ValueError, match="Resolution must be positive"):
            rasterize_tile(src, dst, epsg=6539, resolution=-0.1)
        
        with pytest.raises(ValueError, match="Resolution must be positive"):
            rasterize_tile(src, dst, epsg=6539, resolution=0)
    
    @patch('jpmapper.io.raster.run_pdal_pipeline')
    def test_rasterize_tile_pdal_error(self, mock_run_pdal, tmp_path):
        """Test that PDAL pipeline errors are handled appropriately."""
        # Create empty file to avoid FileNotFoundError
        src = tmp_path / "empty.las"
        src.touch()
        dst = tmp_path / "output.tif"
        
        # Setup mock to raise an error
        mock_run_pdal.side_effect = Exception("PDAL pipeline error")
        
        with pytest.raises(RasterizationError, match="PDAL pipeline error"):
            rasterize_tile(src, dst, epsg=6539, resolution=0.1)
    
    @patch('jpmapper.io.raster.run_pdal_pipeline')
    def test_rasterize_tile_success(self, mock_run_pdal, tmp_path):
        """Test successful rasterization."""
        # Create empty file to avoid FileNotFoundError
        src = tmp_path / "empty.las"
        src.touch()
        dst = tmp_path / "output.tif"
        
        # Setup mock to return successfully
        mock_run_pdal.return_value = None
        
        # Should not raise any errors
        result = rasterize_tile(src, dst, epsg=6539, resolution=0.1)
        assert result == dst
    
    @patch('jpmapper.io.raster.rasterize_tile')
    def test_cached_mosaic_cache_exists(self, mock_rasterize, tmp_path):
        """Test that cached_mosaic returns the cache file if it exists."""
        # Create a mock cache file
        cache = tmp_path / "cache.tif"
        cache.touch()
        
        # Call cached_mosaic
        result = cached_mosaic(tmp_path, cache, epsg=6539, resolution=0.1)
        
        # Verify that rasterize_tile was not called
        mock_rasterize.assert_not_called()
        
        # Verify that the cache file was returned
        assert result == cache
    
    @patch('jpmapper.io.raster.rasterize_tile')
    def test_cached_mosaic_create_cache(self, mock_rasterize, tmp_path):
        """Test that cached_mosaic creates the cache file if it doesn't exist."""
        # Create a mock las directory
        las_dir = tmp_path / "las"
        las_dir.mkdir()
        
        # Create a mock las file
        las_file = las_dir / "test.las"
        las_file.touch()
        
        # Setup cache path
        cache = tmp_path / "cache.tif"
        
        # Setup mock to return the cache path
        mock_rasterize.return_value = cache
        
        # Call cached_mosaic
        result = cached_mosaic(las_dir, cache, epsg=6539, resolution=0.1)
        
        # Verify that rasterize_tile was called
        mock_rasterize.assert_called_once()
        
        # Verify that the cache file was returned
        assert result == cache
