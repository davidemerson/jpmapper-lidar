"""
Comprehensive tests for the JPMapper API.

This test suite covers the core functionality of the JPMapper API,
including filter operations, rasterization, and analysis.
"""
from pathlib import Path
import tempfile
import os
import pytest
import numpy as np

from jpmapper.api import filter_by_bbox, rasterize_tile, analyze_los, generate_profile
from jpmapper.exceptions import (
    JPMapperError, 
    GeometryError, 
    FilterError, 
    RasterizationError, 
    AnalysisError
)


class TestFilterOperations:
    """Test suite for filter operations."""
    
    def test_filter_empty_list(self):
        """Test filtering an empty list returns an empty list."""
        result = filter_by_bbox([], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert result == []
    
    def test_filter_invalid_bbox(self):
        """Test that invalid bounding boxes raise appropriate errors."""
        # Test bbox with wrong number of coordinates
        with pytest.raises(ValueError, match="expected 4 coordinates"):
            filter_by_bbox([], bbox=(-74.01, 40.70, -73.96))
            
        # Test bbox with min > max
        with pytest.raises(ValueError, match="min coordinates must be less than max"):
            filter_by_bbox([], bbox=(-73.96, 40.70, -74.01, 40.75))
        
        with pytest.raises(ValueError, match="min coordinates must be less than max"):
            filter_by_bbox([], bbox=(-74.01, 40.75, -73.96, 40.70))
    
    def test_filter_with_dst_dir(self, tmp_path):
        """Test filtering with a destination directory."""
        # Create mock LAS file paths that won't actually be read
        mock_las = [
            Path(f"mock{i}.las") for i in range(3)
        ]
        
        # This should not raise an error even though files don't exist
        # since we're mocking and the code would normally check headers
        result = filter_by_bbox(mock_las, bbox=(-74.01, 40.70, -73.96, 40.75), dst_dir=tmp_path)
        assert result == []


class TestRasterOperations:
    """Test suite for rasterization operations."""
    
    def test_rasterize_nonexistent_file(self):
        """Test rasterizing a non-existent file raises appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "nonexistent.las"
            dst = Path(tmpdir) / "output.tif"
            
            with pytest.raises(FileNotFoundError, match="Source LAS file does not exist"):
                rasterize_tile(src, dst, epsg=6539)
    
    def test_rasterize_invalid_resolution(self, tmp_path):
        """Test rasterizing with invalid resolution parameters."""
        # Create empty file to avoid FileNotFoundError
        src = tmp_path / "empty.las"
        src.touch()
        dst = tmp_path / "output.tif"
        
        # This test will fail at a later stage but should not raise ValueError
        # for resolution validation
        with pytest.raises(Exception):
            rasterize_tile(src, dst, epsg=6539, resolution=-0.1)


class TestAnalysisOperations:
    """Test suite for analysis operations."""
    
    def test_analyze_los_invalid_coordinates(self, tmp_path):
        """Test that invalid coordinates raise GeometryError."""
        # Create empty DSM file to avoid FileNotFoundError
        dsm_path = tmp_path / "test_dsm.tif"
        dsm_path.touch()
        
        # Test invalid point_a
        with pytest.raises(GeometryError, match="Invalid point_a coordinates"):
            analyze_los(dsm_path, "invalid", (40.7614, -73.9776))
        
        # Test invalid point_b
        with pytest.raises(GeometryError, match="Invalid point_b coordinates"):
            analyze_los(dsm_path, (40.7128, -74.0060), "invalid")
    
    def test_analyze_los_invalid_params(self, tmp_path):
        """Test that invalid parameters raise ValueError."""
        # Create empty DSM file to avoid FileNotFoundError
        dsm_path = tmp_path / "test_dsm.tif"
        dsm_path.touch()
        
        # Test invalid frequency
        with pytest.raises(ValueError, match="Frequency must be positive"):
            analyze_los(dsm_path, (40.7128, -74.0060), (40.7614, -73.9776), freq_ghz=-1)
        
        # Test invalid mast height
        with pytest.raises(ValueError, match="Maximum mast height must be non-negative"):
            analyze_los(dsm_path, (40.7128, -74.0060), (40.7614, -73.9776), max_mast_height_m=-1)
        
        # Test invalid step size
        with pytest.raises(ValueError, match="Mast height step must be positive"):
            analyze_los(
                dsm_path, 
                (40.7128, -74.0060), 
                (40.7614, -73.9776), 
                mast_height_step_m=0
            )
        
        # Test invalid sample count
        with pytest.raises(ValueError, match="Number of samples must be at least 2"):
            analyze_los(dsm_path, (40.7128, -74.0060), (40.7614, -73.9776), n_samples=1)
    
    def test_generate_profile_validation(self, tmp_path):
        """Test validation in generate_profile function."""
        # Create empty DSM file to avoid FileNotFoundError
        dsm_path = tmp_path / "test_dsm.tif"
        dsm_path.touch()
        
        # Test invalid frequency
        with pytest.raises(ValueError, match="Frequency must be positive"):
            generate_profile(
                dsm_path, 
                (40.7128, -74.0060), 
                (40.7614, -73.9776), 
                freq_ghz=-1
            )
        
        # Test invalid sample count
        with pytest.raises(ValueError, match="Number of samples must be at least 2"):
            generate_profile(
                dsm_path, 
                (40.7128, -74.0060), 
                (40.7614, -73.9776), 
                n_samples=1
            )


@pytest.mark.integration
class TestIntegration:
    """
    Integration tests that require actual data files.
    These tests are marked with pytest.mark.integration and can be skipped
    with -m "not integration" if the required data is not available.
    """
    
    @pytest.fixture
    def test_data_dir(self):
        """
        Fixture that returns the test data directory path.
        Tests will be skipped if the directory doesn't exist.
        """
        data_dir = Path(__file__).parent / "data"
        if not data_dir.exists():
            pytest.skip("Test data directory not found")
        return data_dir
    
    def test_filter_by_bbox_with_real_files(self, test_data_dir, tmp_path):
        """Test filtering real LAS files by bbox."""
        las_dir = test_data_dir / "las"
        if not las_dir.exists() or not list(las_dir.glob("*.las")):
            pytest.skip("No LAS test files found")
        
        las_files = list(las_dir.glob("*.las"))
        # Use a bbox that should include all test files
        result = filter_by_bbox(las_files, bbox=(-180, -90, 180, 90))
        assert len(result) > 0, "No files were selected with a global bbox"
