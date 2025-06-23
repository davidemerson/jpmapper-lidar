"""Integration tests for JPMapper workflows."""
import os
import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from jpmapper.api import filter_by_bbox, rasterize_tile, analyze_los, generate_profile
from jpmapper.exceptions import (
    JPMapperError, 
    GeometryError, 
    FilterError, 
    RasterizationError, 
    AnalysisError
)


@pytest.mark.integration
class TestWorkflows:
    """Test end-to-end workflows using mocks."""
    
    @pytest.fixture
    def mock_raster_data(self):
        """Create a mock DSM raster with a simple elevation profile."""
        # Create a 10x10 array with elevation increasing from bottom-left to top-right
        # This will simulate a hill or slope
        data = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                data[i, j] = i + j
        return data
    @patch('jpmapper.api.filter._filter_las_by_bbox')
    def test_filter_workflow(self, mock_filter, tmp_path):
        """Test the filter workflow from API to IO."""
        # Setup mock files
        mock_las_files = [Path(f"file{i}.las") for i in range(3)]
        mock_result = [mock_las_files[0], mock_las_files[2]]
        
        # Setup mock to return two of the three files
        mock_filter.return_value = mock_result
        
        # Call the API function
        result = filter_by_bbox(
            mock_las_files, 
            bbox=(-74.01, 40.70, -73.96, 40.75),
            dst_dir=tmp_path
        )
        
        # Verify that the IO function was called with the correct arguments
        mock_filter.assert_called_once_with(
            mock_las_files, 
            bbox=(-74.01, 40.70, -73.96, 40.75),
            dst_dir=tmp_path
        )
        
        # Verify that the result matches the mock result
        assert result == mock_result
    @patch('jpmapper.api.raster._rasterize_tile')
    def test_rasterize_workflow(self, mock_rasterize, tmp_path):
        """Test the rasterize workflow from API to IO."""
        # Setup mock files
        src = tmp_path / "file.las"
        src.touch()  # Create the file to avoid FileNotFoundError
        dst = tmp_path / "file.tif"
        
        # Setup mock to return the destination file
        mock_rasterize.return_value = dst
          # Call the API function
        result = rasterize_tile(src, dst, epsg=6539, resolution=0.1)
        # Verify that the IO function was called with the correct arguments
        mock_rasterize.assert_called_once_with(
            src, dst, 6539, resolution=0.1
        )
        # Verify that the result matches the mock result
        assert result == dst
        
    @patch('rasterio.open', autospec=True)
    @patch('jpmapper.api.analysis._is_clear', autospec=True)
    def test_analyze_los_workflow(self, mock_is_clear, mock_rasterio_open, tmp_path, mock_raster_data):
        """Test the analyze_los workflow from API to analysis."""
        # Setup mock files
        dsm_path = tmp_path / "dsm.tif"
        dsm_path.touch()
        
        # Setup mock for rasterio.open
        mock_dataset = MagicMock()
        mock_dataset.transform = [0.1, 0, 0, 0, -0.1, 0, 0, 0, 1]
        mock_dataset.crs.to_epsg.return_value = 6539
        mock_rasterio_open.return_value = mock_dataset
        
        # Setup mock for is_clear with the correct return format (matching actual implementation in los.py)
        # (clear, mast_height, ground_A, ground_B, distance)
        mock_is_clear.return_value = (True, 0, 10, 15, 100)
        
        # Call the API function
        result = analyze_los(
            dsm_path,
            (40.7128, -74.0060),
            (40.7614, -73.9776),
            freq_ghz=5.8
        )
        
        # Verify that is_clear was called with the correct parameters
        mock_is_clear.assert_called_once_with(
            mock_dataset, 
            (40.7128, -74.0060), 
            (40.7614, -73.9776),
            freq_ghz=5.8,
            max_mast_height_m=5,
            step_m=1,
            n_samples=256
        )
          # Verify that the result contains the expected fields
        assert "clear" in result
        assert "mast_height_m" in result
        assert "ground_a_m" in result
        assert "ground_b_m" in result
          # Verify that the result values match the mock values
        assert result["clear"] is True
        assert result["mast_height_m"] == 0
        assert result["ground_a_m"] == 10
        assert result["ground_b_m"] == 15
        
    @patch('rasterio.open', autospec=True)
    @patch('jpmapper.api.analysis._profile', autospec=True)
    def test_generate_profile_workflow(self, mock_profile, mock_rasterio_open, tmp_path, mock_raster_data):
        """Test the generate_profile workflow from API to analysis."""
        # Setup mock files
        dsm_path = tmp_path / "dsm.tif"
        dsm_path.touch()
          # Setup mock for rasterio.open
        mock_dataset = MagicMock()
        mock_dataset.transform = [0.1, 0, 0, 0, -0.1, 0, 0, 0, 1]
        mock_dataset.crs.to_epsg.return_value = 6539
        mock_rasterio_open.return_value = mock_dataset
        
        # Setup mock for _profile
        distances = np.linspace(0, 100, 10)
        elevations = np.linspace(10, 20, 10) + np.sin(distances / 10) * 5
        fresnel = np.ones_like(distances) * 5.0
        mock_profile.return_value = (distances, elevations, fresnel)
        
        # Call the API function
        result = generate_profile(
            dsm_path,
            (40.7128, -74.0060),
            (40.7614, -73.9776),
            freq_ghz=5.8,
            n_samples=10
        )
        
        # Verify that _profile was called with the correct parameters
        mock_profile.assert_called_once_with(
            mock_dataset, 
            (40.7128, -74.0060), 
            (40.7614, -73.9776),
            10,  # n_samples
            5.8  # freq_ghz
        )
        
        # Verify that the return value is correct
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        # The API directly returns what _profile returns
        np.testing.assert_array_equal(result[0], distances)  # distances
        np.testing.assert_array_equal(result[1], elevations)  # elevations
        np.testing.assert_array_equal(result[2], fresnel)    # fresnel radius
