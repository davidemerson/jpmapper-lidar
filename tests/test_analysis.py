"""Tests for the analysis module."""
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from jpmapper.analysis.los import (
    is_clear, 
    compute_profile,
    fresnel_radius, 
    point_to_pixel, 
    distance_between_points
)
from jpmapper.exceptions import (
    AnalysisError,
    LOSError,
    GeometryError
)


class TestFunctions:
    """Test suite for individual analysis functions."""
    
    def test_fresnel_radius(self):
        """Test the fresnel_radius function."""
        # Test with known values
        radius = fresnel_radius(
            distance_m=1000,
            distance_total_m=2000,
            frequency_ghz=5.8
        )
        
        # Expected result is approximately 4.5 meters
        # Formula: 17.32 * sqrt(d1 * d2 / (f * D))
        # d1 = 1000, d2 = 1000, f = 5.8, D = 2000
        expected = 17.32 * np.sqrt(1000 * 1000 / (5.8 * 2000))
        assert radius == pytest.approx(expected, rel=1e-6)
        
        # Test with zero distance
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(0, 1000, 5.8)
        
        # Test with zero total distance
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(500, 0, 5.8)
        
        # Test with zero frequency
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(500, 1000, 0)
        
        # Test with negative frequency
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(500, 1000, -5.8)
    
    def test_point_to_pixel(self):
        """Test the point_to_pixel function."""
        # Create a mock transform
        transform = [0.1, 0, 0, 0, -0.1, 0, 0, 0, 1]
        
        # Test with known values
        # Point at origin should be at pixel (0, 0)
        x, y = point_to_pixel((0, 0), transform)
        assert x == 0
        assert y == 0
        
        # Point at (1, 1) should be at pixel (10, -10)
        x, y = point_to_pixel((1, 1), transform)
        assert x == 10
        assert y == -10
        
        # Test with invalid point
        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            point_to_pixel("invalid", transform)
        
        # Test with invalid transform
        with pytest.raises(GeometryError, match="Invalid transform"):
            point_to_pixel((0, 0), "invalid")
    
    def test_distance_between_points(self):
        """Test the distance_between_points function."""
        # Test with known values
        # Points 1 km apart
        distance = distance_between_points((0, 0), (0, 0.01))
        assert distance == pytest.approx(1113.2, rel=1e-2)
        
        # Test with invalid points
        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            distance_between_points("invalid", (0, 0))
        
        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            distance_between_points((0, 0), "invalid")


@pytest.mark.integration
class TestAnalysisWithMocks:
    """Integration tests for analysis functions using mocks."""
    
    @pytest.fixture
    def mock_dsm(self):
        """Create a mock DSM with a simple elevation profile."""
        # Create a 100x100 array with elevation
        # This simulates a terrain with a hill in the middle
        data = np.zeros((100, 100), dtype=np.float32)
        x, y = np.mgrid[0:100, 0:100]
        # Create a hill centered at (50, 50)
        data = 10 + 30 * np.exp(-0.002 * ((x - 50) ** 2 + (y - 50) ** 2))
        return data
    
    @pytest.fixture
    def mock_rasterio_dataset(self, mock_dsm):
        """Create a mock rasterio dataset."""
        mock_dataset = MagicMock()
        mock_dataset.transform = [0.1, 0, 0, 0, -0.1, 0, 0, 0, 1]
        mock_dataset.crs.to_epsg.return_value = 6539
        mock_dataset.read.return_value = mock_dsm[np.newaxis, :, :]
        mock_dataset.shape = mock_dsm.shape
        mock_dataset.height, mock_dataset.width = mock_dsm.shape
        mock_dataset.bounds = (0, 0, 10, 10)
        return mock_dataset
    
    @patch('rasterio.open')
    def test_compute_profile(self, mock_rasterio_open, mock_rasterio_dataset, mock_dsm):
        """Test the compute_profile function."""
        # Setup mock
        mock_rasterio_open.return_value.__enter__.return_value = mock_rasterio_dataset
        
        # Create a mock DSM file
        dsm_path = Path("mock_dsm.tif")
        
        # Test with points that should cross the hill
        point_a = (0, 0)
        point_b = (10, 10)
        
        # Call compute_profile
        distances, elevations, total_distance = compute_profile(
            dsm_path, point_a, point_b, n_samples=10
        )
        
        # Check that the result looks reasonable
        assert len(distances) == 10
        assert len(elevations) == 10
        assert distances[0] == 0
        assert distances[-1] == pytest.approx(total_distance, rel=1e-6)
        
        # The elevations should start low, rise in the middle, and then fall
        # This matches our hill in the middle of the mock DSM
        assert elevations[0] < elevations[len(elevations) // 2]
        assert elevations[-1] < elevations[len(elevations) // 2]
    
    @patch('rasterio.open')
    def test_is_clear_with_clear_path(self, mock_rasterio_open, mock_rasterio_dataset):
        """Test is_clear with a clear path."""
        # Setup mock - make the DSM all zeros for a clear path
        mock_flat_dataset = mock_rasterio_dataset
        mock_flat_dataset.read.return_value = np.zeros((1, 100, 100), dtype=np.float32)
        mock_rasterio_open.return_value.__enter__.return_value = mock_flat_dataset
        
        # Create a mock DSM file
        dsm_path = Path("mock_dsm.tif")
        
        # Test with points
        point_a = (1, 1)
        point_b = (9, 9)
        
        # Call is_clear
        is_clear_result, mast_height, ground_a, ground_b, distance = is_clear(
            dsm_path, point_a, point_b, freq_ghz=5.8
        )
        
        # Check that the path is clear
        assert is_clear_result is True
        assert mast_height == 0
        assert ground_a == 0
        assert ground_b == 0
        assert distance > 0
    
    @patch('rasterio.open')
    def test_is_clear_with_blocked_path(self, mock_rasterio_open, mock_rasterio_dataset, mock_dsm):
        """Test is_clear with a blocked path."""
        # Setup mock
        mock_rasterio_open.return_value.__enter__.return_value = mock_rasterio_dataset
        
        # Create a mock DSM file
        dsm_path = Path("mock_dsm.tif")
        
        # Test with points that should cross the hill
        point_a = (0, 0)
        point_b = (10, 10)
        
        # Call is_clear
        is_clear_result, mast_height, ground_a, ground_b, distance = is_clear(
            dsm_path, point_a, point_b, freq_ghz=5.8
        )
        
        # Check that the path is blocked
        assert is_clear_result is False
        # Mast height should be positive
        assert mast_height > 0
        assert ground_a == pytest.approx(10, rel=1e-1)  # Our mock DSM starts at 10
        assert ground_b == pytest.approx(10, rel=1e-1)
        assert distance > 0
    
    @patch('rasterio.open')
    def test_is_clear_with_mast(self, mock_rasterio_open, mock_rasterio_dataset, mock_dsm):
        """Test is_clear with different mast heights."""
        # Setup mock
        mock_rasterio_open.return_value.__enter__.return_value = mock_rasterio_dataset
        
        # Create a mock DSM file
        dsm_path = Path("mock_dsm.tif")
        
        # Test with points that cross the hill
        point_a = (0, 0)
        point_b = (10, 10)
        
        # First try with max_mast_height_m=0, which should not allow any mast
        is_clear_result, mast_height, _, _, _ = is_clear(
            dsm_path, point_a, point_b, freq_ghz=5.8, max_mast_height_m=0
        )
        
        # Check that the path is blocked and no mast is suggested
        assert is_clear_result is False
        assert mast_height == 0
        
        # Now try with a high enough mast to clear the hill
        is_clear_result, mast_height, _, _, _ = is_clear(
            dsm_path, point_a, point_b, freq_ghz=5.8, max_mast_height_m=50
        )
        
        # Check that the path is now clear with a mast
        assert is_clear_result is True
        assert mast_height > 0
        assert mast_height <= 50
