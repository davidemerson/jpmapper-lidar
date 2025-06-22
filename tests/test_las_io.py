"""Tests for the LAS I/O operations."""
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from jpmapper.io.las import filter_las_by_bbox
from jpmapper.exceptions import FilterError, FileFormatError


class TestLasIO:
    """Test suite for LAS I/O operations."""
    
    def test_filter_las_by_bbox_empty_list(self):
        """Test that filtering an empty list returns an empty list."""
        result = filter_las_by_bbox([], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert result == []
    
    def test_filter_las_by_bbox_invalid_bbox(self):
        """Test that invalid bounding boxes raise appropriate errors."""
        # Test bbox with wrong number of coordinates
        with pytest.raises(ValueError, match="expected 4 coordinates"):
            filter_las_by_bbox([], bbox=(-74.01, 40.70, -73.96))
            
        # Test bbox with min > max
        with pytest.raises(ValueError, match="min coordinates must be less than max"):
            filter_las_by_bbox([], bbox=(-73.96, 40.70, -74.01, 40.75))
        
        with pytest.raises(ValueError, match="min coordinates must be less than max"):
            filter_las_by_bbox([], bbox=(-74.01, 40.75, -73.96, 40.70))
    
    @patch('laspy.open')
    def test_filter_las_by_bbox_file_not_found(self, mock_laspy_open):
        """Test that non-existent files are handled appropriately."""
        # Setup mock to raise FileNotFoundError
        mock_laspy_open.side_effect = FileNotFoundError("File not found")
        
        # Create a non-existent file path
        non_existent_file = Path("non_existent.las")
        
        # Test that the FileNotFoundError is propagated
        with pytest.raises(FileNotFoundError):
            filter_las_by_bbox([non_existent_file], bbox=(-74.01, 40.70, -73.96, 40.75))
    
    @patch('laspy.open')
    def test_filter_las_by_bbox_invalid_file_format(self, mock_laspy_open):
        """Test that invalid LAS files raise FileFormatError."""
        # Setup mock to raise a laspy exception
        mock_laspy_open.side_effect = Exception("Invalid LAS file")
        
        # Create a file path
        invalid_file = Path("invalid.las")
        
        # Test that the Exception is wrapped in a FileFormatError
        with pytest.raises(FileFormatError):
            filter_las_by_bbox([invalid_file], bbox=(-74.01, 40.70, -73.96, 40.75))
    
    @patch('laspy.open')
    def test_filter_las_by_bbox_inside_bbox(self, mock_laspy_open):
        """Test that files inside the bbox are selected."""
        # Create a mock LAS file
        mock_file = MagicMock()
        mock_header = MagicMock()
        
        # Set the header mins and maxs to be inside the bbox
        mock_header.mins = [-74.0, 40.71, 0]
        mock_header.maxs = [-73.97, 40.74, 100]
        
        # Set up the context manager to return the mock header
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_file
        mock_file.header = mock_header
        mock_laspy_open.return_value = mock_context
        
        # Create a file path
        test_file = Path("test.las")
        
        # Test that the file is selected
        result = filter_las_by_bbox([test_file], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert test_file in result
    
    @patch('laspy.open')
    def test_filter_las_by_bbox_outside_bbox(self, mock_laspy_open):
        """Test that files outside the bbox are not selected."""
        # Create a mock LAS file
        mock_file = MagicMock()
        mock_header = MagicMock()
        
        # Set the header mins and maxs to be outside the bbox
        mock_header.mins = [-75.0, 39.0, 0]
        mock_header.maxs = [-74.5, 39.5, 100]
        
        # Set up the context manager to return the mock header
        mock_context = MagicMock()
        mock_context.__enter__.return_value = mock_file
        mock_file.header = mock_header
        mock_laspy_open.return_value = mock_context
        
        # Create a file path
        test_file = Path("test.las")
        
        # Test that the file is not selected
        result = filter_las_by_bbox([test_file], bbox=(-74.01, 40.70, -73.96, 40.75))
        assert test_file not in result
