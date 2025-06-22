"""Tests for the CLI modules."""
import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from jpmapper.cli.main import app
from jpmapper.cli.filter import filter_bbox
from jpmapper.cli.rasterize import rasterize_tile
from jpmapper.cli.analyze import analyze_csv


runner = CliRunner()


class TestCLI:
    """Test suite for the CLI modules."""
    
    @patch('jpmapper.cli.filter.filter_las_by_bbox')
    def test_filter_bbox_command(self, mock_filter):
        """Test the filter bbox command."""
        # Setup mock to return an empty list
        mock_filter.return_value = []
        
        # Call the CLI command
        result = runner.invoke(
            filter_bbox, 
            [
                "test.las", 
                "--bbox", "-74.01", "40.70", "-73.96", "40.75",
                "--dst", "output"
            ]
        )
        
        # Verify that the command ran successfully
        assert result.exit_code == 0
        
        # Verify that filter_las_by_bbox was called with the correct arguments
        mock_filter.assert_called_once()
        args, kwargs = mock_filter.call_args
        assert len(args[0]) == 1  # One file
        assert kwargs["bbox"] == (-74.01, 40.70, -73.96, 40.75)
        assert kwargs["dst_dir"] == Path("output")
    
    @patch('jpmapper.cli.rasterize.rasterize_tile')
    def test_rasterize_tile_command(self, mock_rasterize):
        """Test the rasterize tile command."""
        # Setup mock to return a path
        mock_rasterize.return_value = Path("output.tif")
        
        # Call the CLI command
        result = runner.invoke(
            rasterize_tile, 
            [
                "input.las", 
                "output.tif", 
                "--epsg", "6539",
                "--resolution", "0.1"
            ]
        )
        
        # Verify that the command ran successfully
        assert result.exit_code == 0
        
        # Verify that rasterize_tile was called with the correct arguments
        mock_rasterize.assert_called_once_with(
            Path("input.las"), 
            Path("output.tif"), 
            epsg=6539, 
            resolution=0.1,
            workers=None
        )
    
    @patch('jpmapper.cli.analyze.analyze_csv_file')
    def test_analyze_csv_command(self, mock_analyze):
        """Test the analyze csv command."""
        # Setup mock to return a list of results
        mock_analyze.return_value = [
            {
                "point_a": (40.7128, -74.0060),
                "point_b": (40.7614, -73.9776),
                "clear": True,
                "mast_height_m": 0,
                "distance_m": 5000
            }
        ]
        
        # Call the CLI command
        with patch('builtins.open', MagicMock()):
            result = runner.invoke(
                analyze_csv, 
                [
                    "points.csv", 
                    "--las-dir", "las",
                    "--cache", "cache.tif",
                    "--json-out", "results.json"
                ]
            )
        
        # Verify that the command ran successfully
        assert result.exit_code == 0
        
        # Verify that analyze_csv_file was called with the correct arguments
        mock_analyze.assert_called_once_with(
            Path("points.csv"), 
            las_dir=Path("las"),
            cache=Path("cache.tif"),
            epsg=None,
            resolution=None,
            workers=None,
            max_mast_height_m=5
        )


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for the CLI commands."""
    
    @patch('jpmapper.api.filter_by_bbox')
    def test_filter_command_integration(self, mock_filter_api):
        """Test the filter command integration with the API."""
        # Setup mock to return an empty list
        mock_filter_api.return_value = []
        
        # Call the CLI command
        result = runner.invoke(
            app, 
            [
                "filter", "bbox",
                "test.las", 
                "--bbox", "-74.01", "40.70", "-73.96", "40.75",
                "--dst", "output"
            ]
        )
        
        # Verify that the command ran successfully
        assert result.exit_code == 0
        
        # Verify that the API function was called
        mock_filter_api.assert_called_once()
    
    @patch('jpmapper.api.rasterize_tile')
    def test_rasterize_command_integration(self, mock_rasterize_api):
        """Test the rasterize command integration with the API."""
        # Setup mock to return a path
        mock_rasterize_api.return_value = Path("output.tif")
        
        # Call the CLI command
        result = runner.invoke(
            app, 
            [
                "rasterize", "tile",
                "input.las", 
                "output.tif", 
                "--epsg", "6539",
                "--resolution", "0.1"
            ]
        )
        
        # Verify that the command ran successfully
        assert result.exit_code == 0
        
        # Verify that the API function was called
        mock_rasterize_api.assert_called_once()
    
    @patch('jpmapper.cli.analyze.analyze_csv_file')
    def test_analyze_command_integration(self, mock_analyze):
        """Test the analyze command integration with the API."""
        # Setup mock to return a list of results
        mock_analyze.return_value = [
            {
                "point_a": (40.7128, -74.0060),
                "point_b": (40.7614, -73.9776),
                "clear": True,
                "mast_height_m": 0,
                "distance_m": 5000
            }
        ]
        
        # Call the CLI command
        with patch('builtins.open', MagicMock()):
            result = runner.invoke(
                app, 
                [
                    "analyze", "csv",
                    "points.csv", 
                    "--las-dir", "las",
                    "--cache", "cache.tif",
                    "--json-out", "results.json"
                ]
            )
        
        # Verify that the command ran successfully
        assert result.exit_code == 0
        
        # Verify that the function was called
        mock_analyze.assert_called_once()
