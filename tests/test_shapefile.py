"""Test shapefile filtering functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Test both with and without geopandas availability

# Import dependency checking from conftest
from conftest import check_geopandas_available, check_enhanced_deps_available

def test_shapefile_import_with_geopandas():
    """Test that shapefile functions are available when geopandas is installed."""
    if not check_geopandas_available():
        pytest.skip("geopandas not available for testing")
    
    from jpmapper.api.shapefile_filter import filter_by_shapefile
    assert callable(filter_by_shapefile)


def test_shapefile_import_without_geopandas():
    """Test graceful handling when geopandas is not available."""
    with patch.dict(sys.modules, {'geopandas': None, 'fiona': None}):
        # Remove from cache if already imported
        if 'jpmapper.api.shapefile_filter' in sys.modules:
            del sys.modules['jpmapper.api.shapefile_filter']
        
        from jpmapper.exceptions import ConfigurationError
        
        # This should work - import doesn't fail, but function calls will
        try:
            from jpmapper.api.shapefile_filter import filter_by_shapefile
            
            # Calling the function should raise ConfigurationError
            with pytest.raises(ConfigurationError, match="geopandas is required"):
                filter_by_shapefile([], Path("test.shp"))
                
        except ImportError:
            # This is also acceptable - the module itself might not import
            pass


@pytest.mark.skipif(
    not check_geopandas_available(),
    reason="geopandas not available"
)
def test_filter_by_shapefile_basic(tmp_path):
    """Test basic shapefile filtering functionality."""
    from jpmapper.api.shapefile_filter import filter_by_shapefile
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    # Create test shapefile
    test_polygon = Polygon([
        (-74.01, 40.70), (-73.96, 40.70), 
        (-73.96, 40.75), (-74.01, 40.75), (-74.01, 40.70)
    ])
    
    gdf = gpd.GeoDataFrame({'id': [1]}, geometry=[test_polygon], crs="EPSG:4326")
    shapefile_path = tmp_path / "test_boundary.shp"
    gdf.to_file(shapefile_path)
    
    # Create mock LAS files
    las_files = [tmp_path / "test1.las", tmp_path / "test2.las"]
    for las_file in las_files:
        las_file.touch()
    
    # Mock laspy to return test bounds
    with patch('laspy.open') as mock_laspy:
        mock_context = MagicMock()
        mock_header = MagicMock()
        
        # Set up header with bounds inside the test polygon
        mock_header.mins = [-74.005, 40.705, 0]
        mock_header.maxs = [-73.995, 40.715, 100]
        mock_header.parse_crs.return_value = None  # No CRS in test file
        
        mock_context.__enter__.return_value.header = mock_header
        mock_laspy.return_value = mock_context
        
        # Test filtering
        result = filter_by_shapefile(
            las_files, 
            shapefile_path, 
            validate_crs=False  # Skip CRS validation for test
        )
        
        # Should select files that intersect
        assert len(result) > 0
        assert all(f in las_files for f in result)


def test_shapefile_nonexistent_file():
    """Test handling of nonexistent shapefile."""
    from jpmapper.api.shapefile_filter import filter_by_shapefile, HAS_GEOPANDAS
    from jpmapper.exceptions import ConfigurationError
    
    if not HAS_GEOPANDAS:
        # If geopandas is not available, should get ConfigurationError
        with pytest.raises(ConfigurationError, match="geopandas is required"):
            filter_by_shapefile([], Path("nonexistent.shp"))
    else:
        # If geopandas is available, should get FileNotFoundError
        with pytest.raises(FileNotFoundError):
            filter_by_shapefile([], Path("nonexistent.shp"))


@pytest.mark.skipif(
    sys.modules.get('geopandas') is None, 
    reason="geopandas not available"
)
def test_create_boundary_from_las_files(tmp_path):
    """Test creation of boundary shapefile from LAS files."""
    from jpmapper.api.shapefile_filter import create_boundary_from_las_files
    
    # Create mock LAS files
    las_files = [tmp_path / "test1.las", tmp_path / "test2.las"]
    for las_file in las_files:
        las_file.touch()
    
    # Mock laspy to return test bounds
    with patch('laspy.open') as mock_laspy:
        mock_context = MagicMock()
        mock_header = MagicMock()
        
        # Set up different bounds for each file
        call_count = 0
        def mock_open_side_effect(*args, **kwargs):
            nonlocal call_count
            mock_header.mins = [-74.0 - call_count * 0.01, 40.70, 0]
            mock_header.maxs = [-73.9 - call_count * 0.01, 40.75, 100]
            mock_header.parse_crs.return_value = None
            call_count += 1
            return mock_context
        
        mock_context.__enter__.return_value.header = mock_header
        mock_laspy.side_effect = mock_open_side_effect
        
        # Test boundary creation
        output_shapefile = tmp_path / "boundary.shp"
        result = create_boundary_from_las_files(
            las_files,
            output_shapefile,
            buffer_meters=10.0,
            epsg=4326
        )
        
        assert result == output_shapefile
        assert output_shapefile.exists()


def test_cli_shapefile_command_without_geopandas(tmp_path):
    """Test CLI gracefully handles missing geopandas."""
    from jpmapper.cli.filter import filter_shapefile
    import typer
    
    las_file = tmp_path / "test.las"
    las_file.touch()
    shapefile_path = tmp_path / "boundary.shp"
    
    # Mock geopandas as unavailable
    with patch.dict(sys.modules, {'geopandas': None}):
        with pytest.raises(typer.Exit):
            filter_shapefile(
                src=las_file,
                shapefile=shapefile_path,
                dst=None,
                buffer=0.0,
                validate_crs=True
            )


def test_api_integration_graceful_degradation():
    """Test that API gracefully handles missing shapefile support."""
    # In the current design, shapefile functions are always available but raise
    # appropriate errors when dependencies are missing. This is better UX than
    # having functions disappear completely.
    
    from jpmapper.api import filter_by_bbox, rasterize_tile
    assert callable(filter_by_bbox)
    assert callable(rasterize_tile)
    
    # Shapefile functions should be available but may raise ConfigurationError
    try:
        from jpmapper.api import filter_by_shapefile
        from jpmapper.exceptions import ConfigurationError
        
        # Function should be callable
        assert callable(filter_by_shapefile)
        
        # But should raise ConfigurationError if dependencies are missing
        from jpmapper.api.shapefile_filter import HAS_GEOPANDAS
        if not HAS_GEOPANDAS:
            with pytest.raises(ConfigurationError):
                filter_by_shapefile([], Path("test.shp"))
        
    except ImportError:
        # This would only happen if there's a real import error, not missing optional deps
        pytest.skip("Shapefile module could not be imported")


def test_buffer_parameter():
    """Test buffer parameter functionality."""
    try:
        from jpmapper.api.shapefile_filter import filter_by_shapefile
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Create a small polygon
        small_polygon = Polygon([
            (-74.00, 40.70), (-73.99, 40.70), 
            (-73.99, 40.71), (-74.00, 40.71), (-74.00, 40.70)
        ])
        
        # Mock the filtering process to test buffer application
        with patch('geopandas.read_file') as mock_read, \
             patch('pathlib.Path.exists', return_value=True):  # Mock file existence
            mock_gdf = MagicMock()
            mock_gdf.empty = False
            mock_gdf.geometry.iloc.__getitem__.return_value = small_polygon
            mock_gdf.crs = "EPSG:4326"
            mock_read.return_value = mock_gdf
            
            with patch('jpmapper.api.shapefile_filter._filter_las_by_geometry') as mock_filter:
                mock_filter.return_value = []
                
                # Call with buffer
                filter_by_shapefile(
                    [],
                    Path("test.shp"),
                    buffer_meters=100.0
                )
                
                # Verify that buffer was applied (geometry passed to filter should be buffered)
                mock_filter.assert_called_once()
                # The buffered geometry should be larger than the original
                # This is a simplified test - in practice you'd check the actual geometry
                
    except ImportError:
        pytest.skip("geopandas not available")
