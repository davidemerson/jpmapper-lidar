"""Test metadata-aware rasterization functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import tempfile

# Test both with and without optional dependencies

def _check_geopandas_available():
    """Check if geopandas is available for import."""
    try:
        import geopandas
        return True
    except ImportError:
        return False

def test_metadata_raster_import_with_dependencies():
    """Test that metadata functions are available when dependencies are installed."""
    try:
        from jpmapper.io.metadata_raster import MetadataAwareRasterizer
        from jpmapper.api.enhanced_raster import rasterize_tile_with_metadata
        assert callable(rasterize_tile_with_metadata)
        rasterizer = MetadataAwareRasterizer()
        assert rasterizer is not None
    except ImportError:
        pytest.skip("geopandas/pyproj not available for testing")


def test_metadata_raster_import_without_dependencies():
    """Test graceful handling when optional dependencies are not available."""
    with patch.dict(sys.modules, {'geopandas': None, 'pyproj': None, 'fiona': None}):
        # Remove from cache if already imported
        modules_to_remove = [
            'jpmapper.io.metadata_raster',
            'jpmapper.api.enhanced_raster'
        ]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
        
        # The modules should still import but with limited functionality
        from jpmapper.io.metadata_raster import MetadataAwareRasterizer
        from jpmapper.api.enhanced_raster import rasterize_tile_with_metadata
        
        # Functions should be callable but may have reduced functionality
        assert callable(rasterize_tile_with_metadata)
        rasterizer = MetadataAwareRasterizer()
        assert rasterizer is not None


@pytest.mark.skipif(
    sys.modules.get('geopandas') is None, 
    reason="geopandas not available"
)
def test_metadata_aware_rasterizer_init():
    """Test MetadataAwareRasterizer initialization."""
    from jpmapper.io.metadata_raster import MetadataAwareRasterizer
    
    # Test default initialization
    rasterizer = MetadataAwareRasterizer()
    assert rasterizer.metadata_dir is None
    assert rasterizer._shapefile_cache == {}
    assert rasterizer._crs_cache == {}
    
    # Test with metadata directory
    test_dir = Path("/test/metadata")
    rasterizer = MetadataAwareRasterizer(metadata_dir=test_dir)
    assert rasterizer.metadata_dir == test_dir


def test_find_metadata_files(tmp_path):
    """Test metadata file discovery."""
    try:
        from jpmapper.io.metadata_raster import MetadataAwareRasterizer
    except ImportError:
        pytest.skip("Dependencies not available")
    
    # Create test LAS file
    las_file = tmp_path / "test.las"
    las_file.touch()
    
    # Create various metadata files
    metadata_files = [
        "test.prj",
        "test.cpg", 
        "test.xml",
        "test.shp",
        "test.dbf",
        "test.shx"
    ]
    
    for meta_file in metadata_files:
        (tmp_path / meta_file).touch()
    
    rasterizer = MetadataAwareRasterizer()
    found_files = rasterizer.find_metadata_files(las_file)
    
    # Should find the relevant metadata files
    assert '.prj' in found_files
    assert '.cpg' in found_files
    assert '.xml' in found_files
    assert '.shp' in found_files


def test_find_metadata_files_no_files(tmp_path):
    """Test metadata file discovery when no metadata files exist."""
    try:
        from jpmapper.io.metadata_raster import MetadataAwareRasterizer
    except ImportError:
        pytest.skip("Dependencies not available")
    
    # Create test LAS file with no metadata
    las_file = tmp_path / "test.las"
    las_file.touch()
    
    rasterizer = MetadataAwareRasterizer()
    found_files = rasterizer.find_metadata_files(las_file)
    
    # Should return empty dict
    assert found_files == {}


@pytest.mark.skipif(
    not _check_geopandas_available(),
    reason="geopandas not available"
)
def test_enhanced_rasterization_fallback(tmp_path):
    """Test that enhanced rasterization falls back to standard when metadata unavailable."""
    from jpmapper.api.enhanced_raster import rasterize_tile_with_metadata
    
    # Create a minimal test LAS file (just touch it for the test)
    las_file = tmp_path / "test.las"
    las_file.touch()
    
    output_file = tmp_path / "output.tif"
    
    # Mock the base rasterization function to avoid actual processing
    with patch('jpmapper.api.enhanced_raster._base_rasterize_tile') as mock_raster:
        mock_raster.return_value = output_file
        
        # This should fall back to standard rasterization since no real LAS data
        result_path, metadata_info = rasterize_tile_with_metadata(
            las_file,
            output_file,
            epsg=6539,
            resolution=0.1,
            use_metadata=True
        )
        
        # Should have metadata information
        assert 'metadata_enhanced' in metadata_info
        # Check if fallback was used OR if metadata enhancement succeeded
        assert metadata_info.get('fallback_used', False) or metadata_info.get('metadata_enhanced', False)
        assert result_path == output_file


def test_rasterize_tile_with_metadata_file_not_found():
    """Test error handling when source LAS file doesn't exist."""
    try:
        from jpmapper.api.enhanced_raster import rasterize_tile_with_metadata
    except ImportError:
        pytest.skip("Dependencies not available")
    
    nonexistent_file = Path("/nonexistent/file.las")
    output_file = Path("/output/file.tif")
    
    with pytest.raises(FileNotFoundError):
        rasterize_tile_with_metadata(
            nonexistent_file,
            output_file,
            epsg=6539,
            resolution=0.1
        )


@pytest.mark.skipif(
    sys.modules.get('geopandas') is None, 
    reason="geopandas not available"
)
def test_batch_rasterize_with_metadata_empty_list():
    """Test batch rasterization with empty file list."""
    from jpmapper.api.enhanced_raster import batch_rasterize_with_metadata
    
    results = batch_rasterize_with_metadata(
        [],  # Empty list
        Path("/output"),
        epsg=6539,
        resolution=0.1
    )
    
    assert results == []


def test_generate_processing_report_empty():
    """Test processing report generation with empty results."""
    try:
        from jpmapper.api.enhanced_raster import generate_processing_report
    except ImportError:
        pytest.skip("Dependencies not available")
    
    report = generate_processing_report([])
    
    # Should have basic structure
    assert 'summary' in report
    assert 'files' in report
    assert report['summary']['total_files'] == 0


def test_create_metadata_report(tmp_path):
    """Test metadata report creation."""
    try:
        from jpmapper.io.metadata_raster import create_metadata_report
    except ImportError:
        pytest.skip("Dependencies not available")
    
    # Create some test LAS files
    las_files = [
        tmp_path / "test1.las",
        tmp_path / "test2.las"
    ]
    
    for las_file in las_files:
        las_file.touch()
    
    # Create the report
    report = create_metadata_report(tmp_path)
    
    # Should have basic structure
    assert 'metadata_summary' in report
    assert 'las_files' in report
    assert report['metadata_summary']['total_files'] == len(las_files)


def test_metadata_enhancements_in_api():
    """Test that metadata enhancement functions are properly exported in the API."""
    try:
        from jpmapper.api import rasterize_tile_with_metadata, batch_rasterize_with_metadata, generate_processing_report
        
        # Functions should be importable
        assert callable(rasterize_tile_with_metadata)
        assert callable(batch_rasterize_with_metadata) 
        assert callable(generate_processing_report)
        
    except ImportError:
        # This is acceptable if dependencies aren't available
        pytest.skip("Enhanced rasterization dependencies not available")


def test_metadata_enhancement_graceful_degradation():
    """Test that the system works even with missing metadata files."""
    try:
        from jpmapper.io.metadata_raster import MetadataAwareRasterizer
    except ImportError:
        pytest.skip("Dependencies not available")
    
    # Test with non-existent LAS file (should not crash)
    rasterizer = MetadataAwareRasterizer()
    
    # These should return None or empty without crashing
    fake_las = Path("/nonexistent/test.las")
    
    metadata_files = rasterizer.find_metadata_files(fake_las)
    assert isinstance(metadata_files, dict)
    
    crs = rasterizer.get_crs_from_metadata(fake_las)
    assert crs is None
    
    tile_info = rasterizer.get_tile_info(fake_las)
    assert tile_info is None
    
    accuracy_info = rasterizer.get_accuracy_info(fake_las)
    assert accuracy_info is None


def test_optional_dependency_flags():
    """Test that optional dependency flags are properly set."""
    from jpmapper.io.metadata_raster import HAS_GEOPANDAS, HAS_LASPY, HAS_PYPROJ
    
    # Flags should be boolean
    assert isinstance(HAS_GEOPANDAS, bool)
    assert isinstance(HAS_LASPY, bool)
    assert isinstance(HAS_PYPROJ, bool)


@pytest.mark.skipif(
    sys.modules.get('laspy') is None,
    reason="laspy not available"
)
def test_las_file_handling():
    """Test that the system can handle actual LAS file operations where possible."""
    # This test would require actual LAS file data, so we'll mock it
    from jpmapper.io.metadata_raster import MetadataAwareRasterizer
    
    rasterizer = MetadataAwareRasterizer()
    
    # Test with test data if available
    test_data_dir = Path("tests/data/las")
    if test_data_dir.exists():
        las_files = list(test_data_dir.glob("*.las"))
        if las_files:
            test_las = las_files[0]
            
            # These should not crash even if no metadata is found
            metadata_files = rasterizer.find_metadata_files(test_las)
            assert isinstance(metadata_files, dict)
            
            crs = rasterizer.get_crs_from_metadata(test_las)
            # CRS may be None if no metadata is available
            
            tile_info = rasterizer.get_tile_info(test_las)
            # Tile info may be None if no shapefile is available
            
            accuracy_info = rasterizer.get_accuracy_info(test_las)
            # Accuracy info may be None if no XML is available
