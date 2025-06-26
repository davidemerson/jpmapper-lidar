#!/usr/bin/env python3
"""
Test the metadata-aware rasterization enhancements.
"""

from pathlib import Path
import sys
import tempfile

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from jpmapper.io.metadata_raster import MetadataAwareRasterizer, create_metadata_report


def test_metadata_detection():
    """Test detection of metadata files."""
    print("=== Testing Metadata Detection ===")
    
    test_dir = Path("tests/data/las")
    rasterizer = MetadataAwareRasterizer()
    
    # Test with actual LAS file
    las_files = list(test_dir.glob("*.las"))
    if las_files:
        las_file = las_files[0]
        print(f"Testing with: {las_file}")
        
        metadata_files = rasterizer.find_metadata_files(las_file)
        print(f"Found metadata files: {list(metadata_files.keys())}")
        
        for ext, path in metadata_files.items():
            print(f"  {ext}: {path}")
            
        # Test CRS detection
        crs = rasterizer.get_crs_from_metadata(las_file)
        print(f"Detected CRS: EPSG:{crs}" if crs else "No CRS detected from metadata")
        
        # Test tile info
        tile_info = rasterizer.get_tile_info(las_file)
        if tile_info:
            print(f"Tile info keys: {list(tile_info.keys())}")
            if 'LAS_ID' in tile_info:
                print(f"  LAS_ID: {tile_info['LAS_ID']}")
            if 'bounds' in tile_info:
                bounds = tile_info['bounds']
                print(f"  Bounds: ({bounds['minx']:.1f}, {bounds['miny']:.1f}) to ({bounds['maxx']:.1f}, {bounds['maxy']:.1f})")
        else:
            print("No tile info found")
            
        # Test accuracy info
        accuracy_info = rasterizer.get_accuracy_info(las_file)
        if accuracy_info:
            print(f"Accuracy info: {list(accuracy_info.keys())}")
            for key, acc in accuracy_info.items():
                if isinstance(acc, dict) and 'description' in acc:
                    print(f"  {key}: {acc['description']}")
        else:
            print("No accuracy info found")
    else:
        print("No LAS files found for testing")


def test_enhanced_rasterization():
    """Test the enhanced rasterization process."""
    print("\n=== Testing Enhanced Rasterization ===")
    
    test_dir = Path("tests/data/las")
    las_files = list(test_dir.glob("*.las"))
    
    if not las_files:
        print("No LAS files found for testing")
        return
        
    las_file = las_files[0]
    rasterizer = MetadataAwareRasterizer()
    
    # Test with a temporary output file
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "test_output.tif"
        
        print(f"Rasterizing {las_file.name} to {output_file.name}")
        
        try:
            result_path, metadata_info = rasterizer.enhanced_rasterize(
                las_file,
                output_file,
                epsg=None,  # Let it auto-detect
                resolution=0.1,
                use_metadata=True
            )
            
            print(f"Output written to: {result_path}")
            print(f"File exists: {result_path.exists()}")
            if result_path.exists():
                print(f"File size: {result_path.stat().st_size} bytes")
            
            print(f"Metadata info:")
            for key, value in metadata_info.items():
                if key == 'tile_info' and isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
                elif key == 'accuracy_info' and isinstance(value, dict):
                    print(f"  {key}: {list(value.keys())}")
                else:
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"Error during rasterization: {e}")
            import traceback
            traceback.print_exc()


def test_metadata_report():
    """Test the metadata report generation."""
    print("\n=== Testing Metadata Report ===")
    
    test_dir = Path("tests/data/las")
    
    try:
        report = create_metadata_report(test_dir)
        
        print(f"Report summary:")
        summary = report['metadata_summary']
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
        if report['crs_distribution']:
            print(f"CRS distribution:")
            for crs, count in report['crs_distribution'].items():
                print(f"  EPSG:{crs}: {count} files")
                
        if report['accuracy_summary']:
            print(f"Accuracy summary:")
            acc = report['accuracy_summary']
            for key, value in acc.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if required dependencies are available
    try:
        import geopandas
        import pyproj
        print("Required dependencies available: geopandas, pyproj")
    except ImportError as e:
        print(f"Missing required dependencies: {e}")
        print("Install with: pip install geopandas pyproj")
        sys.exit(1)
    
    test_metadata_detection()
    test_enhanced_rasterization()
    test_metadata_report()
