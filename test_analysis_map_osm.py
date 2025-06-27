#!/usr/bin/env python3
"""Test the actual analysis map function to debug OSM layer issue"""

import logging
from pathlib import Path
from jpmapper.analysis.plots import render_analysis_map

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_analysis_map_osm():
    """Test the actual analysis map function"""
    print("Testing actual analysis map function...")
    
    # Use the existing DSM
    dsm_path = Path("output.tif")
    
    # Test data (similar to what we used before)
    analyzed_points = [
        (40.7128, -74.0060),  # NYC coordinates
        (40.7614, -73.9776)   # Another NYC point
    ]
    
    # Sample points (paths between the analyzed points)
    sample_points = [
        [
            (40.7128, -74.0060),
            (40.7200, -74.0000),
            (40.7300, -73.9900),
            (40.7400, -73.9850),
            (40.7500, -73.9800),
            (40.7614, -73.9776)
        ]
    ]
    
    # Mock obstruction data (some clear, some obstructed)
    sample_obstructions = [
        [False, True, False, True, False, False]  # Mixed obstructions
    ]
    
    # Mock no-data
    sample_no_data = [
        [False, False, False, False, False, False]  # No missing data
    ]
    
    output_path = Path("test_analysis_map_osm.png")
    
    print(f"Calling render_analysis_map with:")
    print(f"  DSM path: {dsm_path}")
    print(f"  Analyzed points: {analyzed_points}")
    print(f"  Sample points: {len(sample_points)} paths")
    print(f"  Output: {output_path}")
    
    success = render_analysis_map(
        dsm_path=dsm_path,
        analyzed_points=analyzed_points,
        sample_points=sample_points,
        output_path=output_path,
        title="Test Enhanced Analysis Map with OSM",
        buffer_km=2.0,
        sample_obstructions=sample_obstructions,
        sample_no_data=sample_no_data
    )
    
    if success:
        print(f"✅ Analysis map test successful! Saved to: {output_path}")
    else:
        print("❌ Analysis map test failed!")

if __name__ == "__main__":
    test_analysis_map_osm()
