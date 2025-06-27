#!/usr/bin/env python3
"""Test script for map rendering functionality."""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_map_rendering():
    """Test the map rendering functionality."""
    print("Testing map rendering functionality...")
    
    try:
        from jpmapper.analysis.plots import render_analysis_map, create_simple_analysis_map
        print("✓ Successfully imported map rendering functions")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test data
    analyzed_points = [
        (40.672699, -73.96444),   # Point A1
        (40.696079, -73.939748),  # Point B1
        (40.70569317, -73.91519213),  # Point A2
        (40.7084572, -73.9218844),    # Point B2
    ]
    
    sample_points = [
        # Sample points along path 1
        [(40.672699, -73.96444), (40.684389, -73.952094), (40.696079, -73.939748)],
        # Sample points along path 2
        [(40.70569317, -73.91519213), (40.707025, -73.918964), (40.7084572, -73.9218844)],
    ]
    
    output_path = Path("test_simple_map.png")
    
    print("Testing simple map rendering...")
    try:
        success = create_simple_analysis_map(
            analyzed_points=analyzed_points,
            sample_points=sample_points,
            output_path=output_path,
            title="Test Analysis Map"
        )
        
        if success and output_path.exists():
            print(f"✓ Simple map created successfully: {output_path}")
            print(f"  File size: {output_path.stat().st_size} bytes")
            return True
        else:
            print("✗ Simple map creation failed")
            return False
            
    except Exception as e:
        print(f"✗ Error creating simple map: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_map_rendering()
    if success:
        print("\n🎉 Map rendering test completed successfully!")
    else:
        print("\n❌ Map rendering test failed!")
    sys.exit(0 if success else 1)
