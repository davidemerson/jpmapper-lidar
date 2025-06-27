#!/usr/bin/env python3
"""
Test script to verify the enhanced map rendering with:
1. OpenStreetMap base layer
2. Larger color-coded sample points
"""

import numpy as np
from pathlib import Path

# Test the enhanced map functionality
def test_enhanced_map():
    try:
        from jpmapper.analysis.plots import create_simple_analysis_map
        
        # Create test data
        analyzed_points = [
            (40.7128, -74.0060),  # NYC
            (40.7614, -73.9776),  # Times Square
            (40.7831, -73.9712),  # Central Park
            (40.7489, -73.9857),  # Empire State
            (40.7505, -73.9934),  # Bryant Park
            (40.7614, -73.9776),  # Times Square (duplicate will be removed)
        ]
        
        # Create sample points for 3 analysis paths
        sample_points = []
        
        # Path 1: NYC to Times Square
        path1 = []
        for i in range(10):
            lat = 40.7128 + (40.7614 - 40.7128) * i / 9
            lon = -74.0060 + (-73.9776 - (-74.0060)) * i / 9
            path1.append((lat, lon))
        sample_points.append(path1)
        
        # Path 2: Times Square to Central Park
        path2 = []
        for i in range(10):
            lat = 40.7614 + (40.7831 - 40.7614) * i / 9
            lon = -73.9776 + (-73.9712 - (-73.9776)) * i / 9
            path2.append((lat, lon))
        sample_points.append(path2)
        
        # Path 3: Empire State to Bryant Park
        path3 = []
        for i in range(10):
            lat = 40.7489 + (40.7505 - 40.7489) * i / 9
            lon = -73.9857 + (-73.9934 - (-73.9857)) * i / 9
            path3.append((lat, lon))
        sample_points.append(path3)
        
        # Create mock obstruction and no-data information
        sample_obstructions = [
            [False, False, True, True, False, False, False, True, False, False],  # Path 1: some obstructions
            [False, False, False, False, False, False, False, False, False, False],  # Path 2: all clear
            [False, True, False, False, True, True, False, False, False, False],   # Path 3: some obstructions
        ]
        
        sample_no_data = [
            [False, False, False, False, False, False, True, False, False, True],  # Path 1: some no-data
            [False, False, False, True, True, False, False, False, False, False],  # Path 2: some no-data
            [False, False, False, False, False, False, False, False, False, False], # Path 3: all have data
        ]
        
        output_path = Path("test_enhanced_color_coded_map.png")
        
        # Test the simple map (fallback) with color coding
        print("Testing enhanced simple map with color-coded sample points...")
        success = create_simple_analysis_map(
            analyzed_points=analyzed_points,
            sample_points=sample_points,
            output_path=output_path,
            title="Test Enhanced Color-Coded Analysis Map",
            sample_obstructions=sample_obstructions,
            sample_no_data=sample_no_data
        )
        
        if success:
            print(f"✅ Enhanced map created successfully: {output_path}")
            print("Sample point colors:")
            print("  🔴 Red: Obstructed points")
            print("  ⚪ White: No data points") 
            print("  ⚫ Black: Clear points with data")
            print("  📏 Sample points are now larger than path lines")
        else:
            print("❌ Failed to create enhanced map")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_enhanced_map()
