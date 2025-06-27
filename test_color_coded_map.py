#!/usr/bin/env python3
"""Test script for color-coded sample points in analysis map"""

import numpy as np
from pathlib import Path
from jpmapper.analysis.plots import render_analysis_map, create_simple_analysis_map

def test_color_coded_map():
    """Test the color-coded sample points functionality"""
    
    # Mock data for testing
    analyzed_points = [
        (40.7128, -74.0060),  # NYC
        (40.7614, -73.9776),  # Times Square
        (40.7589, -73.9851),  # Central Park
        (40.7505, -73.9934),  # Empire State Building
        (40.7831, -73.9712),  # Upper East Side
        (40.7282, -74.0776),  # Battery Park
    ]
    
    # Create sample points along paths (3 paths with different patterns)
    sample_points = []
    
    # Path 1: NYC to Times Square (simulate some obstructions)
    path1 = []
    for i in range(50):
        lat = 40.7128 + (40.7614 - 40.7128) * i / 49
        lon = -74.0060 + (-73.9776 - (-74.0060)) * i / 49
        path1.append((lat, lon))
    sample_points.append(path1)
    
    # Path 2: Central Park to Empire State (simulate mostly clear)
    path2 = []
    for i in range(50):
        lat = 40.7589 + (40.7505 - 40.7589) * i / 49
        lon = -73.9851 + (-73.9934 - (-73.9851)) * i / 49
        path2.append((lat, lon))
    sample_points.append(path2)
    
    # Path 3: Upper East Side to Battery Park (simulate mixed)
    path3 = []
    for i in range(50):
        lat = 40.7831 + (40.7282 - 40.7831) * i / 49
        lon = -73.9712 + (-74.0776 - (-73.9712)) * i / 49
        path3.append((lat, lon))
    sample_points.append(path3)
    
    # Create mock obstruction data
    sample_obstructions = []
    sample_no_data = []
    
    # Path 1: Some obstructions in the middle
    obs1 = [False] * 50
    no_data1 = [False] * 50
    for i in range(20, 30):  # Obstructions in middle
        obs1[i] = True
    for i in range(5, 10):   # Some no-data points
        no_data1[i] = True
    sample_obstructions.append(obs1)
    sample_no_data.append(no_data1)
    
    # Path 2: Mostly clear with few no-data points
    obs2 = [False] * 50
    no_data2 = [False] * 50
    for i in range(0, 5):    # No-data at start
        no_data2[i] = True
    sample_obstructions.append(obs2)
    sample_no_data.append(no_data2)
    
    # Path 3: Mixed pattern
    obs3 = [False] * 50
    no_data3 = [False] * 50
    for i in range(10, 15):  # Some obstructions
        obs3[i] = True
    for i in range(35, 40):  # More obstructions
        obs3[i] = True
    for i in range(45, 50):  # No-data at end
        no_data3[i] = True
    sample_obstructions.append(obs3)
    sample_no_data.append(no_data3)
    
    # Test the simple map first (since we don't have a real DSM)
    output_path = Path("test_color_coded_map.png")
    
    print("Creating color-coded analysis map...")
    success = create_simple_analysis_map(
        analyzed_points=analyzed_points,
        sample_points=sample_points,
        output_path=output_path,
        title="Test Color-Coded Analysis Map",
        sample_obstructions=sample_obstructions,
        sample_no_data=sample_no_data
    )
    
    if success:
        print(f"✅ Color-coded map saved to: {output_path}")
        print("\nColor coding:")
        print("• Black dots: Clear points with valid data")
        print("• Red dots: Obstructed points")
        print("• White dots: No-data points")
        print("• Colored lines: Analysis paths")
        print("• Large red circles: Analysis endpoints")
    else:
        print("❌ Failed to create color-coded map")

if __name__ == "__main__":
    test_color_coded_map()
