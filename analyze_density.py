#!/usr/bin/env python3
"""Analyze LiDAR density and suggest optimal rasterization resolution."""

import rasterio
import numpy as np
from collections import Counter
import laspy

def analyze_las_density(las_file, sample_area_size=100):
    """Analyze point density in a LAS file."""
    print(f"Analyzing LiDAR density in: {las_file}")
    
    with laspy.open(las_file) as las:
        # Read points
        las_data = las.read()
        x = las_data.x[::100]  # Sample every 100th point for speed
        y = las_data.y[::100]
        z = las_data.z[::100]
        
        print(f"Total points (sampled): {len(x):,}")
        
        # Calculate bounds
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        area_width = max_x - min_x
        area_height = max_y - min_y
        total_area = area_width * area_height
        
        print(f"Bounds: {min_x:.1f}, {min_y:.1f} to {max_x:.1f}, {max_y:.1f}")
        print(f"Area: {area_width:.1f}m x {area_height:.1f}m = {total_area:.1f} m²")
        
        # Calculate density
        points_per_m2 = len(x) / total_area * 10000  # Scale back up from sampling
        print(f"Estimated density: {points_per_m2:.2f} points/m²")
        
        # Suggest resolutions
        if points_per_m2 > 100:
            print("✅ High density data - 0.25-0.5m resolution recommended")
        elif points_per_m2 > 25:
            print("✅ Good density data - 0.5-1.0m resolution recommended")
        elif points_per_m2 > 4:
            print("⚠️  Medium density data - 1.0-2.0m resolution recommended")
        elif points_per_m2 > 1:
            print("⚠️  Low density data - 2.0-5.0m resolution recommended")
        else:
            print("❌ Very low density data - 5.0m+ resolution recommended")
        
        return points_per_m2

def analyze_dsm_coverage(dsm_file, sample_size=10000):
    """Analyze nodata coverage in a DSM."""
    print(f"\nAnalyzing DSM coverage in: {dsm_file}")
    
    with rasterio.open(dsm_file) as ds:
        print(f"DSM shape: {ds.shape}")
        print(f"Resolution: {abs(ds.transform[0]):.2f}m")
        print(f"NoData value: {ds.nodata}")
        
        # Sample data to check coverage
        height, width = ds.shape
        sample_rows = np.random.randint(0, height, sample_size)
        sample_cols = np.random.randint(0, width, sample_size)
        
        values = []
        for row, col in zip(sample_rows, sample_cols):
            try:
                val = ds.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                values.append(val)
            except:
                values.append(ds.nodata)
        
        values = np.array(values)
        
        if ds.nodata is not None:
            nodata_count = np.sum(values == ds.nodata)
            nodata_pct = 100 * nodata_count / len(values)
            print(f"NoData coverage: {nodata_pct:.1f}%")
            
            if nodata_pct > 50:
                print("❌ Very sparse coverage - consider lower resolution")
            elif nodata_pct > 25:
                print("⚠️  Moderate coverage gaps - interpolation will be needed")
            elif nodata_pct > 10:
                print("⚠️  Some coverage gaps - minor interpolation needed")
            else:
                print("✅ Good coverage")
        
        # Show elevation statistics for valid data
        valid_data = values[values != ds.nodata] if ds.nodata is not None else values
        if len(valid_data) > 0:
            print(f"Elevation range: {valid_data.min():.1f}m to {valid_data.max():.1f}m")
            print(f"Mean elevation: {valid_data.mean():.1f}m")

def main():
    print("🔍 LiDAR Dataset Analysis\n")
    
    # Test with a sample LAS file
    import glob
    las_files = glob.glob("D:/meshscope/*.las")
    
    if las_files:
        # Analyze first LAS file
        sample_las = las_files[0]
        density = analyze_las_density(sample_las)
        
    # Analyze the DSM
    dsm_file = "D:/dsm_cache.tif"
    analyze_dsm_coverage(dsm_file)
    
    print("\n💡 Recommendations:")
    print("1. If NoData coverage > 25%, consider:")
    print("   - Using 2-3m resolution instead of 1m")
    print("   - Using different aggregation (mean vs max)")
    print("   - Filling gaps with interpolation")
    print("2. For RF analysis, 2-3m resolution is often sufficient")
    print("3. Higher resolutions may create more nodata gaps")

if __name__ == "__main__":
    main()
