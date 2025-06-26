#!/usr/bin/env python3
"""
Examine the metadata files in tests/data/las to understand their structure and content.
"""

import geopandas as gpd
import os
from pathlib import Path

def examine_shapefile():
    """Examine the test shapefile and its attributes."""
    shp_path = Path("tests/data/las/test_sample.shp")
    
    if not shp_path.exists():
        print(f"Shapefile not found at {shp_path}")
        return
    
    print("=== SHAPEFILE EXAMINATION ===")
    print(f"Reading shapefile: {shp_path}")
    
    try:
        # Read the shapefile
        gdf = gpd.read_file(shp_path)
        
        print(f"\nBasic Info:")
        print(f"- Number of features: {len(gdf)}")
        print(f"- CRS: {gdf.crs}")
        print(f"- Geometry type: {gdf.geometry.type.iloc[0] if len(gdf) > 0 else 'None'}")
        
        print(f"\nColumns: {list(gdf.columns)}")
        
        print(f"\nFirst few rows:")
        print(gdf.head())
        
        print(f"\nBounds:")
        print(gdf.bounds)
        
        print(f"\nTotal bounds:")
        print(gdf.total_bounds)
        
        # Show attribute details
        for col in gdf.columns:
            if col != 'geometry':
                print(f"\nColumn '{col}':")
                print(f"  - Type: {gdf[col].dtype}")
                print(f"  - Unique values: {gdf[col].nunique()}")
                if gdf[col].nunique() <= 10:
                    print(f"  - Values: {list(gdf[col].unique())}")
                else:
                    print(f"  - Sample values: {list(gdf[col].unique()[:5])}")
        
    except Exception as e:
        print(f"Error reading shapefile: {e}")

def examine_files():
    """List and examine all metadata files."""
    data_dir = Path("tests/data/las")
    
    print("=== METADATA FILES EXAMINATION ===")
    print(f"Contents of {data_dir}:")
    
    for file_path in sorted(data_dir.iterdir()):
        if file_path.is_file():
            print(f"\n{file_path.name}:")
            print(f"  - Size: {file_path.stat().st_size} bytes")
            
            # Try to read text files
            if file_path.suffix.lower() in ['.txt', '.prj', '.cpg']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if len(content) > 200:
                            print(f"  - Content (first 200 chars): {content[:200]}...")
                        else:
                            print(f"  - Content: {content}")
                except:
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read().strip()
                            if len(content) > 200:
                                print(f"  - Content (first 200 chars): {content[:200]}...")
                            else:
                                print(f"  - Content: {content}")
                    except Exception as e:
                        print(f"  - Could not read as text: {e}")

if __name__ == "__main__":
    examine_files()
    print("\n" + "="*50 + "\n")
    examine_shapefile()
