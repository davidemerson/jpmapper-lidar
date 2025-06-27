#!/usr/bin/env python3
"""Debug script to test OSM base layer functionality"""

import sys
from pathlib import Path

def test_osm_requirements():
    """Test if all required libraries for OSM are available"""
    print("Testing OSM requirements...")
    
    try:
        import contextily as ctx
        print("✅ contextily imported successfully")
        print(f"   Version: {ctx.__version__ if hasattr(ctx, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"❌ contextily import failed: {e}")
        return False
    
    try:
        import geopandas as gpd
        print("✅ geopandas imported successfully")
    except ImportError as e:
        print(f"❌ geopandas import failed: {e}")
        return False
        
    try:
        from shapely.geometry import Point, Polygon, box
        print("✅ shapely imported successfully")
    except ImportError as e:
        print(f"❌ shapely import failed: {e}")
        return False
        
    try:
        from pyproj import Transformer
        print("✅ pyproj imported successfully")
    except ImportError as e:
        print(f"❌ pyproj import failed: {e}")
        return False
        
    try:
        import rasterio
        print("✅ rasterio imported successfully")
    except ImportError as e:
        print(f"❌ rasterio import failed: {e}")
        return False
    
    return True

def test_osm_basemap():
    """Test the OSM basemap functionality directly"""
    print("\nTesting OSM basemap functionality...")
    
    if not test_osm_requirements():
        print("❌ Missing required libraries")
        return
    
    try:
        import matplotlib.pyplot as plt
        import contextily as ctx
        import geopandas as gpd
        from shapely.geometry import box
        
        # Create a simple test map around NYC
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define bounds around NYC
        min_lon, max_lon = -74.05, -73.95
        min_lat, max_lat = 40.70, 40.80
        
        # Create bounding box
        bbox_poly = box(min_lon, min_lat, max_lon, max_lat)
        gdf = gpd.GeoDataFrame([1], geometry=[bbox_poly], crs='EPSG:4326')
        gdf_mercator = gdf.to_crs('EPSG:3857')
        bounds = gdf_mercator.total_bounds
        
        # Set extent
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        
        # Add OSM basemap
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                       zoom='auto', attribution_size=6)
        
        # Add some test points
        from pyproj import Transformer
        tf = Transformer.from_crs(4326, 3857, always_xy=True)
        
        test_points_wgs84 = [(40.7128, -74.0060), (40.7614, -73.9776)]
        test_x, test_y = [], []
        for lat, lon in test_points_wgs84:
            x, y = tf.transform(lon, lat)
            test_x.append(x)
            test_y.append(y)
        
        ax.scatter(test_x, test_y, c='red', s=100, marker='o', zorder=10)
        
        ax.set_title("OSM Basemap Test")
        plt.tight_layout()
        
        output_path = Path("osm_test.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ OSM basemap test successful! Saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ OSM basemap test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_osm_basemap()
