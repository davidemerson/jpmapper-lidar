#!/usr/bin/env python3
"""Debug DSM sampling issues."""

import rasterio
import numpy as np

def debug_dsm_sampling(dsm_path, test_coords):
    """Debug DSM sampling at specific coordinates."""
    print(f"🔍 Testing DSM sampling at coordinates: {test_coords}")
    print(f"📁 DSM file: {dsm_path}")
    
    try:
        with rasterio.open(dsm_path) as ds:
            print(f"📊 DSM info:")
            print(f"   Shape: {ds.shape}")
            print(f"   Bounds: {ds.bounds}")
            print(f"   CRS: {ds.crs}")
            print(f"   NoData: {ds.nodata}")
            
            # Sample a small area to check nodata distribution
            sample_size = min(1000, ds.width, ds.height)
            sample_data = ds.read(1, window=((0, sample_size), (0, sample_size)))
            nodata_count = np.sum(sample_data == ds.nodata) if ds.nodata is not None else 0
            nodata_pct = 100 * nodata_count / sample_data.size
            print(f"   NoData %: {nodata_pct:.1f}% (in {sample_size}x{sample_size} sample)")
            print(f"   Data range: {np.min(sample_data):.1f} to {np.max(sample_data):.1f}")
            
            print(f"\n🎯 Testing coordinates:")
            for i, (x, y) in enumerate(test_coords):
                print(f"  {i+1}. ({x:.1f}, {y:.1f})")
                
                # Check bounds
                in_bounds = (ds.bounds.left <= x <= ds.bounds.right and 
                           ds.bounds.bottom <= y <= ds.bounds.top)
                print(f"     In bounds: {in_bounds}")
                
                if in_bounds:
                    # Sample elevation
                    try:
                        sampled_values = list(ds.sample([(x, y)], 1))
                        elevation = sampled_values[0][0]
                        is_nodata = (ds.nodata is not None and elevation == ds.nodata)
                        print(f"     Elevation: {elevation:.1f}m {'(NODATA)' if is_nodata else '(valid)'}")
                        
                        # Try pixel coordinates
                        row, col = ds.index(x, y)
                        print(f"     Pixel: row={row}, col={col}")
                        
                        # Check surrounding pixels
                        if is_nodata:
                            print(f"     Checking 3x3 neighborhood...")
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    try:
                                        r, c = row + dr, col + dc
                                        if 0 <= r < ds.height and 0 <= c < ds.width:
                                            pixel_val = ds.read(1, window=((r, r+1), (c, c+1)))[0, 0]
                                            is_pixel_nodata = (ds.nodata is not None and pixel_val == ds.nodata)
                                            status = "NODATA" if is_pixel_nodata else f"{pixel_val:.1f}m"
                                            print(f"       [{dr:+2},{dc:+2}]: {status}")
                                    except Exception:
                                        print(f"       [{dr:+2},{dc:+2}]: ERROR")
                        
                    except Exception as e:
                        print(f"     Error sampling: {e}")
                else:
                    print(f"     Outside bounds!")
                    print(f"     DSM bounds: left={ds.bounds.left:.1f}, right={ds.bounds.right:.1f}")
                    print(f"                 bottom={ds.bounds.bottom:.1f}, top={ds.bounds.top:.1f}")
    
    except Exception as e:
        print(f"❌ Error opening DSM: {e}")

if __name__ == "__main__":
    # Test coordinates from the analysis log that showed "no data" warnings
    test_coords = [
        (987256.7, 199308.1),
        (998457.1, 206044.3), 
        (989554.8, 197569.5),
        (985456.2, 198851.1),
        (998467.3, 206031.7),
    ]
    
    debug_dsm_sampling("D:/dsm_cache.tif", test_coords)
