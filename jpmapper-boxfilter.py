import os
import csv
import laspy
from shapely.geometry import box
from datetime import datetime
from tqdm import tqdm

# NYC bounding box (min_lon, min_lat, max_lon, max_lat)
from pyproj import Transformer
# Reproject WGS84 bounding box to EPSG:6539 (NY Long Island ftUS)
transformer = Transformer.from_crs("epsg:4326", "epsg:6539", always_xy=True)
sw_x, sw_y = transformer.transform(-74.945492, 40.096269)
ne_x, ne_y = transformer.transform(-73.016222, 40.972617)
NYC_BBOX = box(sw_x, sw_y, ne_x, ne_y)
LOG_FILE = "las_filter_log.csv"

def get_las_bounds(filepath):
    try:
        with laspy.open(filepath) as las:
            header = las.header
            min_x, min_y = header.mins[0], header.mins[1]
            max_x, max_y = header.maxs[0], header.maxs[1]
            las_bbox = box(min_x, min_y, max_x, max_y)
            return las_bbox, (min_x, min_y, max_x, max_y)
    except Exception as e:
        return None, (None, None, None, None)

def log_action(logfile, filename, action, bbox_tuple):
    timestamp = datetime.now().isoformat()
    row = [timestamp, filename, action] + list(bbox_tuple)
    with open(logfile, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def filter_las_files(folder):
    # Ensure log file has headers if it's new
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "filename", "action",
                "min_lon", "min_lat", "max_lon", "max_lat"
            ])

    las_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".las") or f.lower().endswith(".laz")
    ]

    for filename in tqdm(las_files, desc="Processing LAS files"):
        full_path = os.path.join(folder, filename)
        try:
            las_bbox, bbox_vals = get_las_bounds(full_path)
            if las_bbox:
                print(f"[{'KEEP' if intersects else 'DEL '}] {filename} → "
                      f"Bounds: ({bbox_vals[1]:.5f}, {bbox_vals[0]:.5f}) to ({bbox_vals[3]:.5f}, {bbox_vals[2]:.5f})")
                if intersects:
                    log_action(LOG_FILE, filename, "kept", bbox_vals)
                else:
                    os.remove(full_path)
                    log_action(LOG_FILE, filename, "deleted", bbox_vals)
            else:
                raise ValueError("Invalid LAS bounding box")
        except Exception as e:
            print(f"[ERROR] {filename} — {e}")
            log_action(LOG_FILE, filename, f"error: {e}", (None, None, None, None))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter LAS files by NYC bounding box and log results")
    parser.add_argument("folder", help="Path to folder containing LAS/LAZ files")
    args = parser.parse_args()
    filter_las_files(args.folder)
