#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

def check_dependencies():
    print("[CHECK] Verifying environment dependencies...")

    # Check external commands
    for cmd in ["pdal", "gdal_merge.py"]:
        if shutil.which(cmd) is None:
            print(f"❌ Missing command: '{cmd}'")
            print(f"   Please install it via your system package manager.")
            print(f"   Example (Ubuntu): sudo apt install pdal gdal-bin")
            sys.exit(1)

    # Check Python modules
    try:
        import tqdm  # noqa
    except ImportError:
        print("❌ Missing Python module: 'tqdm'")
        print("   Install with: pip install tqdm")
        sys.exit(1)

    print("✅ All dependencies are satisfied.\n")

def check_cpu_advice(workers_requested):
    total_cores = os.cpu_count()
    if total_cores is None:
        return

    print(f"[INFO] Detected {total_cores} CPU cores available.")
    if workers_requested < total_cores:
        print(f"⚠️  You are using {workers_requested} workers, but {total_cores} cores are available.")
        print(f"💡 Consider increasing --workers to {total_cores} for full CPU utilization.\n")
    else:
        print("✅ Worker count matches or exceeds available cores.\n")

def build_pipeline(input_path, output_path, resolution=1.0):
    return [
        {
            "type": "readers.las",
            "filename": str(input_path)
        },
        {
            "type": "writers.gdal",
            "filename": str(output_path),
            "output_type": "max",
            "resolution": resolution,
            "data_type": "float",
            "gdaldriver": "GTiff",
            "compression": "lzw"
        }
    ]

def rasterize_file(args_tuple):
    laz_path, output_dir, resolution, force = args_tuple
    laz_path = Path(laz_path)
    output_name = laz_path.stem + "_dsm.tif"
    output_path = Path(output_dir) / output_name

    print(f"[START] Processing {laz_path.name}")
    if output_path.exists() and not force:
        print(f"[SKIP] {output_path.name} exists.")
        return str(output_path)

    pipeline = build_pipeline(laz_path, output_path, resolution)
    pipeline_path = output_path.with_suffix(".json")

    with open(pipeline_path, 'w') as f:
        json.dump(pipeline, f)

    try:
        subprocess.run(
            ["pdal", "pipeline", str(pipeline_path)],
            capture_output=True, text=True, check=True
        )
        print(f"[DONE] {output_path.name}")
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {laz_path.name}:\n{e.stderr}")
        return None
    finally:
        pipeline_path.unlink(missing_ok=True)

def merge_tiles(tile_paths, merged_output_path):
    print(f"\n[MOSAIC] Merging {len(tile_paths)} tiles into {merged_output_path.name}")
    try:
        subprocess.run(
            ["gdal_merge.py", "-o", str(merged_output_path), "-of", "GTiff"] + tile_paths,
            check=True
        )
        print(f"[SUCCESS] Merged DSM written to {merged_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Merging failed:\n{e.stderr}")

def cleanup_files(file_paths, label):
    print(f"[CLEANUP] Deleting {len(file_paths)} {label} files...")
    for path in file_paths:
        try:
            Path(path).unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {path}: {e}")

def main():
    check_dependencies()

    parser = argparse.ArgumentParser(description="Rasterize a folder of LIDAR files into a unified DSM GeoTIFF.")
    parser.add_argument("--lasdir", required=True, help="Input directory of .las/.laz files")
    parser.add_argument("--outdir", required=True, help="Directory to write individual DSMs and merged DSM")
    parser.add_argument("--resolution", type=float, default=1.0, help="Raster resolution in meters")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raster tiles")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel processes")
    parser.add_argument("--merged", default="merged_dsm.tif", help="Filename for merged DSM GeoTIFF")
    parser.add_argument("--cleanup-tiles", action="store_true", help="Delete DSM tile .tif files after merge")
    parser.add_argument("--cleanup-lidar", action="store_true", help="Delete input .las/.laz files after rasterizing")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge DSM tiles into a unified raster")

    args = parser.parse_args()

    check_cpu_advice(args.workers)

    input_dir = Path(args.lasdir)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_output_path = output_dir / args.merged

    laz_files = sorted([f for f in input_dir.glob("*.laz")] + [f for f in input_dir.glob("*.las")])
    print(f"🔍 Found {len(laz_files)} LIDAR files. Rasterizing with {args.workers} workers...")

    task_args = [(str(f), output_dir, args.resolution, args.force) for f in laz_files]

    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(rasterize_file, task_args), total=len(task_args)))

    valid_tiles = [r for r in results if r is not None and Path(r).exists()]
    print(f"\n✅ Rasterization complete. {len(valid_tiles)} tiles ready.")

    if valid_tiles and not args.no_merge:
        merge_tiles(valid_tiles, merged_output_path)
        if args.cleanup_tiles:
            cleanup_files(valid_tiles, "tile DSM")
    elif args.no_merge:
        print("ℹ️ Skipping merge step (user set --no-merge).")
    else:
        print("⚠️ No valid tiles to merge. Skipping DSM merge.")

    if args.cleanup_lidar:
        cleanup_files([str(f) for f in laz_files], "LIDAR")

if __name__ == "__main__":
    main()
