#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import argparse
import csv
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

def check_dependencies():
    print("[CHECK] Verifying environment dependencies...")
    if shutil.which("pdal") is None:
        print("❌ Missing command: 'pdal'")
        print("   Please install it via your system package manager.")
        print("   Example (Ubuntu): sudo apt install pdal")
        sys.exit(1)
    try:
        import tqdm, rasterio  # noqa
        from rasterio.merge import merge  # noqa
    except ImportError:
        print("❌ Missing Python modules: 'tqdm' and/or 'rasterio'")
        print("   Install with: pip install tqdm rasterio")
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

def get_epsg_code(las_path):
    try:
        result = subprocess.run(
            ["pdal", "info", "--metadata", str(las_path)],
            capture_output=True, text=True, check=True
        )
        meta = json.loads(result.stdout)
        srs = meta["metadata"].get("srs", {})
        epsg = srs.get("epsg")
        return epsg if isinstance(epsg, int) else None
    except Exception as e:
        print(f"[WARN] Failed to detect EPSG for {las_path}: {e}")
        return None

def build_pipeline(input_path, output_path, resolution=1.0, epsg=None):
    pipeline = [
        {
            "type": "readers.las",
            "filename": str(input_path)
        }
    ]
    if epsg:
        pipeline.append({
            "type": "filters.assign",
            "assignment": f"spatialreference=EPSG:{epsg}"
        })
    pipeline.append({
        "type": "writers.gdal",
        "filename": str(output_path),
        "output_type": "max",
        "resolution": resolution,
        "data_type": "float",
        "gdaldriver": "GTiff"
    })
    return pipeline

def log_raster_extent(tif_path, csv_path):
    import rasterio
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs.to_string() if src.crs else "None"
        zmin, zmax = src.read(1).min(), src.read(1).max()
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                Path(tif_path).name,
                crs,
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                src.res[0], src.res[1],
                float(zmin), float(zmax)
            ])

def rasterize_file(args_tuple):
    laz_path, output_dir, resolution, force, log_csv_path = args_tuple
    laz_path = Path(laz_path)
    output_name = laz_path.stem + "_dsm.tif"
    output_path = Path(output_dir) / output_name

    print(f"[START] Processing {laz_path.name}")
    if output_path.exists() and not force:
        print(f"[SKIP] {output_path.name} exists.")
        return str(output_path)

    epsg = get_epsg_code(laz_path)
    if not epsg:
        print(f"[WARN] EPSG code missing from {laz_path}")
        epsg = prompt_for_epsg()

    pipeline = build_pipeline(laz_path, output_path, resolution, epsg)
    pipeline_path = output_path.with_suffix(".json")

    with open(pipeline_path, 'w') as f:
        json.dump(pipeline, f)

    try:
        subprocess.run(["pdal", "pipeline", str(pipeline_path)],
                       capture_output=True, text=True, check=True)
        print(f"[DONE] {output_path.name}")
        log_raster_extent(output_path, log_csv_path)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {laz_path.name}:\n{e.stderr}")
        return None
    finally:
        pipeline_path.unlink(missing_ok=True)

def prompt_for_epsg():
    print("Please choose a CRS to assign to files with missing EPSG:")
    print("  1. WGS 84 (EPSG:4326)                   – GPS lat/lon")
    print("  2. UTM Zone 11N (WGS84) (EPSG:32611)     – Western US")
    print("  3. NAD83 / UTM Zone 15N (EPSG:26915)     – Central US")
    print("  4. NAD83 / California Albers (EPSG:3310) – California")
    print("  5. Web Mercator (EPSG:3857)              – Online maps")
    print("  q. Quit")
    choice = input("Enter 1–5 to apply EPSG to missing files, or q to quit: ").strip()
    epsg_map = {
        "1": 4326, "2": 32611, "3": 26915, "4": 3310, "5": 3857
    }
    if choice.lower() == "q" or choice == "":
        print("Aborting. Please re-run with proper CRS metadata or assign manually.")
        sys.exit(1)
    if choice not in epsg_map:
        print("Invalid option. Aborting.")
        sys.exit(1)
    print(f"✅ Using EPSG:{epsg_map[choice]} for missing CRS.")
    return epsg_map[choice]

def merge_tiles_rasterio(tile_paths, merged_output_path):
    print(f"\n[MOSAIC] Merging {len(tile_paths)} tiles using rasterio...")
    import rasterio
    from rasterio.merge import merge
    from rasterio.crs import CRS

    crs_map = {}
    datasets = []
    fallback_crs = None

    for path in tile_paths:
        ds = rasterio.open(path)

        if ds.crs is None:
            print(f"\n❌ Missing CRS in file: {path}")
            if fallback_crs is None:
                fallback_epsg = prompt_for_epsg()
                fallback_crs = CRS.from_epsg(fallback_epsg)
            ds.crs = fallback_crs

        datasets.append(ds)
        crs = ds.crs.to_string()
        crs_map.setdefault(crs, []).append(path)

    if len(crs_map) > 1:
        print("\n❌ CRS mismatch detected among input tiles.")
        print("Each input DSM .tif file must use the same Coordinate Reference System.")
        print("The following CRS groups were found:")
        for crs, files in crs_map.items():
            print(f"  - CRS: {crs}")
            for f in files:
                print(f"    • {f}")
        print("\n💡 To fix: split files by CRS and process each batch separately.")
        sys.exit(1)

    mosaic, transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform
    })

    with rasterio.open(merged_output_path, "w", **meta) as dst:
        dst.write(mosaic)

    print(f"[SUCCESS] Merged DSM written to {merged_output_path}")

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
    parser.add_argument("--outdir", required=True, help="Directory to write DSM tiles and merged output")
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
    csv_log_path = output_dir / "dsm_tile_index.csv"

    las_files = sorted([f for f in input_dir.glob("*.laz")] + [f for f in input_dir.glob("*.las")])
    print(f"🔍 Found {len(las_files)} LIDAR files. Rasterizing with {args.workers} workers...")

    with open(csv_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "crs", "xmin", "ymin", "xmax", "ymax", "res_x", "res_y", "zmin", "zmax"])

    task_args = [(str(f), output_dir, args.resolution, args.force, csv_log_path) for f in las_files]

    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(rasterize_file, task_args), total=len(task_args)))

    valid_tiles = [r for r in results if r is not None and Path(r).exists()]
    print(f"\n✅ Rasterization complete. {len(valid_tiles)} tiles ready.")

    if valid_tiles and not args.no_merge:
        merge_tiles_rasterio(valid_tiles, merged_output_path)
        if args.cleanup_tiles:
            cleanup_files(valid_tiles, "tile DSM")
    elif args.no_merge:
        print("ℹ️ Skipping merge step (user set --no-merge).")
    else:
        print("⚠️ No valid tiles to merge. Skipping DSM merge.")

    if args.cleanup_lidar:
        cleanup_files([str(f) for f in las_files], "LIDAR")

if __name__ == "__main__":
    main()
