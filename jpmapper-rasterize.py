#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import argparse
import csv
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio.merge import merge

def check_dependencies():
    print("[CHECK] Verifying environment dependencies...")
    if shutil.which("pdal") is None:
        print("❌ Missing command: 'pdal'")
        sys.exit(1)
    try:
        import tqdm, rasterio
    except ImportError:
        print("❌ Missing Python modules. Try: pip install tqdm rasterio")
        sys.exit(1)
    print("✅ All dependencies are satisfied.\n")

def check_cpu_advice(workers_requested):
    total_cores = os.cpu_count()
    if total_cores and workers_requested < total_cores:
        print(f"⚠️  You are using {workers_requested} workers, but {total_cores} cores are available.")
        print(f"💡 Consider increasing --workers to {total_cores}.\n")

def get_epsg_code(las_path):
    try:
        result = subprocess.run(["pdal", "info", "--metadata", str(las_path)], capture_output=True, text=True, check=True)
        meta = json.loads(result.stdout)
        epsg = meta["metadata"].get("srs", {}).get("epsg")
        return epsg if isinstance(epsg, int) else None
    except Exception:
        return None

def prompt_for_epsg():
    print("Please choose a CRS for files with missing EPSG:")
    print("  1. WGS 84 (EPSG:4326)")
    print("  2. UTM Zone 11N (EPSG:32611)")
    print("  3. NAD83 / UTM Zone 15N (EPSG:26915)")
    print("  4. NAD83 / California Albers (EPSG:3310)")
    print("  5. Web Mercator (EPSG:3857)")
    print("  6. NYC 2021 Data CRS (EPSG:6539 + EPSG:6360)")
    print("  q. Quit")
    choice = input("Enter 1–6 or q: ").strip()
    epsg_map = {
        "1": "EPSG:4326",
        "2": "EPSG:32611",
        "3": "EPSG:26915",
        "4": "EPSG:3310",
        "5": "EPSG:3857",
        "6": ('COMPD_CS["NAD83(2011) / New York Long Island (ftUS) + NAVD88 height (ftUS)",'
              'PROJCS["NAD83(2011) / New York Long Island (ftUS)",'
              'GEOGCS["NAD83(2011)",DATUM["NAD83 (National Spatial Reference System 2011)",'
              'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
              'UNIT["degree",0.0174532925199433]],'
              'PROJECTION["Lambert_Conformal_Conic_2SP"],'
              'PARAMETER["standard_parallel_1",41.03333333333333],'
              'PARAMETER["standard_parallel_2",40.66666666666666],'
              'PARAMETER["latitude_of_origin",40.16666666666666],'
              'PARAMETER["central_meridian",-74],'
              'PARAMETER["false_easting",984250],'
              'PARAMETER["false_northing",0],'
              'UNIT["US survey foot",0.3048006096012192],'
              'AXIS["X",EAST],AXIS["Y",NORTH]],'
              'VERT_CS["NAVD88 height (ftUS)",'
              'VERT_DATUM["North American Vertical Datum 1988",2005],'
              'UNIT["US survey foot",0.3048006096012192],AXIS["Up",UP]]]')
    }
    if choice.lower() == "q":
        sys.exit("Aborted by user.")
    if choice not in epsg_map:
        sys.exit("Invalid option.")
    print(f"✅ Using {epsg_map[choice]}")
    return epsg_map[choice]

def build_pipeline(input_path, output_path, resolution=1.0, epsg=None):
    pipeline = [{"type": "readers.las", "filename": str(input_path)}]
    if epsg:
        pipeline.append({
            "type": "filters.reprojection",
            "in_srs": epsg,
            "out_srs": epsg
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
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        zmin, zmax = src.read(1).min(), src.read(1).max()
        if crs and crs.to_epsg() != 4326:
            gps_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
        else:
            gps_bounds = bounds
        lon_min, lat_min, lon_max, lat_max = gps_bounds
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                Path(tif_path).name, crs.to_string() if crs else "None",
                lat_min, lon_min, lat_max, lon_max,
                src.res[0], src.res[1],
                float(zmin), float(zmax)
            ])

def print_sample_latlon_points(tif_path):
    with rasterio.open(tif_path) as src:
        band = src.read(1)
        nodata = src.nodata
        rows, cols = band.shape
        valid = [(r, c) for r in range(rows) for c in range(cols)
                 if nodata is None or band[r, c] != nodata]
        if len(valid) < 2:
            print("⚠️ Not enough valid points.")
            return
        samples = random.sample(valid, 2)
        for row, col in samples:
            x, y = src.transform * (col, row)
            lon, lat = transform(src.crs, "EPSG:4326", [x], [y])
            print(f"🧪 Sample point: {lat[0]:.6f}, {lon[0]:.6f}")

def rasterize_file(args_tuple):
    laz_path, output_dir, resolution, force, csv_path, epsg = args_tuple
    laz_path = Path(laz_path)
    out_path = Path(output_dir) / (laz_path.stem + "_dsm.tif")
    if out_path.exists() and not force:
        print(f"[SKIP] {out_path.name} exists.")
        return str(out_path)
    pipeline = build_pipeline(laz_path, out_path, resolution, epsg)
    jpath = out_path.with_suffix(".json")
    with open(jpath, 'w') as f:
        json.dump(pipeline, f)
    try:
        subprocess.run(["pdal", "pipeline", str(jpath)], check=True, capture_output=True, text=True)
        print(f"[DONE] {out_path.name}")
        log_raster_extent(out_path, csv_path)
        return str(out_path)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {laz_path.name}: {e.stderr}")
        return None
    finally:
        jpath.unlink(missing_ok=True)

def merge_tiles_rasterio(tile_paths, out_path):
    datasets = []
    crs_map = {}
    fallback_crs = None
    for path in tile_paths:
        ds = rasterio.open(path)
        if ds.crs is None:
            if fallback_crs is None:
                fallback_crs = CRS.from_epsg(int(prompt_for_epsg().split(":")[-1]))
            ds.crs = fallback_crs
        datasets.append(ds)
        crs_map.setdefault(ds.crs.to_string(), []).append(path)
    if len(crs_map) > 1:
        print("❌ CRS mismatch in tiles. Merge failed.")
        sys.exit(1)
    mosaic, transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": transform})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    print(f"[SUCCESS] Merged DSM written to {out_path}")

def cleanup_files(file_paths, label):
    print(f"[CLEANUP] Removing {len(file_paths)} {label} files...")
    for p in file_paths:
        try: Path(p).unlink()
        except Exception as e: print(f"Failed: {p}: {e}")

def main():
    check_dependencies()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lasdir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--merged", default="merged_dsm.tif")
    parser.add_argument("--cleanup-tiles", action="store_true")
    parser.add_argument("--cleanup-lidar", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    check_cpu_advice(args.workers)
    idir = Path(args.lasdir)
    odir = Path(args.outdir)
    odir.mkdir(parents=True, exist_ok=True)
    merged_path = odir / args.merged
    csv_path = odir / "dsm_tile_index.csv"

    files = sorted(idir.glob("*.las") + idir.glob("*.laz"))
    print(f"🔍 Found {len(files)} LIDAR files. Rasterizing...")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "crs", "lat_min", "lon_min", "lat_max", "lon_max", "res_x_m", "res_y_m", "zmin_m", "zmax_m"])

    tasks = []
    for f in files:
        epsg = get_epsg_code(f)
        if not epsg:
            print(f"[WARN] No EPSG in {f}")
            epsg = prompt_for_epsg()
        else:
            epsg = f"EPSG:{epsg}"
        tasks.append((f, odir, args.resolution, args.force, csv_path, epsg))

    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(rasterize_file, tasks), total=len(tasks)))

    valid = [r for r in results if r and Path(r).exists()]
    print(f"\n✅ Rasterization complete. {len(valid)} tiles ready.")

    if valid and not args.no_merge:
        merge_tiles_rasterio(valid, merged_path)
        if args.cleanup_tiles:
            cleanup_files(valid, "tile DSMs")
    elif args.no_merge:
        print("ℹ️ Merge skipped by flag.")
    else:
        print("⚠️ Nothing to merge.")

    if args.cleanup_lidar:
        cleanup_files([str(f) for f in files], "LIDAR")

    if valid:
        print_sample_latlon_points(valid[0])

if __name__ == "__main__":
    main()
