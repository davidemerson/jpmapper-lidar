
import argparse
import os
import glob
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from pyproj import Transformer
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from tqdm import tqdm


def geocode(address):
    geolocator = Nominatim(user_agent="jpmapper")
    location = geolocator.geocode(address)
    if not location:
        raise ValueError(f"Address not found: {address}")
    return location.latitude, location.longitude


def resolve_coords(loc):
    if os.path.isfile(loc):
        raise ValueError("File input not supported for coordinates. Use direct values or addresses.")
    try:
        lat, lon = map(float, loc.split(","))
        return lat, lon
    except:
        return geocode(loc)


def load_dsm_dataset(dsm_path):
    if os.path.isdir(dsm_path):
        tifs = sorted(glob.glob(os.path.join(dsm_path, "*.tif")))
        if not tifs:
            raise FileNotFoundError("No .tif files found in DSM directory.")
        srcs = [rasterio.open(fp) for fp in tifs]
        mosaic, out_trans = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })
        return mosaic[0], meta
    else:
        with rasterio.open(dsm_path) as src:
            return src.read(1), src.meta


def get_elevation_from_dsm(lat, lon, dsm, meta):
    transformer = Transformer.from_crs("EPSG:4326", meta["crs"], always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = ~meta["transform"] * (x, y)
    row, col = int(row), int(col)
    if 0 <= row < dsm.shape[0] and 0 <= col < dsm.shape[1]:
        return dsm[row, col]
    else:
        raise ValueError("Coordinates out of DSM bounds.")


def fresnel_radius(d, f_hz):
    return 17.32 * np.sqrt(d / 1000 / (4 * f_hz / 1e9))


def analyze_path(lat1, lon1, lat2, lon2, elev1, elev2, dsm, meta, freq_ghz):
    d = geodesic((lat1, lon1), (lat2, lon2)).meters
    f_hz = freq_ghz * 1e9
    r = fresnel_radius(d, f_hz)

    # Placeholder analysis logic
    obstruction_count = 0
    partial_count = 0
    suspect_count = 0

    print("
=== Link Summary ===")
    print(f"Total distance: {d:.2f} meters")
    print(f"Point A elevation: {elev1:.2f} m")
    print(f"Point B elevation: {elev2:.2f} m")
    print(f"Frequency: {freq_ghz:.3f} GHz ({f_hz:.0f} Hz)")
    print(f"First Fresnel zone radius (approx): {r:.2f} meters")
    print(f"Obstruction analysis: {obstruction_count} full, {partial_count} partial, {suspect_count} suspect")

    if obstruction_count > 0:
        print("Verdict: Obstructed")
    elif partial_count > 0 or suspect_count > 0:
        print("Verdict: Partially obstructed or unclear")
    else:
        print("Verdict: Clear")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-a", required=True, help="Start point (lat,lon or address)")
    parser.add_argument("--point-b", required=True, help="End point (lat,lon or address)")
    parser.add_argument("--dsm", required=True, help="Path to DSM raster or folder")
    parser.add_argument("--frequency-ghz", type=float, required=True, help="Transmission frequency in GHz")
    parser.add_argument("--override-a", type=float, help="Override elevation at point A (meters)")
    parser.add_argument("--override-b", type=float, help="Override elevation at point B (meters)")
    args = parser.parse_args()

    dsm, meta = load_dsm_dataset(args.dsm)
    lat1, lon1 = resolve_coords(args.point_a)
    lat2, lon2 = resolve_coords(args.point_b)

    elev1 = args.override_a if args.override_a is not None else get_elevation_from_dsm(lat1, lon1, dsm, meta)
    elev2 = args.override_b if args.override_b is not None else get_elevation_from_dsm(lat2, lon2, dsm, meta)

    analyze_path(lat1, lon1, lat2, lon2, elev1, elev2, dsm, meta, args.frequency_ghz)


if __name__ == "__main__":
    main()
