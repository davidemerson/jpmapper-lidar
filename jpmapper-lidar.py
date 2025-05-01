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
            raise FileNotFoundError(f"No .tif files found in {dsm_path}")
        srcs = [rasterio.open(fp) for fp in tifs]
        mosaic, out_trans = merge(srcs)
        out_meta = srcs[0].meta.copy()
        out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
        return mosaic[0], out_meta
    elif os.path.isfile(dsm_path):
        with rasterio.open(dsm_path) as src:
            return src.read(1), src.meta
    else:
        raise FileNotFoundError(f"DSM input path invalid: {dsm_path}")


def interpolate_coords(p1, p2, steps):
    return np.linspace(p1, p2, steps)


def extract_elevation(latlons, dsm_array, dsm_meta):
    elevations = []
    transformer = Transformer.from_crs("EPSG:4326", dsm_meta["crs"], always_xy=True)
    transform = dsm_meta["transform"]

    for lat, lon in tqdm(latlons, desc="Extracting elevation", unit="pt"):
        x, y = transformer.transform(lon, lat)
        col, row = ~transform * (x, y)
        row, col = int(row), int(col)
        if 0 <= row < dsm_array.shape[0] and 0 <= col < dsm_array.shape[1]:
            elevations.append(dsm_array[row, col])
        else:
            elevations.append(np.nan)
    return np.array(elevations)


def fresnel_radius(d1, d2, freq_mhz):
    c = 3e8  # speed of light m/s
    wavelength = c / (freq_mhz * 1e6)
    return np.sqrt(wavelength * d1 * d2 / (d1 + d2))


def plot_profile(dists, elevations, fresnel_curve):
    plt.figure(figsize=(10, 4))
    plt.plot(dists, elevations, label="Elevation (m)")
    plt.plot(dists, fresnel_curve + np.minimum(elevations[0], elevations[-1]), 'r--', label="1st Fresnel Zone")
    plt.title("Terrain and Fresnel Zone Profile")
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_obstructions(coords, dists, elevations, fresnel, start_elev, end_elev):
    obstructed = []
    clearance_line = np.linspace(start_elev, end_elev, len(elevations))
    for i, (latlon, d, elev, fr, line) in enumerate(tqdm(zip(coords, dists, elevations, fresnel, clearance_line),
                                                        total=len(coords), desc="Analyzing obstructions", unit="pt")):
        clearance = elev - (line + fr)
        if clearance > 0:
            obstructed.append({
                "index": i,
                "lat": float(latlon[0]),
                "lon": float(latlon[1]),
                "distance_m": float(d),
                "elevation_m": float(elev),
                "fresnel_radius_m": float(fr),
                "fresnel_clearance_m": float(clearance)
            })
    return obstructed


def main():
    parser = argparse.ArgumentParser(description="Line-of-sight terrain & Fresnel zone analyzer")
    parser.add_argument("--p1", required=True, help="Point 1 (lat,lon or address)")
    parser.add_argument("--p2", required=True, help="Point 2 (lat,lon or address)")
    parser.add_argument("--freq", required=True, type=float, help="Frequency in MHz")
    parser.add_argument("--dsm", required=True, help="Path to DSM GeoTIFF or folder of tiles")
    parser.add_argument("--steps", type=int, default=100, help="Number of interpolation points")
    parser.add_argument("--json", type=str, help="Optional JSON output path for obstruction report")

    args = parser.parse_args()

    latlon1 = resolve_coords(args.p1)
    latlon2 = resolve_coords(args.p2)

    coords = interpolate_coords(np.array(latlon1), np.array(latlon2), args.steps)
    dsm_array, dsm_meta = load_dsm_dataset(args.dsm)
    elevations = extract_elevation(coords, dsm_array, dsm_meta)

    dists = np.array([geodesic(latlon1, tuple(p)).meters for p in tqdm(coords, desc="Computing distances", unit="pt")])
    total_distance = dists[-1]

    fresnel = np.array([fresnel_radius(d, total_distance - d, args.freq) for d in tqdm(dists, desc="Computing Fresnel zone", unit="pt")])

    print(f"\nTotal path distance: {total_distance:.2f} m")
    print(f"Max Fresnel radius: {np.max(fresnel):.2f} m")

    obstructed = analyze_obstructions(coords, dists, elevations, fresnel, elevations[0], elevations[-1])

    print(f"\nNumber of obstructed points: {len(obstructed)}")
    if obstructed:
        print("\nObstructed Points:")
        for o in obstructed:
            print(f"  {o['distance_m']:.1f} m | {o['lat']:.5f}, {o['lon']:.5f} | Elev: {o['elevation_m']:.2f} m | Clearance: {o['fresnel_clearance_m']:.2f} m")

    if args.json:
        output_data = {
            "start_point": {"lat": latlon1[0], "lon": latlon1[1]},
            "end_point": {"lat": latlon2[0], "lon": latlon2[1]},
            "frequency_mhz": args.freq,
            "total_distance_m": total_distance,
            "obstructions": obstructed
        }
        with open(args.json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nObstruction data exported to: {args.json}")

    plot_profile(dists, elevations, fresnel)


if __name__ == "__main__":
    main()
