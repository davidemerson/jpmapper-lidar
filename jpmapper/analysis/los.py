"""Line-of-sight & Fresnel-zone utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import rasterio as rio
from pyproj import Transformer


# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class OutOfBounds(Exception):  # raised when a point is outside the DSM extent
    distance_m: float


# ────────────────────────────────────────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (metres) between two WGS84 lat/lon points."""
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = φ2 - φ1
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _dsm_bbox_wgs84(ds: rio.DatasetReader) -> Tuple[float, float, float, float]:
    """Return DSM bounds in WGS84 (lon_min, lat_min, lon_max, lat_max)."""
    tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
    left, bottom, right, top = ds.bounds
    lon_min, lat_min = tf.transform(left, bottom)
    lon_max, lat_max = tf.transform(right, top)
    return lon_min, lat_min, lon_max, lat_max


def _distance_to_bbox(
    lon: float,
    lat: float,
    bbox: Tuple[float, float, float, float],
) -> float:
    """Shortest haversine distance (metres) from point to bounding box."""
    lon_min, lat_min, lon_max, lat_max = bbox
    # Clip point to bbox to find nearest edge / corner
    lon_clamped = max(min(lon, lon_max), lon_min)
    lat_clamped = max(min(lat, lat_max), lat_min)
    return _haversine(lat, lon, lat_clamped, lon_clamped)


# ────────────────────────────────────────────────────────────────────────────────
def _snap_to_valid(
    ds: rio.DatasetReader,
    lon: float,
    lat: float,
    max_px: int = 3,
) -> Tuple[Tuple[float, float], float]:
    """
    Snap WGS84 lon/lat to nearest DSM pixel with data.
    Raises OutOfBounds if nothing within *max_px*.
    """
    tf_fwd = Transformer.from_crs(4326, ds.crs, always_xy=True)
    tf_inv = Transformer.from_crs(ds.crs, 4326, always_xy=True)

    x, y = tf_fwd.transform(lon, lat)
    col = int((x - ds.transform.c) / ds.transform.a + 0.5)
    row = int((y - ds.transform.f) / ds.transform.e + 0.5)

    nodata = ds.nodata if ds.nodata is not None else -9999
    for d in range(max_px + 1):
        rmin, rmax = row - d, row + d
        cmin, cmax = col - d, col + d
        window = ds.read(1, window=((rmin, rmax + 1), (cmin, cmax + 1)))
        mask = window != nodata
        if mask.any():
            r_off, c_off = np.argwhere(mask)[0]
            r_valid, c_valid = rmin + r_off, cmin + c_off
            x_val = ds.transform.c + c_valid * ds.transform.a
            y_val = ds.transform.f + r_valid * ds.transform.e
            lon_val, lat_val = tf_inv.transform(x_val, y_val)
            elev = float(
                ds.read(1, window=((r_valid, r_valid + 1), (c_valid, c_valid + 1)))[
                    0, 0
                ]
            )
            return (lat_val, lon_val), elev

    # Nothing found – compute distance to bbox and raise
    bbox = _dsm_bbox_wgs84(ds)
    raise OutOfBounds(_distance_to_bbox(lon, lat, bbox))


# ────────────────────────────────────────────────────────────────────────────────
def first_fresnel_radius(distance_m: float, freq_ghz: float) -> np.ndarray:
    lam = 0.3 / freq_ghz  # metres
    return np.sqrt(lam * distance_m / 2)


def profile(
    ds: rio.DatasetReader,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    samples: int,
    freq_ghz: float,
):
    tf = Transformer.from_crs(4326, ds.crs, always_xy=True)
    try:
        (lat_a, lon_a), ground_a = _snap_to_valid(ds, pt_a[1], pt_a[0])
        (lat_b, lon_b), ground_b = _snap_to_valid(ds, pt_b[1], pt_b[0])
    except OutOfBounds as err:
        raise  # re-raise for caller

    x0, y0 = tf.transform(lon_a, lat_a)
    x1, y1 = tf.transform(lon_b, lat_b)

    xs = np.linspace(x0, x1, samples)
    ys = np.linspace(y0, y1, samples)
    rows, cols = ~ds.transform * np.vstack([xs, ys])
    rows = rows.astype(int)
    cols = cols.astype(int)

    terrain = ds.read(1)[rows, cols]
    dist = np.linspace(0, _haversine(lat_a, lon_a, lat_b, lon_b), samples)

    fresnel = first_fresnel_radius(dist, freq_ghz)
    return dist, terrain, fresnel, ground_a, ground_b


# ────────────────────────────────────────────────────────────────────────────────
def analyze_link(
    ds: rio.DatasetReader,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    freq_ghz: float,
    *,
    max_height_m: int,
    raster_units: str,
) -> Dict:
    """
    Return a dict with link metrics.  If either endpoint is outside
    DSM extent, 'coverage' is False and 'oob_m' gives the distance.
    """
    n_samples = 512
    try:
        dist, terrain, fresnel, gA, gB = profile(ds, pt_a, pt_b, n_samples, freq_ghz)
    except OutOfBounds as e:
        return {
            "coverage": False,
            "oob_m": round(e.distance_m, 1),
        }

    # Add antenna heights progressively
    for mast in range(max_height_m + 1):
        tx = gA + mast
        rx = gB + mast
        los = tx + (dist / dist[-1]) * (rx - tx)
        clearance = los - terrain
        off = fresnel - clearance
        if off.max() <= 0:  # clear
            return {
                "coverage": True,
                "clear": True,
                "mast_height_m": mast,
                "min_clearance_m": clearance.min(),
                "worst_offset_m": off.max(),
                "samples": int(n_samples),
                "ground_a_m": gA,
                "ground_b_m": gB,
                "snap_distance_m": 0.0,
            }
    # not clear
    return {
        "coverage": True,
        "clear": False,
        "mast_height_m": -1,
        "min_clearance_m": clearance.min(),
        "worst_offset_m": off.max(),
        "samples": int(n_samples),
        "ground_a_m": gA,
        "ground_b_m": gB,
        "snap_distance_m": 0.0,
    }
