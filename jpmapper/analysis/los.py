"""
Core LOS / Fresnel functions.

Returned tuple from `is_clear`:
    clear (bool)
    mast_height_m (int | -1)
    clr_min_m (float)
    worst_overshoot_m (float)
    n_samples (int)
    ground_A_m (float)
    ground_B_m (float)
    snap_distance_m (float)
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import rasterio
from pyproj import Transformer

# --------------------------------------------------------------------------- geometry helpers


def _first_fresnel_radius(dist: np.ndarray, freq_ghz: float) -> np.ndarray:
    wavelength = 0.3 / freq_ghz  # λ = c / f
    return np.sqrt(wavelength * dist / 2.0)


def _snap_to_valid(
    ds: rasterio.DatasetReader, lon: float, lat: float, max_px: int = 5
) -> Tuple[Tuple[float, float], float, float]:
    """Snap WGS84 lon/lat to nearest valid DSM cell.

    Returns:
      * snapped (lat, lon)
      * ground elevation (m, in DSM units)
      * horizontal distance (m) between requested and snapped point
    """
    tf_wgs84_to_dsm = Transformer.from_crs(4326, ds.crs, always_xy=True)
    tf_dsm_to_wgs84 = Transformer.from_crs(ds.crs, 4326, always_xy=True)

    x, y = tf_wgs84_to_dsm.transform(lon, lat)
    col = int(round((x - ds.transform.c) / ds.transform.a))
    row = int(round((y - ds.transform.f) / ds.transform.e))

    nodata = ds.nodata if ds.nodata is not None else -9999
    for d in range(max_px + 1):
        window = ds.read(
            1,
            window=((row - d, row + d + 1), (col - d, col + d + 1)),
            boundless=True,
            fill_value=nodata,
        )
        mask = window != nodata
        if mask.any():
            r_off, c_off = np.argwhere(mask)[0]
            r_valid, c_valid = row - d + r_off, col - d + c_off
            x_valid = ds.transform.c + c_valid * ds.transform.a
            y_valid = ds.transform.f + r_valid * ds.transform.e
            lon_valid, lat_valid = tf_dsm_to_wgs84.transform(x_valid, y_valid)
            elev = float(
                ds.read(
                    1,
                    window=((r_valid, r_valid + 1), (c_valid, c_valid + 1)),
                    boundless=True,
                    fill_value=nodata,
                )[0, 0]
            )
            dx = math.hypot(lon_valid - lon, lat_valid - lat) * 111_320  # ~ m per deg
            return (lat_valid, lon_valid), elev, dx
    raise ValueError("No valid DSM cell within search radius")


# --------------------------------------------------------------------------- public API


def is_clear(
    ds: rasterio.DatasetReader,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    *,
    freq_ghz: float = 5.8,
    max_height_m: int = 5,
    n_samples: int = 256,
) -> Tuple[bool, int, float, float, int, float, float, float]:
    """
    Return whether path is clear.  If not clear at ground level, try mast
    heights (1 m … max_height_m) at **both** ends until clear.

    Returns *(clear, mast_height_m, clr_min, worst_overshoot, n_samples,
    groundA, groundB, snap_distance)*.
    """
    (lat_a, lon_a), gA, snapA = _snap_to_valid(ds, pt_a[1], pt_a[0])
    (lat_b, lon_b), gB, snapB = _snap_to_valid(ds, pt_b[1], pt_b[0])
    snap = max(snapA, snapB)

    # sample elevations
    tf = Transformer.from_crs(4326, ds.crs, always_xy=True)
    x1, y1 = tf.transform(lon_a, lat_a)
    x2, y2 = tf.transform(lon_b, lat_b)
    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)
    zs = np.empty(n_samples, dtype=float)
    ds.read(1, out=zs, samples=list(zip(xs, ys)), resampling=rasterio.enums.Resampling.nearest)
    distance = np.linspace(0, math.hypot(x2 - x1, y2 - y1), n_samples)

    fresnel = _first_fresnel_radius(distance, freq_ghz)

    def _clear(h: float) -> Tuple[bool, float, float]:
        z_tx = gA + h
        z_rx = gB + h
        z_link = z_tx + (z_rx - z_tx) * (distance / distance[-1])
        clr = z_link - zs - fresnel
        return bool((clr >= 0).all()), clr.min(), -clr.max()

    # ground first
    ok, clr_min, worst = _clear(0)
    if ok:
        return True, 0, clr_min, worst, n_samples, gA, gB, snap

    # try masts
    for h in range(1, max_height_m + 1):
        ok, clr_min, worst = _clear(h)
        if ok:
            return True, h, clr_min, worst, n_samples, gA, gB, snap

    return False, -1, clr_min, worst, n_samples, gA, gB, snap
