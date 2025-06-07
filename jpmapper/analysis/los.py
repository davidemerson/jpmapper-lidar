"""LOS / Fresnel utilities – snaps to nearest valid DSM cell, skips out-of-bounds."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from pyproj import Transformer


# ─────────────────────────── helpers ────────────────────────────────────────────
def first_fresnel_radius(dist_m, freq_ghz: float):
    """First-Fresnel radius (m) – works with ndarray or scalar distance."""
    λ = 0.3 / freq_ghz
    return np.sqrt(λ * np.asarray(dist_m) / 2)


def _sample(ds: rasterio.DatasetReader, px: np.ndarray) -> np.ndarray:
    """Bilinear elevation sampler for float pixel coords (N×2 array)."""
    return np.squeeze(list(ds.sample(px)), axis=1)


def _in_bounds(ds: rasterio.DatasetReader, lon: float, lat: float) -> bool:
    """True if WGS84 lon/lat lies inside raster footprint (simple bbox check)."""
    tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
    left, bottom = tf.transform(ds.bounds.left, ds.bounds.bottom)
    right, top = tf.transform(ds.bounds.right, ds.bounds.top)
    return left <= lon <= right and bottom <= lat <= top


def _snap_to_valid(
    ds: rasterio.DatasetReader,
    lon: float,
    lat: float,
    max_px: int = 15,
) -> Tuple[Tuple[float, float], float]:
    """
    Snap WGS84 lon/lat to nearest non-nodata pixel.

    Returns ((lat, lon), ground_elev_m). Raises ValueError if none found.
    """
    tf_wgs84_to_dsm = Transformer.from_crs(4326, ds.crs, always_xy=True)
    tf_dsm_to_wgs84 = Transformer.from_crs(ds.crs, 4326, always_xy=True)

    x, y = tf_wgs84_to_dsm.transform(lon, lat)
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
            xv = ds.transform.c + c_valid * ds.transform.a
            yv = ds.transform.f + r_valid * ds.transform.e
            lon_v, lat_v = tf_dsm_to_wgs84.transform(xv, yv)
            elev = float(
                ds.read(
                    1, window=((r_valid, r_valid + 1), (c_valid, c_valid + 1))
                )[0, 0]
            )
            return (lat_v, lon_v), elev
    raise ValueError("No valid DSM cell within search radius")


# ─────────────────────────── main routines ──────────────────────────────────────
def profile(
    dsm_path: Path | str,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    n_samples: int = 500,
    freq_ghz: float = 5.8,
):
    """Return (dist_m, ground_m, fresnel_m, gA, gB, snapA_WGS, snapB_WGS)."""
    with rasterio.open(dsm_path) as ds:
        # snap endpoints
        (lat_a, lon_a), gA = _snap_to_valid(ds, pt_a[1], pt_a[0])
        (lat_b, lon_b), gB = _snap_to_valid(ds, pt_b[1], pt_b[0])

        tf = Transformer.from_crs(4326, ds.crs, always_xy=True)
        x0, y0 = tf.transform(lon_a, lat_a)
        x1, y1 = tf.transform(lon_b, lat_b)

        xs = np.linspace(x0, x1, n_samples)
        ys = np.linspace(y0, y1, n_samples)
        cols = (xs - ds.transform.c) / ds.transform.a
        rows = (ys - ds.transform.f) / ds.transform.e
        ground = _sample(ds, np.column_stack([cols, rows]))

        dist = np.linspace(0.0, math.hypot(x1 - x0, y1 - y0), n_samples)
        fresnel = first_fresnel_radius(dist, freq_ghz)

    return dist, ground, fresnel, gA, gB, (lat_a, lon_a), (lat_b, lon_b)


def is_clear(
    dsm_path: Path | str,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    freq_ghz: float = 5.8,
    n_samples: int = 500,
    max_height_m: int = 5,
    step_m: int = 1,
):
    """
    Compute LOS/Fresnel clearance.

    Returns (clear?, first_height_m, groundA, groundB, snap_dist_m)
    ground* = None if point lies outside DSM bounds.
    """
    with rasterio.open(dsm_path) as ds:
        if not (_in_bounds(ds, pt_a[1], pt_a[0]) and _in_bounds(ds, pt_b[1], pt_b[0])):
            return False, -1, None, None, None

    dist, ground, fresnel, gA, gB, snapA, _ = profile(
        dsm_path, pt_a, pt_b, n_samples, freq_ghz
    )
    needed = fresnel * 0.6

    # horizontal snap distance (m) between requested A and snapped A
    snap_dist = math.dist(pt_a[::-1], snapA[::-1]) * 111_000  # rough deg→m

    for h in range(0, max_height_m + 1, step_m):
        line = np.linspace(gA + h, gB + h, n_samples)
        if np.all(line - ground >= needed):
            return True, h, gA, gB, snap_dist
    return False, -1, gA, gB, snap_dist
