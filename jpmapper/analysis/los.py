"""Line-of-sight and Fresnel-zone clearance utilities."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from pyproj import Transformer


def first_fresnel_radius(distance_m: float, freq_ghz: float) -> float:
    """Return radius (metres) of the first Fresnel zone at *distance_m*."""
    wavelength = 0.3 / freq_ghz  # λ = c / f  (c≈3e8 m/s ➜ 0.3 m·GHz)
    return math.sqrt(wavelength * distance_m / 2)


def _sampler(
    dsm: rasterio.DatasetReader, pts_px: np.ndarray
) -> np.ndarray:  # shape (N,)
    """Fast bilinear sampler for terrain elevations at pixel coords."""
    row, col = pts_px[:, 1], pts_px[:, 0]
    return np.squeeze(list(dsm.sample(np.stack([col, row], axis=1))), axis=1)


def profile(
    dsm_path: Path | str,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    n_samples: int = 500,
    freq_ghz: float = 5.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample terrain & Fresnel radius between *pt_a* and *pt_b*.

    Returns three aligned 1-D arrays: `dist_m`, `terrain_m`, `fresnel_m`.
    """
    with rasterio.open(dsm_path) as ds:
        tf = Transformer.from_crs(4326, ds.crs, always_xy=True)
        x0, y0 = tf.transform(*pt_a[::-1])
        x1, y1 = tf.transform(*pt_b[::-1])

        # Geo-distance in plane of projection
        total_dist = math.hypot(x1 - x0, y1 - y0)

        # Interpolate points in map coords ➜ pixel coords
        xs = np.linspace(x0, x1, n_samples)
        ys = np.linspace(y0, y1, n_samples)
        cols, rows = (
            (xs - ds.transform.c) / ds.transform.a,
            (ys - ds.transform.f) / ds.transform.e,
        )
        terrain = _sampler(ds, np.column_stack([cols, rows]))
        dist = np.linspace(0.0, total_dist, n_samples)
        fresnel = first_fresnel_radius(dist, freq_ghz)

    return dist, terrain, fresnel


def is_clear(
    dsm_path: Path | str,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    freq_ghz: float = 5.8,
    n_samples: int = 500,
) -> bool:
    """Return *True* if the LOS path between A and B is unobstructed."""
    dist, terrain, fresnel = profile(dsm_path, pt_a, pt_b, n_samples, freq_ghz)

    # Straight line heights between antennas (assume both at 0 m above ground)
    line = np.linspace(terrain[0], terrain[-1], n_samples)

    clearance = line - terrain
    needed = fresnel * 0.6  # 60 % Fresnel clearance rule-of-thumb

    return np.all(clearance >= needed)
