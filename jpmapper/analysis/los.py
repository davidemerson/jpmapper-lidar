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

import logging
import math
from typing import Tuple, Union, Optional, List
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
import rasterio
from pyproj import Transformer

from jpmapper.exceptions import AnalysisError, NoDataError

# --------------------------------------------------------------------------- geometry helpers


def _first_fresnel_radius(dist: np.ndarray, freq_ghz: float) -> np.ndarray:
    wavelength = 0.3 / freq_ghz  # lambda = c / f
    return np.sqrt(wavelength * dist / 2.0)


def _snap_to_valid(
    ds: rasterio.DatasetReader, lon: float, lat: float, max_px: int = 50
) -> Tuple[Tuple[float, float], float, float]:
    """Snap WGS84 lon/lat to nearest valid DSM cell.

    Returns:
      * snapped (lat, lon)
      * surface elevation (m, in DSM units)
      * horizontal distance (m) between requested and snapped point

    Raises:
      NoDataError: if no valid data found within search radius
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

        if window.ndim == 3:
            window = window[0]

        mask = window != nodata
        if mask.any():
            r_off, c_off = np.argwhere(mask)[0]
            r_valid, c_valid = row - d + r_off, col - d + c_off
            x_valid = ds.transform.c + c_valid * ds.transform.a
            y_valid = ds.transform.f + r_valid * ds.transform.e
            lon_valid, lat_valid = tf_dsm_to_wgs84.transform(x_valid, y_valid)

            elev_window = ds.read(
                1,
                window=((r_valid, r_valid + 1), (c_valid, c_valid + 1)),
                boundless=True,
                fill_value=nodata,
            )
            if elev_window.ndim == 3:
                elev_window = elev_window[0]

            elev = float(elev_window[0, 0])
            dx = math.hypot(lon_valid - lon, lat_valid - lat) * 111_320  # ~ m per deg
            return (lat_valid, lon_valid), elev, dx

    raise NoDataError(
        f"No valid DSM data within {max_px} pixels of ({lat:.6f}, {lon:.6f})"
    )


# --------------------------------------------------------------------------- public API


def is_clear_direct(
    from_lon: float,
    from_lat: float,
    from_alt: float,
    to_lon: float,
    to_lat: float,
    to_alt: float,
    dsm_file: Union[str, Path, rasterio.DatasetReader],
    n_samples: int = 25,
    alt_buffer_m: float = 2.0,
    snap_max_px: int = 50,
) -> bool:
    """Check if line of sight between two points is clear of obstructions.

    Args:
        from_lon: WGS84 longitude of start point
        from_lat: WGS84 latitude of start point
        from_alt: altitude of start point (m)
        to_lon: WGS84 longitude of end point
        to_lat: WGS84 latitude of end point
        to_alt: altitude of end point (m)
        dsm_file: path to DSM GeoTIFF or open rasterio dataset
        n_samples: number of points to sample along path
        alt_buffer_m: buffer (m) to add to points to clear small obstacles
        snap_max_px: max search distance for finding valid DSM cells

    Returns:
        True if line of sight is clear of obstructions
    """
    if isinstance(dsm_file, (str, Path)):
        with rasterio.open(dsm_file) as ds:
            clear, _, _ = _is_clear_with_dataset(
                from_lon, from_lat, from_alt,
                to_lon, to_lat, to_alt,
                ds, n_samples, alt_buffer_m, snap_max_px
            )
            return clear
    else:
        clear, _, _ = _is_clear_with_dataset(
            from_lon, from_lat, from_alt,
            to_lon, to_lat, to_alt,
            dsm_file, n_samples, alt_buffer_m, snap_max_px
        )
        return clear


def _is_clear_with_dataset(
    from_lon: float,
    from_lat: float,
    from_alt: float,
    to_lon: float,
    to_lat: float,
    to_alt: float,
    ds: rasterio.DatasetReader,
    n_samples: int = 25,
    alt_buffer_m: float = 2.0,
    snap_max_px: int = 50,
    freq_ghz: float = 5.8,
) -> Tuple[bool, float, float]:
    """Implementation of is_clear using an open dataset.

    Returns:
        Tuple of (is_clear, min_clearance_m, fresnel_pct):
        - is_clear: True if LOS clears terrain with 60% first Fresnel zone
        - min_clearance_m: Minimum clearance between LOS line and terrain (m)
        - fresnel_pct: Worst-case Fresnel zone obstruction (0.0 = fully clear, 1.0 = fully blocked)
    """
    (from_lat_valid, from_lon_valid), from_ground_alt, from_dx = _snap_to_valid(
        ds, from_lon, from_lat, max_px=snap_max_px
    )
    (to_lat_valid, to_lon_valid), to_ground_alt, to_dx = _snap_to_valid(
        ds, to_lon, to_lat, max_px=snap_max_px
    )

    from_alt_adjusted = from_alt + from_ground_alt
    to_alt_adjusted = to_alt + to_ground_alt

    distances, elevations, total_distance = compute_profile(
        ds,
        (from_lat_valid, from_lon_valid),
        (to_lat_valid, to_lon_valid),
        n_samples=n_samples,
    )

    x = distances / total_distance  # normalized distance 0..1
    los_y = (1 - x) * from_alt_adjusted + x * to_alt_adjusted

    # Geometric clearance: LOS line minus terrain
    clearance = los_y - elevations
    min_clearance = float(clearance.min())

    # Fresnel zone check: compute first Fresnel radius at each sample point
    # F1 = sqrt(lambda * d1 * d2 / D) where d1, d2 are distances to endpoints
    wavelength = 0.3 / freq_ghz  # meters
    d1 = distances  # distance from start
    d2 = total_distance - distances  # distance from end
    # Avoid division by zero at endpoints
    with np.errstate(invalid='ignore'):
        f1_radius = np.sqrt(wavelength * d1 * d2 / total_distance)
    f1_radius = np.nan_to_num(f1_radius, nan=0.0)

    # Required clearance is 60% of F1 radius (standard RF planning threshold)
    required_clearance = 0.6 * f1_radius

    # Fresnel obstruction: how much of the required clearance zone is blocked
    # 0.0 = fully clear, 1.0+ = terrain reaches or exceeds LOS line
    fresnel_margin = clearance - required_clearance
    worst_margin = float(fresnel_margin.min())
    if required_clearance.max() > 0:
        # Normalize: 0 = exactly at 60% F1, negative = obstructed
        worst_f1_idx = fresnel_margin.argmin()
        if required_clearance[worst_f1_idx] > 0:
            fresnel_pct = float(1.0 - clearance[worst_f1_idx] / required_clearance[worst_f1_idx])
        else:
            fresnel_pct = 0.0
    else:
        fresnel_pct = 0.0

    # Path is clear if 60% of first Fresnel zone is unobstructed at all points
    is_clear = bool(worst_margin > 0)

    return is_clear, min_clearance, fresnel_pct


def profile(
    ds: rasterio.DatasetReader,
    pt_a: Tuple[float, float],
    pt_b: Tuple[float, float],
    n_samples: int = 256,
    freq_ghz: float = 5.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate terrain and Fresnel zone profile between two points.

    Args:
        ds: Raster dataset (DSM)
        pt_a: First point as (latitude, longitude)
        pt_b: Second point as (latitude, longitude)
        n_samples: Number of points to sample along the path
        freq_ghz: Frequency in GHz

    Returns:
        Tuple of (distances_m, terrain_heights_m, fresnel_radii_m)
    """
    (lat_a, lon_a), gA, _ = _snap_to_valid(ds, pt_a[1], pt_a[0])
    (lat_b, lon_b), gB, _ = _snap_to_valid(ds, pt_b[1], pt_b[0])

    tf = Transformer.from_crs(4326, ds.crs, always_xy=True)

    x1, y1 = tf.transform(lon_a, lat_a)
    x2, y2 = tf.transform(lon_b, lat_b)
    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)
    ground = np.empty(n_samples, dtype=float)

    raster_bounds = ds.bounds
    nodata_value = ds.nodata

    def _snap_to_valid_elevation(x, y, max_search_px=50):
        """Snap a point to nearest valid elevation data."""
        try:
            row, col = ds.index(x, y)

            if row < 0 or row >= ds.height or col < 0 or col >= ds.width:
                return None, 0.0

            for search_radius in range(1, max_search_px + 1):
                valid_pixels = []
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        if abs(dr) != search_radius and abs(dc) != search_radius:
                            continue

                        test_row = row + dr
                        test_col = col + dc

                        if 0 <= test_row < ds.height and 0 <= test_col < ds.width:
                            try:
                                pixel_val = ds.read(1, window=((test_row, test_row+1), (test_col, test_col+1)))[0, 0]
                                if nodata_value is None or pixel_val != nodata_value:
                                    distance_px = math.sqrt(dr*dr + dc*dc)
                                    distance_m = distance_px * abs(ds.transform[0])
                                    valid_pixels.append((float(pixel_val), distance_m))
                            except Exception:
                                continue

                if valid_pixels:
                    valid_pixels.sort(key=lambda p: p[1])
                    return valid_pixels[0]

            return None, 0.0
        except Exception:
            return None, 0.0

    for i, (x, y) in enumerate(zip(xs, ys)):
        if not (raster_bounds.left <= x <= raster_bounds.right and
                raster_bounds.bottom <= y <= raster_bounds.top):
            elevation, snap_dist = _snap_to_valid_elevation(x, y)
            if elevation is not None:
                ground[i] = elevation
            else:
                ground[i] = 0.0
        else:
            try:
                sampled_values = list(ds.sample([(x, y)], 1))
                elevation = sampled_values[0][0]

                if nodata_value is not None and elevation == nodata_value:
                    snapped_elevation, snap_dist = _snap_to_valid_elevation(x, y)
                    if snapped_elevation is not None:
                        ground[i] = snapped_elevation
                    else:
                        ground[i] = 0.0
                else:
                    ground[i] = float(elevation)
            except Exception:
                snapped_elevation, snap_dist = _snap_to_valid_elevation(x, y)
                if snapped_elevation is not None:
                    ground[i] = snapped_elevation
                else:
                    ground[i] = 0.0

    distance = np.linspace(0, math.hypot(x2 - x1, y2 - y1), n_samples)
    fresnel = _first_fresnel_radius(distance, freq_ghz)

    return distance, ground, fresnel


def compute_profile(
    dsm_path: Union[str, Path, rasterio.DatasetReader],
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    n_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute terrain profile between two points.

    Args:
        dsm_path: Path to DSM GeoTIFF file or open dataset
        point_a: First point as (latitude, longitude)
        point_b: Second point as (latitude, longitude)
        n_samples: Number of samples along the path

    Returns:
        Tuple of (distances_m, terrain_heights_m, total_distance_m)

    Raises:
        AnalysisError: If there's an error processing the DSM
    """
    try:
        if not isinstance(dsm_path, (str, Path)):
            return _compute_profile_with_dataset(dsm_path, point_a, point_b, n_samples)

        with rasterio.open(dsm_path) as ds:
            return _compute_profile_with_dataset(ds, point_a, point_b, n_samples)
    except (AnalysisError, NoDataError):
        raise
    except Exception as e:
        raise AnalysisError(f"Error computing terrain profile: {e}") from e


def _compute_profile_with_dataset(
    ds: rasterio.DatasetReader,
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    n_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Internal implementation to compute profile with an open dataset."""
    (lat_a, lon_a), gA, _ = _snap_to_valid(ds, point_a[1], point_a[0])
    (lat_b, lon_b), gB, _ = _snap_to_valid(ds, point_b[1], point_b[0])

    tf = Transformer.from_crs(4326, ds.crs, always_xy=True)

    x1, y1 = tf.transform(lon_a, lat_a)
    x2, y2 = tf.transform(lon_b, lat_b)

    total_distance = math.hypot(x2 - x1, y2 - y1)

    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)
    elevations = np.empty(n_samples, dtype=float)

    nodata = ds.nodata if ds.nodata is not None else -9999

    for i, (x, y) in enumerate(zip(xs, ys)):
        try:
            sampled = list(ds.sample([(x, y)], 1))
            val = sampled[0][0]
            elevations[i] = 0.0 if (nodata is not None and val == nodata) else float(val)
        except Exception:
            elevations[i] = 0.0

    distances = np.linspace(0, total_distance, n_samples)
    return distances, elevations, total_distance


def compute_profile_with_coords(
    dsm_path: Union[str, Path, rasterio.DatasetReader],
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    n_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, float]]]:
    """Compute terrain profile between two points, including sampled coordinates.

    Args:
        dsm_path: Path to DSM GeoTIFF file or open dataset
        point_a: First point as (latitude, longitude)
        point_b: Second point as (latitude, longitude)
        n_samples: Number of samples along the path

    Returns:
        Tuple of (distances_m, terrain_heights_m, total_distance_m, sample_coords)

    Raises:
        AnalysisError: If there's an error processing the DSM
    """
    try:
        if not isinstance(dsm_path, (str, Path)):
            return _compute_profile_with_coords_dataset(dsm_path, point_a, point_b, n_samples)

        with rasterio.open(dsm_path) as ds:
            return _compute_profile_with_coords_dataset(ds, point_a, point_b, n_samples)
    except (AnalysisError, NoDataError):
        raise
    except Exception as e:
        raise AnalysisError(f"Error computing terrain profile: {e}") from e


def _compute_profile_with_coords_dataset(
    ds: rasterio.DatasetReader,
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    n_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float, List[Tuple[float, float]]]:
    """Internal implementation to compute profile with an open dataset, returning coordinates."""
    distances, elevations, fresnel = profile(ds, point_a, point_b, n_samples, freq_ghz=5.8)
    total_distance = distances[-1] if len(distances) > 0 else 0.0

    tf_wgs84_to_dsm = Transformer.from_crs(4326, ds.crs, always_xy=True)
    tf_dsm_to_wgs84 = Transformer.from_crs(ds.crs, 4326, always_xy=True)

    x1, y1 = tf_wgs84_to_dsm.transform(point_a[1], point_a[0])
    x2, y2 = tf_wgs84_to_dsm.transform(point_b[1], point_b[0])

    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)

    coords = []
    for x, y in zip(xs, ys):
        lon, lat = tf_dsm_to_wgs84.transform(x, y)
        coords.append((lat, lon))

    return distances, elevations, total_distance, coords


def fresnel_radius(distance_m: float, distance_total_m: float, frequency_ghz: float) -> float:
    """Calculate the radius of the first Fresnel zone at a specific point on the path.

    Args:
        distance_m: Distance from one endpoint to the point in meters
        distance_total_m: Total distance between endpoints in meters
        frequency_ghz: Frequency in GHz

    Returns:
        Radius of the first Fresnel zone in meters

    Raises:
        ValueError: If any of the input values are not positive
    """
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    if distance_total_m <= 0:
        raise ValueError("distance_total_m must be positive")
    if frequency_ghz <= 0:
        raise ValueError("frequency_ghz must be positive")

    distance_to_other_m = distance_total_m - distance_m
    return 17.32 * math.sqrt(distance_m * distance_to_other_m / (frequency_ghz * distance_total_m))


def point_to_pixel(point: Tuple[float, float], transform: Union[list, tuple]) -> Tuple[int, int]:
    """Convert a geographic point to pixel coordinates using a transform.

    Args:
        point: Point coordinates as (x, y) or (lon, lat)
        transform: Transform matrix as a list or tuple of 9 elements

    Returns:
        Tuple of (column, row) pixel coordinates

    Raises:
        GeometryError: If point or transform is invalid
    """
    from jpmapper.exceptions import GeometryError

    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise GeometryError("Invalid point coordinates")

    if not isinstance(transform, (list, tuple)) or len(transform) != 9:
        raise GeometryError("Invalid transform")

    try:
        x, y = point
        col = int(round((x - transform[2]) / transform[0]))
        row = int(round((y - transform[5]) / transform[4]))
        return col, row
    except Exception as e:
        raise GeometryError(f"Error converting point to pixel: {e}") from e


def distance_between_points(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    """Calculate the great-circle distance between two points.

    Args:
        point_a: First point as (latitude, longitude)
        point_b: Second point as (latitude, longitude)

    Returns:
        Distance in meters

    Raises:
        GeometryError: If point coordinates are invalid
    """
    from jpmapper.exceptions import GeometryError

    if not isinstance(point_a, (list, tuple)) or len(point_a) != 2:
        raise GeometryError("Invalid point coordinates")

    if not isinstance(point_b, (list, tuple)) or len(point_b) != 2:
        raise GeometryError("Invalid point coordinates")

    try:
        lat1, lon1 = point_a
        lat2, lon2 = point_b

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters

        return c * r
    except Exception as e:
        raise GeometryError(f"Error calculating distance: {e}") from e


def is_clear(
    dsm: Union[str, Path, rasterio.DatasetReader],
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    freq_ghz: float = 5.8,
    max_mast_height_m: int = 5,
    step_m: float = 1.0,
    n_samples: int = 256,
    from_alt: Optional[float] = None,
    to_alt: Optional[float] = None,
) -> Tuple[bool, int, float, float, float]:
    """Check if line of sight between two points is clear of obstructions.

    Args:
        dsm: Path to DSM GeoTIFF or open dataset
        point_a: First point as (latitude, longitude)
        point_b: Second point as (latitude, longitude)
        freq_ghz: Signal frequency in GHz
        max_mast_height_m: Maximum mast height to try (m)
        step_m: Step size for mast height search (m)
        n_samples: Number of points to sample along path
        from_alt: Specific mast height at point A (m)
        to_alt: Specific mast height at point B (m)

    Returns:
        Tuple of (is_clear, mast_height, surface_a, surface_b, snap_distance)
    """
    if isinstance(dsm, (str, Path)):
        with rasterio.open(dsm) as ds:
            return _is_clear_points(ds, point_a, point_b, freq_ghz, max_mast_height_m, step_m, n_samples, from_alt, to_alt)
    else:
        return _is_clear_points(dsm, point_a, point_b, freq_ghz, max_mast_height_m, step_m, n_samples, from_alt, to_alt)


def _is_clear_points(
    ds: rasterio.DatasetReader,
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
    freq_ghz: float = 5.8,
    max_mast_height_m: int = 5,
    step_m: float = 1.0,
    n_samples: int = 256,
    from_alt: Optional[float] = None,
    to_alt: Optional[float] = None,
) -> Tuple[bool, int, float, float, float]:
    """Internal implementation for the API is_clear function."""
    (lat_a, lon_a), ground_a, snap_a = _snap_to_valid(ds, point_a[1], point_a[0])
    (lat_b, lon_b), ground_b, snap_b = _snap_to_valid(ds, point_b[1], point_b[0])

    snap_distance = max(snap_a, snap_b)

    if from_alt is not None and to_alt is not None:
        clear, min_clr, fresnel_pct = _is_clear_with_dataset(
            lon_a, lat_a, from_alt,
            lon_b, lat_b, to_alt,
            ds, n_samples, 2.0, freq_ghz=freq_ghz
        )

        if clear:
            return True, 0, ground_a, ground_b, snap_distance
        else:
            return False, -1, ground_a, ground_b, snap_distance
    else:
        clear, min_clr, fresnel_pct = _is_clear_with_dataset(
            lon_a, lat_a, 0,
            lon_b, lat_b, 0,
            ds, n_samples, 2.0, freq_ghz=freq_ghz
        )

        if clear:
            return True, 0, ground_a, ground_b, snap_distance

        current_height = step_m
        while current_height <= max_mast_height_m:
            clear, min_clr, fresnel_pct = _is_clear_with_dataset(
                lon_a, lat_a, current_height,
                lon_b, lat_b, current_height,
                ds, n_samples, 2.0, freq_ghz=freq_ghz
            )

            if clear:
                return True, int(current_height), ground_a, ground_b, snap_distance

            current_height += step_m

        return False, -1, ground_a, ground_b, snap_distance
