"""
Comprehensive tests for jpmapper.analysis.los module.

Uses real temporary GeoTIFF fixtures instead of production-side test logic.
"""

import math
import numpy as np
import pytest
import rasterio
from pathlib import Path
from unittest.mock import patch

from jpmapper.analysis import los
from jpmapper.exceptions import AnalysisError, GeometryError, NoDataError


class TestFunctions:
    """Test standalone utility functions."""

    def test_first_fresnel_radius(self):
        dist = np.array([100, 500, 1000])
        freq_ghz = 5.8
        result = los._first_fresnel_radius(dist, freq_ghz)

        assert isinstance(result, np.ndarray)
        assert result.shape == dist.shape
        assert result[1] > result[0]
        assert result[2] > result[1]

        wavelength = 0.3 / freq_ghz
        expected = np.sqrt(wavelength * 100 / 2.0)
        assert abs(result[0] - expected) < 0.001

    def test_fresnel_radius_function(self):
        radius = los.fresnel_radius(250, 1000, 5.8)
        assert radius > 0

        with pytest.raises(ValueError, match="distance_m must be positive"):
            los.fresnel_radius(0, 1000, 5.8)
        with pytest.raises(ValueError, match="distance_total_m must be positive"):
            los.fresnel_radius(250, 0, 5.8)
        with pytest.raises(ValueError, match="frequency_ghz must be positive"):
            los.fresnel_radius(250, 1000, 0)

    def test_point_to_pixel(self):
        point = (100.0, 200.0)
        transform = [1.0, 0.0, 0.0, 0.0, -1.0, 250.0, 0.0, 0.0, 1.0]
        col, row = los.point_to_pixel(point, transform)
        assert isinstance(col, int)
        assert isinstance(row, int)

        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            los.point_to_pixel((100,), transform)
        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            los.point_to_pixel("invalid", transform)
        with pytest.raises(GeometryError, match="Invalid transform"):
            los.point_to_pixel(point, [1.0, 2.0])
        with pytest.raises(GeometryError, match="Invalid transform"):
            los.point_to_pixel(point, "invalid")

    def test_distance_between_points(self):
        point_a = (40.0, -74.0)
        point_b = (40.1, -74.1)
        distance = los.distance_between_points(point_a, point_b)
        assert distance > 0
        assert distance < 50000

        distance = los.distance_between_points(point_a, point_a)
        assert abs(distance) < 0.001

        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            los.distance_between_points((40.0,), point_b)
        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            los.distance_between_points(point_a, "invalid")


class TestSnapToValid:
    """Test _snap_to_valid with real GeoTIFF fixtures."""

    def test_snap_to_valid_center(self, flat_dsm):
        """Test snapping at a point that is within the raster."""
        with rasterio.open(flat_dsm) as ds:
            # Convert the center of the raster to WGS84 to get valid lon/lat
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            cx = (ds.bounds.left + ds.bounds.right) / 2
            cy = (ds.bounds.bottom + ds.bounds.top) / 2
            lon, lat = tf.transform(cx, cy)

            (snap_lat, snap_lon), elev, dx = los._snap_to_valid(ds, lon, lat)
            assert isinstance(elev, float)
            assert elev == pytest.approx(3.048, abs=0.5)
            assert dx >= 0

    def test_feet_crs_returns_meters(self, flat_dsm):
        """EPSG:6539 (feet) DSM returns elevation in meters."""
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            cx = (ds.bounds.left + ds.bounds.right) / 2
            cy = (ds.bounds.bottom + ds.bounds.top) / 2
            lon, lat = tf.transform(cx, cy)
            (_, _), elev, _ = los._snap_to_valid(ds, lon, lat)
            assert elev == pytest.approx(10.0 * 0.3048006096012192, rel=0.01)

    def test_meter_crs_no_conversion(self, flat_dsm_meters):
        """EPSG:32618 (metre) DSM returns elevation unchanged."""
        with rasterio.open(flat_dsm_meters) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            cx = (ds.bounds.left + ds.bounds.right) / 2
            cy = (ds.bounds.bottom + ds.bounds.top) / 2
            lon, lat = tf.transform(cx, cy)
            (_, _), elev, _ = los._snap_to_valid(ds, lon, lat)
            assert elev == pytest.approx(10.0, rel=0.01)

    def test_snap_to_valid_no_data(self, tmp_path):
        """Test that _snap_to_valid raises NoDataError when all nodata."""
        import rasterio
        from rasterio.transform import from_bounds

        dsm_path = tmp_path / "nodata_dsm.tif"
        data = np.full((1, 10, 10), -9999.0, dtype=np.float32)
        transform = from_bounds(980000, 190000, 981000, 191000, 10, 10)

        with rasterio.open(
            dsm_path, "w", driver="GTiff",
            height=10, width=10, count=1, dtype="float32",
            crs="EPSG:6539", transform=transform, nodata=-9999,
        ) as dst:
            dst.write(data)

        with rasterio.open(dsm_path) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon, lat = tf.transform(980500, 190500)

            with pytest.raises(NoDataError):
                los._snap_to_valid(ds, lon, lat, max_px=5)


class TestIsClearDirect:
    """Test is_clear_direct with real GeoTIFF fixtures."""

    def test_clear_path_flat_dsm(self, flat_dsm):
        """A high-altitude LOS over flat terrain should be clear."""
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980100, 190500)
            lon2, lat2 = tf.transform(980900, 190500)

        result = los.is_clear_direct(
            lon1, lat1, 50.0,  # 50m above ground
            lon2, lat2, 50.0,
            str(flat_dsm)
        )
        assert result is True

    def test_blocked_path_hill_dsm(self, hill_dsm):
        """Low-altitude LOS across a hill should be blocked."""
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            # Points on opposite edges of the raster, hill in center
            lon1, lat1 = tf.transform(980050, 190500)
            lon2, lat2 = tf.transform(980950, 190500)

        result = los.is_clear_direct(
            lon1, lat1, 0.0,  # ground level
            lon2, lat2, 0.0,
            str(hill_dsm)
        )
        assert result is False

    def test_clear_high_alt_over_hill(self, hill_dsm):
        """High-altitude LOS over a hill should be clear."""
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980050, 190500)
            lon2, lat2 = tf.transform(980950, 190500)

        result = los.is_clear_direct(
            lon1, lat1, 200.0,  # well above hill peak of ~60m
            lon2, lat2, 200.0,
            str(hill_dsm)
        )
        assert result is True


class TestProfile:
    """Test the profile function with real GeoTIFF fixtures."""

    def test_profile_flat_dsm(self, flat_dsm):
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980200, 190500)
            lon2, lat2 = tf.transform(980800, 190500)

            pt_a = (lat1, lon1)
            pt_b = (lat2, lon2)

            distances, terrain, fresnel = los.profile(ds, pt_a, pt_b, n_samples=10)

            assert len(distances) == 10
            assert len(terrain) == 10
            assert len(fresnel) == 10
            assert distances[0] == 0
            assert distances[-1] > 0

    def test_profile_hill_dsm(self, hill_dsm):
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980100, 190500)
            lon2, lat2 = tf.transform(980900, 190500)

            pt_a = (lat1, lon1)
            pt_b = (lat2, lon2)

            distances, terrain, fresnel = los.profile(ds, pt_a, pt_b, n_samples=20)

            assert len(distances) == 20
            # Middle points should be higher than endpoints (hill in center)
            mid = len(terrain) // 2
            assert terrain[mid] > terrain[0]


class TestComputeProfile:
    """Test compute_profile function."""

    def test_compute_profile_with_real_file(self, flat_dsm):
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980200, 190500)
            lon2, lat2 = tf.transform(980800, 190500)

        pt_a = (lat1, lon1)
        pt_b = (lat2, lon2)

        distances, elevations, total_dist = los.compute_profile(
            str(flat_dsm), pt_a, pt_b, n_samples=10
        )

        assert len(distances) == 10
        assert len(elevations) == 10
        assert isinstance(total_dist, float)
        assert total_dist > 0
        assert distances[0] == 0
        assert distances[-1] == pytest.approx(total_dist, rel=1e-6)

    def test_compute_profile_with_dataset(self, flat_dsm):
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980200, 190500)
            lon2, lat2 = tf.transform(980800, 190500)

            pt_a = (lat1, lon1)
            pt_b = (lat2, lon2)

            distances, elevations, total_dist = los.compute_profile(
                ds, pt_a, pt_b, n_samples=8
            )

            assert len(distances) == 8
            assert len(elevations) == 8
            assert isinstance(total_dist, float)

    def test_compute_profile_nonexistent_file(self):
        with pytest.raises(AnalysisError):
            los.compute_profile(
                "nonexistent_file.tif",
                (40.0, -74.0), (40.1, -74.1)
            )


class TestIsClear:
    """Test the main is_clear function."""

    def test_is_clear_flat_terrain(self, flat_dsm):
        """Flat terrain with a small mast (above buffer) should have clear LOS."""
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980200, 190500)
            lon2, lat2 = tf.transform(980800, 190500)

        pt_a = (lat1, lon1)
        pt_b = (lat2, lon2)

        # With max_mast_height_m=5 and step_m=1, the iterative search should
        # find a small mast height that clears the 2m buffer over flat terrain
        clear, mast_height, ground_a, ground_b, snap_dist = los.is_clear(
            str(flat_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=5, step_m=1.0
        )

        assert clear is True
        assert mast_height >= 0
        assert ground_a == pytest.approx(3.048, abs=0.5)
        assert ground_b == pytest.approx(3.048, abs=0.5)

    def test_is_clear_blocked_by_hill(self, hill_dsm):
        """LOS across a hill at ground level should be blocked."""
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980050, 190500)
            lon2, lat2 = tf.transform(980950, 190500)

        pt_a = (lat1, lon1)
        pt_b = (lat2, lon2)

        clear, mast_height, ground_a, ground_b, snap_dist = los.is_clear(
            str(hill_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=0
        )

        assert clear is False

    def test_is_clear_with_mast_over_hill(self, hill_dsm):
        """A tall enough mast should clear the hill."""
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980050, 190500)
            lon2, lat2 = tf.transform(980950, 190500)

        pt_a = (lat1, lon1)
        pt_b = (lat2, lon2)

        clear, mast_height, ground_a, ground_b, snap_dist = los.is_clear(
            str(hill_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=200, step_m=10.0
        )

        assert clear is True
        assert mast_height > 0
        assert mast_height <= 200

    def test_is_clear_with_dataset(self, flat_dsm):
        """Test passing an already-open dataset."""
        with rasterio.open(flat_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980200, 190500)
            lon2, lat2 = tf.transform(980800, 190500)

            pt_a = (lat1, lon1)
            pt_b = (lat2, lon2)

            clear, mast_height, ground_a, ground_b, snap_dist = los.is_clear(
                ds, pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=5, step_m=1.0
            )

            assert clear is True

    def test_is_clear_specific_altitudes(self, hill_dsm):
        """Test with explicit from_alt and to_alt."""
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980050, 190500)
            lon2, lat2 = tf.transform(980950, 190500)

        pt_a = (lat1, lon1)
        pt_b = (lat2, lon2)

        # High altitude should clear the hill
        clear, mast, gA, gB, snap = los.is_clear(
            str(hill_dsm), pt_a, pt_b, from_alt=200.0, to_alt=200.0
        )
        assert clear is True

        # Ground level should be blocked
        clear, mast, gA, gB, snap = los.is_clear(
            str(hill_dsm), pt_a, pt_b, from_alt=0.0, to_alt=0.0
        )
        assert clear is False

    def test_is_clear_nonexistent_file(self):
        """Test with non-existent file raises error."""
        with pytest.raises(Exception):
            los.is_clear(
                "nonexistent.tif",
                (40.0, -74.0), (40.1, -74.1)
            )


class TestComputeProfileWithDataset:
    """Test _compute_profile_with_dataset."""

    def test_with_real_dataset(self, hill_dsm):
        with rasterio.open(hill_dsm) as ds:
            from pyproj import Transformer
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(980100, 190500)
            lon2, lat2 = tf.transform(980900, 190500)

            distances, elevations, total_dist = los._compute_profile_with_dataset(
                ds, (lat1, lon1), (lat2, lon2), n_samples=10
            )

            assert len(distances) == 10
            assert len(elevations) == 10
            assert isinstance(total_dist, float)
            assert total_dist > 0


if __name__ == "__main__":
    pytest.main([__file__])
