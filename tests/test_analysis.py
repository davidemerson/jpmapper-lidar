"""Tests for the analysis module using real GeoTIFF fixtures."""
import math
from pathlib import Path
import pytest
import numpy as np

from jpmapper.analysis.los import (
    is_clear,
    compute_profile,
    fresnel_radius,
    point_to_pixel,
    distance_between_points,
)
from jpmapper.exceptions import (
    AnalysisError,
    GeometryError,
)


class TestFunctions:
    """Test suite for individual analysis functions."""

    def test_fresnel_radius(self):
        radius = fresnel_radius(
            distance_m=1000,
            distance_total_m=2000,
            frequency_ghz=5.8,
        )
        expected = 17.32 / np.sqrt(1000) * np.sqrt(1000 * 1000 / (5.8 * 2000))
        assert radius == pytest.approx(expected, rel=1e-6)

        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(0, 1000, 5.8)
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(500, 0, 5.8)
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(500, 1000, 0)
        with pytest.raises(ValueError, match="must be positive"):
            fresnel_radius(500, 1000, -5.8)

    def test_point_to_pixel(self):
        transform = [0.1, 0, 0, 0, -0.1, 0, 0, 0, 1]

        x, y = point_to_pixel((0, 0), transform)
        assert x == 0
        assert y == 0

        x, y = point_to_pixel((1, 1), transform)
        assert x == 10
        assert y == -10

        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            point_to_pixel("invalid", transform)
        with pytest.raises(GeometryError, match="Invalid transform"):
            point_to_pixel((0, 0), "invalid")

    def test_distance_between_points(self):
        distance = distance_between_points((0, 0), (0, 0.01))
        assert distance == pytest.approx(1113.2, rel=1e-2)

        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            distance_between_points("invalid", (0, 0))
        with pytest.raises(GeometryError, match="Invalid point coordinates"):
            distance_between_points((0, 0), "invalid")


class TestAnalysisWithRealRasters:
    """Integration tests using real temporary GeoTIFF fixtures."""

    def _get_endpoints(self, dsm_path, x1=980200, x2=980800, y=190500):
        """Helper to get WGS84 endpoints from projected coords."""
        import rasterio
        from pyproj import Transformer

        with rasterio.open(dsm_path) as ds:
            tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
            lon1, lat1 = tf.transform(x1, y)
            lon2, lat2 = tf.transform(x2, y)
        return (lat1, lon1), (lat2, lon2)

    def test_compute_profile(self, hill_dsm):
        pt_a, pt_b = self._get_endpoints(hill_dsm, 980100, 980900)

        distances, elevations, total_distance = compute_profile(
            str(hill_dsm), pt_a, pt_b, n_samples=10
        )

        assert len(distances) == 10
        assert len(elevations) == 10
        assert distances[0] == 0
        assert distances[-1] == pytest.approx(total_distance, rel=1e-6)

        # Hill in the middle: mid-profile elevation should be higher than edges
        mid = len(elevations) // 2
        assert elevations[mid] > elevations[0]

    def test_is_clear_with_clear_path(self, flat_dsm):
        pt_a, pt_b = self._get_endpoints(flat_dsm)

        # Flat terrain needs a small mast to clear the 2m alt_buffer
        is_clear_result, mast_height, ground_a, ground_b, distance = is_clear(
            str(flat_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=5, step_m=1.0
        )

        assert is_clear_result is True
        assert mast_height >= 0
        assert ground_a == pytest.approx(3.048, abs=0.5)
        assert ground_b == pytest.approx(3.048, abs=0.5)

    def test_is_clear_with_blocked_path(self, hill_dsm):
        pt_a, pt_b = self._get_endpoints(hill_dsm, 980050, 980950)

        is_clear_result, mast_height, ground_a, ground_b, distance = is_clear(
            str(hill_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=0
        )

        assert is_clear_result is False

    def test_is_clear_with_mast(self, hill_dsm):
        pt_a, pt_b = self._get_endpoints(hill_dsm, 980050, 980950)

        # No mast allowed -> blocked
        is_clear_result, mast_height, _, _, _ = is_clear(
            str(hill_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=0
        )
        assert is_clear_result is False

        # With a tall mast -> should find a solution
        is_clear_result, mast_height, _, _, _ = is_clear(
            str(hill_dsm), pt_a, pt_b, freq_ghz=5.8, max_mast_height_m=200, step_m=10.0
        )
        assert is_clear_result is True
        assert mast_height > 0
        assert mast_height <= 200
