"""Tests for the JPMapper web API."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from jpmapper.web.models import AnalyzeRequest, PointCoord, SnapInfo


# ── Model validation tests ────────────────────────────────────────────────

class TestModels:
    def test_point_coord_valid(self):
        p = PointCoord(lat=40.689, lon=-73.986)
        assert p.lat == 40.689

    def test_point_coord_out_of_range(self):
        with pytest.raises(Exception):
            PointCoord(lat=91, lon=0)
        with pytest.raises(Exception):
            PointCoord(lat=0, lon=181)

    def test_analyze_request_defaults(self):
        req = AnalyzeRequest(
            point_a=PointCoord(lat=40.0, lon=-74.0),
            point_b=PointCoord(lat=40.1, lon=-74.1),
        )
        assert req.mast_a_height_m == 0.0
        assert req.freq_ghz == 5.8

    def test_analyze_request_negative_mast(self):
        with pytest.raises(Exception):
            AnalyzeRequest(
                point_a=PointCoord(lat=40.0, lon=-74.0),
                point_b=PointCoord(lat=40.1, lon=-74.1),
                mast_a_height_m=-1,
            )

    def test_analyze_request_freq_too_high(self):
        with pytest.raises(Exception):
            AnalyzeRequest(
                point_a=PointCoord(lat=40.0, lon=-74.0),
                point_b=PointCoord(lat=40.1, lon=-74.1),
                freq_ghz=500,
            )


# ── API endpoint tests (mocked DSM) ──────────────────────────────────────

@pytest.fixture
def mock_dsm():
    """Create a mock rasterio DatasetReader."""
    ds = MagicMock()
    ds.crs = "EPSG:2263"
    ds.bounds = MagicMock()
    ds.bounds.left = 970000.0
    ds.bounds.right = 1080000.0
    ds.bounds.bottom = 145000.0
    ds.bounds.top = 272000.0
    ds.nodata = -9999.0
    return ds


@pytest.fixture
def client(mock_dsm):
    """Create a test client with mocked DSM."""
    os.environ["JPMAPPER_DSM_PATH"] = "/fake/dsm.tif"
    with patch("jpmapper.web.app.rasterio") as mock_rio:
        mock_rio.open.return_value = mock_dsm
        from jpmapper.web.app import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["dsm_loaded"] is True


class TestBoundsEndpoint:
    def test_bounds(self, client):
        with patch("jpmapper.web.routes.Transformer") as mock_tf_cls:
            mock_tf = MagicMock()
            mock_tf.transform.side_effect = [
                (-74.05, 40.57),  # left, bottom
                (-73.88, 40.79),  # right, top
            ]
            mock_tf_cls.from_crs.return_value = mock_tf
            resp = client.get("/api/bounds")
            assert resp.status_code == 200
            data = resp.json()
            assert "min_lat" in data
            assert "max_lon" in data


class TestAnalyzeEndpoint:
    def test_analyze_success(self, client):
        mock_result = {
            "clear": True,
            "distance_m": 623.4,
            "surface_height_a_m": 33.7,
            "surface_height_b_m": 28.1,
            "clearance_min_m": 4.2,
            "mast_a_height_m": 3.0,
            "mast_b_height_m": 5.0,
        }
        mock_distances = np.linspace(0, 623.4, 10)
        mock_terrain = np.full(10, 30.0)
        mock_fresnel = np.full(10, 0.5)

        def mock_snap(ds, lon, lat, max_px=50):
            return (lat, lon), 30.0, 0.0  # no snap needed

        with patch("jpmapper.web.routes._snap_to_valid", side_effect=mock_snap), \
             patch("jpmapper.web.routes.analyze_los", return_value=mock_result), \
             patch("jpmapper.web.routes.generate_profile", return_value=(mock_distances, mock_terrain, mock_fresnel)):
            resp = client.post("/api/analyze", json={
                "point_a": {"lat": 40.689, "lon": -73.986},
                "point_b": {"lat": 40.694, "lon": -73.990},
                "mast_a_height_m": 3.0,
                "mast_b_height_m": 5.0,
                "freq_ghz": 5.8,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["clear"] is True
            assert data["distance_m"] == 623.4
            assert "profile" in data
            assert len(data["profile"]["distances_m"]) == 10
            assert "obstructions" in data
            assert data["snap_a"] is None
            assert data["snap_b"] is None

    def test_analyze_with_snap(self, client):
        mock_result = {
            "clear": True,
            "distance_m": 623.4,
            "surface_height_a_m": 33.7,
            "surface_height_b_m": 28.1,
            "clearance_min_m": 4.2,
            "mast_a_height_m": 3.0,
            "mast_b_height_m": 5.0,
        }
        mock_distances = np.linspace(0, 623.4, 10)
        mock_terrain = np.full(10, 30.0)
        mock_fresnel = np.full(10, 0.5)

        def mock_snap(ds, lon, lat, max_px=50):
            # Simulate snap: shift lat by a small amount, report 15m snap
            return (lat + 0.0001, lon + 0.0001), 30.0, 15.0

        with patch("jpmapper.web.routes._snap_to_valid", side_effect=mock_snap), \
             patch("jpmapper.web.routes.analyze_los", return_value=mock_result), \
             patch("jpmapper.web.routes.generate_profile", return_value=(mock_distances, mock_terrain, mock_fresnel)):
            resp = client.post("/api/analyze", json={
                "point_a": {"lat": 40.689, "lon": -73.986},
                "point_b": {"lat": 40.694, "lon": -73.990},
                "mast_a_height_m": 3.0,
                "mast_b_height_m": 5.0,
                "freq_ghz": 5.8,
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["snap_a"] is not None
            assert data["snap_a"]["snap_distance_m"] == 15.0
            assert data["snap_a"]["original_lat"] == 40.689
            assert data["snap_b"] is not None

    def test_analyze_validation_error(self, client):
        resp = client.post("/api/analyze", json={
            "point_a": {"lat": 91, "lon": -73.986},
            "point_b": {"lat": 40.694, "lon": -73.990},
        })
        assert resp.status_code == 422

    def test_analyze_outside_dsm(self, client):
        from jpmapper.exceptions import NoDataError
        with patch("jpmapper.web.routes._snap_to_valid", side_effect=NoDataError("Outside DSM")):
            resp = client.post("/api/analyze", json={
                "point_a": {"lat": 40.689, "lon": -73.986},
                "point_b": {"lat": 40.694, "lon": -73.990},
            })
            assert resp.status_code == 400
            data = resp.json()
            assert "No LiDAR coverage" in data["detail"]
            assert "shaded red zones" in data["detail"]


class TestCoverageEndpoint:
    def test_coverage(self, client, mock_dsm):
        # Reset the cache for this test
        import jpmapper.web.routes as routes_mod
        routes_mod._coverage_cache = None

        mock_dsm.shape = (1000, 1000)
        mock_dsm.transform = MagicMock()
        mock_dsm.transform.a = 1.0
        mock_dsm.transform.c = 970000.0
        mock_dsm.transform.e = -1.0
        mock_dsm.transform.f = 272000.0

        # Simulate a raster with some nodata and some valid
        def fake_read(band, window=None):
            r0, r1 = window[0]
            c0, c1 = window[1]
            h = r1 - r0
            w = c1 - c0
            arr = np.full((h, w), 50.0)
            # Make top-left quadrant all nodata
            if r0 < 500 and c0 < 500:
                arr[:] = -9999.0
            return arr

        mock_dsm.read = fake_read

        with patch("jpmapper.web.routes.Transformer") as mock_tf_cls:
            mock_tf = MagicMock()
            mock_tf.transform.side_effect = lambda x, y: (x / 10000 - 170, y / 10000 - 230)
            mock_tf_cls.from_crs.return_value = mock_tf

            resp = client.get("/api/coverage")
            assert resp.status_code == 200
            data = resp.json()
            assert "cells" in data
            assert data["cell_size_px"] == 500
            # Should have the nodata cell
            assert len(data["cells"]) > 0
            # All returned cells should have < 90% coverage
            for cell in data["cells"]:
                assert cell["coverage_pct"] < 90


class TestIndexPage:
    def test_index_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"JPMapper LOS Analyzer" in resp.content

    def test_static_css(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200

    def test_static_js(self, client):
        resp = client.get("/static/app.js")
        assert resp.status_code == 200
