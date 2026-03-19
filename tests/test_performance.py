"""Tests for performance optimization features."""
import os
import multiprocessing
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from jpmapper.io.raster import _get_optimal_workers, _optimize_gdal_cache, HAS_PSUTIL
from jpmapper.cli.analyze_utils import _get_optimal_analysis_workers

psutil_available = pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")


class TestPerformanceOptimizations:
    """Test suite for performance optimization features."""

    def test_get_optimal_workers_explicit(self):
        """Test that explicit worker count is respected."""
        assert _get_optimal_workers(1) == 1
        assert _get_optimal_workers(4) == 4
        assert _get_optimal_workers(8) == 8
        assert _get_optimal_workers(0) == 1
        assert _get_optimal_workers(-5) == 1

    def test_get_optimal_workers_auto_detection(self):
        """Test automatic worker detection."""
        workers = _get_optimal_workers(None)
        assert workers >= 1
        cpu_count = multiprocessing.cpu_count()
        assert workers <= cpu_count

    @psutil_available
    @patch('psutil.virtual_memory')
    def test_get_optimal_workers_memory_limited(self, mock_memory):
        """Test that worker count is limited by available memory."""
        mock_memory.return_value.available = 2 * 1024**3  # 2GB
        workers = _get_optimal_workers(None)
        assert workers == 1

    @psutil_available
    @patch('psutil.virtual_memory')
    def test_get_optimal_workers_memory_abundant(self, mock_memory):
        """Test worker count with abundant memory."""
        mock_memory.return_value.available = 32 * 1024**3  # 32GB
        workers = _get_optimal_workers(None)
        cpu_count = multiprocessing.cpu_count()
        expected_max = max(1, min(8, cpu_count - 1))
        assert workers <= expected_max

    @patch('jpmapper.io.raster.HAS_PSUTIL', False)
    def test_get_optimal_workers_fallback_no_psutil(self):
        """Test fallback behavior when psutil is not available."""
        workers = _get_optimal_workers(None)
        cpu_count = multiprocessing.cpu_count()
        expected = max(1, min(8, cpu_count - 1))
        assert workers == expected

    def test_get_optimal_analysis_workers(self):
        """Test analysis-specific worker optimization."""
        assert _get_optimal_analysis_workers(1) == 1
        assert _get_optimal_analysis_workers(4) == 4

        workers = _get_optimal_analysis_workers(None)
        assert workers >= 1

    @psutil_available
    def test_optimize_gdal_cache_with_psutil(self):
        """Test GDAL cache optimization with psutil available."""
        original_cache = os.environ.get("GDAL_CACHEMAX")
        try:
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            assert cache_size >= 512
            assert cache_size <= 4096
        finally:
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]

    @psutil_available
    @patch('psutil.virtual_memory')
    def test_optimize_gdal_cache_memory_calculation(self, mock_memory):
        """Test GDAL cache calculation based on available memory."""
        original_cache = os.environ.get("GDAL_CACHEMAX")
        try:
            mock_memory.return_value.available = 8 * 1024**3  # 8GB
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            expected = int(8 * 1024 * 0.25)  # 2048MB
            assert cache_size == expected
        finally:
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]

    @psutil_available
    @patch('psutil.virtual_memory')
    def test_optimize_gdal_cache_minimum_and_maximum(self, mock_memory):
        """Test GDAL cache minimum and maximum limits."""
        original_cache = os.environ.get("GDAL_CACHEMAX")
        try:
            mock_memory.return_value.available = 1 * 1024**3  # 1GB
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            assert cache_size >= 512

            mock_memory.return_value.available = 32 * 1024**3  # 32GB
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            assert cache_size <= 4096
        finally:
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]

    @patch('jpmapper.io.raster.HAS_PSUTIL', False)
    def test_optimize_gdal_cache_fallback_no_psutil(self):
        """Test GDAL cache fallback when psutil is not available."""
        original_cache = os.environ.get("GDAL_CACHEMAX")
        try:
            if "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
            _optimize_gdal_cache()
            cache_size = os.environ.get("GDAL_CACHEMAX", "0")
            assert cache_size == "1024"
        finally:
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]

    def test_performance_optimization_integration(self):
        """Test that performance optimizations work together."""
        raster_workers = _get_optimal_workers(None)
        analysis_workers = _get_optimal_analysis_workers(None)

        assert raster_workers > 0
        assert analysis_workers > 0

        original_cache = os.environ.get("GDAL_CACHEMAX")
        try:
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            assert cache_size > 0
        finally:
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]


class TestPerformanceInAPI:
    """Test performance optimizations in API functions."""

    def test_rasterization_worker_optimization(self):
        """Test that rasterization functions use optimal worker detection."""
        auto_workers = _get_optimal_workers(None)
        explicit_workers = _get_optimal_workers(4)

        assert auto_workers >= 1
        assert explicit_workers == 4

        test_inputs = [None, 1, 2, 4, 8]
        for input_workers in test_inputs:
            result = _get_optimal_workers(input_workers)
            assert result >= 1

    def test_worker_parameter_validation(self):
        """Test that worker parameters are properly validated."""
        assert _get_optimal_workers(0) == 1
        assert _get_optimal_workers(-1) == 1
        assert _get_optimal_workers(-10) == 1

        large_workers = _get_optimal_workers(1000)
        cpu_count = multiprocessing.cpu_count()
        assert large_workers <= cpu_count * 2
        assert large_workers >= 1

    @psutil_available
    @patch('jpmapper.io.raster.log')
    def test_performance_logging(self, mock_log):
        """Test that performance optimizations are logged."""
        _optimize_gdal_cache()
        mock_log.info.assert_called()


class TestPerformanceEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch('multiprocessing.cpu_count')
    def test_single_core_system(self, mock_cpu_count):
        """Test behavior on single-core systems."""
        mock_cpu_count.return_value = 1
        assert _get_optimal_workers(None) == 1
        assert _get_optimal_analysis_workers(None) == 1

    @psutil_available
    @patch('psutil.virtual_memory')
    def test_very_low_memory_system(self, mock_memory):
        """Test behavior with very low memory."""
        mock_memory.return_value.available = 512 * 1024**2  # 512MB
        workers = _get_optimal_workers(None)
        assert workers == 1

    def test_worker_bounds_checking(self):
        """Test that worker counts are properly bounded."""
        test_cases = [0, -1, -100, 1, 2, 4, 8, 16, 32, 64, 128, 1000]
        for worker_count in test_cases:
            result = _get_optimal_workers(worker_count)
            assert result >= 1, f"Worker count {worker_count} resulted in {result} < 1"

    def test_environment_variable_preservation(self):
        """Test that GDAL environment variables are handled properly."""
        original_cache = os.environ.get("GDAL_CACHEMAX")
        try:
            os.environ["GDAL_CACHEMAX"] = "2048"
            _optimize_gdal_cache()
            new_cache = os.environ.get("GDAL_CACHEMAX")
            assert new_cache is not None
            assert int(new_cache) > 0
        finally:
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
