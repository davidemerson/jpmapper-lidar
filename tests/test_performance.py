"""Tests for performance optimization features."""
import os
import multiprocessing
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from jpmapper.io.raster import _get_optimal_workers, _optimize_gdal_cache
from jpmapper.cli.analyze_utils import _get_optimal_analysis_workers


class TestPerformanceOptimizations:
    """Test suite for performance optimization features."""
    
    def test_get_optimal_workers_explicit(self):
        """Test that explicit worker count is respected."""
        # Test explicit worker counts
        assert _get_optimal_workers(1) == 1
        assert _get_optimal_workers(4) == 4
        assert _get_optimal_workers(8) == 8
        
        # Test that negative values are clamped to 1
        assert _get_optimal_workers(0) == 1
        assert _get_optimal_workers(-5) == 1
    
    def test_get_optimal_workers_auto_detection(self):
        """Test automatic worker detection."""
        # Test auto-detection (None input)
        workers = _get_optimal_workers(None)
        
        # Should return at least 1 worker
        assert workers >= 1
        
        # Should not exceed total CPU count
        cpu_count = multiprocessing.cpu_count()
        assert workers <= cpu_count
        
        # On multi-core systems, should use multiple workers
        if cpu_count > 1:
            assert workers > 1
    
    @patch('psutil.virtual_memory')
    def test_get_optimal_workers_memory_limited(self, mock_memory):
        """Test that worker count is limited by available memory."""
        # Mock a system with limited memory (2GB available)
        mock_memory.return_value.available = 2 * 1024**3  # 2GB in bytes
        
        workers = _get_optimal_workers(None)
        
        # With 2GB available and 2GB per worker, should get 1 worker
        assert workers == 1
    
    @patch('psutil.virtual_memory')
    def test_get_optimal_workers_memory_abundant(self, mock_memory):
        """Test worker count with abundant memory."""
        # Mock a system with lots of memory (32GB available)
        mock_memory.return_value.available = 32 * 1024**3  # 32GB in bytes
        
        workers = _get_optimal_workers(None)
        
        # Should be limited by CPU count, not memory
        cpu_count = multiprocessing.cpu_count()
        expected_max = max(1, int(cpu_count * 0.75))
        assert workers <= expected_max
    
    @patch('jpmapper.io.raster.psutil')
    def test_get_optimal_workers_fallback_no_psutil(self, mock_psutil):
        """Test fallback behavior when psutil is not available."""
        # Simulate psutil ImportError
        mock_psutil.side_effect = ImportError("psutil not available")
        
        workers = _get_optimal_workers(None)
        
        # Should fall back to CPU-based calculation
        cpu_count = multiprocessing.cpu_count()
        expected = max(1, int(cpu_count * 0.75))
        assert workers == expected
    
    def test_get_optimal_analysis_workers(self):
        """Test analysis-specific worker optimization."""
        # Test explicit worker counts
        assert _get_optimal_analysis_workers(1) == 1
        assert _get_optimal_analysis_workers(4) == 4
        
        # Test auto-detection
        workers = _get_optimal_analysis_workers(None)
        assert workers >= 1
        
        # Analysis workers should be more aggressive than raster workers
        cpu_count = multiprocessing.cpu_count()
        expected_max = max(1, int(cpu_count * 0.9))
        assert workers <= expected_max
        
        # On multi-core systems, should use more CPUs than raster processing
        if cpu_count > 2:
            raster_workers = _get_optimal_workers(None)
            assert workers >= raster_workers
    
    def test_optimize_gdal_cache_with_psutil(self):
        """Test GDAL cache optimization with psutil available."""
        # Store original value
        original_cache = os.environ.get("GDAL_CACHEMAX")
        
        try:
            # Test that the function sets GDAL_CACHEMAX
            _optimize_gdal_cache()
            
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            
            # Should set a reasonable cache size
            assert cache_size >= 512  # At least 512MB
            assert cache_size <= 4096  # No more than 4GB
            
        finally:
            # Restore original value
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
    
    @patch('psutil.virtual_memory')
    def test_optimize_gdal_cache_memory_calculation(self, mock_memory):
        """Test GDAL cache calculation based on available memory."""
        # Store original value
        original_cache = os.environ.get("GDAL_CACHEMAX")
        
        try:
            # Mock 8GB available memory
            mock_memory.return_value.available = 8 * 1024**3  # 8GB in bytes
            
            _optimize_gdal_cache()
            
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            
            # Should be 25% of 8GB = 2GB = 2048MB
            expected = int(8 * 1024 * 0.25)  # 2048MB
            assert cache_size == expected
            
        finally:
            # Restore original value
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
    
    @patch('psutil.virtual_memory')
    def test_optimize_gdal_cache_minimum_and_maximum(self, mock_memory):
        """Test GDAL cache minimum and maximum limits."""
        original_cache = os.environ.get("GDAL_CACHEMAX")
        
        try:
            # Test minimum (very low memory)
            mock_memory.return_value.available = 1 * 1024**3  # 1GB
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            assert cache_size >= 512  # Should enforce minimum of 512MB
            
            # Test maximum (very high memory)
            mock_memory.return_value.available = 32 * 1024**3  # 32GB
            _optimize_gdal_cache()
            cache_size = int(os.environ.get("GDAL_CACHEMAX", "0"))
            assert cache_size <= 4096  # Should cap at 4GB
            
        finally:
            # Restore original value
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
    
    @patch('jpmapper.io.raster.HAS_PSUTIL', False)
    def test_optimize_gdal_cache_fallback_no_psutil(self):
        """Test GDAL cache fallback when psutil is not available."""
        # Store original value
        original_cache = os.environ.get("GDAL_CACHEMAX")
        
        try:
            # Clear existing cache setting
            if "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
            
            _optimize_gdal_cache()
            
            # Should fall back to 1GB default
            cache_size = os.environ.get("GDAL_CACHEMAX", "0")
            assert cache_size == "1024"
            
        finally:
            # Restore original value
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
    
    def test_performance_optimization_integration(self):
        """Test that performance optimizations work together."""
        # This is an integration test that verifies the components work together
        
        # Get optimized worker counts
        raster_workers = _get_optimal_workers(None)
        analysis_workers = _get_optimal_analysis_workers(None)
        
        # Both should be positive
        assert raster_workers > 0
        assert analysis_workers > 0
        
        # Analysis should generally use more workers than rasterization
        cpu_count = multiprocessing.cpu_count()
        if cpu_count > 2:
            assert analysis_workers >= raster_workers
        
        # Test GDAL cache optimization
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
        from jpmapper.io.raster import rasterize_dir_parallel
        
        # Test that the function accepts workers=None and handles it properly
        # We can't easily test the full function without real files, but we can
        # test that the worker optimization logic is integrated
        
        # Test that worker optimization functions work with various inputs
        auto_workers = _get_optimal_workers(None)
        explicit_workers = _get_optimal_workers(4)
        
        assert auto_workers >= 1
        assert explicit_workers == 4
        
        # Test that the functions can handle the types of inputs they'll receive
        # from the rasterization functions
        test_inputs = [None, 1, 2, 4, 8]
        for input_workers in test_inputs:
            result = _get_optimal_workers(input_workers)
            assert result >= 1
            if input_workers is not None and input_workers > 0:
                assert result >= min(input_workers, 1)
    
    def test_worker_parameter_validation(self):
        """Test that worker parameters are properly validated."""
        # Test that functions handle edge cases properly
        
        # Test zero and negative workers
        assert _get_optimal_workers(0) == 1
        assert _get_optimal_workers(-1) == 1
        assert _get_optimal_workers(-10) == 1
        
        # Test very large worker counts (should be capped to reasonable values)
        large_workers = _get_optimal_workers(1000)
        cpu_count = multiprocessing.cpu_count()
        # Should be capped to 2x CPU count (as per implementation)
        assert large_workers <= cpu_count * 2
        assert large_workers >= 1
    
    @patch('jpmapper.io.raster.log')
    def test_performance_logging(self, mock_log):
        """Test that performance optimizations are logged."""
        # Test that functions log their actions
        _optimize_gdal_cache()
        mock_log.info.assert_called()
        
        # Test worker detection logging
        workers = _get_optimal_workers(None)
        # Should have logged the worker count detection
        assert workers > 0


class TestPerformanceEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @patch('multiprocessing.cpu_count')
    def test_single_core_system(self, mock_cpu_count):
        """Test behavior on single-core systems."""
        mock_cpu_count.return_value = 1
        
        # Should still return 1 worker
        workers = _get_optimal_workers(None)
        assert workers == 1
        
        analysis_workers = _get_optimal_analysis_workers(None)
        assert analysis_workers == 1
    
    @patch('psutil.virtual_memory')
    def test_very_low_memory_system(self, mock_memory):
        """Test behavior with very low memory."""
        # Mock system with very low memory (512MB available)
        mock_memory.return_value.available = 512 * 1024**2  # 512MB
        
        workers = _get_optimal_workers(None)
        # Should default to 1 worker when memory is very limited
        assert workers == 1
    
    def test_worker_bounds_checking(self):
        """Test that worker counts are properly bounded."""
        # Test various edge cases
        test_cases = [0, -1, -100, 1, 2, 4, 8, 16, 32, 64, 128, 1000]
        
        for worker_count in test_cases:
            result = _get_optimal_workers(worker_count)
            assert result >= 1, f"Worker count {worker_count} resulted in {result} < 1"
            
            # Should be reasonable (not more than 2x CPU count for explicit values)
            cpu_count = multiprocessing.cpu_count()
            if worker_count > 0:
                assert result <= max(worker_count, cpu_count * 2), \
                    f"Worker count {worker_count} resulted in excessive {result}"
    
    def test_environment_variable_preservation(self):
        """Test that GDAL environment variables are handled properly."""
        # Store original values
        original_cache = os.environ.get("GDAL_CACHEMAX")
        
        try:
            # Set a custom value
            os.environ["GDAL_CACHEMAX"] = "2048"
            
            # Function should still work
            _optimize_gdal_cache()
            
            # Should have updated the value
            new_cache = os.environ.get("GDAL_CACHEMAX")
            assert new_cache is not None
            assert int(new_cache) > 0
            
        finally:
            # Restore original value
            if original_cache is not None:
                os.environ["GDAL_CACHEMAX"] = original_cache
            elif "GDAL_CACHEMAX" in os.environ:
                del os.environ["GDAL_CACHEMAX"]
