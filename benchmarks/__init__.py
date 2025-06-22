"""
JPMapper Benchmarking Package

This package provides tools for benchmarking JPMapper performance.
"""

from benchmarks.core import (
    BenchmarkResult,
    Benchmark,
    BenchmarkSuite,
    print_results,
    plot_results,
    compare_results
)

# Expose memory profiling utilities if available
try:
    from benchmarks.memory_profiler import (
        profile_memory,
        profile_memory_usage,
        MemoryProfiler
    )
    
    # Add memory profiling utilities to __all__
    __all__ = [
        'BenchmarkResult',
        'Benchmark',
        'BenchmarkSuite',
        'print_results',
        'plot_results',
        'compare_results',
        'profile_memory',
        'profile_memory_usage',
        'MemoryProfiler'
    ]
except ImportError:
    # Memory profiling not available
    __all__ = [
        'BenchmarkResult',
        'Benchmark',
        'BenchmarkSuite',
        'print_results',
        'plot_results',
        'compare_results'
    ]
