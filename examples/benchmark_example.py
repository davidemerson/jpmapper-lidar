"""
Example script demonstrating JPMapper benchmarking tools.

This script shows how to use the benchmarking and memory profiling tools
to measure and optimize the performance of JPMapper operations.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

# Import JPMapper API
from jpmapper.api import (
    analyze_los,
    generate_profile,
    filter_by_bbox
)

# Import benchmarking tools
from benchmarks.core import (
    Benchmark,
    BenchmarkSuite,
    print_results,
    plot_results
)

# Import memory profiler if available
try:
    from benchmarks.memory_profiler import (
        profile_memory,
        MemoryProfiler
    )
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print("Memory profiler not available. Install psutil to enable memory profiling.")


def create_mock_data():
    """Create mock data for benchmarking."""
    # Create a temporary directory
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    
    # Create mock LAS files
    las_dir = tmp_path / "las"
    las_dir.mkdir(exist_ok=True)
    for i in range(3):
        with open(las_dir / f"file{i}.las", "wb") as f:
            f.write(os.urandom(1024))  # Mock LAS file
    
    # Create a mock DSM file using numpy and matplotlib
    dsm_path = tmp_path / "dsm.tif"
    
    # This is just a placeholder - in a real script you would
    # create a valid GeoTIFF file using rasterio
    with open(dsm_path, "wb") as f:
        f.write(os.urandom(1024))  # Mock DSM file
    
    return {
        "tmp_dir": tmp_dir,  # Keep reference to delete later
        "tmp_path": tmp_path,
        "las_dir": las_dir,
        "las_files": list(las_dir.glob("*.las")),
        "dsm_path": dsm_path
    }


def run_time_benchmarks(mock_data):
    """Run time benchmarks on JPMapper functions."""
    print("\n=== Running Time Benchmarks ===")
    
    # Create a benchmark suite
    suite = BenchmarkSuite("JPMapper Example Benchmarks")
    
    # Mock parameters
    bbox = (-74.01, 40.70, -73.96, 40.75)  # NYC area
    point_a = (40.7128, -74.0060)  # NYC
    point_b = (40.7614, -73.9776)  # Times Square
    
    # Add benchmarks with increasing complexity
    
    # 1. Benchmark filter_by_bbox with small input
    def _filter_small():
        # In a real benchmark, this would actually process files
        # Here we just simulate processing time for demonstration
        time.sleep(0.1)
        return {}
    
    suite.add_benchmark(Benchmark(
        name="filter_by_bbox_small",
        func=_filter_small,
        iterations=5,
        warmup=1,
        metadata={"files": 1, "bbox": bbox}
    ))
    
    # 2. Benchmark filter_by_bbox with larger input
    def _filter_large():
        # Simulate longer processing time
        time.sleep(0.2)
        return {}
    
    suite.add_benchmark(Benchmark(
        name="filter_by_bbox_large",
        func=_filter_large,
        iterations=5,
        warmup=1,
        metadata={"files": 3, "bbox": bbox}
    ))
    
    # 3. Benchmark analyze_los with different parameters
    for freq in [2.4, 5.8, 60]:
        def _analyze(freq=freq):
            # Simulate analyze_los processing
            time.sleep(0.05 * freq / 2.4)  # Higher frequency takes longer
            return {"clear": True, "fresnel_radius": 10.0}
        
        suite.add_benchmark(Benchmark(
            name=f"analyze_los_freq{freq}",
            func=_analyze,
            iterations=5,
            warmup=1,
            metadata={"point_a": point_a, "point_b": point_b, "freq_ghz": freq}
        ))
    
    # Run all benchmarks
    print("Running benchmarks...")
    results = suite.run()
    
    # Print results
    print("\nBenchmark Results:")
    print_results(results)
    
    # Plot results
    plot_path = mock_data["tmp_path"] / "benchmark_results.png"
    plot_results(results, plot_path)
    print(f"Benchmark plot saved to: {plot_path}")
    
    return results


def run_memory_benchmarks(mock_data):
    """Run memory benchmarks on JPMapper functions."""
    if not HAS_MEMORY_PROFILER:
        print("\n=== Memory Profiling Not Available ===")
        print("Install psutil to enable memory profiling.")
        return
    
    print("\n=== Running Memory Benchmarks ===")
    
    # Create a memory profiler
    profiler = MemoryProfiler(interval=0.01)
    
    # Define a function that allocates memory
    def process_large_array():
        """Function that allocates a large array to demonstrate memory profiling."""
        print("Creating large array...")
        # Create a large array
        arr = np.zeros((1000, 1000, 10))
        
        # Simulate processing
        for i in range(10):
            # Add more data to simulate memory growth
            arr = np.concatenate([arr, np.random.random((100, 100, 10))], axis=0)
            time.sleep(0.1)
        
        return arr
    
    # Profile the function
    print("Profiling memory usage...")
    results = profiler.profile(process_large_array)
    
    # Print results
    print("\nMemory Profiling Results:")
    print(f"  Execution time: {results['execution_time']:.4f} seconds")
    print(f"  Min memory: {results['min_memory_mb']:.2f} MB")
    print(f"  Max memory: {results['max_memory_mb']:.2f} MB")
    print(f"  Average memory: {results['avg_memory_mb']:.2f} MB")
    print(f"  Peak usage: {results['peak_usage_mb']:.2f} MB")
    
    # Plot memory usage
    plot_path = mock_data["tmp_path"] / "memory_profile.png"
    profiler.plot(plot_path, title="Memory Usage Example")
    print(f"Memory profile plot saved to: {plot_path}")
    
    # Demonstrate the decorator
    @profile_memory
    def process_multiple_arrays():
        """Function that processes multiple arrays sequentially."""
        arrays = []
        for size in [500, 1000, 1500]:
            print(f"Processing array of size {size}...")
            arr = np.random.random((size, size))
            arrays.append(arr)
            time.sleep(0.2)
        return arrays
    
    # Run the decorated function
    print("\nTesting profile_memory decorator:")
    process_multiple_arrays()


def main():
    """Main function to demonstrate benchmarking tools."""
    print("JPMapper Benchmarking Example")
    print("-----------------------------")
    
    # Create mock data
    print("Creating mock data...")
    mock_data = create_mock_data()
    
    try:
        # Run time benchmarks
        run_time_benchmarks(mock_data)
        
        # Run memory benchmarks
        run_memory_benchmarks(mock_data)
        
    finally:
        # Clean up temporary directory
        mock_data["tmp_dir"].cleanup()
        print("\nCleaned up temporary files.")


if __name__ == "__main__":
    main()
