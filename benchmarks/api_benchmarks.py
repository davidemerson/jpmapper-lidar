"""
API benchmarks for JPMapper.

This script benchmarks the performance of key JPMapper API functions.
"""

import os
import tempfile
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Check for required dependencies
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Some benchmarks will be disabled.")
    print("Install with: pip install rasterio")

from jpmapper.api import (
    filter_by_bbox,
    rasterize_tile,
    rasterize_directory,
    merge_tiles,
    analyze_los,
    generate_profile
)
from benchmarks.core import (
    Benchmark,
    BenchmarkSuite,
    print_results,
    plot_results
)


def create_mock_las_file(path: Path, size: int = 1000) -> Path:
    """
    Create a mock LAS file for testing.
    
    Note: This is just a placeholder. In a real benchmark, you would use 
    actual LAS files or create valid ones using the laspy library.
    
    Args:
        path: Path to create the file at
        size: Size of the mock file in bytes
        
    Returns:
        Path to the created file
    """
    with open(path, 'wb') as f:
        f.write(os.urandom(size))
    return path


def create_mock_dsm(path: Path, width: int = 100, height: int = 100) -> Path:
    """
    Create a mock DSM GeoTIFF for testing.
    
    Args:
        path: Path to create the file at
        width: Width of the DSM in pixels
        height: Height of the DSM in pixels
        
    Returns:
        Path to the created file
    """
    # Create a synthetic terrain with hills
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    data = 10 + 30 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
    
    if HAS_RASTERIO:
        # Create a GeoTIFF with a simple transform
        transform = Affine(0.01, 0, 0, 0, -0.01, 1)
        
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=data.dtype,
            crs='+proj=latlong',
            transform=transform,
        ) as dst:
            dst.write(data.astype(np.float32), 1)
    else:
        # Create a dummy file if rasterio is not available
        print("Warning: rasterio not installed, creating dummy DSM file")
        with open(path, 'wb') as f:
            f.write(os.urandom(width * height * 4))  # 4 bytes per float32
    
    return path


def benchmark_filter_by_bbox(las_files: list, bbox: tuple, iterations: int = 10) -> None:
    """
    Benchmark the filter_by_bbox function.
    
    Args:
        las_files: List of LAS file paths
        bbox: Bounding box as (min_x, min_y, max_x, max_y)
        iterations: Number of iterations to run
    """
    def _func():
        return filter_by_bbox(las_files, bbox=bbox)
    
    benchmark = Benchmark(
        name="filter_by_bbox",
        func=_func,
        iterations=iterations,
        warmup=1,
        metadata={
            "num_files": len(las_files),
            "bbox": bbox
        }
    )
    
    return benchmark


def benchmark_analyze_los(dsm_path: Path, point_a: tuple, point_b: tuple, 
                        freq_ghz: float = 5.8, iterations: int = 10) -> None:
    """
    Benchmark the analyze_los function.
    
    Args:
        dsm_path: Path to the DSM GeoTIFF
        point_a: First point as (latitude, longitude)
        point_b: Second point as (latitude, longitude)
        freq_ghz: Frequency in GHz
        iterations: Number of iterations to run
    """
    def _func():
        return analyze_los(dsm_path, point_a, point_b, freq_ghz=freq_ghz)
    
    benchmark = Benchmark(
        name="analyze_los",
        func=_func,
        iterations=iterations,
        warmup=1,
        metadata={
            "point_a": point_a,
            "point_b": point_b,
            "freq_ghz": freq_ghz
        }
    )
    
    return benchmark


def benchmark_generate_profile(dsm_path: Path, point_a: tuple, point_b: tuple, 
                             freq_ghz: float = 5.8, n_samples: int = 256, 
                             iterations: int = 10) -> None:
    """
    Benchmark the generate_profile function.
    
    Args:
        dsm_path: Path to the DSM GeoTIFF
        point_a: First point as (latitude, longitude)
        point_b: Second point as (latitude, longitude)
        freq_ghz: Frequency in GHz
        n_samples: Number of samples along the path
        iterations: Number of iterations to run
    """
    def _func():
        return generate_profile(dsm_path, point_a, point_b, freq_ghz=freq_ghz, n_samples=n_samples)
    
    benchmark = Benchmark(
        name="generate_profile",
        func=_func,
        iterations=iterations,
        warmup=1,
        metadata={
            "point_a": point_a,
            "point_b": point_b,
            "freq_ghz": freq_ghz,
            "n_samples": n_samples
        }
    )
    
    return benchmark


def benchmark_rasterize_tile(las_path: Path, dst_path: Path, epsg: int = 6539, 
                           resolution: float = 0.1, iterations: int = 5) -> None:
    """
    Benchmark the rasterize_tile function.
    
    Args:
        las_path: Path to the LAS file
        dst_path: Path to save the GeoTIFF to
        epsg: EPSG code for the coordinate reference system
        resolution: Resolution in coordinate system units
        iterations: Number of iterations to run
    """
    def _func():
        return rasterize_tile(las_path, dst_path, epsg=epsg, resolution=resolution)
    
    benchmark = Benchmark(
        name="rasterize_tile",
        func=_func,
        iterations=iterations,
        warmup=1,
        metadata={
            "epsg": epsg,
            "resolution": resolution
        }
    )
    
    return benchmark


def run_all_benchmarks(iterations: int = 5, output_dir: Path = None) -> None:
    """
    Run all benchmarks and save results.
    
    Args:
        iterations: Number of iterations for each benchmark
        output_dir: Directory to save results to
    """
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create mock data
        print("Creating mock data...")
        las_files = [create_mock_las_file(tmp_path / f"file{i}.las") for i in range(5)]
        dsm_path = create_mock_dsm(tmp_path / "dsm.tif")
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create benchmarks
        print("Setting up benchmarks...")
        suite = BenchmarkSuite("JPMapper API Benchmarks")
        
        # Add filter benchmark
        suite.add_benchmark(benchmark_filter_by_bbox(
            las_files, 
            bbox=(-74.01, 40.70, -73.96, 40.75),
            iterations=iterations
        ))
        
        # Add analyze_los benchmark
        suite.add_benchmark(benchmark_analyze_los(
            dsm_path,
            point_a=(40.7128, -74.0060),
            point_b=(40.7614, -73.9776),
            freq_ghz=5.8,
            iterations=iterations
        ))
        
        # Add generate_profile benchmark
        suite.add_benchmark(benchmark_generate_profile(
            dsm_path,
            point_a=(40.7128, -74.0060),
            point_b=(40.7614, -73.9776),
            freq_ghz=5.8,
            n_samples=100,
            iterations=iterations
        ))
        
        # Add rasterize_tile benchmark
        suite.add_benchmark(benchmark_rasterize_tile(
            las_files[0],
            tmp_path / "output.tif",
            epsg=6539,
            resolution=0.1,
            iterations=iterations
        ))
        
        # Run benchmarks
        print(f"Running benchmarks ({iterations} iterations each)...")
        results = suite.run(progress_callback=lambda name, i, total: 
                          print(f"Running {name} ({i+1}/{total})..."))
        
        # Print results
        print("\nBenchmark Results:")
        print_results(results)
        
        # Save results if output directory is specified
        if output_dir:
            # Save JSON results
            results_path = output_dir / "benchmark_results.json"
            suite.save_results(results_path)
            print(f"Results saved to {results_path}")
            
            # Create and save plot
            plot_path = output_dir / "benchmark_results.png"
            plot_results(results, plot_path, title="JPMapper API Benchmarks")
            print(f"Plot saved to {plot_path}")


def main():
    """Main function to run benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Run JPMapper API benchmarks")
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations for each benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Directory to save results to")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_all_benchmarks(iterations=args.iterations, output_dir=output_dir)


if __name__ == "__main__":
    main()
