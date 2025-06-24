"""
Real-data benchmarks for JPMapper.

This script benchmarks the performance of JPMapper with real-world data.
Unlike api_benchmarks.py which uses mock data, this script uses actual
LAS/LAZ files and DSMs to provide realistic performance measurements.
"""

import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple

# Check for required dependencies
try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False
    print("Warning: laspy not installed. Some benchmarks will be disabled.")
    print("Install with: pip install laspy")

try:
    import rasterio
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


def get_las_files(data_dir: Path, pattern: str = "*.la[sz]") -> List[Path]:
    """
    Get all LAS/LAZ files in a directory.
    
    Args:
        data_dir: Directory to search
        pattern: Glob pattern to match files
        
    Returns:
        List of file paths
    """
    return list(data_dir.glob(pattern))


def get_bounding_boxes(las_files: List[Path], sample_size: int = 3) -> List[Tuple[float, float, float, float]]:
    """
    Generate sample bounding boxes from LAS files.
    
    This function extracts actual bounding boxes from LAS files and 
    creates smaller bounding boxes within them for testing.
    
    Args:
        las_files: List of LAS file paths
        sample_size: Number of bounding boxes to generate
        
    Returns:
        List of bounding boxes as (min_x, min_y, max_x, max_y) tuples
    """
    if not HAS_LASPY:
        print("Warning: laspy not installed, using mock bounding boxes")
        # Return some mock bounding boxes
        return [
            (-74.01, 40.70, -73.96, 40.75),  # NYC area
            (-118.50, 34.00, -118.40, 34.10),  # LA area
            (-87.65, 41.85, -87.60, 41.90),  # Chicago area
        ]
    
    bboxes = []
    # Use the first few LAS files to generate bounding boxes
    for file_path in las_files[:min(sample_size, len(las_files))]:
        try:
            with laspy.open(file_path) as f:
                las = f.read()
                # Get the bounds
                min_x, min_y = las.header.mins[:2]
                max_x, max_y = las.header.maxs[:2]
                
                # Generate a smaller bbox within the file bounds (25% of the area)
                width = max_x - min_x
                height = max_y - min_y
                
                # Create a bbox in the center of the file
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                bbox = (
                    center_x - width * 0.25,  # min_x
                    center_y - height * 0.25,  # min_y
                    center_x + width * 0.25,  # max_x
                    center_y + height * 0.25,  # max_y
                )
                bboxes.append(bbox)
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
    
    return bboxes


def get_dsm_files(data_dir: Path, pattern: str = "*.tif") -> List[Path]:
    """
    Get all DSM files in a directory.
    
    Args:
        data_dir: Directory to search
        pattern: Glob pattern to match files
        
    Returns:
        List of file paths
    """
    return list(data_dir.glob(pattern))


def get_points_from_dsm(dsm_path: Path, num_pairs: int = 3) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Generate sample point pairs from a DSM for line-of-sight analysis.
    
    Args:
        dsm_path: Path to the DSM file
        num_pairs: Number of point pairs to generate
        
    Returns:
        List of point pairs as ((lat1, lon1), (lat2, lon2)) tuples
    """
    if not HAS_RASTERIO:
        print("Warning: rasterio not installed, using mock point pairs")
        # Return some mock point pairs
        return [
            ((40.7128, -74.0060), (40.7614, -73.9776)),  # NYC to Times Square
            ((40.7128, -74.0060), (40.7831, -73.9712)),  # NYC to Central Park
            ((40.7831, -73.9712), (40.7614, -73.9776)),  # Central Park to Times Square
        ]
    
    try:
        with rasterio.open(dsm_path) as src:
            # Get the bounds and transform
            bounds = src.bounds
            transform = src.transform
            
            # Generate point pairs
            pairs = []
            for _ in range(num_pairs):
                # Generate random pixel coordinates within the DSM
                row1, col1 = np.random.randint(0, src.height - 1), np.random.randint(0, src.width - 1)
                row2, col2 = np.random.randint(0, src.height - 1), np.random.randint(0, src.width - 1)
                
                # Convert to geo coordinates
                x1, y1 = transform * (col1, row1)
                x2, y2 = transform * (col2, row2)
                
                pairs.append(((y1, x1), (y2, x2)))  # (lat, lon) format
            
            return pairs
    
    except Exception as e:
        print(f"Warning: Could not process {dsm_path}: {e}")
        # Return some dummy points if we can't read the DSM
        return [
            ((40.7128, -74.0060), (40.7614, -73.9776)),  # NYC to Times Square
            ((40.7128, -74.0060), (40.7831, -73.9712)),  # NYC to Central Park
            ((40.7831, -73.9712), (40.7614, -73.9776)),  # Central Park to Times Square
        ]


def benchmark_filter_with_real_data(las_dir: Path, output_dir: Path, iterations: int = 3) -> List[Benchmark]:
    """
    Benchmark filter operations with real data.
    
    Args:
        las_dir: Directory containing LAS files
        output_dir: Directory to save filtered files to
        iterations: Number of iterations to run
        
    Returns:
        List of benchmarks
    """
    benchmarks = []
    
    # Get LAS files and generate bounding boxes
    las_files = get_las_files(las_dir)
    if not las_files:
        print(f"Warning: No LAS files found in {las_dir}")
        return benchmarks
    
    print(f"Found {len(las_files)} LAS/LAZ files")
    bboxes = get_bounding_boxes(las_files)
    
    if not bboxes:
        print("Warning: Could not generate bounding boxes")
        return benchmarks
    
    # Benchmark filter_by_bbox with different numbers of files
    file_counts = [1, min(5, len(las_files)), min(len(las_files), 10)]
    
    for count in file_counts:
        files_subset = las_files[:count]
        for i, bbox in enumerate(bboxes):
            def _filter_func():
                return filter_by_bbox(
                    files_subset, 
                    bbox=bbox, 
                    output_dir=output_dir / f"filter_benchmark_{count}files"
                )
            
            benchmarks.append(Benchmark(
                name=f"filter_by_bbox_{count}files_bbox{i+1}",
                func=_filter_func,
                iterations=iterations,
                warmup=1,
                metadata={
                    "num_files": count,
                    "bbox": bbox
                }
            ))
    
    return benchmarks


def benchmark_rasterize_with_real_data(las_dir: Path, output_dir: Path, iterations: int = 2) -> List[Benchmark]:
    """
    Benchmark rasterization operations with real data.
    
    Args:
        las_dir: Directory containing LAS files
        output_dir: Directory to save rasterized files to
        iterations: Number of iterations to run
        
    Returns:
        List of benchmarks
    """
    benchmarks = []
    
    # Get LAS files
    las_files = get_las_files(las_dir)
    if not las_files:
        print(f"Warning: No LAS files found in {las_dir}")
        return benchmarks
    
    # Benchmark rasterize_tile with different resolutions
    if las_files:
        for resolution in [1.0, 0.5, 0.1]:
            las_path = las_files[0]
            output_path = output_dir / f"raster_benchmark_res{resolution}.tif"
            
            def _rasterize_func():
                return rasterize_tile(
                    las_path, 
                    output_path, 
                    resolution=resolution
                )
            
            benchmarks.append(Benchmark(
                name=f"rasterize_tile_res{resolution}",
                func=_rasterize_func,
                iterations=iterations,
                warmup=1,
                metadata={
                    "resolution": resolution
                }
            ))
    
    # Benchmark rasterize_directory with different file counts
    file_counts = [1, min(3, len(las_files))]
    
    for count in file_counts:
        files_subset_dir = output_dir / f"subset_{count}files"
        files_subset_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files to subset directory
        for i, file_path in enumerate(las_files[:count]):
            import shutil
            dest_path = files_subset_dir / f"file{i}.las"
            try:
                shutil.copy(file_path, dest_path)
            except Exception as e:
                print(f"Warning: Could not copy {file_path} to {dest_path}: {e}")
        
        output_subdir = output_dir / f"raster_benchmark_dir_{count}files"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        def _rasterize_dir_func():
            return rasterize_directory(
                files_subset_dir,
                output_subdir,
                resolution=0.5
            )
        
        benchmarks.append(Benchmark(
            name=f"rasterize_directory_{count}files",
            func=_rasterize_dir_func,
            iterations=max(1, iterations // 2),  # Reduce iterations for directory benchmarks
            warmup=1,
            metadata={
                "num_files": count,
                "resolution": 0.5
            }
        ))
    
    return benchmarks


def benchmark_analysis_with_real_data(dsm_dir: Path, iterations: int = 5) -> List[Benchmark]:
    """
    Benchmark analysis operations with real data.
    
    Args:
        dsm_dir: Directory containing DSM files
        iterations: Number of iterations to run
        
    Returns:
        List of benchmarks
    """
    benchmarks = []
    
    # Get DSM files
    dsm_files = get_dsm_files(dsm_dir)
    if not dsm_files:
        print(f"Warning: No DSM files found in {dsm_dir}")
        return benchmarks
    
    # Get sample points
    point_pairs = get_points_from_dsm(dsm_files[0])
    
    # Benchmark analyze_los with different frequencies
    for i, (point_a, point_b) in enumerate(point_pairs):
        for freq_ghz in [2.4, 5.8, 60]:
            def _analyze_func():
                return analyze_los(
                    dsm_files[0],
                    point_a,
                    point_b,
                    freq_ghz=freq_ghz
                )
            
            benchmarks.append(Benchmark(
                name=f"analyze_los_pair{i+1}_freq{freq_ghz}",
                func=_analyze_func,
                iterations=iterations,
                warmup=1,
                metadata={
                    "point_a": point_a,
                    "point_b": point_b,
                    "freq_ghz": freq_ghz
                }
            ))
    
    # Benchmark generate_profile with different sample counts
    for i, (point_a, point_b) in enumerate(point_pairs):
        for n_samples in [100, 256, 500]:
            def _profile_func():
                return generate_profile(
                    dsm_files[0],
                    point_a,
                    point_b,
                    n_samples=n_samples
                )
            
            benchmarks.append(Benchmark(
                name=f"generate_profile_pair{i+1}_samples{n_samples}",
                func=_profile_func,
                iterations=iterations,
                warmup=1,
                metadata={
                    "point_a": point_a,
                    "point_b": point_b,
                    "n_samples": n_samples
                }
            ))
    
    return benchmarks


def run_real_data_benchmarks(las_dir: Optional[Path] = None, dsm_dir: Optional[Path] = None, 
                            output_dir: Optional[Path] = None, iterations: int = 3) -> None:
    """
    Run benchmarks with real data.
    
    Args:
        las_dir: Directory containing LAS files
        dsm_dir: Directory containing DSM files
        output_dir: Directory to save results to
        iterations: Number of iterations for each benchmark
    """
    # Create a suite for benchmarks
    suite = BenchmarkSuite("JPMapper Real-Data Benchmarks")
    
    # Create temporary directories for outputs if needed
    import tempfile
    import shutil
    
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run filter benchmarks if LAS directory is provided
        if las_dir:
            las_dir = Path(las_dir)
            print(f"Running filter benchmarks with data from {las_dir}...")
            filter_benchmarks = benchmark_filter_with_real_data(las_dir, output_dir, iterations)
            for benchmark in filter_benchmarks:
                suite.add_benchmark(benchmark)
        
            # Run rasterization benchmarks
            print(f"Running rasterization benchmarks with data from {las_dir}...")
            raster_benchmarks = benchmark_rasterize_with_real_data(las_dir, output_dir, iterations)
            for benchmark in raster_benchmarks:
                suite.add_benchmark(benchmark)
        
        # Run analysis benchmarks if DSM directory is provided
        if dsm_dir:
            dsm_dir = Path(dsm_dir)
            print(f"Running analysis benchmarks with data from {dsm_dir}...")
            analysis_benchmarks = benchmark_analysis_with_real_data(dsm_dir, iterations)
            for benchmark in analysis_benchmarks:
                suite.add_benchmark(benchmark)
        
        # Run all benchmarks
        if suite.benchmarks:
            print(f"Running {len(suite.benchmarks)} benchmarks...")
            results = suite.run(progress_callback=lambda name, i, total: 
                              print(f"Running {name} ({i+1}/{total})..."))
            
            # Print results
            print("\nBenchmark Results:")
            print_results(results)
            
            # Save results
            results_path = output_dir / "real_data_benchmark_results.json"
            suite.save_results(results_path)
            print(f"Results saved to {results_path}")
            
            # Create and save plot
            plot_path = output_dir / "real_data_benchmark_results.png"
            plot_results(results, plot_path, title="JPMapper Real-Data Benchmarks")
            print(f"Plot saved to {plot_path}")
        else:
            print("No benchmarks to run. Please provide LAS and/or DSM directories.")
    
    finally:
        # Clean up temporary directory if created
        if temp_dir:
            temp_dir.cleanup()


def main():
    """Main function to run benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Run JPMapper benchmarks with real data")
    parser.add_argument("--las_dir", type=str, help="Directory containing LAS/LAZ files")
    parser.add_argument("--dsm-dir", type=str, help="Directory containing DSM files")
    parser.add_argument("--output-dir", type=str, default="real_data_benchmark_results",
                      help="Directory to save results to")
    parser.add_argument("--iterations", type=int, default=3,
                      help="Number of iterations for each benchmark")
    args = parser.parse_args()
    
    # Validate inputs
    if not args.las_dir and not args.dsm_dir:
        print("Error: Must provide at least one of --las_dir or --dsm-dir")
        return
    
    las_dir = Path(args.las_dir) if args.las_dir else None
    dsm_dir = Path(args.dsm_dir) if args.dsm_dir else None
    output_dir = Path(args.output_dir)
    
    run_real_data_benchmarks(las_dir, dsm_dir, output_dir, args.iterations)


if __name__ == "__main__":
    main()
