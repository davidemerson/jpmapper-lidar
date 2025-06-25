# JPMapper-LiDAR

A high-performance Python toolkit for LiDAR data processing and wireless link analysis. JPMapper specializes in filtering LAS/LAZ files, generating Digital Surface Models (DSMs), and performing line-of-sight analysis for wireless communication planning.

## Key Features

✅ **High-Performance Processing**: Automatic resource detection and parallel processing  
✅ **LiDAR Data Filtering**: Filter LAS/LAZ files by geographic bounding boxes  
✅ **DSM Generation**: Create Digital Surface Models from first-return LiDAR data  
✅ **Line-of-Sight Analysis**: Analyze wireless link clearance with Fresnel zone calculations  
✅ **CLI & API**: Both command-line tools and Python API for integration  
✅ **Automatic Optimization**: Memory-aware scaling and intelligent worker management  
✅ **Comprehensive Testing**: 80+ tests ensuring reliability and performance

## Quick Start

For immediate use with conda:

```bash
# Clone and setup
git clone https://github.com/davidemerson/jpmapper-lidar.git
cd jpmapper-lidar
conda create -n jpmapper python=3.11
conda activate jpmapper
conda install -c conda-forge pdal python-pdal rasterio laspy shapely pyproj rich typer matplotlib pandas folium psutil
pip install -e .

# Test installation
jpmapper --help

# Basic usage
jpmapper analyze csv your_points.csv --las-dir path/to/las/files --json-out results.json
```

See [Installation](#installation-options) for detailed setup instructions.

## Building Your Environment

This section explains how to set up your development environment for working with JPMapper.

### Prerequisites

- Python 3.9 or newer
- Git (for cloning the repository)
- C/C++ compiler for some dependencies (included with Visual Studio on Windows)

### System Requirements

**Minimum Requirements:**
- 4GB RAM
- 2 CPU cores
- 1GB free disk space

**Recommended for Large Datasets:**
- 16GB+ RAM (allows larger GDAL cache and more parallel workers)
- 8+ CPU cores (enables efficient parallel processing)
- SSD storage (significantly improves I/O performance)
- 10GB+ free disk space (for temporary files and caching)

**Performance Notes:**
- JPMapper automatically scales to available resources
- Memory usage is approximately 2GB per worker for rasterization tasks  
- GDAL cache is automatically set to 25% of available RAM (capped at 4GB)
- CPU usage scales to 75% of cores for rasterization, 90% for analysis tasks
- All performance optimizations are transparent and require no manual configuration

### Installation Options

#### Option 1: Using Conda (Recommended)

The recommended way to set up JPMapper is using Conda, which manages dependencies effectively, especially for packages with complex C/C++ dependencies.

1. **Install Miniconda**:
   - Download from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. **Create a new environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/davidemerson/jpmapper-lidar.git
   cd jpmapper-lidar
     # Create and activate a new conda environment
   conda update conda
   conda config --add channels conda-forge
   conda create -n jpmapper python=3.11
   conda activate jpmapper
   
   # Install core dependencies from conda-forge
   conda install -c conda-forge pdal python-pdal rasterio laspy shapely pyproj rich typer matplotlib pandas folium psutil
   
   # Install development dependencies
   conda install -c conda-forge pytest pytest-cov
   pip install ruff mypy pre-commit
   
   # Install JPMapper in development mode
   pip install -e .
   ```

3. **Verify installation**:
   ```bash
   # Verify command-line interface works
   jpmapper --help
   
   # Run tests
   pytest -q
   ```

#### Option 2: Using pip

If you prefer using pip, you can install JPMapper with the following steps. Note that some dependencies may require additional system libraries.

1. **Create a virtual environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/davidemerson/jpmapper-lidar.git
   cd jpmapper-lidar
   
   # Create and activate a virtual environment
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   # Install all dependencies
   pip install -r requirements.txt
   
   # For development, also install dev dependencies
   pip install -r requirements-dev.txt
   
   # Install JPMapper in development mode
   pip install -e .
   ```

### Dependency Overview

JPMapper depends on the following key packages:

- **PDAL & python-pdal**: Point Data Abstraction Library for processing point cloud data
- **rasterio**: Geospatial raster data processing
- **laspy**: Reading and writing LAS/LAZ LiDAR files
- **shapely**: Manipulation and analysis of geometric objects
- **pyproj**: Cartographic projections and coordinate transformations
- **rich**: Terminal formatting and display
- **typer**: Building command-line interfaces
- **numpy**: Numerical operations
- **pandas**: Data analysis and manipulation
- **matplotlib**: Visualization and plotting (required for analysis and CLI)
- **folium**: Interactive map creation (optional, for HTML map output)

### Development Tools

For development, JPMapper uses:

- **pytest**: Testing framework
- **pytest-cov**: Test coverage reporting
- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for code quality checks

### Benchmarking Dependencies

The benchmarking module has additional dependencies:

```bash
# For basic benchmarking functionality
pip install matplotlib pandas tabulate

# For memory profiling
pip install psutil

# For real data benchmarks (if working with actual data files)
pip install rasterio laspy
```

When using conda, you can install these with:

```bash
# Basic benchmarking
conda install -c conda-forge matplotlib pandas tabulate

# Memory profiling
conda install -c conda-forge psutil

# Real data benchmarks
conda install -c conda-forge rasterio laspy
```

The benchmarking tools are designed to work even if some dependencies are missing, but with reduced functionality.

## Using the Command Line Interface

JPMapper provides a comprehensive command-line interface for filtering, rasterizing, and analyzing LiDAR data:

### Basic Commands

```bash
# Filter LAS/LAZ files by bounding box
jpmapper filter bbox path/to/las/files --bbox "-74.01 40.70 -73.96 40.75" --dst path/to/output

# Rasterize a single LAS/LAZ file to a GeoTIFF DSM
jpmapper rasterize tile input.las output.tif --epsg 6539 --resolution 0.1 --workers auto

# Analyze point-to-point links from a CSV file
jpmapper analyze csv points.csv --las-dir path/to/las/files --json-out results.json --workers auto
```

### Advanced Usage

```bash
# Analyze with custom parameters and caching
jpmapper analyze csv points.csv \
  --las-dir path/to/las/files \
  --cache dsm_cache.tif \
  --epsg 6539 \
  --resolution 0.1 \
  --max-mast-height-m 10 \
  --mast-height-step-m 2 \
  --json-out results.json \
  --map-html interactive_map.html \
  --workers 8

# Rasterize with auto-detected CRS and custom resolution
jpmapper rasterize tile large_file.las output.tif --resolution 0.05

# Filter with specific bounding box (format: min_x min_y max_x max_y)
jpmapper filter bbox "data/las/*.las" \
  --bbox "-74.01 40.70 -73.96 40.75" \
  --dst filtered_output/
```

### Performance Options

All commands support automatic performance optimization:

- `--workers auto` (default): Auto-detects optimal number of workers
- `--workers N`: Use N specific worker processes  
- When `--workers` is omitted, JPMapper automatically optimizes for your system

### CSV File Format for Analysis

When using the `analyze csv` command, your CSV file should have the following columns:

```csv
id,point_a_lat,point_a_lon,point_b_lat,point_b_lon
link_1,40.7128,-74.0060,40.7614,-73.9776
link_2,40.7589,-73.9851,40.7831,-73.9712
```

Required columns:
- `point_a_lat`, `point_a_lon`: Latitude and longitude of the first point
- `point_b_lat`, `point_b_lon`: Latitude and longitude of the second point

Optional columns:
- `id`: Identifier for the link (auto-generated if missing)
- Any other columns will be preserved in the output

### Output Formats

JPMapper supports multiple output formats for analysis results:

**JSON Output** (`--json-out results.json`):
```json
[
  {
    "id": "link_1",
    "point_a": [40.7128, -74.0060],
    "point_b": [40.7614, -73.9776],
    "clear": false,
    "mast_height_m": 3,
    "distance_m": 5420.2,
    "surface_height_a_m": 15.5,
    "surface_height_b_m": 42.1,
    "clearance_min_m": -2.3,
    "freq_ghz": 5.8
  }
]
```

**Interactive Map** (`--map-html map.html`):
- Requires `folium` package
- Creates an interactive HTML map showing link paths
- Color-coded by line-of-sight status (clear/blocked)
- Click on links to see detailed analysis results

## Using the API

JPMapper provides a comprehensive programmatic API for use in Python scripts:

### Basic API Usage

```python
from pathlib import Path
from jpmapper.api import filter_by_bbox, rasterize_tile, analyze_los

# Filter LAS files by bounding box
las_files = list(Path("data/las").glob("*.las"))
bbox = (-74.01, 40.70, -73.96, 40.75)  # min_x, min_y, max_x, max_y
filtered = filter_by_bbox(las_files, bbox=bbox)

# Rasterize a LAS file to a GeoTIFF DSM with auto-optimized workers
rasterize_tile(
    Path("data/las/tile1.las"),
    Path("data/dsm/tile1.tif"),
    epsg=6539,
    resolution=0.1
)

# Analyze line-of-sight between two points
point_a = (40.7128, -74.0060)  # NYC (latitude, longitude)
point_b = (40.7614, -73.9776)  # Times Square
result = analyze_los(
    Path("data/dsm.tif"),
    point_a,
    point_b,
    freq_ghz=5.8,
    max_mast_height_m=5
)

# Check if path is clear
if result["clear"]:
    print("Path is clear!")
else:
    print(f"Path is blocked. Minimum mast height required: {result['mast_height_m']} m")
```

### Advanced API Usage with Performance Optimization

```python
from jpmapper.api import rasterize_directory, cached_mosaic

# Process multiple LAS files in parallel with auto-optimized workers
tiff_files = rasterize_directory(
    Path("data/las/"),
    Path("data/tiffs/"),
    epsg=6539,
    resolution=0.1,
    workers=None  # Auto-detect optimal workers
)

# Create cached mosaic with performance optimization
mosaic_path = cached_mosaic(
    Path("data/las/"),
    Path("data/cached_mosaic.tif"),
    epsg=6539,
    resolution=0.1,
    workers=None,  # Auto-optimized
    force=False    # Use cache if available
)

# Analyze multiple points with performance tuning
analysis_results = []
for point_pair in point_pairs:
    result = analyze_los(
        mosaic_path,
        point_pair[0],
        point_pair[1],
        freq_ghz=5.8,
        max_mast_height_m=10,
        n_samples=512  # Higher resolution for detailed analysis
    )
    analysis_results.append(result)
```
```

## Performance Optimizations

JPMapper is designed to maximize performance on any given machine by automatically utilizing available system resources. The application implements several key optimizations:

### Automatic Resource Detection

JPMapper automatically detects and optimizes for your system's capabilities:

- **CPU Cores**: Auto-detects available CPU cores and uses up to 75% for rasterization tasks and 90% for analysis tasks
- **Memory**: Monitors available RAM and adjusts worker processes to prevent memory exhaustion
- **GDAL Cache**: Dynamically sets GDAL cache size to 25% of available memory (capped at 4GB) for optimal raster processing

### Parallel Processing

The application uses parallel processing throughout:

1. **LAS Rasterization**: Multiple LAS files are processed simultaneously using `ProcessPoolExecutor`
2. **CSV Analysis**: Point-to-point analysis of multiple rows is parallelized when processing large datasets
3. **Memory-Aware Scaling**: Worker processes are limited based on available memory (estimated 2GB per worker for LiDAR processing)

### Performance Configuration

You can override automatic detection by specifying the `--workers` parameter:

```bash
# Use specific number of workers
jpmapper analyze csv points.csv --las-dir path/to/las/files --workers 8

# Let JPMapper auto-detect optimal workers (recommended)
jpmapper analyze csv points.csv --las-dir path/to/las/files
```

### Memory Optimization

- **GDAL Cache**: Automatically optimized based on available system memory
- **Chunked Processing**: Large datasets are processed in memory-efficient chunks
- **Resource Monitoring**: Uses `psutil` to monitor system resources and adjust processing accordingly

### Benchmarking Performance

Use the built-in benchmarking tools to measure performance on your system:

```bash
# Benchmark with real data to see actual performance
python -m benchmarks.real_data_benchmarks --las-dir path/to/las/files

# Benchmark API functions with synthetic data
python -m benchmarks.api_benchmarks --iterations 5 --output-dir results/

# Memory profiling for performance analysis
python -c "
from benchmarks.memory_profiler import MemoryProfiler
profiler = MemoryProfiler()
# Profile your specific workload
results = profiler.profile(your_function, *args)
print(f'Peak memory: {results[\"peak_usage_mb\"]:.1f} MB')
"

# Compare performance between different runs
python -m benchmarks.compare_benchmarks old_results.json new_results.json
```

### Performance Monitoring

JPMapper automatically logs performance information:

```bash
# Enable verbose logging to see performance details
export JPMAPPER_LOG_LEVEL=INFO
jpmapper analyze csv points.csv --las-dir data/las/

# Example output:
# INFO: Auto-detected 6 workers for rasterization (CPU cores: 8, Available memory: 15.2GB)
# INFO: Processing 150 LAS files with 6 workers
# INFO: Auto-detected 7 workers for analysis (CPU cores: 8)
# INFO: Using 7 workers for parallel analysis
# INFO: Set GDAL cache to 3840MB
```

### Performance Tips

1. **SSD Storage**: Store LAS files and output on SSD for faster I/O
2. **Memory**: More RAM allows for larger GDAL cache and more parallel workers
3. **CPU**: Multi-core processors significantly improve processing time for large datasets
4. **Temporary Storage**: Ensure adequate space in temp directory for intermediate files

The performance optimizations are transparent to users - JPMapper will automatically use available resources efficiently without requiring manual configuration.

## Height Calculations and LiDAR Processing

### Digital Surface Model vs. Digital Terrain Model

JPMapper processes LiDAR data as a **Digital Surface Model (DSM)** rather than a Digital Terrain Model (DTM). This is an important distinction:

- **Digital Surface Model (DSM)**: Includes heights of all objects on the terrain surface, including buildings, vegetation, and other structures.
- **Digital Terrain Model (DTM)**: Represents only the bare ground surface without any objects.

### First Return Processing

JPMapper specifically uses **first return** LiDAR data to create DSMs, which ensures:

1. **Building and structure heights are included**: First returns capture the highest points of objects, including rooftops, trees, and other structures.
2. **Maximum elevation per pixel**: The rasterization process uses the maximum Z value in each pixel, preserving tall features.
3. **Realistic obstruction modeling**: Line-of-sight calculations accurately account for all potential obstructions between points.

### How Mast Heights Work

When analyzing line-of-sight between two points, mast heights are added on top of the DSM height (which already includes buildings and structures):

1. The DSM provides the base surface elevation at each point, including any buildings/structures
2. Mast heights are added to these surface elevations (not to bare ground)
3. Line-of-sight is calculated between the resulting elevated points
4. The program can determine the minimum mast height needed for a clear path

This approach ensures that the analysis accurately represents real-world conditions where antennas would be mounted on top of existing structures.

### PDAL Pipeline Details

JPMapper uses PDAL with specific configurations to generate appropriate DSMs:

```
- Uses "output_type": "max" to keep the tallest return per pixel
- Excludes only Classification 7 (noise) while keeping all other points
- Preserves building heights, vegetation, and other structures
```

## Error Handling

JPMapper provides a comprehensive set of exception classes to make error handling more precise:

```python
from jpmapper.api import analyze_los
from jpmapper.exceptions import (
    JPMapperError,  # Base exception class
    GeometryError,  # For coordinate/geometry issues
    AnalysisError,  # For analysis operation failures
    RasterizationError,  # For rasterization failures
    FilterError,  # For filtering failures
    NoDataError,  # For missing data issues
)

try:
    result = analyze_los(
        Path("data/dsm.tif"),
        (40.7128, -74.0060),
        (40.7614, -73.9776),
        freq_ghz=5.8
    )
except GeometryError as e:
    print(f"Invalid coordinates: {e}")
except AnalysisError as e:
    print(f"Analysis failed: {e}")
except FileNotFoundError as e:
    print(f"DSM file not found: {e}")
except JPMapperError as e:
    print(f"Other JPMapper error: {e}")
```

See the `examples` directory for more complete examples of API usage.

## Testing

JPMapper includes a comprehensive test suite to ensure code quality and reliability. The tests are organized into several categories:

### Running Tests

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_api.py
pytest tests/test_exceptions.py

# Run with coverage report
pytest --cov=jpmapper

# Run only tests marked as integration tests
pytest -m integration

# Run all tests except integration tests
pytest -m "not integration"
```

### Test Categories

#### Unit Tests

Unit tests focus on testing individual components in isolation:

- **API Tests** (`test_api.py`): Tests the public API functions
- **Exception Tests** (`test_exceptions.py`): Tests the exception hierarchy
- **I/O Tests** (`test_las_io.py`, `test_raster_io.py`): Tests file I/O operations
- **Analysis Tests** (`test_analysis.py`): Tests the analysis functions
- **Configuration Tests** (`test_config.py`): Tests the configuration system
- **Logging Tests** (`test_logging.py`): Tests the logging system

#### Integration Tests

Integration tests verify that different components work together correctly:

- **API Comprehensive Tests** (`test_api_comprehensive.py`): Tests API functions with more complex scenarios
- **Workflow Tests** (`test_integration_workflows.py`): Tests workflows that combine multiple components
- **End-to-End Tests** (`test_end_to_end.py`): Tests the entire pipeline from filtering to analysis
- **CLI Tests** (`test_cli.py`): Tests the command-line interface

### Test Data Requirements

To run the full test suite, including integration tests, you need to provide the following data:

1. **LAS/LAZ Files**: 
   - Create a directory at `tests/data/las/` if it doesn't exist
   - Add one or more LAS/LAZ files to this directory
   - Files should be small (preferably < 5MB) to keep the repository size manageable
   - The LAS files do not need to cover the exact areas in points.csv - the tests are designed to be flexible
   - The integration tests will automatically use coordinates from within your LAS file's coverage area
   - Files should have varying elevation data for proper line-of-sight testing

2. **Points CSV File**:
   - The repository includes a `tests/data/points.csv` file with test points
   - This file contains pairs of coordinates for line-of-sight testing
   - The default points are in the New York City area (around latitude 40.7, longitude -73.9)
   - Each entry includes two points (A and B), a frequency value, and expected line-of-sight result

3. **Sources for LAS/LAZ Files**:
   - Public repositories such as:
     - [USGS 3DEP LiDAR Explorer](https://apps.nationalmap.gov/lidar-explorer/)
     - [OpenTopography](https://opentopography.org/)
     - [NOAA Digital Coast](https://coast.noaa.gov/dataviewer/)
   - Sample data from LiDAR software providers
   - Converted elevation data using tools like PDAL or LAStools

The test suite is designed to skip tests that require data files when they are not available. This allows basic unit tests to run without any test data, while integration tests will run when appropriate data is provided.

Important notes about test data:
- For end-to-end tests, the system will extract test coordinates from within your LAS file's coverage area
- When testing with points.csv, the test is informational only and doesn't strictly validate that results match the expected values
- If test points from points.csv fall outside your LAS file coverage, those specific test cases will be skipped

### Test Data Structure

The `tests/data` directory should have this structure:
```
tests/data/
├── las/              # Directory for LAS/LAZ files
│   └── (your LAS files here)
├── points.csv        # Test points for line-of-sight analysis
└── README.md         # Information about test data requirements
```

### Test Mocking

For tests that would require real data files, JPMapper uses mocking to simulate file operations:

```python
@patch('laspy.open')
def test_function_with_mock(mock_laspy_open):
    mock_file = MagicMock()
    mock_context = MagicMock()
    mock_context.__enter__.return_value = mock_file
    mock_laspy_open.return_value = mock_context
    
    # Test code that uses laspy.open
```

### Writing New Tests

When adding new functionality to JPMapper, consider adding tests in these categories:

1. **Unit tests** for new functions, classes, or methods
2. **API tests** for new API functions
3. **Exception tests** for error handling
4. **Integration tests** for workflows that combine multiple components

Example of adding a test for a new API function:

```python
def test_new_function():
    """Test that new_function behaves as expected."""
    result = new_function(arg1, arg2)
    assert result == expected_result
    
    # Test error cases
    with pytest.raises(ValueError):
        new_function(invalid_arg)
```

### Code Coverage

To check code coverage of the test suite:

```bash
# Run tests with coverage report
pytest --cov=jpmapper

# Generate HTML coverage report
pytest --cov=jpmapper --cov-report=html
```

Then open `htmlcov/index.html` in a web browser to view the coverage report.

## Benchmarking

JPMapper includes a comprehensive benchmarking system to measure and track performance of key operations. This is essential for optimizing performance and tracking improvements over time.

### Running Benchmarks

#### API Benchmarks

To benchmark the core API functions with mock data:

```bash
# Run API benchmarks with default settings
python -m benchmarks.api_benchmarks

# Specify number of iterations and output directory
python -m benchmarks.api_benchmarks --iterations 10 --output-dir results/api_benchmarks
```

#### Real-Data Benchmarks

For more realistic performance measurements with actual LAS/LAZ files and DSMs:

```bash
# Run benchmarks with real data
python -m benchmarks.real_data_benchmarks --las-dir path/to/las/files --dsm-dir path/to/dsm/files

# Specify number of iterations and output directory
python -m benchmarks.real_data_benchmarks --las-dir path/to/las/files --iterations 5 --output-dir results/benchmarks
```

#### Comparing Benchmark Results

To compare results from multiple benchmark runs and track performance changes:

```bash
# Compare two or more benchmark result files
python -m benchmarks.compare_benchmarks results/api_benchmarks_old.json results/api_benchmarks_new.json

# Add custom labels and save the comparison plot
python -m benchmarks.compare_benchmarks results/benchmark1.json results/benchmark2.json \
  --labels "Before Optimization" "After Optimization" --output comparison.png
```

### Benchmark Structure

JPMapper's benchmarking system is organized as follows:

- **Core Utilities** (`benchmarks.core`): Classes and functions for running benchmarks, measuring times, and analyzing results
- **API Benchmarks** (`benchmarks.api_benchmarks`): Benchmarks for API functions using synthetic data
- **Real-Data Benchmarks** (`benchmarks.real_data_benchmarks`): Benchmarks using actual LAS/LAZ and DSM files
- **Comparison Tools** (`benchmarks.compare_benchmarks`): Tools for comparing results from different benchmark runs

### Creating Custom Benchmarks

You can create custom benchmarks by using the provided classes:

```python
from benchmarks.core import Benchmark, BenchmarkSuite, print_results

# Create a benchmark for your function
benchmark = Benchmark(
    name="my_custom_function",
    func=my_function,
    args=[arg1, arg2],
    kwargs={"param": value},
    iterations=10,
    warmup=1,
    metadata={"description": "My custom benchmark"}
)

# Create a suite and run benchmarks
suite = BenchmarkSuite("My Custom Benchmarks")
suite.add_benchmark(benchmark)
results = suite.run()

# Print and save results
print_results(results)
suite.save_results("my_benchmark_results.json")
```

### Memory Profiling

In addition to timing benchmarks, JPMapper includes tools for memory profiling to help identify memory leaks and optimize memory usage:

```python
from benchmarks.memory_profiler import profile_memory, MemoryProfiler

# Method 1: Use the decorator
@profile_memory
def my_function():
    # Function code here
    pass

# Method 2: Use the MemoryProfiler class
profiler = MemoryProfiler()
results = profiler.profile(my_function, *args, **kwargs)

# Plot memory usage over time
profiler.plot(output_path="memory_profile.png", title="Memory Usage")

# Print results
print(f"Peak memory usage: {results['peak_usage_mb']:.2f} MB")
```

Memory profiling is especially important for functions that process large LiDAR datasets to ensure efficient resource usage.

## Common Issues and Troubleshooting

### Missing Dependencies

If you encounter errors like `ModuleNotFoundError: No module named 'matplotlib'` when running the CLI or benchmarks, you need to install the missing dependency:

```bash
# Using conda
conda install -c conda-forge matplotlib

# Using pip
pip install matplotlib
```

### CLI Command Not Found

If the `jpmapper` command is not found after installation, make sure:

1. You have activated the correct conda environment or virtual environment
2. You have installed the package in development mode with `pip install -e .`
3. Your PATH environment variable includes the directory where pip installs executables

### Import Errors

If you see import errors when running the CLI or API:

1. Ensure all required dependencies are installed
2. Check that you're using the correct Python environment
3. Verify that you've installed JPMapper in development mode with `pip install -e .`

### Test Failures

If you encounter test failures with messages like `ImportError: cannot import name 'xxx' from 'jpmapper.module'`:

1. Make sure you have the latest version of the codebase
2. Check that all the required modules and functions are present in the source code
3. Run the specific failing test with verbose output to get more details:
   ```bash
   pytest tests/test_specific.py -v
   ```

### Performance Issues

If you experience slow performance with large LiDAR datasets:

1. **Check available resources**: JPMapper auto-detects but you can monitor with:
   ```bash
   python -c "
   from jpmapper.io.raster import _get_optimal_workers
   from jpmapper.cli.analyze_utils import _get_optimal_analysis_workers
   print(f'Rasterization workers: {_get_optimal_workers()}')
   print(f'Analysis workers: {_get_optimal_analysis_workers()}')
   "
   ```

2. **Memory optimization**: JPMapper automatically sets GDAL cache, but you can check:
   ```bash
   python -c "
   from jpmapper.io.raster import _optimize_gdal_cache
   import os
   _optimize_gdal_cache()
   print(f'GDAL cache: {os.environ.get(\"GDAL_CACHEMAX\")} MB')
   "
   ```

3. **Storage optimization**: 
   - Use SSD storage for LAS files and output
   - Ensure adequate temp space for intermediate files
   - Consider using `--cache` option to reuse DSM processing

4. **Parallel processing**: 
   - Large datasets benefit from more CPU cores
   - Memory-intensive operations are automatically limited by available RAM
   - Use `--workers 1` to disable parallelization for debugging

5. **File optimization**:
   - Compress LAS files to LAZ format to reduce I/O time
   - Use appropriate resolution (smaller = slower but more detailed)
   - Consider pre-filtering large datasets by geographic area

6. **Monitor system resources**: Use benchmarking tools to identify bottlenecks:
   ```bash
   python -m benchmarks.real_data_benchmarks --las-dir path/to/data --iterations 1
   ```

### Additional Help

If you encounter any issues not covered in this troubleshooting guide:

1. **Check the test suite**: Run `pytest -v` to ensure all components are working correctly
2. **Review the logs**: Enable verbose logging with `export JPMAPPER_LOG_LEVEL=DEBUG` 
3. **Performance validation**: Use the built-in performance tests to verify optimization features:
   ```bash
   pytest tests/test_performance.py -v
   ```

JPMapper is designed to be robust and self-optimizing. Most performance and compatibility issues are automatically handled by the built-in optimization systems.