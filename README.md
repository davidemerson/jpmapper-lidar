# JPMapper-LiDAR

A Python toolkit for LiDAR data filtering, rasterization, and point-to-point link analysis.

## Building Your Environment

This section explains how to set up your development environment for working with JPMapper.

### Prerequisites

- Python 3.9 or newer
- Git (for cloning the repository)
- C/C++ compiler for some dependencies (included with Visual Studio on Windows)

### Installation Options

#### Option 1: Using Conda (Recommended)

The recommended way to set up JPMapper is using Conda, which manages dependencies effectively, especially for packages with complex C/C++ dependencies.

1. **Install Miniconda**:
   - Download from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. **Create a new environment**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/jpmapper-lidar.git
   cd jpmapper-lidar
     # Create and activate a new conda environment
   conda create -n jpmapper python=3.11
   conda activate jpmapper
   
   # Install core dependencies from conda-forge
   conda install -c conda-forge pdal python-pdal rasterio laspy shapely pyproj rich typer matplotlib pandas folium
   
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
   git clone https://github.com/yourusername/jpmapper-lidar.git
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

JPMapper provides a command-line interface for filtering, rasterizing, and analyzing LiDAR data:

```bash
# Filter LAS/LAZ files by bounding box
jpmapper filter bbox path/to/las/files --dst path/to/output

# Rasterize a LAS/LAZ file to a GeoTIFF DSM
jpmapper rasterize tile input.las output.tif --epsg 6539 --resolution 0.1

# Analyze point-to-point links
jpmapper analyze csv points.csv --las-dir path/to/las/files --json-out results.json
```

## Using the API

JPMapper also provides a programmatic API for use in Python scripts:

```python
from pathlib import Path
from jpmapper.api import filter_by_bbox, rasterize_tile, analyze_los

# Filter LAS files by bounding box
las_files = list(Path("data/las").glob("*.las"))
bbox = (-74.01, 40.70, -73.96, 40.75)  # min_x, min_y, max_x, max_y
filtered = filter_by_bbox(las_files, bbox=bbox)

# Rasterize a LAS file to a GeoTIFF DSM
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
    freq_ghz=5.8
)

# Check if path is clear
if result["clear"]:
    print("Path is clear!")
else:
    print(f"Path is blocked. Minimum mast height required: {result['mast_height_m']} m")
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

### Test Data

Some tests require sample LiDAR data. To run these tests:

1. Place sample LAS files in the `tests/data/las` directory
2. Ensure the `points.csv` file in `tests/data` has valid test points

The test suite uses pytest fixtures to automatically skip tests that require data files when those files are not available.

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

1. Make sure you're using the latest versions of key dependencies (rasterio, laspy, etc.)
2. Consider using the memory profiling tools to identify bottlenecks
3. Check the benchmarking results to compare your performance with baseline expectations