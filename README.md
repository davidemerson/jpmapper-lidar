# JPMapper-LiDAR

A Python toolkit for LiDAR data processing and RF line-of-sight analysis. JPMapper filters LAS/LAZ point clouds, generates Digital Surface Models (DSMs), and analyzes wireless link clearance with Fresnel zone calculations.

## Features

- **LiDAR Filtering** -- Select LAS/LAZ tiles by bounding box or shapefile boundary
- **DSM Generation** -- Rasterize first-return point clouds to GeoTIFF with parallel processing
- **Line-of-Sight Analysis** -- Check RF path clearance with mast height optimization
- **Fresnel Zone Profiling** -- Terrain profiles with first Fresnel zone visualization
- **CLI & Python API** -- Full command-line interface and importable Python API
- **Auto-Optimization** -- Memory-aware worker scaling via psutil

## Installation

```bash
# Clone
git clone https://github.com/davidemerson/jpmapper-lidar.git
cd jpmapper-lidar

# Install with conda (recommended for geospatial deps)
conda create -n jpmapper python=3.11
conda activate jpmapper
conda install -c conda-forge pdal python-pdal rasterio laspy shapely pyproj rich typer numpy pandas matplotlib

# Install the package
pip install -e .

# Verify
jpmapper --help
```

### Optional dependencies

```bash
# Shapefile-based filtering
conda install -c conda-forge geopandas fiona

# Performance optimization
pip install psutil

# Interactive maps
pip install folium
```

## CLI Usage

### Filter LAS/LAZ tiles

```bash
# Filter by bounding box
jpmapper filter bbox data/ --bbox '583000 4506000 584000 4507000'
jpmapper filter bbox data/ --bbox '583000 4506000 584000 4507000' --dst filtered/

# Filter by shapefile
jpmapper filter shapefile data/ --shapefile boundary.shp
jpmapper filter shapefile data/ -s boundary.shp --buffer 50 --dst selected/
```

### Rasterize to DSM

```bash
jpmapper rasterize tile input.las output.tif
jpmapper rasterize tile input.las output.tif --epsg 6539 --resolution 0.25
```

### Analyze point-to-point links

```bash
jpmapper analyze csv links.csv --las-dir data/ --epsg 6539
jpmapper analyze csv links.csv --las-dir data/ --json results.json --map map.png
```

The CSV should contain columns: `point_a_lat`, `point_a_lon`, `point_b_lat`, `point_b_lon`, and optionally `point_a_mast`, `point_b_mast`.

## Python API

### Filtering

```python
from pathlib import Path
from jpmapper.api import filter_by_bbox

tiles = list(Path("data/").glob("*.las"))
selected = filter_by_bbox(tiles, bbox=(583000, 4506000, 584000, 4507000))
```

### Rasterization

```python
from jpmapper.api import rasterize_tile

rasterize_tile(
    Path("input.las"),
    Path("output.tif"),
    epsg=6539,
    resolution=0.1,
)
```

### Line-of-Sight Analysis

```python
from jpmapper.api import analyze_los

result = analyze_los(
    Path("dsm.tif"),
    point_a=(40.7128, -74.0060),
    point_b=(40.7614, -73.9776),
    freq_ghz=5.8,
    max_mast_height_m=30,
)
print(f"Clear: {result['clear']}, Mast needed: {result['mast_height_m']}m")
```

### Terrain Profile

```python
from jpmapper.api import generate_profile

distances, terrain, fresnel = generate_profile(
    Path("dsm.tif"),
    point_a=(40.7128, -74.0060),
    point_b=(40.7614, -73.9776),
    n_samples=256,
)
```

## Architecture

```
jpmapper/
  io/            # File I/O layer
    las.py       # LAS/LAZ reading, header parsing, bbox filtering
    raster.py    # PDAL rasterization, tile merging, DSM caching
  analysis/      # Core algorithms
    los.py       # Line-of-sight, Fresnel zone, terrain profiling
    plots.py     # Matplotlib profile visualizations
  api/           # Public API (thin wrappers with validation)
    filter.py    # filter_by_bbox()
    raster.py    # rasterize_tile()
    analysis.py  # analyze_los(), generate_profile()
  cli/           # Typer CLI commands
    filter.py    # jpmapper filter bbox|shapefile
    rasterize.py # jpmapper rasterize tile
    analyze.py   # jpmapper analyze csv
  config.py      # Configuration loading
  exceptions.py  # Exception hierarchy
  logging.py     # Logging setup
```

**Data flow**: LAS files -> IO layer (read/filter/rasterize) -> Analysis (LOS/profile) -> API (validation) -> CLI (user interface)

## Exception Hierarchy

```
JPMapperError
  +-- ConfigurationError
  +-- FileFormatError
  +-- GeoSpatialError
  |     +-- GeometryError
  |     +-- CRSError
  |     +-- NoDataError
  +-- AnalysisError
  |     +-- LOSError
  +-- RasterizationError
  +-- FilterError
```

## Testing

```bash
pip install -e ".[dev]"
pytest                    # Run all tests
pytest tests/test_las_io.py -v  # Run specific module
```

Tests use real temporary GeoTIFF fixtures for LOS/analysis testing. Some integration tests require `pdal` to be installed.

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical operations |
| rasterio | GeoTIFF I/O |
| laspy | LAS/LAZ file reading |
| shapely | Geometric operations |
| pyproj | CRS transformations |
| pdal / python-pdal | Point cloud rasterization |
| rich | Terminal formatting |
| typer | CLI framework |
| pandas | CSV analysis |
| matplotlib | Profile plots |

## License

See [LICENSE](LICENSE).
