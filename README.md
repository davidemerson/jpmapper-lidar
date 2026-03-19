# JPMapper-LiDAR

A Python toolkit for LiDAR data processing and RF line-of-sight analysis. JPMapper filters LAS/LAZ point clouds, generates Digital Surface Models (DSMs), and analyzes wireless link clearance with Fresnel zone calculations.

## Features

- **LiDAR Filtering** -- Select LAS/LAZ tiles by bounding box or shapefile boundary
- **DSM Generation** -- Rasterize first-return point clouds to GeoTIFF with automatic nodata gap-filling and parallel processing
- **Line-of-Sight Analysis** -- Check RF path clearance including 60% first Fresnel zone obstruction
- **Mast Height Optimization** -- Iterative search finds minimum antenna height to clear obstructions
- **Fresnel Zone Profiling** -- Terrain profiles with first Fresnel zone radius at each sample point
- **CLI & Python API** -- Full command-line interface and importable Python API
- **Auto-Optimization** -- Memory-aware worker scaling via psutil

## How It Works

1. **Filter** LAS/LAZ tiles to a geographic area of interest
2. **Rasterize** the filtered point clouds into a Digital Surface Model (DSM) GeoTIFF, with automatic interpolation to fill gaps left by sparse LiDAR returns
3. **Analyze** point-to-point RF links against the DSM: compute terrain profiles, check geometric line-of-sight, and verify 60% first Fresnel zone clearance
4. **Report** results including clearance/obstruction status, minimum mast height needed, surface elevations, and Fresnel zone obstruction percentage

The LOS engine uses the DSM (first-return surface model, which includes buildings and vegetation) rather than a DTM (bare earth), so obstructions like rooftops and tree canopy are accounted for.

## Installation

```bash
# Clone
git clone https://github.com/davidemerson/jpmapper-lidar.git
cd jpmapper-lidar

# Install with conda (recommended — required for pdal)
conda create -n jpmapper python=3.11
conda activate jpmapper
conda install -c conda-forge pdal python-pdal
pip install -e .

# Or install Python deps only (no rasterization without pdal)
pip install numpy rasterio laspy shapely pyproj rich "typer[all]" pandas matplotlib
pip install -e .

# Verify
python -c "from jpmapper.cli.main import app; app(['--help'])"
```

### Optional dependencies

```bash
# Shapefile-based filtering
conda install -c conda-forge geopandas fiona

# Performance optimization (auto worker/memory scaling)
pip install psutil

# Interactive maps
pip install folium
```

## CLI Usage

### Filter LAS/LAZ tiles

```bash
# Filter by bounding box (coordinates in the LAS file's CRS)
jpmapper filter bbox data/ --bbox '583000 4506000 584000 4507000'
jpmapper filter bbox data/ --bbox '583000 4506000 584000 4507000' --dst filtered/

# Filter by shapefile boundary
jpmapper filter shapefile data/ --shapefile boundary.shp
jpmapper filter shapefile data/ -s boundary.shp --buffer 50 --dst selected/
```

### Rasterize to DSM

```bash
# Auto-detect CRS from LAS header
jpmapper rasterize tile input.las output.tif

# Explicit CRS and resolution
jpmapper rasterize tile input.las output.tif --epsg 6539 --resolution 0.25
```

The rasterizer creates a first-return DSM (max Z per pixel) and automatically fills nodata gaps using inverse-distance-weighted interpolation. This is critical for LiDAR data where point cloud density varies and raw rasterization leaves holes.

### Analyze point-to-point links

```bash
jpmapper analyze csv links.csv --las-dir data/ --epsg 6539
jpmapper analyze csv links.csv --las-dir data/ --json results.json --map map.png
```

The CSV should contain columns: `point_a_lat`, `point_a_lon`, `point_b_lat`, `point_b_lon`, and optionally `point_a_mast`, `point_b_mast` (antenna heights in meters above ground), `frequency_ghz`.

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
    resolution=0.1,  # 0.1m cell size
)
```

### Line-of-Sight Analysis

```python
from jpmapper.api import analyze_los

result = analyze_los(
    Path("dsm.tif"),
    point_a=(40.7128, -74.0060),  # lat, lon
    point_b=(40.7614, -73.9776),
    freq_ghz=5.8,
    max_mast_height_m=30,
)
print(f"Clear: {result['clear']}")
print(f"Mast needed: {result['mast_height_m']}m")
print(f"Surface A: {result['surface_height_a_m']}m")
print(f"Surface B: {result['surface_height_b_m']}m")
```

The `is_clear` check verifies that:
1. The geometric line-of-sight clears the terrain surface
2. 60% of the first Fresnel zone is unobstructed at every sample point along the path

If the path is blocked, the mast height search iterates upward at both endpoints until it finds the minimum height that achieves Fresnel clearance, or reports that the maximum height is insufficient.

### Terrain Profile

```python
from jpmapper.api import generate_profile

distances, terrain, fresnel = generate_profile(
    Path("dsm.tif"),
    point_a=(40.7128, -74.0060),
    point_b=(40.7614, -73.9776),
    n_samples=256,
)
# distances: array of meters along path
# terrain: array of surface elevations (m) from DSM
# fresnel: array of first Fresnel zone radii (m)
```

### Direct LOS Check

```python
from jpmapper.analysis.los import is_clear_direct

clear = is_clear_direct(
    from_lon=-74.006, from_lat=40.713, from_alt=10.0,
    to_lon=-73.978, to_lat=40.761, to_alt=10.0,
    dsm_file="dsm.tif",
    n_samples=100,
)
```

## Architecture

```
jpmapper/
  io/              # File I/O layer
    las.py         # LAS/LAZ header reading, bbox intersection filtering
    raster.py      # PDAL rasterization, nodata gap-filling, tile merging, DSM caching
  analysis/        # Core algorithms
    los.py         # LOS geometry, Fresnel zone, terrain profiling, mast optimization
    plots.py       # Matplotlib profile visualizations
  api/             # Public API (validation + thin wrappers)
    filter.py      # filter_by_bbox()
    raster.py      # rasterize_tile()
    analysis.py    # analyze_los(), generate_profile()
  cli/             # Typer CLI commands
    filter.py      # jpmapper filter bbox|shapefile
    rasterize.py   # jpmapper rasterize tile
    analyze.py     # jpmapper analyze csv
  config.py        # Configuration loading
  exceptions.py    # Exception hierarchy
```

**Data flow**: LAS files → IO layer (filter/rasterize/gap-fill) → Analysis (LOS/Fresnel/profile) → API (validation) → CLI (user interface)

### Key algorithms

**`_snap_to_valid()`** — Snaps a WGS84 coordinate to the nearest valid (non-nodata) DSM pixel within a 50-pixel search radius. Handles sparse LiDAR rasters where the query point may fall in a gap.

**`_is_clear_with_dataset()`** — Computes the LOS line between two points (ground elevation + antenna height), samples the terrain profile, then checks:
- Geometric clearance: LOS line is above terrain at all sample points
- Fresnel clearance: 60% of the first Fresnel zone radius is unobstructed at every point

**`_fill_nodata()`** — After PDAL rasterization, fills nodata gaps using GDAL's inverse-distance-weighted interpolation (via `rasterio.fill.fillnodata`). Search distance of 100 pixels.

**`_is_clear_points()`** — Iterative mast height search. Starts at 0, increases mast height at both endpoints by `step_m` until Fresnel clearance is achieved or `max_mast_height_m` is exceeded.

## Exception Hierarchy

```
JPMapperError
  ├── ConfigurationError
  ├── FileFormatError
  ├── GeoSpatialError
  │     ├── GeometryError
  │     ├── CRSError
  │     └── NoDataError
  ├── AnalysisError
  │     └── LOSError
  ├── RasterizationError
  └── FilterError
```

## Testing

```bash
pip install -e ".[dev]"
pytest                              # Run all tests
pytest tests/test_los_coverage.py   # LOS analysis tests
pytest tests/test_raster_io.py      # Rasterization tests
pytest tests/test_las_io.py         # LAS filtering tests
```

Tests use real temporary GeoTIFF fixtures (`flat_dsm`, `hill_dsm`) with proper CRS and transforms for LOS/analysis testing. Integration tests that require `pdal` are skipped when it's not available.

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | Numerical operations | Yes |
| rasterio | GeoTIFF I/O, nodata filling | Yes |
| laspy | LAS/LAZ file reading | Yes |
| shapely | Geometric operations (bbox intersection) | Yes |
| pyproj | CRS transformations (WGS84 ↔ projected) | Yes |
| rich | Terminal formatting | Yes |
| typer | CLI framework | Yes |
| pandas | CSV analysis | Yes |
| matplotlib | Profile plots | Yes |
| pdal / python-pdal | Point cloud rasterization | For rasterize command |
| psutil | Auto worker/memory optimization | Optional |
| geopandas + fiona | Shapefile filtering | Optional |
| folium | Interactive maps | Optional |

## License

See [LICENSE](LICENSE).
