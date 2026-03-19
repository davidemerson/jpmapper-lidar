# JPMapper-LiDAR

A Python toolkit for LiDAR data processing and RF line-of-sight analysis. JPMapper filters LAS/LAZ point clouds, generates Digital Surface Models (DSMs), and analyzes wireless link clearance with Fresnel zone calculations.

## Features

- **LiDAR Filtering** -- Select LAS/LAZ tiles by bounding box or shapefile boundary
- **DSM Generation** -- Rasterize first-return point clouds to GeoTIFF with automatic nodata gap-filling and parallel processing
- **Line-of-Sight Analysis** -- Check RF path clearance including 60% first Fresnel zone obstruction
- **Mast Height Optimization** -- Iterative search finds minimum antenna height to clear obstructions
- **Fresnel Zone Profiling** -- Terrain profiles with first Fresnel zone radius at each sample point
- **Automatic Unit Normalization** -- DSM elevations and distances are converted to meters regardless of the CRS native unit (US survey feet, international feet, etc.)
- **CLI & Python API** -- Full command-line interface and importable Python API
- **Auto-Optimization** -- Memory-aware worker scaling via psutil

## How It Works

1. **Filter** LAS/LAZ tiles to a geographic area of interest
2. **Rasterize** the filtered point clouds into a Digital Surface Model (DSM) GeoTIFF, with automatic interpolation to fill gaps left by sparse LiDAR returns
3. **Analyze** point-to-point RF links against the DSM: compute terrain profiles, check geometric line-of-sight, and verify 60% first Fresnel zone clearance
4. **Report** results including clearance/obstruction status, minimum mast height needed, surface elevations, and Fresnel zone obstruction percentage

The LOS engine uses the DSM (first-return surface model, which includes buildings and vegetation) rather than a DTM (bare earth), so obstructions like rooftops and tree canopy are accounted for.

### Unit Handling

All elevations and distances returned by the analysis API are in **meters**, regardless of the DSM's native coordinate reference system. The LOS engine detects the CRS linear unit (e.g. US survey feet for EPSG:6539, meters for EPSG:32618) and applies the appropriate conversion factor automatically. Mast heights, alt buffers, and Fresnel radii are always specified in meters, so all arithmetic is unit-consistent.

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

### Debug DSM sampling

```bash
# Inspect DSM values at specific projected coordinates
jpmapper debug-dsm dsm.tif '980500,190500'
jpmapper debug-dsm dsm.tif '980100,190500;980500,190500;980900,190500'
```

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
print(f"Distance: {result['distance_m']:.1f}m")
```

Surface heights and distances are always returned in meters, even when the underlying DSM uses feet (e.g. EPSG:6539). The conversion is handled internally.

You can also specify independent mast heights at each endpoint instead of using the iterative search:

```python
result = analyze_los(
    Path("dsm.tif"),
    point_a=(40.7128, -74.0060),
    point_b=(40.7614, -73.9776),
    freq_ghz=5.8,
    mast_a_height_m=10,
    mast_b_height_m=15,
)
```

The `is_clear` check verifies that:
1. The geometric line-of-sight clears the terrain surface
2. 60% of the first Fresnel zone is unobstructed at every sample point along the path

If the path is blocked and `max_mast_height_m` is provided, the mast height search iterates upward at both endpoints until it finds the minimum height that achieves Fresnel clearance, or reports that the maximum height is insufficient.

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

For a simple boolean check without the full result dictionary:

```python
from jpmapper.analysis.los import is_clear_direct

clear = is_clear_direct(
    from_lon=-74.006, from_lat=40.713, from_alt=10.0,
    to_lon=-73.978, to_lat=40.761, to_alt=10.0,
    dsm_file="dsm.tif",
    n_samples=100,
)
```

`from_alt` and `to_alt` are absolute altitudes in meters (ground elevation + antenna height).

## Architecture

```
jpmapper/
  io/                    # File I/O layer
    las.py               # LAS/LAZ header reading, bbox intersection filtering
    raster.py            # PDAL rasterization, nodata gap-filling, tile merging, DSM caching
    metadata_raster.py   # Metadata-aware rasterization (geopandas)
    pdal_utils.py        # PDAL pipeline construction
  analysis/              # Core algorithms
    los.py               # LOS geometry, Fresnel zone, terrain profiling, mast optimization, unit conversion
    plots.py             # Matplotlib/Rich profile visualizations, analysis maps
  api/                   # Public API (validation + thin wrappers)
    filter.py            # filter_by_bbox()
    raster.py            # rasterize_tile()
    enhanced_raster.py   # rasterize_tile_with_metadata()
    analysis.py          # analyze_los(), generate_profile()
    shapefile_filter.py  # Shapefile-based spatial filtering
  cli/                   # Typer CLI commands
    main.py              # Root CLI app, sub-command registration, debug-dsm
    filter.py            # jpmapper filter bbox|shapefile
    rasterize.py         # jpmapper rasterize tile
    analyze.py           # jpmapper analyze csv
    analyze_utils.py     # CSV batch processing, parallel analysis, progress reporting
  config.py              # Configuration loading (~/.jpmapper.json, env vars)
  exceptions.py          # Exception hierarchy
  logging.py             # Rich logging setup
```

**Data flow**: LAS files → IO layer (filter/rasterize/gap-fill) → Analysis (LOS/Fresnel/profile) → API (validation) → CLI (user interface)

### Key algorithms

**`_unit_factor()`** — Detects the CRS linear unit via pyproj and returns the conversion factor to meters. Supports metre, US survey foot, and international foot. Applied in `_snap_to_valid()`, `profile()`, and `_compute_profile_with_dataset()` so that all elevations and distances are normalized to meters before any LOS or Fresnel calculation.

**`_snap_to_valid()`** — Snaps a WGS84 coordinate to the nearest valid (non-nodata) DSM pixel within a configurable search radius (default 50 pixels). Returns the surface elevation in meters after unit conversion. Handles sparse LiDAR rasters where the query point may fall in a gap.

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
pytest                                    # Run all tests (109+)
pytest tests/test_los_coverage.py         # LOS analysis tests
pytest tests/test_analysis.py             # Analysis integration tests
pytest tests/test_raster_io.py            # Rasterization tests
pytest tests/test_las_io.py               # LAS filtering tests
pytest tests/test_api_comprehensive.py    # API validation tests
pytest tests/test_cli.py                  # CLI command tests
pytest tests/test_end_to_end.py           # End-to-end workflow tests
```

Tests use real temporary GeoTIFF fixtures with proper CRS and transforms:

| Fixture | CRS | Description |
|---------|-----|-------------|
| `flat_dsm` | EPSG:6539 (US survey feet) | 100x100 flat raster at 10 ft elevation |
| `flat_dsm_meters` | EPSG:32618 (UTM metres) | 100x100 flat raster at 10 m elevation |
| `hill_dsm` | EPSG:6539 (US survey feet) | 100x100 raster with Gaussian hill (10-60 ft) |

The feet-based fixtures verify that unit conversion works correctly — a 10 ft DSM value should produce ~3.048 m in API results. The metres fixture confirms no double-conversion occurs.

Tests requiring optional dependencies (`pdal`, `geopandas`, `fiona`, `folium`, `psutil`) are automatically skipped when those packages are not installed.

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| numpy | Numerical operations | Yes |
| rasterio | GeoTIFF I/O, nodata filling | Yes |
| laspy | LAS/LAZ file reading | Yes |
| shapely | Geometric operations (bbox intersection) | Yes |
| pyproj | CRS transformations, unit detection | Yes |
| rich | Terminal formatting, progress bars | Yes |
| typer | CLI framework | Yes |
| pandas | CSV processing | Yes |
| matplotlib | Profile plots, analysis maps | Yes |
| pdal / python-pdal | Point cloud rasterization | For rasterize command |
| psutil | Auto worker/memory optimization | Optional |
| geopandas + fiona | Shapefile filtering, metadata-aware rasterization | Optional |
| folium | Interactive HTML maps | Optional |

## License

BSD 3-Clause. See [LICENSE](LICENSE).
