# Updated Dependencies for Enhanced JPMapper-LiDAR

## What Changed

The JPMapper-LiDAR project now includes **metadata-aware rasterization** as a core feature, which requires additional dependencies that were previously optional.

## Updated Dependencies

The following dependencies have been moved from "optional" to "required":

- **geopandas** ≥0.14.0 - Required for shapefile support and metadata-aware rasterization
- **fiona** ≥1.9.0 - Required for geospatial vector data I/O

## Updated Installation Instructions

### Quick Start (Updated)
```bash
# Clone and setup
git clone https://github.com/davidemerson/jpmapper-lidar.git
cd jpmapper-lidar
conda create -n jpmapper --file requirements.txt python=3.11
conda activate jpmapper

# Install ALL dependencies including enhanced features
conda install -c conda-forge pdal python-pdal rasterio laspy shapely pyproj rich typer matplotlib pandas folium psutil geopandas fiona

pip install -e .

# Verify everything is working
python verify_installation.py
```

### For Existing Installations

If you already have JPMapper installed but want to use the enhanced metadata-aware features:

```bash
# Activate your existing environment
conda activate jpmapper

# Install the additional dependencies
conda install -c conda-forge geopandas fiona

# Verify the enhanced features are now available
python verify_installation.py
```

## New Verification Script

A new `verify_installation.py` script has been added that checks:

- ✓ All core dependencies are installed
- ✓ Enhanced functionality dependencies are available  
- ✓ API functions can be imported
- ✓ CLI commands work
- ✓ Summary of what features are available

## Why These Dependencies Are Now Required

The enhanced metadata-aware rasterization features provide significant benefits:

1. **Automatic CRS Detection** - Reads projection from .prj files
2. **Tile Boundary Matching** - Uses shapefile tile index for precise boundaries
3. **Quality Assessment** - Extracts accuracy metrics from XML metadata
4. **Resolution Optimization** - Suggests optimal resolution based on dataset accuracy

These features make LiDAR processing much more reliable and accurate, so they're now considered core functionality rather than optional extras.

## Files Updated

- ✅ `requirements.txt` - Added geopandas and fiona as required dependencies
- ✅ `README.md` - Updated all installation instructions
- ✅ `verify_installation.py` - New verification script to check installation
- ✅ Dependency overview section updated to explain each requirement

## Impact on Tests

With these dependencies installed, the test suite will now run **all 134 tests** instead of skipping the 6 enhanced feature tests. This provides complete validation that all functionality is working correctly.

## Next Steps

1. **Install the dependencies**: Follow the updated installation instructions
2. **Run verification**: Use `python verify_installation.py` to confirm everything works
3. **Run full test suite**: Use `pytest` to verify all 134 tests pass
4. **Try enhanced features**: Use the metadata-aware rasterization functions in your workflows

The project is now ready for full production use with all enhanced features available! 🚀
