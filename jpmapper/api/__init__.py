"""
JPMapper API
-----------

This module provides a programmatic API for the JPMapper LiDAR toolkit.
It allows for filtering, rasterizing, and analyzing LiDAR data without using the CLI.

Main components:
- filter: Functions for filtering LAS/LAZ files
- raster: Functions for rasterizing LiDAR data
- analysis: Functions for line-of-sight and Fresnel zone analysis
"""

from jpmapper.api.filter import filter_by_bbox
from jpmapper.api.raster import rasterize_tile, rasterize_directory, merge_tiles
from jpmapper.api.analysis import analyze_los, generate_profile, save_profile_plot

# Optional shapefile support (requires geopandas)
try:
    from jpmapper.api.shapefile_filter import (
        filter_by_shapefile, 
        create_boundary_from_las_files
    )
    _HAS_SHAPEFILE_SUPPORT = True
except ImportError:
    _HAS_SHAPEFILE_SUPPORT = False

# Optional enhanced rasterization with metadata support
try:
    from jpmapper.api.enhanced_raster import (
        rasterize_tile_with_metadata,
        batch_rasterize_with_metadata,
        generate_processing_report
    )
    _HAS_ENHANCED_RASTER = True
except ImportError:
    _HAS_ENHANCED_RASTER = False

__all__ = [
    # Filter operations
    "filter_by_bbox",
    # Rasterization operations
    "rasterize_tile",
    "rasterize_directory", 
    "merge_tiles",
    # Analysis operations
    "analyze_los",
    "generate_profile",
    "save_profile_plot",
]

# Add shapefile functions if available
if _HAS_SHAPEFILE_SUPPORT:
    __all__.extend([
        "filter_by_shapefile",
        "create_boundary_from_las_files"
    ])

# Add enhanced rasterization to exports if available
if _HAS_ENHANCED_RASTER:
    __all__.extend([
        "rasterize_tile_with_metadata",
        "batch_rasterize_with_metadata", 
        "generate_processing_report"
    ])
