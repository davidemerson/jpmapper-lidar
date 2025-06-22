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
