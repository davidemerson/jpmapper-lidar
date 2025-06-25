"""
JPMapper API - Raster Module
---------------------------

Functions for rasterizing LiDAR data into DSM (Digital Surface Model) GeoTIFFs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

from jpmapper.io.raster import (
    rasterize_tile as _rasterize_tile,
    rasterize_dir_parallel as _rasterize_dir_parallel,
    merge_tiles as _merge_tiles,
    cached_mosaic as _cached_mosaic,
)
from jpmapper.exceptions import RasterizationError, CRSError, NoDataError


def rasterize_tile(
    src_las: Path,
    dst_tif: Path,
    epsg: Optional[int] = None,
    resolution: float = 0.1,
) -> Path:
    """
    Rasterize a single LAS/LAZ file into a GeoTIFF DSM.
    
    Args:
        src_las: Path to the source LAS/LAZ file
        dst_tif: Path where the output GeoTIFF will be written
        epsg: EPSG code for the output CRS. If None, auto-detects from LAS header.
        resolution: Cell size in meters (default: 0.1m)
    
    Returns:
        Path to the created GeoTIFF file
    
    Raises:
        FileNotFoundError: If src_las does not exist
        PermissionError: If dst_tif cannot be written due to permissions
        CRSError: If epsg is None and CRS cannot be determined from LAS header
        RasterizationError: If rasterization fails
    
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import rasterize_tile
        >>> rasterize_tile(
        ...     Path("data/las/tile1.las"),
        ...     Path("data/dsm/tile1.tif"),
        ...     epsg=6539,
        ...     resolution=0.1
        ... )
    """
    # Check if source file exists
    if not src_las.exists():
        raise FileNotFoundError(f"Source LAS file does not exist: {src_las}")
    
    # Check if destination directory is writable
    try:
        dst_tif.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot write to destination directory: {dst_tif.parent}") from e
    
    try:
        _rasterize_tile(src_las, dst_tif, epsg, resolution=resolution)
        return dst_tif
    except ValueError as e:
        if "No EPSG" in str(e):
            raise CRSError(f"Could not determine CRS from LAS file and no EPSG provided: {e}") from e
        raise
    except Exception as e:
        raise RasterizationError(f"Failed to rasterize {src_las}: {e}") from e


def rasterize_directory(
    las_dir: Path,
    out_dir: Path,
    epsg: int = 6539,
    resolution: float = 0.1,
    workers: Optional[int] = None,
) -> List[Path]:
    """
    Rasterize all LAS/LAZ files in a directory to GeoTIFF files.
    
    Args:
        las_dir: Directory containing LAS/LAZ files
        out_dir: Directory where output GeoTIFF files will be written
        epsg: EPSG code for the output CRS
        resolution: Cell size in meters (default: 0.1m)
        workers: Number of parallel workers. If None, auto-detects optimal number.
                 If set to 1, processing is done serially.
    
    Returns:
        List of paths to created GeoTIFF files
    
    Raises:
        FileNotFoundError: If las_dir does not exist or contains no LAS/LAZ files
        PermissionError: If out_dir cannot be written to
        RasterizationError: If rasterization fails
    
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import rasterize_directory
        >>> tifs = rasterize_directory(
        ...     Path("data/las"), 
        ...     Path("data/dsm"),
        ...     epsg=6539,
        ...     resolution=0.1,
        ...     workers=4
        ... )
    """
    # Check if source directory exists
    if not las_dir.exists():
        raise FileNotFoundError(f"LAS directory does not exist: {las_dir}")
    
    # Check if destination directory is writable
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot write to output directory: {out_dir}") from e
    
    try:
        result = _rasterize_dir_parallel(
            las_dir, out_dir, epsg=epsg, resolution=resolution, workers=workers
        )
        
        if not result:
            raise NoDataError(f"No LAS/LAZ files found in {las_dir}")
            
        return result
    except FileNotFoundError as e:
        raise NoDataError(f"No LAS/LAZ files found in {las_dir}") from e
    except Exception as e:
        raise RasterizationError(f"Failed to rasterize files in {las_dir}: {e}") from e


def merge_tiles(
    tifs: Sequence[Path], 
    dst: Path
) -> Path:
    """
    Merge multiple GeoTIFF tiles into a single mosaic.
    
    Args:
        tifs: Sequence of paths to input GeoTIFF files
        dst: Path where the output mosaic will be written
    
    Returns:
        Path to the created mosaic GeoTIFF
    
    Raises:
        ValueError: If tifs is empty
        FileNotFoundError: If any of the input files don't exist
        PermissionError: If dst cannot be written due to permissions
        RasterizationError: If merging fails
    
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import merge_tiles
        >>> dsm_tiles = list(Path("data/dsm").glob("*.tif"))
        >>> merge_tiles(dsm_tiles, Path("data/mosaic.tif"))
    """
    # Validate inputs
    if not tifs:
        raise ValueError("No input GeoTIFF files provided")
    
    # Check if input files exist
    missing = [t for t in tifs if not t.exists()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {', '.join(str(m) for m in missing)}")
    
    # Check if destination directory is writable
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot write to destination directory: {dst.parent}") from e
    
    try:
        _merge_tiles(tifs, dst)
        return dst
    except Exception as e:
        raise RasterizationError(f"Failed to merge tiles: {e}") from e


def create_mosaic(
    las_dir: Path,
    cache_path: Path,
    epsg: int = 6539,
    resolution: float = 0.1,
    workers: Optional[int] = None,
    force: bool = False,
) -> Path:
    """
    Create a mosaic from LAS/LAZ files with caching.
    
    This function creates a cached mosaic from all LAS/LAZ files in a directory.
    If the cache file exists and the input files haven't changed, it returns the
    existing cache instead of regenerating it.
    
    Args:
        las_dir: Directory containing LAS/LAZ files
        cache_path: Path where the cached mosaic will be written
        epsg: EPSG code for the output CRS
        resolution: Cell size in meters (default: 0.1m)
        workers: Number of parallel workers. If None, auto-detects optimal number.
        force: If True, regenerate the mosaic even if the cache exists
    
    Returns:
        Path to the mosaic GeoTIFF
    
    Raises:
        FileNotFoundError: If las_dir does not exist or contains no LAS/LAZ files
        PermissionError: If cache_path cannot be written
        RasterizationError: If rasterization or merging fails
    
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import create_mosaic
        >>> mosaic = create_mosaic(
        ...     Path("data/las"),
        ...     Path("data/cached_mosaic.tif"),
        ...     epsg=6539,
        ...     resolution=0.1
        ... )
    """
    # Check if source directory exists
    if not las_dir.exists():
        raise FileNotFoundError(f"LAS directory does not exist: {las_dir}")
    
    # Check if cache directory is writable
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Cannot write to cache directory: {cache_path.parent}") from e
    
    try:
        return _cached_mosaic(
            las_dir, cache_path, epsg=epsg, resolution=resolution, workers=workers, force=force
        )
    except FileNotFoundError as e:
        raise NoDataError(f"No LAS/LAZ files found in {las_dir}") from e
    except Exception as e:
        raise RasterizationError(f"Failed to create mosaic: {e}") from e
