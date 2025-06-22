"""
JPMapper API - Filter Module
---------------------------

Functions for filtering LAS/LAZ files based on geographic criteria.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, List, Tuple

from jpmapper.io.las import filter_las_by_bbox as _filter_las_by_bbox
from jpmapper import config as _config
from jpmapper.exceptions import FilterError, ConfigurationError


def filter_by_bbox(
    las_files: Iterable[Path],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    dst_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Filter LAS/LAZ files by a bounding box.

    Args:
        las_files: Iterable of Path objects pointing to LAS/LAZ files
        bbox: Bounding box as (min_x, min_y, max_x, max_y). 
              If None, uses the bbox from config.
        dst_dir: Optional destination directory to copy filtered files.
                 If None, original file paths are returned.

    Returns:
        List of Path objects for files that intersect the bounding box.
        If dst_dir is provided, these will be paths to the copied files.
    
    Raises:
        ConfigurationError: If bbox is None and cannot be loaded from config
        FilterError: If filtering operation fails
        ValueError: If bbox coordinates are invalid
    
    Example:
        >>> from pathlib import Path
        >>> from jpmapper.api import filter_by_bbox
        >>> las_dir = Path("data/las")
        >>> filtered = filter_by_bbox(las_dir.glob("*.las"), bbox=(-74.1, 40.5, -73.9, 40.7))
    """
    if bbox is None:
        try:
            cfg = _config.load()
            bbox = cfg.bbox
        except Exception as e:
            raise ConfigurationError(f"Could not load bbox from config: {e}") from e
    
    # Validate bbox
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox: expected 4 coordinates, got {len(bbox)}")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"Invalid bbox: min coordinates must be less than max coordinates")
    
    try:
        return _filter_las_by_bbox(las_files, bbox=bbox, dst_dir=dst_dir)
    except Exception as e:
        raise FilterError(f"Error filtering LAS files: {e}") from e
