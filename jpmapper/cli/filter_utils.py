"""
Functions for filtering LAS/LAZ files based on spatial criteria.
"""
from pathlib import Path
from typing import List, Optional, Tuple, Union

import typer

from jpmapper.exceptions import FilterError, GeometryError


def filter_las_by_bbox(
    src_files: List[Path],
    bbox: Tuple[float, float, float, float],
    dst_dir: Optional[Path] = None
) -> List[Path]:
    """
    Filter LAS/LAZ files by bounding box.
    
    Args:
        src_files: List of LAS/LAZ file paths
        bbox: Bounding box as (min_x, min_y, max_x, max_y)
        dst_dir: Optional destination directory for filtered files
        
    Returns:
        List of file paths that intersect the bounding box
        
    Raises:
        GeometryError: If the bounding box is invalid
        FilterError: If there's an error filtering the files
    """
    from jpmapper.api import filter_by_bbox
    
    try:
        return filter_by_bbox(src_files, bbox=bbox, dst_dir=dst_dir)
    except Exception as e:
        if isinstance(e, GeometryError):
            raise
        raise FilterError(f"Error filtering LAS files: {e}") from e
