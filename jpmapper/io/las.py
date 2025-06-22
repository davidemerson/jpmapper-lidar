"""Helpers for working with LAS/LAZ point-cloud files."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import laspy
from shapely.geometry import box, Polygon

from jpmapper.exceptions import FileFormatError, FilterError, GeometryError

logger = logging.getLogger(__name__)


def _read_header(path: Path) -> Optional[laspy.LasHeader]:
    """
    Return the LAS/LAZ header or None on failure (cheap streaming read).
    
    Args:
        path: Path to the LAS/LAZ file
        
    Returns:
        LAS header if successful, None if the file cannot be read
        
    Raises:
        FileFormatError: If the file format is invalid or corrupted
    """
    if not path.exists():
        logger.warning("File %s does not exist", path)
        return None
        
    try:
        with laspy.open(str(path)) as reader:  # laspy ≥2.4
            return reader.header
    except laspy.errors.LaspyException as err:
        logger.warning("Unable to read %s – laspy error: %r", path, err)
        raise FileFormatError(f"Invalid LAS/LAZ file format: {err}") from err
    except PermissionError as err:
        logger.warning("Permission denied reading %s: %r", path, err)
        return None
    except Exception as err:  # noqa: BLE001
        logger.warning("Unable to read %s – %r", path, err)
        return None


def filter_las_by_bbox(
    las_files: Iterable[Path],
    bbox: Tuple[float, float, float, float],
    *,
    dst_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Return subset of *las_files* whose extent intersects *bbox*.

    Args:
        las_files: Iterable of paths to LAS/LAZ files
        bbox: Bounding box as (min_x, min_y, max_x, max_y)
        dst_dir: Optional destination directory to copy filtered files.
                 If None, original file paths are returned.
    
    Returns:
        List of paths for files that intersect the bounding box.
        If dst_dir is provided, these will be paths to the copied files.
        
    Raises:
        GeometryError: If the bounding box is invalid
        FilterError: If copying files fails
    """
    # Validate bbox
    if not all(isinstance(x, (int, float)) for x in bbox):
        raise GeometryError(f"Invalid bbox coordinates: {bbox}")
    
    try:
        poly: Polygon = box(*bbox)
    except Exception as e:
        raise GeometryError(f"Could not create bbox polygon: {e}") from e
        
    selected: List[Path] = []
    errors: List[str] = []

    for path in las_files:
        hdr = _read_header(path)
        if hdr is None:
            continue

        try:
            file_bbox = box(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1])
            if poly.intersects(file_bbox):
                selected.append(path)
        except Exception as e:
            errors.append(f"{path.name}: {e}")
            continue
    
    if errors:
        logger.warning("Errors occurred while filtering files: %s", ", ".join(errors))

    if dst_dir and selected:
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied: List[Path] = []
        copy_errors: List[str] = []
        
        for src in selected:
            tgt = dst_dir / src.name
            try:
                tgt.write_bytes(src.read_bytes())
                copied.append(tgt)
            except Exception as e:
                copy_errors.append(f"{src.name}: {e}")
        
        if copy_errors:
            error_msg = ", ".join(copy_errors)
            logger.error("Failed to copy some files: %s", error_msg)
            raise FilterError(f"Failed to copy files: {error_msg}")
            
        return copied

    return selected
