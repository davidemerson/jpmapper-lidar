"""Helpers for working with LAS/LAZ point-cloud files."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import laspy
from shapely.geometry import box

from jpmapper.exceptions import FileFormatError, FilterError, GeometryError

logger = logging.getLogger(__name__)


def _read_header(path: Path) -> Optional[laspy.LasHeader]:
    """Return the LAS/LAZ header or None on failure (cheap streaming read).

    Raises:
        FileFormatError: If the file format is invalid or corrupted
    """
    if not path.exists():
        logger.warning("File %s does not exist", path)
        return None

    try:
        with laspy.open(str(path)) as reader:  # laspy >= 2.4
            return reader.header
    except laspy.errors.LaspyException as err:
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
    """Return subset of *las_files* whose extent intersects *bbox*.

    Args:
        las_files: Iterable of paths to LAS/LAZ files
        bbox: Bounding box as (min_x, min_y, max_x, max_y)
        dst_dir: Optional destination directory to copy filtered files.

    Returns:
        List of paths for files that intersect the bounding box.

    Raises:
        GeometryError: If the bounding box is invalid
        FilterError: If copying files fails
    """
    # Validate bbox
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise GeometryError(f"bbox expected 4 coordinates, got {len(bbox) if isinstance(bbox, tuple) else bbox}")

    if not all(isinstance(x, (int, float)) for x in bbox):
        raise GeometryError(f"Invalid bbox coordinates: {bbox}")

    min_x, min_y, max_x, max_y = bbox
    if min_x >= max_x or min_y >= max_y:
        raise GeometryError("min coordinates must be less than max")

    try:
        query_geom = box(min_x, min_y, max_x, max_y)
    except Exception as e:
        raise GeometryError(f"Could not create bbox polygon: {e}") from e

    selected: List[Path] = []

    for path in las_files:
        if not path.exists():
            logger.warning("File %s does not exist", path)
            continue

        try:
            header = _read_header(path)
            if header is None:
                continue

            las_min_x, las_min_y = header.mins[0], header.mins[1]
            las_max_x, las_max_y = header.maxs[0], header.maxs[1]
            las_geom = box(las_min_x, las_min_y, las_max_x, las_max_y)

            if las_geom.intersects(query_geom):
                selected.append(path)

        except FileFormatError:
            raise
        except Exception as e:
            logger.error("Error reading %s: %s", path, e)
            continue

    if dst_dir and selected:
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied: List[Path] = []
        copy_errors: List[str] = []

        for src in selected:
            tgt = dst_dir / src.name
            try:
                shutil.copy2(src, tgt)
                copied.append(tgt)
            except Exception as e:
                copy_errors.append(f"{src.name}: {e}")

        if copy_errors:
            error_msg = ", ".join(copy_errors)
            logger.error("Failed to copy some files: %s", error_msg)
            raise FilterError(f"Failed to copy files: {error_msg}")

        return copied

    return selected
