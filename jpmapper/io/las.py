"""Helpers for working with LAS/LAZ point-cloud files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import laspy
from shapely.geometry import box, Polygon

logger = logging.getLogger(__name__)


def _read_header(path: Path) -> laspy.LasHeader | None:
    """Return the LAS/LAZ header or *None* on failure (cheap streaming read)."""
    try:
        with laspy.open(str(path)) as reader:  # laspy ≥2.4
            return reader.header
    except Exception as err:  # noqa: BLE001
        logger.warning("Unable to read %s – %r", path, err)
        return None


def filter_las_by_bbox(
    las_files: Iterable[Path],
    bbox: tuple[float, float, float, float],
    *,
    dst_dir: Path | None = None,
) -> list[Path]:
    """Return subset of *las_files* whose extent intersects *bbox*.

    If *dst_dir* is given, copy the selected files there and return the new
    paths; otherwise return the original paths.
    """
    poly: Polygon = box(*bbox)
    selected: list[Path] = []

    for path in las_files:
        hdr = _read_header(path)
        if hdr is None:
            continue

        if poly.intersects(box(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1])):
            selected.append(path)

    if dst_dir:
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for src in selected:
            tgt = dst_dir / src.name
            tgt.write_bytes(src.read_bytes())
            copied.append(tgt)
        return copied

    return selected
