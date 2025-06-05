"""Helpers for working with LAS/LAZ point‑cloud files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import laspy
from shapely.geometry import box, Polygon

logger = logging.getLogger(__name__)


def filter_las_by_bbox(
    las_files: Iterable[Path],
    bbox: tuple[float, float, float, float],
    *,
    dst_dir: Path | None = None,
) -> list[Path]:
    """Return subset of *las_files* whose extent intersects *bbox*.

    If *dst_dir* is given, copy the selected files there; otherwise
    return their original paths.
    """
    poly: Polygon = box(*bbox)
    selected: list[Path] = []

    for path in las_files:
        try:
            hdr = laspy.read_header(str(path))
        except Exception as err:  # noqa: BLE001
            logger.warning("Unable to read %s – %r", path, err)
            continue

        if poly.intersects(box(hdr.mins[0], hdr.mins[1], hdr.maxs[0], hdr.maxs[1])):
            selected.append(path)

    if dst_dir:
        dst_dir.mkdir(parents=True, exist_ok=True)
        out = []
        for src in selected:
            tgt = dst_dir / src.name
            tgt.write_bytes(src.read_bytes())
            out.append(tgt)
        return out

    return selected