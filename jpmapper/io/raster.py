"""Rasterization helpers that wrap the PDAL CLI.

Key parameters
--------------
* resolution – defaults to 0.1 m (per user spec).
* epsg       – optional.  When ``None`` we auto-detect CRS from the LAS header
               via laspy ≥ 2.4 (`header.parse_crs().to_epsg()`).

Public API
==========
rasterize_tile(src_las: Path, dst_tif: Path,
               epsg: int | None = None, resolution: float = 0.1)
merge_tiles(tiles: Sequence[Path], dst: Path)
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import laspy

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_epsg(src_las: Path) -> int:
    """Return EPSG code declared in *src_las* header or raise ``ValueError``."""
    with laspy.open(str(src_las)) as reader:
        crs = reader.header.parse_crs()  # pyproj CRS or None
    if crs is None or crs.to_epsg() is None:
        raise ValueError(f"Unable to detect EPSG for {src_las.name}")
    return int(crs.to_epsg())


def _pipeline_json(src: Path, dst: Path, epsg: int, resolution: float) -> str:
    """Return a minimal PDAL pipeline JSON that produces an IDW DSM."""
    return json.dumps(
        {
            "pipeline": [
                str(src),
                {"type": "filters.range", "limits": "Classification![7:7]"},  # drop noise
                {"type": "filters.smrf"},  # ground classification
                {
                    "type": "writers.gdal",
                    "filename": str(dst),
                    "resolution": resolution,
                    "output_type": "idw",
                    "gdaldriver": "GTiff",
                    "spatialreference": f"EPSG:{epsg}",
                },
            ]
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rasterize_tile(
    src_las: Path,
    dst_tif: Path,
    epsg: int | None = None,
    *,
    resolution: float = 0.1,
) -> None:
    """Rasterize *src_las* into *dst_tif* using PDAL.

    If *epsg* is ``None`` we infer it from the LAS header.
    Raises :class:`subprocess.CalledProcessError` on PDAL failure.
    """
    if epsg is None:
        epsg = _detect_epsg(src_las)
        log.debug("Auto-detected EPSG %s from %s", epsg, src_las.name)

    dst_tif.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write(_pipeline_json(src_las, dst_tif, epsg, resolution))

    log.info("PDAL: %s → %s (%.2f m)", src_las.name, dst_tif.name, resolution)
    subprocess.run(["pdal", "pipeline", tmp.name], check=True)


def merge_tiles(tiles: Sequence[Path], dst: Path) -> None:
    """Merge multiple GeoTIFF *tiles* into *dst* using rasterio.merge."""
    import rasterio
    from rasterio.merge import merge as rio_merge

    dst.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env():
        sources = [rasterio.open(str(t)) for t in tiles]
        mosaic, transform = rio_merge(sources)

        meta = sources[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
            }
        )

        with rasterio.open(dst, "w", **meta) as dst_ds:
            dst_ds.write(mosaic)
