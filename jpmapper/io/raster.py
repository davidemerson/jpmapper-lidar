"""Rasterization helpers that wrap the PDAL CLI.

Public API
==========
>>> rasterize_tile(src_las: Path, dst_tif: Path, epsg: int, resolution: float = 1.0)
    Rasterize a single LAS/LAZ tile to a DSM GeoTIFF.

>>> merge_tiles(tiles: Sequence[Path], dst: Path)
    Mosaic a list of GeoTIFF *tiles* into *dst* using rasterio.merge.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pipeline_json(src: Path, dst: Path, epsg: int, resolution: float) -> str:
    """Return a minimal PDAL pipeline JSON that produces an IDW DSM.

    Notes
    -----
    * We drop “noise” points (classification 7) first.
    * `filters.smrf` re-classifies ground as `Classification == 2`.
      If you later want to keep only ground returns explicitly, insert
      another `filters.range` stage after SMRF:
          {"type": "filters.range", "limits": "Classification[2:2]"}
    """
    return json.dumps(
        {
            "pipeline": [
                str(src),
                {"type": "filters.range", "limits": "Classification![7:7]"},
                {"type": "filters.smrf"},  # marks ground points
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
    epsg: int,
    *,
    resolution: float = 1.0,
) -> None:
    """Rasterize *src_las* into *dst_tif* using PDAL.

    Creates parent folders as needed and raises
    :class:`subprocess.CalledProcessError` on PDAL failures.
    """
    dst_tif.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write(_pipeline_json(src_las, dst_tif, epsg, resolution))

    log.info("PDAL: %s → %s", src_las.name, dst_tif.name)
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
