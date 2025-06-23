"""Raster-I/O utilities – writes **first-return DSM** (max Z per pixel)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5
from pathlib import Path
from typing import Sequence, Tuple

import laspy
import rasterio
from rasterio.merge import merge as rio_merge

try:
    from pdal import Pipeline as _PDAL_PIPELINE  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _PDAL_PIPELINE = None  # noqa: N816

log = logging.getLogger(__name__)

# ─────────────────────────── PDAL helpers ───────────────────────────────────────
def _pipeline_dict(src: Path, dst: Path, epsg: int, res: float) -> dict:
    """Return PDAL pipeline dict that keeps **first returns** (max)."""
    return {
        "pipeline": [
            str(src),
            {  # drop only class 7 (noise) – keep everything else including roofs
                "type": "filters.range",
                "limits": "Classification![7:7]",
            },
            {
                "type": "writers.gdal",
                "filename": str(dst),
                "resolution": res,
                "output_type": "max",          # tallest return per pixel
                "gdaldriver": "GTiff",
                "spatialreference": f"EPSG:{epsg}",
            },
        ]
    }


def _run_pdal(pdict: dict) -> None:
    """Execute PDAL pipeline inline if python-pdal present, else via CLI."""
    if _PDAL_PIPELINE:
        _PDAL_PIPELINE(json.dumps(pdict)).execute()
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(json.dumps(pdict))
        subprocess.run(["pdal", "pipeline", tmp.name], check=True)

# Export _run_pdal with a public name for testing
run_pdal_pipeline = _run_pdal


# ─────────────────────────── public API ─────────────────────────────────────────
def rasterize_tile(
    src_las: Path,
    dst_tif: Path,
    epsg: int | None = None,
    resolution: float = 0.1,
    workers: int | None = None,
) -> Path:
    """
    Rasterize a single LAS/LAZ file into a GeoTIFF DSM.
    
    Args:
        src_las: Path to the source LAS/LAZ file
        dst_tif: Path where the output GeoTIFF will be written
        epsg: EPSG code for the output CRS. If None, auto-detects from LAS header.
        resolution: Cell size in meters (default: 0.1m)
        workers: Number of worker processes (unused, for API compatibility)
        
    Returns:
        Path to the created GeoTIFF file
        
    Raises:
        FileNotFoundError: If src_las does not exist
        PermissionError: If dst_tif cannot be written due to permissions
        ValueError: If resolution is not positive or if CRS cannot be determined
        RasterizationError: If rasterization fails
    """
    from jpmapper.exceptions import RasterizationError
    
    # Check if source file exists
    if not src_las.exists():
        raise FileNotFoundError(f"Source LAS file does not exist: {src_las}")
    
    # Validate resolution
    if resolution <= 0:
        raise ValueError(f"Resolution must be positive: {resolution}")
    
    # Auto-detect EPSG if not provided
    if epsg is None:
        try:
            with laspy.open(str(src_las)) as rdr:
                crs = rdr.header.parse_crs()
            if crs is None or crs.to_epsg() is None:
                raise ValueError(f"No EPSG in {src_las}")
            epsg = int(crs.to_epsg())
        except Exception as e:
            raise ValueError(f"Could not determine CRS from LAS header: {e}")

    # Create destination directory if it doesn't exist
    dst_tif.parent.mkdir(parents=True, exist_ok=True)
    
    # Run the PDAL pipeline
    try:
        _run_pdal(_pipeline_dict(src_las, dst_tif, epsg, resolution))
        log.debug("Rasterized %s → %s", src_las.name, dst_tif.name)
        return dst_tif
    except Exception as e:
        raise RasterizationError(f"Failed to rasterize {src_las}: {e}") from e

# ---------- parallel helper (Windows-safe) -------------------------------------
def _rasterize_one(args: Tuple[Path, Path, int, float]) -> Path:
    src, out_dir, epsg, res = args
    dst = out_dir / f"{src.stem}.tif"
    rasterize_tile(src, dst, epsg=epsg, resolution=res)
    return dst


def rasterize_dir_parallel(
    las_dir: Path,
    out_dir: Path,
    *,
    epsg: int = 6539,
    resolution: float = 0.1,
    workers: int | None = None,
) -> list[Path]:
    las_files = list(las_dir.glob("*.las"))
    if not las_files:
        raise FileNotFoundError(f"No .las in {las_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(p, out_dir, epsg, resolution) for p in las_files]

    if workers == 1:
        return [_rasterize_one(t) for t in tasks]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_rasterize_one, tasks))

# ---------- merge & cache unchanged -------------------------------------------
def merge_tiles(tifs: Sequence[Path], dst: Path) -> None:
    if not tifs:
        raise ValueError("merge_tiles() received 0 rasters")

    os.environ.setdefault("GDAL_CACHEMAX", "512")  # MB
    dst.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.Env():
        srcs = [rasterio.open(str(t)) for t in tifs]
        mosaic, transform = rio_merge(srcs, mem_limit=512)
        meta = srcs[0].meta.copy()
        meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform)
        with rasterio.open(dst, "w", **meta) as ds:
            ds.write(mosaic)
    log.debug("Merged %d tiles → %s", len(tifs), dst)


def cached_mosaic(
    las_dir: Path,
    cache_path: Path,
    *,
    epsg: int = 6539,
    resolution: float = 0.1,
    workers: int | None = None,
    force: bool = False,
) -> Path:
    if not force and cache_path.exists():
        return cache_path

    h = md5()
    for p in sorted(las_dir.glob("*.las")):
        st = p.stat()
        h.update(f"{p.name}{st.st_mtime_ns}{st.st_size}".encode())
    sig = h.hexdigest()[:8]

    tmp_dir = cache_path.with_suffix(f".tmp-{sig}")
    tifs = rasterize_dir_parallel(
        las_dir, tmp_dir, epsg=epsg, resolution=resolution, workers=workers
    )
    merge_tiles(tifs, cache_path)
    return cache_path
