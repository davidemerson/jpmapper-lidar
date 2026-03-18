"""Raster-I/O utilities – writes **first-return DSM** (max Z per pixel)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from hashlib import md5
from pathlib import Path
from typing import Sequence, Tuple

import laspy
import rasterio
from rasterio.merge import merge as rio_merge
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

# Optional imports for performance optimization
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from pdal import Pipeline as _PDAL_PIPELINE  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _PDAL_PIPELINE = None  # noqa: N816

from jpmapper.exceptions import RasterizationError

log = logging.getLogger(__name__)


def _get_optimal_workers(workers: int | None = None) -> int:
    """Get optimal number of workers based on available CPU cores and memory."""
    if workers is not None:
        cpu_count = multiprocessing.cpu_count()
        return max(1, min(workers, cpu_count * 2))

    cpu_count = multiprocessing.cpu_count()

    if HAS_PSUTIL:
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_limited_workers = max(1, int(available_memory_gb / 5))
            max_cpu_workers = max(1, min(8, int(cpu_count * 0.25)))
            optimal_workers = min(max_cpu_workers, memory_limited_workers)
            log.info(f"Auto-detected {optimal_workers} workers (CPU cores: {cpu_count}, "
                    f"Available memory: {available_memory_gb:.1f}GB)")
            return optimal_workers
        except Exception:
            pass

    optimal_workers = max(1, min(8, int(cpu_count * 0.25)))
    log.info(f"Auto-detected {optimal_workers} workers (CPU cores: {cpu_count})")
    return optimal_workers


def _optimize_gdal_cache():
    """Set optimal GDAL cache size based on available memory."""
    if HAS_PSUTIL:
        try:
            available_memory_mb = psutil.virtual_memory().available / (1024**2)
            gdal_cache_mb = min(int(available_memory_mb * 0.25), 4096)
            gdal_cache_mb = max(512, gdal_cache_mb)
            os.environ["GDAL_CACHEMAX"] = str(gdal_cache_mb)
            log.info(f"Set GDAL cache to {gdal_cache_mb}MB")
            return
        except Exception:
            pass

    os.environ.setdefault("GDAL_CACHEMAX", "1024")

# ─────────────────────────── PDAL helpers ───────────────────────────────────────
def _pipeline_dict(src: Path, dst: Path, epsg: int, res: float) -> dict:
    """Return PDAL pipeline dict that keeps **first returns** (max)."""
    return {
        "pipeline": [
            str(src),
            {
                "type": "filters.range",
                "limits": "Classification![7:7]",
            },
            {
                "type": "writers.gdal",
                "filename": str(dst),
                "resolution": res,
                "output_type": "max",
                "gdaldriver": "GTiff",
                "spatialreference": f"EPSG:{epsg}",
                "nodata": -9999,
                "gdalopts": "TILED=YES,COMPRESS=LZW,PREDICTOR=3",
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


def _detect_epsg(src_las: Path) -> int:
    """Auto-detect EPSG code from LAS header.

    Raises:
        ValueError: If CRS cannot be determined.
    """
    with laspy.open(str(src_las)) as rdr:
        crs = rdr.header.parse_crs()

    if crs is None:
        raise ValueError(f"No CRS in {src_las}")

    epsg_code = crs.to_epsg()
    if epsg_code is not None:
        return int(epsg_code)

    # Handle compound CRS (3D) - extract horizontal component
    crs_str = str(crs)
    if 'EPSG","6539"' in crs_str:
        log.info(f"Detected compound CRS with EPSG:6539 in {src_las}")
        return 6539
    elif 'New York Long Island' in crs_str and 'ftUS' in crs_str:
        log.info(f"Detected NY Long Island CRS in {src_las}, using EPSG:6539")
        return 6539

    raise ValueError(f"Cannot extract EPSG from compound CRS in {src_las}: {crs_str[:200]}")


# ─────────────────────────── public API ─────────────────────────────────────────
def rasterize_tile(
    src_las: Path,
    dst_tif: Path,
    epsg: int | None = None,
    resolution: float = 0.1,
    workers: int | None = None,
) -> Path:
    """Rasterize a single LAS/LAZ file into a GeoTIFF DSM.

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
        ValueError: If resolution is not positive or if CRS cannot be determined
        RasterizationError: If rasterization fails
    """
    if resolution <= 0:
        raise ValueError(f"Resolution must be positive: {resolution}")

    if not src_las.exists():
        raise FileNotFoundError(f"Source LAS file does not exist: {src_las}")

    if epsg is None:
        epsg = _detect_epsg(src_las)

    dst_tif.parent.mkdir(parents=True, exist_ok=True)

    try:
        _run_pdal(_pipeline_dict(src_las, dst_tif, epsg, resolution))
        log.debug("Rasterized %s -> %s", src_las.name, dst_tif.name)
        return dst_tif
    except Exception as e:
        raise RasterizationError(f"Failed to rasterize {src_las}: {e}") from e


# ---------- parallel helper (Windows-safe) -------------------------------------
def _rasterize_one(args: Tuple[Path, Path, int, float]) -> Path:
    """Rasterize a single LAS file, with error handling to skip problematic files."""
    src, out_dir, epsg, res = args
    dst = out_dir / f"{src.stem}.tif"

    try:
        rasterize_tile(src, dst, epsg=epsg, resolution=res)
        return dst
    except (MemoryError, RasterizationError) as e:
        log.warning(f"Skipping {src.name} due to error: {e}")
        placeholder = out_dir / f"{src.stem}_SKIPPED.txt"
        placeholder.write_text(f"Skipped due to error: {e}")
        return None


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

    optimal_workers = _get_optimal_workers(workers)

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:

        if optimal_workers == 1:
            task_id = progress.add_task(
                f"[cyan]Rasterizing {len(las_files)} LAS files (single-threaded)",
                total=len(tasks)
            )

            results = []
            for i, task in enumerate(tasks):
                src_file = task[0]
                progress.update(task_id, description=f"[cyan]Processing {src_file.name}")
                result = _rasterize_one(task)
                results.append(result)
                progress.advance(task_id)

        else:
            task_id = progress.add_task(
                f"[cyan]Rasterizing {len(las_files)} LAS files ({optimal_workers} workers)",
                total=len(tasks)
            )

            results = []
            completed_count = 0

            with ProcessPoolExecutor(max_workers=optimal_workers) as pool:
                future_to_task = {pool.submit(_rasterize_one, task): task for task in tasks}

                from concurrent.futures import as_completed
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    src_file = task[0]

                    try:
                        result = future.result()
                        results.append(result)

                        if result is not None:
                            status = "[green]done[/green]"
                        else:
                            status = "[yellow]skipped[/yellow]"

                    except Exception as e:
                        log.warning(f"Error processing {src_file.name}: {e}")
                        results.append(None)
                        status = "[red]failed[/red]"

                    completed_count += 1
                    progress.update(
                        task_id,
                        completed=completed_count,
                        description=f"[cyan]Processed {src_file.name} {status}"
                    )

    successful_results = [r for r in results if r is not None]

    if not successful_results:
        raise RasterizationError("All LAS files failed to rasterize")

    skipped_count = len(results) - len(successful_results)
    if skipped_count > 0:
        console.print(f"[yellow]Skipped {skipped_count} problematic files[/yellow]")

    console.print(f"[green]Successfully rasterized {len(successful_results)} of {len(las_files)} LAS files[/green]")
    log.info(f"Successfully rasterized {len(successful_results)} of {len(las_files)} LAS files")
    return successful_results

# ---------- merge & cache unchanged -------------------------------------------
def merge_tiles(tifs: Sequence[Path], dst: Path) -> None:
    if not tifs:
        raise ValueError("merge_tiles() received 0 rasters")

    _optimize_gdal_cache()

    dst.parent.mkdir(parents=True, exist_ok=True)

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:

        merge_task = progress.add_task(f"[magenta]Merging {len(tifs)} tiles into DSM", total=100)

        with rasterio.Env():
            progress.update(merge_task, completed=10, description="[magenta]Opening raster files")
            srcs = [rasterio.open(str(t)) for t in tifs]

            progress.update(merge_task, completed=30, description="[magenta]Computing mosaic")
            mosaic, transform = rio_merge(srcs, mem_limit=512, nodata=-9999)

            progress.update(merge_task, completed=70, description="[magenta]Writing merged DSM")
            meta = srcs[0].meta.copy()
            meta.update(
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                nodata=-9999
            )

            with rasterio.open(dst, "w", **meta) as ds:
                ds.write(mosaic)

            progress.update(merge_task, completed=100, description="[magenta]DSM merge complete")

            for src in srcs:
                src.close()

    console.print(f"[green]Merged {len(tifs)} tiles -> {dst.name}[/green]")
    log.debug("Merged %d tiles -> %s", len(tifs), dst)


def cached_mosaic(
    las_dir: Path,
    cache_path: Path,
    *,
    epsg: int = 6539,
    resolution: float = 0.1,
    workers: int | None = None,
    force: bool = False,
) -> Path:
    console = Console()

    if not force and cache_path.exists():
        console.print(f"[green]Using existing DSM cache: {cache_path.name}[/green]")
        return cache_path

    console.print(f"[cyan]Creating DSM from LAS files in {las_dir}[/cyan]")
    console.print(f"[cyan]Resolution: {resolution}m, EPSG: {epsg}[/cyan]")

    h = md5()
    for p in sorted(las_dir.glob("*.las")):
        st = p.stat()
        h.update(f"{p.name}{st.st_mtime_ns}{st.st_size}".encode())
    sig = h.hexdigest()[:8]

    tmp_dir = cache_path.with_suffix(f".tmp-{sig}")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    optimal_workers = _get_optimal_workers(workers)

    las_files = list(las_dir.glob("*.las"))
    if not las_files:
        raise FileNotFoundError(f"No .las in {las_dir}")

    console.print(f"[cyan]Found {len(las_files)} LAS files to process[/cyan]")

    tifs = rasterize_dir_parallel(
        las_dir, tmp_dir, epsg=epsg, resolution=resolution, workers=optimal_workers
    )

    if tifs:
        merge_tiles(tifs, cache_path)
    else:
        raise RasterizationError("No tiles were successfully rasterized")

    console.print(f"[bold green]DSM creation complete: {cache_path.name}[/bold green]")
    return cache_path
