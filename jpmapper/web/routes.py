"""API route handlers for the web interface."""
from __future__ import annotations

import asyncio
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException
from pyproj import Transformer

from jpmapper.web.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    BoundsResponse,
    CoverageCell,
    CoverageResponse,
    HealthResponse,
    Obstruction,
    ProfileData,
    SnapInfo,
)
from jpmapper.api.analysis import analyze_los, generate_profile
from jpmapper.analysis.los import _snap_to_valid, distance_between_points
from jpmapper.exceptions import GeometryError, NoDataError, AnalysisError, LOSError

SNAP_MAX_PX = 200

router = APIRouter()


def _get_dsm():
    from jpmapper.web.app import dsm_dataset

    if dsm_dataset is None:
        raise HTTPException(status_code=503, detail="DSM not loaded")
    return dsm_dataset


@router.get("/health", response_model=HealthResponse)
async def health():
    from jpmapper.web.app import dsm_dataset

    return HealthResponse(status="ok", dsm_loaded=dsm_dataset is not None)


@router.get("/bounds", response_model=BoundsResponse)
async def bounds():
    ds = _get_dsm()
    tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)
    left, bottom = tf.transform(ds.bounds.left, ds.bounds.bottom)
    right, top = tf.transform(ds.bounds.right, ds.bounds.top)
    # tf returns (lon, lat) since always_xy=True
    return BoundsResponse(
        min_lat=bottom,
        max_lat=top,
        min_lon=left,
        max_lon=right,
    )


_coverage_cache: CoverageResponse | None = None


def _compute_coverage(ds) -> CoverageResponse:
    """Downsample DSM into a grid and report only internal coverage gaps.

    First pass: compute coverage % for every cell.
    Second pass: only keep gap cells that have at least one neighbor with
    significant coverage (>= 50%).  This filters out the empty border region
    around the actual data footprint so only true internal holes are shown.
    """
    global _coverage_cache
    if _coverage_cache is not None:
        return _coverage_cache

    CELL_PX = 500
    rows, cols = ds.shape
    nodata = ds.nodata if ds.nodata is not None else -9999
    tf = Transformer.from_crs(ds.crs, 4326, always_xy=True)

    # First pass — build a grid of coverage percentages
    n_rows = (rows + CELL_PX - 1) // CELL_PX
    n_cols = (cols + CELL_PX - 1) // CELL_PX
    grid = np.zeros((n_rows, n_cols), dtype=float)

    for gr in range(n_rows):
        r0 = gr * CELL_PX
        r1 = min(r0 + CELL_PX, rows)
        for gc in range(n_cols):
            c0 = gc * CELL_PX
            c1 = min(c0 + CELL_PX, cols)
            window = ds.read(1, window=((r0, r1), (c0, c1)))
            total = window.size
            valid = int(np.sum(window != nodata))
            grid[gr, gc] = 100.0 * valid / total if total > 0 else 0

    # Second pass — only report gap cells adjacent to data
    cells = []
    for gr in range(n_rows):
        for gc in range(n_cols):
            pct = grid[gr, gc]
            if pct >= 90:
                continue  # not a gap

            # Check if any neighbor has real data (>= 50% coverage)
            has_data_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and grid[nr, nc] >= 50:
                    has_data_neighbor = True
                    break

            if not has_data_neighbor:
                continue  # border cell, skip

            r0 = gr * CELL_PX
            c0 = gc * CELL_PX
            r1 = min(r0 + CELL_PX, rows)
            c1 = min(c0 + CELL_PX, cols)

            x_min = ds.transform.c + c0 * ds.transform.a
            x_max = ds.transform.c + c1 * ds.transform.a
            y_max = ds.transform.f + r0 * ds.transform.e
            y_min = ds.transform.f + r1 * ds.transform.e

            lon_min, lat_min = tf.transform(x_min, y_min)
            lon_max, lat_max = tf.transform(x_max, y_max)

            cells.append(CoverageCell(
                min_lat=round(lat_min, 6),
                min_lon=round(lon_min, 6),
                max_lat=round(lat_max, 6),
                max_lon=round(lon_max, 6),
                coverage_pct=round(pct, 1),
            ))

    _coverage_cache = CoverageResponse(cell_size_px=CELL_PX, cells=cells)
    return _coverage_cache


@router.get("/coverage", response_model=CoverageResponse)
async def coverage():
    ds = _get_dsm()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(_compute_coverage, ds))
    return result


def _run_analysis(ds, req: AnalyzeRequest):
    """CPU-bound analysis work — runs in executor."""
    orig_a = (req.point_a.lat, req.point_a.lon)
    orig_b = (req.point_b.lat, req.point_b.lon)

    # Snap to nearest valid DSM cells (larger radius than default)
    (lat_a, lon_a), _, snap_dist_a = _snap_to_valid(ds, orig_a[1], orig_a[0], max_px=SNAP_MAX_PX)
    (lat_b, lon_b), _, snap_dist_b = _snap_to_valid(ds, orig_b[1], orig_b[0], max_px=SNAP_MAX_PX)

    point_a = (lat_a, lon_a)
    point_b = (lat_b, lon_b)

    snap_a = None
    if snap_dist_a > 0.5:
        snap_a = SnapInfo(
            original_lat=orig_a[0], original_lon=orig_a[1],
            snapped_lat=lat_a, snapped_lon=lon_a,
            snap_distance_m=round(snap_dist_a, 1),
        )

    snap_b = None
    if snap_dist_b > 0.5:
        snap_b = SnapInfo(
            original_lat=orig_b[0], original_lon=orig_b[1],
            snapped_lat=lat_b, snapped_lon=lon_b,
            snap_distance_m=round(snap_dist_b, 1),
        )

    # LOS check (using already-snapped coords — internal snap will be a no-op)
    result = analyze_los(
        ds,
        point_a,
        point_b,
        freq_ghz=req.freq_ghz,
        mast_a_height_m=req.mast_a_height_m,
        mast_b_height_m=req.mast_b_height_m,
        n_samples=256,
    )

    # Terrain profile
    distances, terrain, fresnel = generate_profile(
        ds, point_a, point_b, n_samples=256, freq_ghz=req.freq_ghz
    )

    # LOS line heights
    surface_a = result["surface_height_a_m"]
    surface_b = result["surface_height_b_m"]
    los_a = surface_a + req.mast_a_height_m
    los_b = surface_b + req.mast_b_height_m
    n = len(distances)
    los_heights = np.linspace(los_a, los_b, n)

    # Find obstructions (terrain above LOS line)
    clearance = los_heights - terrain
    obstructions = []
    obstruction_mask = clearance < 0

    if obstruction_mask.any():
        total_dist = distances[-1] if n > 0 else 0
        # Interpolate lat/lon along the path for obstruction locations
        lats = np.linspace(req.point_a.lat, req.point_b.lat, n)
        lons = np.linspace(req.point_a.lon, req.point_b.lon, n)

        indices = np.where(obstruction_mask)[0]
        # Cluster nearby obstructions — report the worst in each cluster
        clusters = []
        current_cluster = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] <= 3:
                current_cluster.append(indices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [indices[i]]
        clusters.append(current_cluster)

        for cluster in clusters:
            worst_idx = cluster[np.argmin(clearance[cluster])]
            overshoot = float(-clearance[worst_idx])
            if overshoot > 5:
                severity = "severe"
            elif overshoot > 1:
                severity = "moderate"
            else:
                severity = "minor"
            obstructions.append(
                Obstruction(
                    distance_along_path_m=round(float(distances[worst_idx]), 1),
                    lat=round(float(lats[worst_idx]), 6),
                    lon=round(float(lons[worst_idx]), 6),
                    terrain_height_m=round(float(terrain[worst_idx]), 1),
                    obstruction_height_m=round(overshoot, 1),
                    severity=severity,
                )
            )

    return AnalyzeResponse(
        clear=result["clear"],
        distance_m=round(result["distance_m"], 1),
        surface_height_a_m=round(surface_a, 1),
        surface_height_b_m=round(surface_b, 1),
        clearance_min_m=round(result["clearance_min_m"], 1),
        profile=ProfileData(
            distances_m=[round(float(d), 2) for d in distances],
            terrain_heights_m=[round(float(h), 2) for h in terrain],
            los_heights_m=[round(float(h), 2) for h in los_heights],
            fresnel_radii_m=[round(float(r), 2) for r in fresnel],
        ),
        obstructions=obstructions,
        snap_a=snap_a,
        snap_b=snap_b,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    ds = _get_dsm()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(_run_analysis, ds, req))
    except NoDataError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No LiDAR coverage at this location (searched {SNAP_MAX_PX} pixels). "
                "This area appears to be a gap in the source LAS data. "
                "Try placing your point in an area with DSM coverage "
                "(outside the shaded red zones on the map)."
            ),
        )
    except GeometryError as exc:
        if "outside valid DSM" in str(exc).lower() or "no valid DSM" in str(exc).lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    "No LiDAR coverage near this point. "
                    "This area is a gap in the source data — "
                    "check the red shaded zones on the map for coverage gaps."
                ),
            )
        raise HTTPException(status_code=400, detail=str(exc))
    except (AnalysisError, LOSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
