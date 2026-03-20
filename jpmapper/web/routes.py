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
    HealthResponse,
    Obstruction,
    ProfileData,
)
from jpmapper.api.analysis import analyze_los, generate_profile
from jpmapper.analysis.los import distance_between_points
from jpmapper.exceptions import GeometryError, NoDataError, AnalysisError, LOSError

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


def _run_analysis(ds, req: AnalyzeRequest):
    """CPU-bound analysis work — runs in executor."""
    point_a = (req.point_a.lat, req.point_a.lon)
    point_b = (req.point_b.lat, req.point_b.lon)

    # LOS check
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
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    ds = _get_dsm()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, partial(_run_analysis, ds, req))
    except (NoDataError, GeometryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (AnalysisError, LOSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result
