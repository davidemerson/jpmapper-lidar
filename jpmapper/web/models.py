"""Pydantic request/response models for the web API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PointCoord(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class AnalyzeRequest(BaseModel):
    point_a: PointCoord
    point_b: PointCoord
    mast_a_height_m: float = Field(0.0, ge=0)
    mast_b_height_m: float = Field(0.0, ge=0)
    freq_ghz: float = Field(5.8, gt=0)

    @field_validator("freq_ghz")
    @classmethod
    def freq_reasonable(cls, v: float) -> float:
        if v > 300:
            raise ValueError("Frequency must be <= 300 GHz")
        return v


class SnapInfo(BaseModel):
    original_lat: float
    original_lon: float
    snapped_lat: float
    snapped_lon: float
    snap_distance_m: float


class Obstruction(BaseModel):
    distance_along_path_m: float
    lat: float
    lon: float
    terrain_height_m: float
    obstruction_height_m: float
    severity: str  # "minor", "moderate", "severe"


class ProfileData(BaseModel):
    distances_m: List[float]
    terrain_heights_m: List[float]
    los_heights_m: List[float]
    fresnel_radii_m: List[float]


class AnalyzeResponse(BaseModel):
    clear: bool
    distance_m: float
    surface_height_a_m: float
    surface_height_b_m: float
    clearance_min_m: float
    profile: ProfileData
    obstructions: List[Obstruction]
    snap_a: Optional[SnapInfo] = None
    snap_b: Optional[SnapInfo] = None


class CoverageCell(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    coverage_pct: float


class CoverageResponse(BaseModel):
    cell_size_px: int
    cells: List[CoverageCell]


class HealthResponse(BaseModel):
    status: str
    dsm_loaded: bool


class BoundsResponse(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
