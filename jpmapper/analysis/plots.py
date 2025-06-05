"""Plotting helper for terrain/Fresnel profiles."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple


def save_profile_png(
    dist_m: np.ndarray,
    terrain_m: np.ndarray,
    fresnel_m: np.ndarray,
    out_png: Path,
    title: str | None = None,
) -> None:
    """Render and save a cross-section profile to *out_png*."""
    plt.figure(figsize=(8, 4))
    plt.plot(dist_m, terrain_m, label="Terrain")
    plt.plot(dist_m, terrain_m + fresnel_m * 0.6, "--", label="60% Fresnel")
    plt.fill_between(dist_m, terrain_m, terrain_m + fresnel_m * 0.6, alpha=0.2)
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
