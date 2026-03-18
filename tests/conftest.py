"""
Test environment configuration and dependency checking for JPMapper-LiDAR.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List

import pytest


class DependencyChecker:
    """Check and report on dependency availability for JPMapper tests."""

    def __init__(self):
        self.core_deps = [
            ("numpy", "Numerical operations"),
            ("pandas", "Data analysis and manipulation"),
            ("rasterio", "Geospatial raster data processing"),
            ("laspy", "LAS/LAZ file reading and writing"),
            ("shapely", "Geometric operations"),
            ("pyproj", "Cartographic projections"),
            ("rich", "Terminal formatting"),
            ("typer", "Command-line interface"),
        ]

        self.enhanced_deps = [
            ("geopandas", "Geospatial data analysis (metadata-aware rasterization)"),
            ("fiona", "Geospatial vector data I/O (shapefile support)"),
        ]

        self.optional_deps = [
            ("folium", "Interactive map creation"),
            ("psutil", "Performance optimization"),
            ("pdal", "Point cloud processing"),
        ]

        self.check_results = {}
        self._perform_checks()

    def _check_import(self, module_name: str) -> bool:
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _perform_checks(self):
        for deps_list, category in [
            (self.core_deps, "core"),
            (self.enhanced_deps, "enhanced"),
            (self.optional_deps, "optional")
        ]:
            for module_name, description in deps_list:
                available = self._check_import(module_name)
                self.check_results[module_name] = {
                    "available": available,
                    "description": description,
                    "category": category
                }

    def is_available(self, module_name: str) -> bool:
        return self.check_results.get(module_name, {}).get("available", False)

    def get_missing_core_deps(self) -> List[str]:
        return [name for name, info in self.check_results.items()
                if info["category"] == "core" and not info["available"]]

    def get_missing_enhanced_deps(self) -> List[str]:
        return [name for name, info in self.check_results.items()
                if info["category"] == "enhanced" and not info["available"]]

    def get_missing_optional_deps(self) -> List[str]:
        return [name for name, info in self.check_results.items()
                if info["category"] == "optional" and not info["available"]]

    def generate_report(self) -> str:
        lines = []
        lines.append("JPMapper-LiDAR Test Environment Report")
        lines.append("=" * 50)

        for category, title in [("core", "Core Dependencies"),
                               ("enhanced", "Enhanced Dependencies"),
                               ("optional", "Optional Dependencies")]:
            lines.append(f"\n{title}:")
            category_deps = [(name, info) for name, info in self.check_results.items()
                           if info["category"] == category]
            if not category_deps:
                lines.append("  None defined")
                continue
            for name, info in category_deps:
                status = "+" if info["available"] else "-"
                lines.append(f"  {status} {name} - {info['description']}")

        missing_core = self.get_missing_core_deps()
        missing_enhanced = self.get_missing_enhanced_deps()

        lines.append("\n" + "=" * 50)
        lines.append("Test Environment Summary:")

        if not missing_core:
            lines.append("+ Core dependencies: All available")
        else:
            lines.append(f"- Core dependencies: Missing {', '.join(missing_core)}")

        if not missing_enhanced:
            lines.append("+ Enhanced features: All dependencies available")
        else:
            lines.append(f"  Enhanced features: Missing {', '.join(missing_enhanced)}")
            lines.append("  Install with: conda install -c conda-forge geopandas fiona")

        return "\n".join(lines)


# Global dependency checker instance
dependency_checker = DependencyChecker()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_geopandas: mark test as requiring geopandas")
    config.addinivalue_line("markers", "requires_fiona: mark test as requiring fiona")
    config.addinivalue_line("markers", "requires_folium: mark test as requiring folium")
    config.addinivalue_line("markers", "requires_psutil: mark test as requiring psutil")
    config.addinivalue_line("markers", "requires_pdal: mark test as requiring pdal")
    config.addinivalue_line("markers", "integration: integration tests")
    config._dependency_checker = dependency_checker


def pytest_sessionstart(session):
    config = session.config
    missing_core = dependency_checker.get_missing_core_deps()

    if config.option.verbose >= 1 or missing_core:
        print("\n" + dependency_checker.generate_report())
        print()

    if missing_core:
        warnings.warn(
            f"Missing core dependencies: {', '.join(missing_core)}. "
            "Some tests may fail. Please install missing packages.",
            UserWarning
        )

    missing_enhanced = dependency_checker.get_missing_enhanced_deps()
    if missing_enhanced:
        warnings.warn(
            f"Missing enhanced dependencies: {', '.join(missing_enhanced)}. "
            "Enhanced feature tests will be skipped. "
            "Install with: conda install -c conda-forge geopandas fiona",
            UserWarning
        )


def pytest_runtest_setup(item):
    for dep in ["geopandas", "fiona", "folium", "psutil", "pdal"]:
        if item.get_closest_marker(f"requires_{dep}") and not dependency_checker.is_available(dep):
            pytest.skip(f"{dep} not available")


@pytest.fixture(scope="session")
def dependency_info():
    return dependency_checker


@pytest.fixture
def enhanced_deps_available():
    return (dependency_checker.is_available("geopandas") and
            dependency_checker.is_available("fiona"))


@pytest.fixture
def skip_if_no_enhanced_deps():
    if not (dependency_checker.is_available("geopandas") and
            dependency_checker.is_available("fiona")):
        pytest.skip("Enhanced dependencies (geopandas, fiona) not available")


def check_geopandas_available():
    """Helper for checking geopandas availability."""
    return dependency_checker.is_available("geopandas")


def check_fiona_available():
    """Helper for checking fiona availability."""
    return dependency_checker.is_available("fiona")


def check_enhanced_deps_available():
    """Helper for checking if all enhanced dependencies are available."""
    return (dependency_checker.is_available("geopandas") and
            dependency_checker.is_available("fiona"))


# ─── Shared DSM fixtures for LOS/analysis tests ────────────────────────────


@pytest.fixture
def flat_dsm(tmp_path):
    """Create a 100x100 flat raster at 10m elevation with proper CRS/transform."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    dsm_path = tmp_path / "flat_dsm.tif"
    height, width = 100, 100
    # Bounds roughly covering a small area in NYC (EPSG:6539 – NAD83 NY Long Island ftUS)
    # Using projected coordinates in feet
    west, south, east, north = 980000.0, 190000.0, 981000.0, 191000.0
    transform = from_bounds(west, south, east, north, width, height)

    data = np.full((1, height, width), 10.0, dtype=np.float32)

    with rasterio.open(
        dsm_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype="float32",
        crs="EPSG:6539",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data)

    return dsm_path


@pytest.fixture
def hill_dsm(tmp_path):
    """Create a 100x100 raster with a gaussian hill for LOS blockage testing."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    dsm_path = tmp_path / "hill_dsm.tif"
    height, width = 100, 100
    west, south, east, north = 980000.0, 190000.0, 981000.0, 191000.0
    transform = from_bounds(west, south, east, north, width, height)

    y, x = np.mgrid[0:height, 0:width]
    # Hill centered at (50, 50) with peak at 60m, base at 10m
    data = 10.0 + 50.0 * np.exp(-0.005 * ((x - 50)**2 + (y - 50)**2))
    data = data.astype(np.float32)[np.newaxis, :, :]

    with rasterio.open(
        dsm_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=1, dtype="float32",
        crs="EPSG:6539",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(data)

    return dsm_path
