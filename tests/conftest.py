"""
Test environment configuration and dependency checking for JPMapper-LiDAR.

This module provides pytest configuration and environment verification functionality
to ensure tests run in a suitable environment with helpful feedback about missing dependencies.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _perform_checks(self):
        """Perform all dependency checks."""
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
        """Check if a specific module is available."""
        return self.check_results.get(module_name, {}).get("available", False)
    
    def get_missing_core_deps(self) -> List[str]:
        """Get list of missing core dependencies."""
        return [name for name, info in self.check_results.items() 
                if info["category"] == "core" and not info["available"]]
    
    def get_missing_enhanced_deps(self) -> List[str]:
        """Get list of missing enhanced dependencies."""
        return [name for name, info in self.check_results.items() 
                if info["category"] == "enhanced" and not info["available"]]
    
    def get_missing_optional_deps(self) -> List[str]:
        """Get list of missing optional dependencies."""
        return [name for name, info in self.check_results.items() 
                if info["category"] == "optional" and not info["available"]]
    
    def generate_report(self) -> str:
        """Generate a comprehensive dependency report."""
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
                status = "✓" if info["available"] else "✗"
                lines.append(f"  {status} {name} - {info['description']}")
        
        # Summary
        missing_core = self.get_missing_core_deps()
        missing_enhanced = self.get_missing_enhanced_deps()
        missing_optional = self.get_missing_optional_deps()
        
        lines.append("\n" + "=" * 50)
        lines.append("Test Environment Summary:")
        
        if not missing_core:
            lines.append("✓ Core dependencies: All available")
        else:
            lines.append(f"✗ Core dependencies: Missing {', '.join(missing_core)}")
        
        if not missing_enhanced:
            lines.append("✓ Enhanced features: All dependencies available")
        else:
            lines.append(f"⚠ Enhanced features: Missing {', '.join(missing_enhanced)}")
            lines.append("  Install with: conda install -c conda-forge geopandas fiona")
        
        if not missing_optional:
            lines.append("✓ Optional features: All dependencies available")
        else:
            lines.append(f"ℹ Optional features: Missing {', '.join(missing_optional)}")
        
        return "\n".join(lines)
    
    def get_skip_markers(self) -> Dict[str, str]:
        """Get skip markers for pytest based on missing dependencies."""
        markers = {}
        
        for name, info in self.check_results.items():
            if not info["available"]:
                markers[f"skip_if_no_{name}"] = f"{name} not available"
        
        return markers


# Global dependency checker instance
dependency_checker = DependencyChecker()


def pytest_configure(config):
    """Configure pytest with dependency information."""
    # Add custom markers for skipping based on dependencies
    config.addinivalue_line("markers", "requires_geopandas: mark test as requiring geopandas")
    config.addinivalue_line("markers", "requires_fiona: mark test as requiring fiona")
    config.addinivalue_line("markers", "requires_folium: mark test as requiring folium")
    config.addinivalue_line("markers", "requires_psutil: mark test as requiring psutil")
    config.addinivalue_line("markers", "requires_pdal: mark test as requiring pdal")
    
    # Store dependency checker in config for access by other functions
    config._dependency_checker = dependency_checker


def pytest_sessionstart(session):
    """Run at the start of test session to report environment status."""
    config = session.config
    
    # Only show detailed report if verbosity is high or there are missing core deps
    missing_core = dependency_checker.get_missing_core_deps()
    missing_enhanced = dependency_checker.get_missing_enhanced_deps()
    
    if config.option.verbose >= 1 or missing_core:
        print("\n" + dependency_checker.generate_report())
        print()
    
    # Issue warnings for missing dependencies
    if missing_core:
        warnings.warn(
            f"Missing core dependencies: {', '.join(missing_core)}. "
            "Some tests may fail. Please install missing packages.",
            UserWarning
        )
    
    if missing_enhanced:
        warnings.warn(
            f"Missing enhanced dependencies: {', '.join(missing_enhanced)}. "
            "Enhanced feature tests will be skipped. "
            "Install with: conda install -c conda-forge geopandas fiona",
            UserWarning
        )


def pytest_runtest_setup(item):
    """Run before each test to check if dependencies are available."""
    # Check for dependency markers and skip if requirements not met
    if item.get_closest_marker("requires_geopandas") and not dependency_checker.is_available("geopandas"):
        pytest.skip("geopandas not available")
    
    if item.get_closest_marker("requires_fiona") and not dependency_checker.is_available("fiona"):
        pytest.skip("fiona not available")
    
    if item.get_closest_marker("requires_folium") and not dependency_checker.is_available("folium"):
        pytest.skip("folium not available")
    
    if item.get_closest_marker("requires_psutil") and not dependency_checker.is_available("psutil"):
        pytest.skip("psutil not available")
    
    if item.get_closest_marker("requires_pdal") and not dependency_checker.is_available("pdal"):
        pytest.skip("pdal not available")


@pytest.fixture(scope="session")
def dependency_info():
    """Pytest fixture providing dependency information to tests."""
    return dependency_checker


@pytest.fixture
def enhanced_deps_available():
    """Fixture that indicates if enhanced dependencies are available."""
    return (dependency_checker.is_available("geopandas") and 
            dependency_checker.is_available("fiona"))


@pytest.fixture
def skip_if_no_enhanced_deps():
    """Fixture that skips test if enhanced dependencies are not available."""
    if not (dependency_checker.is_available("geopandas") and 
            dependency_checker.is_available("fiona")):
        pytest.skip("Enhanced dependencies (geopandas, fiona) not available")


def check_geopandas_available():
    """Helper function for checking geopandas availability (for backwards compatibility)."""
    return dependency_checker.is_available("geopandas")


def check_fiona_available():
    """Helper function for checking fiona availability (for backwards compatibility)."""
    return dependency_checker.is_available("fiona")


def check_enhanced_deps_available():
    """Helper function for checking if all enhanced dependencies are available."""
    return (dependency_checker.is_available("geopandas") and 
            dependency_checker.is_available("fiona"))
