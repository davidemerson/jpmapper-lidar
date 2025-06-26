#!/usr/bin/env python3
"""
Installation verification script for JPMapper-LiDAR.

This script checks that all required dependencies are properly installed
and that the enhanced metadata-aware features are available.

This script uses the same dependency checking logic as the test suite
to ensure consistency between standalone verification and test environment checks.
"""

import sys
from pathlib import Path

# Try to import the dependency checker from tests, fall back to local implementation
try:
    # Add tests directory to path so we can import conftest
    test_dir = Path(__file__).parent / "tests"
    if test_dir not in sys.path:
        sys.path.insert(0, str(test_dir))
    
    from conftest import DependencyChecker
    
    def main():
        """Main verification function using the shared dependency checker."""
        print("JPMapper-LiDAR Installation Verification")
        print("(Using shared test environment dependency checker)")
        
        checker = DependencyChecker()
        report = checker.generate_report()
        print(report)
        
        # API Tests
        print("\nJPMapper API:")
        try:
            from jpmapper.api import rasterize_tile, analyze_los, filter_by_bbox
            print("✓ Core API functions available")
            api_ok = True
        except ImportError as e:
            print(f"✗ Core API import failed: {e}")
            api_ok = False
        
        if checker.is_available("geopandas") and checker.is_available("fiona"):
            try:
                from jpmapper.api import rasterize_tile_with_metadata, filter_by_shapefile
                print("✓ Enhanced API functions available")
                enhanced_api_ok = True
            except ImportError as e:
                print(f"✗ Enhanced API import failed: {e}")
                enhanced_api_ok = False
        else:
            print("⚠ Enhanced API functions not tested (missing dependencies)")
            enhanced_api_ok = False
        
        # CLI Test
        print("\nCommand-Line Interface:")
        try:
            import subprocess
            # Try the direct command first
            result = subprocess.run(["jpmapper", "--help"], 
                                  capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print("✓ CLI command 'jpmapper' available")
                cli_ok = True
            else:
                # Fallback to python -m if direct command fails
                result = subprocess.run([sys.executable, "-m", "jpmapper.cli.main", "--help"], 
                                      capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
                if result.returncode == 0:
                    print("✓ CLI available via Python module")
                    cli_ok = True
                else:
                    print(f"✗ CLI command failed with code {result.returncode}")
                    print(f"  Error: {result.stderr.strip() if result.stderr else 'No error message'}")
                    cli_ok = False
        except Exception as e:
            print(f"✗ CLI test failed: {e}")
            cli_ok = False
        
        # Final Summary
        missing_core = checker.get_missing_core_deps()
        missing_enhanced = checker.get_missing_enhanced_deps()
        
        print(f"\nPython version: {sys.version}")
        print(f"JPMapper location: {Path(__file__).parent}")
        
        if not missing_core and api_ok and cli_ok and not missing_enhanced and enhanced_api_ok:
            print("\n🎉 Installation complete! All features available.")
            print("✓ Ready for full test suite execution with all features")
            return 0
        elif not missing_core and api_ok and cli_ok:
            print("\n⚠ Installation mostly complete. Enhanced features require additional dependencies.")
            print("ℹ Some tests will be skipped during test suite execution")
            return 1
        else:
            print("\n❌ Installation incomplete. Please check missing dependencies.")
            print("❌ Test suite execution may fail")
            return 2

except ImportError:
    # Fallback to original implementation if conftest not available
    def check_dependency(module_name, description=""):
        """Check if a module can be imported."""
        try:
            __import__(module_name)
            print(f"✓ {module_name} - {description}")
            return True
        except ImportError:
            print(f"✗ {module_name} - {description} (MISSING)")
            return False
    
    def main():
        """Fallback main function with original implementation."""
        print("JPMapper-LiDAR Installation Verification")
        print("(Using fallback dependency checker)")
        print("=" * 50)
        # Core dependencies
        print("\nCore Dependencies:")
        core_deps = [
            ("numpy", "Numerical operations"),
            ("pandas", "Data analysis and manipulation"),
            ("rasterio", "Geospatial raster data processing"),
            ("laspy", "LAS/LAZ file reading and writing"),
            ("shapely", "Geometric operations"),
            ("pyproj", "Cartographic projections"),
            ("rich", "Terminal formatting"),
            ("typer", "Command-line interface"),
            ("matplotlib", "Visualization and plotting"),
        ]
        
        core_ok = all(check_dependency(dep, desc) for dep, desc in core_deps)
        
        # Enhanced functionality dependencies
        print("\nEnhanced Functionality Dependencies:")
        enhanced_deps = [
            ("geopandas", "Geospatial data analysis (metadata-aware rasterization)"),
            ("fiona", "Geospatial vector data I/O (shapefile support)"),
        ]
        
        enhanced_ok = all(check_dependency(dep, desc) for dep, desc in enhanced_deps)
        
        # Optional dependencies
        print("\nOptional Dependencies:")
        optional_deps = [
            ("folium", "Interactive map creation"),
            ("psutil", "Performance optimization"),
            ("pdal", "Point cloud processing"),
        ]
        
        for dep, desc in optional_deps:
            check_dependency(dep, desc)
        
        # Test JPMapper API
        print("\nJPMapper API:")
        try:
            from jpmapper.api import rasterize_tile, analyze_los, filter_by_bbox
            print("✓ Core API functions available")
            api_ok = True
        except ImportError as e:
            print(f"✗ Core API import failed: {e}")
            api_ok = False
        
        if enhanced_ok:
            try:
                from jpmapper.api import rasterize_tile_with_metadata, filter_by_shapefile
                print("✓ Enhanced API functions available")
                enhanced_api_ok = True
            except ImportError as e:
                print(f"✗ Enhanced API import failed: {e}")
                enhanced_api_ok = False
        else:
            print("⚠ Enhanced API functions not tested (missing dependencies)")
            enhanced_api_ok = False
        
        # Test CLI
        print("\nCommand-Line Interface:")
        try:
            import subprocess
            # Try the direct command first
            result = subprocess.run(["jpmapper", "--help"], 
                                  capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
            if result.returncode == 0:
                print("✓ CLI command 'jpmapper' available")
                cli_ok = True
            else:
                # Fallback to python -m if direct command fails
                result = subprocess.run([sys.executable, "-m", "jpmapper.cli.main", "--help"], 
                                      capture_output=True, text=True, timeout=10, encoding='utf-8', errors='replace')
                if result.returncode == 0:
                    print("✓ CLI available via Python module")
                    cli_ok = True
                else:
                    print(f"✗ CLI command failed with code {result.returncode}")
                    print(f"  Error: {result.stderr.strip() if result.stderr else 'No error message'}")
                    cli_ok = False
        except Exception as e:
            print(f"✗ CLI test failed: {e}")
            cli_ok = False
        
        # Summary
        print("\n" + "=" * 50)
        print("INSTALLATION SUMMARY:")
        
        if core_ok and api_ok and cli_ok:
            print("✓ Core JPMapper functionality: READY")
        else:
            print("✗ Core JPMapper functionality: ISSUES FOUND")
        
        if enhanced_ok and enhanced_api_ok:
            print("✓ Enhanced metadata-aware features: READY")
        else:
            print("✗ Enhanced metadata-aware features: MISSING DEPENDENCIES")
            print("  Install with: conda install -c conda-forge geopandas fiona")
        
        print(f"\nPython version: {sys.version}")
        print(f"JPMapper location: {Path(__file__).parent}")
        
        if core_ok and api_ok and cli_ok and enhanced_ok and enhanced_api_ok:
            print("\n🎉 Installation complete! All features available.")
            print("✓ Ready for full test suite execution with all features")
            return 0
        elif core_ok and api_ok and cli_ok:
            print("\n⚠ Installation mostly complete. Enhanced features require additional dependencies.")
            print("ℹ Some tests will be skipped during test suite execution")
            return 1
        else:
            print("\n❌ Installation incomplete. Please check missing dependencies.")
            print("❌ Test suite execution may fail")
            return 2


if __name__ == "__main__":
    sys.exit(main())
