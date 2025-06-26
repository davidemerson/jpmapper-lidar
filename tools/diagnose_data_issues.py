#!/usr/bin/env python3
"""
Diagnostic script to analyze potential data quality issues in jpmapper-lidar rasterization and analysis.

This script helps identify:
1. Issues with the rasterized GeoTIFF DSM
2. Coordinate transformation problems
3. Missing data in LAS files vs GeoTIFF
4. Analysis parameter issues
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def analyze_geotiff_dsm(dsm_path: Path):
    """Analyze the quality of the generated DSM GeoTIFF."""
    console.print(Panel.fit(f"[bold blue]Analyzing DSM: {dsm_path.name}[/bold blue]"))
    
    if not dsm_path.exists():
        console.print(f"[red]❌ DSM file not found: {dsm_path}[/red]")
        return None
    
    try:
        with rasterio.open(dsm_path) as src:
            # Basic metadata
            console.print(f"[cyan]CRS:[/cyan] {src.crs}")
            console.print(f"[cyan]Bounds:[/cyan] {src.bounds}")
            console.print(f"[cyan]Resolution:[/cyan] {src.transform[0]:.3f} x {abs(src.transform[4]):.3f}")
            console.print(f"[cyan]Shape:[/cyan] {src.height} x {src.width}")
            console.print(f"[cyan]NoData value:[/cyan] {src.nodata}")
            
            # Read the data
            data = src.read(1)
            
            # Data statistics
            total_pixels = data.size
            valid_pixels = np.sum(~np.isnan(data))
            if src.nodata is not None:
                valid_pixels = np.sum((~np.isnan(data)) & (data != src.nodata))
            
            zero_pixels = np.sum(data == 0.0)
            
            table = Table(title="DSM Data Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_column("Percentage", style="green")
            
            table.add_row("Total pixels", f"{total_pixels:,}", "100%")
            table.add_row("Valid pixels", f"{valid_pixels:,}", f"{100*valid_pixels/total_pixels:.1f}%")
            table.add_row("Zero elevation pixels", f"{zero_pixels:,}", f"{100*zero_pixels/total_pixels:.1f}%")
            
            if valid_pixels > 0:
                valid_data = data[(~np.isnan(data)) & (data != (src.nodata or -9999))]
                if len(valid_data) > 0:
                    table.add_row("Min elevation", f"{valid_data.min():.2f}m", "")
                    table.add_row("Max elevation", f"{valid_data.max():.2f}m", "")
                    table.add_row("Mean elevation", f"{valid_data.mean():.2f}m", "")
                    table.add_row("Std elevation", f"{valid_data.std():.2f}m", "")
            
            console.print(table)
            
            # Identify potential issues
            issues = []
            if valid_pixels / total_pixels < 0.5:
                issues.append(f"⚠️  Only {100*valid_pixels/total_pixels:.1f}% of pixels have valid data")
            
            if zero_pixels / total_pixels > 0.1:
                issues.append(f"⚠️  {100*zero_pixels/total_pixels:.1f}% of pixels have zero elevation (might indicate missing data)")
            
            if valid_pixels > 0:
                valid_data = data[(~np.isnan(data)) & (data != (src.nodata or -9999))]
                if len(valid_data) > 0 and valid_data.min() < -100:
                    issues.append(f"⚠️  Suspiciously low minimum elevation: {valid_data.min():.2f}m")
                
                if len(valid_data) > 0 and valid_data.max() > 1000:
                    issues.append(f"⚠️  Suspiciously high maximum elevation: {valid_data.max():.2f}m")
            
            if issues:
                console.print("\n[yellow]Potential Issues:[/yellow]")
                for issue in issues:
                    console.print(f"  {issue}")
            else:
                console.print("\n[green]✅ DSM appears to be in good shape![/green]")
            
            return {
                'total_pixels': total_pixels,
                'valid_pixels': valid_pixels,
                'zero_pixels': zero_pixels,
                'crs': str(src.crs),
                'bounds': src.bounds,
                'resolution': (src.transform[0], abs(src.transform[4])),
                'has_issues': len(issues) > 0,
                'issues': issues
            }
            
    except Exception as e:
        console.print(f"[red]❌ Error reading DSM: {e}[/red]")
        return None

def analyze_csv_points(csv_path: Path):
    """Analyze the point pairs in the CSV file."""
    console.print(Panel.fit(f"[bold blue]Analyzing CSV Points: {csv_path.name}[/bold blue]"))
    
    if not csv_path.exists():
        console.print(f"[red]❌ CSV file not found: {csv_path}[/red]")
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        console.print(f"[cyan]Total point pairs:[/cyan] {len(df)}")
        console.print(f"[cyan]Columns:[/cyan] {list(df.columns)}")
        
        # Analyze coordinate ranges
        lat_cols = [col for col in df.columns if 'lat' in col.lower()]
        lon_cols = [col for col in df.columns if 'lon' in col.lower()]
        
        if lat_cols and lon_cols:
            all_lats = pd.concat([df[col] for col in lat_cols])
            all_lons = pd.concat([df[col] for col in lon_cols])
            
            table = Table(title="Coordinate Ranges")
            table.add_column("Coordinate", style="cyan")
            table.add_column("Min", style="magenta")
            table.add_column("Max", style="magenta")
            table.add_column("Range", style="green")
            
            lat_range = all_lats.max() - all_lats.min()
            lon_range = all_lons.max() - all_lons.min()
            
            table.add_row("Latitude", f"{all_lats.min():.6f}°", f"{all_lats.max():.6f}°", f"{lat_range:.6f}°")
            table.add_row("Longitude", f"{all_lons.min():.6f}°", f"{all_lons.max():.6f}°", f"{lon_range:.6f}°")
            
            console.print(table)
            
            # Check if coordinates look reasonable for NYC area
            issues = []
            if not (40.4 <= all_lats.min() <= 41.0 and 40.4 <= all_lats.max() <= 41.0):
                issues.append(f"⚠️  Latitude range {all_lats.min():.6f}° to {all_lats.max():.6f}° doesn't look like NYC area")
            
            if not (-74.5 <= all_lons.min() <= -73.0 and -74.5 <= all_lons.max() <= -73.0):
                issues.append(f"⚠️  Longitude range {all_lons.min():.6f}° to {all_lons.max():.6f}° doesn't look like NYC area")
            
            if issues:
                console.print("\n[yellow]Potential Issues:[/yellow]")
                for issue in issues:
                    console.print(f"  {issue}")
            else:
                console.print("\n[green]✅ Coordinates look reasonable for NYC area![/green]")
                
            return {
                'total_pairs': len(df),
                'lat_range': (all_lats.min(), all_lats.max()),
                'lon_range': (all_lons.min(), all_lons.max()),
                'has_issues': len(issues) > 0,
                'issues': issues
            }
        
    except Exception as e:
        console.print(f"[red]❌ Error reading CSV: {e}[/red]")
        return None

def analyze_results_json(json_path: Path):
    """Analyze the analysis results JSON."""
    console.print(Panel.fit(f"[bold blue]Analyzing Results: {json_path.name}[/bold blue]"))
    
    if not json_path.exists():
        console.print(f"[red]❌ Results file not found: {json_path}[/red]")
        return None
    
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        if not isinstance(results, list):
            console.print(f"[red]❌ Expected list of results, got {type(results)}[/red]")
            return None
        
        console.print(f"[cyan]Total results:[/cyan] {len(results)}")
        
        if len(results) == 0:
            console.print("[red]❌ No results found![/red]")
            return None
        
        # Analyze first result in detail
        first_result = results[0]
        console.print(f"[cyan]Sample result keys:[/cyan] {list(first_result.keys())}")
        
        # Look for missing data issues
        issues = []
        missing_data_count = 0
        zero_elevation_count = 0
        
        for i, result in enumerate(results):
            # Check for missing terrain heights
            if 'sample_points' in result:
                sample_points = result['sample_points']
                for point in sample_points:
                    if point.get('terrain_height_m', 0) == 0.0:
                        zero_elevation_count += 1
            
            # Check for unrealistic values
            min_height = result.get('min_terrain_height_m', None)
            max_height = result.get('max_terrain_height_m', None)
            
            if min_height == 0.0:
                missing_data_count += 1
        
        table = Table(title="Analysis Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total analyses", f"{len(results)}")
        table.add_row("Results with min height = 0", f"{missing_data_count}")
        table.add_row("Sample points with 0 elevation", f"{zero_elevation_count}")
        
        if len(results) > 0:
            clear_count = sum(1 for r in results if r.get('clear', False))
            table.add_row("Clear paths", f"{clear_count} ({100*clear_count/len(results):.1f}%)")
        
        console.print(table)
        
        if missing_data_count > 0:
            issues.append(f"⚠️  {missing_data_count} results have minimum terrain height = 0 (likely missing data)")
        
        if zero_elevation_count > 0:
            issues.append(f"⚠️  {zero_elevation_count} sample points have 0 elevation (likely missing data)")
        
        if issues:
            console.print("\n[yellow]Potential Issues:[/yellow]")
            for issue in issues:
                console.print(f"  {issue}")
        else:
            console.print("\n[green]✅ Results look reasonable![/green]")
        
        return {
            'total_results': len(results),
            'missing_data_count': missing_data_count,
            'zero_elevation_count': zero_elevation_count,
            'has_issues': len(issues) > 0,
            'issues': issues
        }
        
    except Exception as e:
        console.print(f"[red]❌ Error reading results: {e}[/red]")
        return None

def check_coordinate_transformation():
    """Check if coordinate transformation is working correctly."""
    console.print(Panel.fit("[bold blue]Testing Coordinate Transformation[/bold blue]"))
    
    try:
        import pyproj
        
        # Test transformation from WGS84 to NY State Plane
        wgs84 = pyproj.CRS('EPSG:4326')  # WGS84
        ny_state_plane = pyproj.CRS('EPSG:6539')  # NAD83(2011) / New York Long Island ftUS
        
        transformer = pyproj.Transformer.from_crs(wgs84, ny_state_plane, always_xy=True)
        
        # Test point from the CSV (NYC area)
        test_lon, test_lat = -73.956852, 40.655596
        
        x, y = transformer.transform(test_lon, test_lat)
        
        console.print(f"[cyan]Test coordinate transformation:[/cyan]")
        console.print(f"  WGS84: {test_lat:.6f}°N, {test_lon:.6f}°W")
        console.print(f"  NY State Plane (EPSG:6539): {x:.2f}, {y:.2f} ft")
        
        # Check if the transformed coordinates are reasonable for NYC
        # NYC in NY State Plane should be roughly:
        # X: 900,000 - 1,100,000 ft
        # Y: 100,000 - 300,000 ft
        
        issues = []
        if not (800000 <= x <= 1200000):
            issues.append(f"⚠️  X coordinate {x:.0f} ft outside expected NYC range (800k-1200k ft)")
        
        if not (50000 <= y <= 350000):
            issues.append(f"⚠️  Y coordinate {y:.0f} ft outside expected NYC range (50k-350k ft)")
        
        if issues:
            console.print("\n[yellow]Coordinate Transformation Issues:[/yellow]")
            for issue in issues:
                console.print(f"  {issue}")
        else:
            console.print("\n[green]✅ Coordinate transformation looks correct![/green]")
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'test_coordinates': {
                'wgs84': (test_lat, test_lon),
                'ny_state_plane': (x, y)
            }
        }
        
    except Exception as e:
        console.print(f"[red]❌ Error testing coordinate transformation: {e}[/red]")
        return None

def provide_recommendations(dsm_analysis, csv_analysis, results_analysis, coord_analysis):
    """Provide recommendations based on the analysis."""
    console.print(Panel.fit("[bold yellow]Recommendations[/bold yellow]"))
    
    recommendations = []
    
    # DSM issues
    if dsm_analysis and dsm_analysis.get('has_issues'):
        if any('zero elevation' in issue for issue in dsm_analysis['issues']):
            recommendations.append(
                "🔧 Many pixels have zero elevation in the DSM. This suggests:\n"
                "   - LAS files may have missing data in some areas\n"
                "   - PDAL rasterization parameters may need adjustment\n"
                "   - Consider using a different elevation value for 'no data' (like -9999)"
            )
        
        if dsm_analysis['valid_pixels'] / dsm_analysis['total_pixels'] < 0.5:
            recommendations.append(
                "🔧 Less than 50% of DSM pixels have valid data. Consider:\n"
                "   - Checking if LAS files cover the analysis area completely\n"
                "   - Using a coarser resolution for better coverage\n"
                "   - Verifying the bounding box of your analysis area"
            )
    
    # Results issues
    if results_analysis and results_analysis.get('has_issues'):
        if results_analysis['missing_data_count'] > 0:
            recommendations.append(
                "🔧 Some analysis results show minimum terrain height = 0. This suggests:\n"
                "   - The DSM has missing data in analysis areas\n"
                "   - Point coordinates may be outside the DSM coverage area\n"
                "   - Coordinate transformation issues between WGS84 and the DSM CRS"
            )
    
    # Coordinate issues
    if coord_analysis and coord_analysis.get('has_issues'):
        recommendations.append(
            "🔧 Coordinate transformation issues detected. Check:\n"
            "   - EPSG code used for rasterization (should be 6539 for NYC)\n"
            "   - Point coordinates are in the correct format (lat/lon)\n"
            "   - DSM coverage area matches your analysis points"
        )
    
    # General recommendations
    recommendations.extend([
        "📋 To debug further:\n"
        "   1. Create a small test DSM with just a few LAS files\n"
        "   2. Visualize the DSM in QGIS to check coverage and values\n"
        "   3. Test analysis with a single point pair in a known good area\n"
        "   4. Check that LAS files contain data for your analysis coordinates",
        
        "🛠️ Consider using enhanced rasterization:\n"
        "   - Use jpmapper.api.rasterize_tile_with_metadata() for better CRS detection\n"
        "   - Enable metadata-aware processing to automatically optimize parameters\n"
        "   - Use the batch processing functions for consistent results"
    ])
    
    if not recommendations:
        recommendations.append("✅ No issues detected! Your data appears to be in good shape.")
    
    for i, rec in enumerate(recommendations, 1):
        console.print(f"\n{i}. {rec}")

def main():
    """Main diagnostic function."""
    console.print(Panel.fit("[bold green]JPMapper LiDAR Data Quality Diagnostics[/bold green]", width=80))
    
    # File paths
    workspace = Path(".")
    points_csv = workspace / "tests" / "data" / "points.csv"
    results_json = workspace / "meshresults.json"
    
    # Check for alternative DSM locations
    possible_dsm_paths = [
        workspace / "test_output" / "fresh_mosaic.tif",  # Fresh DSM with CRS fix
        workspace / "dsm_cache.tif",
        workspace / "fresh_mosaic.tif",
        workspace / "output.tif"
    ]
    
    dsm_path = None
    for path in possible_dsm_paths:
        if path.exists():
            dsm_path = path
            break
    
    if not dsm_path:
        console.print(f"[yellow]⚠️  Could not find DSM file. Looking for:[/yellow]")
        for path in possible_dsm_paths:
            console.print(f"  - {path}")
        console.print("\n[cyan]Please check if the DSM was created successfully.[/cyan]")
        dsm_analysis = None
    else:
        dsm_analysis = analyze_geotiff_dsm(dsm_path)
    
    console.print("\n" + "="*80 + "\n")
    csv_analysis = analyze_csv_points(points_csv)
    
    console.print("\n" + "="*80 + "\n")
    results_analysis = analyze_results_json(results_json)
    
    console.print("\n" + "="*80 + "\n")
    coord_analysis = check_coordinate_transformation()
    
    console.print("\n" + "="*80 + "\n")
    provide_recommendations(dsm_analysis, csv_analysis, results_analysis, coord_analysis)

if __name__ == "__main__":
    main()
