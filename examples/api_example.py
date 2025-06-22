"""
Example usage of the JPMapper API
--------------------------------

This script demonstrates how to use the JPMapper API programmatically
with proper error handling.
"""

from pathlib import Path
import logging
import sys

from jpmapper.api import (
    filter_by_bbox,
    rasterize_directory,
    merge_tiles,
    analyze_los,
    save_profile_plot,
)
from jpmapper.exceptions import (
    JPMapperError,
    FilterError,
    RasterizationError,
    AnalysisError,
    GeometryError,
    NoDataError
)
from jpmapper.logging import setup as setup_logging, console

# Set up logging
setup_logging(level=logging.INFO)

def main():
    try:
        # Define paths
        las_dir = Path("data/las")
        dsm_dir = Path("data/dsm")
        dsm_mosaic = Path("data/mosaic.tif")
        profile_plot = Path("data/profile.png")
        
        # Define bbox (NYC example: min_x, min_y, max_x, max_y)
        bbox = (-74.01, 40.70, -73.96, 40.75)
        
        # Define points (latitude, longitude)
        point_a = (40.7128, -74.0060)  # NYC
        point_b = (40.7614, -73.9776)  # Times Square
        
        # Create output directories
        dsm_dir.mkdir(parents=True, exist_ok=True)
        profile_plot.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Filter LAS files by bbox
        console.print("[yellow]Filtering LAS files...[/yellow]")
        try:
            las_files = list(las_dir.glob("*.las"))
            if not las_files:
                console.print("[red]No LAS files found in directory[/red]")
            else:
                filtered_las = filter_by_bbox(las_files, bbox=bbox)
                console.print(f"[green]Selected {len(filtered_las)} of {len(las_files)} files[/green]")
        except FilterError as e:
            console.print(f"[red]Error filtering LAS files: {e}[/red]")
            # Continue with remaining steps
        
        # 2. Rasterize LAS files to DSM GeoTIFFs
        console.print("[yellow]Rasterizing LAS files to DSM GeoTIFFs...[/yellow]")
        dsm_tiles = []
        try:
            if not las_files:
                console.print("[red]No LAS files to rasterize[/red]")
            else:
                dsm_tiles = rasterize_directory(
                    las_dir, 
                    dsm_dir, 
                    epsg=6539,  # NY Long Island ftUS
                    resolution=0.1,
                    workers=4,  # Use 4 parallel workers
                )
                console.print(f"[green]Created {len(dsm_tiles)} DSM tiles[/green]")
        except NoDataError as e:
            console.print(f"[red]No data error: {e}[/red]")
        except RasterizationError as e:
            console.print(f"[red]Rasterization error: {e}[/red]")
            # Check if we have existing DSM files we can use
            dsm_tiles = list(dsm_dir.glob("*.tif"))
            if dsm_tiles:
                console.print(f"[yellow]Using {len(dsm_tiles)} existing DSM tiles[/yellow]")
        
        # 3. Merge DSM tiles into a mosaic
        if dsm_tiles:
            console.print("[yellow]Merging DSM tiles into a mosaic...[/yellow]")
            try:
                merge_tiles(dsm_tiles, dsm_mosaic)
                console.print(f"[green]Created mosaic at {dsm_mosaic}[/green]")
            except RasterizationError as e:
                console.print(f"[red]Error merging tiles: {e}[/red]")
                # If the merge fails but we already have a mosaic, try to use it
                if dsm_mosaic.exists():
                    console.print(f"[yellow]Using existing mosaic at {dsm_mosaic}[/yellow]")
                else:
                    console.print("[red]No mosaic available for analysis[/red]")
                    return
        elif dsm_mosaic.exists():
            console.print(f"[yellow]Using existing mosaic at {dsm_mosaic}[/yellow]")
        else:
            console.print("[red]No DSM tiles available for merging and no existing mosaic[/red]")
            return
        
        # 4. Analyze line-of-sight between two points
        console.print("[yellow]Analyzing line-of-sight...[/yellow]")
        try:
            result = analyze_los(
                dsm_mosaic,
                point_a,
                point_b,
                freq_ghz=5.8,
                max_mast_height_m=5,
            )
            
            # Print results
            if result["clear"]:
                console.print("[green]Path is clear![/green]")
            else:
                console.print(f"[red]Path is blocked. Minimum mast height required: {result['mast_height_m']} m[/red]")
            
            console.print(f"Minimum clearance: {result['clearance_min_m']:.2f} m")
            console.print(f"Maximum intrusion: {result['overshoot_max_m']:.2f} m")
            console.print(f"Ground at point A: {result['ground_a_m']:.2f} m")
            console.print(f"Ground at point B: {result['ground_b_m']:.2f} m")
            
            # 5. Generate a profile plot
            console.print("[yellow]Generating profile plot...[/yellow]")
            try:
                save_profile_plot(
                    dsm_mosaic,
                    point_a,
                    point_b,
                    profile_plot,
                    title="NYC to Times Square Link Profile",
                )
                console.print(f"[green]Created profile plot at {profile_plot}[/green]")
            except AnalysisError as e:
                console.print(f"[red]Error creating profile plot: {e}[/red]")
                
        except GeometryError as e:
            console.print(f"[red]Geometry error in analysis: {e}[/red]")
        except AnalysisError as e:
            console.print(f"[red]Analysis error: {e}[/red]")
            
    except JPMapperError as e:
        console.print(f"[red]JPMapper error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print_exception()

if __name__ == "__main__":
    main()
