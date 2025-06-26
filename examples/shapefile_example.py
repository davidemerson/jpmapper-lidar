"""
Example: Using Shapefiles with JPMapper
======================================

This example demonstrates how to use shapefiles for more precise LAS file processing,
including complex boundary filtering and coordinate system handling.
"""

from pathlib import Path
from rich.console import Console

console = Console()

def main():
    """Demonstrate shapefile-based LAS file processing."""
    
    try:
        # Import shapefile functions (requires geopandas)
        from jpmapper.api import (
            filter_by_shapefile, 
            create_boundary_from_las_files,
            rasterize_directory,
            cached_mosaic
        )
        
        console.print("[green]✓ Shapefile support available[/green]")
        
    except ImportError as e:
        console.print(f"[red]✗ Shapefile support not available: {e}[/red]")
        console.print("[yellow]Install with: conda install -c conda-forge geopandas fiona[/yellow]")
        return
    
    # Define paths
    las_dir = Path("data/las")
    shapefile_path = Path("data/boundaries/study_area.shp")
    filtered_dir = Path("data/filtered_las")
    dsm_output = Path("data/study_area_dsm.tif")
    
    console.print("[bold blue]Shapefile-based LAS Processing Workflow[/bold blue]")
    
    # Example 1: Filter LAS files using an existing shapefile
    if shapefile_path.exists():
        console.print(f"\n[yellow]1. Filtering LAS files using shapefile: {shapefile_path}[/yellow]")
        
        try:
            las_files = list(las_dir.glob("*.las"))
            if not las_files:
                console.print("[red]No LAS files found in data/las/[/red]")
                return
            
            # Filter with 50m buffer around shapefile boundary
            filtered_files = filter_by_shapefile(
                las_files,
                shapefile_path,
                dst_dir=filtered_dir,
                buffer_meters=50.0,
                validate_crs=True
            )
            
            console.print(f"[green]✓ Filtered {len(filtered_files)} of {len(las_files)} LAS files[/green]")
            console.print(f"[green]✓ Copied to: {filtered_dir}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error filtering by shapefile: {e}[/red]")
            
    else:
        console.print(f"[yellow]Shapefile not found: {shapefile_path}[/yellow]")
        console.print("[yellow]Creating boundary from LAS files instead...[/yellow]")
        
        # Example 2: Create a boundary shapefile from LAS file extents
        console.print(f"\n[yellow]2. Creating boundary shapefile from LAS extents[/yellow]")
        
        try:
            las_files = list(las_dir.glob("*.las"))
            if not las_files:
                console.print("[red]No LAS files found in data/las/[/red]")
                return
            
            # Create boundary with 100m buffer
            boundary_shapefile = Path("data/boundaries/las_boundary.shp")
            boundary_shapefile.parent.mkdir(parents=True, exist_ok=True)
            
            created_boundary = create_boundary_from_las_files(
                las_files,
                boundary_shapefile,
                buffer_meters=100.0,
                epsg=6539  # NY Long Island State Plane
            )
            
            console.print(f"[green]✓ Created boundary shapefile: {created_boundary}[/green]")
            
            # Now use the created boundary to filter files
            filtered_files = filter_by_shapefile(
                las_files,
                created_boundary,
                dst_dir=filtered_dir,
                buffer_meters=0.0,  # No additional buffer since we already buffered
                validate_crs=True
            )
            
            console.print(f"[green]✓ Filtered {len(filtered_files)} files using created boundary[/green]")
            
        except Exception as e:
            console.print(f"[red]Error creating/using boundary: {e}[/red]")
    
    # Example 3: Process filtered files into a DSM
    console.print(f"\n[yellow]3. Creating DSM from filtered LAS files[/yellow]")
    
    try:
        if filtered_dir.exists() and list(filtered_dir.glob("*.las")):
            dsm_path = cached_mosaic(
                filtered_dir,
                dsm_output,
                epsg=6539,
                resolution=0.5,  # 0.5m resolution
                workers=None,  # Auto-detect
                force=False
            )
            
            console.print(f"[green]✓ Created DSM: {dsm_path}[/green]")
            console.print(f"[green]✓ DSM covers the precise shapefile boundary area[/green]")
            
        else:
            console.print("[yellow]No filtered LAS files available for DSM creation[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error creating DSM: {e}[/red]")
    
    # Example 4: Demonstrate coordinate system handling
    console.print(f"\n[yellow]4. Coordinate System Best Practices[/yellow]")
    
    console.print("[cyan]• Always ensure LAS files and shapefiles use compatible CRS[/cyan]")
    console.print("[cyan]• Use validate_crs=True to catch CRS mismatches early[/cyan]")
    console.print("[cyan]• JPMapper can transform coordinates between compatible CRS[/cyan]")
    console.print("[cyan]• For best results, use projected coordinates (UTM, State Plane)[/cyan]")
    
    console.print(f"\n[bold green]✓ Shapefile workflow demonstration complete![/bold green]")


def demonstrate_advanced_shapefile_usage():
    """Show advanced shapefile techniques."""
    
    console.print("\n[bold blue]Advanced Shapefile Techniques[/bold blue]")
    
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
        
        # Example: Create a custom shapefile programmatically
        console.print("\n[yellow]Creating custom study area shapefile[/yellow]")
        
        # Define a custom polygon (example coordinates)
        study_area_coords = [
            (-74.0059, 40.7128),  # NYC area
            (-74.0059, 40.7628),
            (-73.9559, 40.7628),
            (-73.9559, 40.7128),
            (-74.0059, 40.7128)   # Close the polygon
        ]
        
        study_polygon = Polygon(study_area_coords)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame([1], geometry=[study_polygon], crs="EPSG:4326")
        
        # Save as shapefile
        output_path = Path("data/boundaries/custom_study_area.shp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path)
        
        console.print(f"[green]✓ Created custom shapefile: {output_path}[/green]")
        
        # Example: Multiple processing areas
        console.print("\n[yellow]Handling multiple processing areas[/yellow]")
        
        # Create multiple polygons for different areas
        area1 = Polygon([(-74.1, 40.7), (-74.0, 40.7), (-74.0, 40.8), (-74.1, 40.8), (-74.1, 40.7)])
        area2 = Polygon([(-73.9, 40.7), (-73.8, 40.7), (-73.8, 40.8), (-73.9, 40.8), (-73.9, 40.7)])
        
        multi_gdf = gpd.GeoDataFrame(
            ["Area_1", "Area_2"], 
            geometry=[area1, area2], 
            crs="EPSG:4326",
            columns=["name"]
        )
        
        multi_output = Path("data/boundaries/multiple_areas.shp")
        multi_gdf.to_file(multi_output)
        
        console.print(f"[green]✓ Created multi-area shapefile: {multi_output}[/green]")
        console.print("[cyan]• JPMapper will process LAS files intersecting ANY of these areas[/cyan]")
        
    except ImportError:
        console.print("[yellow]Advanced examples require geopandas for demonstration[/yellow]")
    except Exception as e:
        console.print(f"[red]Error in advanced examples: {e}[/red]")


if __name__ == "__main__":
    main()
    demonstrate_advanced_shapefile_usage()
