"""Plotting helper for terrain/Fresnel profiles."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import rasterio
    import rasterio.plot
    from rasterio.transform import from_bounds
    from shapely.geometry import Point, LineString, box
    import geopandas as gpd
    HAS_GIS_LIBS = True
except ImportError:
    HAS_GIS_LIBS = False
    logger.warning("GIS libraries not available for map rendering (rasterio, geopandas, shapely)")

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    logger.info("Contextily not available for web map tiles. Install with: pip install contextily")


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


def render_analysis_map(
    dsm_path: Union[str, Path],
    analyzed_points: List[Tuple[float, float]],  # [(lat, lon), ...]
    sample_points: List[List[Tuple[float, float]]],  # [[(lat, lon), ...], ...] for each analysis
    output_path: Union[str, Path],
    title: str = "LiDAR Analysis Results",
    buffer_km: float = 1.0,
    sample_obstructions: Optional[List[List[bool]]] = None,  # [[True/False, ...], ...] for each analysis
    sample_no_data: Optional[List[List[bool]]] = None,  # [[True/False, ...], ...] for each analysis
) -> bool:
    """Render an enhanced map with OpenStreetMap base layer, DSM coverage, analyzed points, and sample points.
    
    Args:
        dsm_path: Path to the DSM GeoTIFF file
        analyzed_points: List of analyzed endpoint coordinates as (lat, lon)
        sample_points: List of sample point arrays, one per analysis line
        output_path: Path where to save the map (PNG format)
        title: Title for the map
        buffer_km: Buffer around analyzed points in kilometers
        sample_obstructions: Optional list of obstruction flags for each sample point
        sample_no_data: Optional list of no-data flags for each sample point
        
    Returns:
        True if map was successfully created, False otherwise
    """
    if not HAS_GIS_LIBS:
        logger.warning("Cannot render map: missing required libraries (rasterio, geopandas, shapely)")
        return False
        
    if not analyzed_points:
        logger.warning("No analyzed points provided for map rendering")
        return False
        
    try:
        # Import required libraries
        from pyproj import Transformer
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        
        # Open the DSM to get bounds and CRS info
        with rasterio.open(dsm_path) as dsm:
            dsm_bounds = dsm.bounds
            dsm_crs = dsm.crs
            
            # Create transformers
            tf_wgs84_to_dsm = Transformer.from_crs(4326, dsm_crs, always_xy=True)
            tf_dsm_to_wgs84 = Transformer.from_crs(dsm_crs, 4326, always_xy=True)
            
            # Get extent of analyzed points in WGS84
            lats = [pt[0] for pt in analyzed_points]
            lons = [pt[1] for pt in analyzed_points]
            
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            # Add buffer for better context
            lat_buffer = buffer_km / 111.32  # rough conversion km to degrees
            lon_buffer = buffer_km / (111.32 * np.cos(np.radians((min_lat + max_lat) / 2)))
            
            plot_min_lat = min_lat - lat_buffer
            plot_max_lat = max_lat + lat_buffer
            plot_min_lon = min_lon - lon_buffer
            plot_max_lon = max_lon + lon_buffer
            
            # Create figure with larger size for better detail
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Initialize plotting variables
            plot_crs = 'EPSG:4326'  # Default to WGS84
            tf_wgs84_to_plot = Transformer.from_crs(4326, 4326, always_xy=True)  # Identity transform
            
            # Try to add OpenStreetMap base layer if contextily is available
            if HAS_CONTEXTILY:
                try:
                    import contextily as ctx
                    
                    # Create points GeoDataFrame in WGS84 and convert to Web Mercator
                    all_lats = [pt[0] for pt in analyzed_points]
                    all_lons = [pt[1] for pt in analyzed_points]
                    
                    # Add sample points to get full extent
                    for points in sample_points:
                        for lat, lon in points:
                            all_lats.append(lat)
                            all_lons.append(lon)
                    
                    if all_lats and all_lons:
                        # Calculate bounds in WGS84 with buffer
                        min_lat, max_lat = min(all_lats), max(all_lats)
                        min_lon, max_lon = min(all_lons), max(all_lons)
                        
                        # Add buffer
                        lat_buffer = buffer_km / 111.32
                        lon_buffer = buffer_km / (111.32 * np.cos(np.radians((min_lat + max_lat) / 2)))
                        
                        # Create polygon for the area
                        from shapely.geometry import box
                        bbox_poly = box(min_lon - lon_buffer, min_lat - lat_buffer, 
                                       max_lon + lon_buffer, max_lat + lat_buffer)
                        
                        # Create GeoDataFrame and convert to Web Mercator
                        gdf = gpd.GeoDataFrame([1], geometry=[bbox_poly], crs='EPSG:4326')
                        gdf_mercator = gdf.to_crs('EPSG:3857')
                        bounds = gdf_mercator.total_bounds
                        
                        # Set the map extent in Web Mercator
                        ax.set_xlim(bounds[0], bounds[2])
                        ax.set_ylim(bounds[1], bounds[3])
                        
                        # Add the OpenStreetMap base layer
                        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.OpenStreetMap.Mapnik,
                                       zoom='auto', attribution_size=6)
                        
                        # Update plotting CRS and transformer
                        plot_crs = 'EPSG:3857'
                        tf_wgs84_to_plot = Transformer.from_crs(4326, plot_crs, always_xy=True)
                        
                        logger.info("Added OpenStreetMap base layer")
                    else:
                        logger.warning("No points available for map extent calculation")
                    
                except Exception as e:
                    logger.warning(f"Failed to add OpenStreetMap base layer: {e}")
                    logger.warning(f"Error details: {type(e).__name__}: {e}")
                    # Fall back to plotting in WGS84
                    plot_crs = 'EPSG:4326'
                    tf_wgs84_to_plot = Transformer.from_crs(4326, 4326, always_xy=True)
            else:
                logger.info("OpenStreetMap base layer not available (install contextily: pip install contextily)")
            
            # Add DSM bounds outline (converted to plot CRS)
            dsm_corners_wgs84 = [
                tf_dsm_to_wgs84.transform(dsm_bounds.left, dsm_bounds.bottom),
                tf_dsm_to_wgs84.transform(dsm_bounds.right, dsm_bounds.bottom),
                tf_dsm_to_wgs84.transform(dsm_bounds.right, dsm_bounds.top),
                tf_dsm_to_wgs84.transform(dsm_bounds.left, dsm_bounds.top),
                tf_dsm_to_wgs84.transform(dsm_bounds.left, dsm_bounds.bottom)  # Close the polygon
            ]
            
            dsm_outline_x, dsm_outline_y = [], []
            for lon, lat in dsm_corners_wgs84:
                x, y = tf_wgs84_to_plot.transform(lon, lat)
                dsm_outline_x.append(x)
                dsm_outline_y.append(y)
            
            ax.plot(dsm_outline_x, dsm_outline_y, 'purple', linewidth=3, alpha=0.8, 
                   label='DSM Coverage Area', zorder=3)
            
            # Plot sample points as connected paths (showing actual terrain following)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            for i, points in enumerate(sample_points):
                if points and len(points) > 1:
                    # Convert sample points to plot coordinates
                    path_x, path_y = [], []
                    for lat, lon in points:
                        x, y = tf_wgs84_to_plot.transform(lon, lat)
                        path_x.append(x)
                        path_y.append(y)
                    
                    color = colors[i % len(colors)]
                    # Plot the terrain-following path
                    ax.plot(path_x, path_y, color=color, linewidth=2, alpha=0.8, 
                           label=f'Analysis Path {i+1}' if i < 6 else None, zorder=5)
                    
                    # Plot individual sample points with color coding
                    # For now, we'll use mock obstruction data since detailed per-sample 
                    # obstruction information is not yet available from the LOS analysis
                    # TODO: Enhance LOS analysis to return per-sample obstruction data
                    point_colors = []
                    for j in range(len(points)):
                        if sample_no_data and i < len(sample_no_data) and j < len(sample_no_data[i]) and sample_no_data[i][j]:
                            point_colors.append('white')  # No data points
                        elif sample_obstructions and i < len(sample_obstructions) and j < len(sample_obstructions[i]) and sample_obstructions[i][j]:
                            point_colors.append('red')    # Obstructed points
                        else:
                            point_colors.append('black')  # Clear points with data
                    
                    # Plot points with individual colors and thin black edge for white points visibility
                    ax.scatter(path_x, path_y, c=point_colors, s=15, marker='o', 
                              alpha=0.8, zorder=6, edgecolors='black', linewidth=0.4)
            
            # Plot analyzed endpoint points (large, bold)
            analysis_x, analysis_y = [], []
            for lat, lon in analyzed_points:
                x, y = tf_wgs84_to_plot.transform(lon, lat)
                analysis_x.append(x)
                analysis_y.append(y)
            
            ax.scatter(analysis_x, analysis_y, c='red', s=250, marker='o', 
                      label='Analysis Endpoints', zorder=7, edgecolors='white', linewidth=3)
            
            # Set plot bounds if not using contextily (which sets its own bounds)
            if plot_crs == 'EPSG:4326':
                ax.set_xlim(plot_min_lon, plot_max_lon)
                ax.set_ylim(plot_min_lat, plot_max_lat)
            
            # Set labels based on CRS
            if plot_crs == 'EPSG:3857':
                ax.set_xlabel('Web Mercator X (m)', fontsize=12)
                ax.set_ylabel('Web Mercator Y (m)', fontsize=12)
            else:
                ax.set_xlabel('Longitude (°)', fontsize=12)
                ax.set_ylabel('Latitude (°)', fontsize=12)
            
            ax.set_title(title, fontsize=16, pad=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Set equal aspect ratio to prevent distortion
            ax.set_aspect('equal')
            
            # Tight layout and save
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Enhanced analysis map saved to: {output_path}")
            return True
            
    except ImportError as e:
        logger.warning(f"Missing required packages for enhanced map: {e}")
        # Fall back to simple map
        return create_simple_analysis_map(analyzed_points, sample_points, output_path, title, 
                                        sample_obstructions, sample_no_data)
        
    except Exception as e:
        logger.error(f"Failed to render enhanced analysis map: {e}")
        # Fall back to simple map
        return create_simple_analysis_map(analyzed_points, sample_points, output_path, title,
                                        sample_obstructions, sample_no_data)


def create_simple_analysis_map(
    analyzed_points: List[Tuple[float, float]],  # [(lat, lon), ...]
    sample_points: List[List[Tuple[float, float]]],  # [[(lat, lon), ...], ...] for each analysis
    output_path: Union[str, Path],
    title: str = "LiDAR Analysis Results - Points Only",
    sample_obstructions: Optional[List[List[bool]]] = None,  # [[True/False, ...], ...] for each analysis
    sample_no_data: Optional[List[List[bool]]] = None,  # [[True/False, ...], ...] for each analysis
) -> bool:
    """Create a simple map showing just the analyzed points and paths (no DSM background).
    
    This is a fallback when DSM rendering fails or GIS libraries are not available.
    
    Args:
        analyzed_points: List of analyzed endpoint coordinates as (lat, lon)
        sample_points: List of sample point arrays, one per analysis line  
        output_path: Path where to save the map (PNG format)
        title: Title for the map
        sample_obstructions: Optional list of obstruction flags for each sample point
        sample_no_data: Optional list of no-data flags for each sample point
        
    Returns:
        True if map was successfully created, False otherwise
    """
    try:
        if not analyzed_points:
            logger.warning("No analyzed points provided for simple map rendering")
            return False
            
        # Create the plot with larger size
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Extract coordinates
        lats = [pt[0] for pt in analyzed_points]
        lons = [pt[1] for pt in analyzed_points]
        
        # Plot analysis paths as terrain-following lines with distinct colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for i, points in enumerate(sample_points):
            if points and len(points) > 1:
                sample_lons = [pt[1] for pt in points]
                sample_lats = [pt[0] for pt in points]
                
                color = colors[i % len(colors)]
                # Plot the terrain-following path
                ax.plot(sample_lons, sample_lats, color=color, linewidth=3, alpha=0.8, 
                       label=f'Analysis Path {i+1}' if i < 6 else None, zorder=4)
                
                # Plot individual sample points with color coding
                # For now, we'll use mock obstruction data since detailed per-sample 
                # obstruction information is not yet available from the LOS analysis
                # TODO: Enhance LOS analysis to return per-sample obstruction data
                point_colors = []
                for j in range(len(points)):
                    if sample_no_data and i < len(sample_no_data) and j < len(sample_no_data[i]) and sample_no_data[i][j]:
                        point_colors.append('white')  # No data points
                    elif sample_obstructions and i < len(sample_obstructions) and j < len(sample_obstructions[i]) and sample_obstructions[i][j]:
                        point_colors.append('red')    # Obstructed points
                    else:
                        point_colors.append('black')  # Clear points with data
                
                # Plot points with individual colors and thin black edge for white points visibility
                ax.scatter(sample_lons, sample_lats, c=point_colors, s=15, marker='o', 
                          alpha=0.8, zorder=5, edgecolors='black', linewidth=0.4)
        
        # Plot analyzed endpoint points (large, bold)
        ax.scatter(lons, lats, c='red', s=250, marker='o', 
                  label='Analysis Endpoints', zorder=6, edgecolors='white', linewidth=3)
        
        # Set labels and formatting
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.set_title(title, fontsize=16, pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add some buffer around the points
        if lons and lats:
            lon_range = max(lons) - min(lons)
            lat_range = max(lats) - min(lats)
            buffer = max(lon_range, lat_range) * 0.1  # 10% buffer
            
            ax.set_xlim(min(lons) - buffer, max(lons) + buffer)
            ax.set_ylim(min(lats) - buffer, max(lats) + buffer)
        
        # Tight layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Simple analysis map saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render simple analysis map: {e}")
        return False
